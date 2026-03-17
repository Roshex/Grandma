import pickle
import itertools
import multiprocessing as mp
from tqdm import tqdm
from pathlib import Path
from functools import partial
from typing import List, Dict, Tuple, Union, Optional

from .config import TaskConfig
from .logger import GranLogger
from .models import SmrtTree, MulTree, GroupData, Map, ReconResult, TaskResult, FlatTree, NameRegistry
from .ops import GeneTreeManager

DEFAULT_TARGET = [0]

def _worker_reconcile_single(
    mul_item: Tuple[int, FlatTree],
    flat_gts: Dict[int, FlatTree],
    dup_cost: int,
    loss_cost: int,
    registry: NameRegistry, 
    pickle_dir: str, 
    run_prefix: str, 
    retmap: bool = False,
    optim: bool = False
) -> Tuple[int, int, Optional[Dict[int, ReconResult]]]:
    
    mul_idx, flat_mul = mul_item
    total_score = 0
    gt_results = {} if retmap else None
        
    # Case A: Species Tree 
    if mul_idx == 0:
        for g_num, gt_flat in flat_gts.items():
            res = Reconciler.recon_lca_optimized(gt_flat, flat_mul, dup_cost, loss_cost, registry=registry if retmap else None, retmap=retmap)
            if retmap:
                total_score += res.score
                gt_results[g_num] = res
            else:
                total_score += res
            
    # Case B: MUL-Tree
    else:
        cur_groups = {}
        if mul_idx != 0:
            p_path = Path(pickle_dir) / f"{run_prefix}_{mul_idx}_groups.pickle"
            if p_path.exists():
                with open(p_path, 'rb') as f:
                    cur_groups = pickle.load(f)

        target_map = Reconciler.build_target_map(flat_mul, registry)

        for g_num, gt_flat in flat_gts.items():
            if g_num not in cur_groups: continue
            
            group_data = cur_groups[g_num]
            res = Reconciler.reconcile_permutation(gt_flat, flat_mul, dup_cost, loss_cost, registry, group_data, target_map, retmap=retmap, optim=optim)
            
            if retmap:
                total_score += res.score
                gt_results[g_num] = res
            else:
                total_score += res

    return mul_idx, total_score, gt_results

class Reconciler:
    def __init__(self, config: TaskConfig, logger: GranLogger, num_processes: int = 1, pickle_action: str = 'archive', to_map: bool = False, optim: bool = False):
        self.tcf = config
        self.logger = logger
        self.n_procs = num_processes
        self.pickle_action = pickle_action
        self.to_map = to_map
        self.optim = optim
    
    @staticmethod
    def reconcile_permutation(gt_flat, mul_flat, dup_cost, loss_cost, registry, group_data, target_map, retmap=False, optim=False):
        if optim:
            res = Reconciler.reconcile_permutation_optim(gt_flat, mul_flat, dup_cost, loss_cost, registry, group_data, target_map, retmap=retmap)
        else:
            res = Reconciler.reconcile_permutation_old(gt_flat, mul_flat, dup_cost, loss_cost, registry, group_data, target_map, retmap=retmap)
        return res

    # --------------------------------------------------------------------------
    # COMMON LOGIC
    # --------------------------------------------------------------------------

    @staticmethod
    def build_target_map(mul_flat: FlatTree, registry: NameRegistry) -> Dict[int, List[int]]:
        """
        Calculates the target map ONCE per MUL-tree to avoid redundant string parsing.
        Maps base Species IDs to a list of available MUL-tree node indices.
        """
        target_map = {} 
        for i in range(mul_flat.num_nodes):
            if mul_flat.children_start[i] == mul_flat.children_start[i+1]: 
                name_id = mul_flat.node_to_name_id[i]
                if name_id == -1: continue
                
                sp_name = registry.get_name(name_id)
                base_name = sp_name.replace("*", "")
                base_id = registry.get_id(base_name)
                
                if base_id not in target_map: target_map[base_id] = []
                
                tag_count = sp_name.count("*")
                while len(target_map[base_id]) <= tag_count:
                    target_map[base_id].append(-1)
                target_map[base_id][tag_count] = i
        
        # Fill holes with default (Ancestral/0) if specific copies missing
        for bid, targets in target_map.items():
            if not targets: continue
            valid_target = next((t for t in targets if t != -1), -1)
            for k in range(len(targets)):
                if targets[k] == -1: targets[k] = valid_target
                
        return target_map

    @staticmethod
    def translate_groups_to_ids(gt_flat: FlatTree, group_data: GroupData) -> Tuple[List[List[int]], List[Tuple[List[int], str]]]:
        """
        Modified to accept GroupData that already contains IDs.
        Filters for nodes present in the flattened tree.
        """
        ambig_groups_ids = []
        for grp_ids in group_data.ambiguous_groups:
            # grp_ids is List[int]
            valid_ids = []
            for nid in grp_ids:
                if nid in gt_flat.name_id_to_node_id:
                    valid_ids.append(gt_flat.name_id_to_node_id[nid])
            if valid_ids: ambig_groups_ids.append(valid_ids)

        fixed_groups_ids = []
        for grp_ids, target_idx in group_data.fixed_groups:
            valid_ids = [gt_flat.name_id_to_node_id[nid] for nid in grp_ids if nid in gt_flat.name_id_to_node_id]
            if valid_ids: fixed_groups_ids.append((valid_ids, target_idx))
        
        return ambig_groups_ids, fixed_groups_ids

    @staticmethod
    def recon_lca_optimized(gt: FlatTree, st: FlatTree,
                            dup_cost: int, loss_cost: int,
                            registry: NameRegistry = None, 
                            precalc_map: Dict[int, int] = None, 
                            retmap=False) -> Union[int, ReconResult]:
        """
        O(1) LCA + Integer Array Reconciliation.
        """
        score = 0
        lca_maps = precalc_map.copy() if precalc_map else {}
        
        # Build initial maps if not provided
        if precalc_map is None:
            st_leaf_map = {} 
            for i in range(st.num_nodes):
                if st.children_start[i] == st.children_start[i+1]:
                    st_leaf_map[st.node_to_name_id[i]] = i
            
            for i in range(gt.num_nodes):
                if gt.children_start[i] == gt.children_start[i+1]:
                    name_id = gt.node_to_name_id[i]
                    if name_id in st_leaf_map:
                        lca_maps[i] = st_leaf_map[name_id]

        # Init tracking for detailed map
        node_dups = {}
        node_losses = {}
        
        # Iterate Postorder
        for u in gt.postorder:
            start = gt.children_start[u]
            end = gt.children_start[u+1]
            
            # --- Leaf Case ---
            if start == end: 
                if retmap: 
                    node_dups[u] = 0
                    node_losses[u] = 0
                continue 
            
            # --- Internal Case ---
            c1 = gt.children_flat[start]
            c2 = gt.children_flat[start+1]
            m1 = lca_maps[c1]
            m2 = lca_maps[c2]
            
            m_lca = st.get_lca(m1, m2)
            lca_maps[u] = m_lca
            
            is_dup = 0
            if m_lca == m1 or m_lca == m2:
                is_dup = 1
                score += dup_cost
                if retmap: node_dups[u] = 1
            elif retmap:
                node_dups[u] = 0

            d_lca = st.node_depths[m_lca]
            d_c1 = st.node_depths[m1]
            d_c2 = st.node_depths[m2]    
            
            loss1 = (d_c1 - d_lca - 1) + is_dup 
            loss2 = (d_c2 - d_lca - 1) + is_dup
            
            if loss1 > 0: score += (loss_cost * loss1)
            if loss2 > 0: score += (loss_cost * loss2)
            
            if retmap:
                node_losses[c1] = loss1 if loss1 > 0 else 0
                node_losses[c2] = loss2 if loss2 > 0 else 0
                # Initialize root/parent slots
                node_losses[u] = 0

        # --- Root Loss Penalty ---
        # The root of the GT is the last node in postorder (usually)
        # Or specifically nodes with no parent.
        root_id = gt.postorder[-1]
        if root_id in lca_maps:
            map_root = lca_maps[root_id]
            root_depth = st.node_depths[map_root]
            if root_depth > 0:
                score += (loss_cost * root_depth)
                if retmap: node_losses[root_id] += root_depth

        # --- Generate Map Object ---
        if retmap:
            if not registry: raise ValueError("Registry required for returning maps in flat mode")
            
            final_maps_str = {}
            final_dups_str = {}
            final_losses_str = {}
            
            for u in range(gt.num_nodes):
                if u not in lca_maps: continue
                
                # Get Names
                #u_name_id = gt.node_to_name_id[u]
                target_id = lca_maps[u]
                t_name_id = st.node_to_name_id[target_id]
                
                #u_name = registry.get_name(u_name_id)
                t_name = registry.get_name(t_name_id)

                u_full_id = gt.node_id_to_name_id[u]
                u_full_name = registry.get_name(u_full_id)
                
                final_maps_str[u_full_name] = [t_name]
                if u in node_dups: final_dups_str[u_full_name] = node_dups[u]
                if u in node_losses: final_losses_str[u_full_name] = node_losses[u]
                
            return ReconResult(
                score=score, 
                maps=[Map(
                    n_dups=sum(final_dups_str.values()), 
                    n_losses=sum(final_losses_str.values()), 
                    cor=final_maps_str, 
                    dups=final_dups_str, 
                    losses=final_losses_str
                )]
            )
            
        return score

    # --------------------------------------------------------------------------
    # DIRTY RECONCILIATION LOGIC
    # --------------------------------------------------------------------------
    
    @staticmethod
    def identify_dirty_nodes(gt_flat: FlatTree, dirty_leaves: List[int]) -> Tuple[List[int], List[bool]]:
        """
        Identifies all ancestors of dirty leaves.
        Returns: 
          - dirty_postorder: List of node IDs in post-order that are dirty.
          - dirty_mask: Boolean array of size N.
        """
        is_dirty = [False] * gt_flat.num_nodes
        
        # Mark leaves
        stack = list(dirty_leaves)
        for u in stack:
            if not is_dirty[u]:
                is_dirty[u] = True
                p = gt_flat.parents[u]
                if p != -1:
                    stack.append(p)
                    
        # Filter existing postorder to maintain valid DP order
        dirty_postorder = [u for u in gt_flat.postorder if is_dirty[u]]
        
        return dirty_postorder, is_dirty

    @staticmethod
    def calculate_static_score(gt_flat: FlatTree, st_flat: FlatTree, dup_cost: int, loss_cost: int,
                               lca_maps: Dict[int, int], dirty_mask: List[bool]) -> int:
        """
        Calculates the score contribution of all CLEAN nodes.
        Runs a standard scoring pass but ignores Dirty nodes.
        """
        score = 0
        # Iterate postorder
        for u in gt_flat.postorder:
            if dirty_mask[u]: continue # Skip dirty
            
            # Logic same as recon_lca_optimized
            start = gt_flat.children_start[u]
            end = gt_flat.children_start[u+1]
            
            if start == end: continue # Leaf
            
            c1 = gt_flat.children_flat[start]
            c2 = gt_flat.children_flat[start+1]
            m1 = lca_maps[c1]
            m2 = lca_maps[c2]
            
            # Note: Clean nodes have Clean children (by definition of ancestors).
            # So m1/m2 are static.
            
            m_lca = st_flat.get_lca(m1, m2)
            # IMPORTANT: We assume lca_maps[u] is already set correctly from a base run?
            # Or we compute it? 
            # Ideally we compute it here to ensure consistency.
            lca_maps[u] = m_lca # Update map even if clean
            
            is_dup = 0
            if m_lca == m1 or m_lca == m2:
                is_dup = 1
                score += dup_cost
            
            d_lca = st_flat.node_depths[m_lca]
            d_c1 = st_flat.node_depths[m1]
            d_c2 = st_flat.node_depths[m2]
            
            loss1 = (d_c1 - d_lca - 1) + is_dup 
            loss2 = (d_c2 - d_lca - 1) + is_dup
            
            if loss1 > 0: score += (loss_cost * loss1)
            if loss2 > 0: score += (loss_cost * loss2)

        # Root penalty (if root is clean)
        root_id = gt_flat.postorder[-1]
        if not dirty_mask[root_id] and root_id in lca_maps:
            map_root = lca_maps[root_id]
            root_depth = st_flat.node_depths[map_root]
            if root_depth > 0: score += (loss_cost * root_depth)
             
        return score

    @staticmethod
    def recon_dirty_optimized(gt_flat: FlatTree, st_flat: FlatTree, dup_cost: int, loss_cost: int,
                              dirty_postorder: List[int], dirty_mask: List[bool],
                              base_score: int, lca_maps: Dict[int, int]) -> int:
        """
        Updates maps and score ONLY for dirty nodes.
        """
        current_score = base_score
        
        for u in dirty_postorder:
            start = gt_flat.children_start[u]
            end = gt_flat.children_start[u+1]
            
            if start == end: continue # Leaf (already mapped before calling)
            
            c1 = gt_flat.children_flat[start]
            c2 = gt_flat.children_flat[start+1]
            
            m1 = lca_maps[c1]
            m2 = lca_maps[c2]
            
            m_lca = st_flat.get_lca(m1, m2)
            lca_maps[u] = m_lca
            
            is_dup = 0
            if m_lca == m1 or m_lca == m2:
                is_dup = 1
                current_score += dup_cost
            
            d_lca = st_flat.node_depths[m_lca]
            d_c1 = st_flat.node_depths[m1]
            d_c2 = st_flat.node_depths[m2]
            
            loss1 = (d_c1 - d_lca - 1) + is_dup 
            loss2 = (d_c2 - d_lca - 1) + is_dup
            
            if loss1 > 0: current_score += (loss_cost * loss1)
            if loss2 > 0: current_score += (loss_cost * loss2)
            
        # Root penalty (if root is dirty)
        root_id = gt_flat.postorder[-1]
        if dirty_mask[root_id]:
            map_root = lca_maps[root_id]
            root_depth = st_flat.node_depths[map_root]
            if root_depth > 0: current_score += (loss_cost * root_depth)
             
        return current_score

    @staticmethod
    def reconcile_permutation_optim(gt_flat: FlatTree, mul_flat: FlatTree, dup_cost: int, loss_cost: int,
                            registry: NameRegistry, group_data: GroupData, target_map: Dict[int, List[int]],
                            retmap: bool = False) -> Union[int, ReconResult]:
        
        # Translate Groups
        ambig_groups, fixed_groups = Reconciler.translate_groups_to_ids(gt_flat, group_data)

        # Build Instructions & Identify Dirty Leaves
        instructions = {}
        dirty_leaves = []
        ambig_ranges = []

        for idx, grp_ids in enumerate(ambig_groups):
            # Dynamic Range: limits combinations to the exact number of available copies
            sample_node = grp_ids[0]
            sp_base_id = gt_flat.node_to_name_id[sample_node]

            '''gt_name_id = gt_flat.node_to_name_id[sample_node]
            gt_name = registry.get_name(gt_name_id)
            sp_name = gt_name.split("_")[-1]
            sp_base_id = registry.get_id(sp_name)'''
            
            available_targets = target_map.get(sp_base_id, DEFAULT_TARGET)
            ambig_ranges.append(range(len(available_targets)))

            for nid in grp_ids: 
                instructions[nid] = (0, idx)
                dirty_leaves.append(nid)

        for grp_ids, t_idx in fixed_groups:
            for nid in grp_ids: 
                instructions[nid] = (1, t_idx)

        # --- OPTIMIZATION SETUP ---
        dirty_postorder, dirty_mask = Reconciler.identify_dirty_nodes(gt_flat, dirty_leaves)

        # Base Map (Initialize ALL leaves)
        base_leaf_targets = [None] * gt_flat.num_nodes
        lca_maps = {}
        
        for i in range(gt_flat.num_nodes):
            if gt_flat.children_start[i] == gt_flat.children_start[i+1]:
                sp_base_id = gt_flat.node_to_name_id[i]

                '''sp_name_id = gt_flat.node_to_name_id[i]
                sp_base_name = registry.get_name(sp_name_id).split("_")[-1]
                sp_base_id = registry.get_id(sp_base_name)'''
                
                targets = target_map.get(sp_base_id, DEFAULT_TARGET)
                base_leaf_targets[i] = targets
                
                # Apply initialization fix for "Clean" nodes
                # Check instructions to correctly initialize Fixed vs Ambig vs Singleton
                if i in instructions:
                    type_code, val = instructions[i]
                    if type_code == 1: # Fixed Group
                        # Initialize Fixed nodes to their actual target (e.g. H2)
                        # because they are "Clean" and won't be updated by the loop.
                        choice = val if val < len(targets) else 0
                        lca_maps[i] = targets[choice]
                    else:
                        # Ambig Group: Default to 0 for initial static calc
                        lca_maps[i] = targets[0]
                else:
                    # Singleton/Non-group: Default to 0
                    lca_maps[i] = targets[0]

        # Calculate Static Score (Cost of clean nodes)
        # We must run a full recon once to populate internal lca_maps for clean nodes!
        # Reusing standard recon logic for setup
        Reconciler.recon_lca_optimized(gt_flat, mul_flat, dup_cost, loss_cost, precalc_map=lca_maps)
        # Populate lca_maps for all nodes
        static_score = Reconciler.calculate_static_score(gt_flat, mul_flat, dup_cost, loss_cost, lca_maps, dirty_mask)
        
        # --- Permutation Loop ---
        best_score = 999999
        all_maps = []

        # Use itertools.product unpacked dynamically
        for combo in itertools.product(*ambig_ranges):
            
            # Update ONLY Ambig Leaves
            for u in dirty_leaves:
                if u in instructions:
                    type_code, val = instructions[u]
                    choice = combo[val] if type_code == 0 else val
                    
                    targets = base_leaf_targets[u]
                    if choice >= len(targets): choice = 0 # Out-of-bounds safety
                    
                    lca_maps[u] = targets[choice]

            if retmap:
                # Full recalculation for safety on return map (speed less critical here, happens once)
                res = Reconciler.recon_lca_optimized(gt_flat, mul_flat, dup_cost, loss_cost, registry, lca_maps, True)
                if res.score < best_score:
                    best_score = res.score
                    all_maps = res.maps
                elif res.score == best_score:
                    all_maps.extend(res.maps)
            else:
                # Optimized Scoring
                score = Reconciler.recon_dirty_optimized(gt_flat, mul_flat, dup_cost, loss_cost, dirty_postorder, dirty_mask, static_score, lca_maps)
                if score < best_score:
                    best_score = score
        
        if retmap:
            return ReconResult(best_score, all_maps)
        return best_score
    
    # --------------------------------------------------------------------------
    # RECONCILIATION LOGIC
    # --------------------------------------------------------------------------

    @staticmethod
    def reconcile_permutation_old(gt_flat: FlatTree, mul_flat: FlatTree, dup_cost: int, loss_cost: int,
                            registry: NameRegistry, group_data: GroupData, target_map: Dict[int, List[int]],
                            retmap: bool = False) -> Union[int, ReconResult]:
        
        # Translate Groups
        ambig_groups, fixed_groups = Reconciler.translate_groups_to_ids(gt_flat, group_data)

        # Build Instructions
        # Because the H1 clade is duplicated as a whole, it's enough to sample one node from each group to determine the target options for the entire group.
        instructions = {}
        ambig_ranges = [] 
        for idx, grp_ids in enumerate(ambig_groups):
            # Dynamic Range: limits combinations to the exact number of available copies
            # This works because TreeLinearizer.linearize() of the GT intercepts the leaf
            # nodes and cleans their names before requesting an ID from the registry.
            sample_node = grp_ids[0]
            sp_base_id = gt_flat.node_to_name_id[sample_node]
            available_targets = target_map.get(sp_base_id, DEFAULT_TARGET)
            ambig_ranges.append(range(len(available_targets)))

            for nid in grp_ids: instructions[nid] = (0, idx)

        for grp_ids, t_idx in fixed_groups:
            for nid in grp_ids: 
                instructions[nid] = (1, t_idx)
            
        # Base Map - List of Tuples
        base_leaf_targets = []
        for i in range(gt_flat.num_nodes):
            if gt_flat.children_start[i] == gt_flat.children_start[i+1]:
                sp_base_id = gt_flat.node_to_name_id[i]
                targets = target_map.get(sp_base_id, DEFAULT_TARGET)
                base_leaf_targets.append((i, targets))

        # --- Permutation Loop ---
        best_score = 999999
        all_maps = []

        for combo in itertools.product(*ambig_ranges):
            current_map = {}
            # Iterate and update ONLY the ambiguous nodes
            for u, targets in base_leaf_targets:
                if u in instructions and instructions[u][0] == 0: # Ambig Group
                    choice = combo[instructions[u][1]]
                    if choice >= len(targets): choice = 0 
                    current_map[u] = targets[choice]
                else:
                    current_map[u] = targets[0]
            
            if retmap:
                res = Reconciler.recon_lca_optimized(gt_flat, mul_flat, dup_cost, loss_cost, registry, current_map, True)
                if res.score < best_score:
                    best_score = res.score
                    all_maps = res.maps
                elif res.score == best_score:
                    all_maps.extend(res.maps)
            else:
                score = Reconciler.recon_lca_optimized(gt_flat, mul_flat, dup_cost, loss_cost, precalc_map=current_map, retmap=False)
                if score < best_score:
                    best_score = score
        
        if retmap:
            return ReconResult(best_score, all_maps)
        return best_score

    def recon_all(self, mul_trees: Dict[int, MulTree], gene_trees: Dict[int, SmrtTree],
                  registry: NameRegistry, retmap: bool = False) -> Tuple[List[Tuple[int, int]], Dict[int, Dict[int, ReconResult]]]:
        
        step = "Reconciliation"
        self.logger.report_step(step, "In progress...")
        
        # Flatten Everything
        gene_trees_flat = {}
        for idx, gt in gene_trees.items():
            gt.make_flat(registry)
            gene_trees_flat[idx] = gt.flat_tree

        for idx, mdata in mul_trees.items():
            mdata.mt.make_flat(registry)
            
        all_scores = {}
        detailed_res = {}
        tasks = list(mul_trees.items())
        gene_trees_flat_dict = {k: v.flat_tree for k, v in gene_trees.items()}
        dup_cost, loss_cost = self.tcf.weights
        
        worker_func = partial(_worker_reconcile_single, 
                              flat_gts=gene_trees_flat_dict,
                              dup_cost=dup_cost,
                              loss_cost=loss_cost,
                              registry=registry, 
                              pickle_dir=str(self.tcf.pickle_dir), 
                              run_prefix=self.tcf.run_prefix,
                              retmap=retmap, 
                              optim=self.optim
                            )
        
        if self.n_procs > 1:
            with mp.Pool(processes=self.n_procs) as pool:
                flat_tasks = [(k, v.mt.flat_tree) for k, v in tasks]
                iterator = pool.imap_unordered(worker_func, flat_tasks)
                for idx, score, gt_res in tqdm(iterator, total=len(tasks), desc="# Scoring   ", unit="st", disable=self.logger.disable_tqdm, ncols=177):
                    all_scores[idx] = score
                    if retmap:
                        detailed_res[idx] = gt_res
        else:
            #for k, v in tasks:
            for k, v in tqdm(tasks, total=len(tasks), desc="# Scoring   ", unit="st", disable=self.logger.disable_tqdm, ncols=177):
                item = (k, v.mt.flat_tree)
                idx, score, gt_res = worker_func(item)
                all_scores[idx] = score
                if retmap:
                    detailed_res[idx] = gt_res

        self.logger.report_step(step, "Success", full_update=True)
        return sorted(all_scores.items(), key=lambda x: x[1]), detailed_res
    
    def get_lowest_maps(self, sorted_scores: List[Tuple[int, int]], n_lowest: int, 
                        mul_trees: Dict[int, MulTree], gene_trees: Dict[int, SmrtTree],
                        registry: NameRegistry, enforce_input_tree: bool = False) -> Dict[int, Dict[int, ReconResult]]:
        
        step = "Getting maps for lowest scoring MTs"
        self.logger.report_step(step, "In progress...")

        if enforce_input_tree:
            # Find the ranking of the input tree (MUL-tree 0) and adjust limit to include it if necessary
            # May be None in no-st Mode, but we handle it earlier just in case
            input_rank = next((i for i, (idx, _) in enumerate(sorted_scores) if idx == 0), None)
            if input_rank is not None:
                limit = max(limit, input_rank + 1)
        limit = min(len(sorted_scores), n_lowest)

        detailed_res = {} 
        dup_cost, loss_cost = self.tcf.weights
        
        # Ensure flat structures exist (should be cached from recon_all)
        gt_flat_dict = {k: v.flat_tree for k, v in gene_trees.items()}

        for idx, total in sorted_scores[:limit]:
            mul_data = mul_trees[idx]
            mul_flat = mul_data.mt.flat_tree
            
            # Load Groups
            cur_groups = {}
            if idx != 0:
                p_path = self.tcf.pickle_dir / f"{self.tcf.run_prefix}_{idx}_groups.pickle"
                if p_path.exists():
                    with open(p_path, 'rb') as f:
                        cur_groups = pickle.load(f)
                target_map = Reconciler.build_target_map(mul_flat, registry)
            
            gt_results = {}
            for g_num, gt_flat in gt_flat_dict.items():
                if idx == 0:
                    # ST case
                    res = Reconciler.recon_lca_optimized(gt_flat, mul_flat, dup_cost, loss_cost, registry, retmap=True)
                else:
                    # MUL case
                    group_data = cur_groups.get(g_num, GroupData([], []))
                    res = Reconciler.reconcile_permutation(gt_flat, mul_flat, dup_cost, loss_cost, registry, group_data, target_map, retmap=True, optim=self.optim)
                
                gt_results[g_num] = res
            
            detailed_res[idx] = gt_results
            
        self.logger.report_step(step, f"Success: got {limit}/{len(mul_trees)} maps")
        return detailed_res
        
    def run(self, mul_trees: dict, gene_trees: dict, registry: NameRegistry) -> TaskResult:

        if registry is None: registry = NameRegistry()

        num_mts = len(mul_trees)
        max_select = self.tcf.max_select

        # -n 0 enforces having the input tree maps in the output
        enforce_input_tree = (max_select == 0)
        if self.tcf.mode == 'no-st':
            true_min = 1
            if enforce_input_tree:
                self.logger.log("Warning: Enforce input tree is enabled using -n 0 but mode is 'no-st'. Falling back to -n 1 for reconciliation.", 'w')
                enforce_input_tree = False
                max_select = 1
        else:
            # Full mode may require 2 maps at least, if top is ST
            # no-st mode doesn't need this fix
            true_min = 2
        min_select = min(true_min, num_mts)

        # Normalize selection to the num_mts range - can't select more than available MTs
        max_select_norm = max_select if max_select >= 0 and max_select < num_mts else num_mts

        limit = max(max_select_norm, min_select)
        
        if enforce_input_tree:
            # We don't know the ST rank yet, so we must score without maps first
            high_demand = False
        else:
            # High Map Demand Threshold: 10%
            high_demand = (limit > num_mts * 0.1)

        if self.optim:
            self.logger.log("Using optimized reconciliation method.", 'i')

        try:
            if high_demand:
                self.logger.log(f"High map demand detected ({limit}/{num_mts}). Generating maps directly during scoring.", 'i')
                sorted_scores, detailed_res = self.recon_all(mul_trees, gene_trees, registry, retmap=True)
                
                # Trim the detailed_res down to `limit` to save memory and I/O writing overhead
                # while sorting to keep it consistent with the output format of get_lowest_maps.
                detailed_res = {k: detailed_res[k] for k, _ in sorted_scores[:limit] if k in detailed_res}
            else:
                sorted_scores, _ = self.recon_all(mul_trees, gene_trees, registry, retmap=False)
                detailed_res = self.get_lowest_maps(sorted_scores, limit, mul_trees, gene_trees, registry, enforce_input_tree)
        finally:
            try:
                GeneTreeManager(self.tcf, self.logger, self.n_procs, self.pickle_action).handle_pickles()
            except Exception:
                # Don't re-raise errors from cleanup to avoid masking main results
                self.logger.log("Warning: Failed to clean up pickle files. Please check the pickle directory.", 'w')

        if len(detailed_res) == 2 and max_select == 1 and 0 not in detailed_res:
            # Edge Case: If user requested only 1 tree but the input tree (MUL-tree 0) is not in the top 2,
            # this means we added 1 during min_select in vein, so we must now del the second item in detailed_res
            # to enforce the user's original request of 1 tree.
            keys = list(detailed_res.keys())
            del detailed_res[keys[1]]

        # Write outputs
        self.write_detailed(detailed_res, gene_trees)
        self.write_scores_and_counts(sorted_scores, mul_trees, detailed_res)

        # Instead of keeping ReconResult, keep Maps[0] (Dict[int, Dict[int, Map]] vs Dict[int, Dict[int, ReconResult]] in StepResult)
        detailed_kept = {}
        for mul_idx in detailed_res:
            maps_dict = {g_idx: res.maps[0] for g_idx, res in detailed_res[mul_idx].items()}
            detailed_kept[mul_idx] = maps_dict

        return TaskResult(sorted_scores, mul_trees, detailed_kept, gene_trees)
    
    # --------------------------------------------------------------------------
    # WRITER LOGIC
    # --------------------------------------------------------------------------

    def write_detailed(self, detailed_res: dict, gene_trees: dict):

        def map_formatter(name: str, maps: dict, dups: dict) -> str:
            """
            Dynamically injects [Map-Dup] labels into the Newick string 
            using a formatter, completely avoiding slow tree copying/mutation.
            """
            if name in maps:
                cur_map = maps[name][0]
                # Append '+' if it mapped to the H1 (Base) copy
                if "*" not in cur_map:
                    cur_map += "+"
                dup_count = dups.get(name, 0)
                # Format: Node<|Map-Dups|> -> Node[Map-Dups]
                return f"{name}<|{cur_map}-{dup_count}|>"
            return name
        
        step = "Writing detailed output file"
        self.logger.report_step(step, "In progress...")
        
        p = Path(self.tcf.output_dir) / f"{self.tcf.run_prefix}-detailed.txt"
        with open(p, 'w') as f:
            
            header = "mul.tree\tgene.tree\tdups\tlosses\ttotal.score"
            header += "\tmaps\n" if self.to_map != 0 else "\n"
            f.write(header)

            for mul_idx, res_dict in detailed_res.items():

                f.write(f"# MUL-tree {mul_idx}\n")
                for gene_idx, res in res_dict.items():

                    # Handle multiple maps if present
                    if (maps_len := len(res.maps)) > 1:
                        f.write(f"# GT-{gene_idx+1} to MT-{mul_idx}\t{maps_len} maps found!\n")
                        
                    gt_obj = gene_trees[gene_idx]
                    for map_obj in res.maps:
                        if self.to_map:
                            map_str = gt_obj.to_str(
                                internals=True,
                                name_formatter=map_formatter,
                                maps=map_obj.cor,
                                dups=map_obj.dups
                                )
                            map_str = '\t' + map_str.replace("<|", "[").replace("|>", "]") # Avoid Newick issues with angle brackets
                        else:
                            map_str = ''
                        f.write(f"{mul_idx}\t{gene_idx+1}\t{map_obj.n_dups}\t{map_obj.n_losses}\t{res.score}{map_str}\n")
                                 
        self.logger.report_step(step, f"Success: recorded {len(detailed_res)} MTs{' with maps' if self.to_map else ''}")

    def write_scores_and_counts(self, sorted_scores: list, mul_trees: dict, detailed_res: dict):

        step = "Writing main output files"
        self.logger.report_step(step, "In progress...")
        
        p = Path(self.tcf.output_dir) / f"{self.tcf.run_prefix}-scores.txt"
        with open(p, 'w') as f:
            f.write("mul.tree\th1.node\thx.nodes\tscore\tlabeled.tree\n")
            for idx, score in sorted_scores:
                mul_data = mul_trees[idx]
                tree_str = mul_data.to_marked_str()
                h1_name = mul_data.h1_node.name if mul_data.h1_node else "NA"
                # Handle single or multiple Hx targets
                hx_names = ",".join([n.name for n in mul_data.hx_sisters]) if mul_data.hx_sisters else "NA"  
                f.write(f"{idx}\t{h1_name}\t{hx_names}\t{score}\t{tree_str}\n")

        self.write_dup_loss(detailed_res, mul_trees)

        self.logger.report_step(step, "Success")

    def write_dup_loss(self, detailed_res: dict, mul_trees: dict):
        p_dup = Path(self.tcf.output_dir) / f"{self.tcf.run_prefix}-dup-counts.txt"
        with open(p_dup, 'w') as f:
            f.write("mul.tree\tnode\tdups\tlosses\n")
            for mul_idx, res_dict in detailed_res.items():
                mul_data = mul_trees[mul_idx]
                hybrid_clade = mul_data.h_clade
                ordered_nodes = mul_data.mt.node_order
                
                # Pre-fill dictionary with 0s to guarantee NO missing rows
                main_dups = {node: 0 for node in ordered_nodes}
                main_losses = main_dups.copy()
                
                # Accumulate counts efficiently
                for g_idx, res in res_dict.items():
                    first_map = res.maps[0]
                    cor_maps = first_map.cor
                    for gt_node, count in first_map.dups.items():
                        if count > 0:
                            map_node = cor_maps[gt_node][0]
                            main_dups[map_node] += count
                    for gt_node, count in first_map.losses.items():
                        if count > 0:
                            map_node = cor_maps[gt_node][0]
                            main_losses[map_node] += count

                # Write ordered output
                for node in ordered_nodes:
                    dups = main_dups[node]
                    losses = main_losses[node]
                    out_node = node + "+" if node in hybrid_clade else node
                    f.write(f"{mul_idx}\t{out_node}\t{dups}\t{losses}\n")