import pickle
import itertools
import multiprocessing as mp
from tqdm import tqdm
from pathlib import Path
from functools import partial
from typing import List, Dict, Tuple, Any, Set, Union, Optional

from .config import TaskConfig
from .logger import GranLogger
from .models import SmrtTree, MulTree, GroupData, Map, ReconResult, TaskResult, FlatTree, NameRegistry

GroupsPickle = Dict[int, GroupData]

# --- Worker Function ---

def _worker_reconcile_single(
    mul_item: Tuple[int, Any], 
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
    def __init__(self, config: TaskConfig, logger: GranLogger, num_processes: int = 1, optim: bool = False):
        self.tcf = config
        self.num_processes = num_processes
        self.optim = optim
        self.logger = logger
    
    @staticmethod
    def reconcile_permutation(gt_flat, mul_flat, dup_cost, loss_cost, registry, group_data, target_map, retmap=False, optim=False):
        if optim:
            res = Reconciler.reconcile_permutation_optim(gt_flat, mul_flat, dup_cost, loss_cost, registry, group_data, target_map, retmap=retmap)
        else:
            res = Reconciler.reconcile_permutation_old(gt_flat, mul_flat, dup_cost, loss_cost, registry, group_data, target_map, retmap=retmap)
        return res
    # --------------------------------------------------------------------------
    # GROUP COLLAPSING LOGIC (Object-based, run once per iter)
    # --------------------------------------------------------------------------

    @staticmethod
    def _get_sister_clade_labels(node_obj) -> List[str]:
        if not node_obj or not node_obj.up: return []
        sisters = [ch for ch in node_obj.up.children if ch != node_obj]
        labels = []
        for sis in sisters:
            # Optimize: Avoid iter_leaves if cached, but for MT it's fast enough usually
            labels.extend([l.name.split("_")[-1] for l in sis.iter_leaves()])
        return labels

    @staticmethod
    def _find_node_by_clade(tree: SmrtTree, target_leaves: Set[str]) -> Any:
        # OPTIMIZATION: Use Dictionary Lookup O(1) instead of search_nodes O(N)
        leaf_nodes = []
        for t in target_leaves:
            # Use get_node from GrandmaTree wrapper
            node = tree.get_node(t)
            if node: leaf_nodes.append(node)
        
        if not leaf_nodes: return None
        lca = tree.ete_tree.get_common_ancestor(leaf_nodes)
        lca_leaves = {l.name for l in lca.iter_leaves()}
        if lca_leaves == target_leaves: return lca
        return None

    @staticmethod
    def get_sister_clades(mul_data: MulTree) -> Tuple[Set[str], List[Set[str]]]:
        """
        Returns:
          1. h1_sisters: Set of names indicating H1 (Base) placement.
          2. hx_sisters_list: List[Set[str]] where index 0 -> H2, 1 -> H3.
        """
        if mul_data.h1_node is None:
            return set(), []
            
        h1_target = set(mul_data.h_clade)
        n1_obj = Reconciler._find_node_by_clade(mul_data.mt, h1_target)
        h1_sisters = set(Reconciler._get_sister_clade_labels(n1_obj) if n1_obj else [])

        hx_sisters_list = []
        targets = mul_data.hx_nodes if mul_data.hx_nodes else ([mul_data.h2_node] if mul_data.h2_node else [])
        
        for hx_node in targets:
            if hx_node:
                sisters = Reconciler._get_sister_clade_labels(hx_node)
                if not set(h1_sisters).isdisjoint({l.name for l in hx_node.iter_leaves()}): 
                    h1_sisters = set()
                if n1_obj and not set(sisters).isdisjoint({l.name for l in n1_obj.iter_leaves()}): 
                    sisters = []
                # Append Set of clean names directly to the list
                hx_sisters_list.append({s.replace("*", "") for s in sisters})
            else:
                hx_sisters_list.append(set())

        h1_sisters = {x.replace("*", "") for x in h1_sisters}

        return h1_sisters, hx_sisters_list

    @staticmethod
    def compute_groups(gene_tree: SmrtTree, mul_data: MulTree, registry: NameRegistry,
                       h1_sisters: Set[str] = None, hx_sisters_list: List[Set[str]] = None) -> GroupData:
        """
        Registry-Optimized O(N) implementation.
        Uses integer IDs for Set operations (Union/IsSubset) to achieve significant speedup.
        """
        # 1. Pre-computation: Convert targets to Integer IDs
        # This allows O(1) lookups and fast set operations
        h1_target_ids = set()
        for name in mul_data.h_clade:
            h1_target_ids.add(registry.get_id(name))
            
        groups = {} 
        singles = {} 
        
        # Cache: node -> (species_id_set, leaf_names_list, active_roots)
        # species_id_set: Set[int] - much faster than Set[str]
        node_info = {}

        ginfo = gene_tree.ete_tree

        for node in ginfo.traverse("postorder"):
            if node.is_leaf():
                # Extract name and convert to ID
                sp_name = node.name.split("_")[-1]
                sp_id = registry.get_id(sp_name)
                
                is_h1 = sp_id in h1_target_ids
                
                s_set = {sp_id}
                l_list = [node.name]
                
                if is_h1:
                    singles[node.name] = [] 
                    a_roots = [node.name]
                else:
                    a_roots = []
                
                node_info[node] = (s_set, l_list, a_roots)
                
            else:
                children = node.children
                
                u_s_set = set()
                u_l_list = []
                u_a_roots = []
                
                all_h1_descendants = True
                total_species_count = 0
                
                for child in children:
                    c_s_set, c_l_list, c_a_roots = node_info[child]
                    
                    u_s_set.update(c_s_set)
                    u_l_list.extend(c_l_list)
                    u_a_roots.extend(c_a_roots)
                    total_species_count += len(c_s_set)
                    
                    # Integer set subset check is highly optimized
                    if not c_s_set.issubset(h1_target_ids):
                        all_h1_descendants = False
                
                children_disjoint = (len(u_s_set) == total_species_count)

                if all_h1_descendants and children_disjoint and len(children) > 1:
                    # Valid Group
                    for r in u_a_roots:
                        if r in groups: del groups[r]
                        if r in singles: del singles[r]
                    
                    groups[node.name] = [u_l_list, []]
                    u_a_roots = [node.name]
                
                node_info[node] = (u_s_set, u_l_list, u_a_roots)

        # --- Post-Processing ---
        
        def fill_anc_leaves(n_name, is_group):
            n_obj = gene_tree.get_node(n_name)
            if not n_obj or not n_obj.up: return
            
            p_obj = n_obj.up
            if p_obj in node_info and n_obj in node_info:
                p_leaves = node_info[p_obj][1]
                n_leaves_set = set(node_info[n_obj][1])
                anc_list = [l for l in p_leaves if l not in n_leaves_set]
                
                if is_group:
                    groups[n_name][1] = anc_list
                else:
                    singles[n_name] = anc_list

        for g_name in groups: fill_anc_leaves(g_name, True)
        for s_name in singles: fill_anc_leaves(s_name, False)

        # --- Fixes Logic ---
        final_ambiguous = [] # List of List[int]
        final_fixed = []     # List of (List[int], str)

        # Sister checking
        if mul_data.h1_node and (h1_sisters is None):
            h1_sisters, hx_sisters_list = Reconciler.get_sister_clades(mul_data)

        def to_ids(names):
            return [registry.get_id(n) for n in names]

        def check_fix(unit_nodes, anc_leaves):
            # Convert to Set for fast subset math
            group_sis_specs = {n.split("_")[-1] for n in anc_leaves}
            
            if group_sis_specs:
                if h1_sisters and group_sis_specs.issubset(h1_sisters):
                    # Index 0 corresponds to the Base/H1 target
                    final_fixed.append((to_ids(unit_nodes), 0))
                    return
                
                if hx_sisters_list:
                    for t_idx, sis_set in enumerate(hx_sisters_list):
                        if sis_set and group_sis_specs.issubset(sis_set):
                            # Index t_idx + 1 corresponds to H2 (1), H3 (2), etc.
                            final_fixed.append((to_ids(unit_nodes), t_idx + 1))
                            return
            
            final_ambiguous.append(to_ids(unit_nodes))

        for g_leaves, anc_leaves in groups.values():
            check_fix(g_leaves, anc_leaves)
            
        for s_name, anc_leaves in singles.items():
            check_fix([s_name], anc_leaves)

        return GroupData(final_ambiguous, final_fixed)

    # --------------------------------------------------------------------------
    # RECONCILIATION LOGIC (Unified Flat)
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
        for grp_ids, suffix in group_data.fixed_groups:
            valid_ids = []
            for nid in grp_ids:
                if nid in gt_flat.name_id_to_node_id:
                    valid_ids.append(gt_flat.name_id_to_node_id[nid])
            if valid_ids: fixed_groups_ids.append((valid_ids, suffix))
        
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
            
            d_lca = st.depths[st.first_visit[m_lca]]
            d_c1 = st.depths[st.first_visit[m1]]
            d_c2 = st.depths[st.first_visit[m2]]
            
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
            root_depth = st.depths[st.first_visit[map_root]]
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

    # --- Dirty Node Optimization ---
    
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
            
            d_lca = st_flat.depths[st_flat.first_visit[m_lca]]
            d_c1 = st_flat.depths[st_flat.first_visit[m1]]
            d_c2 = st_flat.depths[st_flat.first_visit[m2]]
            
            loss1 = (d_c1 - d_lca - 1) + is_dup 
            loss2 = (d_c2 - d_lca - 1) + is_dup
            
            if loss1 > 0: score += (loss_cost * loss1)
            if loss2 > 0: score += (loss_cost * loss2)

        # Root penalty (if root is clean)
        root_id = gt_flat.postorder[-1]
        if not dirty_mask[root_id] and root_id in lca_maps:
            map_root = lca_maps[root_id]
            root_depth = st_flat.depths[st_flat.first_visit[map_root]]
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
            
            d_lca = st_flat.depths[st_flat.first_visit[m_lca]]
            d_c1 = st_flat.depths[st_flat.first_visit[m1]]
            d_c2 = st_flat.depths[st_flat.first_visit[m2]]
            
            loss1 = (d_c1 - d_lca - 1) + is_dup 
            loss2 = (d_c2 - d_lca - 1) + is_dup
            
            if loss1 > 0: current_score += (loss_cost * loss1)
            if loss2 > 0: current_score += (loss_cost * loss2)
            
        # Root penalty (if root is dirty)
        root_id = gt_flat.postorder[-1]
        if dirty_mask[root_id]:
             map_root = lca_maps[root_id]
             root_depth = st_flat.depths[st_flat.first_visit[map_root]]
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
            gt_name_id = gt_flat.node_to_name_id[sample_node]
            gt_name = registry.get_name(gt_name_id)
            sp_name = gt_name.split("_")[-1]
            sp_base_id = registry.get_id(sp_name)
            
            available_targets = target_map.get(sp_base_id, [0])
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
        base_leaf_targets = {}
        lca_maps = {}
        
        for i in range(gt_flat.num_nodes):
            if gt_flat.children_start[i] == gt_flat.children_start[i+1]:
                sp_name_id = gt_flat.node_to_name_id[i]
                sp_base_name = registry.get_name(sp_name_id).split("_")[-1]
                sp_base_id = registry.get_id(sp_base_name)
                
                targets = target_map.get(sp_base_id, [0])
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
    
    @staticmethod
    def reconcile_permutation_old(gt_flat: FlatTree, mul_flat: FlatTree, dup_cost: int, loss_cost: int,
                            registry: NameRegistry, group_data: GroupData, target_map: Dict[int, List[int]],
                            retmap: bool = False) -> Union[int, ReconResult]:
        
        # Translate Groups
        ambig_groups, fixed_groups = Reconciler.translate_groups_to_ids(gt_flat, group_data)

        # Build Instructions
        instructions = {}
        ambig_ranges = [] 
        for idx, grp_ids in enumerate(ambig_groups):
            sample_node = grp_ids[0]
            gt_name_id = gt_flat.node_to_name_id[sample_node]
            gt_name = registry.get_name(gt_name_id)
            sp_name = gt_name.split("_")[-1]
            sp_base_id = registry.get_id(sp_name)
            
            available_targets = target_map.get(sp_base_id, [0])
            ambig_ranges.append(range(len(available_targets)))

            for nid in grp_ids: instructions[nid] = (0, idx)

        for grp_ids, t_idx in fixed_groups:
            for nid in grp_ids: 
                instructions[nid] = (1, t_idx)

        # Base Map
        base_leaf_targets = {}
        for i in range(gt_flat.num_nodes):
            if gt_flat.children_start[i] == gt_flat.children_start[i+1]:
                sp_name_id = gt_flat.node_to_name_id[i]
                sp_base_name = registry.get_name(sp_name_id).split("_")[-1]
                sp_base_id = registry.get_id(sp_base_name)
                
                targets = target_map.get(sp_base_id, [0])
                base_leaf_targets[i] = targets

        # --- Permutation Loop ---
        best_score = 999999
        all_maps = []

        for combo in itertools.product(*ambig_ranges):
            current_map = {}
            for u, targets in base_leaf_targets.items():
                if u in instructions:
                    type_code, val = instructions[u]
                    choice = combo[val] if type_code == 0 else val
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
        
        if self.num_processes > 1:
            with mp.Pool(processes=self.num_processes) as pool:
                flat_tasks = [(k, v.mt.flat_tree) for k, v in tasks]
                #for idx, score in pool.imap_unordered(worker_func, flat_tasks):
                iterator = pool.imap_unordered(worker_func, flat_tasks)
                for idx, score, gt_res in tqdm(iterator, total=len(tasks), desc="Scoring   ", unit="st", disable=self.logger.verbosity < 3):
                    all_scores[idx] = score
                    if retmap:
                        detailed_res[idx] = gt_res
        else:
            #for k, v in tasks:
            for k, v in tqdm(tasks, total=len(tasks), desc="Scoring   ", unit="st", disable=self.logger.verbosity < 3):
                item = (k, v.mt.flat_tree)
                idx, score, gt_res = worker_func(item)
                all_scores[idx] = score
                if retmap:
                    detailed_res[idx] = gt_res

        self.logger.report_step(step, "Success")
        return sorted(all_scores.items(), key=lambda x: x[1]), detailed_res
    
    def get_lowest_maps(self, sorted_scores: List[Tuple[int, int]], n_lowest: int, 
                        mul_trees: Dict[int, MulTree], gene_trees: Dict[int, SmrtTree],
                        registry: NameRegistry) -> Dict[int, Dict[int, ReconResult]]:
        
        step = "Getting maps for lowest scoring MTs"
        self.logger.report_step(step, "In progress...")
        detailed_res = {} 
        limit = min(len(sorted_scores), n_lowest)
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
            
        self.logger.report_step(step, "Success")
        return detailed_res
        
    def run(self, mul_trees: dict, gene_trees: dict, registry: NameRegistry, writer: Any) -> TaskResult:

        if registry is None: registry = NameRegistry()

        num_mts = len(mul_trees)
        limit = self.tcf.to_map if self.tcf.to_map >= 0 else num_mts
        # Full mode may require 2 maps at least
        # Can't select more than available MTs
        corrected_max_select = min(self.tcf.max_select+1, num_mts)
        limit = max(limit, corrected_max_select)

        if self.optim:
            self.logger.log("Using optimized reconciliation method.", 'i')
        
        # High Map Demand Threshold: 10%
        high_demand = (limit > num_mts * 0.1)

        if high_demand:
            self.logger.log("High map demand detected. Generating maps directly during scoring.", 'i')
            sorted_scores, detailed_res = self.recon_all(mul_trees, gene_trees, registry, retmap=True)
            
            # Trim the detailed_res down to `limit` to save memory and I/O writing overhead
            # while sorting to keep it consistent with the output format of get_lowest_maps.
            detailed_res = {k: detailed_res[k] for k, _ in sorted_scores[:limit] if k in detailed_res}

        else:
            sorted_scores, _ = self.recon_all(mul_trees, gene_trees, registry, retmap=False)
            detailed_res = self.get_lowest_maps(sorted_scores, limit, mul_trees, gene_trees, registry)

        writer.write_results(sorted_scores, detailed_res, mul_trees, gene_trees)

        # Get the first k,v pair from detailed_res
        # This dict will be "sorted"
        detailed_kept = {}
        is_input_in = 0
        for mul_idx in detailed_res:
            # Instead of keeping ReconResult, keep Maps[0] (Dict[int, Dict[int, Map]] vs Dict[int, Dict[int, ReconResult]] in StepResult)
            maps_dict = {g_idx: res.maps[0] for g_idx, res in detailed_res[mul_idx].items()}
            detailed_kept[mul_idx] = maps_dict
            # Check if idx 0 (input tree) is a key in the dict yet
            if mul_idx == 0:
                is_input_in = 1
            if len(detailed_kept) >= self.tcf.max_select + is_input_in:
                # If input tree is included, allow one extra
                # otherwise, we might not get enough inferred MTs
                break

        return TaskResult(sorted_scores, mul_trees, detailed_kept, gene_trees)