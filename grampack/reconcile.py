import pickle
import itertools
import multiprocessing as mp
from tqdm import tqdm
from pathlib import Path
from functools import partial
from typing import List, Dict, Tuple, Any, Set, Union

from .config import TaskConfig
from .models import SmrtTree, MulTree, GroupData, Map, ReconResult, TaskResult, FlatTree, NameRegistry

GroupsPickle = Dict[int, GroupData]

# --- Worker Function ---

def _worker_reconcile_single(
    mul_item: Tuple[int, Any], 
    flat_gts: Dict[int, FlatTree],
    registry: NameRegistry, 
    pickle_dir: str, 
    run_prefix: str 
) -> Tuple[int, int]:
    
    mul_idx, flat_mul = mul_item
    total_score = 0
        
    # Case A: Species Tree 
    if mul_idx == 0:
        for g_num, gt_flat in flat_gts.items():

            score = Reconciler.recon_lca_optimized(gt_flat, flat_mul)
            total_score += score
            
    # Case B: MUL-Tree
    else:
        
        # Load Groups
        cur_groups = {}
        if mul_idx != 0:
            p_path = Path(pickle_dir) / f"{run_prefix}_{mul_idx}_groups.pickle"
            if p_path.exists():
                #try:
                    with open(p_path, 'rb') as f:
                        cur_groups = pickle.load(f)
                #except Exception:
                #    return mul_idx, 9999999 

        for g_num, gt_flat in flat_gts.items():

            if g_num not in cur_groups: continue
            
            group_data = cur_groups[g_num]
            score = Reconciler.reconcile_permutation(flat_mul, gt_flat, registry, group_data)
            total_score += score

    return mul_idx, total_score

class Reconciler:
    def __init__(self, config: TaskConfig, num_processes: int = 1):
        self.tcf = config
        self.num_processes = num_processes

    # --------------------------------------------------------------------------
    # GROUP COLLAPSING LOGIC (Object-based, run once per iter)
    # --------------------------------------------------------------------------

    def _get_sister_clade_labels(self, node_obj) -> List[str]:
        if not node_obj or not node_obj.up: return []
        sisters = [ch for ch in node_obj.up.children if ch != node_obj]
        labels = []
        for sis in sisters:
            # Optimize: Avoid iter_leaves if cached, but for MT it's fast enough usually
            labels.extend([l.name.split("_")[-1] for l in sis.iter_leaves()])
        return labels

    def _find_node_by_clade(self, tree: SmrtTree, target_leaves: Set[str]) -> Any:
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

    def get_sister_clades(self, mul_data: MulTree) -> Tuple[Set[str], Set[str]]:
        """
        Calculates sister clades ONCE per MUL-tree.
        """
        if mul_data.h1_node is None:
            return set(), set()
            
        h1_target = set(mul_data.h_clade)
        h2_target = {f"{x}*" for x in mul_data.h_clade}
        
        n1_obj = self._find_node_by_clade(mul_data.mt, h1_target)
        n2_obj = self._find_node_by_clade(mul_data.mt, h2_target)
        
        h1_sisters = self._get_sister_clade_labels(n1_obj) if n1_obj else []
        h2_sisters = self._get_sister_clade_labels(n2_obj) if n2_obj else []

        if n2_obj and not set(h1_sisters).isdisjoint({l.name for l in n2_obj.iter_leaves()}): h1_sisters = []
        if n1_obj and not set(h2_sisters).isdisjoint({l.name for l in n1_obj.iter_leaves()}): h2_sisters = []

        h1_clean = {x.replace("*", "") for x in h1_sisters}
        h2_clean = {x.replace("*", "") for x in h2_sisters}
        
        return h1_clean, h2_clean

    def compute_groups(self, gene_tree: SmrtTree, mul_data: MulTree, registry: NameRegistry,
                       h1_sisters: Set[str] = None, h2_sisters: Set[str] = None) -> GroupData:
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

        # Sister checking (Logic using string sets for safety/easier debugging)
        if mul_data.h1_node and (h1_sisters is None):
            h1_sisters, h2_sisters = self.get_sister_clades(mul_data)

        def to_ids(names):
            return [registry.get_id(n) for n in names]

        def check_fix(unit_nodes, anc_leaves):
            # unit_nodes is list of strings
            
            group_sis_specs = [n.split("_")[-1] for n in anc_leaves]
            
            if group_sis_specs:
                if h1_sisters and all(s in h1_sisters for s in group_sis_specs):
                    final_fixed.append((to_ids(unit_nodes), ''))
                    return
                elif h2_sisters and all(s in h2_sisters for s in group_sis_specs):
                    final_fixed.append((to_ids(unit_nodes), '*'))
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
    def translate_groups_to_ids(
        group_data: GroupData, 
        gt_flat: FlatTree, 
        registry: NameRegistry
    ) -> Tuple[List[List[int]], List[Tuple[List[int], str]]]:
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
                score += 1
                if retmap: node_dups[u] = 1
            elif retmap:
                node_dups[u] = 0
            
            d_lca = st.depths[st.first_visit[m_lca]]
            d_c1 = st.depths[st.first_visit[m1]]
            d_c2 = st.depths[st.first_visit[m2]]
            
            loss1 = (d_c1 - d_lca - 1) + is_dup 
            loss2 = (d_c2 - d_lca - 1) + is_dup
            
            if loss1 > 0: score += loss1
            if loss2 > 0: score += loss2
            
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
                score += root_depth
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

    @staticmethod
    def reconcile_permutation(mul_flat: FlatTree, gt_flat: FlatTree, registry: NameRegistry,
                            group_data: GroupData, retmap: bool = False) -> Union[int, ReconResult]:
        
        # 0. Translate Groups (String -> Int)
        ambig_groups, fixed_groups = Reconciler.translate_groups_to_ids(group_data, gt_flat, registry)

        # 1. Pre-calculate Targets (Same as before)
        target_map = {} 
        for i in range(mul_flat.num_nodes):
            if mul_flat.children_start[i] == mul_flat.children_start[i+1]: 
                name_id = mul_flat.node_to_name_id[i]
                if name_id == -1: continue
                sp_name = registry.get_name(name_id)
                base_name = sp_name.replace("*", "")
                base_id = registry.get_id(base_name)
                if base_id not in target_map: target_map[base_id] = [-1, -1]
                if "*" in sp_name: target_map[base_id][1] = i
                else: target_map[base_id][0] = i
        
        for bid, targets in target_map.items():
            if targets[0] == -1: targets[0] = targets[1]
            if targets[1] == -1: targets[1] = targets[0]

        # 2. Build Instructions
        instructions = {}
        for idx, grp_ids in enumerate(ambig_groups):
            for nid in grp_ids: instructions[nid] = (0, idx)
        for grp_ids, suffix in fixed_groups:
            t_idx = 1 if suffix == "*" else 0
            for nid in grp_ids: instructions[nid] = (1, t_idx)

        # 3. Base Map
        base_leaf_targets = {}
        for i in range(gt_flat.num_nodes):
            if gt_flat.children_start[i] == gt_flat.children_start[i+1]:
                sp_name_id = gt_flat.node_to_name_id[i]
                if sp_name_id in target_map:
                    base_leaf_targets[i] = target_map[sp_name_id]

        # 4. Permutation Loop
        best_score = 999999
        all_maps = []

        for combo in itertools.product([0, 1], repeat=len(ambig_groups)):
            current_map = {}
            for u, targets in base_leaf_targets.items():
                if u in instructions:
                    type_code, val = instructions[u]
                    choice = combo[val] if type_code == 0 else val
                    current_map[u] = targets[choice]
                else:
                    current_map[u] = targets[0]
            
            if retmap:
                res = Reconciler.recon_lca_optimized(gt_flat, mul_flat, registry, current_map, True)
                if res.score < best_score:
                    best_score = res.score
                    all_maps = res.maps
                elif res.score == best_score:
                    all_maps.extend(res.maps)
            else:
                score = Reconciler.recon_lca_optimized(gt_flat, mul_flat, precalc_map=current_map, retmap=False)
                if score < best_score:
                    best_score = score
        
        if retmap:
            return ReconResult(best_score, all_maps)
        return best_score

    def recon_all(self, mul_trees: Dict[int, MulTree], gene_trees: Dict[int, SmrtTree], registry: NameRegistry, 
                  pickle_dir: str, run_prefix: str, n_proc: int, logger: Any) -> List[Tuple[int, int]]:
        
        step = "Reconciliation"
        logger.report_step(step, "In progress...")
        
        # Flatten Everything
        gene_trees_flat = {}
        for idx, gt in gene_trees.items():
            gt.make_flat(registry)
            gene_trees_flat[idx] = gt.flat_tree

        for idx, mdata in mul_trees.items():
            mdata.mt.make_flat(registry)
            
        all_scores = {}
        tasks = list(mul_trees.items())
        gene_trees_flat_dict = {k: v.flat_tree for k, v in gene_trees.items()}
        
        worker_func = partial(_worker_reconcile_single, 
                              flat_gts=gene_trees_flat_dict,
                              registry=registry, 
                              pickle_dir=str(pickle_dir), 
                              run_prefix=run_prefix)
        
        if n_proc > 1:
            with mp.Pool(processes=n_proc) as pool:
                flat_tasks = [(k, v.mt.flat_tree) for k, v in tasks]
                #for idx, score in pool.imap_unordered(worker_func, flat_tasks):
                iterator = pool.imap_unordered(worker_func, flat_tasks)
                for idx, score in tqdm(iterator, total=len(tasks), desc="Scoring", unit="mt", disable=logger.verbosity < 3):
                    all_scores[idx] = score
        else:
            #for k, v in tasks:
            for k, v in tqdm(tasks, total=len(tasks), desc="Scoring", unit="mt", disable=logger.verbosity < 3):
                item = (k, v.mt.flat_tree)
                idx, score = worker_func(item)
                all_scores[idx] = score

        logger.report_step(step, "Success")
        return sorted(all_scores.items(), key=lambda x: x[1])
    
    # Updated signature to accept registry
    def get_lowest_maps(self, sorted_scores: List[Tuple[int, int]], n_lowest: int, 
                        mul_trees: Dict[int, MulTree], gene_trees: Dict[int, SmrtTree], registry: NameRegistry,
                        pickle_dir: str, run_prefix: str, logger: Any) -> Dict[int, Dict[int, ReconResult]]:
        
        step = "Getting maps for lowest scoring MTs"
        logger.report_step(step, "In progress...")
        detailed_res = {} 
        limit = min(len(sorted_scores), n_lowest)
        
        # Ensure flat structures exist (should be cached from recon_all)
        gt_flat_dict = {k: v.flat_tree for k, v in gene_trees.items()}

        for idx, total in sorted_scores[:limit]:
            mul_data = mul_trees[idx]
            mul_flat = mul_data.mt.flat_tree
            
            # Load Groups
            cur_groups = {}
            if idx != 0:
                p_path = Path(pickle_dir) / f"{run_prefix}_{idx}_groups.pickle"
                if p_path.exists():
                    with open(p_path, 'rb') as f:
                        cur_groups = pickle.load(f)
            
            gt_results = {}
            for g_num, gt_flat in gt_flat_dict.items():
                if idx == 0:
                    # ST case
                    res = Reconciler.recon_lca_optimized(gt_flat, mul_flat, registry, retmap=True)
                else:
                    # MUL case
                    group_data = cur_groups.get(g_num, GroupData([], []))
                    res = Reconciler.reconcile_permutation(mul_flat, gt_flat, registry, group_data, retmap=True)
                
                gt_results[g_num] = res
            
            detailed_res[idx] = gt_results
            
        logger.report_step(step, "Success")
        return detailed_res
        
    def run(self, mul_trees: dict, gene_trees: dict, registry: NameRegistry, logger: Any, writer: Any) -> TaskResult:

        pickle_dir, run_prefix, = self.tcf.pickle_dir, self.tcf.run_prefix
        n_proc = self.num_processes
        
        if registry is None: registry = NameRegistry()

        sorted_scores = self.recon_all(mul_trees, gene_trees, registry, pickle_dir, run_prefix, n_proc, logger)
        
        # If negative, output all maps
        if self.tcf.to_map < 0:
            n_lowest = len(mul_trees)
        else:
            n_lowest = max(self.tcf.to_map, self.tcf.max_select)
        detailed_res = self.get_lowest_maps(sorted_scores, n_lowest, mul_trees, gene_trees, registry, pickle_dir, run_prefix, logger)
        
        writer.write_results(sorted_scores, detailed_res, mul_trees, gene_trees)

        # Get the first k,v pair from detailed_res
        detailed_res_limited = {}
        for mul_idx in detailed_res:
            # Instead of keeping ReconResult, keep Maps[0] (Dict[int, Dict[int, Map]] vs Dict[int, Dict[int, ReconResult]] in StepResult)
            maps_dict = {g_idx: res.maps[0] for g_idx, res in detailed_res[mul_idx].items()}
            detailed_res_limited[mul_idx] = maps_dict
            # Check if idx 0 (input tree) is a key in the dict yet
            is_input_in = 0 in detailed_res_limited
            if len(detailed_res_limited) >= self.tcf.max_select + int(is_input_in):
                # If input tree is included, allow one extra
                # otherwise, we might not get enough inferred MTs
                break

        return TaskResult(
            sorted_scores=sorted_scores,
            mul_trees=mul_trees,
            kept_mul_maps=detailed_res_limited, # this dict is sorted
            gene_trees=gene_trees
        )