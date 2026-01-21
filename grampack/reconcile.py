import sys
import pickle
import itertools
import multiprocessing as mp
from functools import partial
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Set, Optional, Union

from .config import GrandmaConfig
from .models import TreeNode, SmrtTree, MulTree, GroupData, Map, ReconResult, StepResult

GroupsPickle = Dict[int, GroupData]

# --- Worker Function ---

def _worker_reconcile_single(
    mul_item: Tuple[int, MulTree], 
    gene_trees: Dict[int, SmrtTree], 
    pickle_dir: str, 
    run_prefix: str
) -> Tuple[int, int]:
    
    mul_idx, mul_data = mul_item
    
    # Initialize LCA cache
    mul_data.mt.build_lca_cache()
    
    p_path = Path(pickle_dir) / f"{run_prefix}_{mul_idx}_groups.pickle"
    
    cur_groups_dict: GroupsPickle = {}
    if mul_idx != 0:
        if not p_path.exists():
            return mul_idx, 9999999
        try:
            with open(p_path, 'rb') as f:
                cur_groups_dict = pickle.load(f)
        except Exception:
             return mul_idx, 9999999

    total_score = 0
    
    # --- 1. Species Tree Case (Standard LCA) ---
    if mul_idx == 0:
        st_lookup = {n.name: n for n in mul_data.mt.ete_tree.traverse()}
        
        for g_num, gt_obj in gene_trees.items():
            init_maps = {}
            
            # Generate node list ONCE
            gt_nodes = list(gt_obj.ete_tree.traverse("postorder"))
            
            for n in gt_nodes:
                if n.is_leaf():
                    # FIX: Ensure we split the GeneID from SpeciesID (1_a -> a)
                    # clean_name handles '*' removal, split handles ID extraction
                    raw_name = getattr(n, "clean_name", n.name)
                    sp_name = raw_name.split("_")[-1]
                    
                    if sp_name in st_lookup:
                        init_maps[n] = [st_lookup[sp_name]]
                    else:
                        # PRINT TREE FOR DEBUGGING
                        raise ValueError(f"Gene Tree {g_num} tip '{n.name}' maps to species '{sp_name}', which is not in the Species Tree:",
                                         mul_data.mt.ete_tree.get_ascii(show_internal=True), f"Gene Tree:\n{gt_obj.ete_tree.get_ascii(show_internal=True)}")
                else:
                    init_maps[n] = []
            
            score = Reconciler.recon_lca_optimized(gt_obj, mul_data.mt, init_maps, False, precomputed_postorder=gt_nodes)
            total_score += score
            
    # --- 2. MUL-Tree Case (Permutation Reconciliation) ---
    else:
        for g_num, groups in cur_groups_dict.items():
            if g_num not in gene_trees: continue 
            gt_obj = gene_trees[g_num]
            score = Reconciler.reconcile_one_static(mul_data, gt_obj, groups, False)
            total_score += score
        
    return mul_idx, total_score

class Reconciler:
    def __init__(self, species_tree: SmrtTree, config: Any):
        self.st = species_tree
        self.cfg = config

    # --------------------------------------------------------------------------
    # GROUP COLLAPSING LOGIC (Optimized)
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
            if node:
                leaf_nodes.append(node)
        
        if not leaf_nodes: return None
        
        # If the set matches exactly, the LCA is the node that covers them
        # Note: In a MUL tree, getting LCA of all leaves in a clade returns the clade root
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

    def compute_groups(self, gene_tree: SmrtTree, mul_data: MulTree, 
                       h1_sisters: Set[str] = None, h2_sisters: Set[str] = None) -> GroupData:
        return self._compute_groups(gene_tree, mul_data, h1_sisters, h2_sisters)

    def _compute_groups(self, gene_tree: SmrtTree, mul_data: MulTree, 
                        h1_sisters_clean: Set[str] = None, h2_sisters_clean: Set[str] = None) -> GroupData:
        
        # 1. Identify Candidate Groups (O(N) using DP)
        h1_target = set(mul_data.h_clade)
        groups = {}
        singles = {}
        gt_nodes = list(gene_tree.ete_tree.traverse("postorder"))

        # OPTIMIZATION: Cache the species sets for every node to avoid re-traversing children
        # Map: TreeNode -> Set[SpeciesName]
        node_species_map: Dict[TreeNode, Set[str]] = {}
        # Map: TreeNode -> List[LeafName] (for storing in 'groups'/'singles')
        node_leaf_names: Dict[TreeNode, List[str]] = {}

        for node in gt_nodes:
            if node.is_leaf():
                sp_name = node.name.split("_")[-1]
                node_species_map[node] = {sp_name}
                node_leaf_names[node] = [node.name]
                
                # Check for single match
                if sp_name in h1_target:
                    parent = node.up
                    if parent:
                        # We can't fully know parent's other children yet in postorder loop if we look 'up',
                        # but singles logic relies on parent context.
                        # Legacy logic: "anc_clade = [l.name for l in parent.iter_leaves() if l.name != node.name]"
                        # This part is slightly expensive but only runs for target leaves.
                        # We can optimize by waiting for parent processing, but for now let's leave this 
                        # specific "single" check as-is or optimize slightly.
                        anc_clade = [l.name for l in parent.iter_leaves() if l.name != node.name]
                        singles[node.name] = anc_clade
            else:
                # Internal Node: Aggregate from children
                children = node.children
                current_specs = set()
                current_leaf_names = []
                
                for ch in children:
                    current_specs.update(node_species_map[ch])
                    current_leaf_names.extend(node_leaf_names[ch])
                
                node_species_map[node] = current_specs
                node_leaf_names[node] = current_leaf_names

                if len(children) == 2:
                    d1, d2 = children[0], children[1]
                    d1_specs = node_species_map[d1]
                    d2_specs = node_species_map[d2]
                    
                    # Logic: Both children must be PURELY hybrid
                    # Python's set.issubset is optimized C
                    d1_is_hybrid = d1_specs.issubset(h1_target)
                    d2_is_hybrid = d2_specs.issubset(h1_target)
                    
                    if d1_is_hybrid and d2_is_hybrid:
                        # Logic: Must be disjoint
                        if d1_specs.isdisjoint(d2_specs):
                            # Found a group!
                            # Remove descendants from groups
                            for desc in node.iter_descendants():
                                if desc.name in groups: del groups[desc.name]
                            
                            groups[node.name] = current_leaf_names

        # Cleanup singles
        for grp_node in groups:
            for leaf in groups[grp_node]:
                if leaf in singles: del singles[leaf]

        final_ambiguous = []
        final_fixed = []

        # 2. Fix Groups using Sisters
        if mul_data.h1_node is not None:
            if h1_sisters_clean is None or h2_sisters_clean is None:
                h1_sisters_clean, h2_sisters_clean = self.get_sister_clades(mul_data)

            def process_unit(unit_nodes: List[str]):
                # Re-fetching nodes is fast with node_map in GrandmaTree
                if len(unit_nodes) == 1: 
                    g_node = gene_tree.get_node(unit_nodes[0])
                else:
                    n_objs = [gene_tree.get_node(x) for x in unit_nodes]
                    g_node = gene_tree.ete_tree.get_common_ancestor(n_objs)
                
                if not g_node or not g_node.up:
                    final_ambiguous.append(unit_nodes)
                    return
                
                # Get sisters of the gene node
                # Note: We can optimize this too, but it's only for the formed groups (few)
                gt_sisters = self._get_sister_clade_labels(g_node)
                if not gt_sisters:
                    final_ambiguous.append(unit_nodes)
                    return

                # Check intersection
                if h1_sisters_clean and all(s in h1_sisters_clean for s in gt_sisters):
                    final_fixed.append((unit_nodes, '')) 
                elif h2_sisters_clean and all(s in h2_sisters_clean for s in gt_sisters):
                    final_fixed.append((unit_nodes, '*'))
                else:
                    final_ambiguous.append(unit_nodes)

            for g_list in groups.values(): process_unit(g_list)
            for s_node in singles: process_unit([s_node])
        else:
            final_ambiguous.extend(groups.values())
            for s_node in singles: final_ambiguous.append([s_node])

        return GroupData(final_ambiguous, final_fixed)

    # --------------------------------------------------------------------------
    # RECONCILIATION LOGIC (OPTIMIZED)
    # --------------------------------------------------------------------------

    @staticmethod
    def recon_lca_optimized(gene_tree: SmrtTree, map_target_tree: SmrtTree, 
                         initial_maps: Dict[TreeNode, List[TreeNode]], retmap=False, 
                         precomputed_postorder: List[TreeNode] = None) -> Union[int, ReconResult]:
        """
        Highly optimized LCA reconciliation.
        """
        score = 0
        lca_maps = initial_maps.copy()
        
        node_dups_obj = {}
        node_losses_obj = {}
        if retmap:
            for n in gene_tree.ete_tree.traverse():
                node_dups_obj[n] = 0
                node_losses_obj[n] = 0
        
        nodes = precomputed_postorder if precomputed_postorder else gene_tree.ete_tree.traverse("postorder")
        
        for node in nodes:
            if node.is_leaf(): continue
            
            children = node.children
            if len(children) < 2: continue
            
            # Retrieve maps (guaranteed to exist due to identity checks upstream)
            c1_map = lca_maps[children[0]][0]
            c2_map = lca_maps[children[1]][0]
            
            # Cached LCA lookup
            map_node = map_target_tree.get_lca_obj([c1_map, c2_map])
            lca_maps[node] = [map_node]
            
            is_dup = 0
            if map_node is c1_map or map_node is c2_map:
                is_dup = 1
                score += 1
                if retmap: node_dups_obj[node] += 1
                
            cur_depth = getattr(map_node, "fast_depth", 0)
            
            if node.is_root():
                score += cur_depth
                if retmap: node_losses_obj[node] += cur_depth
            
            for child in [children[0], children[1]]:
                child_map = lca_maps[child][0]
                child_depth = getattr(child_map, "fast_depth", 0)
                loss = (child_depth - cur_depth - 1) + is_dup
                if loss > 0:
                    score += loss
                    if retmap: node_losses_obj[child] += loss

        if retmap:
            final_maps_str = {k.name: [v[0].name] for k, v in lca_maps.items()}
            final_dups_str = {k.name: v for k, v in node_dups_obj.items()}
            final_losses_str = {k.name: v for k, v in node_losses_obj.items()}
            
            final_maps_obj = Map(sum(final_dups_str.values()), sum(final_losses_str.values()), cor=final_maps_str, dups=final_dups_str, losses=final_losses_str)
            return ReconResult(score, [final_maps_obj])
    
        return score

    @staticmethod
    def reconcile_one_static(mul_data: MulTree, gene_tree: SmrtTree, 
                    groups: GroupData, retmap=False) -> Union[int, ReconResult]:
        
        perm_groups = groups.ambiguous_groups
        fixed_groups = groups.fixed_groups

        # 1. Target Cache
        mul_node_lookup = {n.name: n for n in mul_data.mt.ete_tree.traverse()}
        
        # 2. Pre-calculate Target Options
        leaf_targets = {}
        leaf_nodes = list(gene_tree.ete_tree.iter_leaves())
        
        for n in leaf_nodes:
            # FIX: Ensure 1_a -> a
            raw_name = getattr(n, "clean_name", n.name)
            sp_name = raw_name.split("_")[-1]
            
            normal_target = mul_node_lookup.get(sp_name)
            star_target = mul_node_lookup.get(sp_name + "*")
            
            if not normal_target:
                raise ValueError(f"Leaf '{n.name}' maps to species '{sp_name}', but '{sp_name}' is not in the MUL-tree.")
                
            if not star_target: star_target = normal_target
            leaf_targets[n] = (normal_target, star_target)

        # 3. Map Leaf -> Group Index
        leaf_instructions = {}
        for idx, grp in enumerate(perm_groups):
            for n_name in grp:
                node = gene_tree.get_node(n_name)
                if node: leaf_instructions[node] = (0, idx)
                
        for grp, suffix in fixed_groups:
            target_idx = 1 if suffix == "*" else 0
            for n_name in grp:
                node = gene_tree.get_node(n_name)
                if node: leaf_instructions[node] = (1, target_idx)

        # 4. Hoist Traversal
        gt_postorder = list(gene_tree.ete_tree.traverse("postorder"))

        best_score = 999999
        all_maps = []
        
        # 5. Permutation Loop
        for combo in itertools.product([0, 1], repeat=len(perm_groups)):
            current_mapping = {}
            
            for node in leaf_nodes:
                targets = leaf_targets[node] 
                target_idx = 0 
                
                if node in leaf_instructions:
                    type_code, val = leaf_instructions[node]
                    target_idx = combo[val] if type_code == 0 else val
                
                current_mapping[node] = [targets[target_idx]]

            for node in gt_postorder:
                if not node.is_leaf(): current_mapping[node] = []
            
            if retmap:
                res = Reconciler.recon_lca_optimized(gene_tree, mul_data.mt, current_mapping, True, gt_postorder)
                if res.score < best_score:
                    best_score = res.score
                    all_maps = res.maps
                elif res.score == best_score:
                    all_maps.extend(res.maps)
            else:
                score = Reconciler.recon_lca_optimized(gene_tree, mul_data.mt, current_mapping, False, gt_postorder)
                if score < best_score:
                    best_score = score
        
        if retmap:
            return ReconResult(best_score, all_maps)
        
        return best_score

    @staticmethod
    def recon_lca_legacy_static(gene_tree: SmrtTree, map_target_tree: SmrtTree, 
                         initial_maps: Dict[str, List[str]], retmap=False) -> Union[int, ReconResult]:
        obj_maps = {}
        for n in gene_tree.ete_tree.traverse():
             val_list = initial_maps.get(n.name, []) 
             if not val_list and n in initial_maps: val_list = initial_maps[n] 
             
             if val_list:
                 if isinstance(val_list[0], str):
                      obj_maps[n] = [map_target_tree.get_node(val_list[0])]
                 else:
                      obj_maps[n] = val_list
             else:
                 obj_maps[n] = []
                 
        return Reconciler.recon_lca_optimized(gene_tree, map_target_tree, obj_maps, retmap)
        
    def recon_all(self, mul_trees: Dict[int, MulTree], gene_trees: Dict[int, SmrtTree], 
                  pickle_dir: str, run_prefix: str, n_proc: int, logger: Any) -> List[Tuple[int, int]]:
        step = "Reconciliation"
        logger.report_step(step, "In progress...")
        all_scores = {}
        tasks = mul_trees.items()
        worker_func = partial(_worker_reconcile_single, 
                              gene_trees=gene_trees, 
                              pickle_dir=str(pickle_dir), 
                              run_prefix=run_prefix)
        if n_proc > 1:
            with mp.Pool(processes=n_proc) as pool:
                for idx, score in pool.imap_unordered(worker_func, tasks):
                    all_scores[idx] = score
        else:
            for item in tasks:
                idx, score = worker_func(item)
                all_scores[idx] = score
        logger.report_step(step, "Success")
        return sorted(all_scores.items(), key=lambda x: x[1])

    def recon_lca_legacy(self, *args, **kwargs):
        return Reconciler.recon_lca_legacy_static(*args, **kwargs)
        
    def reconcile_one(self, *args, **kwargs):
        return Reconciler.reconcile_one_static(*args, **kwargs)
    
    def get_lowest_maps(self, sorted_scores: List[Tuple[int, int]], n_lowest: int, 
                        mul_trees: Dict[int, MulTree], gene_trees: Dict[int, SmrtTree], 
                        pickle_dir: str, run_prefix: str, logger: Any) -> Dict[int, Dict[int, ReconResult]]:
        
        step = "Getting maps for lowest scoring MTs"
        logger.report_step(step, "In progress...")
        
        # Change: Store in Dict instead of List of Tuples
        detailed_res = {} 
        limit = min(len(sorted_scores), n_lowest)
        
        for idx, total in sorted_scores[:limit]:
            mul_data = mul_trees[idx]
            mul_data.mt.build_lca_cache() 
            
            pickle_path = Path(pickle_dir) / f"{run_prefix}_{idx}_groups.pickle"
            cur_groups_dict = {}
            if idx != 0 and pickle_path.exists():
                with open(pickle_path, 'rb') as f:
                    cur_groups_dict = pickle.load(f)
            
            gt_results = {}
            st_lookup = {n.name: n for n in mul_data.mt.ete_tree.traverse()} if idx == 0 else {}
            
            for g_num, gt_obj in gene_trees.items():
                if idx == 0:
                    init_maps = {}
                    gt_nodes = list(gt_obj.ete_tree.traverse("postorder"))
                    for n in gt_nodes:
                        if n.is_leaf():
                            # FIX: Ensure 1_a -> a here too for consistency
                            raw_name = getattr(n, "clean_name", n.name)
                            sp_name = raw_name.split("_")[-1]
                            if sp_name in st_lookup: init_maps[n] = [st_lookup[sp_name]]
                        else:
                            init_maps[n] = []
                    res = Reconciler.recon_lca_optimized(gt_obj, mul_data.mt, init_maps, True, gt_nodes)
                else:
                    groups = cur_groups_dict[g_num]
                    res = self.reconcile_one(mul_data, gt_obj, groups, True)
                gt_results[g_num] = res
            
            detailed_res[idx] = gt_results
            del cur_groups_dict
            
        logger.report_step(step, "Success")
        return detailed_res
        
    def run(self, mul_trees: dict, gene_trees: dict, cfg: GrandmaConfig, logger: Any, writer: Any) -> StepResult:
        n_lowest, pickle_dir, run_prefix, n_proc = cfg.n_lowest, cfg.pickle_dir, cfg.run_prefix, cfg.num_processes
        sorted_scores = self.recon_all(mul_trees, gene_trees, pickle_dir, run_prefix, n_proc, logger)
        detailed_res = self.get_lowest_maps(sorted_scores, n_lowest, mul_trees, gene_trees, pickle_dir, run_prefix, logger)
        writer.write_results(sorted_scores, detailed_res, mul_trees, gene_trees)

        # Get the first k,v pair from detailed_res
        detailed_res_limited = {}
        for mul_idx in detailed_res:
            # Instead of keeping ReconResult, keep Maps[0] (Dict[int, Dict[int, Map]] vs Dict[int, Dict[int, ReconResult]] in StepResult)
            maps_dict = {g_idx: res.maps[0] for g_idx, res in detailed_res[mul_idx].items()}
            detailed_res_limited[mul_idx] = maps_dict
            if len(detailed_res_limited) >= 1:
                break

        return StepResult(
            sorted_scores=sorted_scores,
            mul_trees=mul_trees,
            kept_mul_maps=detailed_res_limited, # this dict is sorted
            gene_trees=gene_trees
        )