import sys
import pickle
import itertools
import multiprocessing as mp
from functools import partial
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Set, Optional, Union
from .tree_ops import GrandmaTree, MulData
from .io import GrandmaConfig

@dataclass(slots=True, frozen=True)
class ReconResult:
    """
    Immutable container for reconciliation results.
    Replaces the list: [score, n_dups, n_losses, maps, node_dups, node_losses]
    """
    score: int
    n_dups: int
    n_losses: int
    maps: Dict[str, List[str]] = field(default_factory=dict)
    node_dups: Dict[str, int] = field(default_factory=dict)
    node_losses: Dict[str, int] = field(default_factory=dict)

@dataclass(slots=True, frozen=True)
class GroupData:
    """
    Container for the groups of a specific Gene Tree <-> Mul Tree pair.
    Mirrors the list [groups, fixed_groups] from original code.
    """
    ambiguous_groups: List[List[str]]
    fixed_groups: List[Tuple[List[str], str]]

# Type Alias for the content of the pickle file: { GeneID : GroupData }
GroupsPickle = Dict[int, GroupData]

# --- Worker Function (Must be top-level to be picklable) ---

def _worker_reconcile_single(
    mul_item: Tuple[int, MulData], 
    gene_trees: Dict[int, GrandmaTree], 
    pickle_dir: str, 
    run_prefix: str
) -> Tuple[int, int]:
    """
    Worker function to reconcile one MUL-tree against all gene trees.
    Executed in parallel processes.
    """
    mul_idx, mul_data = mul_item
    
    # 1. Load Groups Pickle
    # Note: We reconstruct the path string here to ensure pickle safety
    p_path = Path(pickle_dir) / f"{run_prefix}_{mul_idx}_groups.pickle"
    
    cur_groups_dict: GroupsPickle = {}
    if mul_idx != 0:
        if not p_path.exists():
            # In a worker, we might not want to print to stdout directly or access logger
            return mul_idx, 9999999 # Return high score on error
        
        try:
            with open(p_path, 'rb') as f:
                cur_groups_dict = pickle.load(f)
        except Exception:
             return mul_idx, 9999999

    # 2. Reconcile Gene Trees
    total_score = 0
    
    # Instantiate a temporary Reconciler helper (stateless logic) 
    # or access static methods if refactored. 
    # For now, we reuse the logic via a helper class or method.
    # To keep it simple, we use the Reconciler class methods which we made static-compatible below.

    if mul_idx == 0:
        # Special case ST: Must iterate all gene trees
        for g_num, gt_obj in gene_trees.items():
            init_maps = {}
            # OPTIMIZATION: Cache split names if possible, or do it here
            for n in gt_obj.ete_tree.iter_leaves():
                 init_maps[n.name] = [n.name.split("_")[-1]]
            for n in gt_obj.ete_tree.traverse():
                 if not n.is_leaf(): init_maps[n.name] = []
            
            score = Reconciler.recon_lca_legacy_static(gt_obj, mul_data.mt, init_maps, False)
            total_score += score
    else:
        # Standard MUL-Recon: Iterate GROUPS, not trees (faster if filtering occurred)
        for g_num, groups in cur_groups_dict.items():
            # Direct lookup is faster than iterating all and skipping
            if g_num not in gene_trees:
                continue 
            gt_obj = gene_trees[g_num]
            
            score = Reconciler.reconcile_one_static(mul_data, gt_obj, groups, False)
            total_score += score

    '''for g_num, gt_obj in gene_trees.items():
        if mul_idx == 0:
            # Special case ST
            init_maps = {}
            for n in gt_obj.ete_tree.iter_leaves():
                 init_maps[n.name] = [n.name.split("_")[-1]]
            for n in gt_obj.ete_tree.traverse():
                 if not n.is_leaf(): init_maps[n.name] = []
            score = Reconciler.recon_lca_legacy_static(gt_obj, mul_data.mt, init_maps, False)
        else:
            if g_num not in cur_groups_dict: continue
            groups = cur_groups_dict[g_num]
            score = Reconciler.reconcile_one_static(mul_data, gt_obj, groups, False)
        
        total_score += score'''
        
    return mul_idx, total_score

class Reconciler:
    def __init__(self, species_tree: GrandmaTree, config: Any):
        self.st = species_tree
        self.cfg = config

    # --------------------------------------------------------------------------
    # GROUP COLLAPSING LOGIC
    # --------------------------------------------------------------------------

    def _get_sister_clade_labels(self, node_obj) -> List[str]:
        """Helper to get tip labels of the sister node from a Node Object."""
        if not node_obj or not node_obj.up:
            return []
        
        sisters = [ch for ch in node_obj.up.children if ch != node_obj]
        labels = []
        for sis in sisters:
            labels.extend([l.name.split("_")[-1] for l in sis.iter_leaves()])
        return labels

    def _find_node_by_clade(self, tree: GrandmaTree, target_leaves: Set[str]) -> Any:
        """
        Finds the node in 'tree' whose descendant leaves exactly match 'target_leaves'.
        This avoids relying on node labels which change during MUL-tree construction.
        """
        # Find leaves matching names
        leaf_nodes = []
        for t in target_leaves:
            # Search for exact name match
            matches = tree.ete_tree.search_nodes(name=t)
            if matches:
                leaf_nodes.extend(matches)
        
        if not leaf_nodes:
            return None
            
        # Get LCA of these leaves
        lca = tree.ete_tree.get_common_ancestor(leaf_nodes)
        
        # Verify strict equality (Monophyly)
        # If LCA has extra leaves not in target, it's not the exact clade node
        lca_leaves = {l.name for l in lca.iter_leaves()}
        if lca_leaves == target_leaves:
            return lca
        return None

    # wrapper for multiprocessing compatibility
    def compute_groups(self, gene_tree: GrandmaTree, mul_data: MulData) -> GroupData:
        return self._compute_groups(gene_tree, mul_data)

    def _compute_groups(self, gene_tree: GrandmaTree, mul_data: MulData) -> GroupData:
        """
        Pure computation of groups for a single gene tree against a MUL-tree.
        Returns GroupData object.
        """
        h1_target = set(mul_data.h_clade)
        groups: Dict[str, List[str]] = {}
        singles: Dict[str, List[str]] = {}
        
        # Ensure postorder for bottom-up greedy grouping
        gt_nodes = list(gene_tree.ete_tree.traverse("postorder"))

        # --- 1. Singles & Groups Identification ---
        for node in gt_nodes:
            if node.is_leaf():
                sp_name = node.name.split("_")[-1]
                if sp_name in h1_target:
                    parent = node.up
                    if parent:
                        anc_clade = [l.name for l in parent.iter_leaves() if l.name != node.name]
                        singles[node.name] = anc_clade

        for node in gt_nodes:
            if not node.is_leaf():
                children = node.children
                if len(children) != 2: continue
                d1, d2 = children[0], children[1]
                d1_leaves = [l.name.split("_")[-1] for l in d1.iter_leaves()]
                d2_leaves = [l.name.split("_")[-1] for l in d2.iter_leaves()]
                
                d1_is_hybrid = all(sp in h1_target for sp in d1_leaves)
                d2_is_hybrid = all(sp in h1_target for sp in d2_leaves)
                
                if d1_is_hybrid and d2_is_hybrid:
                    # Disjoint check
                    if not any(sp in d1_leaves for sp in d2_leaves):
                        cur_clade = [l.name for l in node.iter_leaves()]
                        # Greedy delete descendants
                        for desc in node.iter_descendants():
                            if desc.name in groups: del groups[desc.name]
                        groups[node.name] = cur_clade 
        
        # Clean singles swallowed by groups
        for grp_node in groups:
            for leaf in groups[grp_node]:
                if leaf in singles: del singles[leaf]

        # --- 2. Fixing Logic ---
        final_ambiguous: List[List[str]] = []
        final_fixed: List[Tuple[List[str], str]] = []

        if mul_data.h1_node != "NA":
            h2_target = {f"{x}*" for x in mul_data.h_clade}
            n1_obj = self._find_node_by_clade(mul_data.mt, h1_target)
            n2_obj = self._find_node_by_clade(mul_data.mt, h2_target)
            
            h1_sisters = self._get_sister_clade_labels(n1_obj) if n1_obj else []
            h2_sisters = self._get_sister_clade_labels(n2_obj) if n2_obj else []

            # Validity Check: Disjointness
            if n2_obj:
                h2_leaves_exact = {l.name for l in n2_obj.iter_leaves()}
                if not set(h1_sisters).isdisjoint(h2_leaves_exact):
                    h1_sisters = []
            
            if n1_obj:
                h1_leaves_exact = {l.name for l in n1_obj.iter_leaves()}
                if not set(h2_sisters).isdisjoint(h1_leaves_exact):
                    h2_sisters = []

            # Clean sisters (remove *)
            h1_sisters_clean = {x.replace("*", "") for x in h1_sisters}
            h2_sisters_clean = {x.replace("*", "") for x in h2_sisters}

            def process_unit(unit_nodes: List[str]):
                if len(unit_nodes) == 1:
                    g_node = gene_tree.get_node(unit_nodes[0])
                else:
                    n_objs = [gene_tree.get_node(x) for x in unit_nodes]
                    g_node = gene_tree.ete_tree.get_common_ancestor(n_objs)
                
                if not g_node or not g_node.up:
                    final_ambiguous.append(unit_nodes)
                    return

                gt_sisters = self._get_sister_clade_labels(g_node)
                if not gt_sisters:
                    final_ambiguous.append(unit_nodes)
                    return
                    
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
    # RECONCILIATION LOGIC
    # --------------------------------------------------------------------------

    @staticmethod
    def recon_lca_legacy_static(gene_tree: GrandmaTree, map_target_tree: GrandmaTree, 
                         initial_maps: Dict[str, List[str]], retmap=False) -> Union[int, ReconResult]:
        
        score = 0
        dups = {n.name: 0 for n in gene_tree.ete_tree.traverse()}
        losses = {n.name: 0 for n in gene_tree.ete_tree.traverse()}
        nodes = list(gene_tree.ete_tree.traverse("postorder"))
        lca_maps = initial_maps.copy()
        
        for node in nodes:
            if node.is_leaf(): continue
            children = node.children
            # Fix polytomy handling
            if len(children) < 2: continue
            
            # LCA of all children maps
            child_maps = [lca_maps[child.name][0] for child in children]
            map_node = map_target_tree.get_lca(child_maps)
            lca_maps[node.name] = [map_node.name]
            
            is_dup = 0
            # If map matches any child map
            if any(map_node.name == cm for cm in child_maps):
                is_dup = 1
                dups[node.name] += 1
                score += 1
                
            cur_depth = map_target_tree.get_node_depth(map_node.name)
            
            if node.is_root():
                 losses[node.name] += cur_depth
                 score += cur_depth
            
            for i, child in enumerate(children):
                child_map = child_maps[i]
                child_depth = map_target_tree.get_node_depth(child_map)
                c_loss = (child_depth - cur_depth - 1) + is_dup
                if c_loss > 0:
                    score += c_loss
                    losses[child.name] += c_loss
                
        if retmap: 
            return ReconResult(score, sum(dups.values()), sum(losses.values()), lca_maps, dups, losses)
        return score

    @staticmethod
    def reconcile_one_static(mul_data: MulData, gene_tree: GrandmaTree, 
                      groups: GroupData, retmap=False) -> Union[int, ReconResult]:
        """
        Reconciles one gene tree to one MUL-tree using pre-computed groups.
        """
        best_score = 999999
        best_res = None

        perm_groups = groups.ambiguous_groups
        fixed_groups = groups.fixed_groups

        # OPTIMIZATION: Pre-calculate leaf species names once
        leaf_names = {n.name: n.name.split("_")[-1] for n in gene_tree.ete_tree.iter_leaves()}
        
        # OPTIMIZATION: Map Leaf -> Group Index/Suffix ONCE
        # This avoids iterating the groups list 2^N * Leaves times.
        leaf_to_perm_idx = {}
        for idx, grp in enumerate(perm_groups):
            for node in grp:
                leaf_to_perm_idx[node] = idx
        
        leaf_to_fixed_suffix = {}
        for grp, suffix in fixed_groups:
            for node in grp:
                leaf_to_fixed_suffix[node] = suffix
        
        # 3. Permutation Loop (2^N)
        for combo in itertools.product(['', '*'], repeat=len(perm_groups)):
            current_mapping = {}

            # Fast Map Construction (O(Leaves) instead of O(Leaves * Groups)) & using pre-calculated strings
            for n_name, sp_name in leaf_names.items():
                suffix = ""
                
                # O(1) Lookup
                # Check ambiguous
                if n_name in leaf_to_perm_idx:
                    suffix = combo[leaf_to_perm_idx[n_name]]
                # Check fixed
                elif n_name in leaf_to_fixed_suffix:
                    suffix = leaf_to_fixed_suffix[n_name]
                            
                current_mapping[n_name] = [sp_name + suffix]

            '''def get_map_suffix(n_name):
                # Check ambiguous groups
                for i, grp in enumerate(perm_groups):
                    if n_name in grp: return combo[i]
                # Check fixed groups
                for grp, fix_map in fixed_groups:
                    if n_name in grp: return fix_map
                return "" '''

            # Initialize Maps
            '''for node in gene_tree.ete_tree.iter_leaves():
                sp_name = node.name.split("_")[-1]
                suffix = get_map_suffix(node.name)
                current_mapping[node.name] = [sp_name + suffix]'''

            '''for node_name, sp_name in leaf_species.items():
                suffix = get_map_suffix(node_name)
                current_mapping[node_name] = [sp_name + suffix]'''
            
            # Internal nodes init
            for node in gene_tree.ete_tree.traverse():
                if not node.is_leaf(): current_mapping[node.name] = []
                    
            # Run LCA
            if retmap:
                res = Reconciler.recon_lca_legacy_static(gene_tree, mul_data.mt, current_mapping, True)
                if res.score < best_score:
                    best_score = res.score
                    best_res = res # Store the object
                # If equal, we might want to store multiple, but keeping simple for now
            else:
                score = Reconciler.recon_lca_legacy_static(gene_tree, mul_data.mt, current_mapping, False)
                if score < best_score:
                    best_score = score
        
        return best_res if retmap else best_score

    # --------------------------------------------------------------------------
    # DRIVERS WITH PICKLING
    # --------------------------------------------------------------------------

    def recon_all(self, mul_trees: Dict[int, MulData], gene_trees: Dict[int, GrandmaTree], 
                  pickle_dir: str, run_prefix: str, n_proc: int, logger: Any) -> List[Tuple[int, int]]:
        """
        Main driver. Iterates MUL-trees, loads their group pickles, reconciles all gene trees.
        Uses multiprocessing for parallel execution.
        """
        step = "Reconciliation"
        logger.report_step(step, "In progress...")
        
        all_scores = {}
        
        # Prepare arguments for the worker
        # gene_trees object is passed to all workers. 
        # Since it is read-only, multiprocessing should handle it efficiently on Linux (fork).
        # On Windows, it will be pickled, so ensure GrandmaTree is picklable.
        
        tasks = mul_trees.items()
        
        worker_func = partial(_worker_reconcile_single, 
                              gene_trees=gene_trees, 
                              pickle_dir=str(pickle_dir), 
                              run_prefix=run_prefix)
        
        if n_proc > 1:
            with mp.Pool(processes=n_proc) as pool:
                # imap_unordered lets us track progress if needed, but we just want results
                for idx, score in pool.imap_unordered(worker_func, tasks):
                    all_scores[idx] = score
        else:
            # Serial Fallback
            for item in tasks:
                idx, score = worker_func(item)
                all_scores[idx] = score

        logger.report_step(step, "Success")
        return sorted(all_scores.items(), key=lambda x: x[1])

    # Method to keep class compatibility with existing code calling instance methods
    def recon_lca_legacy(self, *args, **kwargs):
        return Reconciler.recon_lca_legacy_static(*args, **kwargs)
        
    def reconcile_one(self, *args, **kwargs):
        return Reconciler.reconcile_one_static(*args, **kwargs)
    
    def get_lowest_maps(self, sorted_scores: List[Tuple[int, int]], n_lowest: int, 
                        mul_trees: Dict[int, MulData], gene_trees: Dict[int, GrandmaTree], 
                        pickle_dir: str, run_prefix: str, logger: Any) -> List[Tuple[int, Dict[int, ReconResult]]]:
        
        step = "Getting maps for lowest scoring MTs"
        logger.report_step(step, "In progress...")
        
        detailed_res = []
        limit = min(len(sorted_scores), n_lowest)
        
        for idx, total in sorted_scores[:limit]:
            mul_data = mul_trees[idx]
            
            # Load Pickle
            pickle_path = Path(pickle_dir) / f"{run_prefix}_{idx}_groups.pickle"
            cur_groups_dict = {}
            if idx != 0 and pickle_path.exists():
                with open(pickle_path, 'rb') as f:
                    cur_groups_dict = pickle.load(f)
            
            # Reconcile All Gene Trees with Maps
            gt_results = {}
            for g_num, gt_obj in gene_trees.items():
                if idx == 0:
                    init_maps = {}
                    for n in gt_obj.ete_tree.iter_leaves():
                         init_maps[n.name] = [n.name.split("_")[-1]]
                    for n in gt_obj.ete_tree.traverse():
                         if not n.is_leaf(): init_maps[n.name] = []
                    res = self.recon_lca_legacy(gt_obj, mul_data.mt, init_maps, True)
                else:
                    groups = cur_groups_dict[g_num]
                    res = self.reconcile_one(mul_data, gt_obj, groups, True)
                
                gt_results[g_num] = res
            
            detailed_res.append((idx, gt_results))
            del cur_groups_dict
            
        logger.report_step(step, "Success")
        return detailed_res
        
    def run(self, mul_trees: dict, gene_trees: dict, cfg: GrandmaConfig, logger: Any, writer: Any) -> Tuple[list, list]:
        n_lowest, pickle_dir, run_prefix, n_proc = cfg.n_lowest, cfg.pickle_dir, cfg.run_prefix, cfg.num_processes
        sorted_scores = self.recon_all(mul_trees, gene_trees, pickle_dir, run_prefix, n_proc, logger)
        detailed_res = self.get_lowest_maps(sorted_scores, n_lowest, mul_trees, gene_trees, pickle_dir, run_prefix, logger)
        writer.write_results(sorted_scores, detailed_res, mul_trees, gene_trees)
        return sorted_scores, detailed_res