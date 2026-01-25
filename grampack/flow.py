import os
import re
import json
import random
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any
from collections import defaultdict
from pathlib import Path
from functools import partial

from .config import GlobalContext
from .models import Tree, TreeNode, SmrtTree, TaskResult, MulTree
from .ops import CommonOps

class HCounterState:
    """Tracks hybridization events to detect nested patterns."""
    def __init__(self, history=None):
        self.base_to_full = defaultdict(set)
        self.sets_by_key = []
        self.h_counter = {}
        if history:
            self._integrate_history(history)
            self._rebuild_h_counter()

    def _integrate_history(self, history):
        for data in history.values():
            combined = data["h1.node"] + data["h2.node"]
            base_names = set()
            for name in combined:
                base = HCounterState.get_base_name(name)
                self.base_to_full[base].add(name)
                base_names.add(base)
            self.sets_by_key.append(base_names)

    def _rebuild_h_counter(self):
        self.h_counter = {}
        for name_set in HCounterState.reduce_group_sets(self.sets_by_key):
            full_names = {n for name in name_set for n in self.base_to_full[name]}
            self.h_counter[frozenset(name_set)] = HCounterState.group_by_suffix(HCounterState.clean_nested(full_names))

    def update(self, new_history_entry):
        # new_history_entry is a single dict {run_id: data}
        self._integrate_history(new_history_entry)
        self._rebuild_h_counter()

    @staticmethod
    def get_base_name(name):
        return name.split('.', 1)[0]

    @staticmethod
    def is_nested(small, big):
        return big.startswith(small + '.')

    @staticmethod
    def clean_nested(names):
        """Keep only the deepest versions of names (no prefixes)."""
        return [n for n in names if not any(HCounterState.is_nested(n, o) for o in names if o != n)]

    #TBD: optimize
    @staticmethod
    def clean_nested_new(names: List[str]) -> List[str]:
        """
        Keep only the deepest versions of names (no prefixes).
        Optimized from O(N^2) to O(N log N).
        """
        if not names:
            return []
        
        # Sort names: A.1 will come before A.1.1
        sorted_names = sorted(names)
        keep = []
        
        for i in range(len(sorted_names)):
            current = sorted_names[i]
            is_prefix = False
            # Check if the next name in sorted order starts with current + "."
            if i + 1 < len(sorted_names):
                next_name = sorted_names[i+1]
                if next_name.startswith(current + "."):
                    is_prefix = True
            
            if not is_prefix:
                keep.append(current)
                
        return keep

    @staticmethod
    def reduce_group_sets(group_list):
        item_to_groups = defaultdict(set)
        for i, s in enumerate(group_list):
            for item in s:
                item_to_groups[item].add(i)
        group_map = defaultdict(set)
        for item, grp_indices in item_to_groups.items():
            group_map[frozenset(grp_indices)].add(item)
        sorted_groups = sorted(group_map.items(), key=lambda x: (len(x[0]), x[0]))
        return [set(group) for _, group in sorted_groups]

    @staticmethod
    def group_by_suffix(name_set):
        groups = defaultdict(list)
        for name in name_set:
            parts = name.split('.')
            suffix = '.'.join(parts[1:]) if len(parts) > 1 else ''
            groups[suffix].append(name)
        return list(groups.values())

class FlowManager:
    def __init__(self, ctx: GlobalContext, mode, logger):
        self.ctx = ctx
        self.mode = mode
        self.h_counter = HCounterState() if self.ctx.ignore_nesting or self.ctx.start_pt == 0 else HCounterState(self.ctx.history)
        self.sample = self.set_sampling_func(2)
        self.logger = logger
        
    # --- Init Methods ---

    def set_sampling_func(self, n):
        if self.ctx.debug:
            if self.ctx.seed:
                def _random_spacing(iterable, n):
                    return sorted(random.sample(range(len(iterable)), min(n, len(iterable))))
                return partial(_random_spacing, n=n)
            else:
                def _equal_spacing(iterable, n):
                    length = len(iterable)
                    if n >= length:
                        return list(range(length))
                    step = length / n
                    return [int(i * step) for i in range(n)]
                return partial(_equal_spacing, n=n)
        def _noop(iterable):
            return []
        return _noop

    # --- Methods for both iterative modes ---

    def _check_if_passed(self, i: int, j: int) -> bool:
        """Returns True if the event should be accepted."""
        cut_type, cut_val = self.ctx.cutoff
        curr_event = self.ctx.history[(i, j)]

        if self.ctx.debug:
            print(i, j, curr_event)

        # In split mode, always compare within current event
        if self.mode == 'split':
            if cut_type == 'rel': cut_val *= curr_event['input_score']
            return cut_val < curr_event['input_score'] - curr_event['nonin_score']
        
        # In full mode, compare to previous event (unless within nested fix or first iteration)
        if j > 0:
            # Called from nested fix
            return True
        else:
            if i == 0:
                if cut_type == 'rel': cut_val *= curr_event['input_score']
                return cut_val < curr_event['input_score'] - curr_event['nonin_score']
            else:
                # harder condition - may want to have a flag for it, but stops infinite loops
                prev_event = self._get_prev_event(i, j)
                if cut_type == 'rel': cut_val *= prev_event['nonin_score']
                return cut_val < prev_event['nonin_score'] - curr_event['nonin_score']

    def _get_prev_event(self, i: int, j: int) -> Optional[dict]:
        """
        Retrieves the previous event data from history.
        """
        if i == 0:
            return None
        else:
            if self.mode == 'split':
                # In the split mode, the depth i and index j represent binary recursion
                return self.ctx.history.get((i-1, j//2))
            else:
                if self.ctx.cutoff[0] == 'auto' and j == 0:
                    prev_event = self.ctx.history.get((i-1, 0))
                else:
                    if j > 0:
                        prev_event = self.ctx.history.get((i, j-1))
                    else:
                        prev_event = self.ctx.history.get((i-1, 0))
        return prev_event

    def _get_nonin_rank(self, res: TaskResult) -> int:
        """Returns the index of the best non-input MulTree."""
        # If input is best (idx 0), return idx of the second-best tree (rank 1)
        # Else, return idx of the best tree (rank 0)
        return 1 if res.mt_idx() == 0 else 0
    
    def _get_nonin_idx(self, res: TaskResult) -> int:
        """Returns the MulTree index of the best non-input MulTree."""
        nonin_rank = self._get_nonin_rank(res)
        return res.mt_idx(nonin_rank)

    def _update_history(self, i, j, res: TaskResult, hold: bool = False) -> None:
        """
        Logs the input and best, or, if input is best, input and second-best
        MulTree data for history tracking.
        Holds off writing to disk if `hold` is True, until nested fixes are done in Full mode.
        """
        best_idx = res.mt_idx()
        nonin_rank = self._get_nonin_rank(res)
        input_score = res.input_score
        nonin_idx = res.mt_idx(nonin_rank)
        nonin_score = res.mt_score(nonin_rank)
        if self.ctx.debug: self.logger.write(f"[DEBUG] Input map score: {input_score}, Best non-input map score: {nonin_score}")
        best_mt = res.mul_trees[best_idx]
        nonin_mt = res.mul_trees[nonin_idx]

        if self.ctx.debug:
            print(nonin_idx, best_idx, nonin_mt)
            self._debug_tree("Best MulTree:", best_mt.mt.ete_tree)
            self._debug_tree("Best Non-input MulTree:", nonin_mt.mt.ete_tree)

        self.ctx.history[(i, j)] = {
            'best_mt': best_mt.mt.ete_tree.write(format=8), # may be input tree
            'nonin_mt': nonin_mt.mt.ete_tree.write(format=8),
            'h1_node': nonin_mt.h1_node.get_leaf_names(),
            'h2_node': nonin_mt.h2_node.get_leaf_names(),
            'input_score': input_score,
            'nonin_score': nonin_score,
            'num_gts': len(res.gene_trees),
        }
        if not hold: # For full mode, until nested fixes are done (to not pollute history with partial events)
            with open(self.ctx.history_file, 'w') as f:
                json.dump({str(k): v for k, v in self.ctx.history.items()}, f, indent=4)

    # --- Output methods ---

    def plot(self, dir_path):
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

        # Get plot data from history
        plot_data = []
        for k, ((i, j), v) in enumerate(self.ctx.history.items()):
            score = v['score_tuple'][1]
            taxa = Tree(CommonOps._fix_semicolon(v['multree']), format=8).get_leaves()
            filled = True if j==0 else False  # If j > 0, it is a nested fix; otherwise, it is the first event
            if (i, j) == (1, 0): # Manually add initial condition
                plot_data.append({
                    'taxa': len(taxa) - len(v['h2.node']),
                    'score': v['score_tuple'][0],
                    'filled': True
                })
            # Last event
            if k == len(self.ctx.history) - 1:
                if v['other_tree'] != '':
                    taxa_len = len(Tree(CommonOps._fix_semicolon(v['other_tree']), format=8).get_leaves())
                else:
                    taxa_len = len(taxa)
                plot_data.append({
                    'taxa': taxa_len,
                    'score': score,
                    'filled': True
                })
            else:
                plot_data.append({
                    'taxa': len(taxa),
                    'score': score,
                    'filled': filled
                })

        df_plot = pd.DataFrame(plot_data)

        df_plot['call'] = range(1, len(df_plot) + 1)
        df_plot['norm_score'] = df_plot['score'] / df_plot['taxa']
        df_plot['filled'] = df_plot['filled'].astype(bool)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for ax, y, title, ylabel in zip(
            axes,
            ['score', 'taxa', 'norm_score'],
            ['MP Score per Call', 'Number of Taxa per Call', 'Normalized MP Score per Call'],
            ['MP Score', 'Taxa Count', 'MP Score / Taxa']
        ):
            # Plot full line
            ax.plot(df_plot['call'], df_plot[y], color='black', linewidth=1)

            # Overlay markers
            filled = df_plot[df_plot['filled']]
            hollow = df_plot[~df_plot['filled']]

            ax.scatter(filled['call'], filled[y], marker='o', color='black', label='Iteration')
            ax.scatter(hollow['call'], hollow[y], marker='o', facecolors='none', edgecolors='black', label='Nested fix')

            ax.set_title(title)
            ax.set_xlabel('Call')
            ax.set_ylabel(ylabel)
            ax.grid(True)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.legend()

        plt.tight_layout()
        output_file = dir_path / 'metrics_plot.png'
        plt.savefig(output_file, dpi=600)
        print(f'\nPlot saved to {output_file}')
        #plt.show()
        plt.close()

    def _debug_tree(self, title: str, ete_tree: Tree) -> None:
        if self.ctx.debug:
            print(f"\n[DEBUG] {title}")
            print(ete_tree.get_ascii(show_internal=True, attributes=['name']))

    # --- Handlers for the Full mode ---

    def _find_subset_key(self, lst, counter_dict):
        lst_set = set(lst)
        for k, v in counter_dict.items():
            for l in v:
                if lst_set.issubset(set(l)): return k
        return None

    @staticmethod
    def _get_base_name(name):
        return name.split('.', 1)[0]
        
    def check_and_fix_nested(self, multree: MulTree, genetrees: List[SmrtTree], iter: int, engine_callback: callable) -> None:
        """Original nested fix logic maintained, but uses unified preparer."""

        if self.ctx.ignore_nesting:
            return
        
        curr_mt, curr_gts = multree, genetrees
        event_id = 0

        while True:
            h1_node, h2_node = curr_mt.h1_node, curr_mt.h2_node
            h1_leaves, h2_leaves = h1_node.get_leaf_names(), h2_node.get_leaf_names()

            self.h_counter.update({
                len(self.h_counter.h_counter): {
                    "h1.node": h1_leaves,
                    "h2.node": h2_leaves,
                }
            })

            if self.ctx.debug:
                self.logger.write(f'\nH1: {h1_node.name} {h1_leaves} | H2: {h2_node.name} {h2_leaves}')
                self.logger.write(f'Current h_counter: {self.h_counter.h_counter}')

            h1_sis = h1_node.get_sisters()[0]
            h2_sis = h2_node.get_sisters()[0]
            
            found_nested_event = False
            for h_sis_node, h_main_node in [(h1_sis, h2_node), (h2_sis, h1_node)]:
                if not h_sis_node: continue
                
                sis_leaves = h_sis_node.get_leaf_names()
                h_group_key = self._find_subset_key(sis_leaves, self.h_counter.h_counter)

                if self.ctx.debug:
                    self.logger.write(f'\nChecking sister node: {h_sis_node.name} with leaves {sis_leaves}')
                    self.logger.write(f'H Group Key found: {h_group_key}')

                if h_group_key is None:
                    continue

                if any(self._get_base_name(leaf) in h_group_key for leaf in h1_leaves):
                    continue

                for targets in self.h_counter.h_counter[h_group_key]:
                    filter_base = {n.split('.', 1)[0] for n in sis_leaves}
                    filtered_targets = [t for t in targets if t.split('.', 1)[0] in filter_base]

                    if not filtered_targets or set(filtered_targets) == set(sis_leaves):
                        continue

                    multree = curr_mt.mt.ete_tree

                    target_lvs = [multree.search_nodes(name=t)[0] for t in filtered_targets]
                    target_node = multree.get_common_ancestor(target_lvs) if len(target_lvs) > 1 else target_lvs[0]

                    if any(ft in h1_leaves or ft in h2_leaves for ft in filtered_targets):
                        continue

                    if self.ctx.debug:
                        self.logger.write(f"\nOriginal targets: {targets}")
                        self.logger.write(f"Filtered targets: {filtered_targets}")
                        self.logger.write(f"Target leaves: {target_lvs}")

                    # Find the sister of the target nodes in the multree
                    target_sis = target_node.get_sisters()[0]
                    target_sis_lvs = target_sis.get_leaf_names()
                    # Check if base names match: target_sis_leaves and H1_leaves
                    base_tsl = {t.split('.', 1)[0] for t in target_sis_lvs}
                    base_h1l = {h.split('.', 1)[0] for h in h1_leaves}
                    if base_h1l == base_tsl:
                        continue

                    self.logger.write(f"\n### Nested Hybridization Detected ###\n")

                    event_id += 1
                    found_nested_event = True
                    fix_dir = self.ctx.root_dir / f'{iter}.{event_id}' / 'output'

                    res = engine_callback(curr_mt.tree, curr_gts, 
                                        ",".join(h_main_node.get_leaf_names()), 
                                        ",".join(filtered_targets),
                                        fix_dir
                                        )

                    curr_mt, curr_gts = self.handle_iteration_result(
                        iter, res, fix_dir, engine_callback, self.logger, j=event_id
                    )

                    # Update hybrid counter for the new nested event
                    self.h_counter.update({
                        len(self.h_counter.h_counter): {
                            "h1.node": curr_mt.h1_node.get_leaf_names(),
                            "h2.node": curr_mt.h2_node.get_leaf_names(),
                        }
                    })

                    break  # go back to outer loop with new mt/gts, save event

                if found_nested_event: break
            if not found_nested_event: break
        self.logger.write(f"# Iteration {iter} produced {event_id} event(s) after nested checks.", level=1)
    
    def _rename_best_mt(self, res: TaskResult, best_mt_idx: int) -> Tuple[SmrtTree, List[str], Dict[str, str]]:

        best_mt = res.mul_trees[best_mt_idx]

        h1_node = best_mt.h1_node
        h2_node = best_mt.h2_node
        h1_leaves = h1_node.get_leaves()
        h2_leaves = h2_node.get_leaves()
        disallowed_lvs = [l.name for l in h1_leaves]

        self._debug_tree(f"Renaming Context: {h1_node.name} (H1) | {h2_node.name} (H2)", best_mt.mt.ete_tree)
        
        distance_map = {}
        for n in best_mt.mt.ete_tree.traverse():
            d1 = n.get_distance(h1_node, topology_only=True)
            d2 = n.get_distance(h2_node, topology_only=True)
            distance_map[n.name] = ".1" if d1 < d2 else ".2"

        # Rename MT nodes for next step logic
        for l in h1_leaves: l.name += ".1"
        for l in h2_leaves: l.name = l.name.replace("*", "") + ".2"

        self._debug_tree("Renamed MT for next iteration:", best_mt.mt.ete_tree)
        
        return best_mt.mt, disallowed_lvs, distance_map

    def _partition_gt_leaves(self, res: TaskResult, best_mt_idx: int, disallowed_lvs: List[str], distance_map: Dict[str, str]) -> List[Tree]:

        new_gts_list = []
        gts = res.gene_trees
        min_maps = res.kept_mul_maps[best_mt_idx]
        
        debug_sample = self.sample(min_maps.keys())

        for g_idx, map_obj in min_maps.items():
            gt_ete = gts[g_idx].ete_tree.copy() # Operations on copies

            if g_idx in debug_sample:
                print(f"ReconResult Map: {map_obj}")
                self._debug_tree(f"Original GT {g_idx}:", gt_ete)

            for l in gt_ete.iter_leaves():
                if l.name.split('_')[1] in disallowed_lvs:
                    # For now, default for the first map when there are multiple options
                    l.name += distance_map[map_obj.cor[l.name][0]]
            new_gts_list.append(gt_ete)

            if g_idx in debug_sample:
                self._debug_tree(f"Renamed GT {g_idx}:", gt_ete)
        
        return new_gts_list

    def handle_iteration_result(self, i: int, res: TaskResult, iter_out: Path, engine_callback, iter_logger, j=0) -> Optional[Tuple[SmrtTree, Dict[int, SmrtTree]]]:
        """
        Handles the end of a 'Full' mode iteration.
        Returns: (next_st, next_gts) or None if stopping.
        """
        # Set logger for this iteration - not applicable for the split mode
        self.logger = iter_logger

        # Rename best non-input MT
        nonin_idx = self._get_nonin_idx(res)
        next_mt, disallowed_lvs, distance_map = self._rename_best_mt(res, nonin_idx)

        # Update history
        self._update_history(i, j, res)

        # Check if passed cutoff
        passed = self._check_if_passed(i, j)
        if not passed:
            self.logger.write(f"# Iteration {i} did not pass cutoff.", level=1)
            return next_mt, None
        
        # Rename Trees for Next Iteration
        new_gts_list = self._partition_gt_leaves(res, nonin_idx, disallowed_lvs, distance_map)

        if j == 0:
            # Check for Nested Hybridization
            # This encapsulates the while-loop for recursive sub-fixes
            self.check_and_fix_nested(
                multree         = res.mul_trees[nonin_idx],
                genetrees       = new_gts_list,
                iter            = i,
                engine_callback = lambda st, gts, h1, h2, out: engine_callback(st, gts, h1, h2, out),
            )

        # Convert list back to Dict for GrandmaTree consumption
        next_st = SmrtTree(tree_obj=next_mt.ete_tree)
        next_gts = {idx: SmrtTree(tree_obj=gt) for idx, gt in enumerate(new_gts_list)}
        
        # Write handoff files for resume support
        CommonOps.write_handoff_files(iter_out.parent, next_mt.ete_tree, new_gts_list)
        
        return next_st, next_gts

    # --- Handlers for the Split mode ---
    
    def extract_subproblems(self, res: TaskResult, depth: int, idx: int) -> None:
        """
        Refined binary recursion split using ETE3-safe surgery and O(N) GT extraction.
        1. Inner: Extracts independent 'pure' subtrees for each hybrid lineage.
        2. Outer: Backbone with H1 clade collapsed to a placeholder leaf.
        """
        best_mt_idx = res.mt_idx()
        best_mt = res.mul_trees[best_mt_idx]

        h1_node = best_mt.h1_node
        h2_node = best_mt.h2_node
        mt_wrapper = best_mt.mt
        min_maps = res.kept_mul_maps[best_mt_idx]
        gts = res.gene_trees

        self._debug_tree(f"Split Context: {h1_node.name} (H1) | {h2_node.name} (H2)", mt_wrapper.ete_tree)

        h_clade_names = [l.name for l in h1_node.get_leaves()]
        h_clade_names.extend([l.name for l in h2_node.get_leaves()])

        outer_gts = {}
        inner_gts = {}
        innie_counter = 0

        debug_sample = self.sample(gts.keys())

        for g_idx, gt_wrapper in gts.items():
            maps = min_maps.get(g_idx)
            if not maps: continue
            source_gt = gt_wrapper.ete_tree

            if g_idx in debug_sample:
                self._debug_tree(f"Pre-split GT {g_idx}:",source_gt)

            rev_map = maps.rev
            inner_leaves = set()
            outer_leaves = set()
            for leaf, v in rev_map.items():
                if leaf in h_clade_names:
                    inner_leaves.update(v)
                else:
                    outer_leaves.update(v)
            source_lvs = {l.name for l in source_gt.get_leaves()}
            outer_leaves = outer_leaves.intersection(source_lvs)
            inner_leaves = inner_leaves.intersection(source_lvs)

            # --- Outer Sub-problem (Backbone) ---
            gt_ete = source_gt.copy()

            if len(outer_leaves) >= self.ctx.min_gt_lvs:
                gt_ete.prune(list(outer_leaves), preserve_branch_length=True)
                outer_gts[g_idx] = SmrtTree(tree_obj=gt_ete)

                if g_idx in debug_sample:
                    self._debug_tree(f"Pruned Outer GT {g_idx}:", gt_ete)

            # --- Inner Sub-problem (Hybrid Clade) ---
            gt_ete = source_gt.copy()

            # Pass 1: Bottom-up purity caching (Postorder)
            node_is_pure = {}
            for node in gt_ete.traverse("postorder"):
                if node.is_leaf():
                    node_is_pure[node] = node.name in inner_leaves
                else:
                    # Internal node is pure only if ALL its children are pure
                    node_is_pure[node] = all(node_is_pure[child] for child in node.children)

            # Pass 2: Top-down extraction (Optimized)
            final_pure_lineages = []
            # We use a manual stack to allow 'skipping' subtrees
            stack = [gt_ete]
            while stack:
                node = stack.pop()
                if node_is_pure.get(node, False):
                    if len(node) >= self.ctx.min_gt_lvs:
                        final_pure_lineages.append(node)
                    # SUCCESS: We found the largest clade for this branch. 
                    # Do NOT add children to the stack; this skips the entire subtree.
                else:
                    # Node isn't pure, so we must check its children
                    stack.extend(node.children)

            for ph_node in final_pure_lineages:
                extracted_gt = ph_node.copy()
                inner_gts[innie_counter] = SmrtTree(tree_obj=extracted_gt)
                
                if g_idx in debug_sample:
                    self._debug_tree(f"Extracted Inner as Lineage {innie_counter}:", extracted_gt)
                
                innie_counter += 1

        # --- Species Tree Surgery ---

        # Simply copy the subtree rooted at H1 for the Inner ST
        inner_st_obj = h1_node.copy()

        # For the Outer ST, we need to remove both H1 and H2
        outer_st_obj = mt_wrapper.ete_tree.copy()
        # Re-locate nodes in the COPY
        h1_in_outer = outer_st_obj.search_nodes(name=h1_node.name)[0]
        h2_in_outer = outer_st_obj.search_nodes(name=h2_node.name)[0]
        # Safely remove H nodes from Outer ST: detach leaf, then delete the resulting knuckle node
        for n in [h1_in_outer, h2_in_outer]:
            n_up = n.up # Save parent before detaching
            n.detach()
            if n_up:
                n_up.delete() # delete() removes the internal node and connects children to parent

        if self.ctx.debug:
            self._debug_tree("Inner Species Tree (Hybrid Clade):", inner_st_obj)
            self._debug_tree("Outer Species Tree (Backbone):", outer_st_obj)
            print(f'len(inner_gts)={len(inner_gts)}, len(outer_gts)={len(outer_gts)}')

        # --- 4. Queue Tasks with Binary IDs ---
        # Only queue tasks if species tree has enough leaves to be valid
        next_tasks = []
        if len(inner_st_obj.get_leaves()) >= self.ctx.min_st_lvs and len(inner_gts) > 0:
            next_tasks.append((SmrtTree(tree_obj=inner_st_obj), inner_gts, f"{depth + 1}.{idx * 2}"))
        if len(outer_st_obj.get_leaves()) >= self.ctx.min_st_lvs and len(outer_gts) > 0:
            next_tasks.append((SmrtTree(tree_obj=outer_st_obj), outer_gts, f"{depth + 1}.{idx * 2 + 1}"))
        return next_tasks

    def handle_split_result(self, bin_id, res: TaskResult, iter_out: Path, iter_logger) -> List[Tuple[SmrtTree, Dict[int, SmrtTree], str]]:
        """
        Processes a split worker result.
        Returns: List of new sub-tasks or empty list.
        """
        backup_logger = self.logger
        self.logger = iter_logger

        # 1. Determine Depth and Index from Binary ID
        depth, idx = (int(x) for x in bin_id.split('.')) if '.' in bin_id else (0, 0)

        self._update_history(depth, idx, res)

        # 2. Validation & Cutoff
        if not self._check_if_passed(depth, idx):
        #if not self._to_proceed(res, f"Depth {depth}, Index {idx}"):
            return []

        self.logger.write(f"# Reticulation found at Depth {depth}, Index {idx} with score {res.mt_score()}.", level=1)
        
        # 3. Extract Subproblems
        #try:
        next_tasks = self.extract_subproblems(res, depth, idx)
        #except Exception as e:
        #    logger.write(f"Error extracting subproblems at Depth {depth}, Index {idx}: {e}", level=1)
            #print(res)
        #    return []
        
        # 4. Update History
        #self.update_history(depth, {idx: res})

        # Write handoff files for resume support
        for task_st, task_gts, task_id in next_tasks:
            task_out = iter_out.parent / task_id
            task_out.mkdir(parents=True, exist_ok=True)
            print(f"# Written handoff files for task {task_id} at {task_out.relative_to(iter_out.parent.parent.parent)}, with {len(task_gts)} GTs.")
            CommonOps.write_handoff_files(task_out, task_st.ete_tree, [gt.ete_tree for gt in task_gts.values()])

        self.logger = backup_logger
        return next_tasks

    # --- Post-processing for Split mode ---

    def glue_split_results(self, output_dir: Path, original_st_path: Path, logger) -> None:
        """
        Recombines recursive sub-analyses into a single global hybridization record.
        Iterates reverse (deepest first), updating leaf names with .1/.2 suffixes.
        Produces both a suffix-separated single-label tree and a clean MUL-tree.
        """
        self.logger = logger
        step = "Recombining Split Results"
        self.logger.report_step(step, "In progress...", start=True)

        if not self.ctx.history:
            self.logger.write("No reticulations found. Nothing to glue.", level=1)
            return

        # 1. Load Base Species Tree
        try:
            with open(original_st_path, 'r') as f:
                st_text = f.read().strip()
                if not st_text.endswith(';'): st_text += ';'
                # Use GrandmaTree wrapper for robust newick parsing
                base_st = SmrtTree(newick=st_text)
        except Exception as e:
            self.logger.write(f"Error loading original ST for gluing: {e}", level=1)
            return

        # 2. Sort Events: Reverse order (Deepest depth/index first)
        # self.history keys are (depth, idx) tuples.
        sorted_keys = sorted(self.ctx.history.keys(), key=lambda x: (x[0], x[1]), reverse=True)
        
        # Working on a copy of the ete3 tree
        current_tree = base_st.ete_tree.copy()
        
        # Helper: Find node in current_tree matching a list of base names
        # Handles cases where leaves have accumulated suffixes (e.g., matching 'x' to 'x.1')
        def find_clade_root(tree, base_names):
            base_set = set(base_names)
            matches = []
            for l in tree.get_leaves():
                # Strip suffixes to find base name: "x.1.2" -> "x"
                # Assumes species names do not contain dots
                # Strip only suffixes like .1, .2, .1.2
                lname_base = re.sub(r'(\.[0-9]+)+$', '', l.name)
                if lname_base in base_set:
                    matches.append(l)
            
            if not matches: return None
            if len(matches) == 1: return matches[0]
            try:
                return tree.get_common_ancestor(matches)
            except ValueError:
                return None # Should not happen if matches exist

        # 3. Apply Events
        for key in sorted_keys:
            event = self.ctx.history[key]
            
            # A. Extract Topology Info from Event
            h1_names = event['h1_node'] # Base names defining the lineage
            
            # We need to find the Sister of H2 to know WHERE to graft.
            # We parse the local event tree string to find this relationship.
            local_mt = Tree(event['best_mt'], format=1) 
            
            # Find H2 leaves in local tree (they usually have '*' or are the second occurence)
            # In split mode history, h2.node names usually come with '*' suffix from the run
            h2_leaves_local = []
            for n in local_mt.get_leaves():
                # Match against recorded h2 names
                if n.name in event['h2_node']:
                    h2_leaves_local.append(n)
            
            if not h2_leaves_local:
                self.logger.write(f"Warning: Could not find H2 leaves in event {key} tree structure.", level=1)
                continue
            
            # Get Sister of H2 in local tree
            local_h2_node = local_mt.get_common_ancestor(h2_leaves_local) if len(h2_leaves_local) > 1 else h2_leaves_local[0]
            sisters = local_h2_node.get_sisters()
            if not sisters:
                self.logger.write(f"Warning: H2 node in event {key} has no sister (Root?).", level=1)
                continue
            
            # Extract base names of the sister clade
            local_sister_leaves = [re.sub(r'(\.[0-9]+)+$', '', n.name.replace('*','')) for n in sisters[0].get_leaves()]

            # B. Locate Nodes in Global Tree
            h1_node_global = find_clade_root(current_tree, h1_names)
            target_sister_global = find_clade_root(current_tree, local_sister_leaves)

            if not h1_node_global or not target_sister_global:
                self.logger.write(f"Warning: Could not map event {key} to global tree. Skipping.", level=1)
                continue

            # C. Grafting Operation
            # 1. Clone H1 (This becomes H2)
            h2_copy = h1_node_global.copy()
            
            # 2. Create new container at target position
            target_parent = target_sister_global.up
            new_internal = TreeNode()
            
            if target_parent:
                target_sister_global.detach()
                target_parent.add_child(new_internal)
            else:
                # Target is root; new node becomes new root
                target_sister_global.detach()
                new_internal.add_child(target_sister_global)
                current_tree = new_internal # Update root reference

            # 3. Attach Sister and New H2 to new internal node
            # (Sister is already attached if we added new_internal to parent? No, we detached sister)
            if target_sister_global not in new_internal.children:
                new_internal.add_child(target_sister_global)
            new_internal.add_child(h2_copy)
            
            # D. Apply Suffixes
            # Original H1 lineage gets .1
            for l in h1_node_global.get_leaves():
                l.name += ".1"
            
            # New H2 lineage gets .2
            for l in h2_copy.get_leaves():
                l.name += ".2"
                
        # 4. Output Files
        out_st = output_dir / "merged_single_label_form.tre"
        out_mul = output_dir / "merged_multree.tre"
        
        sl_str = current_tree.write(format=9)

        # Write SL-Form (with suffixes)
        with open(out_st, 'w') as f:
            f.write(sl_str)

        # Write MUL-Form (Clean names)
        # Clone to avoid modifying the tree we just wrote
        mul_tree = current_tree.copy()
        for l in mul_tree.get_leaves():
            # Remove all suffixes starting with dot followed by digits
            l.name = re.sub(r'(\.[0-9]+)+$', '', l.name)
            
        with open(out_mul, 'w') as f:
            f.write(mul_tree.write(format=9))
            
        self.logger.report_step(step, "Success: Merged trees written.")
        
        # replace .1 with + and .2 with *, and log to logger
        self.logger.write(f"# Merged tree inferred: {sl_str.replace('.1', '+').replace('.2', '*')}")

    def fast_forward_split(self, current_tasks):
        """
        Reconstructs the task queue from disk state based on history.
        Uses BFS to traverse solved nodes and identify the frontier.
        """

        print("Fast-forwarding Split tasks based on history...")
        print(f"Current tasks: {current_tasks}, output_dir: {self.ctx.root_dir}")

        queue = [(current_tasks[0][2], None)] # Start with Root ID (e.g. "0")
        real_tasks = []
        
        while queue:
            q = queue.pop(0)
            nid = q[0]
            
            # Check if this task is already solved in history
            depth, idx = (int(x) for x in nid.split('.')) if '.' in nid else (0, int(nid))
            
            if (depth, idx) in self.ctx.history:
                # Task done. Check for its children directories on disk
                c1 = f"{depth+1}.{idx*2}"
                c2 = f"{depth+1}.{idx*2+1}"

                print(f"Task {nid} done. Checking children: {c1}, {c2} in {self.ctx.root_dir / nid}" )
                
                # If child dir exists, add to traversal queue
                if (self.ctx.root_dir / nid / c1 / "multree.tre").exists():
                    queue.append((c1, nid))
                if (self.ctx.root_dir / nid / c2 / "multree.tre").exists():
                    queue.append((c2, nid))
            else:
                # Task NOT in history -> It is a frontier task to run.
                # Load inputs from disk
                p_nid = q[1]
                print(f"Queueing frontier task: {nid} (child of {p_nid})")

                st_path = self.ctx.root_dir / p_nid / nid / "multree.tre"
                gt_path = self.ctx.root_dir / p_nid / nid / "genetrees.txt"
                
                print(f"Looking for ST at {st_path}, GTs at {gt_path}")

                if st_path.exists() and gt_path.exists():
                    real_tasks.append((st_path, gt_path, nid))
                    self.logger.write(f"Resuming sub-problem: {nid}", level=2)
                else:
                    # Fallback: if root is missing inputs but passed in current_tasks
                    if nid == str(current_tasks[0][2]):
                        print(f"Using provided inputs for root task {nid}, {current_tasks}")
                        real_tasks.extend(current_tasks)
                    else:
                        self.logger.write(f"Missing inputs for resume task {nid}. Skipping.", level=1)

        print(f"Fast-forwarded tasks: {real_tasks}")

        return real_tasks