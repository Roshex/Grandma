import os
import re
import json
import random
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path

from .models import Tree, TreeNode, SmrtTree, StepResult
from .ops import TreeLoader

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
    def __init__(self, iter_num, cutoff_cfg, ignore_nesting, history, history_file, output_dir, seed=42):
        self.curr_i = iter_num
        self.cutoff_cfg = cutoff_cfg
        self.ignore_nesting = ignore_nesting
        self.history = history
        self.history_file = history_file
        self.output_dir = output_dir
        self.seed = seed
        self.h_counter = HCounterState() if ignore_nesting or iter_num == 0 else HCounterState(history)

    ### New from-objs parsing ###

    def update_cutoff_val(self, self_score):
        """Updates the dynamic baseline for 'auto' cutoff mode."""
        self.prev_self_score = self_score

    def check_cutoff(self, self_score, best_score):
        """Returns True if the event should be accepted."""
        # Baseline: if it's not improving over self-mapping, it's usually the end
        if best_score == self_score:
            return False
        
        cuttoff_mode, cutoff_val = self.cutoff_cfg
        
        print(self.cutoff_cfg)
        if cuttoff_mode == 'auto':
            # In 'auto', we compare against 0 improvement or the previous iteration's score
            return best_score < (cutoff_val if cutoff_val is not None else self_score)
        elif cuttoff_mode == 'abs':
            return (self_score - best_score) > cutoff_val
        elif cuttoff_mode == 'rel':
            return ((self_score - best_score) / self_score) > cutoff_val
        return True

    def _to_proceed(self, res: StepResult, i_str: str, logger, debug) -> bool:
        """
        Unified logic to validate scores, apply cutoffs, and rename trees.
        """
        min_idx = res.mt_idx()
        if min_idx == 0:
            logger.write(f"# Cutoff reached: no events found at {i_str}.", level=1)
            return False

        self_score = res.self_score
        best_score = res.mt_score()
        
        if debug: print(f"[DEBUG] Self-mapping score: {self_score}, Best non-self score: {best_score}")
        
        if not self.check_cutoff(self_score, best_score): 
            logger.write(f"# Cutoff reached: no events found at {i_str}.", level=1)
            return False

        return True

    ### Output methods ###

    def _fix_semicolon(self, tree_str: str) -> str:
        """Ensures tree strings end with a semicolon."""
        return tree_str if tree_str.endswith(';') else tree_str + ';'

    def update_history(self, i, new_events):
        for j, v in new_events.items():
            # v is StepResult
            gts = v.gene_trees
            mt = v.mul_trees[v.mt_idx()]
            score = v.mt_score()
            score_tuple = (v.self_score, score)
            other_tree = ''
            self.history[(i, j)] = {
                'multree': mt.mt.ete_tree.write(format=8),
                'gt_file': len(gts) if gts is not None else 'NA',
                'score': score,
                'h1.node': mt.h1_node.get_leaf_names() if isinstance(mt.h1_node, Tree) else mt.h1_node,
                'h2.node': mt.h2_node.get_leaf_names() if isinstance(mt.h2_node, Tree) else mt.h2_node,
                'score_tuple': score_tuple,
                'other_tree': other_tree # already a string
            }
        with open(self.history_file, 'w') as f:
            json.dump({str(k): v for k, v in self.history.items()}, f, indent=4)

    @staticmethod
    def write_handoff_files(st, gts, folder):
        """Writes the trees to disk to allow inspection/resume, matching iter_mode.py."""
        st_path = folder / 'multree.tre'
        gt_path = folder / 'genetrees.txt'
        with open(st_path, 'w') as f: f.write(st.write(format=8))
        with open(gt_path, 'w') as f:
            for gt in gts: f.write(gt.write(format=8) + '\n')

    def plot(self, dir_path):
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

        # Get plot data from history
        plot_data = []
        for k, ((i, j), v) in enumerate(self.history.items()):
            score = v['score_tuple'][1]
            taxa = Tree(self._fix_semicolon(v['multree']), format=8).get_leaves()
            filled = True if j==0 else False  # If j > 0, it is a nested fix; otherwise, it is the first event
            if (i, j) == (1, 0): # Manually add initial condition
                plot_data.append({
                    'taxa': len(taxa) - len(v['h2.node']),
                    'score': v['score_tuple'][0],
                    'filled': True
                })
            # Last event
            if k == len(self.history) - 1:
                if v['other_tree'] != '':
                    taxa_len = len(Tree(self._fix_semicolon(v['other_tree']), format=8).get_leaves())
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

    # --- Nested event detection and handling ---

    def _find_subset_key(self, lst, counter_dict):
        lst_set = set(lst)
        for k, v in counter_dict.items():
            for l in v:
                if lst_set.issubset(set(l)): return k
        return None

    #def check_and_fix_nested(grampa, current_dir, mt_dict, genetrees, h_counter_state, ignore_nesting, debug=False):
    '''def check_and_fix_nested(self, mt_dict, genetrees, engine_callback, h_counter_state=None, ignore_nesting=None, debug=False):
        """Fixes nested events by triggering sub-runs via the engine_callback."""
        new_events = {0: [mt_dict, genetrees]}
        
        if self.ignore_nesting or genetrees is None:
            return new_events

        curr_mt = mt_dict
        curr_gts = genetrees
        event_id = 0

        while True:
            multree = curr_mt["tree"]
            h1_node = curr_mt["h1.node"]
            h2_node = curr_mt["h2.node"]
            h1_lnames = h1_node.get_leaf_names()
            h2_lnames = h2_node.get_leaf_names()

            # Update counter with current state
            self.h_counter.update({
                f"iter_{self.curr_i}_{event_id}": {
                    "h1.node": h1_lnames,
                    "h2.node": h2_lnames,
                }
            })

            h1_sis = h1_node.get_sisters()[0] if h1_node.get_sisters() else None
            h2_sis = h2_node.get_sisters()[0] if h2_node.get_sisters() else None
            
            found_nested_event = False
            for h_sis_node, h_main_node in [(h1_sis, h2_node), (h2_sis, h1_node)]:
                if not h_sis_node: continue
                
                sis_leaves = h_sis_node.get_leaf_names()
                # Find if this sister group is actually a previously collapsed hybrid
                h_group_key = self._find_subset_key(sis_leaves, self.h_counter.h_counter)

                if h_group_key is None: continue

                # Look for targets within that group that aren't yet handled
                for targets in self.h_counter.h_counter[h_group_key]:
                    filter_base = {self.get_base_name(f) for f in sis_leaves}
                    filtered_targets = [t for t in targets if self.get_base_name(t) in filter_base]

                    if not filtered_targets or set(filtered_targets) == set(sis_leaves):
                        continue

                    # Trigger nested sub-run via engine
                    # Note: engine_callback expects (st, gts, h1_str, h2_str)
                    res = engine_callback(
                        multree, curr_gts, 
                        ",".join(h_main_node.get_leaf_names()), 
                        ",".join(filtered_targets)
                    )

                    if res and res['min_idx'] != 0:
                        # Process results of nested run
                        curr_mt, curr_gts = self.process_prev_iter(res)
                        event_id += 1
                        new_events[event_id] = [curr_mt, curr_gts]
                        found_nested_event = True
                        break
                
                if found_nested_event: break
            if not found_nested_event: break

        return new_events'''
        
    def check_and_fix_nested(self, mt_dict, genetrees, engine_callback, curr_i, out):
        """Original nested fix logic maintained, but uses unified preparer."""
        new_events = {0: [mt_dict, genetrees]}
        if self.ignore_nesting or genetrees is None:
            return new_events

        curr_mt, curr_gts = mt_dict, genetrees
        event_id = 0

        while True:
            h1_node, h2_node = curr_mt["h1.node"], curr_mt["h2.node"]
            self.h_counter.update({
                f"iter_{curr_i}_{event_id}": {
                    "h1.node": h1_node.get_leaf_names(),
                    "h2.node": h2_node.get_leaf_names(),
                }
            })

            h1_sis = h1_node.get_sisters()[0] if h1_node.get_sisters() else None
            h2_sis = h2_node.get_sisters()[0] if h2_node.get_sisters() else None
            
            found_nested_event = False
            for h_sis_node, h_main_node in [(h1_sis, h2_node), (h2_sis, h1_node)]:
                if not h_sis_node: continue
                
                h_group_key = self._find_subset_key(h_sis_node.get_leaf_names(), self.h_counter.h_counter)
                if h_group_key is None: continue

                for targets in self.h_counter.h_counter[h_group_key]:
                    filter_base = {n.split('.', 1)[0] for n in h_sis_node.get_leaf_names()}
                    filtered_targets = [t for t in targets if t.split('.', 1)[0] in filter_base]

                    if not filtered_targets or set(filtered_targets) == set(h_sis_node.get_leaf_names()):
                        continue

                    res = engine_callback(curr_mt["tree"], curr_gts, 
                                        ",".join(h_main_node.get_leaf_names()), 
                                        ",".join(filtered_targets),
                                        out / f"nf_{event_id}"
                                        )

                    if res and res['min_idx'] != 0:
                        curr_mt, curr_gts = self._prepare_next_step_data(res, None, None)
                        event_id += 1
                        new_events[event_id] = [curr_mt, curr_gts]
                        found_nested_event = True
                        break
                if found_nested_event: break
            if not found_nested_event: break

        return new_events
    
    # --- Overlapping logic for Full and Split modes ---

    @staticmethod
    def __get_actual_h2(mt_wrapper: SmrtTree, h2_sister_name: str, h_clade: list) -> TreeNode:
        """
        Consistently identifies the H2 insertion node.
        It is the node that is a child of the backbone but contains the H* clade.
        """
        '''# 1. Get any leaf that belongs to the star-clade
        star_leaf_name = f"{h_clade[0]}*"
        star_leaf = mt_wrapper.get_node(star_leaf_name)
        
        # 2. Trace up from the star leaf to find the highest node that 
        # consists ONLY of star-leaves. This is the root of the grafted H clade.
        curr = star_leaf
        while curr.up and all(l.name.endswith('*') for l in curr.up.get_leaves()):
            curr = curr.up
            
        # 3. The 'Actual H2' for surgery purposes is the parent of this grafted clade
        # because that is the node created by to_mul_tree to join the backbone.
        return curr.up if curr.up else curr'''

        # for now, find the common ancestor of all leaves with star suffix
        star_leaves = [mt_wrapper.get_node(f"{name}*") for name in h_clade]
        if len(star_leaves) == 1:
            return star_leaves[0]
        return mt_wrapper.ete_tree.get_common_ancestor(star_leaves)

    @staticmethod
    def _debug_tree(title: str, ete_tree: Tree) -> None:
        print(f"\n[DEBUG] {title}")
        print(ete_tree.get_ascii(show_internal=True, attributes=['name']))

    def __prepare_handled_data(self, res, logger, debug) -> Tuple[Optional[dict], Optional[pd.Series]]:
        """
        Unified logic to validate scores, apply cutoffs, and rename trees.
        Refactored from process_prev_iter.
        """
        min_idx = res['min_idx']
        if min_idx == 0: return None

        # Self map is index 0; score 999999 if missing as a very high score
        self_score = res['sorted_scores_dict'].get(0, 999999)
        best_score = res['min_score']
        if debug: print(f"[DEBUG] Self-mapping score: {self_score}, Best non-self score: {best_score}")
        
        # Cutoff Check (Uses update_mp_cutoff logic from iter_mode)
        if not self.check_cutoff(self_score, best_score): 
            return None

        # Resolve H nodes
        mul_data = res['mul_data']
        # Direct O(1) dictionary lookup in GrandmaTree instead of O(N) search_nodes of ete3
        #h1_node = mul_data.mt.get_node(mul_data.h1_node)
        h1_node = mul_data.h1_node

        # Correctly identify H2 node (sister of the sister provided in results): TBC!!!
        #h2_sister = engine_res['mul_data'].mt.get_node(engine_res['mul_data'].h2_node)
        #h2_node = [c for c in h2_sister.up.get_children() if c != h2_sister][0]
        h2_node = self.get_actual_h2(mul_data.mt, mul_data.h2_node, mul_data.h_clade)
        
        if debug: self._debug_tree(f"Split Context: {mul_data.h1_node} (H1) | {mul_data.h2_node} (H2 [sister]) | {h2_node.name} (H2 [actual])", mul_data.mt.ete_tree)

        # Finalize MT structure for return
        meta = {
            'h1.node': h1_node,
            'h2.node': h2_node,
            'score': best_score,
            'score_tuple': (self_score, best_score),
            'other_tree': ''
        }
        return meta

    # --- Handlers for the Full mode ---

    def rename_trees_for_next_iter(self, res: StepResult, sample: List[int] = None) -> pd.Series:

        best_mt_idx = res.mt_idx()
        best_mt = res.mul_trees[best_mt_idx]

        h1_node = best_mt.h1_node
        h2_node = best_mt.h2_node
        h1_leaves = h1_node.get_leaves()
        h2_leaves = h2_node.get_leaves()
        disallowed_lvs = [l.name for l in h1_leaves]
        
        distance_map = {}
        for n in best_mt.mt.ete_tree.traverse():
            d1 = n.get_distance(h1_node, topology_only=True)
            d2 = n.get_distance(h2_node, topology_only=True)
            distance_map[n.name] = ".1" if d1 < d2 else ".2"

        mapped_gts_list = []
        gts = res.gene_trees
        min_maps = res.kept_mul_maps[best_mt_idx]
        for g_idx, recon_res in min_maps.items():
            gt_ete = gts[g_idx].ete_tree.copy() # Operations on copies

            if sample is not None and g_idx in sample:
                print(f"ReconResult: {recon_res}")
                self._debug_tree(f"Original GT {g_idx}:", gt_ete)

            for l in gt_ete.iter_leaves():
                if l.name.split('_')[1] in disallowed_lvs:
                    # For now, default for the first map when there are multiple options
                    l.name += distance_map[recon_res.maps[l.name][0]]
            mapped_gts_list.append(gt_ete)

            if sample is not None and g_idx in sample:
                self._debug_tree(f"Renamed GT {g_idx}:", gt_ete)
        
        # Rename MT nodes for next step logic
        for l in h1_leaves: l.name += ".1"
        for l in h2_leaves: l.name = l.name.replace("*", "") + ".2"

        if sample is not None:
            self._debug_tree("Renamed MT for next iteration:", best_mt.mt.ete_tree)
        
        return best_mt.mt, mapped_gts_list

    def handle_iteration_result(self, i: int, res: StepResult, iter_out: Path, engine_callback, logger, debug: bool) -> Optional[Tuple[SmrtTree, Dict[int, SmrtTree]]]:
        """
        Handles the end of a 'Full' mode iteration.
        Returns: (next_st, next_gts) or None if stopping.
        """
        # 1. Validation & Cutoff
        if not self._to_proceed(res, f'Iteration {i}', logger, debug):
            return None
        sample = [10, 13] if debug else None

        # 2. Rename Trees for Next Iteration
        next_mt, next_gts = self.rename_trees_for_next_iter(res, sample=sample)
        res.mul_data.mt.ete_tree = next_mt.ete_tree  # Update tree object

        # 3. Check for Nested Hybridization
        # This encapsulates the while-loop for recursive sub-fixes
        new_events = self.check_and_fix_nested(
            mt_dict=res,
            genetrees=next_gts,
            engine_callback=lambda st, gts, h1, h2, out: engine_callback(st, gts, h1, h2, out),
            curr_i=i,
            out=iter_out
        )

        logger.write(f"# Iteration {i} produced {len(new_events)} event(s) after nested checks.", level=1)
        """if genetrees is None:
                    print(f'\nNo further events found. Terminating GRAMPS at iteration {i}.')
                    break"""

        # 4. Update History & Disk
        self.update_history(i, new_events)
        last_idx = max(new_events.keys())
        final_mt, final_gts_list = new_events[last_idx]
        
        # Convert list back to Dict for GrandmaTree consumption
        next_st = SmrtTree(tree_obj=final_mt['tree'])
        next_gts = {idx: SmrtTree(tree_obj=gt) for idx, gt in enumerate(final_gts_list)}
        
        # Write handoff files for resume support
        self.write_handoff_files(final_mt['tree'], final_gts_list, iter_out.parent)
        
        return next_st, next_gts

    # --- Handlers for the Split mode ---
    
    def extract_subproblems(self, res: StepResult, depth: int, idx: int, min_gt_leaves: int = 2, min_st_leaves: int = 1,
                            sample: List[int] = None) -> None:
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

        try:
            h_clade_names = [l.name for l in h1_node.get_leaves()]
        except Exception as e:
            raise RuntimeError(f"Error accessing leaves of H1 node: {e}")

        # --- 2. Inner Sub-problem (Hybrid Clade) ---
        # Species Tree: Copy the subtree rooted at H1
        inner_st_obj = h1_node.copy()
        
        inner_gts = {}
        new_gt_counter = 0
        for g_idx, gt_wrapper in gts.items():
            maps = min_maps.get(g_idx)
            if not maps: continue

            if sample is not None and g_idx in sample:
                self._debug_tree(f"Inner GT {g_idx} (Pre-split):", gt_wrapper.ete_tree)

            gt_ete = gt_wrapper.ete_tree.copy()

            # --- O(N) Strategy for Pure Clade Extraction ---
            # Pass 1: Bottom-up purity caching (Postorder)
            node_is_pure = {}
            for node in gt_ete.traverse("postorder"):
                if node.is_leaf():
                    # Check if the single leaf maps to the hybrid clade
                    mapped_sp = maps[node.name][0].replace("*", "")
                    node_is_pure[node] = mapped_sp in h_clade_names
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
                    if len(node) >= min_gt_leaves:
                        final_pure_lineages.append(node)
                    # SUCCESS: We found the largest clade for this branch. 
                    # Do NOT add children to the stack; this skips the entire subtree.
                else:
                    # Node isn't pure, so we must check its children
                    stack.extend(node.children)

            for ph_node in final_pure_lineages:
                extracted_gt = ph_node.copy()
                inner_gts[new_gt_counter] = SmrtTree(tree_obj=extracted_gt)
                
                if sample is not None and g_idx in sample:
                    self._debug_tree(f"Extracted Inner as Lineage {new_gt_counter}:", extracted_gt)
                
                new_gt_counter += 1

        # --- 3. Outer Sub-problem (Backbone) ---
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

        outer_gts = {}
        for g_idx, gt_wrapper in gts.items():
            maps = min_maps.get(g_idx)
            if not maps: continue

            if sample is not None and g_idx in sample:
                self._debug_tree(f"Outer GT {g_idx} (Pre-prune):", gt_wrapper.ete_tree)

            gt_ete = gt_wrapper.ete_tree.copy()
            # Keep leaves that did NOT map to the hybrid clade
            to_keep = [l for l in gt_ete.iter_leaves() 
                       if maps[l.name][0].replace("*", "") not in h_clade_names]
            
            if len(to_keep) >= min_gt_leaves:
                gt_ete.prune(to_keep, preserve_branch_length=True)
                outer_gts[g_idx] = SmrtTree(tree_obj=gt_ete)

                if sample is not None and g_idx in sample:
                    self._debug_tree(f"Pruned Outer GT {g_idx}:", gt_ete)
        
        if sample is not None:
            self._debug_tree("Inner Species Tree (Hybrid Clade):", inner_st_obj)
            self._debug_tree("Outer Species Tree (Backbone):", outer_st_obj)
            print(f'len(inner_gts)={len(inner_gts)}, len(outer_gts)={len(outer_gts)}')

        # --- 4. Queue Tasks with Binary IDs ---
        # Only queue tasks if species tree has enough leaves to be valid
        next_tasks = []
        if len(inner_st_obj.get_leaves()) >= min_st_leaves and len(inner_gts) > 0:
            next_tasks.append((SmrtTree(tree_obj=inner_st_obj), inner_gts, f"{depth + 1}.{idx * 2}"))
        if len(outer_st_obj.get_leaves()) >= min_st_leaves and len(outer_gts) > 0:
            next_tasks.append((SmrtTree(tree_obj=outer_st_obj), outer_gts, f"{depth + 1}.{idx * 2 + 1}"))
        return next_tasks

    def handle_split_result(self, bin_id, res: StepResult, iter_out, logger, debug):
        """
        Processes a split worker result.
        Returns: List of new sub-tasks or empty list.
        """
        # 1. Determine Depth and Index from Binary ID
        depth, idx = (int(x) for x in bin_id.split('.')) if '.' in bin_id else (0, 0)

        # 2. Validation & Cutoff
        if not self._to_proceed(res, f"Depth {depth}, Index {idx}", logger, debug):
            return []

        logger.write(f"# Reticulation found at Depth {depth}, Index {idx} with score {res.mt_score()}.", level=1)
        sample = [10, 13] if debug else None
        
        # 3. Extract Subproblems
        try:
            next_tasks = self.extract_subproblems(res, depth, idx, sample=sample)
        except Exception as e:
            logger.write(f"Error extracting subproblems at Depth {depth}, Index {idx}: {e}", level=1)
            #print(res)
            return []
        
        # 4. Update History
        self.update_history(depth, {idx: res})
            
        return next_tasks

    # --- Post-processing for Split mode ---

    def glue_split_results(self, output_dir: Path, original_st_path: Path, logger) -> None:
        """
        Recombines recursive sub-analyses into a single global hybridization record.
        Iterates reverse (deepest first), updating leaf names with .1/.2 suffixes.
        Produces both a suffix-separated single-label tree and a clean MUL-tree.
        """
        step = "Recombining Split Results"
        logger.report_step(step, "In progress...", start=True)

        if not self.history:
            logger.write("No reticulations found. Nothing to glue.", level=1)
            return

        # 1. Load Base Species Tree
        try:
            with open(original_st_path, 'r') as f:
                st_text = f.read().strip()
                if not st_text.endswith(';'): st_text += ';'
                # Use GrandmaTree wrapper for robust newick parsing
                base_st = SmrtTree(newick=st_text)
        except Exception as e:
            logger.write(f"Error loading original ST for gluing: {e}", level=1)
            return

        # 2. Sort Events: Reverse order (Deepest depth/index first)
        # self.history keys are (depth, idx) tuples.
        sorted_keys = sorted(self.history.keys(), key=lambda x: (x[0], x[1]), reverse=True)
        
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
                lname_base = l.name.split('.')[0]
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
            event = self.history[key]
            
            # A. Extract Topology Info from Event
            h1_names = event['h1.node'] # Base names defining the lineage
            
            # We need to find the Sister of H2 to know WHERE to graft.
            # We parse the local event tree string to find this relationship.
            local_mt = Tree(event['multree'], format=1) 
            
            # Find H2 leaves in local tree (they usually have '*' or are the second occurence)
            # In split mode history, h2.node names usually come with '*' suffix from the run
            h2_leaves_local = []
            for n in local_mt.get_leaves():
                # Match against recorded h2 names
                if n.name in event['h2.node']:
                    h2_leaves_local.append(n)
            
            if not h2_leaves_local:
                logger.write(f"Warning: Could not find H2 leaves in event {key} tree structure.", level=1)
                continue
            
            # Get Sister of H2 in local tree
            local_h2_node = local_mt.get_common_ancestor(h2_leaves_local) if len(h2_leaves_local) > 1 else h2_leaves_local[0]
            sisters = local_h2_node.get_sisters()
            if not sisters:
                logger.write(f"Warning: H2 node in event {key} has no sister (Root?).", level=1)
                continue
            
            # Extract base names of the sister clade
            local_sister_leaves = [n.name.replace('*','').split('.')[0] for n in sisters[0].get_leaves()]

            # B. Locate Nodes in Global Tree
            h1_node_global = find_clade_root(current_tree, h1_names)
            target_sister_global = find_clade_root(current_tree, local_sister_leaves)

            if not h1_node_global or not target_sister_global:
                logger.write(f"Warning: Could not map event {key} to global tree. Skipping.", level=1)
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
            
        logger.report_step(step, "Success: Merged trees written.")
        
        # replace .1 with + and .2 with *, and log to logger
        logger.write(f"# Merged tree inferred: {sl_str.replace('.1', '+').replace('.2', '*')}")
