import re
import json
import random
import pandas as pd
from typing import Tuple, List, Optional, Dict, Set
from collections import defaultdict
from pathlib import Path
from functools import partial

from .config import GlobalContext
from .models import Tree, TreeNode, SmrtTree, TaskResult, MulTree, HistoryType, ConcurrTask
from .ops import CommonOps
from .logger import GranLogger

class FlowManager:
    def __init__(self, ctx: GlobalContext, mode: str, logger: GranLogger):
        self.ctx = ctx
        self.mode = mode
        self.sample = self.set_sampling_func(2)
        self.logger = logger
        
    # --- Init Methods ---

    def set_sampling_func(self, n: int) -> callable:
        if self.ctx.debug:
            if self.ctx.seed:
                def _random_spacing(iterable, n):
                    """Returns n random values from iterable using fixed seed."""
                    idxs = sorted(random.sample(range(len(iterable)), min(n, len(iterable))))
                    return [iterable[i] for i in idxs]
                return partial(_random_spacing, n=n)
            else:
                def _equal_spacing(iterable, n):
                    """Returns n evenly spaced values from iterable."""
                    length = len(iterable)
                    if n >= length:
                        return list(range(length))
                    step = length / n
                    return [iterable[int(i * step)] for i in range(n)]
                return partial(_equal_spacing, n=n)
        def _noop(iterable):
            return []
        return _noop

    # --- Overlapping Methods for Full and Split Modes ---

    def _check_if_passed(self, i: int, j: int) -> bool:
        """Returns True if the event should be accepted."""
        cut_type, cut_val = self.ctx.cutoff
        curr_event = self.ctx.history[(i, j)]

        self.logger.log(f"Checking if passed for event ({i}, {j}): {curr_event}", 'd')

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

    @staticmethod
    def _get_sis_nodes(node1: TreeNode, node2: TreeNode) -> List[TreeNode]:
        """Returns the sister node of the given node, or None if not found."""
        if node1.up and node2.up and node1.up.name == node2.up.name:
            # autopolyploid case
            node1, node2 = node1.up, node2.up
        res = []
        for node in [node1, node2]:
            parent = node.up
            if parent is None:
                if None not in res:
                    res.append(None)
            else:
                children = parent.get_children()
                if len(children) != 2:
                    raise ValueError("Tree structure invalid for sister retrieval.")
                sister = children[0] if children[1] == node else children[1]
                # Append sister if not already added
                if sister not in res:
                    res.append(sister)
        if len(res) < 1:
            raise ValueError("Sister node not found.")
        return res

    def _update_history(self, i: int, j: int, res: TaskResult, hold: bool = False) -> bool:
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
        self.logger.log(f"Best non-input index: {nonin_idx}, Best index: {best_idx}", 'd')
        self.logger.log(f"Input map score: {input_score}, Best non-input map score: {nonin_score}", 'd')

        best_mt = res.mul_trees[best_idx]
        nonin_mt = res.mul_trees[nonin_idx]
        self._debug_tree("Best MulTree:", best_mt.mt.ete_tree)
        self._debug_tree("Best Non-input MulTree:", nonin_mt.mt.ete_tree, other_attr=['H', 'pure'])

        h_nodes = [nonin_mt.h1_node] + nonin_mt.hx_nodes
        sister_nodes = [nonin_mt.mt.get_sis(n) for n in h_nodes]

        #sis_nodes = self._get_sis_nodes(nonin_mt.h1_node, nonin_mt.h2_node)

        # Embed an attr H in each sis_node
        #for n in sis_nodes:
        #    if n is None: continue
        #    if not hasattr(n, 'H'):
        #        n.add_feature('H', [])
        #    n.H.append(str((i, j))) # Track which events this node was involved in for nested detection / gluing logic

        self.ctx.history[(i, j)] = {
            'best_mt': best_mt.mt.to_str(internals=True),
            'nonin_mt': nonin_mt.mt.to_str(internals=True),
            'h_name': nonin_mt.h1_node.name,
            'h_locs': [n.name if n is not None else 'None' for n in sister_nodes], # H nodes or sisters cannot be the root node! But may be None in source STs
            'h_leaves': nonin_mt.h1_node.get_leaf_names(),
            'num_gts': len(res.gene_trees),
            'input_score': input_score,
            'nonin_score': nonin_score,
        }
        #if self.logger.debug:
        #    # No longer needed to parse iterations, but is very useful for debugging
        #    track_dict = {n.name: n.H for n in nonin_mt.mt.ete_tree.traverse() if hasattr(n, 'H')}
        #    self.ctx.history[(i, j)]['trackers'] = track_dict

        passed = self._check_if_passed(i, j)
        self.ctx.history[(i, j)]['passed'] = passed

        if not hold: # For full mode, until nested fixes are done (to not pollute history with partial events)
            with open(self.ctx.history_file, 'w') as f:
                json.dump({str(k): v for k, v in self.ctx.history.items()}, f, indent=4)

        return passed

    # --- Handlers for the Full mode ---

    def _rename_best_mt(self, res: TaskResult, best_mt_idx: int, i: int, j: int) -> Tuple[MulTree, Dict[str, Set[str]]]:
        """
        Renames the best MulTree's hybrid lineages for the next iteration.
        Format: |{i}.{j~copy_idx} (e.g., Species|1.0, <Internal|1.1>)
        best_mt is modified in place!
        
        Returns: 
            best_mt: The modified MulTree (modified in place).
            suffix_name_map: A mapping from suffix to the set of original names that were suffixed.
        """
        best_mt = res.mul_trees[best_mt_idx]
        mt_wrapper = best_mt.mt
        h1_node = best_mt.h1_node # Source
        hx_nodes = best_mt.hx_nodes # Copies

        hx_str = [f'{n.name} (H{x+2})' for x, n in enumerate(hx_nodes)]

        self._debug_tree(f"Renaming Context: {h1_node.name} (H1) | {' | '.join(hx_str)}", mt_wrapper.ete_tree)

        suffix_name_map = best_mt.rename_marked_nodes(i, j)
        
        self._debug_tree("Renamed MT:", mt_wrapper.ete_tree, other_attr=['H', 'pure'])
        return best_mt, suffix_name_map # suffix -> set of original names (for syncing GTs)

    def _partition_gt_leaves(self, res: TaskResult, best_mt_idx: int, suffix_name_map: Dict[str, Set[str]]) -> Dict[int, SmrtTree]:
        """
        Renames Gene Trees to match the Species Tree renaming logic.
        Uses the mapping generated by _rename_best_mt.
        Returns gt dict, but - GTs are modified in place!
        """
        gts = res.gene_trees
        min_maps = res.kept_mul_maps[best_mt_idx]
        
        debug_sample = self.sample(list(min_maps.keys()))

        for g_idx, map_obj in min_maps.items():
            gt_ete = gts[g_idx].ete_tree

            if g_idx in debug_sample:
                self._debug_tree(f"Original GT {g_idx}:", gt_ete)

            gts[g_idx].rename_leaves_from_mapping(map_obj, suffix_name_map)

            if g_idx in debug_sample:
                self._debug_tree(f"Renamed GT {g_idx}:", gt_ete, other_attr=['H', 'pure'])
        
        return gts

    def find_missing_targets(self, multree: MulTree) -> Set[str]:
        if self.ctx.nesting in {"ignore", "model"}: return set() # Extra safety

        h1_node = multree.h1_node
        matches = multree.mt.match(h1_node.pure)
        get_sis = multree.mt.get_sis

        self.logger.log(f"Nested Fix: Found {[l.name for l in matches]} matches for H1 node in the MT.", 'd')
        all_h2_locs_names = set()
        already_populated = set()
            
        for h_node in matches:

            h_sis = get_sis(h_node)

            h2_locs = multree.mt.get_targets(h_sis)

            # New internality check logic
            skip_match = False
            if self.ctx.nesting == "strict_rectify":
                # Select one other sister as checking candidate for internality
                # We check the other sister, because h_node's P node may be sister's .up...
                other_sis = None
                for sis in h2_locs:
                    if get_sis(sis).pure != h_node.pure:
                        other_sis = sis
                        break
                if other_sis:
                    internal_count = len(h2_locs)
                    # If the parent of one of the other sisters has a count that's different from the sister's copy count,
                    # Then, the sister's lineage is rooted at the sister. Meaning, it is "external" and we should skip it.
                    parental_count = len(multree.mt.match(other_sis.up.pure))
                    if parental_count != internal_count:
                        # Not internal, so skip [strict_rectify only corrects internal nested cases]
                        skip_match = True

            if skip_match:
                h2_locs_names = set()
            else:
                h2_locs_names = set(n.name for n in h2_locs)

            self.logger.log(f"Nested Fix: For sister '{h_sis.name}', found H2 locations {h2_locs_names} after filtering.", 'd')
            all_h2_locs_names.update(h2_locs_names)
            
            already_populated.add(h_sis.name)

        # Targets must appear once, and exclude already populated ones
        # It is not safe to remove within the loop - the next match might find another copy of the same lineage.
        # We must track separately and remove at the end.
        return all_h2_locs_names - already_populated

    ### Old logic:
    # Works, but I found an easier way to check this, and earlier - during target search.
    # Thus it is currently unused.
    def _check_internality(self, t: Tree, node: TreeNode) -> bool:
        """
        Loads all the events from history that contain this node in their H1.
        Out of these events, finds the smallest one (most recent) and checks if the node is the root of the H1 subtree in that event.
        [A single species lineage is root by default]
        """
        containing_events = []
        # FIX 1: We must include the node itself in the check set.
        # If 'node' IS the H1 root, it must match 'h_node'. 
        # iter_ancestors() only gives strict parents.
        lineage_names = {p.name for p in node.iter_ancestors()}
        lineage_names.add(node.name)

        for event_data in self.ctx.history.values():
            locs = event_data.get('h_locs', [])
            for loc in locs:
                # FIX 2: Safety check. History might contain old names not in 't'
                loc_node = t.get_node(loc)
                if not loc_node: 
                    continue
                
                sisters = loc_node.get_sisters()
                if not sisters: 
                    continue

                h_node = sisters[0] # The H1 root is the sister of the graft location
                
                # Check if this H1 root is the node itself or one of its ancestors
                if h_node and h_node.name in lineage_names:
                    # Found a containing event
                    containing_events.append((h_node, len(h_node.get_leaves())))
                    # Optimization: No need to check other locs for the same event
                    break 

        if not containing_events:
            return False

        # FIX 3: Logic Confirmation
        # We find the "smallest" H1 (fewest leaves). In nested scenarios, 
        # the smaller H1 is the more recent nested event inside a larger older H1.
        smallest_event_root, _ = min(containing_events, key=lambda x: x[1])
        
        # Return True only if this node IS the root of that most recent event
        return smallest_event_root.name == node.name

    def autocorrect(self, targets: Set[str], multree: MulTree, genetrees: Dict[int, SmrtTree], iter: int,
                   engine_callback: callable) -> Tuple[MulTree, Dict[int, SmrtTree]]:
        """
        Detects nested hybridization by finding 'orphaned' copies of the H-lineage
        directly in the tree structure using the .match() capability.
        Returns the corrected MulTree and Gene Trees after iteratively fixing each detected nested copy.
        """        
        curr_mt = multree
        curr_gts = genetrees 
        
        # Start looking for Copy 2 (but index starts at 0)
        next_copy_idx = 1

        # Sort targets to ensure deterministic order (important for testing/debugging)
        # We could sort by clade size to avoid needing to update future targets as we rename,
        # But this is less intuitive and not always correct.
        pending_targets = sorted(list(targets))
        
        for t_idx in range(len(pending_targets)):
            target = pending_targets[t_idx]
            
            h2_loc = curr_mt.mt.get_node(target)
            if h2_loc is None:
                self.logger.log(f"Nested Fix: Target node '{target}' not found in current MT. Skipping.", 'w')
                continue

            # If nest_internal_only is True, skip if the target is the most external node in the event that produced it.
            '''if self.ctx.nesting == "strict_rectify" and self._check_internality(curr_mt.mt, h2_loc):
                self.logger.log(f"Nested Fix: Target node '{h2_loc.name}' is root in the smallest event containing it. Skipping due to strict_rectify mode.", 'd')
                continue'''

            # Trigger Nested Fix
            self.logger.log(f"Nested Fix: Nested Event Detected! Locating missing copy at the branch leading to {h2_loc.name}", 'i')

            h1_leaves = curr_mt.h1_node.get_leaves()
            
            fix_dir = self.ctx.root_dir / f'{iter}.{next_copy_idx}' / 'output'
            
            # Run Task to infer reconciliation for this missing copy
            # We treat the 'Missing Candidate' as the H2 (Target) 
            # and the current H1 as the source.
            step = f"Nested Fix Iteration {iter}.{next_copy_idx}"
            self.logger.report_step(step, "In progress...")

            res, _ = engine_callback(
                curr_mt.mt, curr_gts,
                ",".join([l.name for l in h1_leaves]), # H1 is the reference
                ",".join([l.name for l in h2_loc.get_leaves()]), # H2 is the new found copy
                fix_dir
            )

            self.logger.report_step(step, "Success")

            curr_mt, curr_gts, suffix_name_map = self.handle_iteration_result(
                iter, res, engine_callback, fix_dir, self.logger, j=next_copy_idx
            )

            # Update future targets in the list if they were renamed
            suffix = f"{iter}.{next_copy_idx}"
            for k in range(t_idx + 1, len(pending_targets)):
                future_target = pending_targets[k]
                # In the rectify case, there's only one Hx node
                if future_target in suffix_name_map[suffix]:

                    if future_target.startswith("<P*"):
                        new_name = f"{future_target}{iter}|{suffix}"
                    else:
                        new_name = f"{future_target}|{suffix}"

                    pending_targets[k] = new_name
                    self.logger.log(f"Nested Fix: Updated pending target '{future_target}' to '{new_name}'", 'd')

            next_copy_idx += 1

        self.logger.log(f"Nested check complete. Found {next_copy_idx-1} extra copies.", 'i')

        return curr_mt, curr_gts

    def handle_iteration_result(
            self, i: int, res: TaskResult,
            engine_callback: callable,
            iter_out: Path,
            iter_logger: GranLogger,
            j: int = 0
        ) -> Tuple[Optional[MulTree], Optional[Dict[int, SmrtTree]], Dict[str, Set[str]]]:
        """
        Handles the end of a 'Full' mode iteration.
        Returns: (next_st, next_gts) or None if stopping.
        """
        # Set logger for this iteration - not applicable for the split mode
        self.logger = iter_logger

        # Rename best non-input MT
        nonin_idx = self._get_nonin_idx(res)
        next_mt, suffix_name_map = self._rename_best_mt(res, nonin_idx, i, j)

        # Update history & check if passed cutoff
        passed = self._update_history(i, j, res)
        if not passed:
            self.logger.log(f"Cutoff reached: no parsimonious events found at Iteration {i}.", 'i')
            # Returns the input tree, i.e. index 0 in the mt dict.
            return res.mul_trees[0], None, None

        if j == 0:
            self.logger.log(f"Reticulation found at Iteration {i} with score {res.mt_score()}.", 'i')
        # Rename Trees for Next Iteration
        next_gts = self._partition_gt_leaves(res, nonin_idx, suffix_name_map)

        if j == 0 and self.ctx.nesting in {"rectify", "strict_rectify"}:
            # Check for Nested Hybridization
            # This encapsulates the while-loop for recursive sub-fixes
            targets = self.find_missing_targets(next_mt)
            next_mt, next_gts = self.autocorrect(
                targets         = targets,
                multree         = next_mt,
                genetrees       = next_gts,
                iter            = i,
                engine_callback = engine_callback,
            )
        
        # Write handoff files for resume support
        CommonOps.write_handoff_files(iter_out.parent, next_mt.mt.ete_tree, [gt.ete_tree for gt in next_gts.values()])
        
        return next_mt, next_gts, suffix_name_map
    
    # --- Handlers for the Split mode ---
    
    def extract_subproblems(self, res: TaskResult, depth: int, idx: int) -> Tuple[List[ConcurrTask], Dict[int, List[int]]]:
        """
        Refined binary recursion split using ETE3-safe surgery and O(N) GT extraction.
        1. Inner: Extracts independent 'pure' subtrees for each hybrid lineage.
        2. Outer: Backbone with H1 clade collapsed to a placeholder leaf.
        """
        best_mt_idx = res.mt_idx()
        best_mt = res.mul_trees[best_mt_idx]

        h1_node = best_mt.h1_node
        hx_nodes = best_mt.hx_nodes
        mt_wrapper = best_mt.mt
        hx_str = [f'{n.name} (H{x+2})' for x, n in enumerate(hx_nodes)]
        self._debug_tree(f"Splitting Context: {h1_node.name} (H1) | {' | '.join(hx_str)}", mt_wrapper.ete_tree)

        # Pre-compute a mapping from MT nodes to copy indices for O(1) lookup during GT leaf partitioning
        mt_node_to_copy_idx = {}
        for l in h1_node.get_leaves():
            mt_node_to_copy_idx[l.name] = 0
        for x, hx_node in enumerate(hx_nodes):
            for l in hx_node.get_leaves():
                mt_node_to_copy_idx[l.name] = x + 1

        min_maps = res.kept_mul_maps[best_mt_idx]
        gts = res.gene_trees

        outer_gts = {}
        inner_gts = {}
        outie_below_min_counter = 0
        innie_below_min_counter = 0
        innie_counter = 0
        gt_split_dict = defaultdict(list)

        debug_sample = self.sample(list(gts.keys()))

        for g_idx, gt_wrapper in gts.items():
            maps = min_maps.get(g_idx)
            if not maps: continue
            source_gt = gt_wrapper.ete_tree

            if g_idx in debug_sample:
                self._debug_tree(f"Pre-split GT {g_idx}:", source_gt)
                self.logger.log(f"Map for GT {g_idx}: {maps.cor}", 'd')

            # --- Partition leaves by homoeologous copy ---
            rev_map = maps.rev
            outer_gt_lvs = set()
            gt_leaf_to_copy_idx = {} # Inner leaves

            for mt_node, gt_nodes in rev_map.items():
                if mt_node in mt_node_to_copy_idx:
                    copy_idx = mt_node_to_copy_idx[mt_node]
                    # Flatten the mapping so every GT leaf knows its exact state
                    for gt_n in gt_nodes:
                        gt_leaf_to_copy_idx[gt_n] = copy_idx
                else:
                    outer_gt_lvs.update(gt_nodes)
                    
            source_gt_lvs = {l.name for l in source_gt.get_leaves()}
            # We do not need to intersect gt_leaf_to_copy_idx with source_lvs 
            # because during Pass 1, we only query using `node.is_leaf()`.
            outer_gt_lvs = outer_gt_lvs.intersection(source_gt_lvs)

            # --- Outer Sub-problem (Backbone) ---
            gt_ete = source_gt.copy()

            if len(outer_gt_lvs) < self.ctx.min_gt_lvs:
                outie_below_min_counter += 1
            if outer_gt_lvs: # Not empty
                gt_ete.prune(list(outer_gt_lvs), preserve_branch_length=True)
                # Special case to handle single leaf outer GTs which have a bad root: "(Spec);" instead of just "Spec;"
                if len(outer_gt_lvs) == 1:
                    gt_ete = gt_ete.get_leaves()[0].detach()
                outer_gts[g_idx] = SmrtTree(tree_obj=gt_ete)

            if g_idx in debug_sample:
                self._debug_tree(f"Pruned Outer GT {g_idx}:", gt_ete)

            # --- Inner Sub-problem (Hybrid Clade) ---
            gt_ete = source_gt.copy()

            # Pass 1: Bottom-up multi-state purity caching (Postorder)
            node_copy_state = {}
            for node in gt_ete.traverse("postorder"):
                if node.is_leaf():
                    node_copy_state[node] = gt_leaf_to_copy_idx.get(node.name)
                else:
                    child_states = [node_copy_state[child] for child in node.children]
                    # Internal node is pure ONLY if ALL children map to the EXACT SAME copy
                    if child_states and all(s is not None for s in child_states) and len(set(child_states)) == 1:
                        node_copy_state[node] = child_states[0]
                    else:
                        node_copy_state[node] = None

            # Pass 2: Top-down extraction (Optimized)
            final_pure_lineages = []
            # Manual stack allows 'skipping' subtrees
            stack = [gt_ete]
            while stack:
                node = stack.pop()

                copy_idx = node_copy_state.get(node)
                
                # If copy_idx is an integer (0, 1, 2...), this node is pure for that specific copy
                if copy_idx is not None:
                    if len(node) < self.ctx.min_gt_lvs:
                        innie_below_min_counter += 1
                    # SUCCESS: We found the largest pure clade for this homoeolog. 
                    # Nodes are guaranteed to have at least one leaf, and no overlaps
                    final_pure_lineages.append(node)
                    # Do NOT add children to the stack; this skips the entire subtree.
                else:
                    # Node is mixed (eg, contains H1 and H2, or outer leaves), so we must check its children
                    stack.extend(node.children)

            for ph_node in final_pure_lineages:
                # No need to copy() - pure nodes' children aren't added to stack, so no unsafe nested detach()s
                extracted_gt = ph_node.detach()#.copy()
                inner_gts[innie_counter] = SmrtTree(tree_obj=extracted_gt)
                
                if g_idx in debug_sample:
                    self._debug_tree(f"Extracted Inner as Lineage {innie_counter}:", extracted_gt)

                gt_split_dict[g_idx].append(innie_counter)
                innie_counter += 1

        # If all GTs of a given subproblem are below the minimum leaf cutoff, we discard them.
        # Previously, we discarded individual GTs based on this, but it biases the reconciliation score because changes to signal balance.
        if innie_below_min_counter >= len(inner_gts):
            inner_gts = {}
        if outie_below_min_counter >= len(outer_gts):
            outer_gts = {}

        # --- Species Tree Surgery ---

        # Simply copy the subtree rooted at H1 for the Inner ST
        # detach() clears the root (t.up == None)
        inner_st_obj = h1_node.copy().detach()

        # For the Outer ST, we need to remove both H1 and H2
        outer_st_obj = mt_wrapper.ete_tree.copy()
        # Re-locate nodes in the COPY
        h1_in_outer = outer_st_obj.search_nodes(name=h1_node.name)[0]
        hx_in_outer = [outer_st_obj.search_nodes(name=hx_node.name)[0] for hx_node in hx_nodes]
        self.logger.log(f"Removing hybrid clade ({h1_in_outer.name} and {', '.join(n.name for n in hx_in_outer)}) from Outer ST.", 'd')
        outer_st_obj, tree_not_empty = self._trim_outer(outer_st_obj, [h1_in_outer] + hx_in_outer)

        self._debug_tree("Inner Species Tree (Hybrid Clade):", inner_st_obj)
        self._debug_tree(f"Outer Species Tree (Backbone){' is empty' if not tree_not_empty else ''}:", outer_st_obj)
        self.logger.log(f'len(inner_gts)={len(inner_gts)}, len(outer_gts)={len(outer_gts)}', 'd')

        # --- Queue Tasks with Binary IDs ---
        next_tasks = []
        if len(outer_st_obj.get_leaves()) >= self.ctx.min_st_lvs and len(outer_gts) > 0 and tree_not_empty:
            next_tasks.append((SmrtTree(tree_obj=outer_st_obj), outer_gts, f"{depth + 1}.{idx * 2}"))
        if len(inner_st_obj.get_leaves()) >= self.ctx.min_st_lvs and len(inner_gts) > 0:
            next_tasks.append((SmrtTree(tree_obj=inner_st_obj), inner_gts, f"{depth + 1}.{idx * 2 + 1}"))
        return next_tasks, gt_split_dict

    @staticmethod
    def _trim_outer(tree_to_trim: Tree, h_nodes_to_trim: List[TreeNode]) -> Tuple[Tree, List[Optional[str]]]:
        """
        Safely removes both hybrid clades from the Outer Species Tree.
        Returns: (trimmed_tree, trimmed_names)
        trimmed_names: List of names of the parents of the removed H nodes. Empty indicates an empty tree.
        trimmed_tree is modified in place! Must be returned for the special root case!
        """
        # Safely remove H nodes from Outer ST: detach leaf, then delete the resulting knuckle node
        trimmed_names = []
        for n in h_nodes_to_trim:
            n_up = n.up # Save parent before detaching
            if n_up is None:
                return Tree(), [] # Special case: if H node is root, we return an empty tree to indicate no valid outer tree
            trimmed_names.append(n_up.name)
            trackers = getattr(n_up, 'H', []) # Preserve any existing trackers
            n.detach()
            if n_up.up is None:
                children = n_up.get_children()
                if len(children) != 1:
                    raise ValueError("Unexpected structure when removing hybrid clade from Outer ST.")
                # Special case: if parent is root, just promote the child
                child = n_up.get_children()[0]
                tree_to_trim = child.detach()#.copy()
                # Re-assign any trackers to the new root
                if trackers:
                    if not hasattr(tree_to_trim, 'H'):
                        tree_to_trim.add_feature('H', [])
                    tree_to_trim.H.extend(trackers)
            # delete() removes the internal node and connects children to parent
            else:
                n_up_up = n_up.up
                n_up.delete()
                # Re-assign any trackers to the grandparent
                if trackers:
                    if not hasattr(n_up_up, 'H'):
                        n_up_up.add_feature('H', [])
                    n_up_up.H.extend(trackers)
        return tree_to_trim, trimmed_names

    def handle_split_result(
            self, bin_id: str, res: TaskResult,
            iter_out: Path,
            iter_logger: GranLogger
        ) -> Optional[List[ConcurrTask]]:
        """
        Processes a split worker result.
        Returns: List of new sub-tasks or empty list.
        """
        backup_logger = self.logger
        self.logger = iter_logger

        # 1. Determine Depth and Index from Binary ID
        depth, idx = (int(x) for x in bin_id.split('.')) if '.' in bin_id else (0, 0)

        # Update history & check if passed cutoff
        passed = self._update_history(depth, idx, res)
        if not passed:
            self.logger.log(f"Cutoff reached: no parsimonious events found at Depth {depth}, Index {idx}.", 'i')
            return None # Event not taken!

        self.logger.log(f"Reticulation found at Depth {depth}, Index {idx} with score {res.mt_score()}.", 'i')
        
        # 3. Extract Subproblems
        #try:
        next_tasks, gt_split_dict = self.extract_subproblems(res, depth, idx)
        #except Exception as e:
        #    self.logger.log(f"extracting subproblems at Depth {depth}, Index {idx}: {e}", 'e')

        # Write gt_split_dict to a file
        gt_split_path = iter_out.parent / f"gt_splits.json"
        with open(gt_split_path, 'w') as f:
            json.dump(gt_split_dict, f, indent=4)

        # Write handoff files for resume support
        for task_st, task_gts, task_id in next_tasks:
            task_out = iter_out.parent / task_id
            task_out.mkdir(parents=True, exist_ok=True)
            self.logger.log(f"Written handoff files for task {task_id} at {task_out.relative_to(iter_out.parent.parent.parent)}, with {len(task_gts)} GTs.", 's')
            CommonOps.write_handoff_files(task_out, task_st.ete_tree, [gt.ete_tree for gt in task_gts.values()])

        self.logger = backup_logger
        return next_tasks

    def glue_split_results(self) -> SmrtTree:
        """
        Recombines results by recursively diving to the innermost subproblems.
        """
        step = "Recombining Split Results (Recursive)"
        self.logger.report_step(step, "In progress...", start=True)

        # Start the chain from the root task (0,0)
        final_tree = self._iterative_glue((0,0))
        ft_wrapper = SmrtTree(tree_obj=final_tree)

        self.logger.log(f"Final Merged Tree: {final_tree.write(format=9)}", 'i')
        
        self.logger.report_step(step, "Success")
        return ft_wrapper

    def _merge_sister_locs(self, expanded_locs, outer_tree, task_id):
        """
        Merge sister locations to avoid redundant grafts in autopolyploidy cases.
        """
        for loc, all_nodes in expanded_locs.items():
            if len(all_nodes) > 1:
                # Sort by proximity to root (depth)
                all_nodes.sort(key=lambda n: n.get_distance(outer_tree.get_tree_root(), topology_only=True))
                #self.logger.log(f"Glue {task_id}: Multiple targets found for cleaned name {pl}: {[n.up.name for n in all_nodes]}. Attempting to merge sisters.", 'd')
                # Handle pairs each iteration
                i = 0
                final_nodes = []
                flag = False
                while i < len(all_nodes) - 1:
                    nodes = all_nodes[i:i+2]
                    i += 2
                    if not nodes[0].up or not nodes[1].up:
                        final_nodes.extend(nodes)
                        continue
                    if nodes[0].up.name == nodes[1].up.name:
                        parent = nodes[0].up
                        children = parent.get_children()
                        if loc not in children[0].name or loc not in children[1].name:
                            final_nodes.extend(nodes)
                        else:
                            final_nodes.append(parent)
                            flag = True
                if flag:
                    self.logger.log(f"Glue {task_id}: Replacing sister targets with parent {parent.name}.", 'd')
                    expanded_locs[loc] = final_nodes
                else:
                    self.logger.log(f"Glue {task_id}: No sisters merged for loc.", 'd')

    def _iterative_glue(self, root_task_id: Tuple[int, int]) -> Tree:
        """
        Recombines split results using history 'trackers' to identify graft locations.
        (Iterative Stack-Based Implementation)
        Returns:
            the final merged tree for the given root task ID, or;
            the original input tree if the root task was rejected.
        """
        # Stack stores tuples: (task_id, children_visited_flag)
        stack = [(root_task_id, False)]
        results = {}

        while stack:
            task_id, visited = stack.pop()
            
            # Base Case: If this task was never run or didn't pass, 
            # we return None or the input tree.
            if task_id not in self.ctx.history:
                self.logger.log(f"Glue {task_id}: Task {task_id} not found in history.", 'd')
                results[task_id] = None
                continue

            event = self.ctx.history[task_id]

            # Check Pass/Fail
            if not event['passed']:
                self.logger.log(f"Glue {task_id}: Task {task_id} did not pass cutoff.", 'd')
                results[task_id] = None
                continue

            # --- STEP 1: Dive to children (Post-order traversal) ---
            # Outer child: (depth+1, idx*2)
            # Inner child: (depth+1, idx*2 + 1)
            depth, idx = task_id
            outer_id = (depth + 1, idx * 2)
            inner_id = (depth + 1, idx * 2 + 1)
            uid = 2**depth + idx - 1

            if not visited:
                self.logger.log(f"--- Processing Task {task_id} ---", 'd')
                # Push back current node marked as visited
                stack.append((task_id, True))
                # Push children. Inner pushed first so Outer is processed first (LIFO order matches original)
                stack.append((inner_id, False))
                stack.append((outer_id, False))
                continue

            # Post-traversal evaluation for current node
            outer_tree = results.get(outer_id)
            inner_tree = results.get(inner_id)

            # Cleanup memory for children
            if outer_id in results: del results[outer_id]
            if inner_id in results: del results[inner_id]

            self.logger.log(f"Glue {task_id}: Subproblems returned.", 'd')

            # Load "Base State" for this node in the recursion tree of gluing, from the history event.
            # Uses 'best_mt' with format=1
            current_mt = MulTree.from_history_event(event)

            # Account for autopolyploidy
            locs = event['h_locs']
            h1_parent = current_mt.h1_node.up
            if h1_parent.name.startswith('<P*'):
                # Autopolyploidy case - go up by one more to get the actual parent tag
                corrected_loc_node = current_mt.mt.get_sis(h1_parent)
                locs = [corrected_loc_node.name if corrected_loc_node else '<root>']
                h1_parent = h1_parent.up
            parent_tag = h1_parent.name if h1_parent else '<root>'
            self.logger.log(f"Glue {task_id}: Searching for graft locs: {locs}. Found H1 parent {parent_tag}.", 'd')

            if not outer_tree and not inner_tree:
                self.logger.log(f"Glue {task_id}: No Outer and Inner results for task. Returning Current tree.", 'd')
                current_mt.rename_marked_nodes(uid, skip_p_tag=False)
                results[task_id] = current_mt.mt.ete_tree
                continue

            if not inner_tree:
                # Infer inner tree from current tree (best_mt) if missing, since we know the hybrid clade is there
                sister_node = current_mt.mt.get_node(locs[0])
                if sister_node is None:
                    inner_tree = current_mt.mt.ete_tree # <root> case, no sisters
                else:
                    inner_tree = current_mt.mt.get_sis(sister_node)
                self.logger.log(f"Glue {task_id}: No Inner results for task. Retrieved from Current tree using H loc: {locs[0]}.", 'd')
                current_mt.rename_marked_nodes(uid, skip_p_tag=True)

            self._debug_tree(f"Inner Result Tree for Task {task_id}:", inner_tree, other_attr=['H', 'pure'])

            trimmed_names = []
            if not outer_tree:
                nodes_to_detach = []
                for loc in locs:
                    node = current_mt.mt.get_node(loc)
                    sister = current_mt.mt.get_sis(node)
                    nodes_to_detach.append(sister)
                outer_tree = current_mt.mt.ete_tree # Modified in place, but no need to copy - not used later
                outer_tree, trimmed_names = self._trim_outer(outer_tree, nodes_to_detach)
                self.logger.log(f"Glue {task_id}: No Outer results for task. Retrieved from Current tree by removing {trimmed_names} nodes & {[n.name for n in nodes_to_detach]} H locs.", 'd')

            self._debug_tree(f"Outer Result Tree for Task {task_id}:", outer_tree, other_attr=['H', 'pure'])

            outer_tree_wrapper = SmrtTree(tree_obj=outer_tree) # Index
            # inner_tree_wrapper = SmrtTree(tree_obj=inner_tree) # Not needed since we index at the end

            # Expand graft locations
            expanded_locs = {}
            redupl_loc = {}
            for i, loc in enumerate(locs):
                if not loc.startswith('<P*'):
                    matches = outer_tree_wrapper.match(loc)
                    expanded_locs[loc] = matches
                else:
                    redupl_loc[locs[i+1]] = True

            if not expanded_locs:
                self.logger.log(f"Glue {task_id}: Graft locations {locs} not found in Outer tree topology.", 'e')
            self.logger.log(f"Glue {task_id}: Expanded graft locations found: {expanded_locs}", 'd')

            self._merge_sister_locs(expanded_locs, outer_tree, task_id)

            # --- Grafting and renaming logic ---
            flag = False
            for loc, targets in expanded_locs.items():
                self.logger.log(f"Glue {task_id}: Grafting Inner tree to Outer tree at targets for {loc}: {[t.name for t in targets]}.", 'd')
                to_graft = True
                while to_graft:
                    for target in targets:

                        # Extract the surrounding suffix if present (returns empty string if '|' is missing)
                        suffix = target.name.partition('|')[2].rstrip('>')
                        suffix = '|' + suffix if suffix else ''

                        # For copies other than the original, update the parent tag, and preppend the new copy ID to the suffix.
                        if flag:
                            parent_tag = f'<P{uid}>'
                            suffix = f"|{uid}.1{suffix}"

                        # This is an internal node for sure
                        new_name = parent_tag[:-1] + suffix + '>'

                        # Copy the graft and apply the tag suffix immediately
                        graft = SmrtTree.copy_lineage(inner_tree, suffix)

                        # Graft subtree into the outer tree
                        outer_tree = SmrtTree.graft_subtree(outer_tree, target, graft, name=new_name)

                    flag = True

                    if not redupl_loc.get(loc, False):
                        to_graft = False
                    else:
                        # Graft again in the same target if triggered
                        self.logger.log(f"Glue {task_id}: Location {loc} matched previous <P[*]> node. Grafting will be repeated there.", 'd')
                    redupl_loc[loc] = False

            SmrtTree(tree_obj=outer_tree)
            self._debug_tree(f"Post-Graft Tree for Task {task_id}:", outer_tree, other_attr=['H', 'pure'])

            results[task_id] = outer_tree

        if results.get(root_task_id) is None:
            # Fallback to original ST
            self.logger.log("No valid recombination found. Returning the original ST.", 'i')
            # Not finding root in history SHOULD raise an error!
            return Tree(self.ctx.history[root_task_id]['best_mt'], format=1)
        return results.get(root_task_id)

    def fast_forward_split(self, current_tasks: List[ConcurrTask]) -> List[ConcurrTask]:
        """
        Reconstructs the task queue from disk state based on history.
        Uses BFS to traverse solved nodes and identify the frontier.
        """

        self.logger.log("Fast-forwarding Split tasks based on history...", 'i')
        self.logger.log(f"Current tasks: {current_tasks}, output_dir: {self.ctx.root_dir}", 'd')

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

                self.logger.log(f"Task {nid} done. Checking children: {c1}, {c2} in {self.ctx.root_dir / nid}", 'd')
                
                # If child dir exists, add to traversal queue
                if (self.ctx.root_dir / nid / c1 / "multree.tre").exists():
                    queue.append((c1, nid))
                if (self.ctx.root_dir / nid / c2 / "multree.tre").exists():
                    queue.append((c2, nid))
            else:
                # Task NOT in history -> It is a frontier task to run.
                # Load inputs from disk
                p_nid = q[1]
                self.logger.log(f"Queueing frontier task: {nid} (child of {p_nid})", 'd')

                st_path = self.ctx.root_dir / p_nid / nid / "multree.tre"
                gt_path = self.ctx.root_dir / p_nid / nid / "genetrees.txt"
                
                self.logger.log(f"Looking for ST at {st_path}, GTs at {gt_path}", 'd')

                if st_path.exists() and gt_path.exists():
                    real_tasks.append((st_path, gt_path, nid))
                    self.logger.log(f"Resuming sub-problem: {nid}", 's')
                else:
                    # Fallback: if root is missing inputs but passed in current_tasks
                    if nid == str(current_tasks[0][2]):
                        self.logger.log(f"Using provided inputs for root task {nid}, {current_tasks}", 's')
                        real_tasks.extend(current_tasks)
                    else:
                        self.logger.log(f"Missing inputs for resume task {nid}. Skipping.", 'w')

        self.logger.log(f"Fast-forwarded tasks: {real_tasks}", 'd')

        return real_tasks
    
    # --- Output methods ---

    def plot(self):
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator, FixedLocator
        from matplotlib.lines import Line2D
        import numpy as np

        # --- 1. Data Parsing & Grouping ---
        def get_taxa_count(tree_str):
            try:
                t_str = CommonOps._fix_semicolon(tree_str)
                return len(Tree(t_str, format=1).get_leaves())
            except:
                return 0

        # Data container: grouped_history[i] = list of (j, data_dict)
        # Split Mode: i=depth, j=index
        # Full Mode: i=iteration, j=nested_id
        grouped_history = defaultdict(list)
        
        for key, val in self.ctx.history.items():
            i, j = key
            
            # --- Metrics Calculation ---
            out_taxa = get_taxa_count(val['nonin_mt'])
            out_score = val['nonin_score']
            h_lvs = len(val.get('h_leaves', []))
            h_copies = len(val.get('h_locs', []))
            in_taxa = max(0, out_taxa - h_lvs*(h_copies-1 if h_copies > 0 else 0)) # Adjust for hybrid leaves that don't add taxa
            in_score = val['input_score']
            
            out_norm = out_score / out_taxa if out_taxa > 0 else 0
            in_norm = in_score / in_taxa if in_taxa > 0 else 0

            entry = {
                'key': key,
                'in':  {'score': in_score,  'taxa': in_taxa,  'norm': in_norm},
                'out': {'score': out_score, 'taxa': out_taxa, 'norm': out_norm}
            }
            grouped_history[i].append((j, entry))

        if not grouped_history:
            self.logger.log("No history data available to plot.", 'w')
            return

        sorted_iters = sorted(grouped_history.keys())

        # --- 2. Setup Plot ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        metrics = [
            ('score', 'MP Score', 'MP Score (Input vs Inferred)'), 
            ('taxa', 'Taxa Count', 'Taxa Count'), 
            ('norm', 'MP Score / Taxa', 'Normalized Score')
        ]

        # --- 3. Drawing Logic ---
        for ax, (m_key, y_label, title) in zip(axes, metrics):
            
            # Coordinate Cache for connections: Key -> {'in': (x,y), 'out': (x,y)}
            # Key is (depth, idx) for split, or (iter, sub) for full
            coord_map = {}
            
            # Track previous bin's last output for Full mode sequential connections
            last_bin_coords = None 

            for i in sorted_iters:
                events = sorted(grouped_history[i], key=lambda x: x[0])
                n_events = len(events)
                
                # Intra-bin connection tracker
                prev_sub_x, prev_sub_y = None, None
                
                # Bin Logic
                bin_width = 0.8

                for k, (j, data) in enumerate(events):
                    # X Coordinates
                    if self.mode == 'split':
                        # Split Mode: No spreading. All events at this depth share the same center.
                        # The "2 points" (slope) width is fixed.
                        slot_center = i
                        slot_width = bin_width
                    else:
                        # Full Mode: Spread events across the bin width to show nesting order
                        slot_width = bin_width / n_events
                        slot_center = (i - bin_width/2) + (slot_width * (k + 0.5))
                    
                    half_line = (slot_width * 0.8) / 2
                    x_in = slot_center - half_line
                    x_out = slot_center + half_line
                    
                    y_in = data['in'][m_key]
                    y_out = data['out'][m_key]
                    
                    # Store exact coords for later connection logic
                    coord_map[(i, j)] = {'in': (x_in, y_in), 'out': (x_out, y_out)}

                    # Marker Style
                    is_nested = (self.mode != 'split' and j > 0)
                    marker_shape = 'D' if is_nested else 'o'
                    marker_size = 40 if not is_nested else 30

                    # Plot Slope & Points
                    ax.plot([x_in, x_out], [y_in, y_out], color='black', linewidth=1, zorder=3)
                    ax.scatter(x_in, y_in, facecolors='white', edgecolors='black', marker=marker_shape, s=marker_size, zorder=4)
                    ax.scatter(x_out, y_out, color='black', marker=marker_shape, s=marker_size, zorder=4)

                    # --- Connections (Immediate) ---
                    if self.mode != 'split':
                        # Full Mode Intra-bin: Connect Nested to previous in same bin
                        if prev_sub_x is not None:
                            ax.plot([prev_sub_x, x_in], [prev_sub_y, y_in], color='black', linestyle=':', linewidth=1, zorder=2)
                        # Full Mode Inter-bin: Connect 1st event to last event of prev bin
                        elif last_bin_coords is not None:
                            ax.plot([last_bin_coords[0], x_in], [last_bin_coords[1], y_in], color='black', linestyle=':', linewidth=1, zorder=1)

                    prev_sub_x, prev_sub_y = x_out, y_out

                # End of Bin Loop
                last_bin_coords = (prev_sub_x, prev_sub_y)

            # --- Connections (Post-Plotting for Split Mode) ---
            if self.mode == 'split':
                for (depth, idx), curr in coord_map.items():
                    if depth > 0:
                        # Parent Key Logic: Depth-1, Index//2
                        parent_key = (depth - 1, idx // 2)
                        if parent_key in coord_map:
                            prev = coord_map[parent_key]
                            # Connect Parent OUT to Child IN
                            ax.plot([prev['out'][0], curr['in'][0]], 
                                    [prev['out'][1], curr['in'][1]], 
                                    color='black', linestyle=':', linewidth=1, zorder=2)

            # --- 4. Styling & Grid ---
            ax.set_title(title)
            ax.set_ylabel(y_label)
            if self.mode == 'split': ax.set_xlabel('Recursion Depth')
            else: ax.set_xlabel('Iteration Number')

            x_min, x_max = ax.get_xlim()
            x_integers = np.arange(np.floor(x_min), np.ceil(x_max) + 1, 1)
            ax.xaxis.set_major_locator(FixedLocator(x_integers))
            ax.xaxis.set_minor_locator(FixedLocator(x_integers + 0.5))
            
            ax.grid(visible=True, which='minor', axis='x', linestyle='-', alpha=0.5)
            ax.grid(visible=True, which='major', axis='y', linestyle='-', alpha=0.3)
            ax.grid(visible=False, which='major', axis='x')
            
            # --- 5. Legend ---
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label='Input', markerfacecolor='w', markeredgecolor='black', markersize=6),
                Line2D([0], [0], marker='o', color='w', label='Inferred', markerfacecolor='black', markeredgecolor='black', markersize=6),
            ]
            if self.mode != 'split':
                legend_elements.append(
                    Line2D([0], [0], marker='D', color='w', label='Nested fix', markerfacecolor='grey', markeredgecolor='grey', markersize=4)
                )
            
            if m_key == 'score': 
                ax.legend(handles=legend_elements, loc='best', fontsize='small')

        plt.tight_layout()
        output_file = self.ctx.root_dir / 'metrics_plot.png'
        plt.savefig(output_file, dpi=600)
        self.logger.log(f'Plot saved to {output_file}', 'i')
        plt.close()

    def _debug_tree(self, title: str, ete_tree: Tree, key='d', other_attr=[]) -> None:
        self.logger.log(f"{title}", key)
        self.logger.log(ete_tree.get_ascii(show_internal=True, attributes=['name'] + other_attr), key)
