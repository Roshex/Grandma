import json
import random
from typing import Tuple, List, Optional, Dict, Set, Callable, Any
from collections import defaultdict
from pathlib import Path
from functools import partial
from dataclasses import dataclass, field

from .config import GlobalContext
from .models import Tree, TreeNode, SmrtTree, TaskResult, MulTree, Map, HistoryType, ConcurrTask, GraftRecord
from .ops import CommonOps
from .logger import GranLogger

class FlowManager:
    def __init__(self, ctx: GlobalContext, mode: str, logger: GranLogger):
        self.ctx = ctx
        self.mode = mode
        self.sample = self.set_sampling_func(ctx.sample)
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

    def _check_if_passed(self, i: int, j: int, curr_event: dict) -> bool:
        """Returns True if the event should be accepted."""
        cut_type, cut_val = self.ctx.cutoff

        # --- Determine which score to compare against ---

        # In split mode / the first iteration of full mode: compare to the input score of the current event
        if self.mode == 'split' or i == 0:
            comp_score = curr_event['input_score']
        # The rest are full mode iterations after the first one
        elif j > 0:
            # Nested fix always passes
            return True
        else:
            # Look-back: compare to the non-input score of the previous event
            # (i.e., the score of the tree we are modifying, before re-evaluation in the current iteration)
            prev_event = self._get_prev_event(i, j)
            comp_score = prev_event['nonin_score']
            
        # --- Apply type of cutoff ---
        if cut_type == 'rel': cut_val *= comp_score
            
        return cut_val < (comp_score - curr_event['nonin_score'])

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

    def judge_event(self, i: int, j: int, res: TaskResult, transform: Optional[Callable] = None) -> Tuple[bool, int, MulTree, Optional[Any]]:
        """
        Judges whether the current event passes the parsimony cutoff and prepares data for the next iteration.
        Logs the input and best, or, if input is best, input and second-best MulTree data for history tracking.
        """
        step = "Assessing event parsimony"
        self.logger.report_step(step, "In progress...")

        best_idx = res.mt_idx()
        best_mt = res.mul_trees[best_idx]
        best_mt_str = best_mt.mt.to_str(internals=True) # Before renaming if == nonin_mt

        nonin_rank = self._get_nonin_rank(res)
        nonin_idx = res.mt_idx(nonin_rank)
        nonin_score = res.mt_score(nonin_rank)
        nonin_mt = res.mul_trees[nonin_idx]
        h_nodes = [nonin_mt.h1_node] + nonin_mt.hx_nodes

        input_score = res.input_score

        if self.ctx.debug:
            hx_str = [f'{n.name} (H{x+2})' for x, n in enumerate(h_nodes[1:])]
            kind = "Splitting" if self.mode == "split" else "Renaming"
            self._debug_tree(f"{kind} Context: {h_nodes[0].name} (H1) | {' | '.join(hx_str)}", nonin_mt.mt.ete_tree, other_attr=['pure'])

        self.logger.log(f"Best non-input index: {nonin_idx}; Best index: {best_idx}", 'd')
        self.logger.log(f"Input tree score: {input_score}; Best non-input tree score: {nonin_score}", 'd')

        passed = self._check_if_passed(i, j, {'input_score': input_score, 'nonin_score': nonin_score})

        if passed:
            if self.mode == "full" and j > 0:
                self.logger.report_step(step, f"Skip...: deferred by nested assessment")
            else:
                self.logger.report_step(step, f"Success: event accepted w/ score {nonin_score}")
        else:
            self.logger.report_step(step, f"Failed.: parsimony cutoff not met")

        # --- Prepare data for history logging regardless of pass/fail ---

        nonin_mt = res.mul_trees[nonin_idx]

        # Apply operations which must take place BEFORE saving history!
        transform_result = transform(nonin_mt, i, j) if transform else None

        step = "Logging event data to history"
        self.logger.report_step(step, "In progress...")

        sister_nodes = [nonin_mt.mt.get_sis(n) for n in h_nodes]
        # Save event data to history regardless of pass/fail
        self.ctx.history[(i, j)] = {
            'best_mt': best_mt_str,
            'nonin_mt': nonin_mt.mt.to_str(internals=True),
            'h_name': nonin_mt.h1_node.name,
            'h_locs': [n.name if n is not None else 'None' for n in sister_nodes], # H nodes or sisters cannot be the root node! But may be None in source STs
            'h_leaves': nonin_mt.h1_node.get_leaf_names(),
            'num_gts': len(res.gene_trees),
            'input_score': input_score,
            'nonin_score': nonin_score,
            'passed': passed,
        }
        self.update_history()

        self.logger.report_step(step, "Success")

        return passed, nonin_idx, nonin_mt, transform_result
    
    def update_history(self):
        #sis_nodes = self._get_sis_nodes(nonin_mt.h1_node, nonin_mt.h2_node)

        # Embed an attr H in each sis_node
        #for n in sis_nodes:
        #    if n is None: continue
        #    if not hasattr(n, 'H'):
        #        n.add_feature('H', [])
        #    n.H.append(str((i, j))) # Track which events this node was involved in for nested detection / gluing logic

        #if self.logger.debug:
        #    # No longer needed to parse iterations, but is very useful for debugging
        #    track_dict = {n.name: n.H for n in nonin_mt.mt.ete_tree.traverse() if hasattr(n, 'H')}
        #    self.ctx.history[(i, j)]['trackers'] = track_dict

        with open(self.ctx.history_file, 'w') as f:
            json.dump({str(k): v for k, v in self.ctx.history.items()}, f, indent=4)

    # --- Handlers for the Full mode ---

    def _relabel_species_tree(self, best_mt: MulTree, i: int, j: int) -> Dict[str, Set[str]]:
        """
        Renames the best MulTree's hybrid lineages for the next iteration.
        Format: |{i}.{j~copy_idx} (e.g., Species|1.0, <Internal|1.1>)
        best_mt is modified in place!
        
        Returns: 
            suffix_name_map: A mapping from suffix to the set of original names that were suffixed.
        """
        step = "Relabeling top non-input species tree"
        self.logger.report_step(step, "In progress...")

        mt_wrapper = best_mt.mt

        suffix_name_map = best_mt.rename_marked_nodes(i, j)
        
        self._debug_tree("Renamed MT:", mt_wrapper.ete_tree, other_attr=['H', 'pure'])

        self.logger.report_step(step, "Success")
        return suffix_name_map # suffix -> set of original names (for syncing GTs)

    def _relabel_gene_trees(self, res: TaskResult, best_mt_idx: int, suffix_name_map: Dict[str, Set[str]]) -> Dict[int, SmrtTree]:
        """
        Renames Gene Trees to match the Species Tree renaming logic.
        Uses the mapping generated by _rename_best_mt.
        Returns gt dict, but - GTs are modified in place!
        """
        step = "Relabeling gene trees"
        self.logger.report_step(step, "In progress...")

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
        
        self.logger.report_step(step, "Success")
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
                    containing_events.append((h_node, len(h_node)))
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

    def relabel_problem(
            self, i: int, res: TaskResult,
            iter_out: Path,
            iter_logger: GranLogger,
            j: int = 0,
            targets: Optional[List[str]] = None
        ) -> Tuple[Optional[MulTree], Optional[Dict[int, SmrtTree]], Dict[str, Set[str]]]:
        """
        Handles the end of a 'Full' mode iteration using tip renaming for both the best MT and GTs.
        Returns: (next_st, next_gts) or None if stopping.
        """
        self.logger = iter_logger

        # Update history & check if passed cutoff
        # Rename best non-input MT in-place while evaluating the event
        passed, nonin_idx, next_mt, suffix_name_map = self.judge_event(i, j, res, transform=self._relabel_species_tree)

        if not passed:
            # Returns the input tree, i.e. index 0 in the mt dict.
            return res.mul_trees[0], None, None

        # Rename Trees for Next Iteration
        next_gts = self._relabel_gene_trees(res, nonin_idx, suffix_name_map)

        if self.ctx.nesting in {"rectify", "strict_rectify"}:
            step = "Checking for nested events"
            self.logger.report_step(step, "In progress...")
            if j == 0:
                # Check for Nested Hybridization
                # This encapsulates the while-loop for recursive sub-fixes
                targets = self.find_missing_targets(next_mt)
                # Sort targets to ensure deterministic order (important for testing/debugging)
                # We could sort by clade size to avoid needing to update future targets as we rename,
                # But this is less intuitive and not always correct.
                targets = sorted(list(targets))
            else:
                # Update targets
                # Update future targets in the list if they were renamed
                suffix = f"{iter}.{j}"
                for k in range(j, len(targets)):
                    future_target = targets[k]
                    # In the rectify case, there's only one Hx node
                    if future_target in suffix_name_map[suffix]:

                        if future_target.startswith("<P*"):
                            new_name = f"{future_target}{iter}|{suffix}"
                        else:
                            new_name = f"{future_target}|{suffix}"

                        targets[k] = new_name
                        self.flow_logger.log(f"Nested Fix: Updated pending target '{future_target}' to '{new_name}'", 'd')
            if targets:
                if j == 0:
                    self.logger.report_step(step, f"Success: found {len(targets)} targets to fix")
                else:
                    self.logger.report_step(step, f"Success: updated targets, {len(targets)-j} left")
            else:
                self.logger.report_step(step, f"Success: no targets detected")
        else:
            targets = []

        step = "Writing handoff files"
        self.logger.report_step(step, "In progress...")

        # Write handoff files for resume support
        CommonOps.write_handoff_files(iter_out.parent, next_mt.mt.ete_tree, [gt.ete_tree for gt in next_gts.values()])

        if self.ctx.nesting in {"rectify", "strict_rectify"} and targets:
            success_msg = f"ready for task {i}.{j+1}"
        else:
            success_msg = f"ready for task {i+1}"

        self.logger.report_step(step, f"Success: {success_msg}")
        
        return next_mt, next_gts, targets
    
    # --- Handlers for the Split mode ---
    
    def _partition_gene_trees(self, res: TaskResult, best_mt_idx: int) -> Tuple[Dict[int, SmrtTree], Dict[int, SmrtTree], Dict[int, List[int]]]:
        """Splits GTs into backbone (Outer) and hybrid clades (Inner)."""
        step = "Partitioning gene trees"
        self.logger.report_step(step, "In progress...")

        gts = res.gene_trees
        min_maps = res.kept_mul_maps[best_mt_idx]
        h_copy_map = res.mul_trees[best_mt_idx].build_h_copy_map() # mt_node -> copy_idx (0 for H1, 1 for H2, etc.)

        outer_gts, inner_gts = {}, {}
        gt_split_dict = defaultdict(list)
        outie_below, innie_below, innie_counter = 0, 0, 0 # Below min leaf count

        debug_sample = self.sample(list(gts.keys()))

        for g_idx, gt_wrapper in gts.items():
            maps = min_maps.get(g_idx)
            if not maps: continue
            source_gt = gt_wrapper.ete_tree

            if g_idx in debug_sample:
                self._debug_tree(f"Pre-split GT {g_idx}:", source_gt)
                self.logger.log(f"Map for GT {g_idx}: {maps.cor}", 'd')

            # --- Map inner/outer leaves by homoeologous copy ---
            gt_leaf_to_copy_idx = {} # Inner leaves: gt_leaf -> copy_idx (0 for H1, 1 for H2, etc.)

            for mt_node, gt_nodes in maps.rev.items():
                if mt_node in h_copy_map:
                    copy_idx = h_copy_map[mt_node]
                    # Flatten the mapping so every GT leaf knows its exact state
                    for gt_n in gt_nodes:
                        gt_leaf_to_copy_idx[gt_n] = copy_idx
            
            # --- Bottom-up purity caching ---
            node_copy_state = {}
            for node in source_gt.traverse("postorder"):
                if node.is_leaf():
                    node_copy_state[node] = gt_leaf_to_copy_idx.get(node.name)
                else:
                    child_states = [node_copy_state[c] for c in node.children]
                    # Internal node is pure ONLY if ALL children map to the EXACT SAME copy
                    if child_states and all(s is not None for s in child_states) and len(set(child_states)) == 1:
                        node_copy_state[node] = child_states[0]
                    else:
                        node_copy_state[node] = None

            # --- Top-down extraction: manual stack allows 'skipping' subtrees ---
            stack = [source_gt]
            h_lineages = []
            while stack:
                node = stack.pop()
                # If copy_idx is an integer (0, 1, 2...), this node is pure for that specific copy
                if node_copy_state.get(node) is not None:
                    if len(node) < self.ctx.min_gt_lvs:
                        innie_below += 1
                    # SUCCESS: We found the largest pure clade for this homoeolog
                    # Nodes are guaranteed to have at least one leaf, and no overlaps
                    # Do NOT add children to the stack; this skips the entire subtree
                    # No need to copy() - pure nodes' children aren't added to stack, so no unsafe nested detach()s
                    h_lineages.append(node.name)
                else:
                    # Node is mixed (e.g., contains H1 and H2, or outer leaves), so we must check its children
                    stack.extend(node.children)

            is_outer, _, extracts = gt_wrapper.trim_lineages(h_lineages, retain=True)

            # --- Outer GTs ---
            if is_outer:
                if len(gt_wrapper) < self.ctx.min_gt_lvs:
                    outie_below += 1
                outer_gts[g_idx] = gt_wrapper
                if g_idx in debug_sample:
                    self._debug_tree(f"Pruned Outer GT {g_idx}:", gt_wrapper.ete_tree)
            else:
                outie_below += 1 # No leaves retained in outer GT since it's empty
                if g_idx in debug_sample:
                    self.logger.log(f"Outer GT {g_idx} is empty after trimming ", 'd')

            # --- Inner GTs ---
            for extracted_gt in extracts:
                inner_gts[innie_counter] = extracted_gt

                if g_idx in debug_sample:
                    self._debug_tree(f"Extracted Inner as Lineage {innie_counter}:", extracted_gt.ete_tree)

                gt_split_dict[g_idx].append(innie_counter)
                innie_counter += 1

        # Apply global cutoff
        # If all GTs of a given subproblem are below the minimum leaf cutoff, we discard them.
        # Previously, we discarded individual GTs based on this, but it biases the reconciliation score because changes to signal balance.
        if innie_below >= len(inner_gts): inner_gts = {}
        if outie_below >= len(outer_gts): outer_gts = {}

        self.logger.report_step(step, f"Success: got {len(outer_gts)} out. & {len(inner_gts)} in. gts")
        return inner_gts, outer_gts, gt_split_dict

    def _partition_species_tree(self, best_mt: MulTree) -> Tuple[SmrtTree, Optional[SmrtTree]]:
        """Safely creates the Inner and Outer Species Trees."""
        step = "Partitioning species tree"
        self.logger.report_step(step, "In progress...")
            
        names_to_trim = [best_mt.h1_node.name] + [n.name for n in best_mt.hx_nodes]
        outer_wrapper, inner_wrapper, _ = best_mt.partition('h1')

        self._debug_tree("Inner Species Tree (Hybrid Clade):", inner_wrapper.ete_tree)

        if outer_wrapper is None:
            self.logger.log(f"Trimming hybrid clades {names_to_trim} from Species Tree resulted in no outer tree.", 'd')
        else:
            self._debug_tree(f"Outer Species Tree (Backbone) after hybrid clades {names_to_trim} trimming:", outer_wrapper.ete_tree)

        self.logger.report_step(step, f"Success: got {len(outer_wrapper) if outer_wrapper else 0} out. & {len(inner_wrapper)} in. st sizes")
        return inner_wrapper, outer_wrapper

    def extract_subproblems(
            self, bin_id: Tuple[int, int], res: TaskResult,
            iter_out: Path,
            iter_logger: GranLogger
        ) -> Optional[List[ConcurrTask]]:
        """
        Processes a split worker result using ETE3-safe surgery and O(N) GT extraction.
        1. Inner: Extracts independent 'pure' subtrees for each hybrid lineage.
        2. Outer: Backbone with H1 clade collapsed to a placeholder leaf.
        Returns: List of new sub-tasks or None.
        """
        backup_logger = self.logger
        self.logger = iter_logger

        # Determine Depth and Index from Binary ID
        depth, idx = bin_id

        passed, nonin_idx, next_mt, _ = self.judge_event(depth, idx, res)

        if not passed:
            return None # Event not taken!

        # --- Extract Subproblems ---

        # Partition the Gene Trees into Inner and Outer sets
        inner_gts, outer_gts, gt_split_dict = self._partition_gene_trees(res, nonin_idx)

        # Perform topological surgery on the Species Tree
        inner_wrapper, outer_wrapper = self._partition_species_tree(next_mt)

        step = "Extracting inferred event subproblems"
        self.logger.report_step(step, "In progress...")

        # Queue Tasks with binary IDs: Outer first
        next_tasks = []
        if outer_wrapper and len(outer_wrapper) >= self.ctx.min_st_lvs and len(outer_gts) > 0:
            next_tasks.append((outer_wrapper, outer_gts, (depth + 1, idx * 2)))
        if len(inner_wrapper) >= self.ctx.min_st_lvs and len(inner_gts) > 0:
            next_tasks.append((inner_wrapper, inner_gts, (depth + 1, idx * 2 + 1)))

        self.logger.report_step(step, f"Success: extracted {len(next_tasks)} valid subproblems")

        step = "Writing handoff files"
        self.logger.report_step(step, "In progress...")

        # Write gt_split_dict to a file
        gt_split_path = iter_out.parent / f"gt_splits.json"
        with open(gt_split_path, 'w') as f:
            json.dump(gt_split_dict, f, indent=4)

        # Write handoff files for resume support
        task_strs = []
        for task_st, task_gts, task_id in next_tasks:
            task_str = f"{task_id[0]}.{task_id[1]}"
            task_strs.append(task_str)
            task_out = iter_out.parent / task_str
            task_out.mkdir(parents=True, exist_ok=True)
            CommonOps.write_handoff_files(task_out, task_st.ete_tree, [gt.ete_tree for gt in task_gts.values()])

        if task_strs:
            self.logger.report_step(step, f"Success: ready for tasks {', '.join(task_strs)}")
        else:
            self.logger.report_step(step, "Success")
        self.logger = backup_logger
        return next_tasks

    def glue_split_results(self, root_id: Tuple[int, int] = (0, 0)) -> SmrtTree:
        """
        Recombines results by recursively diving to the innermost subproblems.
        """
        self.logger.title_banner("Recombining Split Results")
        self.logger.log("Merging subproblem trees...", 'i')

        ft_wrapper = self._iterative_glue(root_id)
        
        self.logger.log("Success: All subproblems merged successfully.", 's')
        return ft_wrapper

    def _iterative_glue(self, root_task_id: Tuple[int, int]) -> SmrtTree:
        """
        Recombines split results using history 'trackers' to identify graft targets.
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

            # --- Dive to children (Post-order traversal) ---
            depth, idx = task_id
            outer_id = (depth + 1, idx * 2)
            inner_id = (depth + 1, idx * 2 + 1)
            uid = 2**depth + idx - 1
            first_uid_of_depth = 2**depth - 1

            if not visited:
                # Push back current node marked as visited
                # Push children. Inner pushed first so Outer is processed first (LIFO order matches original)
                stack.extend([(task_id, True), (inner_id, False), (outer_id, False)])
                continue

            self.logger.log(f"--- Processing Task {task_id} ---", 'd')

            # --- Load Current Tree and Graft Locations for Terminal Subproblem ---

            # Uses the 'best_mt' with format=1
            current_mt = MulTree.from_history_event(event)
            locs = event['h_locs']

            # --- Initialize Graft Records ---
            all_h_nodes = [current_mt.h1_node] + current_mt.hx_nodes
            records: List[GraftRecord] = []
            for i, h_node in enumerate(all_h_nodes):
                up = h_node.up
                up_up = up.up if up else None
                sisters = up.get_sisters() if up else []
                records.append(GraftRecord(
                    copy_id=i,
                    original=locs[i],
                    corrected=locs[i],
                    parent=up.name if up else '<root>',
                    grandp=up_up.name if up_up else '<root>',
                    aunt=sisters[0].name if sisters else '<none>'
                ))
            self.logger.log(f"Glue {task_id} Initial Records: {[str(r) for r in records]}", 'd')

            # --- Retrieve the results of the children or infer them if missing ---

            # Post-traversal evaluation for current node & cleanup memory for children
            outer_wrapper = results.pop(outer_id, None)
            inner_wrapper = results.pop(inner_id, None)

            if not outer_wrapper and not inner_wrapper:
                self.logger.log(f"Glue {task_id}: No Outer and Inner results from {outer_id} and {inner_id}. Returning Current tree.", 'd')
                current_mt.rename_marked_nodes(uid, skip_p_tag=False)
                results[task_id] = current_mt.mt
                continue

            # Partition, get actual trimmed names, and consume (invalidate) the MulTree obj
            outer_wrapper_curr, inner_wrapper_curr, trimmed_names = current_mt.partition('h1')
            self.logger.log(f"Glue {task_id}: Partitioned current tree. Trimmed names: {trimmed_names}", 'd')

            if not inner_wrapper:
                # Infer inner tree from current tree (best_mt) if missing, since we know the hybrid clade is there
                inner_tree = inner_wrapper_curr.ete_tree
                self.logger.log(f"Glue {task_id}: No Inner results from {inner_id}. Retrieved from Current tree by partitioning H1_node.", 'd')
            else:
                inner_tree = inner_wrapper.ete_tree

            self._debug_tree(f"Inner Result Tree for Task {task_id}:", inner_tree, other_attr=['H', 'pure'])

            if not outer_wrapper:
                outer_wrapper = outer_wrapper_curr
                self.logger.log(f"Glue {task_id}: No Outer results from {outer_id}. Retrieved from Current tree by removing all H clades.", 'd')
            else:
                outer_wrapper_curr.destroy()
            self._debug_tree(f"Outer Result Tree for Task {task_id}:", outer_wrapper.ete_tree, other_attr=['H', 'pure'])

            self.logger.log(f"Glue {task_id}: Subproblems retrieved.", 'd')

            # --- Location Expansion and Grafting Logic using Records ---
            records = self._prepare_graft_records(task_id, outer_wrapper, records, first_uid_of_depth)

            # --- Execute Grafts ---
            outer_wrapper.graft_records(inner_tree, records, uid)
            self._debug_tree(f"Post-Graft Tree for Task {task_id}:", outer_wrapper.ete_tree, other_attr=['H', 'pure'])
            results[task_id] = outer_wrapper

        if results.get(root_task_id) is None:
            # Fallback to original ST
            self.logger.log("No valid recombination found. Returning the original ST.", 'i')
            # Not finding root in history SHOULD raise an error!
            return SmrtTree(tree_obj=Tree(self.ctx.history[root_task_id]['best_mt'], format=1))
        return results[root_task_id]

    def _prepare_graft_records(self, task_id: Tuple[int, int], outer_wrapper: SmrtTree, records: List[GraftRecord], first_uid_of_depth: int) -> List[GraftRecord]:
        """
        Targets need to be corrected for some special cases (e.g., source_wgd, target_wgd, source_ils, target_ils), and also expended to previously split events.
        We can't account for WGD in the source ahead of fetching the subtrees, because of modes like 'split+model from mt' or 'mixed+model',
        where the event may contain some WGD and some "appearant" allopolyploidy, due the the full mode logic.
        """
        
        # --- Case: source_wgd ---

        # if p_i == p_j -> loc_i = p_i.sis, p_i = p_i.up (i.e., autopolyploidy - we graft h1 to the sister of the orignal parent,
        # and then, h2 to h1, which is now in the tree as well)
        p_groups = defaultdict(list)
        for rec in records:
            p_groups[rec.parent].append(rec)
            
        for p_name, group in p_groups.items():
            if len(group) > 1:
                # Find the primary locus safely
                self.logger.log(f"Glue {task_id}: Found duplicate parent '{p_name}', indicating source WGD", 'd')
                assert len(group) == 2, f"Expected exactly 2 records for parent '{p_name}' in source WGD case, found {len(group)}"
                # H1 (clean_rec) has loc marked with *
                clean_rec = next((r for r in group if '*' in r.original), None)
                clean_rec.parent = clean_rec.grandp
                clean_rec.original = clean_rec.aunt
                clean_rec.corrected = clean_rec.aunt

        # --- Case: source_ils ---

        # if p_i == loc_j -> loc_j = loc_i (i.e., we graft both p_i and p_j to the same location loc - the original one)
        # Must be done after source_wgd correction, because WGD correction can create new parent matches that indicate ILS
        for rec in records:
            ils_matches = [r for r in records if r.parent == rec.corrected]
            if ils_matches:
                self.logger.log(f"Glue {task_id}: Found parent match for loc '{rec.corrected}', indicating source ILS", 'd')
                rec.corrected = ils_matches[0].original
            # Invalidate grandp and aunt just in case
            rec.grandp = '<invalid>'
            rec.aunt = '<invalid>'

        # --- Case: target_wgd & target_ils Crawling ---

        # E.g., if y is a loc in (y.1, y.2); or ((a, y.1), y.2);
        # In both cases, we want to graft only above all y's, because if more than one y is in the tree, it's either:
        # [a.] an event in parallel [split mode], so it happened **after** the current event (in which case, we crawl up)
        # [b.] an event from the original input or full mode [in mixed] (here, we mustn't crawl, becasue we want to expand them later)
        # Thus, [a.] has to be done before expansion, because then we won't get redundant locations!
        # Because, we only crawl in [a.], we can distinguish it from [b.] by checking the p depth of y.2, hence these 2 main conditions:
        # [1.] loc.up is an old p (depth < first_uid_of_depth) [distinguish]
        # [2.] loc.up's other child is either == loc [case t_wgd] or == one of loc's children [case t_ils], by .pure
        # Then, we crawl up. This needs to be done until conditions are not met!
        # We check against first_uid_of_depth, because depths have the form, e.g.: [full] 0, 1, 2, [switch to split] 3-6, 7-14...
        # and we break the crawl if the depth is below current
        for rec in records:
            while True:
                loc_node = outer_wrapper.get_node(rec.corrected)
                if not loc_node: break # A delayed wgd loc, will be handled by expansion logic...
                
                parent = loc_node.up
                if not parent: break # Nowhere to crawl up to - root reached
                
                parent_name = parent.name
                left_side = parent_name.split('>', 1)[0].split('|', 1)[0]
                try: left_side = left_side.split('<P')[1]
                except IndexError: break # Not a parent tag
                
                if int(left_side) < first_uid_of_depth:
                    break # Reached historical depth, stop crawling
                
                opts = [loc_node.pure] if loc_node.is_leaf() else [c.pure for c in loc_node.children]
                loc_sis = outer_wrapper.get_sis(loc_node).pure
                
                if loc_sis in opts:
                    self.logger.log(f"Glue {task_id}: Target WGD/ILS found for '{rec.corrected}' by parent's sister '{loc_sis}'. Crawling to '{parent_name}'.", 'd')
                    rec.corrected = parent_name
                else:
                    break # Correction complete

        # --- Expand Locations ---
        for rec in records:
            # Important: we look for the node name, not node.pure!
            # If a lookup of node.pure is needed, it means there's some bug in SmrtTree.graft_records() or downstream from it
            loc_node = outer_wrapper.get_node(rec.corrected)
            if loc_node:
                pure_loc = loc_node.pure
                rec.expanded_targets = outer_wrapper.match(pure_loc)
                rec.corrected = pure_loc
            else:
                # Mark as delayed WGD if missing from outer tree
                rec.expanded_targets = []

        self.logger.log(f"Glue {task_id}: Resolved graft locations with corrections: {[str(r) for r in records]}", 'd')

        return records

    def fast_forward_split(self, current_tasks: List[ConcurrTask]) -> List[ConcurrTask]:
        """
        Reconstructs the task queue from disk state based on history.
        Uses BFS to traverse solved nodes and identify the frontier.
        """

        self.logger.log("Fast-forwarding Split tasks based on history...", 'i')
        self.logger.log(f"Current tasks: {current_tasks}, output_dir: {self.ctx.root_dir}", 'd')

        # queue contains Tuple[Current_Task_ID_Tuple, Parent_Task_ID_Tuple]
        # Start with Root ID (0, 0) and no parent
        queue = [(current_tasks[0][2], None)]
        real_tasks = []
        
        while queue:
            q = queue.pop(0)
            nid = q[0]
            depth, idx = nid
            nid_str = f"{depth}.{idx}"
            
            # Check if this task is already solved in history
            if nid in self.ctx.history:
                # Task done. Check for its children directories on disk
                c1 = (depth + 1, idx * 2)
                c2 = (depth + 1, idx * 2 + 1)

                c1_str = f"{c1[0]}.{c1[1]}"
                c2_str = f"{c2[0]}.{c2[1]}"

                self.logger.log(f"Task {nid_str} done. Checking children: {c1_str}, {c2_str} in {self.ctx.root_dir / nid_str}", 'd')
                
                # If child dir exists, add to traversal queue
                if (self.ctx.root_dir / nid_str / c1_str / "multree.tre").exists():
                    queue.append((c1, nid))
                if (self.ctx.root_dir / nid_str / c2_str / "multree.tre").exists():
                    queue.append((c2, nid))
            else:
                # Task NOT in history -> It is a frontier task to run.
                # Load inputs from disk
                p_nid = q[1]
                p_nid_str = f"{p_nid[0]}.{p_nid[1]}" if p_nid else "None"
                self.logger.log(f"Queueing frontier task: {nid_str} (child of {p_nid_str})", 'd')

                st_path = self.ctx.root_dir / p_nid_str / nid_str / "multree.tre"
                gt_path = self.ctx.root_dir / p_nid_str / nid_str / "genetrees.txt"
                
                self.logger.log(f"Looking for ST at {st_path}, GTs at {gt_path}", 'd')

                if st_path.exists() and gt_path.exists():
                    real_tasks.append((st_path, gt_path, nid))
                    self.logger.log(f"Resuming sub-problem: {nid_str}", 's')
                else:
                    # Fallback: if root is missing inputs but passed in current_tasks
                    if nid == str(current_tasks[0][2]):
                        self.logger.log(f"Using provided inputs for root task {nid_str}, {current_tasks}", 's')
                        real_tasks.extend(current_tasks)
                    else:
                        self.logger.log(f"Missing inputs for resume task {nid_str}. Skipping.", 'w')

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
                return len(Tree(t_str, format=1))
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
        #self.logger.log(f'Plot saved to {output_file}', 'i')
        plt.close()

    def _debug_tree(self, title: str, ete_tree: Tree, key='d', other_attr=[]) -> None:
        self.logger.log(f"{title}", key)
        self.logger.log(ete_tree.get_ascii(show_internal=True, attributes=['name'] + other_attr), key)
