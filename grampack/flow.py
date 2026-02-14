from asyncio import events
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

class HCounterState:
    """Tracks hybridization events to detect nested patterns."""
    def __init__(self, history: Optional[HistoryType] = None):
        self.base_to_full = defaultdict(set)
        self.sets_by_key = []
        self.h_counter = {}
        if history:
            self._integrate_history(history)
            self._rebuild_h_counter()

    def _integrate_history(self, history: HistoryType):
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

    def update(self, new_history_entry: HistoryType):
        # new_history_entry is a single dict {run_id: data}
        self._integrate_history(new_history_entry)
        self._rebuild_h_counter()

    @staticmethod
    def get_base_name(name: str) -> str:
        #return name.split('.', 1)[0]
        # Rather, split by dots, and cut at the first wholy numeric segment (e.g., A.2B.1.2 -> A.2B)
        # TEMP FIX
        parts = name.split('.')
        base_parts = []
        for part in parts:
            if part.isdigit():
                break
            base_parts.append(part)
        return '.'.join(base_parts)

    @staticmethod
    def is_name_nested(small: str, big: str) -> bool:
        return big.startswith(small + '.')

    @staticmethod
    def clean_nested(names: List[str]) -> List[str]:
        """Keep only the deepest versions of names (no prefixes)."""
        return [n for n in names if not any(HCounterState.is_name_nested(n, o) for o in names if o != n)]

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
    def reduce_group_sets(group_list: List[set]) -> List[set]:
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
    def group_by_suffix(name_set: Set[str]) -> List[List[str]]:
        groups = defaultdict(list)
        for name in name_set:
            parts = name.split('.')
            suffix = '.'.join(parts[1:]) if len(parts) > 1 else ''
            groups[suffix].append(name)
        return list(groups.values())

class FlowManager:
    def __init__(self, ctx: GlobalContext, mode: str, logger: GranLogger):
        self.ctx = ctx
        self.mode = mode
        #self.h_counter = HCounterState() if self.ctx.ignore_nesting or self.ctx.start_pt == 0 else HCounterState(self.ctx.history)
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

        sis_nodes = self._get_sis_nodes(nonin_mt.h1_node, nonin_mt.h2_node)

        # Embed an attr H in each sis_node
        for n in sis_nodes:
            if n is None: continue
            if not hasattr(n, 'H'):
                n.add_feature('H', [])
            n.H.append(str((i, j))) # Track which events this node was involved in for nested detection / gluing logic

        self.ctx.history[(i, j)] = {
            'best_mt': best_mt.mt.to_str(internals=True),
            'nonin_mt': nonin_mt.mt.to_str(internals=True),
            'h1_node': nonin_mt.h1_node.get_leaf_names(),
            'h2_node': nonin_mt.h2_node.get_leaf_names(),
            'input_score': input_score,
            'nonin_score': nonin_score,
            'num_gts': len(res.gene_trees),
            'H_locs': [n.name if n else '<auto>' for n in sis_nodes]
        }
        if self.logger.debug:
            # No longer needed to parse iterations, but is very useful for debugging
            track_dict = {n.name: n.H for n in nonin_mt.mt.ete_tree.traverse() if hasattr(n, 'H')}
            self.ctx.history[(i, j)]['trackers'] = track_dict

        passed = self._check_if_passed(i, j)
        self.ctx.history[(i, j)]['passed'] = passed

        if not hold: # For full mode, until nested fixes are done (to not pollute history with partial events)
            with open(self.ctx.history_file, 'w') as f:
                json.dump({str(k): v for k, v in self.ctx.history.items()}, f, indent=4)

        return passed

    # --- Handlers for the Full mode ---

    '''def _find_subset_key(self, lst, counter_dict):
        lst_set = set(lst)
        for k, v in counter_dict.items():
            for l in v:
                if lst_set.issubset(set(l)): return k
        return None

    @staticmethod
    def _get_base_name(name: str) -> str:
        return name.split('.', 1)[0]
        
    def check_and_fix_nested(self, multree: MulTree, genetrees: List[SmrtTree], iter: int, engine_callback: callable) -> None:
        """Original nested fix logic maintained, but uses unified preparer."""

        # Helper to find nodes even if they have been renamed (e.g. x.1 -> x.1.1)
        def _find_node_fuzzy(tree_obj, name):
            # 1. Try exact match
            matches = tree_obj.search_nodes(name=name)
            if matches: return matches[0]
            # 2. Try prefix match (for evolved lineages)
            # This is slower but necessary for nested chains
            for l in tree_obj.iter_leaves():
                if l.name.startswith(name + "."):
                    return l
            return None
            
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

            self.logger.log(f'\nH1: {h1_node.name} {h1_leaves} | H2: {h2_node.name} {h2_leaves}', 'd')
            self.logger.log(f'Current h_counter: {self.h_counter.h_counter}', 'd')

            h1_sis = h1_node.get_sisters()[0]
            h2_sis = h2_node.get_sisters()[0]
            
            found_nested_event = False
            for h_sis_node, h_main_node in [(h1_sis, h2_node), (h2_sis, h1_node)]:
                if not h_sis_node: continue
                
                sis_leaves = h_sis_node.get_leaf_names()
                h_group_key = self._find_subset_key(sis_leaves, self.h_counter.h_counter)

                self.logger.log(f'\nChecking sister node: {h_sis_node.name} with leaves {sis_leaves}', 'd')
                self.logger.log(f'H Group Key found: {h_group_key}', 'd')

                if h_group_key is None:
                    continue

                if any(self._get_base_name(leaf) in h_group_key for leaf in h1_leaves):
                    continue

                for targets in self.h_counter.h_counter[h_group_key]:
                    filter_base = {self._get_base_name(n) for n in sis_leaves}
                    filtered_targets = [t for t in targets if self._get_base_name(t) in filter_base]

                    if not filtered_targets or set(filtered_targets) == set(sis_leaves):
                        continue

                    mt_tree = curr_mt.mt.ete_tree

                    target_lvs = []
                    for t in filtered_targets:
                        found = _find_node_fuzzy(mt_tree, t)
                        if found: target_lvs.append(found)
                    
                    if not target_lvs:
                        continue # Skip if targets are truly lost

                    #target_lvs = [multree.search_nodes(name=t)[0] for t in filtered_targets]
                    target_node = mt_tree.get_common_ancestor(target_lvs) if len(target_lvs) > 1 else target_lvs[0]

                    if any(ft in h1_leaves or ft in h2_leaves for ft in filtered_targets):
                        continue

                    self.logger.log(f"\nOriginal targets: {targets}", 'd')
                    self.logger.log(f"Filtered targets: {filtered_targets}", 'd')
                    self.logger.log(f"Target leaves: {target_lvs}", 'd')

                    # Find the sister of the target nodes in the multree
                    sisters = target_node.get_sisters()
                    if not sisters:
                        continue
                    target_sis = sisters[0]
                    
                    target_sis_lvs = target_sis.get_leaf_names()
                    # Check if base names match: target_sis_leaves and H1_leaves
                    base_tsl = {self._get_base_name(t) for t in target_sis_lvs}
                    base_h1l = {self._get_base_name(h) for h in h1_leaves}
                    if base_h1l == base_tsl:
                        continue

                    self.logger.log(f"\n# --- Nested Hybridization Detected --- #\n", 'i')

                    event_id += 1
                    found_nested_event = True
                    fix_dir = self.ctx.root_dir / f'{iter}.{event_id}' / 'output'

                    res = engine_callback(curr_mt.mt, curr_gts, 
                                        ",".join(h_main_node.get_leaf_names()), 
                                        ",".join(filtered_targets),
                                        fix_dir
                    )

                    curr_mt, curr_gts = self.handle_iteration_result(
                        iter, res, engine_callback, fix_dir, self.logger, j=event_id
                    )

                    # Update hybrid counter for the new nested event
                    # Important: curr_mt should be the new MulTree object, so that h1_node is valid
                    self.h_counter.update({
                        len(self.h_counter.h_counter): {
                            "h1.node": curr_mt.h1_node.get_leaf_names(),
                            "h2.node": curr_mt.h2_node.get_leaf_names(),
                        }
                    })

                    break  # go back to outer loop with new mt/gts, save event

                if found_nested_event: break
            if not found_nested_event: break
        self.logger.log(f"Iteration {iter} found {event_id} event(s) during nested checks.", 'i')
    '''

    def _rename_best_mt(self, res: TaskResult, best_mt_idx: int, suffix: str) -> Tuple[MulTree, Set[str]]:
        """
        Renames the best MulTree's hybrid lineages for the next iteration.
        Format: |{i}.{j~copy_idx} (e.g., Species|1.0, <Internal>|1.1)
        best_mt is modified in place!
        
        Returns: 
            best_mt: The modified MulTree (in place).
            marked_names: Set of original node names for syncing GTs.
        """
        best_mt = res.mul_trees[best_mt_idx]
        mt_wrapper = best_mt.mt
        
        h1_node = best_mt.h1_node # Source
        h2_node = best_mt.h2_node # Copy (Duplicated lineage)

        self._debug_tree(f"Renaming Context: {h1_node.name} (H1) | {h2_node.name} (H2)", mt_wrapper.ete_tree)

        # Build Renaming Map (Traverse to find descendants)
        # We need to map EVERY node in these clades to a suffix.
        # We use a dict to store the decision before applying it, 
        # because applying it changes names and might break traversal if not careful.
        
        marked_nodes = set() # To track which nodes have been marked for renaming
        # Add current
        marked_nodes.add(h2_node)
        # Add descendants
        for desc in h2_node.iter_descendants():
            marked_nodes.add(desc)

        marked_names = set()

        # Apply Renaming
        # We must iterate carefully. The node_to_copy keys are tree nodes.
        # We process them. Note: h1_node and h2_node might share a parent if the tree is small,
        # but their subtrees should be disjoint in a valid MUL-tree.
        
        for node in marked_nodes:
            old_name = node.name
            
            # Store for GTs (Map original name)
            # Gene trees map to the name 'as is' in the recon result.
            marked_names.add(old_name)

            # Handle cases if | is already present - we want to replace it.
            if '|' in old_name: old_name = old_name.split('|')[0]
            
            # Remove any existing '*'
            new_name = old_name.replace('*', '')

            # Generate New Name
            # Leaves: "Species" -> "Species|1.0"
            # Internals: "<1>" -> "<1>|1.0"
            new_name = f"{new_name}|{suffix}"

            # Apply Rename using Wrapper (handles node_map updates)
            mt_wrapper.rename_node(old_name, new_name)
            
        # Rename the <P2> tags for the next iteration logic
        # P nodes of different iterations shouldn't have the same pure name! (hence no '|')
        marked_names.add('<P2>')
        mt_wrapper.rename_node('<P2>', f'<P:{suffix}>')

        '''distance_map = {}
        for n in best_mt.mt.ete_tree.traverse():
            d1 = n.get_distance(h1_node, topology_only=True)
            d2 = n.get_distance(h2_node, topology_only=True)
            distance_map[n.name] = ".1" if d1 < d2 else ".2"'''
        
        self._debug_tree("Renamed MT:", mt_wrapper.ete_tree, other_attr=['H', 'pure'])
        return best_mt, marked_names

    def _partition_gt_leaves(self, res: TaskResult, best_mt_idx: int, marked_names: Set[str], suffix: str) -> List[Tree]:
        """
        Renames Gene Trees to match the Species Tree renaming logic.
        Uses the mapping generated by _rename_best_mt.
        """
        new_gts_list = []
        gts = res.gene_trees
        min_maps = res.kept_mul_maps[best_mt_idx]
        
        debug_sample = self.sample(list(min_maps.keys()))

        for g_idx, map_obj in min_maps.items():
            gt_ete = gts[g_idx].ete_tree.copy() # Work on copy

            if g_idx in debug_sample:
                self._debug_tree(f"Original GT {g_idx}:", gt_ete)

            # Iterate GT leaves
            for l in gt_ete.iter_leaves():
                # Find where this GT leaf mapped in the MT
                # map_obj.cor[gt_leaf_name] returns List[mt_node_names]
                # We take the first mapping (usually only one for optimal recon)
                mt_name_pre_rename = map_obj.cor[l.name][0]
                if mt_name_pre_rename in marked_names:
                    if '|' in l.name: l.name = l.name.split('|')[0]
                    l.name = f"{l.name}|{suffix}"
                    # l.pure stays the same!
                    # No need to modify internal nodes - Reconcile only works on lvs

            new_gts_list.append(gt_ete)

            if g_idx in debug_sample:
                self._debug_tree(f"Renamed GT {g_idx}:", gt_ete, other_attr=['H', 'pure'])
        
        return new_gts_list

    def find_missing_targets(self, multree: MulTree) -> Set[str]:
        if self.ctx.nestedness in {"ignore", "model"}: return set()

        h1_node = multree.h1_node
        matches = multree.mt.match(h1_node.pure)

        self.logger.log(f"Nested Fix: Found {[l.name for l in matches]} matches for H1 node in the MT.", 'd')
        all_h2_locs_names = set()
        already_populated = set()
            
        for h_node in matches:

            # Get sister of the h node
            h_sis = h_node.get_sisters()[0]
            h_sis_name = h_sis.name

            pure_sis, _ = h_sis_name.split('|') if '|' in h_sis_name else (h_sis_name, '')

            if pure_sis != h_sis.pure:
                self.logger.log(f"Nested Fix: Expected pure name '{pure_sis}' to match node's pure attribute '{h_sis.pure}'.", 'e')

            h2_locs = multree.mt.match(pure_sis)
            h2_locs_names = set(n.name for n in h2_locs)

            self.logger.log(f"Nested Fix: For sister '{h_sis.name}', found H2 locations {h2_locs_names} after filtering.", 'd')
            all_h2_locs_names.update(h2_locs_names)
            
            already_populated.add(h_sis_name)

        # Targets must appear once, and exclude already populated ones
        # It is not safe to remove within the loop - the next match might find another copy of the same lineage.
        # We must track separately and remove at the end.
        return all_h2_locs_names - already_populated

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
            locs = event_data.get('H_locs', [])
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

    def autocorrect(self, targets: Set[str], multree: MulTree, genetrees: List[SmrtTree], iter: int,
                   engine_callback: callable) -> None:
        """
        Detects nested hybridization by finding 'orphaned' copies of the H-lineage
        directly in the tree structure using the .match() capability.
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
            if self.ctx.nest_in_only and self._check_internality(curr_mt.mt, h2_loc):
                self.logger.log(f"Nested Fix: Target node '{h2_loc.name}' is root in the smallest event containing it. Skipping due to nest_in_only=True.", 'd')
                continue

            # Trigger Nested Fix
            self.logger.log(f"Nested Fix: Nested Event Detected! Locating missing copy at the branch leading to {h2_loc.name}", 'i')

            h1_leaves = curr_mt.h1_node.get_leaves()
            
            fix_dir = self.ctx.root_dir / f'{iter}.{next_copy_idx}' / 'output'
            
            # Run Task to infer reconciliation for this missing copy
            # We treat the 'Missing Candidate' as the H2 (Target) 
            # and the current H1 as the source.
            gt_dict = {i: SmrtTree(tree_obj=gt) for i, gt in enumerate(curr_gts)}

            step = f"Nested Fix Iteration {iter}.{next_copy_idx}"
            self.logger.report_step(step, "In progress...")

            res, _ = engine_callback(
                curr_mt.mt, gt_dict,
                ",".join([l.name for l in h1_leaves]), # H1 is the reference
                ",".join([l.name for l in h2_loc.get_leaves()]), # H2 is the new found copy
                fix_dir
            )

            self.logger.report_step(step, "Success")

            curr_mt, curr_gts, marked_names = self.handle_iteration_result(
                iter, res, engine_callback, fix_dir, self.logger, j=next_copy_idx
            )

            # Update future targets in the list if they were renamed
            suffix = f"{iter}.{next_copy_idx}"
            for k in range(t_idx + 1, len(pending_targets)):
                future_target = pending_targets[k]
                if future_target in marked_names:
                    clean_name = future_target.split('|')[0] if '|' in future_target else future_target
                    clean_name = clean_name.replace('*', '')
                    new_name = f"{clean_name}|{suffix}"
                    pending_targets[k] = new_name
                    self.logger.log(f"Nested Fix: Updated pending target '{future_target}' to '{new_name}'", 'd')

            next_copy_idx += 1

        self.logger.log(f"Nested check complete. Found {next_copy_idx-1} extra copies.", 'i')

    def handle_iteration_result(
            self, i: int, res: TaskResult,
            engine_callback: callable,
            iter_out: Path,
            iter_logger: GranLogger,
            j: int = 0
        ) -> Tuple[Optional[MulTree], Optional[Dict[int, SmrtTree]], Set[str]]:
        """
        Handles the end of a 'Full' mode iteration.
        Returns: (next_st, next_gts) or None if stopping.
        """
        # Set logger for this iteration - not applicable for the split mode
        self.logger = iter_logger

        suffix = f"{i}.{j}"

        # Rename best non-input MT
        nonin_idx = self._get_nonin_idx(res)
        next_mt, marked_names = self._rename_best_mt(res, nonin_idx, suffix)

        # Update history & check if passed cutoff
        passed = self._update_history(i, j, res)
        if not passed:
            self.logger.log(f"Cutoff reached: no parsimonious events found at Iteration {i}.", 'i')
            return next_mt, None, marked_names

        if j == 0:
            self.logger.log(f"Reticulation found at Iteration {i} with score {res.mt_score()}.", 'i')
        # Rename Trees for Next Iteration
        new_gts_list = self._partition_gt_leaves(res, nonin_idx, marked_names, suffix)

        if j == 0:
            # Check for Nested Hybridization
            # This encapsulates the while-loop for recursive sub-fixes
            targets = self.find_missing_targets(next_mt)
            self.autocorrect(
                targets         = targets,
                multree         = res.mul_trees[nonin_idx],
                genetrees       = new_gts_list,
                iter            = i,
                engine_callback = engine_callback,
            )
            
        # Convert list back to Dict for GrandmaTree consumption
        next_gts = {idx: SmrtTree(tree_obj=gt) for idx, gt in enumerate(new_gts_list)}
        # Refresh the SmrtTree wrapper of MulTree
        next_mt.mt.refresh()
        
        # Write handoff files for resume support
        CommonOps.write_handoff_files(iter_out.parent, next_mt.mt.ete_tree, new_gts_list)
        
        return next_mt, next_gts, marked_names
    
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

        # Bug fix:
        # check if all leaves in best_mt are in h_clade_names
        # if so, after splitting, all remaining leaves in outer will be old internal nodes:
        # the detection was autopolyploidy -> next detection is guaranteed to be the same -> infinite loop
        #mt_lvs_set = set(l.name for l in mt_wrapper.ete_tree.get_leaves())
        #if mt_lvs_set.issubset(set(h_clade_names)):
        #    self.logger.log("All MT leaves belong to duplicated clades.", 'd')

        # Check if H clade only has one species (pure name)
        if len(set(l.pure for l in h1_node.get_leaves())) < 2:
            self.logger.log("All H clade leaves belong to a single species.", 'd')
            self.logger.log(f"Terminal autopolyploidy detected at depth {depth}, index {idx}. Stopping recursive branch.", 'i')
            return []

        # A softer check: MT leaf count == H clade leaf count
        mt_lvs_list = [l.name for l in mt_wrapper.ete_tree.get_leaves()]
        if len(mt_lvs_list) == len(h_clade_names):
            self.logger.log("All MT leaves are H clades leaves.", 'd')
            self.logger.log(f"Terminal autopolyploidy detected at depth {depth}, index {idx}. Stopping recursive branch.", 'i')
            return []

        outer_gts = {}
        inner_gts = {}
        innie_counter = 0

        debug_sample = self.sample(list(gts.keys()))
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
                # No need to copy() - pure nodes' children aren't added to stack, so no unsafe nested detach()s
                extracted_gt = ph_node.detach()#.copy()
                inner_gts[innie_counter] = SmrtTree(tree_obj=extracted_gt)
                
                if g_idx in debug_sample:
                    self._debug_tree(f"Extracted Inner as Lineage {innie_counter}:", extracted_gt)
                
                innie_counter += 1

        # --- Species Tree Surgery ---

        # Simply copy the subtree rooted at H1 for the Inner ST
        # detach() clears the root (t.up == None)
        inner_st_obj = h1_node.copy().detach()

        # For the Outer ST, we need to remove both H1 and H2
        outer_st_obj = mt_wrapper.ete_tree.copy()
        # Re-locate nodes in the COPY
        h1_in_outer = outer_st_obj.search_nodes(name=h1_node.name)[0]
        h2_in_outer = outer_st_obj.search_nodes(name=h2_node.name)[0]
        self.logger.log(f"Removing hybrid clade ({h1_in_outer.name} and {h2_in_outer.name}) from Outer ST.", 'd')
        outer_st_obj, _ = self._trim_outer(outer_st_obj, h1_in_outer, h2_in_outer)

        self._debug_tree("Inner Species Tree (Hybrid Clade):", inner_st_obj)
        self._debug_tree("Outer Species Tree (Backbone):", outer_st_obj)
        self.logger.log(f'len(inner_gts)={len(inner_gts)}, len(outer_gts)={len(outer_gts)}', 'd')

        # --- Queue Tasks with Binary IDs ---
        # Only queue tasks if species tree has enough leaves to be valid
        next_tasks = []
        if len(outer_st_obj.get_leaves()) >= self.ctx.min_st_lvs and len(outer_gts) > 0:
            next_tasks.append((SmrtTree(tree_obj=outer_st_obj), outer_gts, f"{depth + 1}.{idx * 2}"))
        if len(inner_st_obj.get_leaves()) >= self.ctx.min_st_lvs and len(inner_gts) > 0:
            next_tasks.append((SmrtTree(tree_obj=inner_st_obj), inner_gts, f"{depth + 1}.{idx * 2 + 1}"))
        return next_tasks

    @staticmethod
    def _trim_outer(tree_to_trim: Tree, h1_node: TreeNode, h2_node: TreeNode) -> Tuple[Tree, List[str]]:
        """
        Safely removes both hybrid clades from the Outer Species Tree.
        Returns: (trimmed_tree, trimmed_names)
        trimmed_names: List of names of the parents of the removed H nodes.
        trimmed_tree is modified in place! Must be returned for the special root case!
        """
        # Safely remove H nodes from Outer ST: detach leaf, then delete the resulting knuckle node
        trimmed_names = []
        for n in [h1_node, h2_node]:
            n_up = n.up # Save parent before detaching
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
        try:
            next_tasks = self.extract_subproblems(res, depth, idx)
        except Exception as e:
            self.logger.log(f"extracting subproblems at Depth {depth}, Index {idx}: {e}", 'e')

        # Write handoff files for resume support
        for task_st, task_gts, task_id in next_tasks:
            task_out = iter_out.parent / task_id
            task_out.mkdir(parents=True, exist_ok=True)
            self.logger.log(f"Written handoff files for task {task_id} at {task_out.relative_to(iter_out.parent.parent.parent)}, with {len(task_gts)} GTs.", 's')
            CommonOps.write_handoff_files(task_out, task_st.ete_tree, [gt.ete_tree for gt in task_gts.values()])

        self.logger = backup_logger
        return next_tasks

    def glue_split_results(self, output_dir: Path, original_st_path: Path, logger: GranLogger) -> None:
        """
        Recombines results by recursively diving to the innermost subproblems.
        """
        self.logger = logger
        step = "Recombining Split Results (Recursive)"
        self.logger.report_step(step, "In progress...", start=True)

        # Start the chain from the root task (0,0)
        #final_tree = self._recursive_glue("0.0")
        final_tree = self._iterative_glue("0.0")
        # Fallback: original ST
        if final_tree is None:
            self.logger.log("No valid recombination found. Using original species tree.", 'i')
            final_tree = Tree(CommonOps._fix_semicolon(original_st_path.read_text()), format=1)

        # Output the results
        out_st = output_dir / "merged_single_label_form.tre"
        out_mul = output_dir / "merged_multree.tre"
        
        for l in final_tree.get_leaves():
            if l.name.endswith('>'):
                l.name = None

        # Write Suffix form (*) with internal node names
        with open(out_st, 'w') as f:
            f.write(final_tree.write(format=8))

        # Write MUL form (Clean species names)
        mul_tree = final_tree.copy()
        for l in mul_tree.get_leaves():
            # l.name = re.sub(r'(\.[0-9]+)+$', '', l.name)
            l.name = l.name.replace('*', '')
        with open(out_mul, 'w') as f:
            f.write(mul_tree.write(format=9))

        self.logger.log(f"Final Merged Tree: {final_tree.write(format=9)}", 'i')
        
        self.logger.report_step(step, "Success")

    def _iterative_glue(self, root_task_id: str) -> Tuple[Optional[TreeNode], dict]:
        """
        Recombines split results using history 'trackers' to identify graft locations.
        (Iterative Stack-Based Implementation)
        """
        # Stack stores tuples: (task_id, children_visited_flag)
        stack = [(root_task_id, False)]
        results = {}

        while stack:
            task_id, visited = stack.pop()
            
            # Convert "0.0" task_id to the (depth, idx) tuple used in history keys
            depth, idx = (int(x) for x in task_id.split('.'))
            key = (depth, idx)
            
            # Base Case: If this task was never run or didn't pass, 
            # we return None or the input tree.
            if key not in self.ctx.history:
                self.logger.log(f"Glue {task_id}: Task {task_id} not found in history.", 'd')
                results[task_id] = None
                continue

            event = self.ctx.history[key]

            # Check Pass/Fail
            if not event['passed']:
                # Event rejected. If we have an outer_tree (backbone), return it.
                # Otherwise return the Input/Best MT of this node.
                results[task_id] = None
                continue

            # --- STEP 1: Dive to children (Post-order traversal) ---
            # Outer child: (depth+1, idx*2)
            # Inner child: (depth+1, idx*2 + 1)
            outer_id = f"{depth + 1}.{idx * 2}"
            inner_id = f"{depth + 1}.{idx * 2 + 1}"

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

            # "Base State" for this node in the recursion tree.
            current_tree = Tree(event['best_mt'], format=1)
            # Convert to SmrtTree for indexing
            current_tree_wrapper = SmrtTree(tree_obj=current_tree)

            # Get correct H tag
            locs = event['H_locs']
            h_parent = current_tree_wrapper.get_node(locs[0])
            G_tag = f"Graft_{task_id}" # Just in case, shouldn't be needed!
            H_tag = h_parent.up.name if h_parent else f'<auto:{task_id}>'

            # Rename P2 nodes
            current_tree_wrapper.rename_node('<P2>', f"{H_tag}")
            current_tree_wrapper.rename_node('<P2*>', f"{H_tag}", tagged=True)
            # If <P2*> or <P2> in locs, update locs to match the renamed tags
            for i, l in enumerate(locs):
                if l == '<P2>':
                    locs[i] = H_tag
                elif l == '<P2*>':
                    locs[i] = H_tag[:-1] + '*>' # Preserve the '*' in the tag

            self.logger.log(f"Glue {task_id}: Searching for graft locs: {locs}. Found parent {H_tag}.", 'd')

            # Check Redundant Autopolyploidy
            # pass

            if not outer_tree and not inner_tree:
                self.logger.log(f"Glue {task_id}: No Outer and Inner results for task. Returning Current tree.", 'd')
                results[task_id] = current_tree
                continue

            if not inner_tree:
                # Infer inner tree from current tree (best_mt) if missing, since we know the hybrid clade is there
                sister_node = current_tree_wrapper.get_node(locs[0])
                if sister_node is None:
                    inner_tree = current_tree # <root> case, no sisters
                else:
                    inner_tree = sister_node.get_sisters()[0]
                self.logger.log(f"Glue {task_id}: No Inner results for task. Retrieved from Current tree using H loc: {locs[0]}.", 'd')
            self._debug_tree(f"Inner Result Tree for Task {task_id}:", inner_tree, other_attr=['H', 'pure'])

            trimmed_names = []
            if not outer_tree:
                nodes_to_detach = []
                for loc in locs:
                    node = current_tree_wrapper.get_node(loc)
                    sister = node.get_sisters()[0]
                    nodes_to_detach.append(sister)
                outer_tree = current_tree # Modified in place, but no need to copy - not used later
                outer_tree, trimmed_names = self._trim_outer(outer_tree, nodes_to_detach[0], nodes_to_detach[1])
                self.logger.log(f"Glue {task_id}: No Outer results for task. Retrieved from Current tree by removing {trimmed_names} nodes & {[n.name for n in nodes_to_detach]} H locs.", 'd')
            self._debug_tree(f"Outer Result Tree for Task {task_id}:", outer_tree, other_attr=['H', 'pure'])

            outer_tree_wrapper = SmrtTree(tree_obj=outer_tree) # Index
            # inner_tree_wrapper = SmrtTree(tree_obj=inner_tree) # Not needed since we index at the end

            # Expand graft locations, but first, purify
            pure_locs = [l.replace('*', '') for l in locs]
            expanded_locs = {}
            missing_pl = False
            for pl in pure_locs:
                if pl != H_tag and pl != H_tag[:-1] + '*>':
                    matches = outer_tree_wrapper.match(pl)
                    expanded_locs[pl] = matches
                else:
                    missing_pl = True

            if not expanded_locs:
                self.logger.log(f"Glue {task_id}: Graft locations {pure_locs} not found in Outer tree topology.", 'e')
            self.logger.log(f"Glue {task_id}: Expanded graft locations found: {expanded_locs}", 'd')

            # Merge sister locations to avoid redundant grafts in autopolyploidy cases
            for pl, all_nodes in expanded_locs.items():
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
                            if pl not in children[0].name or pl not in children[1].name:
                                final_nodes.extend(nodes)
                            else:
                                final_nodes.append(parent)
                                flag = True
                    if flag:
                        self.logger.log(f"Glue {task_id}: Replacing sister targets with parent {parent.name}.", 'd')
                        expanded_locs[pl] = final_nodes
                    else:
                        self.logger.log(f"Glue {task_id}: No sisters merged for loc.", 'd')

            # Perform grafting of a COPY of inner_tree to each target
            flag = False
            for pl, targets in expanded_locs.items():
                self.logger.log(f"Glue {task_id}: Grafting Inner tree to Outer tree at targets for {pl}: {[t.name for t in targets]}.", 'd')
                for target in targets:
                    tag = '*' if flag else ''

                    # Copy graft, and tag it if not the first pure loc category
                    graft = SmrtTree.copy_lineage(inner_tree, tag)

                    # If root node of the payload has no name, name it
                    if not graft.name:
                        graft.name = f"{G_tag}{tag}"

                    # Pop first trimmed name into new_name
                    new_name = trimmed_names.pop(0) if trimmed_names else f"{H_tag[:-1]}{tag}>"
                    # Use SmrtTree's grafting logic to add as sister to the target node
                    outer_tree = SmrtTree.graft_subtree(outer_tree, target, graft, name=new_name)

                    flag = True

                if missing_pl:
                    self.logger.log(f"Glue {task_id}: First pure loc {pl} matched old <P2> node, at {[target.name for target in targets]}. Subsequent pure locs will be tagged with '*'.", 'd')
                    # Graft again in the same target
                    for target in targets:
                        graft = SmrtTree.copy_lineage(inner_tree, '*')
                        if not graft.name:
                            graft.name = f"{G_tag}*"
                        new_name = f"{H_tag[:-1]}*>"
                        outer_tree = SmrtTree.graft_subtree(outer_tree, target, graft, name=new_name)

            SmrtTree(tree_obj=outer_tree)
            self._debug_tree(f"Post-Graft Tree for Task {task_id}:", outer_tree, other_attr=['H', 'pure'])

            results[task_id] = outer_tree

        return results.get(root_task_id)

    def _recursive_glue(self, task_id: str) -> Tuple[Optional[TreeNode], dict]:
        """
        Recombines split results using history 'trackers' to identify graft locations.
        """
        # Convert "0.0" task_id to the (depth, idx) tuple used in history keys
        depth, idx = (int(x) for x in task_id.split('.'))
        key = (depth, idx)
        
        # Base Case: If this task was never run or didn't pass, 
        # we return None or the input tree.
        if key not in self.ctx.history:
            self.logger.log(f"Glue {task_id}: Task {task_id} not found in history.", 'd')
            return None

        event = self.ctx.history[key]
        self.logger.log(f"--- Processing Task {task_id} ---", 'd')

        # --- STEP 1: Dive to children (Post-order traversal) ---
        # Outer child: (depth+1, idx*2)
        # Inner child: (depth+1, idx*2 + 1)
        outer_id = f"{depth + 1}.{idx * 2}"
        inner_id = f"{depth + 1}.{idx * 2 + 1}"

        # Check Pass/Fail
        if not event['passed']:
            # Event rejected. If we have an outer_tree (backbone), return it.
            # Otherwise return the Input/Best MT of this node.
            return None
        
        # "Base State" for this node in the recursion tree.
        current_tree = Tree(event['best_mt'], format=1)
        # Convert to SmrtTree for indexing
        current_tree_wrapper = SmrtTree(tree_obj=current_tree)

        # Get correct H tag
        locs = event['H_locs']
        h_parent = current_tree_wrapper.get_node(locs[0])
        G_tag = f"Graft_{task_id}" # Just in case, shouldn't be needed!
        H_tag = h_parent.up.name if h_parent else f'<auto:{task_id}>'

        # Rename P2 nodes
        current_tree_wrapper.rename_node('<P2>', f"{H_tag}")
        current_tree_wrapper.rename_node('<P2*>', f"{H_tag}", tagged=True)
        # If <P2*> or <P2> in locs, update locs to match the renamed tags
        for i, l in enumerate(locs):
            if l == '<P2>':
                locs[i] = H_tag
            elif l == '<P2*>':
                locs[i] = H_tag[:-1] + '*>' # Preserve the '*' in the tag

        self.logger.log(f"Glue {task_id}: Searching for graft locs: {locs}. Found parent {H_tag}.", 'd')

        # Check Redundant Autopolyploidy
        # If this event's tree is fully contained in the PARENT event's tree, it's redundant.
        """if depth > 0:
            h1_names = event['h1_node']
            h2_names = event['h2_node']
            # Skip autopolyploidy events, where H1 + H2 == all leaves
            all_leaves = set(current_tree.get_leaf_names())
            if set(h1_names).union(set(h2_names)) == all_leaves:
                parent_key = (depth - 1, idx // 2)
                # this check is not enough, because sometimes we DO want to keep autopolyploidy events (if not excessive)
                # we can check if the new mt is a subtree of its parental (by key) best_mt
                if parent_key in self.ctx.history:
                    parent_best_mt = Tree(self.ctx.history[parent_key]['best_mt'], format=1)
                    self._debug_tree(f"Checking autopolyploidy at {key} against parent {parent_key}", parent_best_mt)
                    self._debug_tree(f"Non-input MT at {key}:", current_tree)
                    # by lca
                    try:
                        equivelent_subtree = parent_best_mt.get_common_ancestor([parent_best_mt.search_nodes(name=lv)[0] for lv in all_leaves])
                        if all_leaves == set(equivelent_subtree.get_leaf_names()):
                            self.logger.log(f"Skipping terminal autopolyploidy event at {key}.", 'd')
                            return None
                    except Exception as e:
                        pass
                        # Not equivelent, keep the event"""

        outer_tree = self._recursive_glue(outer_id)
        inner_tree = self._recursive_glue(inner_id)

        self.logger.log(f"Glue {task_id}: Subproblems returned.", 'd')

        if not outer_tree and not inner_tree:
            self.logger.log(f"Glue {task_id}: No Outer and Inner results for task. Returning Current tree.", 'd')
            return current_tree

        if not inner_tree:
            # Infer inner tree from current tree (best_mt) if missing, since we know the hybrid clade is there
            sister_node = current_tree_wrapper.get_node(locs[0])
            if sister_node is None:
                inner_tree = current_tree # <root> case, no sisters
            else:
                inner_tree = sister_node.get_sisters()[0]
            self.logger.log(f"Glue {task_id}: No Inner results for task. Retrieved from Current tree using H loc: {locs[0]}.", 'd')
        self._debug_tree(f"Inner Result Tree for Task {task_id}:", inner_tree, other_attr=['H', 'pure'])

        trimmed_names = []
        if not outer_tree:
            nodes_to_detach = []
            for loc in locs:
                node = current_tree_wrapper.get_node(loc)
                sister = node.get_sisters()[0]
                nodes_to_detach.append(sister)
            outer_tree = current_tree # Modified in place, but no need to copy - not used later
            outer_tree, trimmed_names = self._trim_outer(outer_tree, nodes_to_detach[0], nodes_to_detach[1])
            self.logger.log(f"Glue {task_id}: No Outer results for task. Retrieved from Current tree by removing {trimmed_names} nodes & {[n.name for n in nodes_to_detach]} H locs.", 'd')
        self._debug_tree(f"Outer Result Tree for Task {task_id}:", outer_tree, other_attr=['H', 'pure'])

        outer_tree_wrapper = SmrtTree(tree_obj=outer_tree) # Index
        # inner_tree_wrapper = SmrtTree(tree_obj=inner_tree) # Not needed since we index at the end

        # Expand graft locations, but first, purify
        pure_locs = [l.replace('*', '') for l in locs]
        expanded_locs = {}
        missing_pl = False
        for pl in pure_locs:
            if pl != H_tag and pl != H_tag[:-1] + '*>':
                matches = outer_tree_wrapper.match(pl)
                expanded_locs[pl] = matches
            else:
                missing_pl = True

        if not expanded_locs:
            self.logger.log(f"Glue {task_id}: Graft locations {pure_locs} not found in Outer tree topology.", 'e')
        self.logger.log(f"Glue {task_id}: Expanded graft locations found: {expanded_locs}", 'd')

        # Merge sister locations to avoid redundant grafts in autopolyploidy cases
        for pl, all_nodes in expanded_locs.items():
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
                        if pl not in children[0].name or pl not in children[1].name:
                            final_nodes.extend(nodes)
                        else:
                            final_nodes.append(parent)
                            flag = True
                if flag:
                    self.logger.log(f"Glue {task_id}: Replacing sister targets with parent {parent.name}.", 'd')
                    expanded_locs[pl] = final_nodes
                else:
                    self.logger.log(f"Glue {task_id}: No sisters merged for loc.", 'd')

        # Perform grafting of a COPY of inner_tree to each target
        flag = False
        for pl, targets in expanded_locs.items():
            self.logger.log(f"Glue {task_id}: Grafting Inner tree to Outer tree at targets for {pl}: {[t.name for t in targets]}.", 'd')
            for target in targets:
                tag = '*' if flag else ''

                # Copy graft, and tag it if not the first pure loc category
                graft = SmrtTree.copy_lineage(inner_tree, tag)

                # If root node of the payload has no name, name it
                if not graft.name:
                    graft.name = f"{G_tag}{tag}"

                # Pop first trimmed name into new_name
                new_name = trimmed_names.pop(0) if trimmed_names else f"{H_tag[:-1]}{tag}>"
                # Use SmrtTree's grafting logic to add as sister to the target node
                outer_tree = SmrtTree.graft_subtree(outer_tree, target, graft, name=new_name)

                flag = True

            if missing_pl:
                self.logger.log(f"Glue {task_id}: First pure loc {pl} matched old <P2> node, at {[target.name for target in targets]}. Subsequent pure locs will be tagged with '*'.", 'd')
                # Graft again in the same target
                for target in targets:
                    graft = SmrtTree.copy_lineage(inner_tree, '*')
                    if not graft.name:
                        graft.name = f"{G_tag}*"
                    new_name = f"{H_tag[:-1]}*>"
                    outer_tree = SmrtTree.graft_subtree(outer_tree, target, graft, name=new_name)

        SmrtTree(tree_obj=outer_tree)
        self._debug_tree(f"Post-Graft Tree for Task {task_id}:", outer_tree, other_attr=['H', 'pure'])

        return outer_tree

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
            h2_len = len(val.get('h2_node', []))
            in_taxa = max(0, out_taxa - h2_len)
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
