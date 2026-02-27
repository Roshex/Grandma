import os
import re
import sys
import bisect
import pickle
import tempfile
import multiprocessing as mp
from tqdm import tqdm
from pathlib import Path
from itertools import chain
from typing import Tuple, List, Optional, Dict, Union

from .config import TaskConfig
from .logger import GranLogger
from .models import Tree, SmrtTree, MulTree, GroupData, NameRegistry

class CommonOps:
    @staticmethod
    def _fix_semicolon(tree_str: str) -> str:
        """Ensures tree strings end with a semicolon."""
        tree_str = tree_str.strip()
        return tree_str if tree_str.endswith(';') else tree_str + ';'

    @staticmethod
    def write_handoff_files(dir: Path, st: Tree=None, gts: Optional[List[Tree]]=None):
        """Writes the trees to disk to allow inspection/resume, matching iter_mode.py."""
        ### Bug : we need to save with all names - internals and root too!
        if st:
            st_path = dir / 'multree.tre'
            with open(st_path, 'w') as f: f.write(st.write(format=0))
        if gts:
            gt_path = dir / 'genetrees.txt'
            with open(gt_path, 'w') as f:
                for gt in gts: f.write(gt.write(format=0) + '\n')

    @staticmethod
    def _identify_path(p: Union[str, Path]) -> Tuple[str, List[Path]]:

        if not isinstance(p, (Path, str)):
            return "invalid", []
        
        if isinstance(p, str) and not os.path.exists(p):
            return "raw", []

        p = Path(p)

        if p.exists():
            return ("file" if p.is_file() else "directory"), [p]

        parent = p.parent if p.parent != Path('.') else Path.cwd()
        matches = list(parent.glob(p.name))

        if matches:
            return "pattern", matches

        return "nonexistent", []
    
    @staticmethod
    def _load_single_content(input: Union[Path, str], desc: str, logger: GranLogger, key: str="e") -> str:
        """Loads a single input string from Path or String."""
        kind, paths = CommonOps._identify_path(input)
        if kind == "raw":
            return input
        elif kind == "file":
            try:
                return paths[0].read_text()
            except Exception as e:
                logger.log(f"reading {desc} file '{paths[0]}': {e}", key)
        elif kind == "nonexistent":
            logger.log(f"{desc} file '{input}' not found.", key)
        else:
            logger.log(f"Invalid input type for {desc}.", key)

    @staticmethod
    def _load_multi_content(input: Union[Path, str], desc: str, logger: GranLogger, key: str="e") -> Tuple[List[str], Optional[List[str]]]:
        """
        Handles File, Raw String, or Folder input.
        Returns List of (content, source_name).
        """
        kind, paths = CommonOps._identify_path(input)
        if kind == "raw":
            return input, None
        elif kind == "file":
            try:
                return paths[0].read_text().splitlines(), None
            except Exception as e:
                logger.log(f"reading {desc} file '{paths[0]}': {e}", key)
        elif kind == "directory" or kind == "pattern":
            if kind == "directory":
                paths = list(paths[0].glob('*'))
            content = []
            for p in paths:
                try:
                    txt = p.read_text().strip()
                    if txt:
                        content.append(txt)
                except Exception as e:
                    logger.log(f"Could not read {p}: {e}", key)
            return content, [f'from {p.name}' for p in paths]
        elif kind == "nonexistent":
            logger.log(f"{desc} file '{input}' not found.", key)
        else:
            logger.log(f"Invalid input type for {desc}.", key)

class TreeLoader:
    """
    Handles loading, verification, and optional repais of Species and Gene trees.
    """
    @staticmethod
    def spec_tree(tcf: TaskConfig, logger: GranLogger) -> SmrtTree:

        if isinstance(tcf.st, SmrtTree):
            # Do nothing, already loaded
            step = "Loading species tree from memory"
            logger.report_step(step, "In progress...")
            logger.report_step(step, "Success: species tree loaded")
            return tcf.st

        step = "Reading species tree"
        logger.report_step(step, "In progress...")

        # Input Validation
        if not tcf.st:
            logger.log("Species tree not found. Please check the input.", 'e')

        # Load Raw Content
        line = CommonOps._load_single_content(tcf.st, "species tree", logger)
        
        # Basic Formatting
        line = TreeLoader._sanitize_line(line)
        line = CommonOps._fix_semicolon(line)

        # MUL-Tree processing
        line = TreeLoader._process_mul_status(tcf, line, logger)

        # Parse
        try:
            # Format 1 allows internal node names, which we might need
            t = Tree(line, format=0)
        except Exception as e:
            logger.log(f"reading species tree file: {e}", 'e')

        # Topology Fixes & Checks (Polytomies, Rooting)
        # We error on ST issues unless repair is requested, as ST must be robust.
        is_valid, msg = TreeLoader._fix_and_validate_topology(t, tcf.repair)
        if not is_valid:
            logger.log(f"Species tree invalid: {msg}", 'e')

        if tcf.repair:
            CommonOps.write_handoff_files(tcf.output_dir, st=t)

        logger.report_step(step, "Success: species tree read")
        return SmrtTree(tree_obj=t)

    @staticmethod
    def gene_trees(tcf: TaskConfig, logger: GranLogger) -> Optional[Dict[int, SmrtTree]]:

        if isinstance(tcf.gts, dict):
            step = "Loading gene trees from memory"
            logger.report_step(step, "In progress...")
            logger.report_step(step, f"Success: {len(tcf.gts)} gene trees loaded")
            return tcf.gts

        step = "Reading gene trees"
        logger.report_step(step, "In progress...")

        # Input Validation
        if tcf.gts is None:
            if tcf.mode == 'build-mts':
                logger.report_step(step, "Skipped: 'build-mts' mode")
                return {}
            else:
                logger.log(f"Gene trees input is missing. Required in all modes except 'build-mts' (here: '{tcf.mode}' mode).", 'e')

        # Load Raw Contents (File, String, or Folder)
        tree_list, origins = CommonOps._load_multi_content(tcf.gts, "gene trees", logger, key="e")
        
        # Process Trees
        valid_gts = []
        #st_taxa = {n.name for n in tcf.st.ete_tree.get_leaves()}
        for i, line in enumerate(tree_list):

            origin = origins[i] if origins else "on line " + str(i+1)

            # Check Empty
            if not line.strip():
                logger.log(f"Empty GT in {origin} -- Filtering.", 'w')
                continue

            # Semicolon
            line = CommonOps._fix_semicolon(line)

            # Parse
            try:
                gt = Tree(line, format=0)
            except Exception:
                logger.log(f"Error reading tree {origin}! -- Filtering.", 'w')
                continue

            # Topology Repair/Check
            is_valid, msg = TreeLoader._fix_and_validate_topology(gt, tcf.repair)
            if not is_valid:
                logger.log(f"Gene tree {origin}: {msg} -- Filtering.", 'w')
                continue

            # Tips Repair/Check } if tcf.repair
            #TreeLoader._repair_tips(gt)

            # Taxa Repair/Check } if tcf.repair
            '''gt_taxa = {n.name for n in gt.ete_tree.get_leaves()}
            if not gt_taxa.issubset(st_taxa):
                # Optionally, prune here if fixing is enabled? 
                # For now, strict filtering based on prompts.
                logger.write(f"Warning: Gene tree {origin} contains taxa not in Species Tree -- Filtering.")
                continue'''

            valid_gts.append(gt)

        if len(valid_gts) == 0:
            logger.log(f"No valid gene trees survived filtering (required in {tcf.mode} mode).", 'e')
        
        if tcf.repair:
            CommonOps.write_handoff_files(tcf.output_dir, gts=valid_gts)
                
        logger.report_step(step, f"Success: {len(valid_gts)} gene trees read")
        return {idx: SmrtTree(tree_obj=gt) for idx, gt in enumerate(valid_gts)}

    # --- Helpers ---

    @staticmethod
    def _fix_and_validate_topology(t: Tree, attempt_fix: bool) -> Tuple[bool, str]:
        """
        Checks for polytomies and unrooted-ness.
        If attempt_fix is True, resolves polytomies and midpoint roots (if needed).
        """
        # Rooting
        # ETE3 logic: unrooted trees often loaded as rooted with trifurcation at top.
        # If we just resolved polytomies, we might have arbitrarily binary-ized the root.
        # A simple check: leaves = internal + 1 is true for any binary tree.
        # We mainly ensure it effectively acts rooted.
        if len(t.children) > 2 or not t.get_tree_root():
            if attempt_fix:
                # Quickest fix for unrooted top-level polytomy
                t.resolve_polytomy() 
            else:
                return False, "Tree root is not rooted"

        # 2. Rooting via (Num Internal = Num Tips - 1)
        # Note: ETE3 handles rooting differently, but strict bifurcating tree property holds:
        # Leaves = N, Internal = N-1.
        leaves = len(t.get_leaves())
        internal = len([n for n in t.traverse() if not n.is_leaf()])
        if internal != (leaves - 1):
            # This usually happens if the root has only 2 children? 
            # Actually for rooted bifurcating: N leaves -> N-1 internal (including root).
            pass

        # Polytomies
        has_poly = False
        for n in t.traverse():
            if len(n.children) > 2:
                has_poly = True
                break
        if has_poly:
            if attempt_fix:
                t.resolve_polytomy(recursive=True)
            else:
                return False, "Tree contains non-bifurcating nodes"

        return True, ""

    @staticmethod
    def _shrink_ret_depths(predef_rets: dict) -> dict:
        """
        Compresses a depth-keyed dictionary of reticulation pairs into continuous 
        iteration levels (0, 1, ...).
        
        Independent (non-overlapping) reticulations are grouped into the same level 
        so they can be evaluated concurrently in the same iteration.
        """
        def get_taxa(pair):
            """Extracts a set of clean leaf names from an (h1_str, h2_str) pair."""
            taxa = set()
            for group in pair:
                for t in group.split(','):
                    t = t.strip()
                    if t:
                        taxa.add(t)
            return taxa

        # 1. Flatten the dict and sort by original depth (ascending)
        # Items look like: (original_depth, (h1_str, h2_str))
        flat_items = []
        for depth, pairs in predef_rets.items():
            for pair in pairs:
                flat_items.append((int(depth), pair))
                
        # Shallower depths (ancestors) must be processed before deeper depths (descendants)
        flat_items.sort(key=lambda x: x[0])
        
        # 2. Determine the new iteration level for each pair
        new_levels = []
        for i, (depth_i, pair_i) in enumerate(flat_items):
            taxa_i = get_taxa(pair_i)
            
            max_parent_level = -1
            # Check all previously processed (shallower) pairs
            for j in range(i):
                depth_j, pair_j = flat_items[j]
                taxa_j = get_taxa(pair_j)
                
                # If they share leaves, the deeper one (pair_i) is nested in the shallower one (pair_j)
                if taxa_i.intersection(taxa_j):
                    max_parent_level = max(max_parent_level, new_levels[j])
                    
            # Assign to the level immediately following its deepest parent
            # If it has no parents (no intersection), it defaults to -1 + 1 = 0.
            new_levels.append(max_parent_level + 1)
            
        # 3. Rebuild the dictionary grouped by the new iteration levels
        shrunk_dict = {}
        for level, (_, pair) in zip(new_levels, flat_items):
            if level not in shrunk_dict:
                shrunk_dict[level] = []
            shrunk_dict[level].append(pair)
            
        return shrunk_dict

    @staticmethod
    def _process_mul_status(tcf: TaskConfig, t_line: str, logger: GranLogger) -> str:

        from Reticulate_Tree.reticulate_tree import ReticulateTree
        from collections import Counter
        
        def singlify_tree(t: Tree) -> None:
            '''Remove duplicate labels.'''
            leaves_to_keep = set()
            for l in t.get_leaves():
                if not l.name or l.name in leaves_to_keep:
                    l.name = None
                else:
                    leaves_to_keep.add(l.name)
            t.prune(leaves_to_keep, preserve_branch_length=True)

        def singlify_enewick(t_line: str) -> Tree:
            t = Tree(t_line, format=1) # ENewick parsing to handle reticulation labels
            # Identify valid backbone leaves (excluding any node with #H in the name)
            backbone_leaves = [n for n in t.get_leaves() if not (n.name and "#H" in n.name)]
            # Prune strips the hybrid leaves and automatically collapses 
            # any resulting unbranched internal nodes, preserving branch lengths.
            t.prune(backbone_leaves, preserve_branch_length=True)
            # Clean up any remaining internal branching nodes that were part of the reticulation
            for n in t.traverse():
                if n.name and "#H" in n.name:
                    # Safely strip just the #H tag in case it had a real name prefix (e.g. 'NodeA#H1' -> 'NodeA')
                    clean_name = n.name.split('#H')[0]
                    n.name = clean_name if clean_name else ''
                    '''# When wrapping a single leaf
                    if len(n.children) == 1:
                        n.delete()'''
            return t

        # --- Autodetect MULTree/eNewick and collapse if needed ---
        rt = ReticulateTree(t_line, is_multree=True)
        t = rt.tree
        G = rt.dag

        # rt.visualize() # For debugging, can be removed later
        
        if tcf.is_mul_input and not rt.get_reticulation_count():
            logger.log("Warning: --multree was given but the input species tree is not a MUL-tree or eNewick.", 'w')
        if not tcf.is_mul_input and rt.get_reticulation_count():
            logger.log("You have not entered a tree type (--multree) of multree, but there are labels in your tree that appear more than once.", 'w')

        # No reticulations found, just return original string
        if not rt.retnodes:
            return t_line
        
        logger.log(f"Tree:\n{t.get_ascii(show_internal=True)}", 'd')
        
        g_depths = rt.compute_depths(G)
        sorted_rets = sorted(rt.retnodes, key=lambda n: g_depths[n])#, reverse=True) # Process deeper reticulations first (descendants before ancestors)

        ret_struct = {}
        for ret_node in sorted_rets:
            preds = list(G.predecessors(ret_node))
            succs = list(G.successors(ret_node))
            logger.log(f"Reticulation node '{ret_node}' has predecessors {len(preds)} and successors {len(succs)}", 'd')
            ret_struct[ret_node] = (G.nodes[succs[0]]['ete'], [G.nodes[p]['ete'] for p in preds])

        '''if len(ret_struct) > 1:
            logger.log("Multiple reticulations detected in the input tree. This is currently not supported as input.", 'e')
        
        first_ret = next(iter(ret_struct.values()))
        if len(first_ret[1]) != 2:
            counts = Counter(n.name for n in t.traverse() if n.name)
            violators = [k for k, v in counts.items() if v > 2]
            logger.log(f"Reticulation node with degree != 2 detected (all violators: {violators}). This is currently not supported as input.", 'e')

        '''

        predefined_rets = {}

        # Process ALL reticulations for Guided Iterative Search
        for ret_node, (succ, parents) in ret_struct.items():
            if len(parents) != 2:
                counts = Counter(n.name for n in t.traverse() if n.name)
                violators = [k for k, v in counts.items() if v > 2]
                logger.log(f"Reticulation node with degree != 2 detected (all violators: {violators}). This is currently not supported as input.", 'e')
                raise NotImplementedError("Reticulations with degree != 2 not supported.")

            n1, n2 = parents

            logger.log(f"Reticulation structure: Succ: {succ.get_leaf_names()} N1: {n1.get_leaf_names()} N2: {n2.get_leaf_names()}", 'd')
            logger.log(f"Succ Tree:\n{succ.get_ascii(show_internal=True)}", 'd')
            logger.log(f"Succ up Tree:\n{succ.up.get_ascii(show_internal=True)}", 'd')
            logger.log(f"N1 Tree:\n{n1.get_ascii(show_internal=True)}", 'd')
            logger.log(f"N2 Tree:\n{n2.get_ascii(show_internal=True)}", 'd')

            n1_children = n1.get_children()
            n2_children = n2.get_children()
            
            succ_leaves = set(succ.get_leaf_names())
            if set(n2_children[0].get_leaf_names()) == succ_leaves:
                n2_sister = n2_children[1]
            else:
                n2_sister = n2_children[0]

            if set(n1_children[0].get_leaf_names()) == succ_leaves:
                n1_sister = n1_children[1]
            else:
                n1_sister = n1_children[0]

            h_str = ",".join(succ.get_leaf_names())
            p1_str = ",".join(n1_sister.get_leaf_names())
            p2_str = ",".join(n2_sister.get_leaf_names())
            
            logger.log(f"Hybrid clade (h): {h_str}", 'd')
            logger.log(f"Hybrid clade parent 1 (p1): {p1_str}", 'd')
            logger.log(f"Hybrid clade parent 2 (p2): {p2_str}", 'd')

            ret_depth = g_depths[ret_node]
            if ret_depth in predefined_rets:
                predefined_rets[ret_depth].append((h_str, p1_str, p2_str))
            else:
                predefined_rets[ret_depth] = [(h_str, p1_str, p2_str)]

        if '#H' in t_line:
            t = singlify_enewick(t_line)
        else:
            singlify_tree(t) # in-place

        logger.log(f"T after collapsing reticulations to singly-labeled:\n{t.get_ascii(show_internal=True)}", 'd')

        logger.log(f"Predefined reticulations for iterative search: {predefined_rets}", 'd')

        predefined_rets = TreeLoader._shrink_ret_depths(predefined_rets)
        for level, pairs in predefined_rets.items():
            fixed_pairs = []
            for h_str, p1_str, p2_str in pairs:
                p1 = t.get_common_ancestor(p1_str.split(',')) if ',' in p1_str else t.search_nodes(name=p1_str)[0]
                p1_sis = p1.get_sisters()[0]
                if set(p1_sis.get_leaf_names()) == set(h_str.split(',')):
                    fixed_pairs.append((h_str, p2_str))
                else:
                    fixed_pairs.append((h_str, p1_str))
            predefined_rets[level] = fixed_pairs

        # Inject the queue into the config: update() won't work here
        object.__setattr__(tcf, 'predefined_rets', predefined_rets)
        logger.log(f"Predefined reticulations for iterative search: {tcf.predefined_rets}", 'd')
        
        # Pass the collapsed tree string forward
        t_line = t.write(format=1)

        return t_line

    @staticmethod
    def _sanitize_line(s: str) -> str:
        # Remove everything between "'[" and "]'" (Astral cleaning)
        return re.sub(r"'\[.*?\]'", '', s)
    
    @staticmethod
    def _repair_tips(t: Tree) -> None:
        leaf_counts = {}
        for node in t.traverse():
            if node.name:
                if node.is_leaf():
                    clean_name = TreeLoader._sanitize_tip(node.name)
                    # Ensure unique names
                    leaf_counts[clean_name] = leaf_counts.get(clean_name, 0) + 1
                    node.name = f'{leaf_counts[clean_name]}_{clean_name}'
                else:
                    node.name = None

    @staticmethod
    def _sanitize_tip(s: str, old: List[str] = ['_', '.'], new: str = '-') -> str:
        # Replace any chars in old with new
        for char in old:
            s = s.replace(char, new)
        return s

class MulTreeManager:
    def __init__(self, config: TaskConfig, st: SmrtTree, logger: GranLogger) -> Optional[Dict[str, int]]:
        self.tcf = config
        self.st = st
        self.logger = logger
        self.ploidies = self._parse_ploidy_file(self.tcf.ploidies, logger)

        # DFS Cache for optimizations (Lazy loaded)
        self.dfs_order = None
        self.subtree_range = None
        self.st_adj = None

    def _linearize_tree(self):
        """
        Performs DFS to map topological relationships to integer ranges.
        O(N) Pre-computation.
        """
        if self.dfs_order: return # Already done

        # 1. Build Adjacency for traversal
        # SmrtTree/Ete3 structure: node.children is list of objects
        root = self.st.ete_tree
        
        self.dfs_order = {}      # obj -> int
        self.subtree_range = {}  # obj -> (min, max)
        
        timer = 0
        # Stack: (node, state) 0=Enter, 1=Exit
        work_stack = [(root, 0)]
        
        while work_stack:
            node, state = work_stack.pop()
            
            if state == 0:
                # Entry
                timer += 1
                self.dfs_order[node] = timer
                # Push exit
                work_stack.append((node, 1))
                # Push children
                for child in node.children:
                    work_stack.append((child, 0))
            else:
                # Exit
                start = self.dfs_order[node]
                self.subtree_range[node] = (start, timer)

    def countMULTrees_Optimized(self, h1_nodes: List[str], h2_nodes: List[str]) -> int:
        """
        O(H1 * log H2) Counting using DFS Ranges.
        """
        # Ensure linearization
        self._linearize_tree()
        
        # Map strings back to objects
        h1_objs = [self.st.get_node(n) for n in h1_nodes if self.st.get_node(n)]
        h2_objs = [self.st.get_node(n) for n in h2_nodes if self.st.get_node(n)]
        
        # Sort H2 DFS indices (O(M log M))
        h2_indices = []
        for h2 in h2_objs:
            if h2 in self.dfs_order:
                h2_indices.append(self.dfs_order[h2])
        h2_indices.sort()
        
        total_h2 = len(h2_indices)
        valid_count = 0
        
        for h1 in h1_objs:
            if h1.is_leaf():
                # Leaf has empty clade (in terms of internal nesting check)
                valid_count += total_h2
                continue
            
            if h1 not in self.subtree_range: continue
            
            L, R = self.subtree_range[h1]

            # [FIX] Strict Inequality (bisect_right of L)
            # Naive iter_descendants() excludes self. 
            # To match naive, we want strictly descendants: (L, R]
            
            # Find first element > L (Excludes self)
            start_idx = bisect.bisect_right(h2_indices, L) 
            # Find first element > R
            end_idx = bisect.bisect_right(h2_indices, R)

            nested_count = end_idx - start_idx
            valid_count += (total_h2 - nested_count)
            
        return valid_count

    def _find_all_targets(self, primary_h2: str) -> List[str]:
        """
        For 'Model' mode: Finds all nodes in the ST that belong to the same lineage as primary_h2.
        Useful when ST is already a MUL-tree (e.g. Iteration > 0).
        """
        # 1. Identify the 'Pure' lineage name
        # If input is 'Species|1.0', pure is 'Species'. 
        # If input is 'Species', pure is 'Species'.
        if '|' in primary_h2:
            pure_name = primary_h2.split('|')[0]
        else:
            pure_name = primary_h2
            
        pure_name = pure_name.replace('*', '')

        # 2. Find all matches in the current ST
        matches = self.st.match(pure_name)
        
        # 3. Return their unique names
        # We sort them to ensure deterministic behavior (Primary H2 usually comes first naturally or via sort)
        target_names = sorted([n.name for n in matches])
        
        # Ensure the requested primary_h2 is in the list (it should be if logic is correct)
        if primary_h2 not in target_names:
            # Fallback for edge cases where .match() might behave differently on leaves vs internals
            target_names.append(primary_h2)
            
        return target_names



    @staticmethod
    def _parse_ploidy_file(ploidies: Optional[Union[Path, str, Dict[str, int]]], logger: GranLogger) -> Dict[str, int]:
        # No need to reload
        if isinstance(ploidies, dict):
            return ploidies
        if not ploidies:
            return {}
        step = "Reading ploidy file"
        logger.report_step(step, "In progress...")
        ploidies = CommonOps._load_single_content(ploidies, "ploidies", logger, key="e")
        ploid_dict = {}
        try:
            lines = ploidies.splitlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 2:
                    species, ploidy = parts
                    ploid_dict[species] = int(ploidy)
        except Exception as e:
            logger.log(f"reading ploidy file: {e}", 'e')
        if not ploid_dict:
            logger.log("Ploidy file is empty or invalid.", 'w')
        logger.report_step(step, f"Success: Loaded ploidies for {len(ploid_dict)} species")
        return ploid_dict

    ### Beta
    def _count_effective_lineages(self) -> Dict[str, Tuple[int, int]]:
        """
        Counts effective lineages for each species in the current ST.
        Returns a dict: {species: (number_of_pure_groups, max_size_of_pure_group)}
        Logic:
        1. Pure Groups: A clade where all descendants are the same species.
        2. Polytomies: Siblings of the same pure species at a mixed node are aggregated 
           into a single 'group' (e.g., (x,x,y) counts as one x-group of size 2).
        3. Nested Pure Groups: Only the maximal pure group is counted (e.g., ((x,x),x) is 1 group of size 3).
        """
        # Data structure: species -> [count, max_size]
        counts = {}
        
        def update_counts(species, size):
            if species not in counts:
                counts[species] = [0, 0]
            counts[species][0] += 1
            counts[species][1] = max(counts[species][1], size)

        def get_state(node):
            """
            Returns (species, size) if the node represents a pure clade.
            Returns None if the node is mixed.
            """
            if node.is_leaf():
                # Extract species name (remove '*' if present from previous MUL-tree ops)
                '''sp = node.name.replace("*", "")
                return (sp, 1)'''
                return (node.pure, 1)

            # Get states of all children
            child_states = [get_state(child) for child in node.children]

            # Check purity: Are all children pure and of the same species?
            first_sp = child_states[0][0] if child_states and child_states[0] else None
            all_same = True
            total_size = 0
            
            for state in child_states:
                if state is None or state[0] != first_sp:
                    all_same = False
                    break
                total_size += state[1]

            if all_same and first_sp is not None:
                # This node extends a pure lineage
                return (first_sp, total_size)
            
            # If mixed (or children have different species), finalize the pure children
            # Aggregate by species to handle polytomies like (x, x, y)
            current_level_groups = {} # species -> total_size
            
            for state in child_states:
                if state is not None:
                    sp, size = state
                    current_level_groups[sp] = current_level_groups.get(sp, 0) + size
            
            # Record these finalized groups
            for sp, size in current_level_groups.items():
                update_counts(sp, size)

            return None

        # Start traversal from root
        root_state = get_state(self.st.ete_tree)

        # Edge case: If the entire tree is pure (e.g., ((x,x),x)), 
        # get_state returns a value for root that hasn't been recorded yet.
        if root_state is not None:
            update_counts(root_state[0], root_state[1])

        # Convert lists to tuples for return type consistency
        return {k: tuple(v) for k, v in counts.items()}
        
    ### Beta
    def _apply_ploidy_constraints(self, h1_candidates: List[str], bin_id: Optional[int] = None) -> List[str]:
        """
        Filters H1 candidates based on the ploidy file loaded at init.
        Only H1 is filtered because H1 is the lineage being duplicated.
        """
        filtered_h1 = []
        check_fn = None

        # 1. Get current counts: {species: (num_groups, max_size)}
        if not bin_id:
            current_stats = self._count_effective_lineages()

            check_fn = lambda current, limit: current >= limit # In this mode: num_of_groups >= limit: # or max_size*2 > limit: ??
        else:
            # Number of 1s in the binary representation indicates how many copies of this lineage will exist after gluing.
            max_size = bin_id.bit_count()
            # Eg, if the count is 2, this tree paricipated in 2 "inner" subproblems
            # Meaning, it should have 2**2 = 4 copies already (if glued)
            # Meaning, next iteration could create 2**3 = 8 copies - this is the number we should check!
            max_size = 2**(max_size+1)
            # Pure may not be needed here; for safety
            current_stats = {l.pure: (max_size, None) for l in self.st.ete_tree.get_leaves()}

            check_fn = lambda future, limit: future > limit # In this mode, it's very clear when we need to check
        
        for node_name in h1_candidates:
            node = self.st.get_node(node_name)
            if not node: continue
            
            clade_species = {l.name.replace("*", "") for l in node.iter_leaves()}
            
            is_valid = True
            for sp in clade_species:
                # Ploidy Limit Logic:
                # Interpretation: The dict value (e.g., x:2) is the Max Number of Groups allowed.
                # If current_stats[sp] (group count) >= Limit, we cannot add another group.
                
                limit = self.ploidies.get(sp, 999) # Default to infinite if not in file
                
                num_of_groups, max_size = current_stats.get(sp, (0,0))
                
                if check_fn(num_of_groups, limit):
                    is_valid = False
                    break
            
            if is_valid:
                filtered_h1.append(node_name)

        # H2 candidates are not filtered by ploidy count, as H2 is the *target* of insertion, not the source of duplication.
        return filtered_h1

    def _resolve_h_inputs(self, raw_input: str, h_type: str) -> List[str]:
        """
        Resolves h1/h2 inputs into a list of node names.
        Optimized to use get_leaf_names() for faster set operations.
        """
        if not raw_input:
            # Return Tips first (get_leaves), then Internal nodes (Post-order)
            # This matches the legacy GRAMPA behavior where `nodes` dict was built tips-first.
            tips = [n.name for n in self.st.ete_tree.get_leaves()]
            internal = [n.name for n in self.st.ete_tree.traverse("postorder") if not n.is_leaf()]
            return tips + internal

        if isinstance(raw_input, list):
            clade_lists = [g.split(",") for g in raw_input]
        elif " " in raw_input:
            groups = raw_input.split(" ")
            clade_lists = [g.split(",") for g in groups]
        else:
            clade_lists = [raw_input.split(",")]

        h_nodes = []
        for clade in clade_lists:
            cleaned_clade = []
            for item in clade:
                name_to_check = f"<{item}>" if item.isdigit() else item
                if not self.st.get_node(name_to_check):
                    self.logger.log(f"Node {name_to_check} not found in tree (specified in -{h_type}).", 'e')
                    sys.exit(1)
                cleaned_clade.append(name_to_check)

            if len(cleaned_clade) == 1:
                val = cleaned_clade[0]
                if val not in h_nodes:
                    h_nodes.append(val)
            else:
                nodes_obj = [self.st.get_node(name) for name in cleaned_clade]
                lca_node = self.st.ete_tree.get_common_ancestor(nodes_obj)
                
                # OPTIMIZATION: Use get_leaf_names() instead of iter_leaves()
                # ETE3 get_leaf_names is significantly faster as it avoids creating Node objects
                lca_leaves = set(lca_node.get_leaf_names())
                input_set = set(cleaned_clade)
                
                # Check subset relationship
                if len(lca_leaves) != len(input_set) or not input_set.issubset(lca_leaves):
                    self.logger.log(f"All hybrid clades specified {h_type} must be monophyletic. LCA produced {lca_leaves} and supercedes input {input_set}", 'd')

                if lca_node.name not in h_nodes:
                    h_nodes.append(lca_node.name)
                    
        return h_nodes

    ### multi_H to debug the new builder
    def build(self, optim: bool = False, nestedness: str ='ignore') -> dict:
        mul_trees = {}
        
        # --- GUIDED ITERATIVE INTERCEPT ---
        if hasattr(self.tcf, 'predefined_rets') and self.tcf.predefined_rets:
            step = "Building Predefined MUL-trees"
            self.logger.report_step(step, "In progress...")
            
            mul_trees[0] = MulTree(mt=self.st) # Index 0 is always the ST
            mul_num = 1

            prerets = list(self.tcf.predefined_rets.values())
            # Flatten the lists
            prerets = [pair for sublist in prerets for pair in sublist]
            print("Predefined reticulations for this iteration:", prerets)
            
            for h1_str, h2_str in prerets:
                # Resolve the raw strings against the current ST context (which may now contain <P2> tags from previous iters)
                h1_res = self._resolve_h_inputs(h1_str, "h1")
                h2_res = self._resolve_h_inputs(h2_str, "h2")
                
                if not h1_res or not h2_res: continue
                
                h1, h2 = h1_res[0], h2_res[0]
                h1_st_node = self.st.get_node(h1)
                h_clade = [l.name for l in h1_st_node.iter_leaves()]
                
                mt_wrapper, h1_obj, h2_obj = self.st.to_mul_tree(h1, h2)
                if mt_wrapper:
                    mul_trees[mul_num] = MulTree(mt_wrapper, h_clade, h1_obj, hx_nodes=[h2_obj])
                    mul_num += 1
                    
            self.logger.report_step(step, f"Success: {mul_num-1} Predefined MUL-trees built")
            return mul_trees, [], [], self.ploidies

        if self.tcf.mode != "st-only":
            step = "Parsing hybrid clades"
            self.logger.report_step(step, "In progress...")
            h1_resolved = self._resolve_h_inputs(self.tcf.h1_nodes, "h1")
            h2_resolved = self._resolve_h_inputs(self.tcf.h2_nodes, "h2")
            self.logger.report_step(step, "Success: got H nodes")

            if self.ploidies:
                step = "Applying ploidy constraints"
                self.logger.report_step(step, "In progress...")
                h1_resolved = self._apply_ploidy_constraints(h1_resolved, bin_id = self.tcf.binary_id)
                self.logger.log(f"After ploidy filtering, {len(h1_resolved)} H1 candidates remain: {h1_resolved}", 'd')
                self.logger.report_step(step, "Success: identified compatible H nodes")

            step = "Counting MUL-trees to generate"
            self.logger.report_step(step, "In progress...")
            
            if optim:
                num_mul_trees = self.countMULTrees_Optimized(h1_resolved, h2_resolved)
            else:
            
                num_mul_trees = 0
                for n1_name in h1_resolved:
                    n1_node = self.st.get_node(n1_name)
                    if n1_node.is_leaf():
                        n1_clade_names = set()
                    else:
                        n1_clade_names = {n.name for n in n1_node.iter_descendants()}
                    
                    ni = 0
                    for n2_name in h2_resolved:
                        if n2_name not in n1_clade_names:
                            ni += 1
                    num_mul_trees += ni
                
            self.logger.report_step(step, f"Success: {num_mul_trees} total MUL-trees")
        else:
            h1_resolved = []
            h2_resolved = []

        # Index 0 is species tree itself regardless of mode
        mul_trees[0] = MulTree(mt=self.st)
        
        if self.tcf.mode != "st-only":
            step = "Building MUL-trees"
            self.logger.report_step(step, "In progress...")
            
            mul_num = 1
            for h1 in h1_resolved:
                h1_st_node = self.st.get_node(h1)
                h_clade = [l.name for l in h1_st_node.iter_leaves()]

                # Pre-calculate set for naive check to speed up (sets are O(1))
                # Only needed if NOT optim
                if not optim and not h1_st_node.is_leaf():
                    n1_descendants = {n.name for n in h1_st_node.iter_descendants()}
                else:
                    n1_descendants = set()

                processed_targets = set()

                for h2 in h2_resolved:
                    
                    is_nested = False
                    
                    if optim:
                        # DFS Logic
                        n1_node = self.st.get_node(h1)
                        if not n1_node.is_leaf():
                            if n1_node in self.subtree_range:
                                L, R = self.subtree_range[n1_node]
                                h2_node = self.st.get_node(h2)
                                if h2_node in self.dfs_order:
                                    idx = self.dfs_order[h2_node]
                                    # [FIX] Strict L < idx (Exclude self from nested definition to match naive)
                                    if L < idx <= R:
                                        is_nested = True
                    else:
                        # [FIX] RESTORE NAIVE LOGIC
                        # If optim is OFF, we MUST still check!
                        if h2 in n1_descendants:
                            is_nested = True
                            
                    if is_nested: continue # Skip this pair

                    # --- MODEL MODE LOGIC ---
                    if nestedness == 'model':
                        if h2 in processed_targets: continue
                        
                        all_targets = self._find_all_targets(h2)
                        processed_targets.update(all_targets)
                        
                        mt_wrapper, h1_obj, hx_objs = self.st.to_mul_tree_multi(h1, all_targets)
                        
                        if mt_wrapper:
                            mul_trees[mul_num] = MulTree(mt_wrapper, h_clade, h1_obj, hx_nodes=hx_objs)
                            mul_num += 1

                    # --- STANDARD LOGIC (Rectify/Ignore) ---
                    else:
                        mt_wrapper, h1_obj, h2_obj = self.st.to_mul_tree(h1, h2)
                        if mt_wrapper:
                            # Wrap single H2 in list for consistency
                            mul_trees[mul_num] = MulTree(mt_wrapper, h_clade, h1_obj, hx_nodes=[h2_obj])
                            mul_num += 1
                    
                    """mt_wrapper, h1_obj, h2_obj = self.st.to_mul_tree(h1, h2)
                    if mt_wrapper:
                        mul_trees[mul_num] = MulTree(mt_wrapper, h_clade, h1_obj, h2_obj)
                        mul_num += 1"""

            self.logger.report_step(step, f"Success: {mul_num-1} MUL-trees built")
            
        return mul_trees, h1_resolved, h2_resolved, self.ploidies

    # --- Legacy Support ---

    def report_num_trees(self):
        """Replicates the --numtrees output from legacy mul_tree.py"""
        
        # 1. Parse H inputs (logic borrowed from build())
        h1_resolved = self._resolve_h_inputs(self.tcf.h1_nodes, "h1")
        h2_resolved = self._resolve_h_inputs(self.tcf.h2_nodes, "h2")
        
        # 2. Count Logic (Legacy implementation)
        # Note: We use the already loaded self.st which is a SmrtTree
        st_ete = self.st.ete_tree
        nt = len([n for n in st_ete.traverse()])
        n_tips = len(st_ete.get_leaves())
        
        num_mul_trees = 0
        for n1 in h1_resolved:
            n1_node = self.st.get_node(n1)
            # Get clade for containment check
            if n1_node.is_leaf():
                n1_clade_names = set()
            else:
                n1_clade_names = {n.name for n in n1_node.iter_descendants()}

            ni = 0
            for n2 in h2_resolved:
                if n2 not in n1_clade_names:
                    ni += 1
            num_mul_trees += ni

        # 3. Print Block
        # Using print directly as this is a specific CLI report tool
        print()
        print(f"Total nodes in species tree: {nt}")
        print(f"Total tips in species tree.: {n_tips}")
        print(f"H1 nodes...................: {','.join(h1_resolved)}")
        print(f"H2 nodes...................: {','.join(h2_resolved)}")
        print(f"Possible MUL-trees.........: {num_mul_trees}")
        print()

    def report_build_multrees(self):
        """Replicates --buildmultrees output loop."""
        # Reuse existing build() logic
        mul_trees, h1_res, h2_res, _ = self.build()
        
        # Legacy prints headers to log/screen depending on verbosity
        # GRANDMA has unified logging. We log as 'i' (Info/High Priority).
        
        # Headers: mul.tree, h1.node, h2.node, labeled.tree
        headers = ["mul.tree", "h1.node", "h2.node", "labeled.tree"]
        self.logger.log("\t".join(headers), 'i')
        
        for idx in sorted(mul_trees.keys()):
            if idx == 0: continue # Legacy buildmultrees skips the ST (Index 0)
            
            mt_data = mul_trees[idx]
            
            # Format tree string (add + to hybrid clade)
            tree_str = mt_data.mt.to_str(internals=True)
            for spec in mt_data.h_clade:
                # Regex: spec not followed by *
                import re
                tree_str = re.sub(f"{spec}(?!\\*)", f"{spec}+", tree_str)
                tree_str = tree_str.replace("+*", "*")
            
            h1_name = mt_data.h1_node.name if mt_data.h1_node else "NA"
            h2_name = mt_data.h2_node.name if mt_data.h2_node else "NA"
            
            line = f"{idx}\t{h1_name}\t{h2_name}\t{tree_str}"
            self.logger.log(line, 'i')

# --- Multiprocessing Workers ---

# --- Global Holders for Worker Processes ---
_worker_gene_trees = {}
_worker_registry = None

def _init_collapse_worker(data_path: Optional[Union[Path, str]] = None) -> None:
    """
    Universal Initializer:
    Loads data from the temporary dump file.
    """
    global _worker_gene_trees, _worker_registry
    
    # Load Data (Trees & Registry)
    if data_path:
        try:
            # Determine how to open based on path type
            # (pathlib objects work in open(), but safety cast doesn't hurt)
            with open(str(data_path), 'rb') as f:
                data_payload = pickle.load(f)
            
            _worker_gene_trees = data_payload['trees']
            _worker_registry = data_payload['registry']
            
        except Exception as e:
            # Critical failure reporting
            print(f"CRITICAL: Worker process failed to load temp data from {data_path}: {e}", file=sys.stderr)
            raise e

def _collapse_worker(payload: Tuple[int, MulTree]) -> Tuple[int, Dict[int, GroupData]]:
    """
    Standard worker logic using global data.
    """
    m_idx, m_data = payload
    
    # Access globals
    h1_sis, hx_sis_list = m_data.get_sister_clades()
    
    current_mt_groups = {}
    for g_idx, gt_obj in _worker_gene_trees.items():
        group_data = gt_obj.compute_groups(m_data, _worker_registry, h1_sis, hx_sis_list)
        current_mt_groups[g_idx] = group_data
    
    return m_idx, current_mt_groups

# --- Worker for parallel filtering ---
def _check_and_write_worker(payload: Tuple[int, MulTree, Path, List[int], int, bool]) -> Tuple[int, str, Dict[int, List[int]]]:
    """
    Worker to process a single MUL-tree's checknums logic and create formatted output string.
    Payload: (m_idx, m_data, pickle_path, sorted_gene_ids, group_cap, is_mul_input)
    Returns: (m_idx, formatted_string_buffer, failures_dict)
    """
    m_idx, m_data, pickle_path, sorted_gene_ids, group_cap, is_mul_input = payload
    
    buffer = []
    local_failures = {} # g_idx -> list of m_idxs (just [m_idx])

    # Format Tree String
    mt_str = m_data.mt.to_marked_str(m_data.h1_node)

    h_info = ""
    if not is_mul_input:
        h1_name = m_data.h1_node.name if m_data.h1_node else "NA"
        hx_sisters = m_data.hx_sisters
        if hx_sisters:
            hx_str = "\t".join([f'H{i+2} Node:{hx.name}' for i, hx in enumerate(hx_sisters)])
        else:
            hx_str = "Hx Nodes:NA"
        h_info = f"\tH1 Node:{h1_name}\t{hx_str}"
    
    buffer.append(f"# MT-{m_idx}:{mt_str}{h_info}\n")

    # 2. Process Pickle Data
    if os.path.exists(pickle_path):
        try:
            with open(pickle_path, 'rb') as pf:
                current_mt_groups = pickle.load(pf)
                
            for g_idx in sorted_gene_ids:
                if g_idx not in current_mt_groups: continue
                
                groups = current_mt_groups[g_idx]
                num_ambig = len(groups.ambiguous_groups)
                num_fixed = len(groups.fixed_groups)
                total_groups = num_ambig + num_fixed
                
                over_cap = "N"
                if num_ambig > group_cap:
                    over_cap = "Y"
                    local_failures[g_idx] = [m_idx]

                combos = 1 << num_ambig 
                buffer.append(f"{m_idx}\t{g_idx+1}\t{total_groups}\t{num_fixed}\t{combos}\t{over_cap}\n")
                
        except Exception:
            # If pickle fails in worker, we might log or ignore? 
            # For now, append error to buffer so it appears in file
            buffer.append(f"# Error processing groups for MT-{m_idx}\n")

    buffer.append(f"# ----------------------------------\n")
    return m_idx, "".join(buffer), local_failures

class GeneTreeManager:
    def __init__(self, config: TaskConfig, logger: GranLogger, num_processes: int = 1):
        self.tcf = config
        self.logger = logger
        self.n_procs = num_processes
        
    def cull(self, mul_trees: Dict[int, MulTree], gene_trees: Dict[int, SmrtTree], registry: NameRegistry):
        if self.tcf.mode != "st-only":
            self.collapse_groups(mul_trees, gene_trees, registry)
        original_count = max(gene_trees.keys()) + 1 if gene_trees else 0
        gt_failures = self.filter_and_check(mul_trees, gene_trees)
        self.write_filtered_trees(gene_trees, gt_failures, original_count)

    def _check_registry_safety(self, registry_path: Path, registry: NameRegistry) -> bool:
        """Registry Persistence Logic"""
        registry_loaded = False
        
        # Try to load the registry snapshot from the previous run
        if registry_path.exists() and not self.tcf.overwrite:
            try:
                with open(registry_path, 'rb') as f:
                    saved_state = pickle.load(f)
                    registry.set_state(saved_state)
                registry_loaded = True
            except Exception as e:
                self.logger.log(f"Could not load registry pickle ({e}). Regenerating all groups.", 'w')
        
        # If we couldn't load the registry, we cannot trust the group pickles 
        # (they contain IDs that map to the old registry) -> overwrite them.
        return not registry_loaded

    def collapse_groups(self, mul_trees: Dict[int, MulTree], gene_trees: Dict[int, SmrtTree], registry: NameRegistry):
        """
        Computes groups for all MUL-trees using registry-optimized logic and DUMPS to pickle immediately.
        Does NOT retain data in memory.
        """
        step = "Collapsing gene tree groupings"
        self.logger.report_step(step, "In progress...")

        # --- PRIME REGISTRY ---
        # We must register all gene tree taxa in the main registry BEFORE forking.
        # Otherwise, workers create divergent ID mappings (e.g., 'TaxonA' is ID 5 in Worker 1, but ID 8 in Main).
        # We assume compute_groups uses registry.get_id(name). 
        # By iterating and 'touching' names here, we lock the IDs globally.
        if self.n_procs > 1:
            # Optimization: Use C-level iteration to collect all keys at once
            # Note: iterating a dict (gt.node_map) yields its keys automatically
            all_names = set(chain.from_iterable(gt.node_map for gt in gene_trees.values()))
            # Register all at once
            for name in all_names:
                registry.get_id(name)
        
        pickle_dir = self.tcf.pickle_dir
        pickle_dir.mkdir(parents=True, exist_ok=True)

        registry_path = pickle_dir / f"{self.tcf.run_prefix}_registry.pickle"
        force_regenerate = self._check_registry_safety(registry_path, registry)
     
        # Prepare Tasks (skip 0)
        # Only process if pickle doesn't exist or forced
        tasks = []
        for m_idx, m_data in mul_trees.items():
            if m_idx == 0: continue
            
            pickle_path = pickle_dir / f"{self.tcf.run_prefix}_{m_idx}_groups.pickle"
            if pickle_path.exists() and not self.tcf.overwrite and not force_regenerate:
                continue
            tasks.append((m_idx, m_data))

        # --- Parallel Execution ---
        
        def save_result(res):
            m_idx, groups = res
            pickle_path = pickle_dir / f"{self.tcf.run_prefix}_{m_idx}_groups.pickle"
            try:
                with open(pickle_path, 'wb') as f:
                    pickle.dump(groups, f)
            except Exception as e:
                self.logger.log(f"saving pickle {pickle_path}: {e}", 'e')

        if tasks:
            if self.n_procs > 1:
                # Create a temporary file to hold the massive data
                # delete=False is safer for Windows (prevents access errors if opened twice)
                # We manually unlink later.
                fd, temp_file_path = tempfile.mkstemp(suffix=".pkl", prefix="grandma_worker_data_")
                os.close(fd) 

                try:
                    # Dump data once
                    #self.logger.log("Serializing worker data to temp file...", 'd')
                    payload = {'trees': gene_trees, 'registry': registry}
                    with open(temp_file_path, 'wb') as f:
                        pickle.dump(payload, f)
                    
                    # Init Workers pointing to file
                    # This works on ALL OSs
                    with mp.Pool(processes=self.n_procs, initializer=_init_collapse_worker, initargs=(temp_file_path,)) as pool:
                        for res in tqdm(pool.imap_unordered(_collapse_worker, tasks), total=len(tasks), desc="Collapsing", unit="mt", 
                                      disable=self.logger.disable_tqdm):
                            save_result(res)
                            
                finally:
                    # Cleanup Temp File
                    if os.path.exists(temp_file_path):
                        try:
                            os.remove(temp_file_path)
                            #self.logger.log(f"Cleaned up temp file {temp_file_path}", 'd')
                        except OSError as e:
                            self.logger.log(f"Failed to delete temp file {temp_file_path}: {e}", 'w')

            else:
                # Serial Execution (Single Core)
                # Manually set globals to avoid rewriting worker logic
                global _worker_gene_trees, _worker_registry
                _worker_gene_trees = gene_trees
                _worker_registry = registry
                
                # We don't need init function, just call worker directly
                for item in tqdm(tasks, desc="Collapsing", unit="mt", disable=self.logger.disable_tqdm):
                    res = _collapse_worker(item)
                    save_result(res)
                
                # Cleanup globals
                _worker_gene_trees = {}
                _worker_registry = None
            
        # Save the registry state so the next run can interpret the IDs in the group pickles
        try:
            with open(registry_path, 'wb') as f:
                pickle.dump(registry.get_state(), f)
        except Exception as e:
            self.logger.log(f"saving registry pickle: {e}", 'e')
            
        self.logger.report_step(step, "Success")

    def filter_and_check(self, mul_trees: Dict[int, MulTree], gene_trees: Dict[int, SmrtTree]) -> Dict[int, int]:
        """
        Writes checknums file and filters trees exceeding group cap.
        [UPDATED] Uses multiprocessing to prepare string buffers.
        """
        step = "Filtering gene trees over group cap"
        self.logger.report_step(step, "In progress...")
        
        check_path = Path(self.tcf.output_dir) / f"{self.tcf.run_prefix}-checknums.txt"
        
        gt_failures = {} 
        sorted_mul_ids = sorted(mul_trees.keys())
        sorted_gene_ids = sorted(gene_trees.keys())

        # Prepare Tasks
        tasks = []
        for m_idx in sorted_mul_ids:
            if m_idx == 0: continue
            
            m_data = mul_trees[m_idx]
            pickle_path = self.tcf.pickle_dir / f"{self.tcf.run_prefix}_{m_idx}_groups.pickle"
            
            # Payload: (m_idx, m_data, pickle_path, gene_ids, cap, is_mul_input)
            tasks.append((
                m_idx, m_data, str(pickle_path), 
                sorted_gene_ids, self.tcf.group_cap, self.tcf.is_mul_input
            ))
        
        # Results container: Map m_idx -> string buffer
        results_map = {}

        # Execution
        if tasks:
            if self.n_procs > 1:
                with mp.Pool(processes=self.n_procs) as pool:
                    for res in tqdm(pool.imap_unordered(_check_and_write_worker, tasks), total=len(tasks), desc="Checking  ", unit="mt", disable=self.logger.disable_tqdm):
                        m_idx, buf, fails = res
                        results_map[m_idx] = buf
                        # Merge failures
                        for g_idx, failures in fails.items():
                            if g_idx not in gt_failures: gt_failures[g_idx] = []
                            gt_failures[g_idx].extend(failures)
            else:
                for task in tqdm(tasks, desc="Checking  ", unit="mt", disable=self.logger.disable_tqdm):
                    m_idx, buf, fails = _check_and_write_worker(task)
                    results_map[m_idx] = buf
                    for g_idx, failures in fails.items():
                        if g_idx not in gt_failures: gt_failures[g_idx] = []
                        gt_failures[g_idx].extend(failures)

        # Write to file (Sequential, in order)
        with open(check_path, 'w') as f:
            f.write("mul.tree\tgene.tree\tgroups\tfixed\tcombinations\tover.cap.filtered\n")
            for m_idx in sorted_mul_ids:
                if m_idx in results_map:
                    f.write(results_map[m_idx])

        self.logger.report_step(step, f"Success: {len(gt_failures)} gts over cap")
        
        for g_idx in sorted(gt_failures.keys()):
            if g_idx in gene_trees:
                gt_failures[g_idx] = len(gt_failures[g_idx])
                self.logger.log(f"Gene tree on line {g_idx+1} is over the group cap in {gt_failures[g_idx]} MTs and will be filtered.", 'w')
                del gene_trees[g_idx]
        return gt_failures
                
    def write_filtered_trees(self, gene_trees: Dict[int, SmrtTree], gt_failures: Dict[int, int], original_count: int):
        """Matches GRAMPA's filterOut logic."""
        step = "Writing filtered gene trees to file"
        self.logger.report_step(step, "In progress...")
        
        p = Path(self.tcf.output_dir) / f"{self.tcf.run_prefix}-trees-filtered.txt"
        count = 0
        with open(p, 'w') as f:
            for idx in range(original_count):
                if idx in gt_failures:
                    f.write(f"# Over group cap in {gt_failures[idx]} MUL-trees\n")
                elif idx in gene_trees:
                    f.write(gene_trees[idx].to_str(internals=True) + "\n")
                    count += 1
                else:
                    # This happens in later iterations, especially by the split mode
                    f.write(f"# Already filtered out\n")
                
        self.logger.report_step(step, f"Success: {count} gene trees written")
