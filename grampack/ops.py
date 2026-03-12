import os
import re
import sys
import pickle
import tempfile
import multiprocessing as mp
from tqdm import tqdm
from pathlib import Path
from shutil import rmtree, make_archive, unpack_archive
from itertools import chain
from functools import partial
from typing import Tuple, List, Optional, Dict, Union, Set

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
        
        # Universal tree characters that Windows paths explicitly forbid
        if isinstance(p, str) and any(c in p for c in ('(', ')', ';')):
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
    _SANITIZE_TRANS = str.maketrans('_.', '--') # Replace any chars in '_.' with '-'

    @staticmethod
    def spec_tree(tcf: TaskConfig, logger: GranLogger) -> Optional[SmrtTree]:

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
        st_wrapper = SmrtTree(tree_obj=t)

        if tcf.mode == "label-sp":
            logger.log(f"The input species tree with internal nodes labeled:", 'i')
            logger.log(f"{st_wrapper.to_str(internals=True)}", 'i', prefix='')
            logger.log(f"Label-sp Mode terminates before parsing the species tree: exiting...", 'i')
            return None
        return st_wrapper

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
            if tcf.mode in ('build-mts', 'count-mts', 'label-sp'):
                logger.report_step(step, f"Skipped: '{tcf.mode}' mode")
                return {}
            else:
                logger.log(f"Gene trees input is missing. Required in all modes except 'build-mts' (here: '{tcf.mode}' mode).", 'e')

        # Load Raw Contents (File, String, or Folder)
        tree_list, origins = CommonOps._load_multi_content(tcf.gts, "gene trees", logger, key="e")
        
        # Process Trees
        valid_gts = []
        #st_taxa = set(tcf.st.ete_tree.iter_leaf_names())
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
            '''gt_taxa = set(gt.ete_tree.iter_leaf_names())
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
        leaves = len(t)
        internal = sum(not n.is_leaf() for n in t.traverse())
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
            for l in t.iter_leaves():
                if not l.name or l.name in leaves_to_keep:
                    l.name = None
                else:
                    leaves_to_keep.add(l.name)
            t.prune(leaves_to_keep, preserve_branch_length=True)

        def singlify_enewick(t_line: str) -> Tree:
            t = Tree(t_line, format=1) # ENewick parsing to handle reticulation labels
            # Identify valid backbone leaves (excluding any node with #H in the name)
            backbone_leaves = [n for n in t.iter_leaves() if not (n.name and "#H" in n.name)]
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
            succ_leaves = set(succ.iter_leaf_names())

            logger.log(f"Reticulation structure: Succ: {succ_leaves} N1: {n1.get_leaf_names()} N2: {n2.get_leaf_names()}", 'd')
            logger.log(f"Succ Tree:\n{succ.get_ascii(show_internal=True)}", 'd')
            logger.log(f"Succ up Tree:\n{succ.up.get_ascii(show_internal=True)}", 'd')
            logger.log(f"N1 Tree:\n{n1.get_ascii(show_internal=True)}", 'd')
            logger.log(f"N2 Tree:\n{n2.get_ascii(show_internal=True)}", 'd')

            n1_children = n1.get_children()
            n2_children = n2.get_children()

            n2c0_leaves = n2_children[0].get_leaf_names()
            n2_sister_children = n2c0_leaves if set(n2c0_leaves) != succ_leaves else n2_children[1].get_leaf_names()

            n1c0_leaves = n1_children[0].get_leaf_names()
            n1_sister_children = n1c0_leaves if set(n1c0_leaves) != succ_leaves else n1_children[1].get_leaf_names()

            h_str = ",".join(succ_leaves)
            p1_str = ",".join(n1_sister_children)
            p2_str = ",".join(n2_sister_children)
            
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
                if set(p1_sis.iter_leaf_names()) == set(h_str.split(',')):
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
    def _sanitize_tip(s: str) -> str:
        return s.translate(TreeLoader._SANITIZE_TRANS)

class MulTreeManager:
    __slots__ = ['tcf', 'st', 'logger', 'ploidies']

    def __init__(self, config: TaskConfig, st: SmrtTree, logger: GranLogger) -> None:
        self.tcf = config
        self.st = st
        self.logger = logger
        self.ploidies: Dict[str, int] = self._parse_ploidy_file(self.tcf.ploidies, logger)

    @staticmethod
    def _parse_ploidy_file(ploidies: Optional[Union[Path, str, Dict[str, int]]], logger: GranLogger) -> Dict[str, int]:
        # No need to reload
        if isinstance(ploidies, dict):
            return ploidies
        if not ploidies:
            return {}
        step = "Reading ploidy file"
        logger.report_step(step, "In progress...")
        ploidy_content = CommonOps._load_single_content(ploidies, "ploidies", logger, key="e")
        ploidy_dict: Dict[str, int] = {}
        try:
            for line in ploidy_content.splitlines():
                parts = line.strip().split()
                if len(parts) == 2:
                    species, ploidy = parts
                    ploidy_dict[species] = int(ploidy)
        except Exception as e:
            logger.log(f"reading ploidy file: {e}", 'e')
        if not ploidy_dict:
            logger.log("Ploidy file is empty or invalid.", 'w')
        logger.report_step(step, f"Success: Loaded ploidies for {len(ploidy_dict)} species")
        return ploidy_dict

    @staticmethod
    def _count_effective_lineages(tree: Tree) -> Dict[str, Tuple[int, int]]:
        """
        Counts effective lineages for each species in the current ST.
        Populates self.ploidy_stats: {species: (number_of_pure_groups, max_size_of_pure_group)}
        Logic:
        1. Pure Groups: A clade where all descendants are the same species.
        2. Polytomies: Siblings of the same pure species at a mixed node are aggregated 
           into a single 'group' (e.g., (x,x,y) counts as one x-group of size 2).
        3. Nested Pure Groups: Only the maximal pure group is counted (e.g., ((x,x),x) is 1 group of size 3).
        """
        # Data structure: species -> [count, max_size]
        counts: Dict[str, List[int]] = {}
        
        def update_counts(species: str, size: int) -> None:
            if species not in counts:
                counts[species] = [0, 0]
            counts[species][0] += 1
            counts[species][1] = max(counts[species][1], size)

        def get_state(node: Tree) -> Optional[Tuple[str, int]]:
            """
            Returns (species, size) if the node represents a pure clade.
            Returns None if the node is mixed.
            """
            if node.is_leaf():
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
            current_level_groups: Dict[str, int] = {} # species -> total_size
            
            for state in child_states:
                if state is not None:
                    sp, size = state
                    current_level_groups[sp] = current_level_groups.get(sp, 0) + size
            
            # Record these finalized groups
            for sp, size in current_level_groups.items():
                update_counts(sp, size)

            return None

        # Start traversal from root
        root_state = get_state(tree)

        # Edge case: If the entire tree is pure (e.g., ((x,x),x)), 
        # get_state returns a value for root that hasn't been recorded yet.
        if root_state is not None:
            update_counts(root_state[0], root_state[1])

        # Convert lists to tuples for return type consistency
        return {k: tuple(v) for k, v in counts.items()}
        
    def _apply_ploidy_constraints(self, h1_candidates: List[str], bin_id: Optional[int], is_strict: bool) -> Tuple[List[str], Dict[str, float]]:
        """
        Filters H1 candidates and calculates how many NEW copies each can tolerate.
        Centralizes all ploidy math for Simple, Full, Split, and Mixed modes.
        Contains both the complex 'effective lineage' logic and the strict 'exact match count' logic.
        Only H1 is filtered because H1 is the lineage being duplicated, but the allowance is calculated for H2/x for when grafting.
        """
        filtered_h1: List[str] = []
        h1_allowances: Dict[str, float] = {}

        # Determine Current Statistics based on Stricter Option vs Default
        if is_strict:
            # STRICT MODE: Count exactly how many disjoint pure matches currently exist in the tree
            ploidy_stats = {}
            for sp in self.ploidies.keys():
                # Find a representative leaf to get the targets
                rep_leaf = next((l for l in self.st.ete_tree.iter_leaves() if l.pure == sp), None)
                if rep_leaf:
                    ploidy_stats[sp] = (len(self.st.get_targets(rep_leaf)), 1)
                else:
                    ploidy_stats[sp] = (0, 0)
        else:
            # DEFAULT MODE: Complex topological aggregation
            ploidy_stats = self.count_effective_lineages(self.st.ete_tree)

        # Determine the depth multiplier (1 for Full/Mixed baseline, 2^N for inner splits)
        multiplier = 1
        if bin_id is not None:
            # Number of 1s in the binary representation indicates how many copies of this lineage will exist
            # after gluing, e.g., if the count is 2, this tree paricipated in 2 "inner" subproblems.
            # Thus, it should have 2**2 = 4 copies (if glued) of what is **in the tree** (may be more than 1 in mixed mode, etc).
            multiplier = 2 ** bin_id.bit_count()
        
        for node_name in h1_candidates:
            node = self.st.get_node(node_name)
            
            clade_species = {l.replace("*", "").split('|')[0] for l in node.iter_leaf_names()}
            min_allowance = float('inf')
            
            for sp in clade_species:
                limit = self.ploidies.get(sp, 999) # Default to infinite if not in file           
                current_count, max_group_size = self.ploidy_stats.get(sp, (0,0))
                
                # Math: (Current + Allowed) * Multiplier <= Limit
                # Therefore: Allowed <= (Limit // Multiplier) - Current
                allowed = (limit // multiplier) - current_count
                
                if allowed < min_allowance:
                    min_allowance = allowed
            
            # If allowance is < 1, this H1 cannot participate in ANY grafts. Filter it out.
            if min_allowance >= 1:
                filtered_h1.append(node_name)
                h1_allowances[node_name] = float(min_allowance)

        return filtered_h1, h1_allowances

    def _resolve_h_inputs(self, raw_input: str, h_type: str) -> List[str]:
        """
        Resolves h1/h2 inputs into a list of node names.
        Optimized to use get_leaf_names() for faster set operations.
        """
        if not raw_input:
            # Return Tips first (get_leaves), then Internal nodes (Post-order)
            # This matches the legacy GRAMPA behavior where `nodes` dict was built tips-first.
            tips = self.st.ete_tree.get_leaf_names()
            internal = [n.name for n in self.st.ete_tree.traverse("postorder") if not n.is_leaf()]
            return tips + internal

        if isinstance(raw_input, list):
            clade_lists = [g.split(",") for g in raw_input]
        elif " " in raw_input:
            groups = raw_input.split(" ")
            clade_lists = [g.split(",") for g in groups]
        else:
            clade_lists = [raw_input.split(",")]

        h_nodes: List[str] = []
        for clade in clade_lists:
            cleaned_clade: List[str] = []
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
                
                lca_leaves = set(lca_node.iter_leaf_names())
                input_set = set(cleaned_clade)
                
                # Check subset relationship
                if len(lca_leaves) != len(input_set) or not input_set.issubset(lca_leaves):
                    self.logger.log(f"All hybrid clades specified {h_type} must be monophyletic. LCA produced {lca_leaves} and supercedes input {input_set}", 'd')

                if lca_node.name not in h_nodes:
                    h_nodes.append(lca_node.name)
                    
        return h_nodes

    @staticmethod
    def _is_redundant_graft(matches: List[Tree], h1_st_node: Tree, allow_redundant: bool) -> bool:
        if not allow_redundant:
            for target in matches:
                sisters = target.get_sisters()
                if len(sisters)==1 and sisters[0].pure == h1_st_node.pure:
                    return True
        return False

    def _compile_h2_targets(self, h1_st_node: Tree, h2_resolved: List[str], nesting: str, n1_pure_descendants: Set[str],
                            allowance: float, allow_redundant: bool) -> List[List[Tree]]:
        """
        Evaluates all H2 candidates for a given H1 and returns a list of valid match groups.
        Each item is a list of ETE3 nodes that should be grafted onto simultaneously.
        """
        valid_match_groups: List[List[Tree]] = []
        processed_targets: Set[str] = set()
        
        for h2 in h2_resolved:
            h2_node = self.st.get_node(h2)
            
            # Topological / Semantic Nesting
            if h2_node.pure in n1_pure_descendants:
                # For this check to work, .pure must not be None for any node (which is correct, but worth noting)
                continue
                
            # Mode grouping
            if nesting == 'model':
                if h2_node.pure in processed_targets:
                    continue
                matches = self.st.get_targets(h2_node)
            else:
                matches = [h2_node]
                
            # EDGE CASE: Prevent redundant grafting "below", e.g., (H1_old, (H1_new, C)). Only allow: (H1_new, (H1_old, C))
            # If any target's sister is a pure copy of H1, grafting here is topologically
            # identical to grafting onto the target's parent. Because the parent will ALSO
            # be evaluated as a target in this loop, skipping this prevents duplicate MUL-trees
            # and safely protects the internal integrity of previously marked <P> clades.
            is_redundant = self._is_redundant_graft(matches, h1_st_node, allow_redundant)
            # Compared to normal Grampa we produce (num_nodes-1) less MTs in the first iter too,
            # because each node that is not the root would be able to be grafted below itself,
            # but this is not a bug! It can still graft above itself... And the root node is not effected...
            # Allow_redundant (false by default) is for recreating this **wrong** behavior if desired or for testing.
            
            # Ploidy check and redundancy blocking
            if is_redundant or len(matches) > allowance:
                if nesting == 'model': processed_targets.add(h2_node.pure)
                continue
                    
            if nesting == 'model':
                processed_targets.add(h2_node.pure)
                
            valid_match_groups.append(matches)
            
        return valid_match_groups

    def build(self,
              nesting: str='model',
              strict_constraint: bool=False,
              allow_redundant_mts: bool=False) -> Tuple[Dict[int, MulTree], List[str], List[str], Dict[str, int]]:
        mul_trees: Dict[int, MulTree] = {}

        # --- ADD SPECIES TREE (INDEX 0) ---
        # Index 0 is species tree itself regardless of mode
        if self.tcf.mode != "no-st":
            mul_trees[0] = MulTree(mt=self.st)
        
        # --- GUIDED ITERATIVE INTERCEPT ---
        if hasattr(self.tcf, 'predefined_rets') and self.tcf.predefined_rets:
            step = "Building Predefined MUL-trees"
            self.logger.report_step(step, "In progress...")
            
            mul_num = 1

            prerets = [pair for sublist in self.tcf.predefined_rets.values() for pair in sublist]
            self.logger.log(f"Predefined reticulations for this iteration: {prerets}", 'd')
            
            for h1_str, h2_str in prerets:
                # Resolve the raw strings against the current ST context (which may now contain <P2> tags from previous iters)
                h1_res = self._resolve_h_inputs(h1_str, "h1")
                h2_res = self._resolve_h_inputs(h2_str, "h2")
                
                if not h1_res or not h2_res: continue
                
                h1, h2 = h1_res[0], h2_res[0]
                h1_st_node = self.st.get_node(h1)
                h_clade = h1_st_node.get_leaf_names()
                
                mt_wrapper, h1_obj, h2_obj = self.st.to_mul_tree(h1, h2)
                if mt_wrapper:
                    mul_trees[mul_num] = MulTree(mt_wrapper, h_clade, h1_obj, hx_nodes=[h2_obj])
                    mul_num += 1
                    
            self.logger.report_step(step, f"Success: {mul_num-1} Predefined MUL-trees built")
            return mul_trees, [], [], self.ploidies

        # --- EARLY RETURN FOR ST-ONLY ---
        if self.tcf.mode == "st-only":
            self.logger.log("INFO   : ST-only mode skips hybrid parsing and MUL-tree generation", 'i')
            return mul_trees, [], [], self.ploidies
        
        step = "Parsing hybrid clades"
        self.logger.report_step(step, "In progress...")
        h1_resolved_original = self._resolve_h_inputs(self.tcf.h1_nodes, "h1")
        h2_resolved = self._resolve_h_inputs(self.tcf.h2_nodes, "h2")
        self.logger.report_step(step, "Success: got H nodes")

        if self.ploidies:
            step = "Applying ploidy constraints"
            self.logger.report_step(step, "In progress...")
            h1_resolved, h1_allowances = self._apply_ploidy_constraints(h1_resolved_original, self.tcf.binary_id, strict_constraint)
            self.logger.log(f"After ploidy filtering, {len(h1_resolved)} H1 candidates remain: {h1_resolved}", 'd')
            self.logger.report_step(step, "Success: identified compatible H nodes")
        else:
            h1_resolved = h1_resolved_original.copy()
            h1_allowances: Dict[str, float] = {h: float('inf') for h in h1_resolved}

        # --- O(N) Bottom-Up Semantic Cache ---
        # Pre-calculate pure descendants for O(1) lookup
        # This catches BOTH standard topological descendants 
        # AND any previously grafted hybrid copies of those descendants!
        # Used for both counting and building blocks...
        # The issue was that the target could be a **subset** of H1's clade, e.g., H1=(A,B,C), H2=(A,B).
        # Then we need to recursively "autocorrect" H1 into insertions of H1 inside itself -- this would be a bug!!!
        # Even if in old -m we catch this, it is still a bug in the -r mode, and in -m, in that the Hx list is corrupted (e.g., [**, ****], missing 1 & 3 stars).
        # Is it enough to simply block this behavior? Yes!
        # Why? Becasue biologically, what we try to do is have H1=(A,B,C), and H2_prev=(A,B,C)=H1*
        # This is checked in the prev iteration, and back then, we already checked if duplicating (A,B,C) is more parsimonious than (A,B)!
        # There is no scenario where we would want to insert only a subset of a hybrid clade, inside itself, **partially**.
        # Later iterations can still doublicate (A,B) inside (A,B,C) if parsimonious!
        # Additionally, we must be careful to still allow duplicating (A,B,C) itself, but each local copy.
        # Hence we use iter_descendants() to not include self, or even better, a DFS cache to get strictly descendants in a single pass:
        pure_desc_cache: Dict[Tree, Set[str]] = self.st.desc_pure_cache

        # --- PRE-CALCULATION & COUNTING STEP ---
        # Pre-calculate valid pairings and use this for both counting and building steps
        # This ensures consistency and avoids redundant calculations
        
        step = "Counting MUL-trees" if self.tcf.mode == "count-mts" else "Counting MUL-trees to generate"
        self.logger.report_step(step, "In progress...")
        
        num_mul_trees = 0
        valid_pairings: Dict[str, List[List[Tree]]] = {}
        for h1 in h1_resolved:
            h1_st_node = self.st.get_node(h1)
            n1_pure_descendants = pure_desc_cache.get(h1_st_node, set())
            
            # Get the definitive list of valid target groupings
            match_groups = self._compile_h2_targets(h1_st_node, h2_resolved, nesting, n1_pure_descendants, h1_allowances[h1], allow_redundant_mts)
            valid_pairings[h1] = match_groups
            num_mul_trees += len(match_groups)
            
        self.logger.report_step(step, f"Success: {num_mul_trees} total MUL-trees")
                        
        if self.tcf.mode == "count-mts":
            self.report_mt_count(self.st.ete_tree, h1_resolved_original, h2_resolved, num_mul_trees, nesting, bool(self.ploidies), allow_redundant_mts)
            return {}, [], [], {} # Returns empty dict to signal main.py to exit
 
        # --- BUILDING STEP ---
        step = "Building MUL-trees"
        self.logger.report_step(step, "In progress...")
        
        mul_num = 1
        for h1 in h1_resolved:
            h1_st_node = self.st.get_node(h1)
            h_clade = h1_st_node.get_leaf_names()
            
            # We just iterate over the pre-calculated, validated groups!
            for matches in valid_pairings[h1]:
                all_targets = sorted([n.name for n in matches])
                
                # to_multi_mul_tree handles BOTH Simple and Model modes seamlessly!
                # (If simple, all_targets just has 1 item)
                mt_wrapper, h1_obj, hx_objs = self.st.to_multi_mul_tree(h1, all_targets)

                if mt_wrapper:
                    mul_trees[mul_num] = MulTree(mt_wrapper, h_clade, h1_obj, hx_nodes=hx_objs)
                    mul_num += 1
        
        self.logger.report_step(step, f"Success: {mul_num-1} MUL-trees built")

        if self.tcf.mode == "build-mts":
            self.report_mt_build(mul_trees, nesting)
            return {}, [], [], {} # Returns empty dict to signal main.py to exit

        if not mul_trees:
            self.logger.log("No valid MUL-trees could be generated with the given constraints.", 'w')
        if len(mul_trees) < (1 if self.tcf.mode in {"no-st", "st-only"} else 2):
            self.logger.log("Too few MUL-trees built. Check your H1/H2 and ploidy constraints.", 'w')
            
        return mul_trees, h1_resolved, h2_resolved, self.ploidies

    def report_mt_count(self, st_ete: Tree, h1_resolved: List[str], h2_resolved: List[str], num_mul_trees: int,
                        nesting: str, has_ploidies: bool, allow_redundant_mts: bool) -> None:
        """Prints a report of the MUL-tree count, replicating legacy mul_tree.py output."""
        n_tips = len(st_ete)
        n_nodes = n_tips*2 - 1 # Assumes full bifurcation (a valid species tree)
        # Max possible calculation
        name_to_node = {n.name: n for n in st_ete.traverse()}
        h2_set = set(h2_resolved)
        max_possible = 0
        for h1_name in h1_resolved:
            h1_node = name_to_node.get(h1_name)
            # Find descendants of H1
            desc_names = {d.name for d in h1_node.get_descendants()}
            # Valid H2s are the total H2s MINUS the H2s that are descendants of this H1
            invalid_h2s = desc_names.intersection(h2_set)
            max_possible += len(h2_resolved) - len(invalid_h2s)

        self.logger.log("MUL-tree Count Report: exiting after", 'i')
        self.logger.log(f"Total nodes in species tree: {n_nodes}", 'i')
        self.logger.log(f"Total tips in species tree.: {n_tips}", 'i')
        self.logger.log(f"H1 nodes...................: {','.join(h1_resolved)}", 'i')
        self.logger.log(f"H2 nodes...................: {','.join(h2_resolved)}", 'i')
        self.logger.log(f"Possible MUL-trees.........: {max_possible}", 'i')
        self.logger.log(f"Modeling Nested constraints: {'On' if nesting == 'model' else 'Off'}", 'i')
        self.logger.log(f"Ploidy constraints.........: {'On' if has_ploidies else 'Off'}", 'i')
        self.logger.log(f"Redundancy filter..........: {'Off' if allow_redundant_mts else 'On'}", 'i')
        self.logger.log(f"Actual MUL-trees...........: {num_mul_trees}", 'i')

    def report_mt_build(self, mul_trees: Dict[int, MulTree], nesting: str) -> None:
        """Prints a report of the built MUL-trees, replicating legacy mul_tree.py output."""
        self.logger.log("MUL-tree Build Report: exiting after", 'i')
        hx_colname = "hx.node" if nesting == "model" else "h2.node"
        self.logger.log("\t".join(["mul.tree", "h1.node", hx_colname, "labeled.tree"]), 'i', prefix='')
        for mul_num, mt_data in mul_trees.items():
            # Skip the base species tree like in the legacy code
            if mul_num == 0: 
                continue
            tree_str = mt_data.to_marked_str()
            h1_name = mt_data.h1_node.name
            hx_sisters = ",".join(n.name for n in mt_data.hx_sisters)
            self.logger.log(f"{mul_num}\t{h1_name}\t{hx_sisters}\t{tree_str}", 'i', prefix='')

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
def _check_and_write_worker(payload: Tuple[int, str, str, str], sorted_gene_ids: List[int] = None, group_cap: int = 8) -> Tuple[int, str, Dict[int, List[int]]]:
    """
    Worker to process a single MUL-tree's checknums logic.
    Payload: (m_idx, mt_str, h_info, pickle_path)
    Returns: (m_idx, formatted_string_buffer, failures_dict)
    """
    m_idx, mt_str, h_info, pickle_path = payload
    
    buffer = []
    local_failures = {} # g_idx -> list of m_idxs (just [m_idx])

    # Formatting tree string is done upstream
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
    def __init__(self, config: TaskConfig, logger: GranLogger, num_processes: int = 1, pickle_action: str = 'archive'):
        self.tcf = config
        self.logger = logger
        self.n_procs = num_processes
        self.pickle_action = pickle_action

    def cull(self, mul_trees: Dict[int, MulTree], gene_trees: Dict[int, SmrtTree], registry: NameRegistry) -> bool:
        if self.tcf.mode != "st-only":
            self.collapse_groups(mul_trees, gene_trees, registry)
        original_count = max(gene_trees.keys()) + 1 if gene_trees else 0
        gt_failures = self.filter_and_check(mul_trees, gene_trees)
        self.write_filtered_trees(gene_trees, gt_failures, original_count)
        if self.tcf.mode != "check-nums":
            return True
        self.logger.log("Check-nums Mode terminates before reconciliation: exiting...", 'i')
        self.handle_pickles()
        return False

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

    def handle_pickles(self) -> None:
        """Handles post-iteration pickle cleanup or compression."""
        pickle_dir = self.tcf.pickle_dir
        action = self.pickle_action
        if not pickle_dir.exists() or action.startswith('k'): # keep
            return

        if action.startswith('c'): # clean
            step = "Cleaning up pickle directory"
            self.logger.report_step(step, "In progress...")
            try:
                rmtree(pickle_dir)
                self.logger.report_step(step, "Success")
            except Exception as e:
                self.logger.log(f"Failed to clean pickle directory: {e}", 'w')

        elif action.startswith('a'): # archive
            step = "Archiving pickle directory"
            self.logger.report_step(step, "In progress...")
            try:
                # shutil.make_archive creates 'pkls.tar.gz' 
                archive_base_path = str(pickle_dir) 
                make_archive(
                    base_name=archive_base_path, 
                    format='gztar', 
                    root_dir=pickle_dir.parent, 
                    base_dir=pickle_dir.name
                )
                # Delete the uncompressed directory to free up the space and inodes
                rmtree(pickle_dir)
                self.logger.report_step(step, f"Success: created {pickle_dir.name}.tar.gz")
            except Exception as e:
                self.logger.log(f"Failed to archive pickle directory: {e}", 'w')

        else:
            self.logger.log(f"Unknown pickle handling action: '{action}'", 'w')

    def unpack_archive(self, pickle_dir: Path) -> None:
        archive_path = Path(str(pickle_dir) + '.tar.gz')
        if not pickle_dir.exists() and archive_path.exists() and not self.tcf.overwrite:
            step_unpack = "Unpacking pickle archive"
            self.logger.report_step(step_unpack, "In progress...")
            try:
                unpack_archive(archive_path, extract_dir=pickle_dir.parent)
                self.logger.report_step(step_unpack, "Success")
            except Exception as e:
                self.logger.log(f"Failed to unpack archive: {e}", 'w')

    def collapse_groups(self, mul_trees: Dict[int, MulTree], gene_trees: Dict[int, SmrtTree], registry: NameRegistry):
        """
        Computes groups for all MUL-trees using registry-optimized logic and DUMPS to pickle immediately.
        Does NOT retain data in memory.
        """
        pickle_dir = self.tcf.pickle_dir
        self.unpack_archive(pickle_dir) # Handle archived pickles if needed
        pickle_dir.mkdir(parents=True, exist_ok=True)
     
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
                        for res in tqdm(pool.imap_unordered(_collapse_worker, tasks), total=len(tasks), desc="# Collapsing", unit="mt", 
                                      disable=self.logger.disable_tqdm, ncols=177):
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
                for item in tqdm(tasks, desc="# Collapsing", unit="mt", disable=self.logger.disable_tqdm, ncols=177):
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
            
        self.logger.report_step(step, "Success", full_update=True)

    def filter_and_check(self, mul_trees: Dict[int, MulTree], gene_trees: Dict[int, SmrtTree]) -> Dict[int, int]:
        """
        Writes checknums file and filters trees exceeding group cap.
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
            
            # String Formatting
            mt_str = m_data.to_marked_str()
            h1_name = m_data.h1_node.name if m_data.h1_node else "NA"
            hx_sisters = m_data.hx_sisters
            if hx_sisters:
                hx_str = "\t".join([f'H{i+2} Node:{hx.name}' for i, hx in enumerate(hx_sisters)])
            else:
                hx_str = "Hx Nodes:NA"
            h_info = f"\tH1 Node:{h1_name}\t{hx_str}"
            
            tasks.append((m_idx, mt_str, h_info, str(pickle_path)))
        
        # Results container: Map m_idx -> string buffer
        results_map = {}

        # Bind repetitive arguments to avoid sending them per-task
        worker_func = partial(_check_and_write_worker, sorted_gene_ids=sorted_gene_ids, group_cap=self.tcf.group_cap)

        if tasks:
            if self.n_procs > 1:
                with mp.Pool(processes=self.n_procs) as pool:
                    iterator = pool.imap_unordered(worker_func, tasks)
                    for res in tqdm(iterator, total=len(tasks), desc="# Filtering ", unit="mt", disable=self.logger.disable_tqdm, ncols=177):
                        m_idx, buf, fails = res
                        results_map[m_idx] = buf
                        # Merge failures
                        for g_idx, failures in fails.items():
                            if g_idx not in gt_failures: gt_failures[g_idx] = []
                            gt_failures[g_idx].extend(failures)
            else:
                for task in tqdm(tasks, desc="# Filtering ", unit="mt", disable=self.logger.disable_tqdm, ncols=177):
                    m_idx, buf, fails = worker_func(task)
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

        self.logger.report_step(step, f"Success: {len(gt_failures)} gts over cap", full_update=True)
        
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
