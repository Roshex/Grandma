import os
import re
import sys
import pickle
import multiprocessing as mp
from tqdm import tqdm
from pathlib import Path
from functools import partial
from typing import Tuple, List, Optional, Dict, Union

from .config import TaskConfig
from .logger import GranLogger
from .models import Tree, SmrtTree, MulTree, GroupData, NameRegistry
from .reconcile import Reconciler

class CommonOps:
    @staticmethod
    def _fix_semicolon(tree_str: str) -> str:
        """Ensures tree strings end with a semicolon."""
        tree_str = tree_str.strip()
        return tree_str if tree_str.endswith(';') else tree_str + ';'

    @staticmethod
    def write_handoff_files(dir: Path, st: Tree=None, gts: Optional[List[Tree]]=None):
        """Writes the trees to disk to allow inspection/resume, matching iter_mode.py."""
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
        """Loads a single tree string from Path or String."""
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

        # MUL-Tree Validation
        TreeLoader._validate_mul_status(t, tcf.is_mul_input, logger)

        if tcf.repair:
            CommonOps._write_handoff_files(tcf.output_dir, st=t)

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
            CommonOps._write_handoff_files(tcf.output_dir, gts=valid_gts)
                
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
    def _validate_mul_status(t: Tree, is_mul_input: bool, logger: GranLogger):
        """Validates species counts against the flag."""
        leaves = [n.name for n in t.get_leaves()]
        counts = {}
        for l in leaves:
            counts[l] = counts.get(l, 0) + 1
        
        # Detect actual status
        has_dups = any(v > 1 for v in counts.values())
        
        if is_mul_input:
            # Appear exactly once or twice
            # Check if any appear > 2 (or 0, implicit)
            bad_taxa = [k for k, v in counts.items() if v > 2]
            if bad_taxa:
                logger.log(f"You have entered a tree type (--multree) of multree, species in your tree should appear exactly once or twice. (Violators: {bad_taxa})", 'w')
        else:
            # Standard mode, no dups allowed
            if has_dups:
                bad_taxa = [k for k, v in counts.items() if v > 1]
                logger.log("You have not entered a tree type (--multree) of multree, but there are labels in your tree that appear more than once!", 'w')

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
            with open(ploidies, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        species, ploidy = parts
                        ploid_dict[species] = int(ploidy)
        except Exception as e:
            logger.log(f"reading ploidy file: {e}", 'e')
        if not ploid_dict:
            logger.log("Ploidy file is empty or invalid.", 'w')
        logger.report_step(step, f"Success: Loaded ploidy info for {len(ploid_dict)} species")
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
                sp = node.name.replace("*", "")
                return (sp, 1)

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
    def _apply_ploidy_constraints(self, h1_candidates: List[str]) -> List[str]:
        """
        Filters H1 candidates based on the ploidy file loaded at init.
        Only H1 is filtered because H1 is the lineage being duplicated.
        """
        filtered_h1 = []

        # 1. Get current counts: {species: (num_groups, max_size)}
        current_stats = self._count_effective_lineages()
        
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
                
                if num_of_groups >= limit: # or max_size*2 > limit:
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

        if " " in raw_input:
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
                if len(lca_leaves) != len(input_set) and not input_set.issubset(lca_leaves):
                     pass

                if lca_node.name not in h_nodes:
                    h_nodes.append(lca_node.name)
                    
        return h_nodes

    def build(self) -> dict:
        mul_trees = {}
        
        if self.tcf.mode != "st-only":
            step = "Parsing hybrid clades"
            self.logger.report_step(step, "In progress...")
            h1_resolved = self._resolve_h_inputs(self.tcf.h1_nodes, "h1")
            h2_resolved = self._resolve_h_inputs(self.tcf.h2_nodes, "h2")
            self.logger.report_step(step, "Success: got H nodes")

            if self.ploidies:
                step = "Applying ploidy constraints"
                self.logger.report_step(step, "In progress...")
                h1_resolved = self._apply_ploidy_constraints(h1_resolved)
                self.logger.report_step(step, "Success: identified compatible H nodes")

            step = "Counting MUL-trees to generate"
            self.logger.report_step(step, "In progress...")
            
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
                for h2 in h2_resolved:

                    mt_wrapper, h1_obj, h2_obj = self.st.to_mul_tree(h1, h2)
                    if mt_wrapper:
                        mul_trees[mul_num] = MulTree(mt_wrapper, h_clade, h1_obj, h2_obj)
                        mul_num += 1

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
            tree_str = mt_data.mt.to_string(internal_labels=True)
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

_worker_data = {}

def _init_collapse_worker(gts, reg, rec):
    _worker_data['gts'] = gts
    _worker_data['reg'] = reg
    _worker_data['rec'] = rec

def _collapse_worker(item):
    m_idx, m_data = item
    rec = _worker_data['rec']
    h1_sis, h2_sis = rec.get_sister_clades(m_data)
    
    current_mt_groups = {}
    for g_idx, gt_obj in _worker_data['gts'].items():
        # Reconciler logic is pure, safe to call here
        group_data = rec.compute_groups(gt_obj, m_data, _worker_data['reg'], h1_sis, h2_sis)
        current_mt_groups[g_idx] = group_data
    return m_idx, current_mt_groups

# Worker for parallel filtering
def _check_worker(payload):
    """
    Worker to process a single MUL-tree's checknums logic.
    Returns: (m_idx, formatted_string_buffer, failures_dict)
    """
    m_idx, m_data, pickle_path, sorted_gene_ids, group_cap, is_mul_input = payload
    
    buffer = []
    local_failures = {} # g_idx -> list of m_idxs (just [m_idx])

    # 1. Format Tree String (Regex logic localized)
    # We re-compile regex per worker task (negligible overhead for one regex)
    # Pattern: \b(SpecA|SpecB)(?!\*)
    pattern_str = r'\b(' + '|'.join(map(re.escape, m_data.h_clade)) + r')(?!\*)'
    regex = re.compile(pattern_str)
    
    mt_str = m_data.mt.to_string(internal_labels=True)
    if mt_str.endswith(';'): mt_str = mt_str[:-1]
    
    mt_str = regex.sub(r'\1+', mt_str)
    mt_str = mt_str.replace("+*", "*")
    
    h_info = ""
    if not is_mul_input:
        h1_name = m_data.h1_node.name if m_data.h1_node else "NA"
        h2_name = m_data.h2_node.name if m_data.h2_node else "NA"
        h_info = f"\tH1 Node:{h1_name}\tH2 Node:{h2_name}"
    
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
                buffer.append(f"{m_idx}\t{g_idx}\t{total_groups}\t{num_fixed}\t{combos}\t{over_cap}\n")
                
        except Exception:
            # If pickle fails in worker, we might log or ignore? 
            # For now, append error to buffer so it appears in file
            buffer.append(f"# Error processing groups for MT-{m_idx}\n")
            
    return m_idx, "".join(buffer), local_failures

class GeneTreeManager:
    def __init__(self, config: TaskConfig, reconciler: Reconciler, logger: GranLogger):
        self.tcf = config
        self.reconciler = reconciler
        self.logger = logger
        
    def cull(self, mul_trees: Dict[int, MulTree], gene_trees: Dict[int, SmrtTree], registry: NameRegistry):
        if self.tcf.mode != "st-only":
            self.collapse_groups(mul_trees, gene_trees, registry)
        self.filter_and_check(mul_trees, gene_trees)
        self.write_filtered_trees(gene_trees)

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

        n_proc = self.reconciler.num_processes

        # --- CRITICAL FIX: PRIME REGISTRY ---
        # We must register all gene tree taxa in the main registry BEFORE forking.
        # Otherwise, workers create divergent ID mappings (e.g., 'TaxonA' is ID 5 in Worker 1, but ID 8 in Main).
        # We assume compute_groups uses registry.get_id(name). 
        # By iterating and 'touching' names here, we lock the IDs globally.
        if n_proc > 1:
            for gt in gene_trees.values():
                for node in gt.ete_tree.traverse():
                    if node.name:
                        registry.get_id(node.name)
        # This step may be costly for large GT sets - if we later fix the tips to follow an easier scheme, we can optimize this.
        # This is because the st will have all the species
        # ------------------------------------
        
        pickle_dir = self.tcf.pickle_dir
        pickle_dir.mkdir(parents=True, exist_ok=True)

        registry_path = pickle_dir / f"{self.tcf.run_prefix}_registry.pickle"
        force_regenerate = self._check_registry_safety(registry_path, registry)

        '''#for m_idx, m_data in mul_trees.items():
        # Disable if verbosity < 3
        for m_idx, m_data in tqdm(mul_trees.items(), desc="Processing MUL-trees", unit="mt", disable=self.logger.verbosity < 3):
            if m_idx == 0: continue
            
            pickle_path = pickle_dir / f"{self.tcf.run_prefix}_{m_idx}_groups.pickle"
            # Skip only if pickles exist AND we successfully restored the registry state
            if pickle_path.exists() and not self.tcf.overwrite and not force_regenerate:
                continue
                
            # Compute groups using the registry for speed
            current_mt_groups: Dict[int, GroupData] = {}
            
            # Pre-calculate sister clades (returned as Sets of Strings)
            h1_sis, h2_sis = self.reconciler.get_sister_clades(m_data)

            for g_idx, gt_obj in gene_trees.items():
                group_data = self.reconciler.compute_groups(gt_obj, m_data, registry, h1_sis, h2_sis)
                current_mt_groups[g_idx] = group_data
            
            try:
                with open(pickle_path, 'wb') as f:
                    pickle.dump(current_mt_groups, f)
            except Exception as e:
                self.logger.log(f"saving pickle {pickle_path}: {e}", 'e')
            
            del current_mt_groups'''
        
        # Prepare Tasks (skip 0)
        # Only process if pickle doesn't exist or forced
        tasks = []
        for m_idx, m_data in mul_trees.items():
            if m_idx == 0: continue
            
            pickle_path = pickle_dir / f"{self.tcf.run_prefix}_{m_idx}_groups.pickle"
            if pickle_path.exists() and not self.tcf.overwrite and not force_regenerate:
                continue
            tasks.append((m_idx, m_data))

        # Parallel Execution
        
        def save_result(res):
            m_idx, groups = res
            pickle_path = pickle_dir / f"{self.tcf.run_prefix}_{m_idx}_groups.pickle"
            try:
                with open(pickle_path, 'wb') as f:
                    pickle.dump(groups, f)
            except Exception as e:
                self.logger.log(f"saving pickle {pickle_path}: {e}", 'e')

        if tasks:
            if n_proc > 1:
                # Pass gene_trees and registry once via initializer to avoid repeated pickling overhead
                with mp.Pool(processes=n_proc, initializer=_init_collapse_worker, initargs=(gene_trees, registry, self.reconciler)) as pool:
                    for res in tqdm(pool.imap_unordered(_collapse_worker, tasks), total=len(tasks), desc="Collapsing", unit="mt", disable=self.logger.verbosity < 3):
                        save_result(res)
            else:
                _init_collapse_worker(gene_trees, registry, self.reconciler)
                for item in tqdm(tasks, desc="Collapsing", unit="mt", disable=self.logger.verbosity < 3):
                    res = _collapse_worker(item)
                    save_result(res)
            
        # Save the registry state so the next run can interpret the IDs in the group pickles
        try:
            with open(registry_path, 'wb') as f:
                pickle.dump(registry.get_state(), f)
        except Exception as e:
            self.logger.log(f"saving registry pickle: {e}", 'e')
            
        self.logger.report_step(step, "Success")

    def filter_and_check(self, mul_trees: Dict[int, MulTree], gene_trees: Dict[int, SmrtTree]):
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
                m_idx, 
                m_data, 
                str(pickle_path), 
                sorted_gene_ids, 
                self.tcf.group_cap, 
                self.tcf.is_mul_input
            ))

        n_proc = self.reconciler.num_processes
        
        # Results container: Map m_idx -> string buffer
        results_map = {}

        # Execution
        if tasks:
            if n_proc > 1:
                with mp.Pool(processes=n_proc) as pool:
                    for res in tqdm(pool.imap_unordered(_check_worker, tasks), total=len(tasks), desc="Checking  ", unit="mt", disable=self.logger.verbosity < 3):
                        m_idx, buf, fails = res
                        results_map[m_idx] = buf
                        # Merge failures
                        for g_idx, failures in fails.items():
                            if g_idx not in gt_failures: gt_failures[g_idx] = []
                            gt_failures[g_idx].extend(failures)
            else:
                for task in tqdm(tasks, desc="Checking  ", unit="mt", disable=self.logger.verbosity < 3):
                    m_idx, buf, fails = _check_worker(task)
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
                fail_count = len(gt_failures[g_idx])
                self.logger.log(f"Gene tree on line {g_idx+1} is over the group cap in {fail_count} MTs and will be filtered.", 'w')
                del gene_trees[g_idx]
                
    def filter_and_check2(self, mul_trees: Dict[int, MulTree], gene_trees: Dict[int, SmrtTree]):
        """
        Writes checknums file and filters trees exceeding group cap.
        Optimized regex formatting and pre-sorted iteration.
        """
        step = "Filtering gene trees over group cap"
        self.logger.report_step(step, "In progress...")
        
        check_path = Path(self.tcf.output_dir) / f"{self.tcf.run_prefix}-checknums.txt"
        
        gt_failures = {} 
        sorted_mul_ids = sorted(mul_trees.keys())
        sorted_gene_ids = sorted(gene_trees.keys())

        # OPTIMIZATION: Memoize regex patterns for formatting
        # Since h_clade is constant for a given MUL-tree, we compile the regex once per MT.
        mt_regex_cache = {}

        def format_mt_optimized(mt, h_clade, m_idx):
            s = mt.to_string(internal_labels=True)
            if s.endswith(';'): s = s[:-1]
            
            # Use cached compiled regex for this MUL-tree
            if m_idx not in mt_regex_cache:
                # Matches any species in h_clade that is NOT followed by a *
                # Pattern: \b(SpecA|SpecB|SpecC)(?!\*)
                pattern_str = r'\b(' + '|'.join(map(re.escape, h_clade)) + r')(?!\*)'
                mt_regex_cache[m_idx] = re.compile(pattern_str)
            
            # Single pass replacement using regex engine
            s = mt_regex_cache[m_idx].sub(r'\1+', s)
            s = s.replace("+*", "*")
            return s

        with open(check_path, 'w') as f:
            f.write("mul.tree\tgene.tree\tgroups\tfixed\tcombinations\tover.cap.filtered\n")
            
            for m_idx in sorted_mul_ids:
                if m_idx == 0: continue
                m_data = mul_trees[m_idx]
                
                # Use optimized formatter
                mt_str = format_mt_optimized(m_data.mt, m_data.h_clade, m_idx)
                
                h_info = ""
                if not self.tcf.is_mul_input:
                    h1_name = m_data.h1_node.name if m_data.h1_node else "NA"
                    h2_name = m_data.h2_node.name if m_data.h2_node else "NA"
                    h_info = f"\tH1 Node:{h1_name}\tH2 Node:{h2_name}"
                
                f.write(f"# MT-{m_idx}:{mt_str}{h_info}\n")

                pickle_path = self.tcf.pickle_dir / f"{self.tcf.run_prefix}_{m_idx}_groups.pickle"
                if not pickle_path.exists(): continue
                
                current_mt_groups = {}
                try:
                    with open(pickle_path, 'rb') as pf:
                        current_mt_groups = pickle.load(pf)
                except Exception:
                    continue
                
                buffer = []
                
                for g_idx in sorted_gene_ids:
                    if g_idx not in current_mt_groups: continue
                    
                    groups = current_mt_groups[g_idx]
                    num_ambig = len(groups.ambiguous_groups)
                    num_fixed = len(groups.fixed_groups)
                    total_groups = num_ambig + num_fixed
                    
                    over_cap = "N"
                    if num_ambig > self.tcf.group_cap:
                        over_cap = "Y"
                        if g_idx not in gt_failures: gt_failures[g_idx] = []
                        gt_failures[g_idx].append(m_idx)

                    combos = 1 << num_ambig 
                    buffer.append(f"{m_idx}\t{g_idx}\t{total_groups}\t{num_fixed}\t{combos}\t{over_cap}\n")
                
                if buffer:
                    f.write("".join(buffer))
                
                del current_mt_groups

        self.logger.report_step(step, f"Success: {len(gt_failures)} gts over cap")
        
        for g_idx in sorted(gt_failures.keys()):
            if g_idx in gene_trees:
                fail_count = len(gt_failures[g_idx])
                self.logger.log(f"Gene tree on line {g_idx} is over the group cap in {fail_count} MTs and will be filtered.", 'w')
                del gene_trees[g_idx]

    def write_filtered_trees(self, gene_trees: Dict[int, SmrtTree]):
        """Matches GRAMPA's filterOut logic."""
        step = "Writing filtered gene trees to file"
        self.logger.report_step(step, "In progress...")
        
        p = Path(self.tcf.output_dir) / f"{self.tcf.run_prefix}-trees-filtered.txt"
        count = 0
        with open(p, 'w') as f:
            for idx in sorted(gene_trees.keys()):
                # GRAMPA writes the original newick string
                f.write(gene_trees[idx].to_string(internal_labels=False) + "\n")
                count += 1
                
        self.logger.report_step(step, f"Success: {count} gene trees written")
