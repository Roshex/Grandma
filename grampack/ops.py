import os
import re
import pickle
import gzip, tarfile
from math import log2
from pathlib import Path
from collections import defaultdict
from shutil import rmtree, unpack_archive
from itertools import combinations
from functools import partial
from typing import Any, Collection, Dict, List, Optional, Set, Tuple, Union

from .config import TaskConfig
from .logger import GranLogger
from .models import Tree, SmrtTree, TreeCache, MulTree, NameRegistry, FlatTree, GroupData, splitSpec, ENGINE_RULE
from .core import compute_groups, compute_units
from .parallel import WorkPool


# Archiving: gzip level 1, not shutil's default.
# The group pickles are int-list dicts, so level 1 already should capture
#  nearly all the compression.
ARCHIVE_COMPRESSLEVEL = 2

MAX_LG_COMB = 24

def plan_dedup(gene_trees: Dict[int, FlatTree], threshold: float, enabled: bool=True, latched_off: bool=False,
               logger: Optional[GranLogger]=None, label: str=""):
    """
    Returns (rep_of, latch_off_downstream).

    Once the signature pass has run its cost is sunk, so the map is USED in the task
    that computed it whatever the duplicate fraction. `threshold` decides only whether
    the fraction is high enough to be worth re-paying in the tasks that inherit from
    this one: below it, descendants of a SERIAL task latch de-duplication off (their
    duplicate fraction cannot recover - relabelling only refines the leaf labelling).
    Split regimes always re-assess, because partitioning creates many duplicates.

    Hard off: clear bit 0 of --optim (enabled=False), or set the threshold to >= 1.0,
    which skips even the signature pass.
    """
    if not enabled or latched_off or not gene_trees or threshold >= 1.0:
        return {}, True

    first, rep_of = {}, {}
    for idx, gt in gene_trees.items():
        rep = first.setdefault(gt.canon_sig, idx)
        if rep != idx:
            rep_of[idx] = rep

    frac = len(rep_of) / len(gene_trees)
    latch = frac < threshold
    if logger is not None:
        tag = f"{label} " if label else ""
        kept = len(gene_trees) - len(rep_of)
        tail = (f" < {100*threshold:.1f}% threshold - used here, disabled "
                f"for subsequent iterations." if latch else ".")
        logger.log(f"{tag}Gene-tree de-duplication: {len(gene_trees)} trees -> {kept} "
                f"distinct labelled topologies ({100*frac:.1f}%){tail}", 'i')
    return rep_of, latch


class CommonOps:
    @staticmethod
    def _fix_semicolon(tree_str: str) -> str:
        """Ensures tree strings end with a semicolon."""
        tree_str = tree_str.strip()
        return tree_str if tree_str.endswith(';') else tree_str + ';'

    @staticmethod
    def export_tree_files(dir: Path, st: Tree=None, gts: Optional[List[Tree]]=None, suffix: str="") -> None:
        """Writes the trees to disk to allow inspection/resume, matching iter_mode.py."""
        if st:
            st_path = dir / f'multree{suffix}.tre'
            with open(st_path, 'w') as f:
                f.write(SmrtTree._to_str(st, internals=True) + '\n')
        if gts:
            gt_path = dir / f'genetrees{suffix}.txt'
            with open(gt_path, 'w') as f:
                for gt in gts: f.write(SmrtTree._to_str(gt, internals=True) + '\n')

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
            logger.log(f"{desc.capitalize()} file '{input}' not found.", key)
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
            logger.log(f"{desc.capitalize()} file '{input}' not found.", key)
        else:
            logger.log(f"Invalid input type for {desc}.", key)

    @staticmethod
    def _make_gztar(pickle_dir: Path, level: int = ARCHIVE_COMPRESSLEVEL) -> str:
        out = str(pickle_dir) + ".tar.gz"
        with gzip.GzipFile(out, 'wb', compresslevel=level) as gz:
            with tarfile.open(fileobj=gz, mode='w') as tf:
                tf.add(pickle_dir, arcname=pickle_dir.name)
        return out

    @staticmethod
    def _make_tar(pickle_dir: Path) -> str:
        out = str(pickle_dir) + ".tar"
        with tarfile.open(out, mode='w') as tf:
            tf.add(pickle_dir, arcname=pickle_dir.name)
        return out


class TreeLoader:
    """
    Handles loading, verification, and optional repais of Species and Gene trees.
    """
    _SANITIZE_TRANS = str.maketrans('_.', '--') # Replace any chars in '_.' with '-'

    @staticmethod
    def spec_tree(tcf: TaskConfig, logger: GranLogger, root_str: Optional[str] = None) -> Optional[SmrtTree]:

        if isinstance(tcf.st, SmrtTree):
            # Do nothing, already loaded
            step = "Loading species tree from memory"
            logger.report_step(step, "In progress...")
            logger.report_step(step, "Success: species tree loaded")
            return tcf.st

        repair = True if tcf.repair in ('fast', 'best') else False
        repair_str = " & repairing" if repair else ""
        step = f"Reading{repair_str} species tree"
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
            # But it also reads support vals as names!!!
            t = Tree(line, format=1)
        except Exception as e:
            logger.log(f"reading species tree file: {e}", 'e')

        if root_str and not TreeLoader.root_on_str(t, root_str):
            logger.log(f"Could not root species tree on provided root: '{root_str}' - skipping.", 'w')

        # Fixes & Checks
        # We error on ST issues unless repair is requested, as ST must be robust.
        is_valid, msg = TreeLoader._check_and_fix_names(t, repair, kind="st")
        if not is_valid:
            logger.log(f"Species tree invalid: {msg}", 'e')

        # Topology (Polytomies, Rooting)
        is_valid, msg = TreeLoader._check_and_fix_topology(t, repair)
        if not is_valid:
            logger.log(f"Species tree invalid: {msg}", 'e')

        logger.report_step(step, "Success: species tree read")
        st_wrapper = SmrtTree(tree_obj=t)

        st_wrapper.assert_len

        if repair or root_str:
            CommonOps.export_tree_files(tcf.output_dir, st=st_wrapper.ete_tree, suffix="_repaired")

        if tcf.mode == "label-sp":
            logger.log(f"The input species tree with internal nodes labeled:", 'i')
            logger.log(f"{st_wrapper.to_str(internals=True)}", 'i', prefix='')
            logger.log(f"Label-sp Mode terminates before parsing the species tree: exiting...", 'i')
            return None
        return st_wrapper

    @staticmethod
    def gene_trees(tcf: TaskConfig, logger: GranLogger, ref: SmrtTree,
                registry: Optional[NameRegistry] = None) -> Optional[Dict[int, SmrtTree]]:

        if isinstance(tcf.gts, dict):
            step = "Loading gene trees from memory"
            logger.report_step(step, "In progress...")

            # --- Q block ---
            quotas_path = tcf.output_dir / f"{tcf.run_prefix}-gt-quotas.tsv"
            TreeLoader.proc_harmonic_quota(tcf.quota_gts, tcf.gts, ref, logger, quotas_path)

            logger.report_step(step, f"Success: {len(tcf.gts)} gene trees loaded")
            return tcf.gts
        
        if tcf.repair in ('fast', 'best'):
            repair = True
            name_fixer = partial(TreeLoader._check_and_fix_names, repair=True, ref=ref)
            if tcf.repair == 'best':
                topo_fixer = partial(TreeLoader._check_and_fix_topology, repair=True, ref=ref, weights=tcf.weights, registry=registry)
            else:
                topo_fixer = partial(TreeLoader._check_and_fix_topology, repair=True)
        else:
            repair = False
            name_fixer = partial(TreeLoader._check_and_fix_names, repair=False, ref=ref)
            topo_fixer = partial(TreeLoader._check_and_fix_topology, repair=False)
        repair_str = " & repairing" if repair else ""
        step = f"Reading{repair_str} gene trees"
        logger.report_step(step, "In progress...")

        # Input Validation
        if tcf.gts is None:
            if tcf.mode in ('build-mts', 'count-mts', 'label-sp', 'repair'):
                logger.report_step(step, f"Skipped: '{tcf.mode}' mode")
                return None if tcf.mode == 'repair' else {}
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
                # Format 0 allows for support values (and branch lengths) - no internal names allowed for gts
                gt = Tree(line, format=0)
            except Exception:
                logger.log(f"Error reading tree {origin}! -- Filtering.", 'w')
                continue

            # Label Repair/Check
            is_valid, msg = name_fixer(gt)
            if not is_valid:
                logger.log(f"Gene tree {origin}: {msg} -- Filtering.", 'w')
                continue

            # Topology Repair/Check
            is_valid, msg = topo_fixer(gt)
            if not is_valid:
                logger.log(f"Gene tree {origin}: {msg} -- Filtering.", 'w')
                continue

            valid_gts.append(gt)

        if len(valid_gts) == 0:
            logger.log(f"No valid gene trees survived filtering (required in {tcf.mode} mode).", 'e')

        # Make sure to index first
        gt_dict = {idx: SmrtTree(tree_obj=gt) for idx, gt in enumerate(valid_gts)}

        # Most important validation
        for _, gt in gt_dict.items():
            gt.assert_len

        # --- Q block ---
        quotas_path = tcf.output_dir / f"{tcf.run_prefix}-gt-quotas.tsv"
        TreeLoader.proc_harmonic_quota(tcf.quota_gts, gt_dict, ref, logger, quotas_path)

        if repair:
            CommonOps.export_tree_files(tcf.output_dir, gts=valid_gts, suffix="_repaired")
                
        logger.report_step(step, f"Success: {len(valid_gts)} gene trees read")
        if tcf.mode == "repair":
            logger.log("Repair Mode finished successfully: repaired trees have been exported. Exiting...", 'i')
            return None

        return gt_dict

    @staticmethod
    def proc_harmonic_quota(
            quota_gts: str, gt_dict: Dict[int, SmrtTree], ref: Optional[SmrtTree], logger: GranLogger, quotas_path: Path
    ) -> None:

        if quota_gts == 'harmonic' and ref is not None:
            # Extract total unique species directly from the ST
            total_species = len(set(n.pure for n in ref.ete_tree.iter_leaves()))
            
            debug_file = None
            if logger.debug:
                try:
                    debug_file = open(quotas_path, 'w')
                    debug_file.write("gt_idx\tO\tR\tQ\n")
                except Exception as e:
                    logger.log(f"Could not open debug file for GT quotas: {e}", 'w')

            all_R_are_one = True
            for g_idx, gt in gt_dict.items():
                O, R, Q, has_support = gt.calculate_Q(total_species)
                if debug_file:
                    debug_file.write(f"{g_idx+1}\t{O:.4f}\t{R:.4f}\t{Q:.4f}\n")
                    if not has_support:
                        logger.log(f"GT-{g_idx+1} has no support values", 'w')
                if R != 1.0:
                    all_R_are_one = False
            if all_R_are_one:
                logger.log("All GTs have R=1.0, indicating no support values or all support values are 1.0. Consider disabling quotas ('-q equal').", 'w')
            
            if debug_file:
                debug_file.close()

    # --- Structural Repair Algorithms ---

    @staticmethod
    def remove_knuckles(t: Tree) -> None:
        """Removes unary internal nodes to ensure pure bifurcations."""
        def transfer_props_before_removal(n: Tree) -> None:
            """Transfers properties from n1 to n2 if n2 is empty."""
            p = n.up
            if n.name and not p.name:
                p.name = n.name
            if n.dist not in (None, 0.0) and p.dist not in (None, 0.0):
                p.dist += n.dist
            if n.support and p.support:
                p.support = min(n.support, p.support)

        for n in list(t.traverse("postorder")):
            if not n.is_leaf() and len(n.children) == 1 and not n.is_root():
                # If has name, transfer to the parent (if has no name)
                transfer_props_before_removal(n)
                n.delete(prevent_nondicotomic=False)
        
        # Safely resolve root knuckle if present
        if len(t.children) == 1:
            child = t.children[0]
            transfer_props_before_removal(child)
            child.detach()
            for c in list(child.children):
                t.add_child(c.detach())
    
    @staticmethod
    def root_on_str(t: Tree, root: str) -> bool:
        """Attempts to root on a candidate name, returns success."""
        if ',' in root:
            clade = root.split(',')
            clade = [t.search_nodes(name=c.strip())[0] for c in clade if t.search_nodes(name=c.strip())]
            if not clade:
                return False
            lca = t.get_common_ancestor(clade)
            t.set_outgroup(lca)
            return True
        else:
            clade = t.search_nodes(name=root.strip())
            if clade:
                t.set_outgroup(clade[0])
                return True
        print("clade", clade, root.strip(), t.write(format=1)) ##
        return False

    @staticmethod
    def _check_and_fix_topology(t: Tree, repair: bool,
                                   ref: Optional[SmrtTree] = None,
                                   weights: Optional[Tuple[int, int]] = None,
                                   registry: Optional[NameRegistry] = None) -> Tuple[bool, str]:
        """
        Checks for polytomies and unrooted-ness.
        """
        # Standardize by removing knuckles
        TreeLoader.remove_knuckles(t)

        # Rooting
        # ETE3 logic: unrooted trees often loaded as rooted with trifurcation at top.
        # If we just resolved polytomies, we might have arbitrarily binary-ized the root.
        if len(t.children) > 2:
            if repair:
                if ref is not None:
                    print(t.write(format=0)) ##
                    TreeLoader.root_by_recon(t, ref, weights, registry)
                    print(t.write(format=0)) ##
                else:
                    t.resolve_polytomy(recursive=False)
            else:
                return False, "Tree root is not rooted"

        # Polytomies
        has_polytomies = any(len(n.children) > 2 for n in t.traverse())
        if has_polytomies:
            if repair:
                if ref is not None:
                    print(t.write(format=0)) ##
                    TreeLoader.resolve_polytomies_by_recon(t, ref, weights, registry)
                    print(t.write(format=0)) #t.get_ascii(show_internal=True)) ##
                else:
                    t.resolve_polytomy(recursive=True)
            else:
                return False, "Tree contains non-bifurcating nodes"

        return True, ""

    @staticmethod
    def _score_topology(test_gt: Tree, st_wrapper: SmrtTree, dup_cost: int, loss_cost: int, registry: NameRegistry) -> int:
        """
        Leverages the highly optimized FlatTree engine in reconcile.py to score a topology.
        MATHEMATICAL SAFEGUARD: Because recon_lca_optimized strictly assumes bifurcating trees,
        we must temporarily binarize any remaining polytomies (e.g., ancestors when resolving bottom-up)
        so the scoring engine doesn't silently ignore 3rd+ children and drop lineages.
        """
        from .core import PairwiseRecon
        from .models import SmrtTree
        
        # Flatten the species tree once if it hasn't been already
        if not st_wrapper.flat_tree:
            st_wrapper.make_flat(registry)
            
        # 1. Create a sandbox copy to avoid modifying the actual permutation we are testing
        temp_gt = test_gt.copy(method="cpickle")
        
        # 2. Force strictly bifurcating structure for the FlatTree array math
        temp_gt.resolve_polytomy(recursive=True)
        TreeLoader.remove_knuckles(temp_gt)
        
        # 3. Wrap, flatten, and score safely
        temp_wrapper = SmrtTree(tree_obj=temp_gt)
        temp_wrapper.make_flat(registry)
        
        return PairwiseRecon(dup_cost, loss_cost, True).reconcile_sl(temp_wrapper.flat_tree, st_wrapper.flat_tree, registry=registry)[0]

    @staticmethod
    def root_by_recon(gt: Tree, st_wrapper: SmrtTree, weights: Tuple[int, int], registry: NameRegistry) -> None:
        """Notung Algorithm: Reroots unrooted GT by testing every edge and picking minimum D/L score."""
        best_score = float('inf')
        best_edge_node_id = None
        
        # Tag original nodes to track them across deep copies
        for i, n in enumerate(gt.traverse()):
            n.add_feature("temp_id", i)
            
        dup_cost, loss_cost = weights
        root = gt.get_tree_root()
        for edge_target in gt.traverse():
            if edge_target == root: continue
            
            test_gt = gt.copy()
            target_in_test = test_gt.search_nodes(temp_id=edge_target.temp_id)[0]
            
            # Reroot and clean
            test_gt.set_outgroup(target_in_test)
            TreeLoader.remove_knuckles(test_gt)
            
            # Score using reconcile.py
            score = TreeLoader._score_topology(test_gt, st_wrapper, dup_cost, loss_cost, registry)
            
            if score < best_score:
                best_score = score
                best_edge_node_id = edge_target.temp_id
                
        # Apply the optimal root
        if best_edge_node_id is not None:
            best_target = gt.search_nodes(temp_id=best_edge_node_id)[0]
            gt.set_outgroup(best_target)
            TreeLoader.remove_knuckles(gt)
            
        for n in gt.traverse():
            if hasattr(n, "temp_id"):
                n.del_feature("temp_id")

    @staticmethod
    def resolve_polytomies_by_recon(gt: Tree, st_wrapper: SmrtTree, weights: Tuple[int, int], registry: NameRegistry, polytomy_size_limit: int = 6) -> None:
        """
        Notung Algorithm: Resolves polytomies by building all binary topologies and picking min D/L.
        Because the number of binary resolutions grows super-exponentially with polytomy size, we solve one polytomy at a time.
        The polytomies are solved in postorder to ensure score remains relativelt invariant to the resolution of polytomies upstream.
        Polytomy formula: (2k-3)!!
        E.g., for k=3: 3, k=4: 15, k=5: 105, k=6: 945, k=7: 10395, k=8: 135135 ...
        Hence, we put a safeguard cap at polytomy_size_limit to avoid combinatorial explosion, and simply resolve arbitrarily beyond that.
        """
        def build_balanced_tree(nodes: List[Tree]) -> Tree:
            """Helper to arbitrarily resolve paralogs/in-paralogs into a clean binary tree."""
            if not nodes: return None
            if len(nodes) == 1: return nodes[0]
            while len(nodes) > 1:
                nxt = []
                for i in range(0, len(nodes), 2):
                    if i + 1 < len(nodes):
                        parent = Tree(name="<R>")
                        parent.add_child(nodes[i])
                        parent.add_child(nodes[i+1])
                        nxt.append(parent)
                    else:
                        nxt.append(nodes[i])
                nodes = nxt
            return nodes[0]

        def resolve_large_polytomy(poly_node: Tree) -> None:
            """Species-Tree-Guided bottom-up heuristic for large polytomies."""
            from .core import PairwiseRecon
            st_bins = defaultdict(list)

            pr = PairwiseRecon(dup_cost, loss_cost, True)
            
            # 1. Map each child to its ST LCA using exact Reconciliation (Flawless MUL-tree handling)
            for c in list(poly_node.children):
                c_detached = c.detach()
                
                # Assign a temp name to the root of this clade so we can reliably fetch its mapping
                original_name = c_detached.name
                temp_name = original_name if original_name else f"<TR_{id(c_detached)}>"
                c_detached.name = temp_name
                
                # Flatten the standalone clade
                c_wrapper = SmrtTree(tree_obj=c_detached)
                c_wrapper.make_flat(registry)
                
                st_lca = None

                _score, maps = pr.reconcile_sl(
                    c_wrapper.flat_tree, st_wrapper.flat_tree,
                    registry=registry, retmap=True)
                if maps:
                    mapped_st_name = maps[0].cor[temp_name][0]
                    st_lca = st_wrapper.get_node(mapped_st_name)
                
                # Restore original name
                if not original_name:
                    c_detached.name = ""
                    
                if not st_lca:
                    st_lca = st_wrapper.ete_tree # Fallback to ST root if heavily pruned
                    
                st_bins[st_lca].append(c_detached)
                
            # 2. Bottom-up postorder merge along the ST topology
            accumulated = defaultdict(list)
            for st_node in st_wrapper.ete_tree.traverse("postorder"):
                child_gts = []
                for st_child in st_node.children:
                    if accumulated[st_child]:
                        merged = build_balanced_tree(accumulated[st_child])
                        if merged: child_gts.append(merged)
                        
                if len(child_gts) > 1:
                    accumulated[st_node].append(build_balanced_tree(child_gts))
                elif len(child_gts) == 1:
                    accumulated[st_node].append(child_gts[0])
                    
                if st_node in st_bins:
                    accumulated[st_node].extend(st_bins[st_node])
                    
            # 3. Apply the final resolved binary structure
            final_nodes = accumulated[st_wrapper.ete_tree]
            resolved_root = build_balanced_tree(final_nodes)
            
            if resolved_root:
                if resolved_root.name == "<R>":
                    for c in list(resolved_root.children):
                        poly_node.add_child(c.detach())
                else:
                    poly_node.add_child(resolved_root.detach())

        def get_rooted_topologies(items):
            if len(items) == 1: return [items[0]]
            if len(items) == 2: return [(items[0], items[1])]
            res = []
            first = items[0]
            rest = items[1:]
            for i in range(len(rest)):
                for left_rest in combinations(rest, i):
                    left_set = [first] + list(left_rest)
                    right_set = [x for x in rest if x not in left_set]
                    if not right_set: continue
                    for lt in get_rooted_topologies(left_set):
                        for rt in get_rooted_topologies(right_set):
                            res.append((lt, rt))
            return res

        def apply_topology(node, topology):
            if not isinstance(topology, tuple):
                node.add_child(topology)
                return
            left_node = Tree(name="<L>")
            right_node = Tree(name="<R>")
            node.add_child(left_node)
            node.add_child(right_node)
            apply_topology(left_node, topology[0])
            apply_topology(right_node, topology[1])

        dup_cost, loss_cost = weights
        has_poly = True
        while has_poly:
            has_poly = False
            poly_node = None
            for n in gt.traverse("postorder"):
                if len(n.children) > 2:
                    poly_node = n
                    break
                    
            if poly_node:
                has_poly = True

                # Remeber to iterating over a static copy
                children = list(poly_node.children)

                # Safeguard: (2*polytomy_size_limit-3)!! possibilities. Fallback to arbitrary resolution.
                if len(children) > polytomy_size_limit:
                    resolve_large_polytomy(poly_node)
                    # poly_node.resolve_polytomy(recursive=False)
                    continue
                    
                best_score = float('inf')
                best_topology = None
                topologies = get_rooted_topologies(children)

                print(f"Resolving polytomy with {len(children)} children and {len(topologies)} topologies to evaluate...") ##
                scores = []
                
                for topo in topologies:
                    # Apply permutation
                    for c in list(poly_node.children): c.detach()
                    apply_topology(poly_node, topo)
                    
                    # Score using reconcile.py
                    #print(gt.write(format=0)) ##
                    #print(st_wrapper.to_str(internals=True)) ##
                    #print(poly_node.get_ascii(show_internal=True)) ##
                    score = TreeLoader._score_topology(gt, st_wrapper, dup_cost, loss_cost, registry)
                    scores.append(score)
                    
                    if score < best_score:
                        best_score = score
                        best_topology = topo
                        
                    # Revert permutation
                    for c in list(poly_node.children): c.detach()
                    for c in children: poly_node.add_child(c)

                print(f"Best score for this polytomy: {best_score}") ##
                print(f"Scores for this polytomy: {scores}") ##
                        
                # Apply optimal topology
                if best_topology:
                    for c in list(poly_node.children): c.detach()
                    apply_topology(poly_node, best_topology)

    # --- Specialized MUL-tree processing for iterative search ---

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

    # --- Name Parsing and Sanitization ---

    @staticmethod
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    @staticmethod
    def _sanitize_line(s: str) -> str:
        # Remove everything between "'[" and "]'" (Astral cleaning)
        return re.sub(r"'\[.*?\]'", '', s)
    
    @staticmethod
    def _check_and_fix_names(t: Tree, repair: bool,
                             ref: Optional[SmrtTree] = None, kind: str = "gt") -> Tuple[bool, str]:

        table = str.maketrans('|', '.', '<>')
        illegal_chars = set('|<>')

        # ST-specific:
        if kind == "st":
            seen_st_nodes = set()
            for node in t.traverse():
                if node.is_leaf():
                    if not node.name:
                        return False, "Species tree contains empty leaf names"

                    if not repair:
                        if any(c in node.name for c in illegal_chars):
                            return False, f"Leaf name '{node.name}' contains illegal characters '|', '<', or '>'"
                        
                        # Uniqueness strictly enforced
                        if node.name in seen_st_nodes:
                            return False, f"Species tree contains duplicate leaf name: '{node.name}'"
                        seen_st_nodes.add(node.name)
                    else:
                        # Translate FIRST, then check uniqueness
                        node.name = node.name.translate(table)
                        if node.name in seen_st_nodes:
                            return False, f"Species tree contains duplicate leaf name (after repair): '{node.name}'"
                        seen_st_nodes.add(node.name)                        
                else:
                    # INTERNAL NODES
                    if node.name:
                        # Branch support read as internal node name, which is not allowed (e.g. '(A:0.1,B:0.2)0.5:0.3;')
                        if TreeLoader.is_number(node.name):
                            node.name = None
                            continue

                        if not repair:
                            if not (node.name.startswith("<") and node.name.endswith(">")):
                                return False, f"Internal node names must be enclosed in angle brackets (e.g. '<Node1>'), but found '{node.name}'"
                            internal_ = node.name[1:-1]
                            if any(c in internal_ for c in illegal_chars) or internal_.startswith('P*'):
                                return False, f"Internal node name '{node.name}' contains illegal characters inside brackets"
                                
                            # Uniqueness strictly enforced
                            if node.name in seen_st_nodes:
                                return False, f"Species tree contains duplicate internal name: '{node.name}'"
                            seen_st_nodes.add(node.name)
                        else:
                            # Format and translate FIRST
                            if not node.name.startswith("<"):
                                node.name = "<" + node.name
                            if not node.name.endswith(">"):
                                node.name = node.name + ">"
                            translated = node.name[1:-1].translate(table)
                            if translated.startswith('P'):
                                translated = translated.replace('P', 'p').replace('*', '.') # Additional safeguard against P* prefix after translation
                            new_name = f"<{translated}>"
                            
                            # Check for collisions safely
                            if new_name in seen_st_nodes:
                                node.name = None # Safely drop duplicate internal names
                            else:
                                node.name = new_name
                                seen_st_nodes.add(node.name)
            return True, ""

        # GT-specific:
        # Don't care about internal nodes
        
        st_leaf_names = set(ref.ete_tree.iter_leaf_names()) if ref else set()

        seen_names = set()
        base_counts = {}
        to_prune = set()
        
        for node in t.traverse():
            if not node.is_leaf():
                node.name = None # Clear internal nodes safely
                continue
                
            if not node.name:
                to_prune.add(node)
                continue

            if not repair:
                # Check Illegal Characters
                if any(c in node.name for c in illegal_chars):
                    return False, f"Leaf name '{node.name}' contains illegal characters '|', '<', or '>'"
                    
                # Check ST Membership
                if node.name not in st_leaf_names and splitSpec(node.name) not in st_leaf_names:
                    return False, f"Taxon '{node.name}' or '{splitSpec(node.name)}' not found in species tree"
                
                # Check Uniqueness (using a properly tracking set)
                if node.name in seen_names:
                    return False, f"Leaf name '{node.name}' is not unique"
                seen_names.add(node.name)
            else:
                # Sanitize raw characters
                clean_name = node.name.translate(table)
                
                spec_id = None
                gene_id = ""
                
                # Extract Species ID and Gene ID
                # Case A: Entire name is an exact match to an ST leaf (missing gene ID)
                if clean_name in st_leaf_names:
                    spec_id = clean_name
                else:
                    # Case B: Rely on configured splitSpec to find the ST leaf
                    parsed_spec = splitSpec(clean_name)
                    if parsed_spec in st_leaf_names:
                        spec_id = parsed_spec
                        # Extract the gene_id prefix by removing the spec_id and delimiter
                        if clean_name.endswith("_" + spec_id):
                            gene_id = clean_name[:-(len(spec_id) + 1)]
                    else:
                        to_prune.add(node)
                        continue
                
                # Format Gene ID to immunize against splitSpec parsing variations
                # By converting internal '_' to '.', we guarantee that gene_id_spec_id 
                # behaves identically whether split by first '_' or last '_'
                gene_id = gene_id.replace("_", ".")
                if not gene_id:
                    gene_id = ""
                    
                # Reconstruct and Uniquify
                base_new_name = f"{gene_id}_{spec_id}"
                
                # Fast O(1) counter lookup
                count = base_counts.get(base_new_name, 0)

                infix = '.' if gene_id and gene_id[-1].isnumeric() else ''
                
                if count == 0 and base_new_name not in seen_names and not base_new_name.startswith('_'):
                    new_name = base_new_name
                    base_counts[base_new_name] = 0
                else:
                    # Jump ahead using the dictionary to skip the while loop overhead
                    count = max(0, count) 
                    while True:
                        count += 1
                        new_name = f"{gene_id}{infix}{count}_{spec_id}"
                        # The set protects against natural name collisions
                        if new_name not in seen_names:
                            break
                    base_counts[base_new_name] = count
                
                node.name = new_name
                seen_names.add(new_name)

        # If repairing, prune any leaves that couldn't be matched to the ST after sanitization and counting
        if to_prune:
            if repair:
                # Safely collect only valid leaves for ETE3 pruning
                valid_leaves = [n for n in t.iter_leaves() if n not in to_prune]
                if len(valid_leaves) >= 2:
                    try:
                        t.prune(valid_leaves, preserve_branch_length=True)
                    except Exception:
                        return False, "Pruning left tree in invalid state: cannot repair"
                else:
                    return False, "No taxa are in the species tree: cannot repair"
            else:
                return False, "Contains taxa not in Species Tree or has empty names"

        return True, ""

    @staticmethod
    def _sanitize_tip(s: str) -> str:
        return s.translate(TreeLoader._SANITIZE_TRANS)
    

class MulTreeManager:
    __slots__ = ['tcf', 'st', 'logger', 'ploidies']

    def __init__(self, config: TaskConfig, st: SmrtTree, logger: GranLogger) -> None:
        self.tcf = config
        self.st = st
        self.logger = logger
        self.ploidies: Dict[str, int] = self._parse_ploidy_file(self.tcf.ploidies, st, logger)

    @staticmethod
    def _parse_ploidy_file(ploidies: Optional[Union[Path, str, Dict[str, int]]], st: SmrtTree, logger: GranLogger) -> Dict[str, int]:
        # No need to reload
        if isinstance(ploidies, dict):
            return ploidies
        if not ploidies:
            return {}
        step = "Reading ploidy file"
        logger.report_step(step, "In progress...")
        ploidy_content = CommonOps._load_single_content(ploidies, "ploidies", logger, key="e")
        ploidy_dict: Dict[str, int] = {}
        allowed_species = st.node_map.keys()
        for line in ploidy_content.splitlines():
            parts = line.strip().split()
            # Allow comments and skip empty lines
            if parts[0].startswith('#') or not parts:
                continue
            if len(parts) != 2:
                logger.log(f"Invalid ploidy file line format: '{line}'. Expected 'species_name positive_int_ploidy', i.e., singly whitespace separated. Skipping.", 'w')
                continue
            species, ploidy = parts
            if species not in allowed_species:
                logger.log(f"Species '{species}' in ploidy file not found in species tree. CHECK THE INPUT FILES. Skipping.", 'w')
                continue
            if species in ploidy_dict:
                logger.log(f"Duplicate entry for species '{species}' in ploidy file. CHECK THE INPUT FILES. Skipping.", 'w')
                continue
            try:
                x = float(ploidy)
                assert x.is_integer() and x > 0
            except (ValueError, AssertionError):
                logger.log(f"Invalid ploidy value for species '{species}': '{ploidy}'. Expected a positive integer. Skipping.", 'w')
                continue
            ploidy_dict[species] = int(x)
        if not ploidy_dict:
            logger.log("Ploidy file is empty or invalid.", 'w')
        logger.report_step(step, f"Success: Loaded ploidies for {len(ploidy_dict)} species")
        return ploidy_dict
        
    def _apply_ploidy_constraints(self, h1_candidates: List[str], is_strict: bool) -> Tuple[List[str], Dict[str, float]]:
        """
        Filters H1 candidates and calculates how many NEW copies each can tolerate.
        Centralizes all ploidy math for Simple, Full, Split, and Mixed modes.
        Contains both the complex 'effective lineage' logic and the strict 'exact match count' logic.
        Only H1 is filtered because H1 is the lineage being duplicated, but the allowance is calculated for H2/x for when grafting.
        """
        filtered_h1: List[str] = []
        h1_allowances: Dict[str, float] = {}

        # --- Prepare Tree Cahces and Calculate the Gluing Multiplier (Inner Case) ---
        multiplier = 1
        if self.tcf.global_tree_cache is not None:
            tree_cache = self.tcf.global_tree_cache
            try:
                # Ask the global tree how many times our local root appears. 
                # This is EXACTLY how many times our local grafts will be duplicated during gluing!
                local_root_pure = self.st.ete_tree.pure
                multiplier = len(tree_cache.st.match(local_root_pure))
            except Exception:
                pass
            if multiplier == 0: multiplier = 1 # Safety fallback
        else:
            tree_cache = TreeCache(self.st)

        # Settings are the same for all subproblems of a given depth
        tree_cache.populate(self.ploidies, is_strict)

        target_st = tree_cache.st
        ploidy_stats = tree_cache.ploidy_stats
        clade_counts = tree_cache.clade_counts

        rejected_clades: Set[str] = set() 

        for node_name in h1_candidates:

            node = target_st.get_node(node_name)
            if node is None:
                self.logger.log(f"H1 candidate '{node_name}' exists in a local subproblem but not found in the Global ST cache.", 'e')
            
            clade_species = clade_counts.get(node_name)

            if not node.is_leaf():
                if any(child.name in rejected_clades for child in node.children):
                    rejected_clades.add(node_name)
                    self.logger.log(f"Skipping H1 candidate '{node_name}': ancestor of excluded clade(s).", 'd')
                    continue

            min_allowance_grafts = float('inf')
            
            for pure_sp, sp_copies_in_clade in clade_species.items():
                limit = self.ploidies.get(pure_sp, 999) # Default to infinite if not in file           
                current_count, max_group_size = ploidy_stats.get(pure_sp, (0,0))
            
                # How many total new copies of this species are we allowed to add?
                available = limit - current_count
                
                # How many GLOBAL copies does 1 local graft create?
                global_cost_per_graft = sp_copies_in_clade * multiplier

                # How many GRAFT EVENTS of this clade does that translate to?
                # (Integer division normalizes the allowance by the cost of the graft)
                allowed_grafts = available // global_cost_per_graft
                
                if allowed_grafts < min_allowance_grafts:
                    min_allowance_grafts = allowed_grafts
            
            # If we are allowed to graft this clade at least 1 time, keep it.
            if min_allowance_grafts >= 1:
                filtered_h1.append(node_name)
                # Pass the NORMALIZED graft count directly to build()
                h1_allowances[node_name] = float(min_allowance_grafts)

                self.logger.log(f"Including H1 candidate '{node_name}' with species {clade_species}. Max allowed grafts: {min_allowance_grafts}", 'd')
            
            else:
                rejected_clades.add(node_name)
                self.logger.log(f"Excluding H1 candidate '{node_name}' with species {clade_species} due to insufficient allowance.", 'd')

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
            
            # Mark as seen regardless of ploidy / redundancy check results - to block the entire subtree
            if nesting == 'model': processed_targets.add(h2_node.pure)
            
            # Ploidy check and redundancy blocking
            if is_redundant or len(matches) > allowance:
                continue
                    
            valid_match_groups.append(matches)
            
        return valid_match_groups

    def index(self,
              nesting: str='model',
              strict_constraint: bool=False,
              allow_redundant_mts: bool=False,
              sweep_mode: bool=False) -> Tuple[Dict[int, MulTree], List[str], List[str], Dict[str, int]]:
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
            h1_resolved, h1_allowances = self._apply_ploidy_constraints(h1_resolved_original, strict_constraint)
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
        
        step = "Counting MUL-trees" if self.tcf.mode == "count-mts" or sweep_mode else "Counting MUL-trees to generate"
        self.logger.report_step(step, "In progress...")

        multi: List[int] = []
        single: List[int] = []
        mt_meta: Dict[int, Tuple[str, List]] = {}
        idx = 1 # idx 0 reserved for species tree!

        for h1 in h1_resolved:
            h1_st_node = self.st.get_node(h1)
            n1_pure_descendants = pure_desc_cache.get(h1_st_node, set())
            
            # Get the definitive list of valid target groupings
            # _compile returns valid_pairings, a list of matches, for the given H1
            for matches in self._compile_h2_targets(h1_st_node, h2_resolved, nesting, n1_pure_descendants, h1_allowances[h1], allow_redundant_mts):
                mt_meta[idx] = (h1, matches)
                (single if len(matches) == 1 else multi).append(idx)
                idx += 1

        num_mul_trees = len(mt_meta) # already excludes the species tree (index 0)

        self.logger.report_step(step, f"Success: {len(single)} single- & {len(multi)} multi-MTs"
                                if len(multi) else f"Success: {num_mul_trees} single MUL-trees")

        if self.tcf.mode == "count-mts":
            self.report_mt_count(self.st.ete_tree, h1_resolved_original, h2_resolved, num_mul_trees, nesting, bool(self.ploidies), allow_redundant_mts)
            return {}, [], [], {} # Returns empty dict to signal main.py to exit

        if sweep_mode:
            self.logger.log("Sweep mode enabled: skipping MUL-tree building step.", 'i')
            return (mt_meta, single, multi), h1_resolved, h2_resolved, self.ploidies

        # --- BUILDING STEP ---
        self.build(mt_meta, mul_trees)

        if self.tcf.mode == "build-mts":
            self.report_mt_build(mul_trees, nesting)
            return {}, [], [], {} # Returns empty dict to signal main.py to exit

        if not mul_trees:
            self.logger.log("No valid MUL-trees could be generated with the given constraints.", 'w')
        if len(mul_trees) < (1 if self.tcf.mode in {"no-st", "st-only"} else 2):
            self.logger.log("Too few MUL-trees built. Check your H1/H2 and ploidy constraints.", 'w')
            
        return mul_trees, h1_resolved, h2_resolved, self.ploidies

    def build(self,
              mt_meta: Dict[int, Tuple[str, List]],
              mul_trees: Optional[Dict[int, MulTree]] = None,
              keep: Optional[Collection[int]] = None,
              label: str = ''
              ) -> Dict[int, MulTree]:
        """
        Builds MUL-trees from the provided index metadata.
        Metadata indexes must not be already present in mul_trees to avoid overwriting!
        If mul_trees is provided, it will be updated in-place; otherwise, a new dictionary will be created.

        keep=None: build all
        keep=collection[int]: build only the specified indices
        keep=empty_collection: build none (used for counting only)
        """
        step = f"Building {label}MUL-trees"
        self.logger.report_step(step, "In progress...")

        counter = 0
        if mul_trees is None: mul_trees = {}
        for idx, (h1, matches) in mt_meta.items():
            if keep is None or idx in keep:

                assert idx not in mul_trees, f"Index {idx} already exists in mul_trees. This should not happen."

                h1_st_node = self.st.get_node(h1)
                h_clade = h1_st_node.get_leaf_names()
                # to_multi_mul_tree handles BOTH Simple and Model modes seamlessly!
                # (If simple, matches just has 1 item)
                mt_wrapper, h1_obj, hx_objs = self.st.to_multi_mul_tree(h1, sorted([n.name for n in matches]))

                if mt_wrapper:
                    mul_trees[idx] = MulTree(mt_wrapper, h_clade, h1_obj, hx_nodes=hx_objs)
                    counter += 1
        
        self.logger.report_step(step, f"Success: {counter} {label}MUL-trees built")
        return mul_trees

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

def log2_combinations(n_amb: int, n_amb_multi: int, c_max: int = 2,
                      use_exact: bool = False) -> float:
    """
    log2 of the number of allele assignments a gene tree contributes for one MUL-tree.
 
    n_amb        ambiguous (free) units; fixed/pinned units have one state, contributing 0
    n_amb_multi  of those, how many have more than one leaf (these gain the mixed state)
    c_max        copies of the most-duplicated species in this MUL-tree
    use_exact    whether multi-leaf units carry the extra mixed state
 
    For heterogeneous multi-MTs, c_max upper-bounds each unit's own c, so this
    over-estimates log2 P and filters a SUPERSET - conservative, never under-filtering.
    At c_max = 2 with use_exact = False it returns exactly n_amb, i.e. GRAMPA's rule, so
    the default pathway and GRAMPA parity are unchanged.
    """
    if n_amb <= 0:
        return 0.0
    if not use_exact:
        return n_amb * log2(c_max)
    return (n_amb - n_amb_multi) * log2(c_max) + n_amb_multi * log2(2 * c_max - 1)

def check_workload(work: float) -> None:
    # Tractability, on the enumeration that will ACTUALLY run. Raising rather than
    # filtering keeps the dataset fixed: a run either analyses every gene tree that
    # passed the cap, or it stops and says why.
    if work > MAX_LG_COMB:
        raise RuntimeError(
            f"2^{work:.1f} allele assignments for this gene tree are above 'MAX_LG_COMB' \
            ({MAX_LG_COMB}). Lower --cap, or clear --optim bit 3 for this dataset.")

def _collapse_task(state: Dict[str, Any], payload: Tuple[int, MulTree]
                   ) -> Tuple[int, List[int], Optional[Dict[int, Tuple[int, int, float]]]]:
    """
    Compute the groups of every gene tree against one MUL-tree.
 
    Returns (m_idx, fails, counts).
 
    `state` is moved once per worker by WorkPool - inherited under fork, read from a
    single spill file under spawn - so the per-task payload is just (index, MUL-tree).
 
    With dedup on: gene trees that are isomorphic as species-labelled trees have
    isomorphic GroupData against ANY MUL-tree, because compute_groups reads only the
    topology, the leaf species and the MUL-tree's sister sets - gene-copy identifiers
    enter solely at the final registry.get_ids(). So compute_groups runs once per class
    and the units are transported to the other members through the canonical leaf order.
 
    `fails` and `counts` are produced here because the numbers the filtering stage needs
    are already in hand; re-reading every pickle afterwards to recover them was a second
    full I/O pass over what we had just written.
    """
    gene_trees: Dict[int, SmrtTree] = state['trees']
    registry: NameRegistry = state['registry']
    cap: int = state['group_cap']
    want_counts: bool = state['want_counts']
    rep_of: Dict[int, int] = state['rep_of'] or {}
    rule: int = state['rule']
    cap_by_work: bool = state['cap_by_work']
    use_exact: bool = state['use_exact']
    pickle_dir, run_prefix = Path(state['pickle_dir']), state['run_prefix']
 
    m_idx, m_data = payload
    h1_sis, hx_sis_list = m_data.get_sister_clades()
 
    # Only representatives that actually HAVE members are worth converting: the position
    # transport costs about as much as compute_groups, so paying it for a singleton class
    # is pure loss. This is what made de-duplication a net cost at a low duplicate rate.
    has_members = set(rep_of.values())
 
    groups_out: Dict[int, GroupData] = {}          # what gets pickled and reconciled
    cap_cache: Dict[int, int] = {}                 # rep -> GRAMPA-rule ambiguous count
    fails: List[int] = []
    positions: Dict[int, Any] = {}
    counts: Optional[Dict[int, Tuple[int, int, float]]] = {} if want_counts else None

    #from collections import Counter
    #c_max = max(Counter(l.pure for l in m_data.mt.ete_tree.iter_leaves()).values())
    # for now multi-MT with c_max > 2 shouldn't filter excessively [and we'll hope it
    # won't lock up the enumeration]. Much more important to limit the power instead...
    c_max = 2
 
    for g_idx, gt_obj in gene_trees.items():
        rep = rep_of.get(g_idx)
        if rep is None:
            gd = compute_groups(gt_obj, m_data, registry, h1_sis, hx_sis_list, rule)
            if g_idx in has_members:
                positions[g_idx] = gt_obj.groups_to_positions(gd)
        else:
            pos = positions.get(rep)
            if pos is None:
                # Order-independent: a member may precede its representative if the
                # gene-tree dict is ever re-ordered. Cheaper than an assertion that
                # fails deep inside a worker.
                rep_gd = compute_groups(gene_trees[rep], m_data, registry,
                                        h1_sis, hx_sis_list, rule)
                pos = positions[rep] = gene_trees[rep].groups_to_positions(rep_gd)
            gd = gt_obj.groups_from_positions(pos)
 
        groups_out[g_idx] = gd
        n_amb = len(gd.ambiguous_groups)
        n_multi = sum(1 for u in gd.ambiguous_groups if len(u) > 1)
        work = log2_combinations(n_amb, n_multi, c_max, use_exact)

        # ---- the FILTER metric: what decides the dataset ---------------------
        if cap_by_work:
            metric = work
        elif rule == ENGINE_RULE:
            metric = n_amb                       # already GRAMPA's quantity
        else:
            # The filter must not move when the algorithm does, so recover GRAMPA's
            # count explicitly. Paid ONLY under a non-default rule, and only once per
            # de-duplication class (members reuse their representative's groups).
            key = rep if rep is not None else g_idx
            metric = cap_cache.get(key)
            if metric is None:
                src = gt_obj if rep is None else gene_trees[rep]
                eng_gd = compute_groups(src, m_data, registry, h1_sis, hx_sis_list,
                                        ENGINE_RULE)
                metric = cap_cache[key] = len(eng_gd.ambiguous_groups)

        if want_counts:
            counts[g_idx] = (n_amb, len(gd.fixed_groups), work)
        if metric > cap:
            fails.append(g_idx)
            continue

        # Work guard on gts that did not fail
        check_workload(work)

    p_path = pickle_dir / f"{run_prefix}_{m_idx}_groups.pickle"
    tmp = p_path.with_suffix('.pickle.tmp')
    try:
        # Write-then-rename: a crash mid-write must not leave a truncated pickle that a
        # resumed run would happily read.
        with open(tmp, 'wb') as f:
            pickle.dump(groups_out, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, p_path)
    except OSError as e:
        raise RuntimeError(f"MUL-tree {m_idx}: cannot write group data '{p_path}': {e}") from e

    return m_idx, fails, counts

def _sweep_filter_task(state: Dict[str, Any], h_item: Tuple[str, int]) -> List[bytes]:
    """
    Decide the cap for ONE donor clade, over every DISTINCT gene tree.
 
    Returns a list of over-cap signatures.
    Keyed by signature rather than gene-tree index because both decisions are structural:
    isomorphic gene trees give identical unit counts, so one decision serves the whole
    de-duplication class and the payload back stays tiny.
    """
    from .core import sweep_engine
 
    st_flat: FlatTree = state['st_flat']
    flats: Dict[bytes, FlatTree] = state['flats_by_sig']     # one per signature
    clade_ids: Dict[str, Set[int]] = state['clade_ids']
    valid_t: Dict[str, List[int]] = state['valid_t']
    cap: int = state['group_cap']
    rule: int = state['rule']
    cap_by_work: bool = state['cap_by_work']
    use_exact: bool = state['use_exact']

    # Real weights are needed because sweep_engine is a
    # single-slot per-worker cache keyed partly on cost, so using the run's weights lets
    # the scoring phase reuse this engine - and its scalar/leaf caches - instead of
    # rebuilding at the phase transition.
    pin_states_ = sweep_engine(st_flat, *state['weights']).pin_states
 
    h1_name, h = h_item
    sp_ids, targets = clade_ids[h1_name], valid_t[h1_name]

    def _sweep_cap_decision(gt_flat: FlatTree) -> bool:
        """
        filtered? for one (gene tree, donor clade).

        The FILTER is measured under GRAMPA's rule unless --cap-by-work is set, so the
        set of analysed gene trees does not move with --optim or --unit-rule and stays
        comparable with the pairwise engine and with GRAMPA. The WORK is measured under
        the rule actually in use, because that is what the sweep will enumerate. The
        sweep models exactly two copies, so c_max is always 2 here.
        """
        filt_rule = rule if cap_by_work else ENGINE_RULE
        filt_units = compute_units(gt_flat, sp_ids, rule=filt_rule)
        if not filt_units:
            return False

        # SHORT-CIRCUIT. pin_states costs an O(units*depth + N*g) pass, but pinning can
        # only ever REDUCE the free-unit count: free[t] <= len(units) for every target.
        # So when the un-pinned bound already clears the cap, no target can exceed it and
        # the pass is pure waste - which is the common case (97.5% of decisions measured
        # at the default cap). The decision, and therefore every log line, is identical.
        g = len(filt_units)
        n_big = sum(1 for u in filt_units if len(u) > 1)
        bound = (log2_combinations(g, n_big, 2, use_exact)
                 if cap_by_work else g)
        if bound <= cap:
            over = False
            st_map = free = None
        else:
            st_map, free = pin_states_(gt_flat, h, filt_units, st_flat, targets)
            if cap_by_work:
                big = [i for i, u in enumerate(filt_units) if len(u) > 1]
                over = any(log2_combinations(free[t],
                            sum(1 for i in big if st_map[t][i] is None), 2,
                            use_exact) > cap for t in targets)
            else:
                over = any(free[t] > cap for t in targets)
        if over:
            return True
 
        # Same argument for the work guard: pinning only shrinks the enumeration, so the
        # un-pinned count is an upper bound. Skip the analysis when it already clears.
        if filt_rule == rule:
            run_units, run_bound = filt_units, log2_combinations(g, n_big, 2, use_exact)
        else:
            run_units = compute_units(gt_flat, sp_ids, rule=rule)
            if not run_units:
                return False
            run_bound = log2_combinations(
                len(run_units), sum(1 for u in run_units if len(u) > 1), 2,
                use_exact)
        if run_bound <= MAX_LG_COMB:
            return False

        if filt_rule == rule and st_map is not None:
            rst, rfree = st_map, free           # reuse the pinning we already paid for
        else:
            rst, rfree = pin_states_(gt_flat, h, run_units, st_flat, targets)
        big = [i for i, u in enumerate(run_units) if len(u) > 1]
        worst = max((log2_combinations(rfree[t],
                                       sum(1 for i in big if rst[t][i] is None),
                                       2, use_exact)
                     for t in targets), default=0.0)
        check_workload(worst)                   # Only situation where you need to raise
        
        return False

    return [sig for sig, gt_flat in flats.items() if _sweep_cap_decision(gt_flat)]

class GeneTreeManager:
    def __init__(self, config: TaskConfig, logger: GranLogger, num_processes: int = 1,
                 pickle_action: str = 'archive', pool: Optional[WorkPool] = None):
        self.tcf = config
        self.logger = logger
        self.n_procs = num_processes
        self.pickle_action = pickle_action

        self.dedup_threshold = getattr(config, 'disable_dedup_below', 0.05)
        self.rep_of: Dict[int, int] = {}        # the decision, read back by main
        self.dedup_latch_next = True            # ditto, drives the serial latch

        self._pool = pool
        self._owns_pool = pool is None
 
    @property
    def pool(self) -> WorkPool:
        """The shared worker pool, created on demand when the caller did not supply one."""
        if self._pool is None:
            self._pool = WorkPool(self.n_procs, spill_dir=self.tcf.pickle_dir,
                                  logger=self.logger, name=self.tcf.run_prefix)
        return self._pool
 
    def cleanup(self) -> None:
        """Release the workers and any spilled state files. A no-op when the pool was
        supplied by the caller, which then owns its lifetime."""
        if self._owns_pool and self._pool is not None:
            self._pool.cleanup()
            self._pool = None

    def cull(self, mul_trees: Dict[int, MulTree], gene_trees: Dict[int, SmrtTree], registry: NameRegistry) -> bool:

        fails_by_mt, counts_by_mt, reused = {}, {}, []
        if self.tcf.mode != "st-only":
            fails_by_mt, counts_by_mt, reused = self.collapse_groups(mul_trees, gene_trees, registry)
        original_count = max(gene_trees.keys()) + 1 if gene_trees else 0
        gt_failures = self.filter_and_check(mul_trees, gene_trees, fails_by_mt, counts_by_mt, reused)
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

        elif action.startswith('s'): # store
            step = "Storing pickle directory"
            self.logger.report_step(step, "In progress...")
            try:
                CommonOps._make_tar(pickle_dir)
                self.logger.report_step(step, f"Success: created {pickle_dir.name}.tar")
            except Exception as e:
                self.logger.log(f"Failed to store pickle directory: {e}", 'w')

        elif action.startswith('a'): # archive
            step = "Archiving pickle directory"
            self.logger.report_step(step, "In progress...")
            try:
                CommonOps._make_gztar(pickle_dir)
                # Delete the uncompressed directory to free up the space and inodes
                rmtree(pickle_dir)
                self.logger.report_step(step, f"Success: created {pickle_dir.name}.tar.gz")
            except Exception as e:
                self.logger.log(f"Failed to archive pickle directory: {e}", 'w')

        else:
            self.logger.log(f"Unknown pickle handling action: '{action}'", 'w')

    def unpack_archive_(self, pickle_dir: Path) -> None:
        archive_path = Path(str(pickle_dir) + '.tar.gz')
        storage_path = Path(str(pickle_dir) + '.tar')
        extant_path = archive_path if archive_path.exists() else storage_path if storage_path.exists() else None
        if not pickle_dir.exists() and extant_path and not self.tcf.overwrite:
            step_unpack = "Unpacking pickle archive"
            self.logger.report_step(step_unpack, "In progress...")
            try:
                unpack_archive(extant_path, extract_dir=pickle_dir)
                self.logger.report_step(step_unpack, "Success")
            except Exception as e:
                self.logger.log(f"Failed to unpack archive: {e}", 'w')

    def _want_checknums_file(self) -> bool:
        """The checknums file holds one line per (MUL-tree, gene tree) pair - millions of
        lines on a large run, formatted and written every time for a diagnostic that only
        check-nums mode consumes."""
        return self.tcf.mode == "check-nums" or getattr(self.tcf, 'write_checknums', False)

    def collapse_groups(self, mul_trees: Dict[int, MulTree], gene_trees: Dict[int, SmrtTree], registry: NameRegistry, label: str = ''):
        """
        Computes groups for all MUL-trees and dumps them to pickle immediately.
        Returns (fails_by_mt, counts_by_mt, reused_mt_ids); the counts feed
        filter_and_check directly so it never has to read the pickles back.
        """
        tcf = self.tcf
        pickle_dir = tcf.pickle_dir
        self.unpack_archive_(pickle_dir)
        pickle_dir.mkdir(parents=True, exist_ok=True)

        step = f"Collapsing {label}gene tree groupings"
        self.logger.report_step(step, "In progress...")

        # --- PRIME CACHES BEFORE FORKING ------------------------------------------
        # linearize() registers every species name, full leaf name and internal name, so
        # flattening here also locks the registry IDs globally (workers must not invent
        # divergent ones). The gene-tree flats carry no Euler tour / sparse table: they
        # are only ever traversed, never LCA-queried. Reconciliation reuses them as-is.
        for gt in gene_trees.values():
            gt.make_flat(registry)
        # ONE decision for the whole task; reconcile reuses it rather than re-deciding.
        self.rep_of, self.dedup_latch_next = plan_dedup(
            gene_trees, self.dedup_threshold, enabled=tcf.dedup_gts,
            latched_off=getattr(tcf, 'dedup_latch', False),
            logger=self.logger)
        if self.rep_of:
            # canon is computed from the flat tree, which __getstate__ does not ship;
            # priming it here is what lets the workers use it without rebuilding.
            for gt in gene_trees.values():
                gt.canon                        # prime for the workers (see __getstate__)

        registry_path = pickle_dir / f"{tcf.run_prefix}_registry.pickle"
        force_regenerate = self._check_registry_safety(registry_path, registry)
        if tcf.unit_rule != ENGINE_RULE and not tcf.cap_by_work:
            # A reused pickle holds groups under whatever rule wrote it, so GRAMPA's
            # ambiguous-unit count - the filter metric - is not recoverable from it.
            # Regenerate rather than let a resumed run filter differently from a fresh one.
            force_regenerate = True

        tasks, reused = [], []
        for m_idx, m_data in mul_trees.items():
            if m_idx == 0:
                continue
            pickle_path = pickle_dir / f"{tcf.run_prefix}_{m_idx}_groups.pickle"
            if pickle_path.exists() and not tcf.overwrite and not force_regenerate:
                reused.append(m_idx)             # counts for these must come from disk
                continue
            tasks.append((m_idx, m_data))

        want_counts = self._want_checknums_file()
        cap = self.tcf.group_cap
        fails_by_mt: Dict[int, List[int]] = {}
        counts_by_mt: Dict[int, Dict[int, Tuple[int, int, float]]] = {}

        state = {'trees': gene_trees, 'registry': registry, 'group_cap': cap,
            'want_counts': want_counts, 'rep_of': self.rep_of, 'rule': tcf.unit_rule,
            'cap_by_work': tcf.cap_by_work, 'use_exact': tcf.use_exact,
            'pickle_dir': str(pickle_dir), 'run_prefix': tcf.run_prefix}

        # A pool costs interpreter start-up (~0.3 s for 8 workers under spawn) plus a full
        # serialisation of every gene tree. Below this threshold it never repays itself.
        for res in self.pool.map_unordered(_collapse_task, tasks, state=state,
                                           desc="# Collapsing", unit="mt",
                                           disable=self.logger.disable_tqdm,
                                           min_parallel=2 * self.n_procs):
            m_idx, fails, counts = res
            if fails:
                fails_by_mt[m_idx] = fails
            if want_counts:
                counts_by_mt[m_idx] = counts

        try:
            with open(registry_path, 'wb') as f:
                pickle.dump(registry.get_state(), f)
        except Exception as e:
            self.logger.log(f"saving registry pickle: {e}", 'e')

        self.logger.report_step(step, "Success", full_update=True)
        return fails_by_mt, counts_by_mt, reused

    def _counts_and_fails_from_pickle(self, m_idx: int, cap: int, want_counts: bool) -> Tuple[List[int], Optional[Dict[int, Tuple[int, int, float]]]]:
        """Only for MUL-trees whose group pickle was REUSED from a previous run."""
        p = self.tcf.pickle_dir / f"{self.tcf.run_prefix}_{m_idx}_groups.pickle"
        fails, counts = [], ({} if want_counts else None)
        try:
            with open(p, 'rb') as f:
                groups = pickle.load(f)
        except OSError as e:
            raise RuntimeError(f"MUL-tree {m_idx}: cannot read reused group data '{p}': {e}")
        for g_idx, gd in groups.items():
            n_amb = len(gd.ambiguous_groups)
            n_multi = sum(1 for u in gd.ambiguous_groups if len(u) > 1)
            work = log2_combinations(n_amb, n_multi, 2, self.tcf.use_exact)
            # force_regenerate guarantees rule == ENGINE_RULE here, so n_amb IS GRAMPA's
            # quantity; applying a different test than _collapse_task would make a
            # resumed run filter differently from a fresh one.
            metric = work if self.tcf.cap_by_work else n_amb
            if metric > cap:
                fails.append(g_idx)
            if want_counts:
                counts[g_idx] = (n_amb, len(gd.fixed_groups), work)
        return fails, counts

    def filter_by_sweep_cap(self, gene_trees: Dict[int, SmrtTree], st_flat: FlatTree,
                            clade_ids: Dict[str, Set[int]], h_id: Dict[str, int],
                            valid_t: Dict[str, Set[int]], registry: NameRegistry) -> Dict[int, int]:
        """
        The sweep's equivalent of filter_and_check: the standard path caps on units
        remaining AFTER sister-pinning, which is target-dependent, and pin_states gives
        that same count for every target - so this removes exactly the gene trees the
        standard path removes, and writes the same filtered-trees output.
        """
        step = "Filtering gene trees over group cap"
        self.logger.report_step(step, "In progress...")

        # TODO: If a gene tree was already dropped earlier in the run (e.g. by the loader), the two paths
        # will report slightly different "original" counts in write_filtered_trees. Worth passing
        # original_count in from run_sweep for consistency with cull.
        original_count = max(gene_trees.keys()) + 1 if gene_trees else 0
        for gt in gene_trees.values():
            gt.make_flat(registry)          # Traversed only: no Euler tour / or RMQ
 
        # One representative per signature: the decision is structural, so isomorphic
        # gene trees share it. This is also what keeps the worker payload small.
        flats_by_sig: Dict[bytes, FlatTree] = {}
        members: Dict[bytes, List[int]] = {}
        for g_idx, gt in gene_trees.items():
            sig = gt.canon_sig
            flats_by_sig.setdefault(sig, gt.flat_tree)
            members.setdefault(sig, []).append(g_idx)

        tcf = self.tcf
        state = {'st_flat': st_flat, 'flats_by_sig': flats_by_sig, 'clade_ids': clade_ids, 'valid_t': valid_t,
                 'group_cap': tcf.group_cap, 'rule': tcf.unit_rule, 'cap_by_work': tcf.cap_by_work,
                 'use_exact': tcf.use_exact, 'weights': tcf.weights}
        donors = [(name, h_id[name]) for name in clade_ids if name in h_id]
 
        gt_failures: Dict[int, int] = {}
        for over in self.pool.map_unordered(
                _sweep_filter_task, donors, state=state, desc="# Filtering ", unit="h1",
                disable=self.logger.disable_tqdm):
            for sig in over:
                for g_idx in members[sig]:
                    gt_failures[g_idx] = gt_failures.get(g_idx, 0) + 1

        for g_idx in sorted(gt_failures):
            self.logger.log(f"Gene tree on line {g_idx+1} is over the group cap for "
                            f"{gt_failures[g_idx]} donor clades and will be filtered.", 'w')
            del gene_trees[g_idx]
        self.logger.report_step(step, f"Success: {len(gt_failures)} gts over cap", full_update=True)
 
        self.write_filtered_trees(gene_trees, gt_failures, original_count)
        return gt_failures

    def filter_and_check(self, mul_trees: Dict[int, MulTree], gene_trees: Dict[int, SmrtTree],
                         fails_by_mt=None, counts_by_mt=None, reused=()) -> Dict[int, int]:
        """
        Applies the group cap. The counts come from collapse_groups, which already had
        them, so there is no second worker pool and no second pass over the pickles.
        """
        step = "Filtering gene trees over group cap"
        self.logger.report_step(step, "In progress...")

        cap = self.tcf.group_cap
        want_counts = self._want_checknums_file()
        fails_by_mt = dict(fails_by_mt or {})
        counts_by_mt = dict(counts_by_mt or {})

        for m_idx in reused:
            f, c = self._counts_and_fails_from_pickle(m_idx, cap, want_counts)
            if f:
                fails_by_mt[m_idx] = f
            if want_counts:
                counts_by_mt[m_idx] = c

        gt_failures: Dict[int, int] = {}
        for fails in fails_by_mt.values():
            for g_idx in fails:
                gt_failures[g_idx] = gt_failures.get(g_idx, 0) + 1

        # Must happen before the deletion loop below
        if want_counts:
            self._write_checknums(mul_trees, gene_trees, counts_by_mt, fails_by_mt)

        self.logger.report_step(step, f"Success: {len(gt_failures)} gts over cap")#, full_update=True)

        for g_idx in sorted(gt_failures):
            if g_idx in gene_trees:
                self.logger.log(f"Gene tree on line {g_idx+1} is over the group cap in "
                                f"{gt_failures[g_idx]} MTs and will be filtered.", 'w')
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

    def _write_checknums(self, mul_trees: Dict[int, MulTree], gene_trees: Dict[int, SmrtTree],
                         counts_by_mt: Dict[int, Dict[int, Tuple[int, int, float]]],
                         fails_by_mt: Dict[int, List[int]]) -> None:
        """
        The per-(MUL-tree, gene tree) diagnostic.

        The 'combinations' column is the REAL enumeration size, 2**work - not 2**groups,
        because a multi-MT gives a unit c states and exact grouping gives a multi-leaf
        unit 2c-1. The 'over.cap.filtered' column is read from the authoritative `fails`
        list rather than recomputed: under a non-default --unit-rule the filter metric is
        GRAMPA's ambiguous-unit count recovered separately, which is NOT the n_amb stored
        here, so any recomputation would silently disagree with what was actually filtered.
        """
        check_path = Path(self.tcf.output_dir) / f"{self.tcf.run_prefix}-checknums.txt"
        sorted_gene_ids = sorted(gene_trees.keys())
        with open(check_path, 'w') as f:
            f.write("mul.tree\tgene.tree\tgroups\tfixed\tcombinations\tover.cap.filtered\n")
            for m_idx in sorted(mul_trees.keys()):
                if m_idx == 0 or m_idx not in counts_by_mt:
                    continue
                m_data = mul_trees[m_idx]
                h1_name = m_data.h1_node.name if m_data.h1_node else "NA"
                hx_sisters = m_data.hx_sisters
                hx_str = ("\t".join(f'H{i+2} Node:{hx.name}' for i, hx in enumerate(hx_sisters))
                          if hx_sisters else "Hx Nodes:NA")
                f.write(f"# MT-{m_idx}:{m_data.to_marked_str()}\tH1 Node:{h1_name}\t{hx_str}\n")

                counts = counts_by_mt[m_idx]
                filtered = set(fails_by_mt.get(m_idx, ()))
                for g_idx in sorted_gene_ids:
                    c = counts.get(g_idx)
                    if c is None:
                        continue
                    n_amb, n_fix, work = c
                    f.write(f"{m_idx}\t{g_idx+1}\t{n_amb + n_fix}\t{n_fix}\t"
                            f"{2 ** work:.0f}\t{'Y' if g_idx in filtered else 'N'}\n")
                f.write("# ----------------------------------\n")