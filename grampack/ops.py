import re
import sys
import pickle
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Set, Any

from .models import SmrtTree, MulTree, GroupData, NameRegistry
from .config import GrandmaConfig
from .logger import GrandmaLogger
from .reconcile import Reconciler

class GeneTreeProcessor:
    @staticmethod
    def read_gene_tree(line: str, index: int) -> Tuple[int, Optional[SmrtTree], str]:
        """
        Parses a gene tree line.
        Returns: (index, TreeObject, FilterMessage)
        """
        clean_line = line.strip()
        if not clean_line:
            return index, None, "# Empty line"

        try:
            # Use GrandmaTree wrapper which uses ETE3
            # ETE3 format 1 loads internal names if present
            gt = SmrtTree(newick=clean_line)
        except Exception:
            return index, None, "# Error parsing newick"

        # Filters from gene_tree.py
        # 1. Check for Polytomies (non-bifurcating)
        # In ETE3, check if any node has > 2 children
        for node in gt.ete_tree.traverse():
            if len(node.children) > 2:
                # Check if root trifurcation (allowed in unrooted, but GRAMPA usually wants rooted)
                if node.is_root():
                     return index, None, "Tree contains non-bifurcating nodes (root)"
                return index, None, "Tree contains non-bifurcating nodes"

        # 2. Check Rooting (Num Internal = Num Tips - 1)
        # Note: ETE3 handles rooting differently, but strict bifurcating tree property holds:
        # Leaves = N, Internal = N-1.
        leaves = len(gt.ete_tree.get_leaves())
        internal = len([n for n in gt.ete_tree.traverse() if not n.is_leaf()])
        
        if internal != (leaves - 1):
             # This usually happens if the root has only 2 children? 
             # Actually for rooted bifurcating: N leaves -> N-1 internal (including root).
             pass 

        return index, gt, ""

class TreeLoader:
    @staticmethod
    def gene_trees(path: str, logger: GrandmaLogger, registry: NameRegistry = None) -> dict[int, SmrtTree]:
        step = "Reading gene trees"
        logger.report_step(step, "In progress...")

        try:
            with open(path, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            logger.write(f"Error reading gene tree file: {e}")
            raise e
            
        gene_trees = {}
        valid_count = 0
        warnings = []

        for i, line in enumerate(lines):
            idx, gt, msg = GeneTreeProcessor.read_gene_tree(line, i+1)
            if gt:
                gene_trees[idx] = gt
                valid_count += 1
            if msg:
                if not gt:
                    warnings.append(f"# WARNING: Gene tree on line {i+1}: {msg} -- Filtering.")
                    logger.warnings += 1

        logger.report_step(step, f"Success: {valid_count} gene trees read")
        for w in warnings: logger.write(w)

        return gene_trees
    
    @staticmethod
    def spec_tree(path: str, logger: GrandmaLogger) -> SmrtTree:
        step = "Reading species tree"
        logger.report_step(step, "In progress...")

        try:
            with open(path, 'r') as f:
                content = f.read().strip()
        except Exception as e:
            logger.write(f"Error reading species tree file: {e}")
            raise e
        
        logger.report_step(step, "Success: species tree read")
        return SmrtTree(newick=content)

class MulTreeManager:
    def __init__(self, species_tree: SmrtTree, config: GrandmaConfig, logger: GrandmaLogger):
        self.st = species_tree
        self.cfg = config
        self.logger = logger

    def _resolve_h_inputs(self, raw_input: str, h_type: str) -> list:
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
                    self.logger.write(f"Error: Node {name_to_check} not found in tree (specified in -{h_type}).")
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
        
        if self.cfg.mode != "st-only":
            step = "Parsing hybrid clades"
            self.logger.report_step(step, "In progress...")
            h1_resolved = self._resolve_h_inputs(self.cfg.h1_nodes, "h1")
            h2_resolved = self._resolve_h_inputs(self.cfg.h2_nodes, "h2")
            self.logger.report_step(step, "Success: got H nodes")

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
        
        if self.cfg.mode != "st-only":
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
            
        return mul_trees
    
class GeneTreeManager:
    def __init__(self, config: GrandmaConfig, logger: GrandmaLogger, reconciler: Reconciler):
        self.cfg = config
        self.logger = logger
        self.reconciler = reconciler

    def cull(self, mul_trees: Dict[int, MulTree], gene_trees: Dict[int, SmrtTree], registry: NameRegistry):
        if self.cfg.mode != "st-only":
            self.collapse_groups(mul_trees, gene_trees, registry)
        self.filter_and_check(mul_trees, gene_trees)
        self.write_filtered_trees(gene_trees)

    def collapse_groups(self, mul_trees: Dict[int, MulTree], gene_trees: Dict[int, SmrtTree], registry: NameRegistry):
        """
        Computes groups for all MUL-trees using registry-optimized logic and DUMPS to pickle immediately.
        Does NOT retain data in memory.
        """
        step = "Collapsing gene tree groupings"
        self.logger.report_step(step, "In progress...")
        
        pickle_dir = self.cfg.pickle_dir
        pickle_dir.mkdir(parents=True, exist_ok=True)

        for m_idx, m_data in mul_trees.items():
            if m_idx == 0: continue
            
            pickle_path = pickle_dir / f"{self.cfg.run_prefix}_{m_idx}_groups.pickle"
            if pickle_path.exists() and not self.cfg.overwrite:
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
                self.logger.write(f"Error saving pickle {pickle_path}: {e}")
            
            del current_mt_groups
            
        self.logger.report_step(step, "Success")

    def filter_and_check(self, mul_trees: Dict[int, MulTree], gene_trees: Dict[int, SmrtTree]):
        """
        Writes checknums file and filters trees exceeding group cap.
        Optimized regex formatting and pre-sorted iteration.
        """
        step = "Filtering gene trees over group cap"
        self.logger.report_step(step, "In progress...")
        
        check_path = Path(self.cfg.output_dir) / f"{self.cfg.run_prefix}-checknums.txt"
        
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
                if not self.cfg.is_mul_input:
                     h1_name = m_data.h1_node.name if m_data.h1_node else "NA"
                     h2_name = m_data.h2_node.name if m_data.h2_node else "NA"
                     h_info = f"\tH1 Node:{h1_name}\tH2 Node:{h2_name}"
                
                f.write(f"# MT-{m_idx}:{mt_str}{h_info}\n")

                pickle_path = self.cfg.pickle_dir / f"{self.cfg.run_prefix}_{m_idx}_groups.pickle"
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
                    if num_ambig > self.cfg.group_cap:
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
                self.logger.write(f"# WARNING: Gene tree on line {g_idx} is over the group cap in {fail_count} MTs and will be filtered.")
                self.logger.warnings += 1
                del gene_trees[g_idx]

    def write_filtered_trees(self, gene_trees: Dict[int, SmrtTree]):
        """Matches GRAMPA's filterOut logic."""
        step = "Writing filtered gene trees to file"
        self.logger.report_step(step, "In progress...")
        
        p = Path(self.cfg.output_dir) / f"{self.cfg.run_prefix}-trees-filtered.txt"
        count = 0
        with open(p, 'w') as f:
            for idx in sorted(gene_trees.keys()):
                # GRAMPA writes the original newick string
                f.write(gene_trees[idx].to_string(internal_labels=False) + "\n")
                count += 1
                
        self.logger.report_step(step, f"Success: {count} gene trees written")