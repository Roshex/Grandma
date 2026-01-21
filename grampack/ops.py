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
        # Default: All nodes
        if not raw_input:
            return [n.name for n in self.st.ete_tree.traverse()]

        if " " in raw_input:
            groups = raw_input.split(" ")
            clade_lists = [g.split(",") for g in groups]
        else:
            clade_lists = [raw_input.split(",")]

        h_nodes = []
        for clade in clade_lists:
            if len(clade) == 1:
                val = clade[0]
                node_name = f"<{val}>" if val.isdigit() else val
                if not self.st.get_node(node_name):
                     self.logger.write(f"Error: Node {node_name} not found in tree.")
                     sys.exit(1)
                if node_name not in h_nodes:
                    h_nodes.append(node_name)
            else:
                # --- FIX: Use ete3 directly since get_lca was removed from SmrtTree ---
                nodes_obj = []
                for name in clade:
                    node_obj = self.st.get_node(name)
                    if not node_obj:
                        self.logger.write(f"Error: Node {name} not found in tree during LCA resolution.")
                        sys.exit(1)
                    nodes_obj.append(node_obj)
                
                lca_node = self.st.ete_tree.get_common_ancestor(nodes_obj)
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

    def cull(self, mul_trees: Dict[int, MulTree], gene_trees: Dict[int, SmrtTree]):
        if self.cfg.mode != "st-only":
            self.collapse_groups(mul_trees, gene_trees)
        self.filter_and_check(mul_trees, gene_trees)
        self.write_filtered_trees(gene_trees)

    def collapse_groups(self, mul_trees: Dict[int, MulTree], gene_trees: Dict[int, SmrtTree]):
        """
        Computes groups for all MUL-trees and DUMPS to pickle immediately.
        Does NOT retain data in memory.
        """
        step = "Collapsing gene tree groupings"
        self.logger.report_step(step, "In progress...")
        
        pickle_dir = self.cfg.pickle_dir
        pickle_dir.mkdir(parents=True, exist_ok=True)

        for m_idx, m_data in mul_trees.items():
            if m_idx == 0: continue
            
            pickle_path = pickle_dir / f"{self.cfg.run_prefix}_{m_idx}_groups.pickle"
            
            # Skip if exists and not overwrite
            if pickle_path.exists() and not self.cfg.overwrite:
                continue
                
            # OPTIMIZATION: Get sisters once for this MUL-tree
            h1_sis, h2_sis = self.reconciler.get_sister_clades(m_data)

            # Compute groups for ALL gene trees for THIS mul-tree
            current_mt_groups: Dict[int, GroupData] = {}
            
            for g_idx, gt_obj in gene_trees.items():
                group_data = self.reconciler.compute_groups(gt_obj, m_data, h1_sis, h2_sis)
                current_mt_groups[g_idx] = group_data
            
            # Dump to disk
            try:
                with open(pickle_path, 'wb') as f:
                    pickle.dump(current_mt_groups, f)
            except Exception as e:
                self.logger.write(f"Error saving pickle {pickle_path}: {e}")
            
            # CLEAR MEMORY
            del current_mt_groups
            
        self.logger.report_step(step, "Success")

    def filter_and_check(self, mul_trees: Dict[int, MulTree], gene_trees: Dict[int, SmrtTree]):
        step = "Filtering gene trees over group cap"
        self.logger.report_step(step, "In progress...")
        
        check_path = Path(self.cfg.output_dir) / f"{self.cfg.run_prefix}-checknums.txt"
        trees_to_remove: Set[int] = set()
        
        with open(check_path, 'w') as f:
            f.write("mul.tree\tgene.tree\tgroups\tfixed\tcombinations\tover.cap.filtered\n")
            
            for m_idx, m_data in mul_trees.items():
                if m_idx == 0: continue
                
                # Load Pickle to check counts
                pickle_path = Path(self.cfg.output_dir) / f"{self.cfg.run_prefix}_{m_idx}_groups.pickle"
                if not pickle_path.exists(): continue
                
                current_mt_groups: Dict[int, GroupData] = {}
                with open(pickle_path, 'rb') as pf:
                    current_mt_groups = pickle.load(pf)
                
                mt_str = m_data.mt.to_string(internal_labels=True)[:-1] 
                f.write(f"# MT-{m_idx}: {mt_str} | H1: {m_data.h1_node}, H2: {m_data.h2_node}\n")
                                
                for g_idx in gene_trees:
                    if g_idx not in current_mt_groups: continue
                    
                    groups = current_mt_groups[g_idx]
                    num_groups = len(groups.ambiguous_groups)
                    num_fixed = len(groups.fixed_groups)
                    combos = 2**num_groups
                    
                    over_cap = "N"
                    if num_groups > self.cfg.group_cap:
                        over_cap = "Y"
                        trees_to_remove.add(g_idx)
                        
                    f.write(f"{m_idx}\t{g_idx}\t{num_groups}\t{num_fixed}\t{combos}\t{over_cap}\n")
                
                f.write("# ----------------------------------\n")
                
                # CLEAR MEMORY
                del current_mt_groups

        self.logger.report_step(step, f"Success: {len(trees_to_remove)} gts over cap")
        
        for idx in trees_to_remove:
            if idx in gene_trees:
                del gene_trees[idx]
                self.logger.warnings += 1

    def write_filtered_trees(self, gene_trees: Dict[int, SmrtTree]):
        step = "Writing filtered gene trees to file"
        self.logger.report_step(step, "In progress...")
        
        p = Path(self.cfg.output_dir) / f"{self.cfg.run_prefix}-trees-filtered.txt"
        count = 0
        with open(p, 'w') as f:
            for idx, gt_obj in gene_trees.items():
                f.write(gt_obj.to_string(internal_labels=False) + "\n")
                count += 1
                
        self.logger.report_step(step, f"Success: {count} gene trees written")