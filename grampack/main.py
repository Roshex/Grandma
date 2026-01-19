import sys
import re
from pathlib import Path
from .io import parse_args, GrandmaConfig, GrandmaWriter
from .logger import GrandmaLogger
from .tree_ops import GrandmaTree
from .gene_ops import TreeLoader, GeneTreeManager, MulTreeManager
from .reconcile import Reconciler
from .orthology import OrthologyLabeler

class GrandmaEngine:
    def __init__(self, config: GrandmaConfig):
        self.cfg = config
        self.log_path = Path(self.cfg.output_dir) / f"{self.cfg.run_prefix}.log"
        self.logger = GrandmaLogger(self.log_path, self.cfg.verbosity)
        self.writer = GrandmaWriter(self.cfg, self.logger)
        
        self.spec_tree = None
        self.gene_trees = {} 
        self.mul_trees = {}  
        
        # Components
        self.reconciler = None
        self.mul_mgr = None
        self.gene_mgr = None

    def run(self):
        # 0. Banner
        self.logger.print_start_banner(self.cfg, {})
        self.logger.report_step("", "", start=True)

        # 1. Species Tree
        self.spec_tree = TreeLoader.spec_tree(self.cfg.species_tree_path, self.logger)

        # Init components
        self.reconciler = Reconciler(self.spec_tree, self.cfg)
        self.mul_mgr = MulTreeManager(self.spec_tree, self.cfg, self.logger)
        self.gene_mgr = GeneTreeManager(self.cfg, self.logger, self.reconciler)

        # 2. MUL-Trees
        self.mul_trees = self.mul_mgr.build()
        if self.cfg.lca_opt == "mul-gen-only":
            self._end_prog()
            return

        # 3. Gene Trees
        self.gene_trees = TreeLoader.gene_trees(self.cfg.gene_tree_path, self.logger)

        # DEBUG - print the string representation of all gene trees and the species tree and the first MUL tree
        '''for mul_idx, md in self.mul_trees.items():
            print(f"MUL Tree {mul_idx}: {md.mt.to_string(internal_labels=True)}, \
                  H1: {md.h1_node}, H2: {md.h2_node}, Hybrid Clade: {md.h_clade}")
        for gene_num, gt_obj in self.gene_trees.items():
            ginfo = {}
            for n in gt_obj.ete_tree.traverse():
                # in a list, store: branch length, parent name, "tip"/"internal"/"root", support
                n_type = "tip" if n.is_leaf() else ("root" if n.is_root() else "internal")
                ginfo[n.name] = [n.dist, n.up.name if n.up else None, n_type, n.support]
            print(f"Gene Tree {gene_num}: {gt_obj.to_string(internal_labels=True)}, Node Info: {ginfo}")
        print(f"Species Tree: {self.spec_tree.to_string(internal_labels=True)}")'''

        # 4. Collapse & Filter Groups
        self.gene_mgr.cull(self.mul_trees, self.gene_trees)
        if self.cfg.lca_opt == "check-nums":
            self._end_prog()
            return

        # DEBUG
        '''for gene_num, (gt_obj, x) in self.gene_trees.items():
            print(f"Gene Tree {gene_num}: {gt_obj.to_string(internal_labels=True)}, {x}")
        '''

        # 7. Reconciliation and MUL-tree Selection
        sorted_scores, detailed_res = self.reconciler.run(self.mul_trees, self.gene_trees,
                                                          self.cfg, self.logger, self.writer)

        
        # 10. Orthology
        if self.cfg.orth_opt and detailed_res:
            min_idx = sorted_scores[0][0]
            min_data = self.mul_trees[min_idx]
            min_maps = detailed_res[0][1]
            OrthologyLabeler.run(self.gene_trees, min_maps, min_data[0], 
                                min_data[2], self.cfg.output_dir, self.cfg.run_prefix)

        # 9. End
        min_idx = sorted_scores[0][0]
        min_score = sorted_scores[0][1]
        min_data = self.mul_trees[min_idx]
        min_tree_str = min_data.mt.to_string(internal_labels=True)
        
        h_clade = min_data.h_clade
        for spec in h_clade:
            min_tree_str = re.sub(f"{spec}(?!\*)", f"{spec}+", min_tree_str)
            min_tree_str = min_tree_str.replace("+*", "*")

        self._end_prog((min_idx, min_score, min_tree_str))

    def _end_prog(self, min_info=None):
        self.logger.print_end_prog(self.cfg, min_info)
        sys.exit(0)

def main():
    config = parse_args()
    engine = GrandmaEngine(config)
    engine.run()

if __name__ == "__main__":
    main()