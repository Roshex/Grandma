'''
Replaces params.py and global_vars.py. Holds constants and configuration dataclasses.
Handles the input parsing logic from opt_parse.py and spec_tree.py
'''

import argparse
import time
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from .tree_ops import GrandmaTree, MulData
from .logger import GrandmaLogger

@dataclass(frozen=True, slots=True)
class GrandmaConfig:
    # Input/Output
    species_tree_path: str = ""
    gene_tree_path: str = ""
    output_dir: str = ""

    # Execution Mode
    mode: str = "single"  # options: single, split, full, parallel

    run_prefix: str = "grandma"
    overwrite: bool = False

    pickle_dir: str = ""
    n_lowest: int = 6
    
    # Algorithm Options
    h1_nodes: Optional[str] = None
    h2_nodes: Optional[str] = None
    group_cap: int = 8
    maps_opt: bool = False
    orth_opt: bool = False
    
    # Execution
    num_processes: int = 1
    verbosity: int = 3
    info_only: bool = False
    
    # Modes
    is_mul_input: bool = False
    lca_opt: str = "default"  # options: default, st-only, no-st, check-nums
    
    # Metadata
    authors: str = "Ronen Shtein"
    doi: str = "TBD"
    github: str = "TBD"
    http: str = "TBD"
    release: str = "TBD 2026"
    version: str = "2.0.0 (Modern)"

    # GRAMPA Source Metadata
    source_authors: str = "Gregg Thomas, S. Hussain Ather, Matthew Hahn"
    source_doi: str = "https://doi.org/10.1093/sysbio/syx044"
    source_github: str = "https://github.com/gwct/grampa"
    source_http: str = "https://gwct.github.io/grampa/"
    source_release: str = "June 2024"
    source_version: str = "1.4.0"

def parse_args() -> GrandmaConfig:
    parser = argparse.ArgumentParser(description="GRAMPA (Modern): Gene-tree Reconciliation Algorithm with MUL-trees.")
    
    # Required
    parser.add_argument("-s", dest="spec_tree", required=True, help="Species tree file")
    parser.add_argument("-g", dest="gene_tree", help="Gene tree file (required unless --buildmultrees)")
    
    # Base Options
    parser.add_argument("-h1", dest="h1", help="Hybrid clade 1 (space separated nodes/tips)")
    parser.add_argument("-h2", dest="h2", help="Hybrid clade 2 (space separated nodes/tips)")
    parser.add_argument("-c", dest="cap", type=int, default=8, help="Max groups (cap)")
    parser.add_argument("-o", dest="outdir", help="Output directory")
    parser.add_argument("-p", dest="procs", type=int, default=1, help="Number of processes")
    parser.add_argument("-f", dest="prefix", default="grandma", help="Output file prefix")
    parser.add_argument("-v", dest="verbosity", type=int, default=3, help="Verbosity (0-3)")

    # Mode Options
    parser.add_argument("-m", dest="mode", choices=["single", "split", "full", "parallel"], default="single",
                        help="Execution mode (single, split, full, parallel)")
    parser.add_argument('-i', '--iter', type=int, default=0, help='Number of iterations; <int>, non-positive for infinite mode')
    parser.add_argument('--prep', type=str, help='Preprocess input files; "0/D/default" for default settings, or <path> for a config json')
    parser.add_argument('--start', type=str, default='auto', help='Start point when finishing a previous execution; positive <int>, or "auto" for auto-detection')
    parser.add_argument('--cutoff', type=str, default='auto', help='Stopping condition mode; "auto" for abs:0+lookback, "rel:<float>" for relative, or "abs:<int>" for absolute')
    parser.add_argument('--ignore-nesting', action='store_true', help='Do not automatically fix nested hybridization events; let GRAMPA iterate normally')

    parser.add_argument('--plot', action='store_true', help='Plot taxon count, MP score, and normalized score over iterations')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode for additional output')

    # Flags
    parser.add_argument("--buildmultrees", action="store_true", help="Only build MUL-trees and exit")
    parser.add_argument("--checknums", action="store_true", help="Count groups and exit")
    parser.add_argument("--no-st", action="store_true", help="Skip singly-labeled tree")
    parser.add_argument("--st-only", action="store_true", help="Only run singly-labeled tree")
    parser.add_argument("--maps", action="store_true", help="Output detailed maps")
    parser.add_argument("--orthologies", action="store_true", help="Run orthology labeling (Beta)")
    parser.add_argument("--force", action="store_true", dest="overwrite", help="Overwrite existing output")
    
    # Compatibility flags (ignored or handled implicitly)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output")

    args = parser.parse_args()
    
    # Determine Mode logic before initialization
    lca_opt = "default"
    if args.buildmultrees: lca_opt = "mul-gen-only"
    elif args.checknums:   lca_opt = "check-nums"
    elif args.st_only:     lca_opt = "st-only"
    elif args.no_st:       lca_opt = "no-st"
        
    # Handle Output Dir
    out_dir = args.outdir if args.outdir else f"grampa_out_{int(time.time())}"

    # Instantiate the Immutable Config
    return GrandmaConfig(
        species_tree_path = args.spec_tree,
        gene_tree_path    = args.gene_tree if args.gene_tree else "",
        output_dir        = out_dir,

        mode              = args.mode,

        run_prefix        = args.prefix,
        overwrite         = args.overwrite,
        h1_nodes          = args.h1,
        h2_nodes          = args.h2,
        group_cap         = args.cap,
        num_processes     = args.procs,
        verbosity         = args.verbosity,
        lca_opt           = lca_opt,
        # is_mul_input defaults to False, logic handled by mode usually
        maps_opt          = args.maps,
        orth_opt          = args.orthologies,
        pickle_dir        = Path(out_dir) / "pkls/",
        n_lowest          = 6
    )

class GrandmaWriter:
    def __init__(self, config: GrandmaConfig, logger: GrandmaLogger):
        self.cfg = config
        self.logger = logger

    def write_results(self, sorted_scores: list, detailed_res: list, mul_trees: dict, gene_trees: dict):
        self._write_detailed(detailed_res, gene_trees)
        self._write_scores(sorted_scores, mul_trees)
        self._write_dup_counts(detailed_res, mul_trees)

    def _write_detailed(self, detailed_res: list, gene_trees: dict):
        step = "Writing detailed output file"
        self.logger.report_step(step, "In progress...")
        
        p = Path(self.cfg.output_dir) / f"{self.cfg.run_prefix}-detailed.txt"
        with open(p, 'w') as f:
            f.write("mul.tree\tgene.tree\tdups\tlosses\ttotal.score\tmaps\n")
            for mul_idx, res_dict in detailed_res:
                for gene_idx, res in res_dict.items():
                    gt_obj = gene_trees[gene_idx]
                    if isinstance(res, list):
                        f.write(f"# GT-{gene_idx} to MT-{mul_idx}\t{len(res)} maps found!\n")
                    else:
                        res_list = [res]
                    for res in res_list:
                        map_str = GrandmaWriter.detailed_out_string(gt_obj, res.maps, res.node_dups)
                        f.write(f"{mul_idx}\t{gene_idx}\t{res.n_dups}\t{res.n_losses}\t{res.score}\t{map_str}\n")
                         
        self.logger.report_step(step, "Success")

    @staticmethod
    def detailed_out_string(gt: GrandmaTree, maps: dict, dups: dict) -> str:
        """
        Recreates the regex-based string manipulation from detailedOut in gene_tree.py
        to produce the [Map-Dup] labels in the output string.
        """
        # We work on a copy to not mutate the main tree
        out_tree = gt.ete_tree.copy()
        
        for node in out_tree.traverse():
            if node.name in maps:
                cur_map = maps[node.name][0]
                if "*" not in cur_map:
                    cur_map += "+"
                
                dup_count = dups.get(node.name, 0)
                # Format: Node[Map-Dups]
                node.name = f"{node.name}[{cur_map}-{dup_count}]"
                
        return out_tree.write(format=8) # Format 8 = All names

    def _write_scores(self, sorted_scores: list, mul_trees: dict):
        step = "Writing main output file"
        self.logger.report_step(step, "In progress...")
        
        p = Path(self.cfg.output_dir) / f"{self.cfg.run_prefix}-scores.txt"
        import re
        with open(p, 'w') as f:
            f.write("mul.tree\th1.node\th2.node\tscore\tlabeled.tree\n")
            for idx, score in sorted_scores:
                mul_data = mul_trees[idx]
                tree_str = mul_data.mt.to_string(internal_labels=True) 
                for spec in mul_data.h_clade:
                    tree_str = re.sub(f"{spec}(?!\*)", f"{spec}+", tree_str)
                    tree_str = tree_str.replace("+*", "*")
                     
                f.write(f"{idx}\t{mul_data.h1_node}\t{mul_data.h2_node}\t{score}\t{tree_str}\n")
        self.logger.report_step(step, "Success")

    def _write_dup_counts(self, detailed_res: list, mul_trees: dict):
        p_dup = Path(self.cfg.output_dir) / f"{self.cfg.run_prefix}-dup-counts.txt"
        with open(p_dup, 'w') as f:
            f.write("mul.tree\tnode\tdups\n")
            for i in range(len(detailed_res)):
                mul_idx = detailed_res[i][0]
                res_dict = detailed_res[i][1]
                hybrid_clade = mul_trees[mul_idx].h_clade
                
                main_dups = {}
                for g_idx, res_list in res_dict.items():
                    if isinstance(res_list, list):
                        res_list = res_list[0]  # Take first if multiple maps
                    dups = res_list.node_dups
                    maps = res_list.maps
                    for gt_node, count in dups.items():
                        if count != 0:
                            map_node = maps[gt_node][0]
                            main_dups[map_node] = main_dups.get(map_node, 0) + 1
                            
                for node, count in main_dups.items():
                    out_node = node + "+" if node in hybrid_clade else node
                    f.write(f"{mul_idx}\t{out_node}\t{count}\n")