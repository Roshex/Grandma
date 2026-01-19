'''
Replaces params.py and global_vars.py. Holds constants and configuration dataclasses.
Handles the input parsing logic from opt_parse.py and spec_tree.py
'''

import os
import re
import sys
import ast
import json
import time
import shutil
import argparse
from glob import glob
from ete3 import Tree
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union
from dataclasses import dataclass


from .tree_ops import GrandmaTree, MulData
from .logger import GrandmaLogger

@dataclass(frozen=True, slots=True)
class GrandmaMetadata:
    """Immutable metadata about the GRANDMA software."""
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

'''
missing:

seed: int = 42
max_select: int = 1 # max mts to select for processing per Run (non-overlapping H clades & scoring above ST)
ploidy_file: str = "" # path to ploidy file
min_st_lvs: int = 1 # for split mode
min_gt_lvs: int = 2 # for split mode

better mem and cpu/processing options?

'''

@dataclass(frozen=True, slots=True)
class GrandmaConfig:
    """Immutable configuration object for a GRANDMA run."""
    # Input/Output
    #species_tree_path: str = ""
    #gene_tree_path: str = ""
    #output_dir: str = ""

    species_tree_path: Path
    gene_tree_path: Optional[Path]
    output_dir: Optional[Path]
    log_path: Optional[Path]

    # Execution Mode
    # options: single, split, full, build-mts, st-only, no-st, check-nums
    mode: str = "single"

    # Iteration / Full Mode Control
    max_iter: Union[int, float] = 0        
    history: Dict[Tuple[int, int], Any] = None      
    history_file: Optional[Path] = None
    start_pt: int = 0      
    cutoff: Tuple[str, Optional[Union[int, float]]] = ("auto", None)
    ignore_nesting: bool = False

    # Prep
    #prep: Optional[str] = None # Corresponds to --prep

    run_prefix: str = "grandma"
    overwrite: bool = False

    pickle_dir: Path = Path("pkls/")
    n_lowest: int = 6
    
    # Algorithm Options
    h1_nodes: Optional[str] = None
    h2_nodes: Optional[str] = None
    group_cap: int = 8
    maps_opt: bool = False
    orth_opt: bool = False
    
    # Modes
    is_mul_input: bool = False
    
    # Execution
    num_processes: int = 1
    verbosity: int = 3
    debug: bool = False
    plot: bool = True
    info_only: bool = False

    def __post_init__(self):
        """Debug logging if enabled."""
        if self.debug:
            # We initialize a temporary logger for debug output
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger = GrandmaLogger(log_path=self.log_path, verbosity=self.verbosity, clear_log=False)
            '''logger.write("GRANDMA Configuration Initialized:", level=0)
            for field_name in self.__slots__:
                logger.write(f"  {field_name}: {getattr(self, field_name)}", level=0)'''

# --- Utility Functions ---

def parse_cutoff(val: str) -> Tuple[str, Optional[Union[float, int]]]:
    """Parses stopping condition strings into a typed tuple."""
    if val == 'auto': 
        return ('auto', None)
    if val.startswith('rel:'):
        try: 
            return ('rel', float(val.split(':')[1]))
        except (ValueError, IndexError): 
            raise argparse.ArgumentTypeError(f'Invalid relative cutoff: {val}')
    if val.startswith('abs:'):
        try: 
            return ('abs', int(val.split(':')[1]))
        except (ValueError, IndexError): 
            raise argparse.ArgumentTypeError(f'Invalid absolute cutoff: {val}')
    raise argparse.ArgumentTypeError(f'Invalid cutoff format: "{val}". Use "auto", "rel:<float>", or "abs:<int>".')

def load_history(history_file: Path) -> Dict[Tuple[int, int], Any]:
    """Loads and deserializes the iteration history."""
    with open(history_file, 'r') as f:
        # Convert string keys back to tuples (int, int)
        return {ast.literal_eval(k): v for k, v in json.load(f).items()}

def start_up_prep(start_point: Union[str, int], base_output_dir: Path, logger: GrandmaLogger) -> Tuple[int, Dict, Path]:
    """Handles folder cleanup and history loading before the engine starts."""
    i = 0
    iter_dirs = []
    history = {}
    history_file = base_output_dir / 'history.json'

    if base_output_dir.exists():
        if history_file.exists():
            history = load_history(history_file)
            iter_dirs = sorted(base_output_dir.glob('*/output/'), key=lambda x: int(x.parent.name))
        else:
            logger.write(f"No history file found at {history_file}. Starting from scratch.", level=1)
    else:
        logger.write(f"Previous output directory {base_output_dir} does not exist. Starting from scratch.", level=1)
    
    # Resolve start point
    if isinstance(start_point, int):
        if start_point < 1:
            raise ValueError(f"--start must be a strictly positive integer or 'auto', got {start_point}.")
        i = start_point - 1  # We'll read from iteration (start-1)
        
        if (i+1, 0) not in history:
            raise RuntimeError(f'Missing history entry for iteration {i}. Cannot resume at iteration {i+1}.')

        logger.write(f"Resuming from iteration {i+1} (manually specified).", level=1)
    else:
        # Auto-detect: try to infer starting iteration from history
        if iter_dirs:
            for iter_dir in reversed(iter_dirs):
                j = int(iter_dir.parent.name) + 1
                if (j, 0) in history:
                    i = j
                    logger.write(f"Auto-detected resume point: Iteration {i+1}.", level=1)
                    break

    # Clean up folders from iteration i onward for fresh resume
    for iter_dir in iter_dirs:
        j = int(iter_dir.parent.name)
        if j >= i:
            logger.write(f'Removing existing directory from previous run: {iter_dir.parent}', level=1)
            shutil.rmtree(iter_dir.parent, ignore_errors=True)

    # Clean up history keys
    keys_to_delete = [k for k in history if k[0] > i]
    for k in keys_to_delete:
        del history[k]
    if keys_to_delete and history_file.exists():
        logger.write(f'Removed history entries: {keys_to_delete}', level=1)
        with open(history_file, 'w') as f:
            json.dump({str(k): v for k, v in history.items()}, f, indent=4)

    return i, history, history_file

def check_loop_length(n: int, i: int, st_file: Path, history: Dict[Tuple[int, int], Any], logger: GrandmaLogger) -> Union[int, float]:
    """Determines the true loop length based on input and history."""
    # If n is non-positive, run while True, otherwise run n times
    if n <= 0:
        logger.write('\nImportant: -i (--iter) is set to 0 or less, running indefinitely until no new H-nodes are found.', level=1)
        n = float('inf')
    if i > 0:
        if history[(i, 0)]['gt_file'] == 'NA' and st_file.exists():
            logger.write(f'\nPrevious run was completed. No further iterations needed.', level=1)
            n = -1
    return n

def resolve_mode_logic(args: argparse.Namespace, logger: GrandmaLogger) -> str:
    """
    Consolidates modern --mode and legacy flags into a single mode string.
    Returns resolved mode.
    """
    lca_opt = "default"
    if args.buildmultrees: lca_opt = "build-mts"
    elif args.checknums: lca_opt = "check-nums"
    elif args.st_only: lca_opt = "st-only"
    elif args.no_st: lca_opt = "no-st"
    if args.buildmultrees + args.checknums + args.st_only + args.no_st > 1:
        logger.write("Warning: Only a single legacy lca_opt flag (build-mts, checknums, no-st, st-only) can be applied! One will be chosen according to precedence.", level=1)

    # Priority 1: Direct --mode selection (if not single)
    if args.mode != "single" and args.mode != lca_opt:
        logger.write("Warning: --mode flag takes precedence over legacy lca_opt flags (build-mts, checknums, no-st, st-only) and will override them!", level=1)
        return args.mode
    
    # Priority 2: If mode is single and no legacy flags are set
    if lca_opt == "default":
        return "single"
        
    return lca_opt


# --- Main Parsing ---

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
    parser.add_argument("-v", dest="verbosity", type=int, default=3, choices=range(4), help="Verbosity (0-3)")

    # Mode Options
    parser.add_argument("-m", "--mode", choices=["single", "split", "full", "build-mts", "checknums", "no-st", "st-only"], default="single",
                        help="Execution mode (single, split, full, build-mts, checknums, no-st, st-only")
    parser.add_argument('-i', '--iter', type=int, default=0, help='Number of iterations; <int>, non-positive for infinite mode')
    parser.add_argument('--prep', type=str, help='Preprocess input files; "0/D/default" for default settings, or <path> for a config json')
    parser.add_argument('--start', type=str, default='auto', help='Start point when finishing a previous execution; positive <int>, or "auto" for auto-detection')
    parser.add_argument('--cutoff', type=str, default='auto', help='Stopping condition mode; "auto" for abs:0+lookback, "rel:<float>" for relative, or "abs:<int>" for absolute')
    parser.add_argument('--ignore-nesting', action='store_true', help='Do not automatically fix nested hybridization events; let GRAMPA iterate normally')

    parser.add_argument('--plot', action='store_true', help='Plot taxon count, MP score, and normalized score over iterations')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode for additional output')

    # Legacy mode flags
    parser.add_argument("--buildmultrees", action="store_true", help="Only build MUL-trees and exit")
    parser.add_argument("--checknums", action="store_true", help="Count groups and exit")
    parser.add_argument("--no-st", action="store_true", help="Skip singly-labeled tree")
    parser.add_argument("--st-only", action="store_true", help="Only run singly-labeled tree")

    # Other legacy flags
    parser.add_argument("--maps", action="store_true", help="Output detailed maps")
    parser.add_argument("--orthologies", action="store_true", help="Run orthology labeling (Beta)")
    parser.add_argument("--force", action="store_true", dest="overwrite", help="Overwrite existing output")
    
    # Compatibility flags (ignored or handled implicitly)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output")

    '''
    # Required Groups
    req = parser.add_argument_group("Required Arguments")
    req.add_argument("-s", dest="spec_tree", required=True, help="Species tree file")
    req.add_argument("-g", dest="gene_tree", help="Gene tree file (required for analysis)")
    
    # Execution Modes
    modes = parser.add_argument_group("Execution Modes")
    modes.add_argument("-m", "--mode", choices=["single", "split", "full", "build-mts", "check-nums", "no-st", "st-only"], 
                       default="single", help="Analysis workflow mode")
    modes.add_argument('-i', '--iter', type=int, default=0, help='Max iterations (0 for infinite)')
    modes.add_argument('--start', default='auto', help='Resume point ("auto" or iteration number)')
    modes.add_argument('--cutoff', default='auto', help='Stopping condition (e.g., auto, rel:0.01, abs:5)')
    
    # Algorithm Logic
    algo = parser.add_argument_group("Algorithm Options")
    algo.add_argument("-h1", dest="h1", help="Hybrid clade 1 (space/comma separated nodes)")
    algo.add_argument("-h2", dest="h2", help="Hybrid clade 2 (space/comma separated nodes)")
    algo.add_argument("-c", dest="cap", type=int, default=8, help="Maximum groups per gene tree")
    algo.add_argument("--ignore-nesting", action='store_true', help='Disable automatic nested fix detection')

    # Output / Performance
    out = parser.add_argument_group("Output & Performance")
    out.add_argument("-o", dest="outdir", help="Output directory")
    out.add_argument("-f", dest="prefix", default="grandma", help="Output filename prefix")
    out.add_argument("-p", dest="procs", type=int, default=1, help="Multiprocessing cores")
    out.add_argument("-v", dest="verbosity", type=int, default=3, choices=[0, 1, 2, 3], help="Verbosity level")
    out.add_argument("--force", "--overwrite", action="store_true", dest="overwrite", help="Overwrite existing files")
    out.add_argument('--plot', action='store_true', help='Generate metrics plot after iteration')

    # Legacy Flags (Compatibility)
    leg = parser.add_argument_group("Legacy Compatibility Flags")
    leg.add_argument("--buildmultrees", action="store_true", help="Alias for --mode build-mts")
    leg.add_argument("--checknums", action="store_true", dest="checknums_flag", help="Alias for --mode check-nums")
    leg.add_argument("--no-st", action="store_true", help="Alias for --mode no-st")
    leg.add_argument("--st-only", action="store_true", help="Alias for --mode st-only")
    leg.add_argument("--maps", action="store_true", help="Detailed mapping output")
    leg.add_argument("--orthologies", action="store_true", help="Pairwise orthology labeling")

    # for n_lowest maps ?
    parser.add_argument(
    "--maps", 
    dest="maps_opt", 
    nargs='?', 
    const=6,      # Default value if flag is present but empty
    type=int,     # Ensures the value is treated as an integer
    default=0,    # Value if the flag is not present at all
    help="Output detailed maps (optional: specify number of maps, default: 6)"
    )
    '''

    args = parser.parse_args()
    
    # Resolve Output Dir
    out_dir = args.outdir if args.outdir else f"grampa_out_{int(time.time())}" 
    out_dir = Path(out_dir).resolve()
    
    log_path = out_dir / f"{args.prefix}.log"
    logger = GrandmaLogger(log_path=log_path, verbosity=args.verbosity)

    # Resolve Mode Logic
    mode = resolve_mode_logic(args, logger)

    # Handle folder deletion and history loading BEFORE the engine starts
    start = args.start if args.start == 'auto' else int(args.start)
    i, history, history_file = start_up_prep(start, out_dir, logger)

    # Determine initial file paths based on iteration/resume point
    if i == 0:
        st_file = Path(args.spec_tree).resolve()
        gt_file = Path(args.gene_tree).resolve() if args.gene_tree else ""
    else:
        # Load the processed files from the PREVIOUS successful iteration
        st_file = out_dir / str(i-1) / 'multree.tre'
        gt_file = out_dir / str(i-1) / 'genetrees.txt'

    '''# Print setup info
    print(f'\nSetup:')
    print(f'Iterations: {max_iter} (provided start-point {start})')
    print(f'Cutoff: {mp_cutoff}')
    print(f'Handle Nested Hybridizations: {not ignore_nesting}')
    print(f'Preprocessing Config: {prep_config if prep_config else "None"}')
    print(f'Output Directory: {base_output_dir}')
    print(f'Plotting Enabled: {args.plot}')
    print(f'Debug Mode: {debug}')
    print(f'Other Args: {unknown_args}')
    '''

    # Determine true loop length
    max_iter = check_loop_length(args.iter, i, st_file, history, logger)

    # Optional: Data Preparation
    if args.prep:
        step = "Preprocessing"
        GrandmaLogger().report_step(step, "Preprocessing input files...", start=True)
        if i == 0:
            st_file, gt_file = DataPreparer.run(
                species_tree_path = st_file,
                gene_tree_path    = gt_file,
                output_dir        = out_dir,
                prep_config       = args.prep,
                logger            = logger
            )
            GrandmaLogger().report_step(step, "Success")
        else:
            GrandmaLogger().report_step(step, 'Skipped') ### TBD: logger
            print(f'Skipping preprocessing because a previous run was loaded.')

    # Instantiate the Immutable Config
    return GrandmaConfig(
        species_tree_path = st_file,
        gene_tree_path    = gt_file,
        output_dir        = out_dir,

        mode              = mode,
        cutoff            = parse_cutoff(args.cutoff),
        ignore_nesting    = args.ignore_nesting,
        debug             = args.debug,
        plot              = args.plot,
        log_path          = log_path,

        start_pt          = i,
        max_iter          = max_iter,
        history           = history,
        history_file      = history_file,

        run_prefix        = args.prefix,
        overwrite         = args.overwrite,
        h1_nodes          = args.h1,
        h2_nodes          = args.h2,
        group_cap         = args.cap,
        num_processes     = args.procs,
        verbosity         = args.verbosity,

        # is_mul_input defaults to False, logic handled by mode usually
        maps_opt          = args.maps,
        orth_opt          = args.orthologies,
        pickle_dir        = Path(out_dir) / "pkls/",
        n_lowest          = 6
    )

class DataPreparer:
    @staticmethod
    def run(species_tree_path: str, gene_tree_path: str, output_dir: str, prep_config: str = None, logger: GrandmaLogger = None) -> Tuple[str, str]:
        """
        Preprocesses input files to ensure they are rooted, bifurcating, and have unique leaf IDs.
        """
        cnfg = {
            'gt': {
                'sfx_lookup': '.treefile', #None, #'.tre',
                'clean_fn': DataPreparer._gt_clean_fn,
                'name_lambda': lambda x: x.replace('_', '-'), #lambda x: x.rsplit('_', 1)[1],
                'out_fmt': 9,  # Leafs only
            },
            'st': {
                'clean_fn': DataPreparer._st_clean_fn,
                'name_lambda': lambda x: x.replace('_', '-'),
                'out_fmt': 9,  # Leafs only
            }
        }

        # Parse config override
        if prep_config and prep_config not in ['0', 'D', 'default']:
            if os.path.isfile(prep_config):
                with open(prep_config, 'r') as f:
                    cnfg_ = json.load(f)
                    for key in cnfg_:
                        cnfg[key] = cnfg_[key]
            else:
                raise ValueError(f'Error: Preprocessing config {prep_config} is not valid.')

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Process Gene Trees
        new_gt_path = DataPreparer._process_treefiles(gene_tree_path, out_dir, **cnfg['gt'])
        
        # Process Species Tree
        new_st_path = DataPreparer._process_spec_tree(species_tree_path, out_dir, **cnfg['st'])

        return str(new_st_path), str(new_gt_path)

    @staticmethod
    def _gt_clean_fn(tree):
        """Clean a tree by removing polytomies and ensuring it is rooted."""
        def root_tree(t):
            # Placeholder for rooting logic if needed (ETE3 usually handles unrooted okay for some ops)
            tre = t.get_tree_root()
            #t.unroot()
            return tre

        tree.resolve_polytomy(recursive=True)
        if ancestral_node := root_tree(tree):
            #tree.set_outgroup(ancestral_node)
            return True
        return False

    @staticmethod
    def _st_clean_fn(x, step='pre'):
        if step == 'pre':
            # Remove everything between "'[" and "]'" (legacy cleaning)
            return re.sub(r"'\[.*?\]'", '', x)
        return x

    @staticmethod
    def _process_treefiles(path, out_dir, clean_fn, name_lambda, sfx_lookup='.treefile', out_fmt=9):
        path = Path(path)
        tree_txts = []
        
        if sfx_lookup and path.is_dir():
            # Directory mode
            files = glob(os.path.join(path, f'*{sfx_lookup}'))
            if not files:
                print(f"No gene tree files found in directory '{path}' with suffix '{sfx_lookup}'.") ### TBD: logger
                return path
            tree_txts = [Path(t).read_text().rstrip() for t in files]
        else:
            # File mode
            if not path.exists():
                print(f"Gene tree file '{path}' does not exist.") ### TBD: logger
                return path 
            tree_txts = path.read_text().splitlines()

        if not tree_txts:
            print(f"No gene trees found in '{path}'.") ### TBD: logger
            return path

        clean_lines = []
        fix_semicolon = lambda x: x if x.endswith(';') else x + ';'

        for i, txt in enumerate(tree_txts):
            try:
                t = Tree(fix_semicolon(txt.strip()))
                if not clean_fn(t):
                    print(f"Skipping tree {i+1} due to cleanup failure.") ### TBD: logger
                    continue
                
                leaf_counts = {}
                for node in t.traverse():
                    if node.name:
                        if node.is_leaf():
                            clean_name = name_lambda(node.name)
                            # Ensure unique names
                            leaf_counts[clean_name] = leaf_counts.get(clean_name, 0) + 1
                            node.name = f'{leaf_counts[clean_name]}_{clean_name}'
                        else:
                            node.name = None
                clean_lines.append(t.write(format=out_fmt))
            except Exception as e:
                print(f"Error processing tree {i+1}: {e}") ### TBD: logger
                continue
        print(f'Processed {len(clean_lines)} gene trees.') ### TBD: logger
        
        out_path = out_dir / (path.stem + '.txt')
        with open(out_path, 'w') as f:
            f.write("\n".join(clean_lines))
        
        return out_path

    @staticmethod
    def _process_spec_tree(path, out_dir, clean_fn, name_lambda, out_fmt=9):
        path = Path(path)
        if not path.exists():
            print(f"Spec tree file '{path}' does not exist.") ### TBD: logger
            return path
        
        txt = path.read_text().strip()
        txt = clean_fn(txt if txt.endswith(';') else txt+';', step='pre')
        
        t = Tree(txt, format=1)
        for node in t.traverse():
            if node.name:
                if node.is_leaf():
                    node.name = name_lambda(node.name)
                else:
                    node.name = None
        
        # Post-clean (noop by default)
        t = clean_fn(t, step='post')
        
        out_path = out_dir / (path.stem + '.tre')
        with open(out_path, 'w') as f:
            f.write(t.write(format=out_fmt))
        
        print(f'Processed the species tree.') ### TBD: logger
        return out_path

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