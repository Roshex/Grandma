'''
Replaces params.py and global_vars.py. Holds constants and configuration dataclasses.
Handles the input parsing logic from opt_parse.py and spec_tree.py
'''

import ast
import json
import shutil
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union, List
from dataclasses import dataclass, field, replace, fields

from .models import SmrtTree, Map
from .logger import GrandmaLogger

import datetime

# Replicating the dynamic default logic from the original opt_parse.py
def get_default_outdir() -> str:
    return "grandma_out_" + datetime.datetime.now().strftime("%m-%d-%Y.%I-%M-%S")

'''
missing:

better mem and cpu/processing options?

'''

# --- Package Metadata (Literal) ---
@dataclass(frozen=True, slots=True)
class GrandmaMetadata:
    """Immutable metadata about the GRANDMA software."""
    authors: str = "Ronen Shtein"
    doi: str = "TBD"
    github: str = "TBD"
    http: str = "TBD"
    release: str = "TBD 2026"
    version: str = "2.6.0 (Modern)"

    # GRAMPA Source Metadata
    source_authors: str = "Gregg Thomas, S. Hussain Ather, Matthew Hahn"
    source_doi: str = "https://doi.org/10.1093/sysbio/syx044"
    source_github: str = "https://github.com/gwct/grampa"
    source_http: str = "https://gwct.github.io/grampa/"
    source_release: str = "June 2024"
    source_version: str = "1.4.0"

# --- Global Environment (Static) ---
@dataclass(frozen=True, slots=True)
class GlobalContext:
    """
    Environmental settings that remain constant throughout the application lifecycle.
    Passed to the Engine/Run classes upon initialization.
    """
    # System Resources
    num_processes: int = 1
    verbosity: int = 3
    seed: int = 42
    
    # Global Flags
    plot: bool = False
    norun: bool = False
    nolog: bool = False
    debug: bool = False

    # Global Algorithm Options
    orth_opt: bool = False
    max_iter: Union[int, float] = 0
    cutoff: Tuple[str, Optional[Union[int, float]]] = ("auto", None)
    ignore_nesting: bool = False # for full mode
    min_st_lvs: int = 1 # for split mode
    min_gt_lvs: int = 2 # for split mode
    
    # Paths that define the "Session"
    root_dir: Path = field(default_factory=lambda: Path(get_default_outdir()))
    log_file: Optional[Path] = None
    
    # History Tracking (Global State)
    history: Dict[Tuple[Any, int], Any] = field(default_factory=dict)
    start_pt: int = 0

    @property
    def history_file(self) -> Path:
        """Dynamic property derived from the root output dir."""
        return self.root_dir / 'history.json'
    
    def update(self, **changes) -> 'GlobalContext':
        """
        Returns a new GlobalContext with specific fields updated.
        Validates that keys exist to prevent silent errors.
        """
        # Certain fields should not be allowed to be changed
        forbidden_keys = {"seed", "plot", "norun", "nolog", "orth_opt", "max_iter",
        "min_gt_lvs", "min_st_lvs", "root_dir", "log_file", "history", "start_pt"}
        # Safety check to ensure we aren't inventing new fields
        valid_fields = {f.name for f in fields(self)}
        for key in changes:
            if key not in valid_fields:
                raise TypeError(f"GlobalContext got an unexpected keyword argument '{key}'")
            if key in forbidden_keys:
                raise ValueError(f"GlobalContext field '{key}' is not allowed to be changed")

        return replace(self, **changes)


# --- Unit of Reconciliation (Dynamic) ---
@dataclass(frozen=True, slots=True)
class TaskConfig:
    """
    Configuration for a SINGLE execution Step.
    Contains sanitized arguments specific to one reconciliation task (~Grampa).
    """
    # I/O for this specific step
    output_dir: Path
    st: Union[str, Path, SmrtTree]
    gts: Optional[Union[str, Path, Dict[int, SmrtTree]]] = None
    run_prefix: str = "grandma"

    # Mode Targets
    mode: str = "single"
    overwrite: bool = False
    repair: bool = False
    
    # Algorithm Tunables
    h1_nodes: Optional[Union[str, List[str]]] = None
    h2_nodes: Optional[Union[str, List[str]]] = None
    ploidies: Optional[Union[Path, str, Dict[str, int]]] = None
    group_cap: int = 8
    to_map: int = 0 # False, True, int for max maps to keep, -1 for all
    max_select: int = 1 # max mts to select for processing per Run (non-overlapping H clades & scoring above ST)

    # Legacy Flags
    is_mul_input: bool = False

    @property
    def pickle_dir(self) -> Path:
        """Dynamic property derived from the step's output dir."""
        return self.output_dir / "pkls"
    
    @property
    def log_file(self) -> Path:
        """Dynamic property for the log file path."""
        return self.output_dir / f"{self.run_prefix}.log"
    
    def update(self, **changes) -> 'TaskConfig':
        """
        Returns a new TaskConfig with specific fields updated.
        Validates that keys exist to prevent silent errors.
        """
        # Certain fields should not be allowed to be changed
        forbidden_keys = {'group_cap', 'to_map', 'max_select', 'run_prefix'}
        # Safety check to ensure we aren't inventing new fields
        valid_fields = {f.name for f in fields(self)}
        for key in changes:
            if key not in valid_fields:
                raise TypeError(f"TaskConfig got an unexpected keyword argument '{key}'")
            if key in forbidden_keys:
                raise ValueError(f"TaskConfig field '{key}' is not allowed to be changed")

        return replace(self, **changes)

# --- Utility Functions ---

def load_history(history_file: Path) -> Dict[Tuple[Any, int], Any]:
    """Loads and deserializes the iteration history."""
    with open(history_file, 'r') as f:
        # Convert string keys back to tuples. 
        # Note: Split mode keys might be (depth, idx) or strings in future, 
        # but ast.literal_eval handles "(0, 0)" safely.
        return {ast.literal_eval(k): v for k, v in json.load(f).items()}

def start_up_prep_2(start_point: Union[str, int], base_output_dir: Path, logger: GrandmaLogger, mode: str = "full") -> Tuple[int, Dict, Path]:
    """
    Handles folder cleanup and history loading.
    Supports both integer-based iteration folders (Full) and Depth.Index folders (Split).
    """
    history = {}
    history_file = base_output_dir / 'history.json'
    
    # Check for resume
    is_resume = False
    if base_output_dir.exists() and history_file.exists():
        history = load_history(history_file)
        is_resume = True
    elif not base_output_dir.exists():
        logger.write(f"Creating new output directory at {base_output_dir}", level=2)
    else:
        logger.write(f"No history file found at {history_file}. Starting from scratch.", level=1)

    # Resolve start point logic
    if mode == "split":
        # For split mode, 'start_point' is less about a linear index and more about cleaning up partial runs.
        # We assume auto-resume if history exists.
        # Clean up any folder that is NOT in history? Or just trust history?
        # A simple approach: Trust history. If user passes --start 0, we wipe everything.
        if isinstance(start_point, int) and start_point == 0:
             logger.write("Split mode: --start 0 implies fresh run. Wiping history.", level=1)
             is_resume = False
             history = {}
             # shutil.rmtree(base_output_dir) # Dangerous? Better to just clear history and let overwrite handle it.
             # Actually, we should clear folders to avoid confusion.
             if base_output_dir.exists():
                for item in base_output_dir.iterdir():
                    if item.is_dir(): shutil.rmtree(item)
                    elif item.name == "history.json": item.unlink()

        return 0, history, history_file

    # Full Mode Logic (Linear 0, 1, 2...)
    i = 0
    iter_dirs = sorted(base_output_dir.glob('*/output/'), key=lambda x: int(x.parent.name) if x.parent.name.isdigit() else -1)
    
    if isinstance(start_point, int):
        if start_point < 1:
             # Manual override to 0 means fresh start
             logger.write("Manual start at 0. Wiping previous history.", level=1)
             i = 0
             keys_to_delete = list(history.keys()) # Delete all
        else:
            i = start_point - 1
            if (i, 0) not in history and is_resume:
                # If we want to start at 5, we need history for 4.
                # If history is empty, we can't start at 5.
                raise RuntimeError(f'Missing history entry for iteration {i}. Cannot resume at iteration {i+1}.')
            logger.write(f"Resuming from iteration {i+1} (manually specified).", level=1)
            # We delete everything AFTER i
            keys_to_delete = [k for k in history if isinstance(k[0], int) and k[0] > i]
    else:
        # Auto-detect
        if iter_dirs and is_resume:
            # Find last valid entry
            # We look for the highest folder that is present in history
            valid_i = 0
            for iter_dir in reversed(iter_dirs):
                if iter_dir.parent.name.isdigit():
                    j = int(iter_dir.parent.name)
                    if (j, 0) in history:
                        valid_i = j
                        break
            i = valid_i + 1
            logger.write(f"Auto-detected resume point: Iteration {i+1}.", level=1)
            keys_to_delete = [k for k in history if isinstance(k[0], int) and k[0] >= i] # Should be empty usually
        else:
            keys_to_delete = []

    # Execute Cleanup
    # 1. Folders
    for item in base_output_dir.iterdir():
        if item.is_dir() and item.name.isdigit():
            if int(item.name) >= i and i > 0: # Don't delete 0 if we are starting at 0, overwrites happen later
                 logger.write(f'Removing future directory: {item}', level=1)
                 shutil.rmtree(item, ignore_errors=True)
            elif i == 0 and item.name.isdigit():
                 shutil.rmtree(item, ignore_errors=True)

    # 2. History
    for k in keys_to_delete:
        del history[k]
    
    if keys_to_delete and history_file.exists():
        with open(history_file, 'w') as f:
            json.dump({str(k): v for k, v in history.items()}, f, indent=4)

    return i, history, history_file

def start_up_prep(start_point: Union[str, int], base_output_dir: Path, logger: GrandmaLogger, mode: str) -> Tuple[int, Dict, Path]:
    """
    Handles folder cleanup and history loading before the engine starts.
    Robustly handles both Integer folders (Full mode) and Dot-Notation folders (Split mode).
    """
    i = 0
    history = {}
    history_file = base_output_dir / 'history.json'

    # 1. Load History if exists
    if base_output_dir.exists():
        if history_file.exists():
            try:
                history = load_history(history_file)
                logger.write(f"Loaded existing history ({len(history)} entries).", level=1)
            except Exception as e:
                logger.write(f"Warning: Failed to load history file ({e}). Starting fresh.", level=1)
                history = {}
        else:
            logger.write(f"No history file found at {history_file}. Starting from scratch.", level=1)
    else:
        logger.write(f"Previous output directory {base_output_dir} does not exist. Starting from scratch.", level=1)
    
    # 2. Resolve Start Point & Mode Logic
    if mode == "split":
        # Split mode resumes based on History content (Graph Traversal), not a linear index.
        # We generally DO NOT delete folders in split mode unless forced, to preserve the recursion tree.
        if start_point == 0: # Explicit restart request
             logger.write("Split Mode: Explicit start at 0. Clearing previous history.", level=1)
             history = {}
             # We let existing folders be overwritten by the run logic
        return 0, history, history_file

    # --- Full Mode (Linear) Logic Below ---
    
    # Identify existing iteration folders (0, 1, 2...)
    # Filter out non-integer folders (like 'pkls' or split folders)
    iter_dirs = []
    if base_output_dir.exists():
        for d in base_output_dir.iterdir():
            if d.is_dir() and d.name.isdigit():
                iter_dirs.append(d)
    iter_dirs.sort(key=lambda x: int(x.name))

    # Determine 'i' (Next Iteration Index)
    if isinstance(start_point, int):
        if start_point < 1:
            # Explicit restart
            logger.write("Manual start at 0. Wiping previous history.", level=1)
            i = 0
        else:
            # Resume at specific point
            i = start_point - 1  # We read output from (start-1) to begin (start)
            # Check if required history exists
            if (i, 0) not in history:
                logger.write(f"Warning: Missing history for iteration {i}. Cannot resume perfectly. Starting at {i} anyway.", level=1)
            logger.write(f"Resuming from iteration {i+1} (manually specified).", level=1)
    else:
        # Auto-detect from folders + history
        # Find the highest folder index that is ALSO in history
        valid_i = -1
        for folder in reversed(iter_dirs):
            idx = int(folder.name)
            if (idx, 0) in history:
                valid_i = idx
                break
        
        if valid_i >= 0:
            i = valid_i + 1
            logger.write(f"Auto-detected resume point: Iteration {i}.", level=1)
        else:
            i = 0

    # 3. Cleanup Future Data (Full Mode Only)
    # Delete folders >= i (The iteration we are about to run should be fresh)
    # Exception: If we have pickles for 'i', we might want to keep them?
    # Ops.py handles overwrite logic. Ideally we keep the folder but clear the history key.
    
    # Clean History Keys > i-1 (We keep history UP TO the previous completed run)
    keys_to_delete = [k for k in history if isinstance(k[0], int) and k[0] >= i]
    
    # If resuming, we usually want to start 'i' fresh, so we might delete folder 'i'
    # But to support "Resume using pickles", we must NOT delete folder 'i'.
    # We only delete folders > i.
    for iter_dir in iter_dirs:
        j = int(iter_dir.name)
        if j > i:
            logger.write(f'Removing future directory: {iter_dir}', level=1)
            shutil.rmtree(iter_dir, ignore_errors=True)

    # Update History File
    if keys_to_delete:
        for k in keys_to_delete:
            del history[k]
        if history_file.exists():
            logger.write(f'Pruning history entries >= {i}', level=1)
            with open(history_file, 'w') as f:
                json.dump({str(k): v for k, v in history.items()}, f, indent=4)

    return i, history, history_file

def check_loop_length(n: int, i: int, st_file: Path, history: Dict, logger: GrandmaLogger) -> Union[int, float]:
    """Determines the true loop length based on input and history."""
    # If n is non-positive, run while True, otherwise run n times
    if n <= 0:
        logger.write('\nImportant: -i (--iter) is set to 0 or less, running indefinitely until no new H-nodes are found.', level=1)
        n = float('inf')
    if i > 0:
        if history[(i, 0)]['gt_file'] == 'NA' and st_file.exists():
            logger.write(f'\nPrevious run was completed. No further iterations needed.', level=1)
            n = -1
    '''
    # Check if the "previous" run actually signaled completion
    # In history, if 'gt_file' is 'NA', it means no genes were mapped or valid, often implying end of road.
    # Adjust logic based on how flow.py writes history.
    if i > 0 and (i-1, 0) in history:
        if history[(i-1, 0)].get('gt_file') == 'NA' and st_file.exists():
            logger.write(f'\nPrevious run seems completed (No GTs passed forward). No further iterations needed.', level=1)
            n = -1
    '''

    return n





# --- Main Parsing ---

class InitParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="GRANDMA: Gene-tree Reconciliation Algorithm with MUL-trees.", ###### TBD
            epilog="For full documentation, visit: TBD", ###### TBD
            formatter_class=argparse.RawTextHelpFormatter
        )
        self._add_arguments()
        self.logger = None

    def _add_arguments(self):
        # --- Required Inputs ---
        g_required = self.parser.add_argument_group("Required Inputs")
        g_required.add_argument("-s", dest="spec_input", required=True, type=str,
            help="A file or string containing a newick formatted species tree on which to search "
                 "for polyploid events. May be a singly-labeled or multi-labeled tree.")

        # --- Base Options ---
        g_general = self.parser.add_argument_group("General Options")
        g_general.add_argument("-g", dest="genes_input", default=None, type=str,
            help="A file or string containing one or more newick formatted gene trees to reconcile. Known gene "
                 "names if present should be in the format 'geneID_speciesID', where IDs are separated by the 1st "
                 "underscore. May be a singly-labeled or multi-labeled tree (required unless --buildmultrees).")
        g_general.add_argument("-o", "--out", dest="outdir", default=get_default_outdir(), type=str,
            help="Output directory. If it does not exist, it will be created. Default = 'grandma_out_' + timestamp.")
        g_general.add_argument("-f", "--prefix", default="grandma", type=str,
            help="A string prepended to all output files. Default = 'grandma'.")
        g_general.add_argument("-p", "--procs", type=int, default=1,
            help="Number of processes to use for parallelizable tasks. Default = 1.")
        g_general.add_argument("-v", "--verbosity", type=int, default=3, choices=range(4),
            help="Level of verbosity printed to the screen. 0 = none; 1 = run info; 2 = standard; "
                 "3 = loud / soft debugging. Default = 3.")

        # --- Algorithmic Options ---
        g_algo = self.parser.add_argument_group("Algorithmic Options")
        g_algo.add_argument("-h1", "--h1", type=str,
            help="A space separated nodes/leaves list of one or more node labels from the species tree to be "
                 "considered as parental node 1 (H1). If no labels are specified, all nodes "
                 "in the species tree are considered.")
        g_algo.add_argument("-h2", "--h2", type=str, help="As -h1, but for parental node 2 (H2).")
        g_algo.add_argument("-x", "--ploidy", type=str, default=None,
            help="Ploidy file formatted as a Polyphest Multiset file. If provided, H1 and H2 nodes will be enforced by "
            "ploidy levels. Default: None.")
        g_algo.add_argument("-c", "--cap", type=int, default=8,
            help="The maximum number of groups a gene tree is allowed to have. A gene tree with more than --cap "
                 "number of groups for a given MUL-tree, will be skipped. Default = 8.")
        g_algo.add_argument("-n", "--max_select", type=int, default=1,
            help="Maximum MUL-trees to select per run for parallel inference heuristics. Default: 1, i.e., only the best "
            "scoring MT is considered per iteration.")
        g_algo.add_argument("--min_st_lvs", type=int, default=1,
            help="Minimum species tree leaves for a species tree to be considered valid. Specifically relevant for the "
            "split mode. Default: 1.")
        g_algo.add_argument("--min_gt_lvs", type=int, default=2,
            help="Minimum gene trees leaves per tree for a gene tree to be considered valid. Specifically relevant for "
            "the split mode. Default: 2.")

        # --- Flow Control Options ---
        g_flow = self.parser.add_argument_group("Flow Control Options")
        g_flow.add_argument("-m", "--mode", type=str, default="single",
            choices=["single", "split", "full", "build-mts", "check-nums", "no-recon", "no-st", "st-only"], 
            help="Execution mode. Options: single => simple run [default] | split => parallelized binary-recursive | "
            "full => fully sequencial with nestedness inference | build-mts => build MUL-trees only | check-nums => "
            "count groups only | no-recon => build MTs & count groups | no-st => skip reconciliation to input | st-only "
            "=> reconciliation to input only.")
        g_flow.add_argument('-i', '--iter', type=int, default=0,
            help="Maximun number of iterations for iterative modes; <int>, non-positive to be unlimited. Default = 0.")
        g_flow.add_argument('-r', '--repair', action='store_true',
            help="If set, attempt to repair input files by forcing bifurcating trees, rooting, valid tip names, and more.")
        g_flow.add_argument('--start', type=str, default='auto',
            help="Start point when resuming a previous execution; positive <int>, or 'auto' [default] for auto-detection.")
        g_flow.add_argument('--cutoff', type=str, default='auto',
            help="Stopping condition when comparing MP score; 'auto' [default] for abs:0+lookback, 'abs:<int>' for "
            "absolute, or 'rel:<float>' for relative.")
        g_flow.add_argument('--ignore-nesting', action='store_true',
            help="If set, do not automatically fix nested hybridization events; let GRANDMA iterate normally without "
            "corrections. Ignored in all modes except 'full'.")
        g_flow.add_argument("--orthologies", action="store_true",
            help="If set, will output an additional file containing the pairwise orthology "
                 "relationships for each gene tree to the lowest scoring MUL-tree.")

        # --- Output Options ---
        g_output = self.parser.add_argument_group("Output Options")
        g_output.add_argument("--maps", nargs='?', const=1, default=0, type=int,
            help="If set, the detailed output file will contain node mappings for each gene tree to the lowest "
                 "scoring MUL-tree. Specify number to retreive for multiple lowest MTs (default if present: 1, all: -1).")
        g_output.add_argument('--plot', action='store_true',
            help="Plot taxon count, MP score, and normalized score over iterations. Relevant only for iterative modes.")
        g_output.add_argument('--debug', action='store_true',
            help="Enable debug mode for additional outputs.")
        g_output.add_argument("--overwrite", action="store_true",
            help="If set, overwrite existing files in the output directory. Default: exit if files exist.")
        g_output.add_argument("--norun", action="store_true",
            help="If set, only print the run info and exit.")
        g_output.add_argument("--nolog", action="store_true",
            help="If set, do not write a log file.")
        g_output.add_argument('--seed', type=int, default=42,
            help="Random seed for sampling and reproducibility. Default = 42.")

        # --- Legacy Flags ---
        g_legacy = self.parser.add_argument_group("Legacy Support")
        g_legacy.add_argument("--multree", dest="is_mul_input", action="store_true",
            help="If set, the input species tree is parsed as a MUL-tree. The H1 and H2 nodes are inferred "
                 "from tree topology (parents of the 'H*' and 'H' clades). First iteration / single mode will perform "
                 "reconciliation only - i.e., no alternative MT search. Auto-detected for multi-labeled input.")
        g_legacy.add_argument("--buildmultrees", action="store_true",
            help="If set, only build the MUL-trees from the species tree and exit. "
                 "No reconciliation will be performed. Equivalent to -m build-mts.")
        g_legacy.add_argument("--checknums", action="store_true",
            help="If set, only count the number of groups in each gene tree for each MUL-tree and write them "
                 "to file. No reconciliation will be performed. Equivalent to -m check-nums.")
        g_legacy.add_argument("--no-st", dest="no_st", action="store_true",
            help="If set, the standard reconciliation (reconciling to the singly-labeled species tree) "
                 "will be skipped. Equivalent to -m no-st.")
        g_legacy.add_argument("--st-only", dest="st_only", action="store_true",
            help="If set, only the standard reconciliation (reconciling to the singly-labeled species tree) "
                 "will be performed. Equivalent to -m st-only.")

    def parse_cutoff(self, val: str) -> Tuple[str, Optional[Union[float, int]]]:
        """Parses stopping condition strings into a typed tuple."""
        if val == 'auto': 
            return ('auto', None)
        if val.startswith('rel:'):
            try: 
                return ('rel', float(val.split(':')[1]))
            except (ValueError, IndexError): 
                self.logger.write(f'Error: Invalid relative cutoff: {val}', level=0)
        if val.startswith('abs:'):
            try: 
                return ('abs', int(val.split(':')[1]))
            except (ValueError, IndexError): 
                self.logger.write(f'Error: Invalid absolute cutoff: {val}', level=0)
        self.logger.write(f'Error: Invalid cutoff format: "{val}". Use "auto", "rel:<float>", or "abs:<int>".', level=0)

    def resolve_mode_logic(self, mode, build_mts, check_nums, st_only, no_st) -> str:
        """
        Consolidates modern --mode and legacy flags into a single mode string.
        Returns resolved mode.
        """
        m = "default"
        if build_mts: m = "build-mts"
        elif check_nums: m = "check-nums"
        elif st_only: m = "st-only"
        elif no_st: m = "no-st"

        if build_mts and check_nums: m = "no-recon"
        elif sum([build_mts, check_nums, st_only, no_st]) > 1:
            self.logger.write("Warning: Multiple legacy [build-mts and/or check-nums, no-st, st-only] flags set! One will be chosen according to precedence.", level=1)

        # Priority 1: Direct --mode selection (if not single)
        if mode != "single" and mode != m:
            self.logger.write("Warning: --mode flag overrides legacy [build-mts, check-nums, no-st, st-only] flags!", level=1)
            return mode
        
        # Priority 2: If mode is single and no legacy flags are set
        if m == "default":
            return "single"
            
        return m

    def parse(self, args_=None) -> Tuple[GlobalContext, TaskConfig]:
        """
        Parses arguments and returns strictly typed configuration objects.
        """
        args = self.parser.parse_args(args_)
        
        # --- Setup Environment and Banner ---
        out_dir = args.outdir if args.outdir else get_default_outdir()
        out_dir = Path(out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = out_dir / f"{args.prefix}.log"
        self.logger = GrandmaLogger(log_path=log_file, verbosity=args.verbosity)

        self.logger.log_software_banner(GrandmaMetadata())
        self.logger.write("=" * 73, level=1)

        ####
        # Logging strategy:
        # log the original paths in the banner only!
        # split start run:
        # before parsing args: log original args
        # after parsing args: log sanitized args in tcf - when running the Task
        ###

        # --- Resolve Argument Logistics ---
        mode = self.resolve_mode_logic(
            args.mode,
            args.buildmultrees,
            args.checknums,
            args.st_only,
            args.no_st
        )

        # Handle folder deletion and history loading BEFORE the engine starts
        start = args.start if args.start == 'auto' else int(args.start)
        
        # Prepare history/folders
        i, history, history_file = start_up_prep(start, out_dir, self.logger, mode)

        # --- Build Global Context ---
        ctx = GlobalContext(
            num_processes=  args.procs,
            verbosity=      args.verbosity,
            seed=           args.seed,
            plot=           args.plot,
            norun=          args.norun,
            nolog=          args.nolog,
            debug=          args.debug,
            orth_opt=       args.orthologies,
            max_iter=       check_loop_length(args.iter, i, None, history, self.logger),
            cutoff=         self.parse_cutoff(args.cutoff),
            ignore_nesting= args.ignore_nesting,
            min_gt_lvs=     args.min_gt_lvs,
            min_st_lvs=     args.min_st_lvs,
            root_dir=       out_dir,
            log_file=       log_file,
            history=        history,
            start_pt=       i
        )

        # --- Prepare Step Config ---
        tcf = TaskConfig(
            output_dir=     ctx.root_dir,
            st=             args.spec_input,
            gts=            args.genes_input,
            run_prefix=     args.prefix,
            mode=           mode,
            overwrite=      args.overwrite,
            repair=         args.repair,
            h1_nodes=       args.h1,
            h2_nodes=       args.h2,
            ploidies=       args.ploidy,
            group_cap=      args.cap,
            to_map=         args.maps,
            max_select=     args.max_select,
            is_mul_input=   args.is_mul_input,
        )

        return ctx, tcf

class GrandmaWriter:
    def __init__(self, config: TaskConfig, logger: GrandmaLogger):
        self.tcf = config
        self.logger = logger

    def write_results(self, sorted_scores: list, detailed_res: dict, mul_trees: dict, gene_trees: dict):
        self._write_detailed(detailed_res, gene_trees)
        self._write_scores(sorted_scores, mul_trees)
        self._write_dup_counts(detailed_res, mul_trees)

    def _write_detailed(self, detailed_res: dict, gene_trees: dict):
        step = "Writing detailed output file"
        self.logger.report_step(step, "In progress...")
        
        p = Path(self.tcf.output_dir) / f"{self.tcf.run_prefix}-detailed.txt"
        with open(p, 'w') as f:
            f.write("mul.tree\tgene.tree\tdups\tlosses\ttotal.score\tmaps\n")
            for mul_idx, res_dict in detailed_res.items():
                for gene_idx, res in res_dict.items():
                    gt_obj = gene_trees[gene_idx]
                    # Handle multiple maps if present
                    if (maps_len := len(res.maps)) > 1:
                        f.write(f"# GT-{gene_idx} to MT-{mul_idx}\t{maps_len} maps found!\n")
                    for map in res.maps:
                        map_str = GrandmaWriter.detailed_out_string(gt_obj, map.cor, map.dups)
                        f.write(f"{mul_idx}\t{gene_idx}\t{map.n_dups}\t{map.n_losses}\t{res.score}\t{map_str}\n")
                         
        self.logger.report_step(step, "Success")

    @staticmethod
    def detailed_out_string(gt: SmrtTree, maps: Map, dups: dict) -> str:
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
        
        p = Path(self.tcf.output_dir) / f"{self.tcf.run_prefix}-scores.txt"
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

    def _write_dup_counts(self, detailed_res: dict, mul_trees: dict):
        p_dup = Path(self.tcf.output_dir) / f"{self.tcf.run_prefix}-dup-counts.txt"
        with open(p_dup, 'w') as f:
            f.write("mul.tree\tnode\tdups\n")
            for mul_idx, res_dict in detailed_res.items():
                hybrid_clade = mul_trees[mul_idx].h_clade
                main_dups = {}
                for g_idx, res in res_dict.items():
                    first_map = res.maps[0]
                    maps = first_map.cor
                    for gt_node, count in first_map.dups.items():
                        if count != 0:
                            map_node = maps[gt_node][0]
                            main_dups[map_node] = main_dups.get(map_node, 0) + 1
                            
                for node, count in main_dups.items():
                    out_node = node + "+" if node in hybrid_clade else node
                    f.write(f"{mul_idx}\t{out_node}\t{count}\n")