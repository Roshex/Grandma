'''
Replaces params.py and global_vars.py. Holds constants and configuration dataclasses.
Handles the input parsing logic from opt_parse.py and spec_tree.py
'''

from html import parser
import re
import os
import ast
import json
import shutil
import argparse
import multiprocessing
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union, List
from dataclasses import dataclass, field, replace, fields

from .logger import GranLogger
from .models import SmrtTree, Map, HistoryType

import datetime

# Replicating the dynamic default logic from the original opt_parse.py
def get_default_outdir() -> str:
    return "grandma_out_" + datetime.datetime.now().strftime("%m-%d-%Y.%I-%M-%S")

# --- Package Metadata (Literal) ---
@dataclass(frozen=True, slots=True)
class GranMetadata:
    """Immutable metadata about the GRANDMA software."""
    authors: str = "Ronen Shtein"
    doi: str = "TBD"
    github: str = "https://github.com/Roshex/Grandma"
    http: str = "TBD"
    release: str = "TBD 2026"
    version: str = "3.0.5"

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
    optim: bool = False

    # Global Flow/Algorithm Options
    orth_opt: bool = False
    max_iter: Union[int, float] = 0
    mixed_switch: int = 0
    cutoff: Tuple[str, Optional[Union[int, float]]] = ("auto", None)
    # For full mode
    # replaces ignore_nesting: bool = False -> ignore (False), rectify (True), strict_rectify (New), model (New)
    nesting: str = "model"
    # For split mode
    min_st_lvs: int = 1 
    min_gt_lvs: int = 2
      
    # Paths that define the "Session"
    root_dir: Path = field(default_factory=lambda: Path(get_default_outdir()))
    log_file: Optional[Path] = None
    
    # History Tracking (Global State)
    history: HistoryType = field(default_factory=dict)
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
        forbidden_keys = {"seed", "plot", "norun", "nolog", "orth_opt",
        "max_iter", "nesting", "mixed_switch", "min_gt_lvs", "min_st_lvs",
        "root_dir", "log_file", "history", "start_pt"}
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
    binary_id: Optional[int] = None # For split mode, identifies the current subproblem (the j number)
    predefined_rets: Dict[int, List[Tuple[str, str]]] = field(default_factory=dict)
    group_cap: int = 8
    weights: Tuple[int, int] = (1, 1) # (w_dup, w_loss)
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

def start_up_prep_2(start_point: Union[str, int], base_output_dir: Path, logger: GranLogger, mode: str = "full") -> Tuple[int, Dict, Path]:
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
        logger.log(f"Creating new output directory at {base_output_dir}", 's')
    else:
        logger.log(f"No history file found at {history_file}. Starting from scratch.", 'i')

    # Resolve start point logic
    if mode == "split":
        # For split mode, 'start_point' is less about a linear index and more about cleaning up partial runs.
        # We assume auto-resume if history exists.
        # Clean up any folder that is NOT in history? Or just trust history?
        # A simple approach: Trust history. If user passes --start 0, we wipe everything.
        if isinstance(start_point, int) and start_point == 0:
             logger.log("Split mode: --start 0 implies fresh run. Wiping history.", 'i')
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
             logger.log("Manual start at 0. Wiping previous history.", 'i')
             i = 0
             keys_to_delete = list(history.keys()) # Delete all
        else:
            i = start_point - 1
            if (i, 0) not in history and is_resume:
                # If we want to start at 5, we need history for 4.
                # If history is empty, we can't start at 5.
                raise RuntimeError(f'Missing history entry for iteration {i}. Cannot resume at iteration {i+1}.')
            logger.log(f"Resuming from iteration {i+1} (manually specified).", 'i')
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
            logger.log(f"Auto-detected resume point: Iteration {i+1}.", 'i')
            keys_to_delete = [k for k in history if isinstance(k[0], int) and k[0] >= i] # Should be empty usually
        else:
            keys_to_delete = []

    # Execute Cleanup
    # 1. Folders
    for item in base_output_dir.iterdir():
        if item.is_dir() and item.name.isdigit():
            if int(item.name) >= i and i > 0: # Don't delete 0 if we are starting at 0, overwrites happen later
                 logger.log(f'Removing future directory: {item}', 'i')
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

def start_up_prep(start_point: Union[str, int], base_output_dir: Path, logger: GranLogger, mode: str) -> Tuple[int, Dict, Path]:
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
                logger.log(f"Loaded existing history ({len(history)} entries).", 'i')
            except Exception as e:
                logger.log(f"Failed to load history file ({e}). Starting fresh.", 'w')
                history = {}
        else:
            pass
            #logger.log(f"No history file found at {history_file}. Starting from scratch.", 'i')
    else:
        logger.log(f"Previous output directory {base_output_dir} does not exist. Starting from scratch.", 'i')
    
    # 2. Resolve Start Point & Mode Logic
    if mode == "split":
        # Split mode resumes based on History content (Graph Traversal), not a linear index.
        # We generally DO NOT delete folders in split mode unless forced, to preserve the recursion tree.
        if start_point == 0: # Explicit restart request
             logger.log("Split Mode: Explicit start at 0. Clearing previous history.", 'i')
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
            logger.log("Manual start at 0. Wiping previous history.", 'i')
            i = 0
        else:
            # Resume at specific point
            i = start_point - 1  # We read output from (start-1) to begin (start)
            # Check if required history exists
            if (i, 0) not in history:
                logger.log(f"Missing history for iteration {i}. Cannot resume perfectly. Starting at {i} anyway.", 'w')
            logger.log(f"Resuming from iteration {i+1} (manually specified).", 'i')
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
            logger.log(f"Auto-detected resume point: Iteration {i}.", 'i')
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
            logger.log(f'Removing future directory: {iter_dir}', 'i')
            shutil.rmtree(iter_dir, ignore_errors=True)

    # Update History File
    if keys_to_delete:
        for k in keys_to_delete:
            del history[k]
        if history_file.exists():
            logger.log(f'Pruning history entries >= {i}', 'i')
            with open(history_file, 'w') as f:
                json.dump({str(k): v for k, v in history.items()}, f, indent=4)

    return i, history, history_file

def check_loop_length(n: int, i: int, st_file: Path, history: Dict, logger: GranLogger) -> Union[int, float]:
    """Determines the true loop length based on input and history."""
    # If n is non-positive, run while True, otherwise run n times
    if n <= 0:
        #logger.log('Important: -i (--iter) is set to 0 or less, running indefinitely until no new H-nodes are found.', 'i')
        n = float('inf')
    if i > 0:
        if history[(i, 0)]['gt_file'] == 'NA' and st_file.exists():
            logger.log(f'Previous run was completed. No further iterations needed.', 'i')
            n = -1
    '''
    # Check if the "previous" run actually signaled completion
    # In history, if 'gt_file' is 'NA', it means no genes were mapped or valid, often implying end of road.
    # Adjust logic based on how flow.py writes history.
    if i > 0 and (i-1, 0) in history:
        if history[(i-1, 0)].get('gt_file') == 'NA' and st_file.exists():
            logger.log(f'\nPrevious run seems completed (No GTs passed forward). No further iterations needed.', 'i')
            n = -1
    '''

    return n

def assess_restart_compatibility(logger, ctx, tcf) -> bool:
    """Pull the args from the log, if present.
    Then determine if you can restart, or if the args are incompatible.
    If no .log file, issue an error. [Log file not found. Cannot assess restart compatibility (not supported for --nolog runs).]
    """

    # use regex to get the line "# The program was called as: {args}\n"
    log_file = logger.log_file
    if not log_file or not log_file.exists():
        logger.log("No log file found. Cannot assess restart compatibility (not supported for --nolog runs).", 'e')
        return False
    with open(log_file, 'r') as f:
        log_content = f.read()
    match = re.search(r"# The program was called as: (.+)", log_content)
    if not match:
        logger.log("No call signature found in log file. Cannot assess restart compatibility.", 'e')
        return False
    old_args_str = match.group(1)
    old_ctx, old_tcf = InitParser().parser(old_args_str.split()) # unsafe as it may change the folders!
    # move the output changing funtionality out of parser and into main?

    # Compare relevant fields (those that affect the run logic and output structure)
    pass



    





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
        # --- Base Options ---
        g_general = self.parser.add_argument_group("General Options")
        g_general.add_argument("-s", dest="spec_input", default=None, type=str,
            help="A file or string containing a newick formatted species tree on which to search "
                 "for polyploid events. May be a singly-labeled or multi-labeled tree. Required argument, "
                 "unless resuming a previous run or step (such as --plot).")
        g_general.add_argument("-g", dest="genes_input", default=None, type=str,
            help="A file or string containing one or more newick formatted gene trees to reconcile. Known gene "
                 "names if present should be in the format 'geneID_speciesID', where IDs are separated by the 1st "
                 "underscore. May be a singly-labeled or multi-labeled tree (required unless --buildmultrees).")
        g_general.add_argument("-o", "--out", dest="outdir", default=get_default_outdir(), type=str,
            help="Output directory. If it does not exist, it will be created. Default = 'grandma_out_' + timestamp.")
        g_general.add_argument("-f", "--prefix", default="grandma", type=str,
            help="A string prepended to all output files. Default = 'grandma'.")
        g_general.add_argument("-p", "--procs", type=int, default=1,
            help="Number of processes to use for parallelizable tasks. Default = 1. Non-positive to autodetect and " 
                 "use all available cores.")
        g_general.add_argument("-v", "--verbosity", type=int, default=2, choices=range(5),
            help="Level of verbosity printed to the screen. 0 = none; 1 = run info; 2 = standard; "
                 "3 = debug; 4 = verbose debugging. Default = 2.")

        # --- Algorithmic Options ---
        g_algo = self.parser.add_argument_group("Algorithmic Options")
        g_algo.add_argument("-h1", "--h1", type=str, nargs='+',
            help="Node name or a comma-separated leaves list of one or more (space-separated) nodes from the species "
                 "tree to be considered as parental node 1 (H1). For each node given as a list, the LCA is taken. "
                 "If no labels are specified, all nodes in the species tree are considered.")
        g_algo.add_argument("-h2", "--h2", type=str, nargs='+', help="As -h1, but for parental node 2 (H2).")
        g_algo.add_argument("-x", "--ploidy", type=str, default=None,
            help="Ploidy file or string formatted as a line-separated counter of diploid subgenomes (e.g., 'A 1' = "
                 "A is at most diploid). If provided, H1 and H2 nodes will be enforced by ploidy levels. Default: None.")
        g_algo.add_argument("-c", "--cap", type=int, default=8,
            help="The maximum number of groups a gene tree is allowed to have. A gene tree with more than --cap "
                 "number of groups for a given MUL-tree, will be skipped. Default = 8 [to be raised to 15 on release].")
        g_algo.add_argument("-w", "--weights", type=int, nargs=2, default=[1, 1],
            help="Space-separated integer weights for the parsimony score calculation: 'w_dup w_loss'. Default = '1 1'.")
        g_algo.add_argument("-n", "--max_select", type=int, default=1,
            help="Maximum MUL-trees to select per run for parallel inference heuristics. Default: 1, i.e., only the best "
                 "scoring MT is considered per iteration.")
        g_algo.add_argument("--min_st_lvs", type=int, default=1,
            help="Minimum species tree leaves for a species tree to be considered valid. Specifically relevant for the "
                 "split mode. Default: 1.")
        g_algo.add_argument("--min_gt_lvs", type=int, default=2,
            help="Minimum gene trees leaves per tree for a gene tree to be considered valid. Specifically relevant for "
                 "the split mode. Default: 2.")
        g_algo.add_argument('--nesting', type=str, choices=['ignore', 'i', 'rectify', 'r', 'model', 'm', 'strict_rectify', 's'], default='model',
            help="Behavior for nested hybridization events treatment during the full and mixed modes. "
                 "(i)gnore: Do nothing, ignore nesting scenarios. "
                 "(r)ectify: Autocorrect nested events between iterations, including via sister relationships [default]. "
                 "(s)trict_rectify: Rectify, but only consider internally nested events when tracing missing subgenomes. "
                 "(m)odel: Model nested copies during MT creation. Computationally heaviest, but most exact.")
        
        g_algo.add_argument('--optim', dest='optim', action='store_true',
            help="If set, will run alternative algorithms for [1.] MT construction and [2.] reconciliation: " 
                "these are unrelated and can be mixed if merged into default. However, testing them I observe no significant speedup. "
                "[1.] may be worth to keep due to safety (?) but [2.] is over-engineered and seems to not be worth it at all.")

        # --- Flow Control Options ---
        g_flow = self.parser.add_argument_group("Flow Control Options")
        g_flow.add_argument("-m", "--mode", type=str, default="single",
            help="Execution mode. Supported options: "
                 "single: Simple run with a single iteration, equivalent to running Grampa [default]. "
                 "full: Fully sequential search allowing for nested hybridization event inference [computationally expensive]. "
                 "split: Binary split search reducing each depth into outer/inner subproblems [cheap; supports subproblem parallelism]. "
                 "mixed-<int>: Start with full mode, then switch to split mode after <int> iterations [default w/o int: mixed-3]. "
                 "label-sp: Only label input species tree internal nodes. "
                 "count-mts: Count possible MUL-trees only. "
                 "build-mts: Build MUL-trees only. "
                 "check-nums: Count groups only. "
                 "no-recon: Build MTs & count groups. "
                 "no-st: Skip reconciliation to input. "
                 "st-only: Reconciliation to input only.")
        g_flow.add_argument('-i', '--iter', type=int, default=0,
            help="Maximun number of iterations or (~depth) event num for iterative modes; <int>, non-positive to be unlimited. Default = 0.")
        g_flow.add_argument('-r', '--repair', action='store_true',
            help="If set, attempt to repair input files by forcing bifurcating trees, rooting, valid tip names, and more.")
        g_flow.add_argument('--start', type=str, default='auto',
            help="Start point when resuming a previous execution; positive <int>, or 'auto' [default] for auto-detection.")
        g_flow.add_argument('--cutoff', type=str, default='auto',
            help="Stopping condition when comparing MP score; 'auto' [default] for abs:0+lookback, 'abs:<int>' for "
                 "absolute, or 'rel:<float>' for relative.")
        g_flow.add_argument("--orthologies", action="store_true",
            help="If set, will output an additional file containing the pairwise orthology "
                 "relationships for each gene tree to the lowest scoring MUL-tree.")

        # --- Output Options ---
        g_output = self.parser.add_argument_group("Output Options")
        g_output.add_argument("--generate", type=str, metavar="JSON_CONFIG",
            help="Path to JSON configuration file. Enters Generation Mode (ignores other flags).")
        g_output.add_argument("--maps", nargs='?', const=1, default=0, type=int,
            help="If set, the detailed output file will contain node mappings for each gene tree to the lowest "
                 "scoring MUL-tree. Specify number to retreive for multiple lowest MTs (default if present: 1, all: -1).")
        g_output.add_argument('--plot', action='store_true',
            help="Plot taxon count, MP score, and normalized score over iterations. Relevant only for iterative modes.")
        g_output.add_argument('--debug', action='store_true',
            help="Enable debug mode for additional outputs to the log file (whereas --v 3 only prints debug messages to screen).")
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
        g_legacy.add_argument("--labeltree", action="store_true",
            help="If set, the program will read the species tree, print it with internal nodes labeled, and exit.")
        g_legacy.add_argument("--numtrees", action="store_true",
            help="If set, the program will count the number of possible MUL-trees from the inputs and exit.")
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
            return ('auto', 0)
        if val.startswith('rel:'):
            try: 
                return ('rel', float(val.split(':')[1]))
            except (ValueError, IndexError): 
                self.logger.log(f'Invalid relative cutoff: {val}', 'e')
        if val.startswith('abs:'):
            try: 
                return ('abs', int(val.split(':')[1]))
            except (ValueError, IndexError): 
                self.logger.log(f'Invalid absolute cutoff: {val}', 'e')
        self.logger.log(f'Invalid cutoff format: "{val}". Use "auto", "rel:<float>", or "abs:<int>".', 'e')

    def resolve_mode_logic(self, mode, label_sp, count_mts, build_mts, check_nums, st_only, no_st) -> Tuple[str, int]:
        """
        Consolidates modern --mode and legacy flags into a single mode string.
        Returns resolved mode and mixed switch value.
        """

        mixed_switch = 3 # default switch point
        if mode.startswith("mixed"):
            if '-' in mode:
                try:
                    _, mixed_switch = mode.split('-')
                    mixed_switch = int(mixed_switch)
                    assert mixed_switch > 0
                except (ValueError, AssertionError):
                    self.logger.log(f"Invalid mixed mode format: {mode}. Expected 'mixed' or 'mixed-<int>', where <int> is strictly positive.", 'e')
            mode = "mixed"

        m = "default"
        if label_sp: m = "label-sp"
        elif count_mts: m = "count-mts"
        elif build_mts: m = "build-mts"
        elif check_nums: m = "check-nums"
        elif st_only: m = "st-only"
        elif no_st: m = "no-st"

        if build_mts and check_nums: m = "no-recon"
        elif sum([label_sp, count_mts, build_mts, check_nums, st_only, no_st]) > 1:
            self.logger.log("Multiple legacy flags set! One will be chosen according to precedence.", 'w')

        # Priority 1: Direct --mode selection (if not single)
        if mode != "single" and mode != m:
            self.logger.log("--mode flag overrides legacy flags!", 'w')
            return mode, mixed_switch
        
        # Priority 2: If mode is single and no legacy flags are set
        if m == "default":
            return "single", mixed_switch
            
        return m, mixed_switch

    @staticmethod
    def resolve_nesting(val: str) -> str:
        if val in ['ignore', 'i']: return 'ignore'
        if val in ['rectify', 'r']: return 'rectify'
        if val in ['strict_rectify', 's']: return 'strict_rectify'
        if val in ['model', 'm']: return 'model'
        return 'rectify'
        
    def plot_and_exit(self):
        # plot hardcoded data here for convinience of testing

        import matplotlib.pyplot as plt
        import numpy as np
        import matplotlib.patches as mpatches
        import sys
        from matplotlib.ticker import FixedLocator

        # --- 1. Hardcoded Data ---
        y_label = "Elapsed time (s) [Log Scale]"
        # --- 1. Hardcoded Data ---
        #y_labels = ["Elapsed time (s)", "Elapsed time (h)"]
        x_label = "Step (computing the first event)"
        
        datasets_for_legend = ['Kalanchoe backbone', 'Bendiksby et al. 2011', 'Díaz-Pérez et al. 2018']
        tools_for_legend = ['Grampa', 'Grandma: base-mode']
        
        # Data structure: List of tuples corresponding to datasets. 
        # Tuple: (Time_Grampa, Time_Grandma)
        time_steps = {
            'Build MTs': [
                (0.4926, 0.16781),   # Dataset 1
                (10.36343, 5.13517),   # Dataset 2
                (1.58598, 0.41652)   # Dataset 3
            ],
            'Load GTs': [
                (1.34652, 0.22395),    # Dataset 1 (Usually similar I/O)
                (0.47741, 0.76329),    # Dataset 2
                (0.71326, 0.69493)   # Dataset 3
            ],
            'Collapse & Filter GTs': [
                (59.32991+4.95699, 0.02194+2.52801),    # Dataset 1
                (6557.39175+61.27786, 0.21467+8.58491),   # Dataset 2
                (382.01572+10.56089, 0.15849+4.31518)    # Dataset 3
            ],
            'Reconcile': [
                (156.28378, 16.03081),  # Dataset 1 (Core algo speed similar per tree)
                (16735.90709, 429.51364),  # Dataset 2
                (1041.99585, 77.38802) # Dataset 3
            ],
        }
        '''# Group 1: Fast / Low Magnitude
        steps_fast = {
            'Build MTs': [
                (0.4926, 0.16781),   # Dataset 1
                (10.36343, 5.13517),   # Dataset 2
                (1.58598, 0.41652)   # Dataset 3
            ],
            'Load GTs': [
                (1.34652, 0.22395),    # Dataset 1 (Usually similar I/O)
                (0.47741, 0.76329),    # Dataset 2
                (0.71326, 0.69493)   # Dataset 3
            ]
        }
        
        # Group 2: Slow / High Magnitude
        steps_slow = {
            'Collapse & Filter GTs': [
                (59.32991+4.95699, 0.02194+2.52801),    # Dataset 1
                (6557.39175+61.27786, 0.21467+8.58491),   # Dataset 2
                (382.01572+10.56089, 0.15849+4.31518)    # Dataset 3
            ],
            'Reconcile': [
                (156.28378, 16.03081),  # Dataset 1 (Core algo speed similar per tree)
                (16735.90709, 429.51364),  # Dataset 2
                (1041.99585, 77.38802) # Dataset 3
            ]
        }'''

        # --- 2. Plotting Setup ---
        step_names = list(time_steps.keys())
        n_steps = len(step_names)
        n_datasets = len(datasets_for_legend)
        n_tools = len(tools_for_legend)

        # Layout calculations
        x = np.arange(n_steps)  # Base positions for steps
        total_width = 0.8       # Width of the entire group for one step
        group_width = total_width / n_datasets  # Width for one dataset block
        bar_width = group_width / n_tools       # Width for a single tool bar
        
        fig, ax = plt.subplots(figsize=(12, 7))

        # Colors for Datasets
        colors = ['#66c2a5', '#fc8d62', '#8da0cb'] # Qualitative Set2 style
        # Hatches for Tools: Empty for Grampa, Hatched for Grandma
        hatches = ['', '////'] 

        # --- 3. The Plotting Loop ---
        for i, step in enumerate(step_names):
            data_for_step = time_steps[step] # List of tuples
            
            # Center the entire group on the x tick
            group_start_x = x[i] - (total_width / 2)
            
            for j in range(n_datasets):
                # Calculate start x for this dataset block
                dataset_x_start = group_start_x + (j * group_width)
                
                tool_times = data_for_step[j] # (Time_Tool1, Time_Tool2)
                
                for k in range(n_tools):
                    # Calculate exact x for this bar
                    bar_x = dataset_x_start + (k * bar_width)
                    time_val = tool_times[k]
                    
                    ax.bar(bar_x, time_val, width=bar_width, 
                           color=colors[j], edgecolor='black', linewidth=0.7,
                           hatch=hatches[k], align='edge',
                           label=f"{datasets_for_legend[j]}" if i==0 and k==0 else "")

        # --- 4. Formatting ---
        ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
        ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
        ax.set_title("Single Event Performance: GRAMPA vs. GRANDMA", fontsize=14, pad=20)
        
        ax.set_xticks(x)
        ax.set_xticklabels(step_names, fontsize=11)
        
        # 1. LOG SCALE
        ax.set_yscale('log')
        
        # Add grid for readability
        ax.yaxis.grid(True, linestyle='--', alpha=0.7, which='major')
        ax.yaxis.grid(True, linestyle=':', alpha=0.4, which='minor')
        ax.set_axisbelow(True)

        # 2. SECONDARY Y-AXIS (Time Units)
        ax2 = ax.twinx()
        ax2.set_yscale('log')
        ax2.set_ylim(ax.get_ylim()) # Sync limits
        
        # Define ticks: 1s, 1m, 1h, 6h
        human_ticks = [1, 10, 60, 600, 3600, 21600]
        human_labels = ['1s', '10s', '1m', '10m', '1h', '6h']
        
        ax2.yaxis.set_major_locator(FixedLocator(human_ticks))
        ax2.set_yticklabels(human_labels)
        #ax2.set_ylabel("Human Readable Time", fontsize=12, rotation=270, labelpad=15)
        
        # --- 5. Custom Legend Construction ---
        # Legend 1: Colors (Datasets)
        color_handles = [mpatches.Patch(facecolor=colors[i], edgecolor='black', label=datasets_for_legend[i]) 
                         for i in range(n_datasets)]
        
        # Legend 2: Patterns (Tools)
        # Create neutral grey patches to show the pattern
        pattern_handles = [mpatches.Patch(facecolor='lightgrey', edgecolor='black', hatch=hatches[i], label=tools_for_legend[i]) 
                           for i in range(n_tools)]

        # Add Legends
        first_legend = ax.legend(handles=color_handles, title="Datasets", loc='upper left', frameon=True)
        ax.add_artist(first_legend) # Add back mainly because the next call overwrites it
        
        ax.legend(handles=pattern_handles, title="Tools", loc='upper left', bbox_to_anchor=(0, 0.75), frameon=True)

        plt.tight_layout()
        #plt.show()
        out_dir = Path.cwd()
        print(f"Saving figure to: {out_dir}")
        plt.savefig(out_dir / "performance_comparison.png", dpi=600)
        sys.exit()

        '''# For the "All Operations" plot, we will use all steps
        steps_fast.update(steps_slow)
        # In slow, convert to hours for the second plot
        for step in steps_slow:
            steps_slow[step] = [(t[0]/3600, t[1]/3600) for t in steps_slow[step]]

        step_groups = [steps_fast, steps_slow]
        titles = ["All Operations", "Intensive Operations"]

        # --- 2. Plotting Setup ---
        # 2 Rows, 1 Column
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Colors for Datasets
        colors = ['#66c2a5', '#fc8d62', '#8da0cb'] # Qualitative Set2 style
        # Hatches for Tools: Empty for Grampa, Hatched for Grandma
        hatches = ['', '////'] 

        n_datasets = len(datasets_for_legend)
        n_tools = len(tools_for_legend)
        
        # --- 3. The Plotting Loop ---
        for ax_idx, ax in enumerate(axes):
            current_steps = step_groups[ax_idx]
            step_names = list(current_steps.keys())
            n_steps = len(step_names)
            
            x = np.arange(n_steps)
            total_width = 0.8
            group_width = total_width / n_datasets
            bar_width = group_width / n_tools

            for i, step in enumerate(step_names):
                data_for_step = current_steps[step] # List of tuples
                group_start_x = x[i] - (total_width / 2)
                
                for j in range(n_datasets):
                    dataset_x_start = group_start_x + (j * group_width)
                    tool_times = data_for_step[j]
                    
                    for k in range(n_tools):
                        bar_x = dataset_x_start + (k * bar_width)
                        time_val = tool_times[k]
                        
                        rects = ax.bar(bar_x, time_val, width=bar_width, 
                               color=colors[j], edgecolor='black', linewidth=0.7,
                               hatch=hatches[k], align='edge')

                        # --- ADD VALUE LABELS ---
                        # Loop over the bars just created (usually just 1 per iteration here)
                        for rect in rects:
                            height = rect.get_height()
                            if height < 11 and ax_idx == 0:
                                continue
                            if height < 0.3 and ax_idx == 1:
                                continue
                            
                            # Formatting: If small (<100), show decimal. If large, show integer.
                            # Also handle the unit conversion (h vs s) for display if needed.
                            # Since we already divided by 3600 for the slow plot, the 'height' is in hours.
                            
                            if height < 10:
                                label_text = f'{height:.1f}'
                            else:
                                label_text = f'{int(height)}'
                            
                            text_h = 11.05 if ax_idx == 0 else 0.306

                            # Place text slightly above the bar
                            ax.text(rect.get_x() + rect.get_width() / 2, text_h, label_text,
                                    ha='center', va='bottom', fontsize=9, fontweight='bold',
                                    rotation=45, clip_on=False)

            # --- 4. Formatting ---
            ax.set_ylabel(y_labels[ax_idx], fontsize=12, fontweight='bold')
            ax.set_title(titles[ax_idx], fontsize=12, loc='left', pad=10)

            if step_groups[ax_idx] == steps_fast:
                ax.set_ylim(0, 11)
            else:
                ax.set_ylim(0, 0.3)
            
            ax.set_xticks(x)
            ax.set_xticklabels(step_names, fontsize=11, fontweight='bold')
            
            # Independent scales for each plot (Linear is fine since we split by magnitude)
            # ax.set_yscale('log') # Uncomment if log scale is still desired
            
            ax.yaxis.grid(True, linestyle='--', alpha=0.7)
            ax.set_axisbelow(True)

        # --- 5. Legends (Add to Top Plot) ---
        color_handles = [mpatches.Patch(facecolor=colors[i], edgecolor='black', label=datasets_for_legend[i]) 
                         for i in range(n_datasets)]
        pattern_handles = [mpatches.Patch(facecolor='lightgrey', edgecolor='black', hatch=hatches[i], label=tools_for_legend[i]) 
                           for i in range(n_tools)]

        ax_top = axes[0]
        first_legend = ax_top.legend(handles=color_handles, title="Datasets", loc='upper left', frameon=True)
        ax_top.add_artist(first_legend)
        ax_top.legend(handles=pattern_handles, title="Tools", loc='upper left', bbox_to_anchor=(0, 0.65), frameon=True)

        axes[1].set_xlabel(x_label, fontsize=12, fontweight='bold')
        fig.suptitle("Performance Comparison: GRAMPA vs. GRANDMA", fontsize=16, y=0.95)

        plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # Adjust for suptitle
        # save the figure if needed
        out_dir = Path.cwd()
        print(f"Saving figure to: {out_dir}")
        plt.savefig(out_dir / "performance_comparison.png", dpi=600)
        sys.exit()'''
    
    def parse(self, args_=None) -> Tuple[GlobalContext, TaskConfig]:
        """
        Parses arguments and returns strictly typed configuration objects.
        """
        args = self.parser.parse_args(args_)
        
        #self.plot_and_exit()
        
        # --- Setup Environment and Banner ---
        out_dir = args.outdir if args.outdir else get_default_outdir()
        out_dir = Path(out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = out_dir / f"{args.prefix}.log"
        self.logger = GranLogger(log_file=log_file, verbosity=args.verbosity, no_log=args.nolog, debug=args.debug)

        # --- GENERATION MODE INTERCEPT ---
        if args.generate:
            from .generator import DatasetGenerator
            
            if not args.spec_input:
                # We reuse the -s flag for the base species tree
                self.logger.log("Generation mode requires a base species tree template via -s.", 'e')
                
            if not os.path.exists(args.generate):
                self.logger.log(f"Generator requires a valid JSON configuration file. Not found: {args.generate}", 'e')
            
            generator = DatasetGenerator(args, out_dir, self.logger)
            
            try:
                generator.run()
                print("Generation complete.")
            except Exception as e:
                print(f"Generation failed: {e}")
                import traceback
                traceback.print_exc()
            
            import sys  
            sys.exit(0)

        self.logger.log_software_banner(GranMetadata())
        self.logger.log("=" * 73, 'i')

        ####
        # Logging strategy:
        # log the original paths in the banner only!
        # split start run:
        # before parsing args: log original args
        # after parsing args: log sanitized args in tcf - when running the Task
        ###

        # --- Resolve Argument Logistics ---
        mode, mixed_switch = self.resolve_mode_logic(
            args.mode,
            args.labeltree,
            args.numtrees,
            args.buildmultrees,
            args.checknums,
            args.st_only,
            args.no_st
        )
        """legacy_flags = {
            'label_sp': args.labeltree, 'count_mts': args.numtrees,
            'build_mts': args.buildmultrees, 'check_nums': args.checknums,
            'st_only': args.st_only, 'no_st': args.no_st
        }
        mode, mixed_switch = self.resolve_mode_logic(args.mode, legacy_flags)"""

        nesting = self.resolve_nesting(args.nesting)

        # Handle folder deletion and history loading BEFORE the engine starts
        start = args.start if args.start == 'auto' else int(args.start)
        
        # Prepare history/folders
        i, history, history_file = start_up_prep(start, out_dir, self.logger, mode)

        if args.spec_input is None:
            # Try loading from folder of i
            prev_st = out_dir / str(i-1) / "spectree.tre"
            if args.plot:
                self.logger.log("Plotting mode detected. Program will not run any analysis.", 'i')
                pass
                # To be implemented: Load history and plot only
            elif prev_st.exists():
                args.spec_input = str(prev_st)
                self.logger.log(f"Resuming trees input from previous run at {prev_st.parent}", 'i')
                if args.genes_input is None:
                    try:
                        prev_gt = out_dir / str(i-1) / "genetrees.txt"
                        if prev_gt.exists():
                            args.genes_input = str(prev_gt)
                    except Exception:
                        pass
            else:
                self.logger.log("Species tree input (-s) is required for this mode.", 'e')

        # 1. Determine CPU Count
        if args.procs > 0:
            # User specified value takes precedence
            n_procs = args.procs
        else:
            # Check Cluster Environment Variables
            if 'SLURM_CPUS_PER_TASK' in os.environ:
                # Slurm
                n_procs = int(os.environ['SLURM_CPUS_PER_TASK'])
            elif 'PBS_NP' in os.environ:
                # PBS / Torque
                n_procs = int(os.environ['PBS_NP'])
            elif 'LSB_DJOB_NUMPROC' in os.environ:
                # LSF
                n_procs = int(os.environ['LSB_DJOB_NUMPROC'])
            elif 'NSLOTS' in os.environ:
                # GridEngine
                n_procs = int(os.environ['NSLOTS'])
            else:
                # Local Machine fallback
                try:
                    n_procs = multiprocessing.cpu_count()
                except NotImplementedError:
                    n_procs = 1

        # --- Build Global Context ---
        ctx = GlobalContext(
            num_processes = n_procs,
            verbosity     = args.verbosity,
            seed          = args.seed,
            plot          = args.plot,
            norun         = args.norun,
            nolog         = args.nolog,
            debug         = args.debug,
            optim         = args.optim,
            orth_opt      = args.orthologies,
            max_iter      = check_loop_length(args.iter, i, None, history, self.logger),
            mixed_switch  = mixed_switch,
            cutoff        = self.parse_cutoff(args.cutoff),
            nesting       = nesting,
            min_gt_lvs    = args.min_gt_lvs,
            min_st_lvs    = args.min_st_lvs,
            root_dir      = out_dir,
            log_file      = log_file,
            history       = history,
            start_pt      = i
        )

        # --- Prepare Step Config ---
        tcf = TaskConfig(
            output_dir    = ctx.root_dir,
            st            = args.spec_input,
            gts           = args.genes_input,
            run_prefix    = args.prefix,
            mode          = mode,
            overwrite     = args.overwrite,
            repair        = args.repair,
            h1_nodes      = args.h1,
            h2_nodes      = args.h2,
            ploidies      = args.ploidy,
            group_cap     = args.cap,
            weights       = tuple(args.weights),
            to_map        = args.maps,
            max_select    = max(args.max_select, 1),
            is_mul_input  = args.is_mul_input,
        )

        return ctx, tcf
    