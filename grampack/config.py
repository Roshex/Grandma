'''
Replaces params.py and global_vars.py. Holds constants and configuration dataclasses.
Handles the input parsing logic from opt_parse.py and spec_tree.py
'''

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
from .models import SmrtTree, HistoryType, TreeCache, ProtectedDict

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
    version: str = "2.5.1"

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
    seed: int = 42
    
    # Output & Logging Controls
    verbosity: int = 3
    pickles: str = "archive" # 'keep', 'clean', 'store', 'archive'
    maps: bool = False
    plot: bool = False
    norun: bool = False
    nolog: bool = False
    bench: bool = False
    debug: bool = False
    sample: int = 2

    # Global Flow/Algorithm Options
    orth_opt: bool = False
    lookahead: int = 0
    breadth_max: int = 0
    max_iter: Union[int, float] = 0
    start_pt: int = 0
    mixed_switch: int = 0
    root_spec: Optional[str] = None
    # For full mode
    # replaces ignore_nesting: bool = False -> ignore (False), rectify (True), strict_rectify (New), model (New)
    nesting: str = "model"
    # For split mode
    min_st_lvs: int = 1 
    min_gt_lvs: int = 2
    # For mt selection
    strict_max: bool = False
    allow_redun: bool = False
      
    # Paths that define the "Session"
    root_dir: Path = field(default_factory=lambda: Path(get_default_outdir()))
    log_file: Optional[Path] = None
    
    # History Tracking (Global State)
    history: HistoryType = field(default_factory=lambda: HistoryType(Path(get_default_outdir()) / "history.json"))

    @property
    def history_file(self) -> Path:
        """Dynamic property derived from the root output dir."""
        return self.root_dir / 'history.json'
    
    @property
    def beam_file(self) -> Path:
        """Dynamic property for the beam search file path."""
        return self.root_dir / 'beam_tracker.json'
    
    def update(self, **changes) -> 'GlobalContext':
        """
        Returns a new GlobalContext with specific fields updated.
        Validates that keys exist to prevent silent errors.
        """
        # Certain fields should not be allowed to be changed
        forbidden_keys = {"seed", "pickles", "maps", "plot", "norun", "nolog", "bench", "sample",
        "orth_opt", "lookahead", "breadth_max", "max_iter", "start_pt", "mixed_switch", "nesting",
        "min_gt_lvs", "min_st_lvs", "root_spec", "strict_max", "allow_redun", "root_dir", "log_file", "history"}
        # Safety check to ensure we aren't inventing new fields
        valid_fields = {f.name for f in fields(self)}
        for key in changes:
            if key not in valid_fields:
                raise TypeError(f"GlobalContext got an unexpected keyword argument '{key}'")
            if key in forbidden_keys:
                raise ValueError(f"GlobalContext field '{key}' is not allowed to be changed")

        return replace(self, **changes)
    
    def get_task_dir(self, task_id: Optional[Tuple[int, int]] = None, state_id: str = "S0", is_beam: bool = False) -> Path:
        """Centralized path management for consistent task output directory resolution."""
        if task_id is None:
            return self.root_dir / "output"
        
        depth, idx = task_id
        task_str = f"{depth}.{idx}" if idx is not None else f"{depth}"
        
        # Branch into isolated state folders only if beam search is actively splitting paths
        if is_beam and state_id != "S0":
            return self.root_dir / task_str / state_id / "output"
        
        return self.root_dir / task_str / "output"


# --- Unit of Reconciliation (Dynamic) ---
@dataclass(frozen=True, slots=True)
class TaskConfig:
    """
    Configuration for a SINGLE execution Step.
    Contains sanitized arguments specific to one reconciliation task (~Grampa).
    """
    # I/O for this specific step
    st: Union[str, Path, SmrtTree]
    gts: Optional[Union[str, Path, Dict[int, SmrtTree]]] = None
    output_dir: Path = field(default_factory=lambda: Path(get_default_outdir()))
    run_prefix: str = "grandma"
    repair: str = 'none'
    overwrite: bool = False

    # Mode Target
    mode: str = "single"

    # Algorithm Tunables
    h1_nodes: Optional[Union[str, List[str]]] = None
    h2_nodes: Optional[Union[str, List[str]]] = None
    ploidies: Optional[Union[Path, str, Dict[str, int]]] = None
    optim: int = 0
    unit_rule: int = 1 # 0=Strict, 1=Engine, 2=Maximal
    group_cap: int = 15
    cap_by_work: bool = False
    quota_gts: str = "equal"
    weights: Tuple[int, int] = (1, 1) # (w_dup, w_loss)
    n_best: int = 1 # max mts to select for processing per Run (non-overlapping H clades & scoring above ST)
    cutoff: Tuple[str, str, Union[float, int]] = ('input', 'abs', 0) # (Reference, DiffFunc, Offset)

    disable_dedup_below: float = 0.05      # duplicate fraction under which dedup is skipped
    dedup_latch: bool = False              # set on child tasks once the serial phase has
                                           # decided against it (see plan_dedup)

    # Legacy Flags
    is_mul_input: bool = False

    # --- Global State Tracking ---
    # For Split/Mixed mode ploidy constraints
    global_tree_cache: Optional[TreeCache] = None
    # For MulTree inputs with pre-defined reconciliations
    predefined_rets: Dict[int, List[Tuple[str, str]]] = field(default_factory=dict)
    # For tracking multiple MT selection searches (i.e., not a simple greedy selection)
    search_state_id: str = "S0"
    prev_score: Optional[float] = None

    def __post_init__(self):
        """Validate the optim bitmask right after dataclass initialization."""
        if not isinstance(self.optim, int) or not (0 <= self.optim <= 15):
            raise ValueError(f"Invalid optimization level: {self.optim}. Must be 0-15.")

    # --- Optimization Switches (Decoded dynamically on access) ---

    @property
    def dedup_gts(self) -> bool:
        """Bit 0 (+1): One representative per class of isomorphic gene trees."""
        return bool(self.optim & 1)

    @property
    def use_gray(self) -> bool:
        """Bit 1 (+2): Gray-code enumeration of ambiguous groups / units."""
        return bool(self.optim & 2)

    @property
    def use_sweep(self) -> bool:
        """Bit 2 (+4): Target sweep for single-target candidates."""
        return bool(self.optim & 4)

    @property
    def use_exact(self) -> bool:
        """Bit 3 (+8): Exact unit states. Without it, reproduces GRAMPA's collapsing exactly."""
        return bool(self.optim & 8)

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
        forbidden_keys = {'optim', 'unit_rule', 'group_cap', 'cap_by_work', "quota_gts", 'weights',
                          'max_select', 'run_prefix', 'search_state_id'}
        # Safety check to ensure we aren't inventing new fields
        valid_fields = {f.name for f in fields(self)}
        for key in changes:
            if key not in valid_fields:
                raise TypeError(f"TaskConfig got an unexpected keyword argument '{key}'")
            if key in forbidden_keys:
                raise ValueError(f"TaskConfig field '{key}' is not allowed to be changed")

        return replace(self, **changes)

# --- Utility Functions ---

def decode_optim(optim: int) -> Tuple[bool, bool, bool, bool]:
    """
      bit 0 (+1)  dedup_gts    one representative per class of isomorphic gene trees
      bit 1 (+2)  use_gray     Gray-code enumeration of ambiguous groups / units
      bit 2 (+4)  use_sweep    target sweep for single-target candidates
      bit 3 (+8)  use_exact    exact unit states (see core.unit_states); without it
                               both engines reproduce GRAMPA's collapsing exactly.
    """
    if not isinstance(optim, int) or not (0 <= optim <= 15):
        raise ValueError(f"Invalid optimization level: {optim}. Must be 0-15.")
    return bool(optim & 1), bool(optim & 2), bool(optim & 4), bool(optim & 8)

def load_history_ng(history_file: Path) -> Dict[Tuple[Any, int], Any]:
    """Loads and deserializes the iteration history."""
    with open(history_file, 'r') as f:
        # Convert string keys back to tuples. 
        # Note: Split mode keys might be (depth, idx) or strings in future, 
        # but ast.literal_eval handles "(0, 0)" safely.
        return {ast.literal_eval(k): v for k, v in json.load(f).items()}

def load_history(history_file: Path) -> ProtectedDict:
    """Loads and deserializes the iteration history into a ProtectedDict."""
    with open(history_file, 'r') as f:
        raw_history = json.load(f)
        
    def _recursive_protect(d):
        if isinstance(d, dict):
            pt = ProtectedDict()
            for k, v in d.items():
                pt[k] = _recursive_protect(v)
            return pt
        return d
        
    protected_hist = ProtectedDict()
    for k, v in raw_history.items():
        # ast.literal_eval handles the tuple keys like "(0, 0)"
        protected_hist[ast.literal_eval(k)] = _recursive_protect(v)
        
    return protected_hist

def start_up_prep_2(start_point: Union[str, int], base_output_dir: Path, logger: GranLogger, mode: str = "full") -> Tuple[int, Dict, Path]:
    """
    Handles folder cleanup and history loading.
    Supports both integer-based iteration folders (Full) and Depth.Index folders (Split).
    """
    history = ProtectedDict()
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
             history = ProtectedDict()
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
    history = ProtectedDict()
    history_file = base_output_dir / 'history.json'

    # 1. Load History if exists
    if base_output_dir.exists():
        if history_file.exists():
            try:
                history = load_history(history_file)
                logger.log(f"Loaded existing history ({len(history)} entries).", 'i')
            except Exception as e:
                logger.log(f"Failed to load history file ({e}). Starting fresh.", 'w')
                history = ProtectedDict()
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
             history = ProtectedDict()
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

def check_loop_length_greedy(n: int, i: int, st_file: Path, history: Dict, logger: GranLogger) -> Union[int, float]:
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

def check_loop_length(n: int, i: int, st_file: Path, history: Dict, logger: GranLogger) -> Union[int, float]:
    """Determines the true loop length based on input and history."""
    if n <= 0:
        n = float('inf')
    if i > 0:
        last_task = (i-1, 0)
        if last_task in history:
            passed_any = False
            # Iterate through the new nested state structure
            for state_id, events in history[last_task].items():
                for child_id, event_data in events.items():
                    if child_id != "In" and event_data.get('passed', False):
                        passed_any = True
                        break
                if passed_any: break
            
            if not passed_any and st_file.exists():
                logger.log(f'Previous run was completed. No further iterations needed.', 'i')
                n = -1
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
        g_general.add_argument('-r', '--repair', nargs='?', const='best', default='none', choices=['fast', 'f', 'best', 'b', 'none', 'n'],
            help="If set, attempt to repair input files by forcing bifurcating trees, rooting, valid tip names, and more. "
                 "If used without arguments, defaults to 'best' (Notung-like optimal rooting and polytomy resolution).")
        g_general.add_argument("-p", "--procs", type=int, default=1,
            help="Number of processes to use for parallelizable tasks. Default = 1. Non-positive to autodetect and " 
                 "use all available cores.")
        g_general.add_argument("-v", "--verbosity", type=int, default=2, choices=range(5),
            help="Level of verbosity printed to the screen. 0 = none; 1 = run info; 2 = standard; "
                 "3 = debug; 4 = verbose debugging. Default = 2.")
        g_general.add_argument("--overwrite", action="store_true",
            help="If set, overwrite existing files in the output directory. Default: exit if files exist.")
        g_general.add_argument("--generate", type=str, metavar="JSON_CONFIG",
            help="Path to JSON configuration file. Enters Generation Mode (ignores other flags).")

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
        g_algo.add_argument("-c", "--cap", type=int, default=15,
            help="The maximum number of groups with polyploid species a gene tree is allowed to have given a MUL-tree. "
                 "A GT with more than --cap groups for at least one MT, will be skipped entirely. Default = 15.")
        g_algo.add_argument("-q", "--quota-gts", dest="quota_gts", type=str, choices=['equal', 'e', 'harmonic', 'h', 'per-clade', 'p'], default='equal',
            help="Method for balancing the influence of gene trees during reconciliation. "
                 "(e)qual: No balancing; all GTs contribute equally to the parsimony score [default]. "
                 "(h)armonic: Balance by the harmonic mean Q of topology and support. GTs with Q will have less influence. "
                 "(p)er-clade: Balance directly during reconciliation. Clades with low support will have less influence [Not implemented].")
        g_algo.add_argument("-w", "--weights", type=int, nargs=2, default=[1, 1],
            help="Space-separated integer weights for the parsimony score calculation: 'w_dup w_loss'. Default = '1 1'.")
        g_algo.add_argument("-n", "--n-best", dest="n_best", type=int, default=1,
            help="Maximum top-ranked MUL-trees to select per run for postprocessing and detailed outputs. Default: 1, i.e., only the best "
                 "scoring MT is considered per iteration (Default: 1, All_until_input_st_inclusive: 0, ALL: any negative int).")
        g_algo.add_argument("--min-st-lvs", dest="min_st_lvs", type=int, default=1,
            help="Minimum species tree leaves for a species tree to be considered valid. Specifically relevant for the "
                 "split mode. Default: 1.")
        g_algo.add_argument("--min-gt-lvs", dest="min_gt_lvs", type=int, default=2,
            help="Minimum gene trees leaves per tree for a gene tree to be considered valid. Specifically relevant for "
                 "the split mode. Default: 2.")
        g_algo.add_argument('--nesting', type=str, choices=['ignore', 'i', 'rectify', 'r', 'model', 'm', 'strict_rectify', 's'], default='model',
            help="Behavior for nested hybridization events treatment during the full and mixed modes. "
                 "(i)gnore: Do nothing, ignore nesting scenarios. "
                 "(r)ectify: Autocorrect nested events between iterations, including via sister relationships [default]. "
                 "(s)trict_rectify: Rectify, but only consider internally nested events when tracing missing subgenomes. "
                 "(m)odel: Model nested copies during MT creation. Computationally heaviest, but most exact.")
        g_algo.add_argument('--strict-constraint', dest='strict_constraint', action='store_true',
            help="If set, ploidy constraints will apply strictly to number of copies of a species, rather than to "
                 "number of their monophyletic clades.")
        g_algo.add_argument('--allow-redundant-mts', dest='allow_redundant_mts', action='store_true',
            help="If set, the program will not filter out redundant MUL-trees where taxa are grafted below themselves (i.e., having "
                 "identical groupings and scores with those grafted above themselves). This may be useful for debugging and for exact "
                 "reproduction of legacy-GRAMPA output, but is not recommended for general use due to increased runtime and algorithmic " 
                 "incompatibility with full/mixed modes.")
        
        g_algo.add_argument('--optim', dest='optim', type=int, default=0,
            help="Developer option for optimization level. 0 = default, 1 = deduplication only, 2 = backbone only, 3 = combined optimizations.")
        g_algo.add_argument('--disable-dedup-below', dest='disable_dedup_below', type=float, default=0.05, metavar='FRAC',
            help="Duplicate fraction below which gene-tree de-duplication is disabled for "
                "SUBSEQUENT iterations of a serial chain. The map is always used in the task "
                "that computed it (the signature pass is already paid). 1.0 disables it "
                "entirely, including the signature pass; to switch it off completely, clear "
                "bit 0 of --optim.")
        g_algo.add_argument('-u', '--unit-rule', dest='unit_rule', default='auto', choices=['auto', 0, 'strict', 1, 'engine', 2, 'maximal'],
        help="How gene-copy clades are collapsed into mapping units. 'auto' (default) uses 'maximal' when exact grouping "
              "is enabled (--optim bit 3) and 'engine' otherwise, which is what preserves GRAMPA score parity. "
              "'maximal' without exact grouping is NOT sound. 'strict' (0) is duplicate-freeness (transitive), "
              "'engine' (2) is sibling-disjointedness, and 'maximal' (3) is maximally-movable.")
        g_algo.add_argument('--cap-by-work', dest='cap_by_work', action='store_true',
        help="Apply --cap to log2 of the number of allele assignments under the grouping actually in use, instead "
              "of to GRAMPA's unit count ('engine' rule). With coarse units (--unit-rule maximal) this retains "
              "more gene trees, but scores are then NOT comparable with runs that use the default metric.")
        
        # --- Flow Control Options ---
        g_flow = self.parser.add_argument_group("Flow Control Options")
        g_flow.add_argument("-m", "--mode", type=str, default="single",
            help="Execution mode. Supported options: "
                 "single: Simple run with a single iteration, equivalent to running Grampa [default]. "
                 "full: Fully sequential search allowing for nested hybridization event inference [computationally expensive]. "
                 "split: Binary split search reducing each depth into outer/inner subproblems [cheap; supports subproblem parallelism]. "
                 "mixed-<int>: Start with full mode, then switch to split mode after <int> iterations [default w/o int: mixed-3]. "
                 "label-sp: Only label input species tree internal nodes. "
                 "repair: Load and repair input trees (defaults to '-r best'), export them, and exit. "
                 "count-mts: Count possible MUL-trees only. "
                 "build-mts: Build MUL-trees only. "
                 "check-nums: Count groups only. "
                 "no-st: Skip reconciliation to input. "
                 "st-only: Reconciliation to input only.")
        g_flow.add_argument("-l", "--lookahead", type=int, default=0,
            help="Controls the 'delay' in path selection: a lookahead of `l` at depth `d` evaluates paths which originated at depth `d-l`. "
                 "Keeps only the paths which originated from the origin that produced the best path (at depth `d`), capping the maximum "
                 "active paths back to n^l. Only relevant for iterative modes with n>1. Default: 0 (disabled), Unbounded (=disabled): any "
                 "non-positive int.")
        g_flow.add_argument("-b", "--breadth-max", dest="breadth_max", type=int, default=0,
            help="Controls the maximum number of active paths allowed to proceed at any given depth, keeping only the top `b` paths. "
                 "Only relevant for iterative modes with n>1. Default: 0 (disabled), Unbounded (=disabled): any non-positive int.")
        g_flow.add_argument('-i', '--iter', type=int, default=0,
            help="Maximun number of iterations or (~depth) event num for iterative modes; <int>, non-positive to be unlimited. Default = 0.")
        g_flow.add_argument('--start', type=str, default='auto',
            help="Start point when resuming a previous execution; positive <int>, or 'auto' [default] for auto-detection.")
        
        # Cutoff Mode (-c, --cutoff-mode)
        """
        Controls how the top non-input Multi-Trees are filtered during reconciliation.
        Options:
        'input' (0)     : (Default) Filters based on the absolute/relative difference from the input score (using the existing threshold logic).
        'fvall' (-1)    : Creates a score distribution histogram of the current reconciliation. Finds the first local minimum (valley) in frequency and uses its score as the cutoff threshold.
        'lvall' (-2)     : Creates a score distribution histogram of the current reconciliation. Finds the first local minimum (valley) in frequency and uses its score as the cutoff threshold.
        'rvall' (-3)    : Creates a score distribution histogram of the current reconciliation. Finds the first local minimum (valley) in frequency and uses its score as the cutoff threshold.
        'none' (<=-4)     : No cutoff. All top `n_best` MTs will be accepted.
        """

        g_flow.add_argument('--cutoff', type=str, nargs='+', default=['auto'],
            help="Strategy to pass or filter the top non-input MT(s) during reconciliation. Recieves up to 3 space-separated arguments: "
                 "- Reference: 'input' [default] for input tree's score, 'fvall' for first valley in score distribution, 'lvall' for valley "
                 "             left of the input score, 'rvall' for valley right of the input score, or 'none' for no cutoff. "
                 "- Difference function (if Ref is not 'none'): 'abs' for absolute difference [default], or 'rel' for relative difference. "
                 "- Offset (if Ref is not 'none'): <int> for absolute, or <float> for relative. Default = 0 (i.e., no difference from Ref). "
                 "These are optional and may be given in any order. 'auto' [default] is a shorthand for 'input abs 0'. ")
        
        g_flow.add_argument("--root", type=str, default=None,
            help="Root the species tree on the specified node/leaf string or comma-separated clade.")
        g_flow.add_argument("--orthologies", action="store_true",
            help="If set, will output an additional file containing the pairwise orthology "
                 "relationships for each gene tree to the lowest scoring MUL-tree.")

        # --- Output Options ---
        g_output = self.parser.add_argument_group("Output Options")
        g_output.add_argument("--pickles", type=str, choices=['keep', 'k', 'clean', 'c', 'archive', 'a', 'store', 's'], default='archive',
            help="Action to take on the pickle directory after an inference step: "
                 "(k)eep: leave files untouched for instant resuming. "
                 "(c)lean: delete the directory (Warning: prevents resuming). "
                 "(s)tore: store the entire directly as a .tar uncompressed file. "
                 "(a)rchive: compresses the entire directory into a single .tar.gz file (Default; auto-resumed).")
        g_output.add_argument("--maps", action='store_true',
            help="If set, the detailed output file will contain node mappings for each gene tree to each of the lowest "
                 "scoring MUL-tree in the file.")
        g_output.add_argument('--plot', action='store_true',
            help="Plot taxon count, MP score, and normalized score over iterations. Relevant only for iterative modes.")
        g_output.add_argument("--norun", action="store_true",
            help="If set, only print the run info and exit.")
        g_output.add_argument("--nolog", action="store_true",
            help="If set, do not write a log file.")
        g_output.add_argument("--bench", action="store_true",
            help="If set, write a benchmark tsv file with runtime for each step. Only applicable if --norun is not set.")
        g_output.add_argument('--debug', action='store_true',
            help="Enable debug mode for additional outputs to the log file (whereas --v 3 only prints debug messages to screen).")
        g_output.add_argument('--seed', type=int, default=42,
            help="Random seed for sampling and reproducibility. Default = 42.")
        g_output.add_argument('--sample', type=int, default=2,
            help="Number of samples to sample from large iterables. Default = 2.")

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

    @staticmethod
    def resolve_unit_rule(unit_rule: str, use_exact: bool, use_sweep: bool) -> str:
        """
        'maximal' is the coarsest valid decomposition and is strictly cheaper under exact
        grouping - merging k units trades a factor >= 2^k in the outer product for O(|U|) in
        the local DP. It is NOT sound without exact states: collapsing a clade that is not
        duplicate-free is exactly GRAMPA's second grouping defect.
        """
        unit_rule_2_num = {'maximal': 2, 'engine': 1, 'strict': 0}
        if unit_rule == 'auto':
            return 2 if use_exact else 1
        if unit_rule in (2, 'maximal') and not use_exact:
            raise ValueError("--unit-rule maximal requires exact grouping (--optim bit 3): "
                            "without the mixed unit states it under-counts the score.")
        if unit_rule in (0, 'strict') and not (not use_exact and use_sweep):
            raise ValueError("--unit-rule strict is incompatible with pairwise recon, and with "
                            "exact grouping (--optim bit 3) + sweep (--optim bit 2): the strict "
                            "rule produces many, fragmented unit states for sweep to not lock up.")
        if unit_rule in (0, 1, 2, '0', '1', '2'):
            return int(unit_rule)
        try:
            return unit_rule_2_num[unit_rule]
        except KeyError:
            raise ValueError(f"Invalid unit rule: {unit_rule}")

    def parse_cutoff(self, vals: List[str]) -> Tuple[str, str, Union[float, int]]:
        """Parses stopping condition list into a typed tuple (Reference, DiffFunc, Offset)."""
        # Failsafe if argparse hasn't converted it to a list yet
        if isinstance(vals, str):
            vals = [vals]
        if len(vals) > 3:
            self.logger.log(f'Cutoff argument received too many values: {vals}', 'e')
        vals = [v.strip().lower() for v in vals]

        # Shorthand overrides
        if 'auto' in vals:
            if len(vals) > 1:
                self.logger.log(f'"auto" cutoff overrides other cutoff arguments. Ignoring: {[v for v in vals if v != "auto"]}', 'w')
            return ('input', 'abs', 0)
        if 'none' in vals:
            if len(vals) > 1:
                self.logger.log(f'"none" cutoff overrides other cutoff arguments. Ignoring: {[v for v in vals if v != "none"]}', 'w')
            return ('none', 'abs', 0)

        # Defaults
        ref, func, offset = 'input', 'abs', 0

        # Order-independent parsing
        for val in vals:
            if val in ('input', 'fvall', 'lvall', 'rvall'):
                ref = val
            elif val in ('abs', 'rel'):
                func = val
            else:
                try:
                    if '.' in val:
                        offset = float(val)
                    else:
                        offset = int(val)
                except ValueError:
                    self.logger.log(f'Ignored invalid cutoff argument: "{val}"', 'w')
                    
        return (ref, func, offset)

    def resolve_mode_logic(self, mode: str, lgflags: Dict[str, bool]) -> Tuple[str, int]:
        """
        Consolidates modern --mode and legacy flags into a single mode string.
        Returns resolved mode and mixed switch value.
        """
        # Sum boolean values safely
        num_set_legacy = sum(bool(v) for v in lgflags.values())
        
        if num_set_legacy > 1:
            self.logger.log("Multiple legacy flags set! One will be chosen according to precedence.", 'w')

        # Priority 1: Direct --mode selection (if not single)
        if mode != "single" and num_set_legacy:
            self.logger.log("Flag --mode overrides legacy flags!", 'w')

        # Parse Complex / Modern Modes
        if mode.startswith("mixed"):
            mixed_switch = 3 # default switch point in mixed mode
            if '-' in mode:
                try:
                    _, mixed_switch_str = mode.split('-')
                    mixed_switch = int(mixed_switch_str)
                    assert mixed_switch > 0
                except (ValueError, AssertionError):
                    self.logger.log(f"Invalid mixed mode format: {mode}. Expected 'mixed' or 'mixed-<int>', where <int> is strictly positive.", 'e')
            mode = "mixed"
        else:
            mixed_switch = 0 # for bin_id in normal split mode
            
            # Map valid legacy aliases from the --mode string into the dictionary
            if mode in {"label-sp", "label_sp", "labeltree"}:
                lgflags['label-sp'] = True
            elif mode in {"count-mts", "count_mts", "numtrees"}:
                lgflags['count-mts'] = True
            elif mode in {"build-mts", "build_mts", "buildmultrees"}:
                lgflags['build-mts'] = True
            elif mode in {"check-nums", "check_nums", "checknums"}:
                lgflags['check-nums'] = True
            elif mode in {"st-only", "st_only"}:
                lgflags['st-only'] = True
            elif mode in {"no-st", "no_st"}:
                lgflags['no-st'] = True
            elif mode not in {"single", "split", "full", "repair"}:
                self.logger.log(f"Unknown mode '{mode}': using fallback to 'single' or a set legacy flag.", 'w')
                mode = "single"

        # Standardize the final output string based on precedence
        if mode in {"mixed", "split", "full", "repair"}:
            # Major computational modes override all legacy booleans
            pass 
        else:
            # Resolve legacy modes in strict order from "earliest exit" to "longest execution"
            precedence_order = ['label-sp', 'count-mts', 'build-mts', 'check-nums', 'st-only', 'no-st']
            for flag in precedence_order:
                if lgflags.get(flag):
                    mode = flag
                    break
            # No-break clause of the for-else loop: if no legacy flags are set True, default to "single"
            else:
                mode = "single"
                
        return mode, mixed_switch

    @staticmethod
    def resolve_nesting(val: str) -> str:
        if val in ['ignore', 'i']: return 'ignore'
        # if val in ['rectify', 'r']: return 'rectify'
        if val in ['strict_rectify', 's']: return 'strict_rectify'
        if val in ['model', 'm']: return 'model'
        return 'rectify'

    @staticmethod
    def resolve_repair(val: str) -> str:
        if val in ['best', 'b']: return 'best'
        if val in ['fast', 'f']: return 'fast'
        # if val in ['none', 'n']: return 'none'
        return 'none'

    @staticmethod
    def resolve_quota(val: str) -> str:
        if val in ['harmonic', 'h']: return 'harmonic'
        if val in ['per-clade', 'p']: return 'per-clade'
        return 'equal'

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
        self.logger = GranLogger(log_file, args.verbosity, args.debug, no_log=args.nolog)

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

        ####
        # Logging strategy:
        # log the original paths in the banner only!
        # split start run:
        # before parsing args: log original args
        # after parsing args: log sanitized args in tcf - when running the Task
        ###

        quota_gts = self.resolve_quota(args.quota_gts) # <--- ADDED
        
        if quota_gts == 'per-clade': # <--- ADDED
            raise NotImplementedError("The 'per-clade' balancing option is not yet implemented.")

        # --- Resolve Argument Logistics ---
        legacy_mode_flags = {
            'label-sp': args.labeltree, 'count-mts': args.numtrees,
            'build-mts': args.buildmultrees, 'check-nums': args.checknums,
            'st-only': args.st_only, 'no-st': args.no_st
        }
        mode, mixed_switch = self.resolve_mode_logic(args.mode, legacy_mode_flags)

        repair = self.resolve_repair(args.repair)
        if mode == 'repair' and repair == 'none':
            self.logger.log("Mode 'repair' selected without choosing a level from --repair. Defaulting to 'best'.", 'i')
            repair = 'best'

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

        _, _, use_sweep, use_exact = decode_optim(args.optim)

        # --- Build Global Context ---
        ctx = GlobalContext(
            num_processes = n_procs,
            verbosity     = args.verbosity,
            pickles       = args.pickles,
            maps          = args.maps,
            plot          = args.plot,
            norun         = args.norun,
            nolog         = args.nolog,
            bench         = args.bench,
            debug         = args.debug,
            seed          = args.seed,
            sample        = args.sample,
            orth_opt      = args.orthologies,
            lookahead     = args.lookahead,
            breadth_max   = args.breadth_max,
            max_iter      = check_loop_length(args.iter, i, None, history, self.logger),
            start_pt      = i,
            mixed_switch  = mixed_switch,
            root_spec     = args.root,
            nesting       = nesting,
            min_gt_lvs    = args.min_gt_lvs,
            min_st_lvs    = args.min_st_lvs,
            strict_max    = args.strict_constraint,
            allow_redun   = args.allow_redundant_mts,
            root_dir      = out_dir,
            log_file      = log_file,
            history       = HistoryType(out_dir / "history.json", initial_data=history)
        )

        if not 0.0 <= args.disable_dedup_below <= 1.0:
            self.parser.error("--disable-dedup-below must be in [0, 1]")

        # --- Prepare Step Config ---
        tcf = TaskConfig(
            st            = args.spec_input,
            gts           = args.genes_input,
            output_dir    = ctx.root_dir,
            run_prefix    = args.prefix,
            repair        = repair,
            overwrite     = args.overwrite,
            mode          = mode,
            h1_nodes      = args.h1,
            h2_nodes      = args.h2,
            ploidies      = args.ploidy,
            optim         = args.optim,
            unit_rule     = self.resolve_unit_rule(args.unit_rule, use_exact=use_exact, use_sweep=use_sweep),
            group_cap     = args.cap,
            cap_by_work   = args.cap_by_work,
            quota_gts     = quota_gts,
            weights       = tuple(args.weights),
            n_best        = args.n_best,
            cutoff        = self.parse_cutoff(args.cutoff),
            is_mul_input  = args.is_mul_input,

            disable_dedup_below = args.disable_dedup_below,
        )

        return ctx, tcf
    