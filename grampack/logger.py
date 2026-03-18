'''
Replaces reconcore.py printing logic to ensure identical output format.
'''

import os
import sys
import time
import datetime
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Any, NamedTuple

# This block is ignored at runtime, and solves the circular dependency of typing
if TYPE_CHECKING:
    from .config import GranMetadata, GlobalContext, TaskConfig
    from .models import SmrtTree

# Try importing psutil for memory logging
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

class LogInheritance(NamedTuple):
    """
    NamedTuple to encapsulate the state of a logger for inheritance by following loggers.
    These fields shouldn't change when a logger passes its inheritance.
    Can't pass parent_logger here due to possible multiprocessing pickling issues, so that is handled separately in the GranLogger __init__.
    """
    # Init arg params
    log_file: Path
    no_log: bool
    label: str
    # Init default params
    bench: bool
    benchmarks: Optional[list]
    start_time: float
    warnings: int

class GranLogger:

    __slots__ = ['log_file', 'no_log', 'label', 'verbosity', 'debug', 'step_start_time', 'parent_logger',
                 'bench', 'benchmarks', 'start_time', 'warnings', 'step_active', 'step_buffer', 'step_interrupt', 'pids']
    
    def __init__(self, log_file: Path, verbosity: int = 4, debug: bool = False,
                 catch_exceptions: bool = True, parent_logger: Optional['GranLogger'] = None,
                 no_log: bool = False, clear_log: bool = True, label: str = "",
                 inheritance: Optional[LogInheritance] = None) -> None:
        self.log_file = log_file
        self.no_log = no_log          # Controls if ANY messages go to file
        self.label = label            # Optional task label for this logger if a task logger (not main logger)

        self.verbosity = verbosity    # Controls screen output (0-4)
        self.debug = debug            # Controls if 'd' messages go to file

        self.parent_logger = parent_logger

        # --- INHERITANCE INJECTION ---
        if inheritance:
            clear_log = False
            self.log_file = inheritance.log_file
            self.no_log = inheritance.no_log
            self.label = inheritance.label
            self.bench = inheritance.bench
            self.benchmarks = inheritance.benchmarks
            self.start_time = inheritance.start_time
            self.warnings = inheritance.warnings
        else:
            self.bench = False
            self.benchmarks = None
            self.start_time = time.time()
            self.warnings = 0

        self.pids = [psutil.Process(os.getpid())] if HAS_PSUTIL else []
        self.step_start_time = 0
        self.step_interrupt = True

        # States for Warning Buffering
        self.step_active = False
        self.step_buffer = []
        
        # Ensure dir exists & clear log file
        if not self.no_log:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            if clear_log:
                with open(self.log_file, 'w') as f:
                    f.write("")

        if catch_exceptions and not self.no_log and self.log_file:
            self.catch_all_exceptions()

    @classmethod
    def dummy(cls) -> 'GranLogger':
        """
        Returns a completely silent dummy logger that bypasses all file I/O 
        and terminal output. Safe to pass anywhere a GranLogger is expected.
        """
        return cls(
            log_file=None,          # No Path object to accidentally write to
            verbosity=-1,           # -1 guarantees even level 0 errors ('e') fail the self.verbosity >= level check
            debug=False,
            catch_exceptions=False, # Crucial: prevents the dummy from hijacking sys.excepthook
            no_log=True,            # Bypasses all file open/write blocks
            clear_log=False,
            label="dummy"
        )

    @property
    def inheritance(self) -> LogInheritance:
        """Returns the current logger's state for inheritance by follower loggers."""
        return LogInheritance(
            # Init arg params
            log_file=self.log_file,
            no_log=self.no_log,
            label=self.label,
            # Init default params
            bench=self.bench,
            benchmarks=self.benchmarks,
            start_time=self.start_time,
            warnings=self.warnings
        )

    @property
    def disable_tqdm(self) -> bool:
        stream = sys.stderr
        return self.verbosity < 3 or not (hasattr(stream, "isatty") and stream.isatty())

    def log(self, msg: str, key: str, to_screen: bool = True, kill_on_error: bool = True, prefix: str = "# ") -> None:
        """
        Unified logging function.
        Writes to log file and optionally to screen based on verbosity.
        Args:
            msg: The message to log.
            key: Priority key indicating level and type.
                'e' -> Error (Level 0) - Always prints/logs, Exits.
                'i' -> Info (Level 1) - Start/End banners, "In progress...".
                'w' -> Warning (Level 2) - Warnings.
                's' -> Standard (Level 2) - Additional input and options info.
                'd' -> Debug (Level 3) - General debug info.
                'dx'-> Debug (Level 4) - Tree strings, detailed matrix info.
            to_screen: Whether to allow printing to stdout (subject to verbosity).
            kill_on_error: If True, exits program on 'e'.
        """
        # 1. Map Key to Int Level (for Screen)
        # 'e'=0 (always), 'i'=1, 'w'/'s'=2, 'd'=3, 'dx'=4
        key_map = {'e': 0, 'i': 1, 'w': 2, 's': 2, 'd': 3, 'dx': 4}
        
        level = key_map.get(key, 0)
        if key == 'e': 
            prefix = "# ERROR: " if str(msg)[0].isupper() else "# Error "
        elif key == 'w':
            prefix = "# WARNING: "
            self.warnings += 1
        elif key in ('d', 'dx'):
            prefix = "# [DEBUG]: "
        elif key not in key_map:
            prefix += f"ERROR: Unknown log mode: {key}\n# "
            key = 'e'

        formatted_msg = prefix + str(msg)

        # Logic: Write if NOT nolog. 
        # Exception: If key is 'd', only write if debug is True.
        log_to_file = not self.no_log
        if key in ('d', 'dx') and not self.debug:
            log_to_file = False

        if log_to_file:
            # Propagate to Parent (File only)
            if self.parent_logger:
                self.parent_logger.log(msg, key, to_screen=False, kill_on_error=False, prefix=prefix)

            try:
                # Append to log file
                with open(self.log_file, "a") as f:
                    f.write(formatted_msg + "\n")
                    f.flush()
                    os.fsync(f.fileno())
            except Exception:
                pass # Fail silently if the logger itself crashes, so it doesn't mask the original error

        # Screen Output & Buffering
        if to_screen and self.verbosity >= level:
            if key == 'w' and self.step_active:
                # Buffer warnings to prevent breaking "In progress..." lines
                self.step_buffer.append(formatted_msg)
            else:
                if not self.step_interrupt:
                    self.step_interrupt = True
                    print("") # Print newline first to clear "In progress..." line before printing the message
                print(formatted_msg)
            
        # Kill program on error
        if key == 'e' and kill_on_error:
            # Flush buffered warnings if any before exiting
            if self.step_buffer:
                print("\n".join(self.step_buffer))
            sys.exit(1)

    # check the original code for when it was screen printing!
    # combine level and key but make required

    def catch_all_exceptions(self) -> None:
        """
        Overrides the global Python exception hook to route all unhandled 
        crashes through this logger's 'e' state before exiting.
        """
        def handle_exception(exc_type, exc_value, exc_traceback):
            # Let Ctrl+C (KeyboardInterrupt) kill the program normally without logging a massive error
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return

            # Extract the full traceback as a formatted string
            tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            tb_string = "".join(tb_lines)

            # Log the crash using your custom file-flushing 'e' state
            self.log(f"UNEXPECTED FATAL EXCEPTION:\n{tb_string}", 'e')

        # Bind the custom handler to Python's global hook
        sys.excepthook = handle_exception

    def assimilate(self, worker_log_path: Path, warnings: int = 0) -> None:
        """Safely appends a finished worker's log into the main log."""
        self.warnings += warnings

        if not worker_log_path.exists(): 
            return
            
        if not self.no_log:
            with open(self.log_file, 'a') as main_f:
                with open(worker_log_path, 'r') as worker_f:
                    # Read the worker log and write it directly
                    main_f.write(worker_f.read())

            # Pass to parent logger if exists (for multi-level workers)
            if self.parent_logger:
                self.parent_logger.assimilate(worker_log_path, warnings)

    def space(self, s: Any, width: int) -> str:
        return str(s) + " " * (width - len(str(s)))

    def get_date_time(self) -> str:
        return datetime.datetime.now().strftime("%m.%d.%Y  %H:%M:%S")

    def report_step(self, step_name: str, status: str, start: bool = False, full_update: bool = False, enable_benchmark: bool = False) -> None:
        """Mimics the specific table-like reporting of the old GRAMPA."""

        # Standard visual widths (Total width including the "# " prefix)
        col_widths = [12, 10, 50, 40, 20, 16]
        if HAS_PSUTIL:
            col_widths += [18, 10]

        current_time = time.time()

        if start:
            self.benchmarks = []
            if enable_benchmark:
                self.bench = True

            headers = ["Date", "Time", "Current step", "Status", "Elapsed time (s)", "Step time (s)"]
            if HAS_PSUTIL:
                headers += ["Current mem (MB)", "Virtual mem (MB)"]

            # Adjust first column: log() adds "# " (2 chars), so we subtract 2 from the first header spacing
            #header_widths = list(col_widths)
            #header_widths[0] -= 2 
                
            header_str = "".join([self.space(h, w) for h, w in zip(headers, col_widths)])
            border = "-" * (175 if HAS_PSUTIL else 150)
            
            self.log(border, 'i')
            self.log(header_str, 'i')
            self.log(border, 'i')
            return

        # Time calcs
        prog_elapsed = f"{current_time - self.start_time:.5f}"
        
        # Memory calcs
        mem_str, vmem_str = "", ""
        if HAS_PSUTIL:
            mem = round(sum([p.memory_info().rss for p in self.pids]) / float(2 ** 20), 5)
            vmem = round(sum([p.memory_info().vms for p in self.pids]) / float(2 ** 20), 5)
            mem_str = str(mem)
            vmem_str = str(vmem)

        if status == "In progress...":
            # Step start
            self.step_active = True
            self.step_interrupt = False
            self.step_buffer = []

            self.step_start_time = current_time
            out_parts = [
                datetime.datetime.now().strftime('%m.%d.%Y'),
                datetime.datetime.now().strftime('%H:%M:%S'),
                step_name,
                status
            ]
            line = "".join([self.space(p, w) for p, w in zip(out_parts, col_widths[:4])])
            
            if self.verbosity > 1:
                sys.stdout.write("# " + line)
                sys.stdout.flush()
        else:
            # Step completion
            self.step_active = False

            step_elapsed = f"{current_time - self.step_start_time:.5f}"

            if self.benchmarks is not None:
                self.benchmarks.append((step_name, step_elapsed))
            
            # Legacy logic: File gets full line, Screen gets partial update or full update
            full_parts = [
                datetime.datetime.now().strftime('%m.%d.%Y'),
                datetime.datetime.now().strftime('%H:%M:%S'),
                step_name,
                status,
                prog_elapsed,
                step_elapsed
            ]
            if HAS_PSUTIL:
                full_parts += [mem_str, vmem_str]
            
            file_line = "".join([self.space(p, w) for p, w in zip(full_parts, col_widths)])
            
            # Screen output
            if self.verbosity > 1:
                if full_update or self.step_interrupt:
                    sys.stdout.write("# " + file_line + "\n")
                else:
                    # Clear "In progress..."
                    sys.stdout.write("\b" * 40)
                    # Construct just the status part onwards
                    screen_parts = [status, prog_elapsed, step_elapsed]
                    if HAS_PSUTIL:
                        screen_parts += [mem_str, vmem_str]
                    
                    # Columns start from index 3 (Status)
                    screen_widths = col_widths[3:]
                    screen_line = "".join([self.space(p, w) for p, w in zip(screen_parts, screen_widths)])
                    sys.stdout.write(screen_line + "\n")
                
                sys.stdout.flush()

                # FLUSH WARNINGS after status line is printed
                if self.step_buffer:
                    print("\n".join(self.step_buffer))
                    self.step_buffer = [] # Clear
            
            # Write full line to log
            self.log(file_line, 'i', to_screen=False)
            self.step_interrupt = True

    def log_software_banner(self, meta: 'GranMetadata') -> None:
        """Prints the static software info (Authors, DOI, Version)."""
        # This replaces the first half of the old print_start_banner
        key = 'i' if self.verbosity == 0 else 's'
        log_ = lambda msg: self.log(msg, key)
        
        log_("")
        log_("=" * 73)
        log_(f"Welcome to GRANDMA -- {meta.version} .")
        log_(f"Version {meta.version} released on {meta.release}")
        log_(f"GRANDMA was developed by {meta.authors}")
        log_(f"\t\tinspired by GRAMPA [Gene tree reconciliations with MUL-trees] by {meta.source_authors}")
        log_(f"Citation:      {meta.doi}")
        log_(f"Website:       {meta.http}")
        log_(f"Report issues: {meta.github}")
        log_("")
        log_(f"The date and time at the start is:  {self.get_date_time()}")
        log_(f"Using Python executable located at: {sys.executable}")
        log_(f"Using Python version:               {'.'.join(map(str, sys.version_info[:3]))}")
        log_(f"\n# The program was called as:          {' '.join(sys.argv)}\n#")

    def norun_banner(self) -> None:
        key = 'i' if self.verbosity == 0 else 's'
        self.log("-" * 125, key)
        self.log("--norun SET. EXITING AFTER PRINTING OPTIONS INFO...", 'i')
        self.log("", key)

    def title_banner(self, title: str) -> None:
        key = 'i' if self.verbosity == 0 else 's'
        self.log("=" * 73, 'i')
        if title:
            self.log(f"--- {title.upper()} ---", key)

    def start_info(self, ctx: 'GlobalContext', tcf: 'TaskConfig') -> None:
        """Prints the overarching configuration once at the start of the entire Engine run."""
        key = 'i' if self.verbosity == 0 else 's'
        pad = 38 # was 40, to account for "# " prefix
        
        # Aliases for clean readability
        log_ = lambda msg: self.log(msg, key)
        space = self.space
        mode = tcf.mode

        # Comdition aliases
        is_task = bool(self.label)
        is_iter = mode in ("split", "full", "mixed")
        is_top_level = not is_task and is_iter
        is_singleton = not is_task and not is_iter
        is_terminal = is_singleton or is_task # == not is_top_level

        if is_task:
            log_("=" * 125)
            log_(f"--- BEGIN TASK LOG: {self.label} ---")
            log_("=" * 125)
        else:
            log_("-" * 125)

        # --- Input / Output Info ---
        
        log_("INPUT / OUTPUT FILES:")
        st_str = str(tcf.st) if isinstance(tcf.st, (Path, str)) else ("None" if not tcf.st else "Memory object")
        log_(space("Species tree input:", pad) + st_str)
        gts_str = str(tcf.gts) if isinstance(tcf.gts, (Path, str)) else ("None" if not tcf.gts else "Memory object")
        log_(space("Gene tree input:", pad) + gts_str)

        if is_terminal:
            log_(space("Output directory:", pad) + str(tcf.output_dir))
            logging_str = "Off" if ctx.nolog else str(tcf.output_dir / "grandma.log")
            log_(space("Log file:", pad) + logging_str)
            if mode not in ("label-sp", "count-mts", "build-mts"):
                if mode != "check-nums":
                    log_(space("Score file:", pad) + str(tcf.output_dir / f"{tcf.run_prefix}-scores.txt"))
                log_(space("Filtered gene trees:", pad) + str(tcf.output_dir / f"{tcf.run_prefix}-trees-filtered.txt"))
                log_(space("Check nums file:", pad) + str(tcf.output_dir / f"{tcf.run_prefix}-checknums.txt"))
                if mode != "check-nums":
                    log_(space("Detailed mapping file:", pad) + str(tcf.output_dir / f"{tcf.run_prefix}-detailed.txt"))
                    log_(space("Duplication count file:", pad) + str(tcf.output_dir / f"{tcf.run_prefix}-dup-counts.txt"))
            if ctx.bench:
                log_(space("Benchmarks file:", pad) + str(tcf.output_dir / f"{tcf.run_prefix}-benchmarks.txt"))
        else:
            log_(space("Root output directory:", pad) + str(ctx.root_dir))
            logging_str = "Off" if ctx.nolog else str(ctx.root_dir / "grandma.log")
            log_(space("Root log file:", pad) + logging_str)

        if not is_task:
            if ctx.plot and mode in ("split", "full", "mixed"):
                log_(space("Flow plot file:", pad) + str(ctx.root_dir / "metrics_plot.png"))
            if mode in ("split", "full", "mixed", "single", "no-st"):
                log_(space("Event history file:", pad) + str(ctx.root_dir / "history.json"))
            if mode in ("split", "full", "mixed", "single", "no-st", "st-only"):
                log_(space("Final multi- & SL tree files:", pad) + str(ctx.root_dir / "final_*.tre"))

        # --- Execution Settings ---

        log_("-" * 125)
        log_("EXECUTION SETTINGS:")
        m_switch = ctx.mixed_switch
        mixed_switch_str = "" if mode != "mixed" and not m_switch else f" [switch at {str(m_switch)}]"
        log_(space("------ MODE ------:", pad) + f"{str(mode).upper()}{mixed_switch_str}")
        if is_top_level:
            iter_text = "Unlimited" if ctx.max_iter == float('inf') else str(ctx.max_iter)
            log_(space("Max iterations:", pad) + iter_text)
            log_(space("Start iteration:", pad) + str(ctx.start_pt))
        if not is_task or ctx.debug:
            log_(space("Automatic tree repair:", pad) + str("On" if tcf.repair else "Off"))
        if not is_task and mode not in ("label-sp", "count-mts", "build-mts", "check-nums"):
            log_(space("Orthology labeling analysis:", pad) + str("On" if ctx.orth_opt else "Off"))

        # --- Algorithmic Settings ---

        if mode != "label-sp":
            log_("-" * 125)
            log_("ALGORITHMIC SETTINGS:")
            if mode != "st-only":
                if is_terminal:
                    if tcf.predefined_rets:
                        log_(space("Predefined reconciliations from MT input:", pad) + str(tcf.predefined_rets))
                    h1_str = tcf.h1_nodes if tcf.h1_nodes else "All"
                    log_(space("H1 search space:", pad) + str(h1_str))
                    h2_str = tcf.h2_nodes if tcf.h2_nodes else "All"
                    log_(space("H2 search space:", pad) + str(h2_str))
                if tcf.ploidies:
                    ploidies_str = str(tcf.ploidies) if isinstance(tcf.ploidies, (str, Path)) else "Memory object"
                    log_(space("Ploidy constraint input:", pad) + ploidies_str)
                    log_(space("Ploidy constraint behavior:", pad) + "Strict" if ctx.strict_max else "Lineage-based")
                    if is_task and mode in ("split", "mixed"):
                        log_(space("Depth ploidy constraint:", pad) + str(tcf.binary_id))
                log_(space("Redundant MT filter:", pad) + str("Off" if ctx.allow_redun else "On"))
                if ctx.nesting == "model" and mode not in ("full", "mixed"):
                    log_(space("Nestedness behavior:", pad) + str(ctx.nesting).capitalize())
            if mode not in ("count-mts", "build-mts"):
                log_(space("GT polyploid group cap:", pad) + str(tcf.group_cap))
                if mode != "check-nums":
                    log_(space("Optimized reconciliation:", pad) + str("On" if ctx.optim else "Off"))
                    log_(space("Parsimony penalty weights:", pad) + f"Dup: {tcf.weights[0]}, Loss: {tcf.weights[1]}")
                    max_select_str = str(tcf.max_select) if tcf.max_select > 0 else ("Up to input ST (inclusive)" if not tcf.max_select else "All")
                    if mode != "st-only":
                        log_(space("Max number of MTs to select:", pad) + max_select_str)
                    if mode in ("split", "full", "mixed"):
                        log_(space("Parsimony score cutoff:", pad) + f"Type: {ctx.cutoff[0]}, Value: {ctx.cutoff[1]}")
                        if mode in ("full", "mixed"):
                            log_(space("Nestedness behavior:", pad) + str(ctx.nesting).capitalize())
                        if mode in ("split", "mixed"):
                            log_(space("Min extracted tree leaves:", pad) + f"Species: {ctx.min_st_lvs}, Genes: {ctx.min_gt_lvs}")
        
        log_("-" * 125)
        log_("SYSTEM & OUTPUT:")
        if not is_task:
            log_(space("Outputs prefix:", pad) + str(tcf.run_prefix))
        log_(space("Parallel processes:", pad) + str(ctx.num_processes))
        log_(space("Verbosity level:", pad) + str(self.verbosity))
        if not is_task and mode not in ("label-sp", "count-mts", "build-mts"):
            log_(space("Pickle directory handling:", pad) + str(ctx.pickles).capitalize())
            if mode != "check-nums":
                log_(space("Detailed maps output:", pad) + str("On" if ctx.maps else "Off"))
        if ctx.debug:
            log_(space("Debugging state:", pad) + f"Seed={ctx.seed}, N={ctx.sample}")
        
        if self.verbosity == 1:
            log_("-" * 125)
            log_(f"{self.get_date_time()} INFO: Starting GRANDMA. With -v 1 set, minimal screen output will be printed.")

    def end_report(self, min_score: int = 0, min_idx: int = 0, min_tree_str: str = "") -> None:
        """Replicates endProg from reconcore.py"""
        total_time = time.time() - self.start_time
        output_dir = self.log_file.parent
        fn_prefix = self.log_file.stem
        key = 'i' if self.verbosity == 0 else 's'

        # Aliases for clean readability
        log_ = lambda msg: self.log(msg, key)

        log_("=" * 175)
        log_("\n# Done!")
        log_(f"The date and time at the end is: {self.get_date_time()}")
        log_(f"Total execution time:            {round(total_time, 3)} seconds.")
        log_(f"Output directory for this run:   {output_dir}")
        if not self.no_log:
            log_(f"Log file for this run:           {self.log_file}")
        if self.bench:
            bench_file = output_dir / f"{fn_prefix}-benchmarks.txt"
            try:
                with open(bench_file, 'w') as f:
                    f.write("Step\tTime_Seconds\n")
                    for step_name, elapsed in self.benchmarks:
                        f.write(f"{step_name}\t{elapsed}\n")
                log_(f"Benchmarks saved to:             {bench_file}")
            except Exception as e:
                self.log(f"Failed to write benchmarks: {e}", 'w')
            # Reset bench state for potential next reports
            self.bench = False
        # Clear benchmarks after end report's writing
        self.benchmarks = None
        if self.warnings > 0:
            log_(f"\n# Task finished with {self.warnings} WARNINGS -- check log file for more info")

        if min_tree_str:
            log_("-" * 125)
            if min_idx != 0:
                log_(f"The MUL-tree with the minimum parsimony score is MT-{min_idx}:\t{min_tree_str}")
            else:
                log_(f"The tree with the minimum parsimony score is the singly-labeled tree (ST):\t{min_tree_str}")
            log_(f"Score = {min_score}")

        if self.label:
            log_("=" * 125)
            log_(f"--- END TASK LOG: {self.label} ---")
            log_("=" * 125)
        else:
            log_("-" * 125)

        log_("")
        return None # Explicitly return None to reset benchmarks

    def final_report(self, final_tree: Optional['SmrtTree'], ret_tree: Optional[Any], orig_tree: Optional[Any], is_iter: bool = False, to_plot: bool = False) -> None:
        """Prints the overarching global summary for the entire GRANDMA run."""
        key = 'i' if self.verbosity == 0 else 's'
        log_ = lambda msg: self.log(msg, key)

        out_dir = self.log_file.parent

        log_("")

        if is_iter:
            log_("=" * 125)
            log_("--- ANALYSIS SUMMARY ---")
            log_("=" * 125)
            
            total_time = time.time() - self.start_time
            log_(f"The date and time at the end is: {self.get_date_time()}")
            log_(f"Total execution time:            {round(total_time, 3)} seconds.")

            log_(f"Output directory for this run:   {out_dir}")
            log_(f"Main log file:                   {self.log_file}")
            
            plot_file = out_dir / "metrics_plot.png"
            if to_plot and plot_file.exists():
                log_(f"Metrics plot saved to:           {plot_file}")
                
            if ret_tree is not None:
                viz_file = out_dir / "final_tree.png"
                ret_tree.visualize(filename=viz_file, launch=False)
                log_(f"Tree visualization saved to:     {viz_file}")

        if final_tree:
            log_(f"Multi -labelled form written to: {out_dir / 'final_multree.tre'}")
            log_(f"Singly-labelled form written to: {out_dir / 'final_single_label_form.tre'}")
            log_(f"Enriched Newick form written to: {out_dir / 'final_enriched_newick.tre'} [Not Implemented Yet]")
            if not final_tree.are_all_nodes_unique():
                self.log("Final tree contains non-unique node labels, which may cause issues for some downstream applications.", 'w')
            if not final_tree.contains(orig_tree):
                self.log("The original input species tree topology or node names were NOT perfectly preserved in the final merged tree!", 'w')

        if is_iter and self.warnings > 0:
            log_(f"\n# Pipeline finished with {self.warnings} WARNINGS -- check log file for more info")

        if final_tree:
            log_("-" * 125)
            log_("FINAL TREE (Newick):")
            log_(final_tree.ete_tree.write(format=9))
            
            log_("-" * 125)
            log_("FINAL TREE (ASCII):")
            log_(final_tree.ete_tree.get_ascii(show_internal=True))
            log_("")

        log_("=" * 125)
        log_("")