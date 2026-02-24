'''
Replaces reconcore.py printing logic to ensure identical output format.
'''

from multiprocessing.util import debug
import sys
import time
import datetime
import os
from pathlib import Path
from typing import TYPE_CHECKING

# This block is ignored at runtime, and solves the circular dependency of typing
if TYPE_CHECKING:
    from .config import GranMetadata, GlobalContext, TaskConfig

# Try importing psutil for memory logging
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

class GranLogger:
    def __init__(self, log_file: Path, verbosity: int = 4, debug: bool = False, no_log: bool = False,
                 parent_logger: 'GranLogger' = None, clear_log: bool = True):
        self.log_file = log_file
        self.verbosity = verbosity    # Controls screen output (0-4)
        self.debug = debug       # Controls if 'd' messages go to file
        self.no_log = no_log          # Controls if ANY messages go to file
        self.parent_logger = parent_logger
        self.start_time = time.time()
        self.step_start_time = 0
        self.pids = [psutil.Process(os.getpid())] if HAS_PSUTIL else []
        self.warnings = 0
        # States for Warning Buffering
        self.step_active = False
        self.step_buffer = []
        
        # Ensure dir exists & clear log file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        if clear_log:
            with open(self.log_file, 'w') as f:
                f.write("")

    def log(self, msg: str, key: str, to_screen: bool = True, kill_on_error: bool = True):
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
        prefix = "# "
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
            # Append to log file
            with open(self.log_file, "a") as f:
                f.write(formatted_msg + "\n")
            
            # Propagate to Parent (File only)
            if self.parent_logger:
                self.parent_logger.log(msg, key, to_screen=False, kill_on_error=False)

        # Screen Output & Buffering
        if to_screen and self.verbosity >= level:
            if key == 'w' and self.step_active:
                # Buffer warnings to prevent breaking "In progress..." lines
                self.step_buffer.append(formatted_msg)
            else:
                print(formatted_msg)
            
        # Kill program on error
        if key == 'e' and kill_on_error:
            sys.exit(1)

    # check the original code for when it was screen printing!
    # combine level and key but make required

    def spaced(self, s, width):
        return str(s) + " " * (width - len(str(s)))

    def get_date_time(self):
        return datetime.datetime.now().strftime("%m.%d.%Y  %H:%M:%S")

    def report_step(self, step_name: str, status: str, start: bool = False, full_update: bool = False):
        """Mimics the specific table-like reporting of the old GRAMPA."""

        # Standard visual widths (Total width including the "# " prefix)
        col_widths = [14, 10, 50, 40, 20, 16]
        if HAS_PSUTIL:
            col_widths += [18, 10]

        current_time = time.time()

        if start:
            headers = ["Date", "Time", "Current step", "Status", "Elapsed time (s)", "Step time (s)"]
            if HAS_PSUTIL:
                headers += ["Current mem (MB)", "Virtual mem (MB)"]

            # Adjust first column: log() adds "# " (2 chars), so we subtract 2 from the first header spacing
            header_widths = list(col_widths)
            header_widths[0] -= 2 
                
            header_str = "".join([self.spaced(h, w) for h, w in zip(headers, header_widths)])
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
            self.step_buffer = []

            self.step_start_time = current_time
            out_parts = [
                f"# {datetime.datetime.now().strftime('%m.%d.%Y')}",
                datetime.datetime.now().strftime('%H:%M:%S'),
                step_name,
                status
            ]
            line = "".join([self.spaced(p, w) for p, w in zip(out_parts, col_widths[:4])])
            
            if self.verbosity > 1:
                sys.stdout.write(line)
                sys.stdout.flush()
        else:
            # Step completion
            self.step_active = False

            step_elapsed = f"{current_time - self.step_start_time:.5f}"
            
            # Legacy logic: File gets full line, Screen gets partial update or full update
            full_parts = [
                f"# {datetime.datetime.now().strftime('%m.%d.%Y')}",
                datetime.datetime.now().strftime('%H:%M:%S'),
                step_name,
                status,
                prog_elapsed,
                step_elapsed
            ]
            if HAS_PSUTIL:
                full_parts += [mem_str, vmem_str]
            
            file_line = "".join([self.spaced(p, w) for p, w in zip(full_parts, col_widths)])
            
            # Screen output
            if self.verbosity > 1:
                if full_update:
                    sys.stdout.write(file_line + "\n")
                else:
                    # Clear "In progress..."
                    sys.stdout.write("\b" * 40)
                    # Construct just the status part onwards
                    screen_parts = [status, prog_elapsed, step_elapsed]
                    if HAS_PSUTIL:
                        screen_parts += [mem_str, vmem_str]
                    
                    # Columns start from index 3 (Status)
                    screen_widths = col_widths[3:]
                    screen_line = "".join([self.spaced(p, w) for p, w in zip(screen_parts, screen_widths)])
                    sys.stdout.write(screen_line + "\n")
                
                sys.stdout.flush()

                # FLUSH WARNINGS after status line is printed
                if self.step_buffer:
                    print("\n".join(self.step_buffer))
                    self.step_buffer = [] # Clear
            
            # Write full line to log
            with open(self.log_file, "a") as f:
                f.write(file_line + "\n")

    def log_software_banner(self, meta: 'GranMetadata'):
        """Prints the static software info (Authors, DOI, Version)."""
        # This replaces the first half of the old print_start_banner
        key = 'i' if self.verbosity == 0 else 's'
        
        self.log("=" * 73, key)
        self.log(f"Welcome to GRANDMA -- {meta.version} .", key)
        self.log(f"Version {meta.version} released on {meta.release}", key)
        self.log(f"GRANDMA was developed by {meta.authors}", key)
        self.log(f"\t\tinspired by GRAMPA [Gene tree reconciliations with MUL-trees] by {meta.source_authors}", key)
        self.log(f"Citation:      {meta.doi}", key)
        self.log(f"Website:       {meta.http}", key)
        self.log(f"Report issues: {meta.github}", key)
        self.log("", key)
        self.log(f"The date and time at the start is:  {self.get_date_time()}", key)
        self.log(f"Using Python executable located at: {sys.executable}", key)
        self.log(f"Using Python version:               {'.'.join(map(str, sys.version_info[:3]))}", key)
        self.log(f"\n# The program was called as:          {' '.join(sys.argv)}\n#", key)

    def start_run(self, ctx: 'GlobalContext', tcf: 'TaskConfig'):
        """Replicates startProg from opt_parse.py"""
        if ctx.norun: return
        
        key = 'i' if self.verbosity == 0 else 's'

        self.log("-" * 125, key)
        self.log("INPUT/OUTPUT INFO:", key)

        pad = 38 # NEW: was 40, to account for "# " prefix
        
        # Files
        if isinstance(tcf.st, (Path, str)):
            self.log(self.spaced("Species tree file:", pad) + str(tcf.st), key)
        if not tcf.is_mul_input and isinstance(tcf.gts, (Path, str)):
             self.log(self.spaced("Gene tree file:", pad) + (str(tcf.gts) if tcf.gts else ""), key)
        
        self.log(self.spaced("Output directory:", pad) + str(tcf.output_dir), key)
        
        if not tcf.is_mul_input:
            self.log(self.spaced("Score file:", pad) + str(Path(tcf.output_dir) / f"{tcf.run_prefix}-scores.txt"), key)
            self.log(self.spaced("Filtered gene trees:", pad) + str(Path(tcf.output_dir) / f"{tcf.run_prefix}-trees-filtered.txt"), key)
            self.log(self.spaced("Check nums file:", pad) + str(Path(tcf.output_dir) / f"{tcf.run_prefix}-checknums.txt"), key)
            if tcf.mode != "check-nums":
                self.log(self.spaced("Detailed mapping file:", pad) + str(Path(tcf.output_dir) / f"{tcf.run_prefix}-detailed.txt"), key)
                self.log(self.spaced("Duplication count file:", pad) + str(Path(tcf.output_dir) / f"{tcf.run_prefix}-dup-counts.txt"), key)

        self.log("-" * 125, key)
        self.log("OPTIONS INFO:", key)
        self.log(self.spaced("Option", pad) + self.spaced("Current setting", 30) + "Current action", key)
        
        # Options Table
        # -h1
        h1_str = tcf.h1_nodes if tcf.h1_nodes else "All"
        self.log(self.spaced("-h1", pad) + self.spaced(h1_str, 30) + "GRAMPA will search these H1 nodes. If none are specified, all nodes will be searched as H1 nodes.", key)
        # -h2
        h2_str = tcf.h2_nodes if tcf.h2_nodes else "All"
        self.log(self.spaced("-h2", pad) + self.spaced(h2_str, 30) + "GRAMPA will search these H2 nodes. If none are specified, all nodes will be searched as H2 nodes.", key)
        # -c
        self.log(self.spaced("-c", pad) + self.spaced(str(tcf.group_cap), 30) + "Gene trees with more than this number of groups/clades with polyploid species for a given h1/h2 combination will be skipped.", key)
        # -f
        self.log(self.spaced("-f", pad) + self.spaced(tcf.run_prefix, 30) + "All output files generated will have this string preprended to them.", key)
        # -p
        self.log(self.spaced("-p", pad) + self.spaced(str(ctx.num_processes), 30) + "GRAMPA will use this number of processes for LCA mapping.", key)
        # -v
        self.log(self.spaced("-v", pad) + self.spaced(str(self.verbosity), 30) + "Controls the amount of info printed to the screen as GRAMPA is running.", key)
        # --multree
        mul_str = "The tree input with -s will be read as a MUL-tree." if tcf.is_mul_input else "The tree input with -s will be read as singly-labeled tree."
        self.log(self.spaced("--multree", pad) + self.spaced(str(tcf.is_mul_input), 30) + mul_str, key)
        # --checknums
        cn_str = "GRAMPA will count groups to filter gene trees and exit." if tcf.mode == "check-nums" else "GRAMPA will count groups to filter gene trees and then perform reconciliations."
        self.log(self.spaced("--checknums", pad) + self.spaced(str(tcf.mode == "check-nums"), 30) + cn_str, key)
        # --no-st, --st-only
        st_opt_str = "default"
        if tcf.mode == "no-st": st_opt_str = "no-st"
        if tcf.mode == "st-only": st_opt_str = "st-only"
        st_desc = "GRAMPA will perform reconciliations to all MUL-trees specified by -h1 and -h2 and the input species tree."
        if tcf.mode == "no-st": st_desc = "GRAMPA will perform reconciliations to only the MUL-trees specified by -h1 and -h2."
        if tcf.mode == "st-only": st_desc = "GRAMPA will perform reconciliations to only the input species tree."
        self.log(self.spaced("--no-st, --st-only", pad) + self.spaced(st_opt_str, 30) + st_desc, key)
        # --maps
        if tcf.mode != "check-nums":
             map_desc = "GRAMPA will output node mappings for the lowest scoring tree in the detailed output file." if tcf.to_map else "GRAMPA will only output duplication and loss counts in the detailed output file."
             self.log(self.spaced("--maps", pad) + self.spaced(str(tcf.to_map), 30) + map_desc, key)
        # --overwrite
        if tcf.overwrite:
             self.log(self.spaced("--overwrite", pad) + self.spaced("True", 30) + "GRAMPA will OVERWRITE the existing files in the specified output directory.", key)

        if self.verbosity == 1:
            self.log("-" * 125, key)
            self.log(f"{self.get_date_time()} INFO: Starting GRAMPA. With -v 1 set, no more information will be printed to the screen until the end of the run.", key)

    def print_end_prog(self, tcf, min_score=0, min_idx=0, min_data=None):
        """Replicates endProg from reconcore.py"""
        total_time = time.time() - self.start_time
        key = 'i' if self.verbosity == 0 else 's' 
        self.log("=" * 175, key)
        self.log("\n# Done!", key)
        self.log(f"The date and time at the end is: {self.get_date_time()}", key)
        self.log(f"Total execution time:            {round(total_time, 3)} seconds.", key)
        self.log(f"Output directory for this run:   {tcf.output_dir}", key)
        self.log(f"Log file for this run:           {self.log_file}", key)
        if self.warnings > 0:
            self.log(f"\n# Task finished with {self.warnings} WARNINGS -- check log file for more info", key)

        if min_data:
            min_tree_str = min_data.mt.to_marked_str(min_data.h1_node)
            self.log("-" * 40, key)
            if min_idx != 0:
                 self.log(f"The MUL-tree with the minimum parsimony score is MT-{min_idx}:\t{min_tree_str}", key)
            else:
                 self.log(f"The tree with the minimum parsimony score is the singly-labeled tree (ST):\t{min_tree_str}", key)
            self.log(f"Score = {min_score}", key)
            self.log("-" * 40, key)

        self.log("=" * 175, key)
        self.log("", key)