'''
Replaces reconcore.py printing logic to ensure identical output format.
'''

import sys
import time
import datetime
import os
from pathlib import Path

# Try importing psutil for memory logging
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

class GrandmaLogger:
    def __init__(self, log_path: Path, verbosity: int = 3):
        self.log_path = log_path
        self.verbosity = verbosity
        self.start_time = time.time()
        self.step_start_time = 0
        self.warnings = 0
        self.pids = [psutil.Process(os.getpid())] if HAS_PSUTIL else []
        
        # Ensure dir exists
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # Clear log file
        with open(self.log_path, 'w') as f:
            f.write("")

    def write(self, message: str, level: int = 2, to_screen: bool = True):
        """Writes to log file and optionally to screen based on verbosity."""
        with open(self.log_path, "a") as f:
            f.write(message + "\n")
        if to_screen and self.verbosity >= level:
            print(message)

    def spaced(self, s, width):
        return str(s) + " " * (width - len(str(s)))

    def get_date_time(self):
        return datetime.datetime.now().strftime("%m.%d.%Y  %H:%M:%S")

    def report_step(self, step_name: str, status: str, start: bool = False, full_update: bool = False):
        """Mimics the specific table-like reporting of the old GRAMPA."""
        current_time = time.time()
        
        # Formatting constants
        col_widths = [14, 10, 50, 40, 20, 16]
        if HAS_PSUTIL:
            col_widths += [18, 10]

        if start:
            headers = ["# Date", "Time", "Current step", "Status", "Elapsed time (s)", "Step time (s)"]
            if HAS_PSUTIL:
                headers += ["Current mem (MB)", "Virtual mem (MB)"]
                
            header_str = "".join([self.spaced(h, w) for h, w in zip(headers, col_widths)])
            border = "# " + "-" * (175 if HAS_PSUTIL else 150) # Legacy dashed line length logic
            
            self.write(border, level=1)
            self.write(header_str, level=1)
            self.write(border, level=1)
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
            # Completion
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
            
            # Write full line to log
            with open(self.log_path, "a") as f:
                f.write(file_line + "\n")

    def print_start_banner(self, cfg, globs_dict):
        """Replicates startProg from opt_parse.py"""
        start_v = 1 if self.verbosity == 0 else 3
        
        self.write("# =========================================================================", level=start_v)
        self.write("# Welcome to GRANDMA -- XXX .", level=start_v)
        self.write(f"# Version {cfg.version} released on {cfg.release}", level=start_v)
        self.write(f"# GRANDMA was developed by {cfg.authors}", level=start_v)
        self.write(f"# \t\tbased on GRAMPA [Gene tree reconciliations with MUL-trees] by {cfg.source_authors}", level=start_v)
        self.write(f"# Citation:      {cfg.doi}", level=start_v)
        self.write(f"# Website:       {cfg.http}", level=start_v)
        self.write(f"# Report issues: {cfg.github}", level=start_v)
        self.write("#", level=start_v)
        self.write(f"# The date and time at the start is:  {self.get_date_time()}", level=start_v)
        self.write(f"# Using Python executable located at: {sys.executable}", level=start_v)
        self.write(f"# Using Python version:               {'.'.join(map(str, sys.version_info[:3]))}", level=start_v)
        self.write(f"#\n# The program was called as:          {' '.join(sys.argv)}\n#", level=start_v)

        if cfg.info_only: return

        pad = 40
        self.write("# " + "-" * 125, level=start_v)
        self.write("# INPUT/OUTPUT INFO:", level=start_v)
        
        # Files
        self.write(self.spaced("# Species tree file:", pad) + cfg.species_tree_path, level=start_v)
        if not cfg.is_mul_input:
             self.write(self.spaced("# Gene tree file:", pad) + (cfg.gene_tree_path if cfg.gene_tree_path else ""), level=start_v)
        
        self.write(self.spaced("# Output directory:", pad) + cfg.output_dir, level=start_v)
        
        if not cfg.is_mul_input:
            self.write(self.spaced("# Score file:", pad) + str(Path(cfg.output_dir) / f"{cfg.run_prefix}-scores.txt"), level=start_v)
            self.write(self.spaced("# Filtered gene trees:", pad) + str(Path(cfg.output_dir) / f"{cfg.run_prefix}-trees-filtered.txt"), level=start_v)
            self.write(self.spaced("# Check nums file:", pad) + str(Path(cfg.output_dir) / f"{cfg.run_prefix}-checknums.txt"), level=start_v)
            if cfg.lca_opt != "check-nums":
                self.write(self.spaced("# Detailed mapping file:", pad) + str(Path(cfg.output_dir) / f"{cfg.run_prefix}-detailed.txt"), level=start_v)
                self.write(self.spaced("# Duplication count file:", pad) + str(Path(cfg.output_dir) / f"{cfg.run_prefix}-dup-counts.txt"), level=start_v)

        self.write("# " + "-" * 125, level=start_v)
        self.write("# OPTIONS INFO:", level=start_v)
        self.write(self.spaced("# Option", pad) + self.spaced("Current setting", 30) + "Current action", level=start_v)
        
        # Options Table
        # -h1
        h1_str = cfg.h1_nodes if cfg.h1_nodes else "All"
        self.write(self.spaced("# -h1", pad) + self.spaced(h1_str, 30) + "GRAMPA will search these H1 nodes. If none are specified, all nodes will be searched as H1 nodes.", level=start_v)
        # -h2
        h2_str = cfg.h2_nodes if cfg.h2_nodes else "All"
        self.write(self.spaced("# -h2", pad) + self.spaced(h2_str, 30) + "GRAMPA will search these H2 nodes. If none are specified, all nodes will be searched as H2 nodes.", level=start_v)
        # -c
        self.write(self.spaced("# -c", pad) + self.spaced(str(cfg.group_cap), 30) + "Gene trees with more than this number of groups/clades with polyploid species for a given h1/h2 combination will be skipped.", level=start_v)
        # -f
        self.write(self.spaced("# -f", pad) + self.spaced(cfg.run_prefix, 30) + "All output files generated will have this string preprended to them.", level=start_v)
        # -p
        self.write(self.spaced("# -p", pad) + self.spaced(str(cfg.num_processes), 30) + "GRAMPA will use this number of processes for LCA mapping.", level=start_v)
        # -v
        self.write(self.spaced("# -v", pad) + self.spaced(str(self.verbosity), 30) + "Controls the amount of info printed to the screen as GRAMPA is running.", level=start_v)
        # --multree
        mul_str = "The tree input with -s will be read as a MUL-tree." if cfg.is_mul_input else "The tree input with -s will be read as singly-labeled tree."
        self.write(self.spaced("# --multree", pad) + self.spaced(str(cfg.is_mul_input), 30) + mul_str, level=start_v)
        # --checknums
        cn_str = "GRAMPA will count groups to filter gene trees and exit." if cfg.lca_opt == "check-nums" else "GRAMPA will count groups to filter gene trees and then perform reconciliations."
        self.write(self.spaced("# --checknums", pad) + self.spaced(str(cfg.lca_opt == "check-nums"), 30) + cn_str, level=start_v)
        # --no-st, --st-only
        st_opt_str = "default"
        if cfg.lca_opt == "no-st": st_opt_str = "no-st"
        if cfg.lca_opt == "st-only": st_opt_str = "st-only"
        st_desc = "GRAMPA will perform reconciliations to all MUL-trees specified by -h1 and -h2 and the input species tree."
        if cfg.lca_opt == "no-st": st_desc = "GRAMPA will perform reconciliations to only the MUL-trees specified by -h1 and -h2."
        if cfg.lca_opt == "st-only": st_desc = "GRAMPA will perform reconciliations to only the input species tree."
        self.write(self.spaced("# --no-st, --st-only", pad) + self.spaced(st_opt_str, 30) + st_desc, level=start_v)
        # --maps
        if cfg.lca_opt != "check-nums":
             map_desc = "GRAMPA will output node mappings for the lowest scoring tree in the detailed output file." if cfg.maps_opt else "GRAMPA will only output duplication and loss counts in the detailed output file."
             self.write(self.spaced("# --maps", pad) + self.spaced(str(cfg.maps_opt), 30) + map_desc, level=start_v)
        # --overwrite
        if cfg.overwrite:
             self.write(self.spaced("# --overwrite", pad) + self.spaced("True", 30) + "GRAMPA will OVERWRITE the existing files in the specified output directory.", level=start_v)

        if self.verbosity == 1:
            self.write("# " + "-" * 125, level=1)
            self.write(f"# {self.get_date_time()} INFO: Starting GRAMPA. With -v 1 set, no more information will be printed to the screen until the end of the run.", level=1)

    def print_end_prog(self, cfg, min_info=None):
        """Replicates endProg from reconcore.py"""
        total_time = time.time() - self.start_time
        self.write("# " + "=" * 175, level=self.verbosity)
        self.write("#\n# Done!", level=self.verbosity)
        self.write(f"# The date and time at the end is: {self.get_date_time()}", level=self.verbosity)
        self.write(f"# Total execution time:            {round(total_time, 3)} seconds.", level=self.verbosity)
        self.write(f"# Output directory for this run:   {cfg.output_dir}", level=self.verbosity)
        self.write(f"# Log file for this run:           {self.log_path}", level=self.verbosity)

        if self.warnings > 0:
            self.write(f"\n# GRAMPA finished with {self.warnings} WARNINGS -- check log file for more info", level=self.verbosity)

        if min_info:
            # min_info = (min_num, min_score, tree_string)
            self.write("# " + "-" * 40, level=self.verbosity)
            if min_info[0] != 0:
                 self.write(f"# The MUL-tree with the minimum parsimony score is MT-{min_info[0]}:\t{min_info[2]}", level=self.verbosity)
            else:
                 self.write(f"# The tree with the minimum parsimony score is the singly-labled tree (ST):\t{min_info[2]}", level=self.verbosity)
            self.write(f"# Score = {min_info[1]}", level=self.verbosity)
            self.write("# " + "-" * 40, level=self.verbosity)

        self.write("# " + "=" * 175, level=self.verbosity)
        self.write("#", level=self.verbosity)
