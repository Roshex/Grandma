import os
import re
import sys
import random
import multiprocessing as mp
from functools import partial
from pathlib import Path
from dataclasses import replace

from .config import parse_args, GrandmaConfig, GrandmaWriter, GrandmaMetadata
from .logger import GrandmaLogger
from .models import SmrtTree
from .ops import TreeLoader, GeneTreeManager, MulTreeManager
from .flow import FlowManager
from .reconcile import Reconciler
from .orthology import OrthologyLabeler

import psutil
HAS_PSUTIL = psutil is not None

# --- Top-level Helper Functions ---

def _is_memory_st(st):
    """Check if st is a GrandmaTree object (Memory) or Path/str (Disk)."""
    return not isinstance(st, (str, Path))

def _is_not_none(obj):
    return obj is not None

# --- Standalone workers function supporting parallel processing --- #

def run_worker(task_data, config, test_func, verbosity=0, logger=None):
    """
    Unified worker for both Iterative (Full) and Recursive (Split) modes.
    task_data: (st_obj_or_path, gt_dict_or_path, id)
    """
    st, gts, id = task_data
    
    # TEMP FIX: If st is a GrandmaTree object (passed from split mode), refresh it
    if hasattr(st, 'refresh'):
        st.refresh()

    # Ensure ID-specific output and pickle sub-directories
    out = config.output_dir / id / "output"
    out.mkdir(parents=True, exist_ok=True)
    pkl_dir = config.pickle_dir.parent / id / 'output' / config.pickle_dir.name

    # Handle Gene Tree Input Type (Disk path vs Memory Dict)
    # If gts is a path/string, we must update the config so Run() loads it, 
    # and pass None to Run() so it doesn't think we gave it an empty dict.
    iter_cfg = replace(config, output_dir=out, pickle_dir=pkl_dir, verbosity=verbosity)
    # To be checked again!
    '''if isinstance(gts, (str, Path)):
        iter_cfg = replace(iter_cfg, gene_tree_path=Path(gts))
        gts = None  # Signal to Run class to use the loader'''

    # If no logger provided (preferred in multiprocessing), create local one
    if not logger:
        log_path = out / f"{config.run_prefix}.log"
        logger = GrandmaLogger(log_path, verbosity)

    run_inst = Run(iter_cfg, spec_tree=st, gene_trees=gts, logger=logger)
    res = run_inst.execute(from_memory=test_func(st))

    return id, out, res

# --- Core Classes --- #

class Run:
    """
    Represents a discrete execution unit of GRAMPA analysis.
    It contains its own logger, writer, and configuration state specific to this execution.
    It does not know about iteration history or other runs.
    """
    def __init__(self, config: GrandmaConfig, logger: GrandmaLogger = None, 
                 spec_tree: SmrtTree = None, gene_trees: dict = None):
        self.cfg = config
        
        # 1. Setup Logging/Writing for this specific run
        Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)
        if logger:
            self.logger = logger
        else:
            # Create a new logger if one wasn't passed (standard behavior)
            log_path = Path(self.cfg.output_dir) / f"{self.cfg.run_prefix}.log"
            self.logger = GrandmaLogger(log_path, self.cfg.verbosity)
        self.writer = GrandmaWriter(self.cfg, self.logger)
        
        # 2. State
        self.spec_tree = spec_tree
        self.gene_trees = gene_trees if gene_trees is not None else {}
        self.mul_trees = {}
        
        # 3. Components (Lazy init)
        self.reconciler = None
        self.mul_mgr = None
        self.gene_mgr = None

    def execute(self, from_memory=False):
        """
        Executes the analysis pipeline for this run context.
        Returns the results dict or None on failure/check-nums.
        """
        # 0. Banner
        self.logger.start_run(self.cfg, {})
        self.logger.report_step("", "", start=True)

        # 1. Load Species Tree
        if not from_memory or self.spec_tree is None:
        #if not from_memory and self.spec_tree is None:
            self.spec_tree = TreeLoader.spec_tree(self.cfg.species_tree_path, self.logger)

        # (Re)Init components with current trees
        self.reconciler = Reconciler(self.spec_tree, self.cfg)
        self.mul_mgr = MulTreeManager(self.spec_tree, self.cfg, self.logger)
        self.gene_mgr = GeneTreeManager(self.cfg, self.logger, self.reconciler)

        # 2. Build MUL-Trees
        self.mul_trees = self.mul_mgr.build()
        if self.cfg.mode == "build-mts": return None

        # 3. Load Gene Trees
        if not from_memory or not self.gene_trees:
        # We only want to load from disk if we are NOT running from memory AND the trees are missing.
        #if not from_memory and not self.gene_trees:
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
        # Note: In iterative modes, we are filtering the *memory* gene trees, 
        # which effectively filters them for subsequent iterations too.
        # TBD !!!
        self.gene_mgr.cull(self.mul_trees, self.gene_trees)
        if self.cfg.mode == "check-nums": return None

        # DEBUG
        '''for gene_num, (gt_obj, x) in self.gene_trees.items():
            print(f"Gene Tree {gene_num}: {gt_obj.to_string(internal_labels=True)}, {x}")
        '''

        # 5. Reconciliation and MUL-tree Selection
        step_result = self.reconciler.run(self.mul_trees, self.gene_trees,
                                                          self.cfg, self.logger, self.writer)

        '''if not step_result.sorted_scores:
            self.logger.write("No valid MUL-trees scored.", level=1)
            return None'''

        # 6. Extract Best Result (using StepResult properties)
        min_idx = step_result.mt_idx()
        min_data = step_result.mul_trees[min_idx]
        
        # kept_mul_results is Dict[mul_idx, Dict[g_idx, Maps]]
        min_maps = step_result.kept_mul_maps[min_idx]
        
        # 7. Orthology (Optional)
        if self.cfg.orth_opt and min_maps:
             # OrthologyLabeler expects dict of results. We pass min_maps directly.
             # NOTE: OrthologyLabeler might need adjustment if it expects a list vs dict, 
             # but standard dict iteration works for both.
             # We pass keys as gene_num, values as ReconResult
             # OrthologyLabeler.run signature: gene_trees, min_maps_dict
             # But ReconResult wraps maps. Orthology.py uses res[3] (maps) and res[4] (dups)
             # or attributes if updated. Assuming strict compat, we pass the wrapper.
             OrthologyLabeler.run(self.gene_trees, min_maps, min_data.mt, 
                                min_data.h_clade, self.cfg.output_dir, self.cfg.run_prefix)

        # 8. Final Report
        min_tree_str = min_data.mt.to_string(internal_labels=True)
        h_clade = min_data.h_clade
        for spec in h_clade:
            min_tree_str = re.sub(f"{spec}(?!\*)", f"{spec}+", min_tree_str)
            min_tree_str = min_tree_str.replace("+*", "*")
            
        self.logger.print_end_prog(self.cfg, (min_idx, step_result.mt_score(), min_tree_str))

        return step_result
        
class Engine:
    """
    Orchestrates the analysis flow based on the selected mode.
    Manages global state and sub-runs.
    """
    def __init__(self, config: GrandmaConfig):
        self.cfg = config
        
        # In Single mode, we don't need a separate engine logger.
        # In Full mode, we might want a flow logger.
        # We initialize it lazily or just use print for high-level flow messages.
        self.flow_logger = None

    def _init_flow_logger(self):
        Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)
        log_path = Path(self.cfg.output_dir) / f"{self.cfg.run_prefix}.log"
        self.flow_logger = GrandmaLogger(log_path, self.cfg.verbosity, clear_log=False)
        '''# In resume, append to log, don't clear
        is_resume = (self.cfg.history is not None and len(self.cfg.history) > 0)
        self.flow_logger = GrandmaLogger(log_path, self.cfg.verbosity, clear_log=not is_resume)'''

    def run(self):
        # 1. Seed Initialization
        if self.cfg.seed is not None:
            random.seed(self.cfg.seed)
            # if 'numpy' in sys.modules: import numpy; numpy.random.seed(self.cfg.seed)

        self._init_flow_logger()
        meta = GrandmaMetadata()
        # Print software info once for the whole session
        self.flow_logger.log_software_banner(meta)
        self.flow_logger.write(f"# Running in Mode: {self.cfg.mode} | Seed: {self.cfg.seed}", level=1)
        self.flow_logger.write("# " + "=" * 73, level=1)
        if self.cfg.mode in ["no-st", "st-only", "build-mts", "check-nums", "single"]:
            self.run_single()
        elif self.cfg.mode == "full":
            self.run_full()
        elif self.cfg.mode == "split":
            self.run_split()
        else:
            sys.exit(f"Unknown mode {self.cfg.mode}")

    def run_single(self):
        """
        Executes a single run using the global configuration directly.
        Matches legacy behavior exactly: one run, one output folder, one log.
        """
        # Create Run with global config
        run = Run(self.cfg, logger=self.flow_logger)
        res = run.execute(from_memory=False)
        return res
        
    def run_full(self):
        """
        Iterative mode. 
        Creates a dedicated folder and Run instance for each iteration.
        Passes tree objects in memory to avoid reloading.
        """
        self.flow_logger.write("# Starting Fully Sequential Mode", level=1)

        # Setup: initial parameters must have been parsed already in io.py
        i = self.cfg.start_pt
        max_iter = self.cfg.max_iter
        base_output_dir = self.cfg.output_dir

        flow_mgr = FlowManager(
            iter_num=i,
            cutoff_cfg=self.cfg.cutoff,
            ignore_nesting=self.cfg.ignore_nesting,
            history=self.cfg.history,
            history_file=self.cfg.history_file,
            output_dir=base_output_dir,
            seed = self.cfg.seed
        )

        current_st = None
        current_gts = None

        # If resuming, we rely on Run() loading from files via cfg paths in the first iteration loop,
        # OR we load them here. Actually, io.py sets cfg.species_tree_path to the previous output 
        # if i > 0. So current_st can remain None for the first loop.
       
        # Iteration Loop
        try:
            while i < max_iter:
                i += 1
                self.flow_logger.write(
                    f'#\n##### Iteration {i}' + (' (inf mode) #####\n#' if max_iter == float('inf') else f' of {int(max_iter)} #####\n#'))#∞

                # Use run_worker for the actual execution
                task_data = (current_st, current_gts, str(i-1))
                iter_out = self.cfg.output_dir / str(i-1) / "output"
                iter_logger = GrandmaLogger(iter_out / f"{self.cfg.run_prefix}.log", self.cfg.verbosity, parent_logger=self.flow_logger)
                _, _, res = run_worker(
                    task_data, self.cfg, 
                    test_func = _is_not_none,
                    verbosity = self.cfg.verbosity,
                    logger = iter_logger
                )

                '''# Prepare Task
                # If current_st is None (first loop or resume), we pass Paths from cfg.
                # If current_st is Obj (subsequent loops), we pass Objs.
                task_st = current_st if current_st else self.cfg.species_tree_path
                task_gts = current_gts if current_gts else self.cfg.gene_tree_path
                
                # Setup output folder for this iteration
                task_id = str(i-1)
                iter_out = self.cfg.output_dir / task_id / "output"
                iter_logger = GrandmaLogger(iter_out / f"{self.cfg.run_prefix}.log", 
                                            self.cfg.verbosity, parent_logger=self.flow_logger)
                '''

                if not res:
                    self.flow_logger.write("# No results generated in iteration.")
                    break

                # Process result and handle potential nesting
                # This returns the trees prepared for the NEXT iteration
                next_trees = flow_mgr.handle_iteration_result(
                    i, res, iter_out, 
                    engine_callback=lambda st, gts, h1, h2, out: self._run_nested_subproblem(st, gts, h1, h2, out),
                    logger=iter_logger,
                    debug=self.cfg.debug
                )
                
                if not next_trees: break
                current_st, current_gts = next_trees
                
        except KeyboardInterrupt:
            self.flow_logger.write("# Interrupted by user.", level=1)

        if self.cfg.plot:
            flow_mgr.plot(base_output_dir)
        
        self.flow_logger.write("# Fully Sequential Mode Finished.")

    def _run_nested_subproblem(self, st_obj, gt_objs, h1_str, h2_str, fix_dir):
            """Helper to run a constrained nested fix run."""
            pkl_dir = fix_dir / "pkls"
            fix_cfg = replace(self.cfg, 
                            output_dir=fix_dir,
                            pickle_dir=pkl_dir,
                            h1_nodes=h1_str, 
                            h2_nodes=h2_str,
                            mode="no-st",
                            verbosity=0)

            mem_st = SmrtTree(tree_obj=st_obj)
            mem_gts = {k: SmrtTree(tree_obj=v) for k, v in enumerate(gt_objs)}
            
            run_inst = Run(fix_cfg, spec_tree=mem_st, gene_trees=mem_gts)
            return run_inst.execute(from_memory=True)    

    def run_split(self):
        """
        Binary-Recursive Mode: Executes sub-problems in parallel where possible.
        Each depth of the recursion tree is dispatched to the process pool.
        Tracking should be Process-Safe and Unified.
        Matches the recursive sub-problem architecture: folder 'Depth.Index' (as tracked in 'history').
        """
        self.flow_logger.write("# Starting Parallelized Split Mode (Binary Recursive Search)", level=1)
        
        # Initialize Unified FlowManager (Main Process Only)
        flow_mgr = FlowManager(
            iter_num=0,
            cutoff_cfg=self.cfg.cutoff, 
            ignore_nesting=self.cfg.ignore_nesting, 
            history=self.cfg.history, 
            history_file=self.cfg.history_file, 
            output_dir=self.cfg.output_dir,
            seed = self.cfg.seed
        )

        # 1. Initialize Task Queue
        # Start with the root problem
        root_task = (self.cfg.species_tree_path, self.cfg.gene_tree_path, "0")
        current_tasks = [root_task]
        
        # 2. Fast-Forward (Resume) Logic
        # If we have history, we might have completed the root or others.
        # We need to reconstruct the frontier.
        if self.cfg.history:
            self.flow_logger.write(f"# History found ({len(self.cfg.history)} entries). Checking for resume...", level=1)
            current_tasks = flow_mgr.fast_forward_split(current_tasks, self.flow_logger)

        pool = mp.Pool(processes=self.cfg.num_processes)
        depth = 0
        
        # Adjust depth display if resuming deep
        if current_tasks and "." in str(current_tasks[0][2]):
             depth = int(str(current_tasks[0][2]).split('.')[0])

        while current_tasks:
            self.flow_logger.write(f"# Dispatching {len(current_tasks)} sub-problems at Depth {depth}...", level=1)
            step = f"Processing Depth {depth}"
            self.flow_logger.report_step(step, "", start=True)

            # Dispatch workers (Workers do not have access to flow_mgr)
            worker_func = partial(run_worker, config=self.cfg, test_func=_is_memory_st)
            batch_results = pool.map(worker_func, current_tasks)
            self.flow_logger.report_step(step, f"Success: Extracting new sub-problems.")
            
            # Process results sequentially in the main process (safe access to flow_mgr)
            next_tasks = []
            for bin_id, iter_out, res in batch_results:
                if not res: continue
                # Logic to determine branching vs termination moved to flow_mgr
                next_tasks.extend(
                    flow_mgr.handle_split_result(bin_id, res, iter_out, logger=self.flow_logger, debug=self.cfg.debug)
                )

            # Move to the next depth of the recursion
            current_tasks = next_tasks
            depth += 1

        pool.close()
        pool.join()

        flow_mgr.glue_split_results(self.cfg.output_dir, self.cfg.species_tree_path, self.flow_logger)

        if self.cfg.plot:
            flow_mgr.plot(self.cfg.output_dir)

        self.flow_logger.write("# Parallelized Split Mode Finished.")

def main():
    config = parse_args()
    engine = Engine(config)
    engine.run()

if __name__ == "__main__":
    main()