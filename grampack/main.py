import os
import re
import sys
from pathlib import Path
from dataclasses import replace

from .io import parse_args, GrandmaConfig, GrandmaWriter, GrandmaMetadata
from .logger import GrandmaLogger
from .tree_ops import GrandmaTree
from .gene_ops import TreeLoader, GeneTreeManager, MulTreeManager
from .flow import FlowManager
from .reconcile import Reconciler
from .orthology import OrthologyLabeler

import psutil
HAS_PSUTIL = psutil is not None

# --- Top-level Helper Functions for Pickling ---

def _is_memory_st(st):
    """Replacement for lambda: Check if st is a memory object rather than a path."""
    return not isinstance(st, (str, Path))

def _is_not_none(obj):
    """Replacement for lambda: Check if object is initialized."""
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

    # Ensure ID-specific sub-directory
    out = config.output_dir / id / "output"
    out.mkdir(parents=True, exist_ok=True)
    
    # Update pickle dir to be sub-directory specific
    pkl_dir = config.pickle_dir
    pkl_dir = pkl_dir.parent / id / 'output' / pkl_dir.name

    # Localized configuration
    iter_cfg = replace(config, output_dir=out, pickle_dir=pkl_dir, verbosity=verbosity)

    # If no logger provided (preferred in multiprocessing), create local one
    if not logger:
        log_path = out / f"{config.run_prefix}.log"
        logger = GrandmaLogger(log_path, verbosity)

    run_inst = Run(iter_cfg, spec_tree=st, gene_trees=gts, logger=logger)
    res = run_inst.execute(from_memory=test_func(st))

    # Return ID and out path for history/glueing
    return id, out, res

# --- Core Classes --- #

class Run:
    """
    Represents a discrete execution unit of GRAMPA analysis.
    It contains its own logger, writer, and configuration state specific to this execution.
    It does not know about iteration history or other runs.
    """
    def __init__(self, config: GrandmaConfig, logger: GrandmaLogger = None, 
                 spec_tree: GrandmaTree = None, gene_trees: dict = None):
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
        sorted_scores, detailed_res = self.reconciler.run(self.mul_trees, self.gene_trees,
                                                          self.cfg, self.logger, self.writer)

        # 6. Extract Best Result
        min_idx = sorted_scores[0][0]
        min_data = self.mul_trees[min_idx]
        min_maps = detailed_res[0][1] if detailed_res else {}
        
        # 7. Orthology (Optional)
        if self.cfg.orth_opt and detailed_res:
             OrthologyLabeler.run(self.gene_trees, min_maps, min_data[0], 
                                min_data[2], self.cfg.output_dir, self.cfg.run_prefix)

        # 8. Final Report
        min_score = sorted_scores[0][1]
        min_tree_str = min_data.mt.to_string(internal_labels=True)
        h_clade = min_data.h_clade
        for spec in h_clade:
            min_tree_str = re.sub(f"{spec}(?!\*)", f"{spec}+", min_tree_str)
            min_tree_str = min_tree_str.replace("+*", "*")
            
        self.logger.print_end_prog(self.cfg, (min_idx, min_score, min_tree_str))

        # Return structured result
        return {
            "min_idx": min_idx,
            "min_score": min_score,
            "mul_data": min_data,
            "min_maps": min_maps,
            "sorted_scores_dict": {k:v for k,v in sorted_scores},
            "gene_trees": self.gene_trees 
        }
        
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

    def run(self):

        meta = GrandmaMetadata()
        self._init_flow_logger()
        # Print software info once for the whole session
        self.flow_logger.log_software_banner(meta)

        self.flow_logger.write(f"# Running in mode: {self.cfg.mode}", level=1)
        self.flow_logger.write("# " + "=" * 73, level=1)
        if self.cfg.mode in ["no-st", "st-only", "build-mts", "check-nums", "single"]:
            self.run_single()
        elif self.cfg.mode == "full":
            self.run_full()
        elif self.cfg.mode == "split":
            self.run_split()
        else:
            self.flow_logger.write(f"# Error: Unknown mode {self.cfg.mode}", level=1)
            sys.exit(1)

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
            output_dir=base_output_dir
        )

        current_st = None # Holds GrandmaTree
        current_gts = None # Holds Dict[int, GrandmaTree]
       
        # Iteration Loop
        try:
            while i < self.cfg.max_iter:
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
                # Clean up memory/PIDs
                self.flow_logger.pids = [psutil.Process(os.getpid())] if HAS_PSUTIL else []

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

            mem_st = GrandmaTree(tree_obj=st_obj)
            mem_gts = {k: GrandmaTree(tree_obj=v) for k, v in enumerate(gt_objs)}
            
            run_inst = Run(fix_cfg, spec_tree=mem_st, gene_trees=mem_gts)
            return run_inst.execute(from_memory=True)    

    def run_split(self):
        """
        Binary-Recursive Mode: Executes sub-problems in parallel where possible.
        Each depth of the recursion tree is dispatched to the process pool.
        Tracking should be Process-Safe and Unified.
        Matches the recursive sub-problem architecture: folder 'Depth.Index' (as tracked in 'history').
        """
        import multiprocessing as mp
        from functools import partial

        self.flow_logger.write("# Starting Parallelized Split Mode (Binary Recursive Search)", level=1)
        
        # Initialize Unified FlowManager (Main Process Only)
        flow_mgr = FlowManager(
            iter_num=0,
            cutoff_cfg=self.cfg.cutoff, 
            ignore_nesting=self.cfg.ignore_nesting, 
            history=self.cfg.history, 
            history_file=self.cfg.history_file, 
            output_dir=self.cfg.output_dir
        )

        # Initial task: (SpeciesTreePath/Obj, GeneTreesPath/Dict, BinaryID)
        current_tasks = [(self.cfg.species_tree_path, self.cfg.gene_tree_path, "0")]

        # Initialize the Pool
        pool = mp.Pool(processes=self.cfg.num_processes)

        depth = 0
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