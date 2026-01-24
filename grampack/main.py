import os
import re
import sys
import random
import multiprocessing as mp
from functools import partial
from pathlib import Path
from dataclasses import replace
from typing import Optional, Tuple, Dict, Any, List, Union

from .config import InitParser, GlobalContext, TaskConfig, GrandmaWriter
from .logger import GrandmaLogger
from .models import SmrtTree, NameRegistry, StepResult
from .ops import TreeLoader, GeneTreeManager, MulTreeManager
from .flow import FlowManager
from .reconcile import Reconciler
from .orthology import OrthologyLabeler

import psutil
HAS_PSUTIL = psutil is not None

# --- Top-level Helper Functions --- #
# --- Standalone workers function supporting parallel processing --- #

def task_worker(payload: Tuple[Any, Any, str], context: GlobalContext, config: TaskConfig, verbosity=0, parent_logger=None):
    """
    Unified worker for both Iterative (Full) and Recursive (Split) modes.
    Returns: (task_id, result, updates, log_file)
    """
        
    st, gts, task_id = payload

    # BETA: TEMP FIX: If st is a GrandmaTree object (passed from split mode), refresh it
    if hasattr(st, 'refresh'):
        st.refresh()

    # Ensure ID-specific output directory
    out = context.root_dir / task_id / "output"
    out.mkdir(parents=True, exist_ok=True)
    # Note: pickle_dir is a property of TaskConfig derived from output_dir, 
    # so we don't set it via replace.

    # Create local TaskConfig for this step
    iter_tcf = config.update(
        output_dir=out,
        st=st,
        gts=gts
    )

    # Create local GlobalContext (e.g., to override verbosity for workers)
    iter_ctx = replace(context, verbosity=verbosity)

    # If no parent logger provided (preferred in multiprocessing)
    # create logger without forwarding to parent
    logger = GrandmaLogger(iter_tcf.log_file, verbosity, parent_logger=parent_logger)

    # Execute Task
    # Task.execute returns (StepResult, updates)
    # 'updates' contains ONLY the hydrated persistent config data (parsed ploidies, etc.)
    task = Task(iter_ctx, logger=logger)
    res, updates = task.execute(iter_tcf)

    return task_id, res, updates, logger.log_path

# --- Core Classes --- #

class Task:
    """
    Represents a discrete execution unit of GRAMPA analysis.
    It contains its own logger, writer, and configuration state specific to this execution.
    It does not know about iteration history or other runs.
    """
    def __init__(self, context: GlobalContext, logger: GrandmaLogger = None):
        self.ctx = context
        self.logger = logger
        
        # State containers
        self.spec_tree = None
        self.gene_trees = {}
        self.mul_trees = {}
        self.registry = NameRegistry()
        
        # Components
        self.reconciler = None
        self.mul_mgr = None
        self.gene_mgr = None
        self.writer = None # Initialized in execute when we have a TaskSpec
             
    def execute(self, tcf: TaskConfig) -> Tuple[Optional[StepResult], Dict[str, Any]]:

        # 0. Setup for this task & log banner
        Path(tcf.output_dir).mkdir(parents=True, exist_ok=True)
        if not self.logger:
            # Create a new logger if one wasn't passed (standard behavior)
            log_file = Path(tcf.output_dir) / f"{tcf.run_prefix}.log"
            self.logger = GrandmaLogger(log_file, tcf.verbosity)
        self.writer = GrandmaWriter(tcf, self.logger)
        self.logger.start_run(self.ctx, tcf)
        self.logger.report_step("", "", start=True)

        # 1. Load Species Tree
        self.spec_tree = TreeLoader.spec_tree(tcf, self.logger)

        # (Re)Init components with current trees
        self.reconciler = Reconciler(tcf, self.ctx.num_processes)
        self.mul_mgr = MulTreeManager(tcf, self.spec_tree, self.logger)
        self.gene_mgr = GeneTreeManager(tcf, self.reconciler, self.logger)

        # 2. Build MUL-Trees
        self.mul_trees, h1_nodes, h2_nodes, ploidies = self.mul_mgr.build()
        if tcf.mode == "build-mts": return None, {}

        # 3. Load Gene Trees
        self.gene_trees = TreeLoader.gene_trees(tcf, self.logger)

        # DEBUG - print the string representation of all gene trees and the species tree and the first MUL tree
        """for mul_idx, md in self.mul_trees.items():
            print(f"MUL Tree {mul_idx}: {md.mt.to_string(internal_labels=True)}, \
                  H1: {md.h1_node}, H2: {md.h2_node}, Hybrid Clade: {md.h_clade}")
        for gene_num, gt_obj in self.gene_trees.items():
            ginfo = {}
            for n in gt_obj.ete_tree.traverse():
                # in a list, store: branch length, parent name, "tip"/"internal"/"root", support
                n_type = "tip" if n.is_leaf() else ("root" if n.is_root() else "internal")
                ginfo[n.name] = [n.dist, n.up.name if n.up else None, n_type, n.support]
            print(f"Gene Tree {gene_num}: {gt_obj.to_string(internal_labels=True)}, Node Info: {ginfo}")
        print(f"Species Tree: {self.spec_tree.to_string(internal_labels=True)})"""

        # 4. Collapse & Filter Groups
        # Note: In iterative modes, we are filtering the *memory* gene trees, 
        # which effectively filters them for subsequent iterations too.
        # TBD !!!
        self.gene_mgr.cull(self.mul_trees, self.gene_trees, self.registry)
        if tcf.mode == "check-nums": return None, {}

        # DEBUG
        """for gene_num, (gt_obj, x) in self.gene_trees.items():
            print(f"Gene Tree {gene_num}: {gt_obj.to_string(internal_labels=True)}, {x}")
        """

        # 5. Reconciliation and MUL-tree Selection
        step_result = self.reconciler.run(self.mul_trees, self.gene_trees, self.registry, self.logger, self.writer)

        """if not step_result.sorted_scores:
            self.logger.write("No valid MUL-trees scored.", level=1)
            return None"""

        # 6. Extract Best Result (using StepResult properties)
        min_idx = step_result.mt_idx()
        min_data = step_result.mul_trees[min_idx]
        
        # kept_mul_results is Dict[mul_idx, Dict[g_idx, Maps]]
        min_maps = step_result.kept_mul_maps[min_idx]
        
        # 7. Orthology (Optional)
        if self.ctx.orth_opt and min_maps:
             # OrthologyLabeler expects dict of results. We pass min_maps directly.
             # NOTE: OrthologyLabeler might need adjustment if it expects a list vs dict, 
             # but standard dict iteration works for both.
             # We pass keys as gene_num, values as ReconResult
             # OrthologyLabeler.run signature: gene_trees, min_maps_dict
             # But ReconResult wraps maps. Orthology.py uses res[3] (maps) and res[4] (dups)
             # or attributes if updated. Assuming strict compat, we pass the wrapper.
             OrthologyLabeler.run(self.gene_trees, min_maps, min_data.mt, 
                                min_data.h_clade, tcf.output_dir, tcf.run_prefix)

        # 8. Final Report
        min_tree_str = min_data.mt.to_string(internal_labels=True)
        h_clade = min_data.h_clade
        for spec in h_clade:
            min_tree_str = re.sub(f"{spec}(?!\*)", f"{spec}+", min_tree_str)
            min_tree_str = min_tree_str.replace("+*", "*")

        # Important:
        # ST and GTs are not needed here, as they are prepared from StepResult in FlowManager
        tcf_updates = {
            'ploidies': ploidies,
            #'h1_nodes': h1_nodes, to be supported later
            #'h2_nodes': h2_nodes, to be supported later
            'repair': False # Disable repair for subsequent runs
        }
            
        self.logger.print_end_prog(tcf, (min_idx, step_result.mt_score(), min_tree_str))

        return step_result, tcf_updates
        
class Engine:
    """
    Orchestrates the analysis flow based on the selected mode.
    Manages global state and sub-runs.
    """
    def __init__(self, context: GlobalContext, config: TaskConfig):
        self.ctx = context
        self.tcf = config
        
        # In Single mode, we don't need a separate engine logger.
        # In Full mode, we might want a flow logger.
        self.flow_logger = GrandmaLogger(self.ctx.log_file, self.ctx.verbosity, clear_log=False)
        """# In resume, append to log, don't clear
        is_resume = (self.tcf.history is not None and len(self.tcf.history) > 0)
        self.flow_logger = GrandmaLogger(log_path, self.cfg.verbosity, clear_log=not is_resume)"""

        # Flow Manager for Iterative Modes, lazy init
        self.flow_mgr = None

        # Seed Initialization
        if 'numpy' in sys.modules:
            import numpy; numpy.random.seed(self.ctx.seed)
        else:
            random.seed(self.ctx.seed)

    def run(self):
        final_res = None
        if self.tcf.mode in ["full", "split"]:
            # Flow Manager Init for Iterative Modes
            self.flow_mgr = FlowManager(self.ctx)
            if self.tcf.mode == "full":
                final_res = self.run_full()
            else:
                final_res = self.run_split()
        else:
            # Other modes
            final_res = Task(self.ctx, self.flow_logger).execute(self.tcf)
        return final_res
        
    def run_full(self):
        """
        Iterative mode. 
        Creates a dedicated folder and Run instance for each iteration.
        Passes tree objects in memory to avoid reloading.
        """
        self.flow_logger.write("# Starting Fully Sequential Mode", level=1)

        # Setup: initial parameters must have been parsed already in io.py
        i = self.ctx.start_pt
        max_iter = self.ctx.max_iter
        
        perm_tcf = self.tcf
        current_st = perm_tcf.st
        current_gts = perm_tcf.gts

        # If resuming, we rely on Run() loading from files via cfg paths in the first iteration loop,
        # OR we load them here. Actually, io.py sets cfg.species_tree_path to the previous output 
        # if i > 0. So current_st can remain None for the first loop.
       
        # Iteration Loop
        try:
            while i < max_iter:
                i += 1
                self.flow_logger.write(
                    f'#\n##### Iteration {i}' + (' (inf mode) #####\n#' if max_iter == float('inf') else f' of {int(max_iter)} #####\n#'))#∞

                # Run worker
                # We pass the persistent config (which might have parsed ploidies from iter 1)
                # worker will apply transient updates (output_dir, st, gts) internally
                _, res, updates, iter_log = task_worker(
                    payload=(current_st, current_gts, str(i-1)), # ID is 0-indexed usually
                    context=self.ctx,      # Pass Global Context
                    config=perm_tcf,       # Pass updated Task Config 
                    verbosity=self.ctx.verbosity,
                    parent_logger=self.flow_logger
                )
                iter_logger = GrandmaLogger(iter_log, self.ctx.verbosity, parent_logger=self.flow_logger, clear_log=False)

                if not res:
                    self.flow_logger.write("# No results generated in iteration.")
                    break

                # Update persistent config for next iteration
                perm_tcf = perm_tcf.update(**updates)

                # Process result and handle potential nesting
                # This returns the trees prepared for the NEXT iteration
                next_trees = self.flow_mgr.handle_iteration_result(
                    i, res, perm_tcf.output_dir, 
                    engine_callback=lambda st, gts, h1, h2, out: self._run_nested_subproblem(st, gts, h1, h2, out),
                    iter_logger=iter_logger,
                )

                if not next_trees: break
                current_st, current_gts = next_trees

        except KeyboardInterrupt:
            self.flow_logger.write("# Interrupted by user.", level=1)

        if self.cfg.plot:
            self.flow_mgr.plot(self.ctx.root_dir)
        
        self.flow_logger.write("# Fully Sequential Mode Finished.")

    def _run_nested_subproblem(self, st_obj, gt_objs, h1_str, h2_str, fix_dir):
        """Helper to run a constrained nested fix run."""
        # Use self.tcf as base, explicitly overriding params
        # Note: verbosity is passed in a quiet context to the Task.

        mem_st = SmrtTree(tree_obj=st_obj)
        mem_gts = {k: SmrtTree(tree_obj=v) for k, v in enumerate(gt_objs)}
        
        fix_tcf = replace(self.tcf,
                        st = mem_st,
                        gts = mem_gts,
                        output_dir=fix_dir,
                        h1_nodes=h1_str, 
                        h2_nodes=h2_str,
                        mode="no-st")
        
        # Create a quiet context
        fix_ctx = replace(self.ctx, verbosity=0)
        
        return Task(fix_ctx, logger=None).execute(fix_tcf)
    
    def run_split(self):
        """
        Binary-Recursive Mode: Executes sub-problems in parallel where possible.
        Each depth of the recursion tree is dispatched to the process pool.
        Tracking should be Process-Safe and Unified.
        Matches the recursive sub-problem architecture: folder 'Depth.Index' (as tracked in 'history').
        """
        self.flow_logger.write("# Starting Parallelized Split Mode (Binary Recursive Search)", level=1)
        
        # Initialize Task Queue to the root problem
        perm_tcf = self.tcf
        root_task = (perm_tcf.st, perm_tcf.gts, "0")
        current_tasks = [root_task]
        
        # Fast-Forward (Resume) Logic
        # If we have history, we might have completed the root or others.
        # We need to reconstruct the frontier.
        if self.ctx.history:
            self.flow_logger.write(f"# History found ({len(self.ctx.history)} entries). Checking for resume...", level=1)
            current_tasks = self.flow_mgr.fast_forward_split(current_tasks)

        pool = mp.Pool(processes=self.ctx.num_processes)
        depth = 0
        
        # Adjust depth display if resuming deep
        if current_tasks and "." in str(current_tasks[0][2]):
            depth = int(str(current_tasks[0][2]).split('.')[0])

        while current_tasks:

            #self.flow_logger.write(f"# Dispatching {len(current_tasks)} sub-problems at Depth {depth}...", level=1)
            #step = f"Processing Depth {depth}"
            #self.flow_logger.report_step(step, "", start=True)
            self.flow_logger.report_step(f"Depth {depth}", f"Dispatching {len(current_tasks)} tasks", start=True)

            # Dispatch workers (Workers do not have access to flow_mgr)
            worker_func = partial(task_worker, context=self.ctx, config=perm_tcf)
            # Map returns list of: (id, res, updates, logger)
            batch_results = pool.map(worker_func, current_tasks)

            self.flow_logger.report_step(f"Depth {depth}", f"Success: Extracting new sub-problems.")
            
            # Optimizing the Config (The "Split" Trick)
            # If this is the first run, the workers just parsed the ploidies/h-nodes.
            # We grab the updates from the *first valid result* and update perm_tcf.
            # The next depth's workers will receive the PARSED dicts, not file paths.
            for _, _, updates, _ in batch_results:
                if updates:
                    perm_tcf = perm_tcf.update(**updates)
                    break # Only need to do once

            # Process Results & Generate Next Tasks
            # Done sequentially in the main process (safe access to flow_mgr)
            next_tasks = []
            for task_id, res, _, iter_log in batch_results:
                if not res: continue

                # Logic to determine branching vs termination moved to flow_mgr
                # Note: We reconstruct path here to avoid passing it back from workers
                iter_logger = GrandmaLogger(iter_log, self.ctx.verbosity, parent_logger=self.flow_logger, clear_log=False)
                next_tasks.extend(
                    self.flow_mgr.handle_split_result(
                        task_id, res,
                        iter_out = self.ctx.root_dir / task_id / "output",
                        iter_logger = iter_logger
                    )
                )

            current_tasks = next_tasks
            depth += 1

        pool.close(); pool.join()

        self.flow_mgr.glue_split_results(self.ctx.root_dir, self.tcf.st, self.flow_logger)

        if self.ctx.plot:
            self.flow_mgr.plot(self.ctx.root_dir)

        self.flow_logger.write("# Parallelized Split Mode Finished.")

def main():
    ctx, tcf = InitParser().parse()
    engine = Engine(ctx, tcf)
    engine.run()

if __name__ == "__main__":
    main()