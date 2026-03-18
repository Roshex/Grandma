import os

# --- CLUSTER ENVIRONMENT SAFETY ---
# For imports of C-level libraries such as numpy to run in single-threaded mode,
# allowing Grandma's Process Pool to manage the parallelism without contention.
# May not be the best idea to set this globally, but TBD.
# Must appear before any imports of such libraries.
"""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
"""

import sys
import random
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Union, Set

from .config import InitParser, GlobalContext, TaskConfig
from .flow import FlowManager
from .logger import GranLogger, LogInheritance
from .models import SmrtTree, MulTree, NameRegistry, TaskResult, HistoryType, ConcurrTask
from .ops import TreeLoader, GeneTreeManager, MulTreeManager
from .orthology import OrthologyLabeler
from .reconcile import Reconciler

import psutil
HAS_PSUTIL = psutil is not None

import multiprocessing.pool

# --- Top-level Helper Functions --- #

# --- Helper Classes for Nested Parallelism ---
class NoDaemonProcess(multiprocessing.Process):
    """A Process that can spawn children."""
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

class NoDaemonPool(multiprocessing.pool.Pool):
    """A Pool that creates NoDaemonProcesses, allowing nested pools."""
    def Process(self, *args, **kwds):
        proc = super(NoDaemonPool, self).Process(*args, **kwds)
        proc.__class__ = NoDaemonProcess
        return proc

# --- Standalone workers function supporting parallel processing --- #

def task_worker(
        payload: ConcurrTask, context: GlobalContext, config: TaskConfig,
        verbosity: int = 0, label: str = '', parent_logger: Optional[GranLogger] = None
    ) -> Tuple[str, Optional[TaskResult], Dict[str, Any], LogInheritance]:
    """
    Unified worker for both Sequential (Full) and Binary (Split) modes.
    Returns: (task_id, result, updates, logger_inheritance)
    """
    st, gts, task_id = payload
    depth, idx = task_id
    task_str = f"{depth}.{idx}" if idx is not None else f"{depth}"

    # Ensure ID-specific output directory
    out = context.root_dir / task_str / "output"
    out.mkdir(parents=True, exist_ok=True)
    # Note: pickle_dir is a property of TaskConfig derived from output_dir, 
    # so we don't set it via replace.

    # Temp fix
    if config.mode == "split":
        binary_id = idx - context.mixed_switch
    else:
        binary_id = None

    # Create local TaskConfig for this step
    iter_tcf = config.update(output_dir=out, st=st, gts=gts, binary_id=binary_id)

    # Create local GlobalContext (e.g., to override verbosity for workers)
    iter_ctx = context.update(verbosity=verbosity)

    # If no parent logger provided (preferred in multiprocessing)
    # create logger without forwarding to parent
    iter_logger = GranLogger(iter_tcf.log_file, verbosity, context.debug, parent_logger=parent_logger, no_log=context.nolog, label=label)

    # BETA: TEMP FIX: If st is a GrandmaTree object (passed from split mode), refresh it
    if hasattr(st, 'refresh'):
        st.refresh()

    # Execute Task
    # Task.execute returns (StepResult, updates)
    # 'updates' contains ONLY the hydrated persistent config data (parsed ploidies, etc.)
    iter_logger.start_info(iter_ctx, iter_tcf)
    task = Task(iter_ctx, logger=iter_logger)
    res, updates = task.execute(iter_tcf)

    return task_id, res, updates, iter_logger.inheritance

# --- Core Classes --- #

class Task:
    """
    Represents a discrete execution unit of GRAMPA analysis.
    It contains its own logger and configuration state specific to this execution.
    It does not know about iteration history or other runs.
    """
    def __init__(self, context: GlobalContext, logger: GranLogger = None):
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
             
    def execute(self, tcf: TaskConfig) -> Tuple[Optional[TaskResult], Dict[str, Any]]:

        # Setup task
        Path(tcf.output_dir).mkdir(parents=True, exist_ok=True)
        if not self.logger:
            # Create a new logger if one wasn't passed
            log_file = Path(tcf.output_dir) / f"{tcf.run_prefix}.log"
            self.logger = GranLogger(log_file, self.ctx.verbosity, self.ctx.debug, no_log=self.ctx.nolog)

        if self.ctx.norun:
            return self.logger.norun_banner(), {}

        self.logger.report_step("", "", start=True, enable_benchmark=self.ctx.bench)

        # Load Species Tree
        self.spec_tree = TreeLoader.spec_tree(tcf, self.logger)

        # If the ST is None, terminal label-sp mode successfully finished or a warning was logged. Exit smoothly.
        if not self.spec_tree:
            return self.logger.end_report(), {}
        
        # Re-init component with current task
        self.mul_mgr = MulTreeManager(tcf, self.spec_tree, self.logger)

        # Build MUL-Trees
        self.mul_trees, h1_nodes, h2_nodes, ploidies = self.mul_mgr.build(self.ctx.nesting, self.ctx.strict_max, self.ctx.allow_redun)

        # If build() returns an empty dict, a terminal mode (count-mts or build-mts) successfully finished or a warning was logged. Exit smoothly.
        if not self.mul_trees:
            return self.logger.end_report(), {}

        # Load Gene Trees
        self.gene_trees = TreeLoader.gene_trees(tcf, self.logger)

        # Re-init component with current task
        self.reconciler = Reconciler(tcf, self.logger, self.ctx.num_processes, self.ctx.pickles, self.ctx.maps, self.ctx.optim)
        self.gene_mgr = GeneTreeManager(tcf, self.logger, self.ctx.num_processes, self.ctx.pickles)

        # Collapse & Filter Groups
        # Note: In iterative modes, we are filtering the *memory* gene trees, 
        # which effectively filters them for subsequent iterations too.
        # TBD ?
        proceed = self.gene_mgr.cull(self.mul_trees, self.gene_trees, self.registry)

        # In check-nums mode, cull returns False after logging the counts and handling pickles.
        if not proceed:
            return self.logger.end_report(), {}

        # Reconciliation and MUL-tree Selection
        step_result = self.reconciler.run(self.mul_trees, self.gene_trees, self.registry)

        """if not step_result.sorted_scores:
            self.logger.write("No valid MUL-trees scored.", level=1)
            return None"""

        # Extract Best Result from StepResult properties: kept_mul_maps is Dict[mul_idx, Dict[g_idx, Maps]]
        min_idx = step_result.mt_idx()
        min_data = step_result.mul_trees[min_idx]
        min_maps = step_result.kept_mul_maps[min_idx]
        
        # Orthology (TBD)
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

        # Final Cleanup and Prep for Next Iteration
        self.gene_mgr.handle_pickles()

        # Important:
        # ST and GTs are not needed here, as they are prepared from StepResult in FlowManager; do NOT pass them here!
        tcf_updates = {
            'ploidies': ploidies,
            #'h1_nodes': h1_nodes, to be supported later
            #'h2_nodes': h2_nodes, to be supported later
            'repair': False # Disable repair for subsequent runs
        }

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
        self.flow_logger = GranLogger(self.ctx.log_file, self.ctx.verbosity, self.ctx.debug, no_log=self.ctx.nolog, clear_log=False)
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

    def run(self) -> Dict[str, Union[SmrtTree, HistoryType, Any]]:

        run_mode = self.tcf.mode
        # Init FlowManager for any iterative mode, and single - to return history
        if run_mode in ("full", "split", "mixed", "single", "st-only", "no-st") and not self.ctx.norun:
            self.flow_mgr = FlowManager(self.ctx, run_mode, self.flow_logger)

        self.flow_logger.start_info(self.ctx, self.tcf)

        final_smtree = None
        if run_mode == "mixed":
            final_smtree = self.run_mixed()
        elif run_mode == "full":
            final_smtree, _ = self.run_full()
        elif run_mode == "split":
            final_smtree = self.run_split()
        else:
            final_smtree = self.run_noniter()

        if final_smtree:
            final_smtree.write_forms(self.ctx.root_dir)
            is_iter = run_mode in ("full", "split", "mixed")
            task_tree_ascii = self.flow_mgr.create_problem_tree_ascii() if is_iter else None
            # Visualize with reticulate tree's built-in function (requires matplotlib)
            rt = final_smtree.to_rt() if self.ctx.debug else None
            orig_tree = TreeLoader.spec_tree(self.tcf, GranLogger.dummy()).ete_tree
            self.flow_logger.final_report(final_smtree, orig_tree, rt, task_tree_ascii, is_iter, self.ctx.plot)

        return {
            'final_tree': final_smtree,
            'history': self.ctx.history,
            'maps': None # to be supported later
        }

    def run_noniter(self) -> Optional[SmrtTree]:
        res, _ = Task(self.ctx, self.flow_logger).execute(self.tcf)
        # Extract best SmrtTree if applicable
        if res:
            min_score, min_idx, min_mult = res.unpacked_min_mt
            if self.tcf.mode != "st-only":
                self.flow_mgr.judge_event(0, 0, res)
            # Terminal report for single-run modes (single, st-only, no-st)
            self.flow_logger.end_report(min_score, min_idx, min_mult.to_marked_str())
            return min_mult.mt
        return None

    def run_mixed(self) -> SmrtTree:
        """
        Executes Full mode until iteration N, then switches to Split mode.
        """
        self.flow_logger.log(f"Starting Mixed Mode: Full until iter {self.ctx.mixed_switch}, then Split.", 'i')
        
        full_limit = self.ctx.mixed_switch
        
        # Run Full Mode up to switch point
        self.flow_mgr.mode = "full"
        last_st, last_gts = self.run_full(limit_override=full_limit)
        
        if not last_gts:
            self.flow_logger.log("Full Mode terminated before reaching switch point. Split Mode will not be executed.", 'i')
            return last_st
        self.flow_logger.log(f"Switching to Split Mode at iteration {full_limit}.", 'i')
        
        # Prepare the root task for split mode. 
        # The ID is simply the iteration number (e.g., "5"), representing depth 5, index 0 effectively.
        # Split logic expects "Depth.Index": since we ran linear 0..4, the next depth is 5 (and index 0).
        root_id = (full_limit, 0)

        self.flow_mgr.mode = "split"
        return self.run_split(initial_payload=(last_st, last_gts, root_id))

    def _run_nested_subproblem(self, mem_st: SmrtTree, mem_gts: Dict[int, SmrtTree], h1_str: str, h2_str: str, fix_dir: Path) -> TaskResult:
        """Helper to run a constrained nested fix run."""
        # Use self.tcf as base, explicitly overriding params
        # Note: verbosity is passed in a quiet context to the Task.
        
        fix_tcf = self.tcf.update(
                        st = mem_st,
                        gts = mem_gts,
                        output_dir=fix_dir,
                        h1_nodes=h1_str, 
                        h2_nodes=h2_str,
                        mode="no-st")
        
        # Create a quiet context
        fix_ctx = self.ctx.update(verbosity=0)
        
        return Task(fix_ctx, logger=None).execute(fix_tcf)
        
    def autocorrect(self, targets: List[str], multree: MulTree, genetrees: Dict[int, SmrtTree], iter: int) -> Tuple[MulTree, Dict[int, SmrtTree]]:
        """
        Detects nested hybridization by finding 'orphaned' copies of the H-lineage
        directly in the tree structure using the .match() capability.
        Returns the corrected MulTree and Gene Trees after iteratively fixing each detected nested copy.
        """        
        curr_mt = multree
        curr_gts = genetrees 

        self.flow_logger.log(f"Nested Fix: Detected {len(targets)} missing copies to fix: {targets}", 'i')
        
        for t_id in range(len(targets)):
            # Start looking for Copy 2 (but index starts at 0, so start = 1)
            next_copy_idx = t_id + 1

            target = targets[t_id]
            h2_loc = curr_mt.mt.get_node(target)
            if h2_loc is None:
                self.flow_logger.log(f"Nested Fix: Target node '{target}' not found in current MT. Skipping.", 'w')
                continue

            # If nest_internal_only is True, skip if the target is the most external node in the event that produced it.
            '''if self.ctx.nesting == "strict_rectify" and self.flow_mgr._check_internality(curr_mt.mt, h2_loc):
                self.flow_logger.log(f"Nested Fix: Target node '{h2_loc.name}' is root in the smallest event containing it. Skipping due to strict_rectify mode.", 'd')
                continue'''

            # Trigger Nested Fix
            self.flow_logger.log(f"Nested Fix: Nested Event Detected! Locating missing copy at the branch leading to {h2_loc.name}", 'i')
            h1_node = curr_mt.h1_node
            
            fix_dir = self.ctx.root_dir / f'{iter}.{next_copy_idx}' / 'output'
            
            # Run Task to infer reconciliation for this missing copy
            # We treat the 'Missing Candidate' as the H2 (Target) 
            # and the current H1 as the source.
            step = f"Nested Fix Iteration {iter}.{next_copy_idx}"
            self.flow_logger.report_step(step, "In progress...")

            res, _ = self._run_nested_subproblem(
                curr_mt.mt, curr_gts,
                ",".join(h1_node.iter_leaf_names()), # H1 is the reference
                ",".join(h2_loc.iter_leaf_names()), # H2 is the new found copy
                fix_dir
            )

            self.flow_logger.report_step(step, "Success")

            curr_mt, curr_gts, targets = self.flow_mgr.relabel_problem(
                iter, res, fix_dir, self.flow_logger, j=next_copy_idx, targets=targets
            )

        self.flow_logger.log(f"Nested check complete. Found {next_copy_idx-1} extra copies.", 'i')
        return curr_mt, curr_gts

    def run_full(self, limit_override: int = None) -> Tuple[SmrtTree, Optional[Dict[int, SmrtTree]]]:
        """
        Iterative mode. Returns final (st, gts) if limit reached, or (None, None) if finished naturally.
        """
        self.flow_logger.title_banner("Starting Fully Sequential Mode")

        # Setup: initial parameters must have been parsed already in io.py
        i = self.ctx.start_pt
        max_iter = limit_override if limit_override is not None else self.ctx.max_iter
        if max_iter > self.ctx.max_iter:
            self.flow_logger.log(f"Provided Mixed Mode switch ({max_iter}) exceeds global max_iter ({self.ctx.max_iter}). Using override.", 'w')
        iter_msg = '(inf mode)' if max_iter == float('inf') else f'of {int(max_iter)}' # ∞
        iter_labeller = lambda it: f"Iteration {it} {iter_msg}"
        
        perm_tcf = self.tcf
        current_st = perm_tcf.st
        current_gts = perm_tcf.gts

        # If resuming, we rely on Run() loading from files via cfg paths in the first iteration loop,
        # OR we load them here. Actually, io.py sets cfg.species_tree_path to the previous output 
        # if i > 0. So current_st can remain None for the first loop.
       
        # Iteration Loop
        try:
            while True:
                
                if i >= max_iter:
                    if limit_override is not None:
                        self.flow_logger.log(f"Reached Mixed Mode switch point. Terminating with handoff.", 'i')
                        return current_st, current_gts
                    self.flow_logger.log(f"Reached maximum valid events set by user ({max_iter}). Terminating.", 'i')
                    break

                self.flow_logger.title_banner("")

                # Run worker
                # We pass the persistent config (which might have parsed ploidies from iter 1)
                # worker will apply transient updates (output_dir, st, gts) internally
                _, res, updates, log_inheritance = task_worker(
                    payload       = (current_st, current_gts, (i, None)),
                    context       = self.ctx,      # Pass Global Context
                    config        = perm_tcf,      # Pass updated Task Config 
                    verbosity     = self.ctx.verbosity,
                    label         = iter_labeller(i),
                    parent_logger = self.flow_logger
                )

                min_score, min_idx, min_mult = res.unpacked_min_mt
                min_mult_str = min_mult.to_marked_str()

                # Update persistent config for next iteration
                perm_tcf = perm_tcf.update(**updates)

                iter_logger = GranLogger(None, self.ctx.verbosity, self.ctx.debug, parent_logger=self.flow_logger, inheritance=log_inheritance)
                # Process result and handle potential nesting
                # This returns the trees prepared for the NEXT iteration
                next_mt, next_gts, targets = self.flow_mgr.relabel_problem(
                    i, res, iter_out = self.ctx.root_dir / str(i) / "output", 
                    iter_logger = iter_logger,
                )

                iter_logger.end_report(min_score, min_idx, min_mult_str)

                # Save for return even if breaking (e.g. if no events found, get ST)
                if not next_gts:
                    self.flow_logger.log(f"No further events found. Terminating at iteration {i}.", 'i')
                    current_st = next_mt.mt
                    break

                if targets:
                    next_mt, next_gts = self.autocorrect(targets, next_mt, next_gts, i)
        
                i += 1
                current_st = next_mt.mt
                current_gts = next_gts
                
        except KeyboardInterrupt:
            self.flow_logger.log("Interrupted by user.", 'i')

        if self.ctx.plot: self.flow_mgr.plot()
        
        self.flow_logger.title_banner("Fully Sequential Mode Finished")
        return current_st, None # Natural finish
    
    def run_split(self, initial_payload: Optional[ConcurrTask] = None) -> SmrtTree:
        """
        Binary Split Mode:
        Executes sub-problems in parallel where possible.
        Can accept an initial_payload (st, gts, id) to resume/mixed-start.
        Each depth of the recursion tree is dispatched to the process pool.
        Tracking should be Process-Safe and Unified.
        Matches the recursive sub-problem architecture: folder 'Depth.Index' (as tracked in 'history').
        Logic:
        1. Few Tasks (< num_procs): Run tasks in parallel, give each task MULTIPLE cores (Low-level).
        2. Many Tasks (> num_procs): Run many tasks in parallel, give each task ONE core (High-level).
        Even though both small and large tasks get about the same resources, in practice, for some
        reason this is faster than:
        1. running each sequentially with all cores, or
        2. scheduling batches based on task weights (e.g., number of species leaves).
        Possibly due to overhead of scheduling and load balancing.
        """
        self.flow_logger.title_banner("Starting Binary Split Mode")

        perm_tcf = self.tcf
        current_tasks = []
        
        # Initialize Tasks
        if initial_payload:
            # Mixed Mode Handoff
            current_tasks = [initial_payload]
            events_found = initial_payload[2][0] # Depth is index 0 of the tuple
            root_task_id = initial_payload[2]
        else:
            # Standard Start: initialize Task Queue to the root problem
            root_task_id = (0, 0)
            root_task = (perm_tcf.st, perm_tcf.gts, root_task_id)
            current_tasks = [root_task]
            # Fast-Forward (Resume) Logic
            # If we have history, we might have completed the root or others.
            # We need to reconstruct the frontier.
            events_found = 0
            if self.ctx.history:
                self.flow_logger.log(f"History found ({len(self.ctx.history)} entries). Checking for resume...", 'i')
                current_tasks = self.flow_mgr.fast_forward_split(current_tasks)
                events_found = 0 # to be implemented !!!

        # Adjust depth display if resuming deep
        depth = current_tasks[0][2][0] if current_tasks else 0

        max_iter = self.ctx.max_iter
        iter_labeller = lambda it: f"Branch {it[0]}.{it[1]}"

        while current_tasks:
    
            if events_found >= max_iter:
                self.flow_logger.log(f"Reached maximum valid events set by user ({events_found} actual >= {max_iter}). Terminating.", 'i')
                break

            num_tasks = len(current_tasks)
            if num_tasks == 0:
                break

            self.flow_logger.title_banner("")
            self.flow_logger.log(f"Dispatching {num_tasks} tasks at Depth {depth}", 'i')

            # --- DYNAMIC CALCULATION ---
            total_procs = self.ctx.num_processes
            # Don't exceed available cores for outer pool
            outer_pool_size = min(num_tasks, total_procs)
            # How many cores does each task get? (Distribute remainders)
            # e.g., 8 cores, 2 tasks -> inner = 4. 
            # e.g., 8 cores, 20 tasks -> inner = 1.
            inner_pool_size = max(1, total_procs // outer_pool_size)
            reminder = total_procs % outer_pool_size

            reminder_str = f" (+1 for {reminder} tasks)" if reminder > 0 else ""
            self.flow_logger.log(f"Resource Strategy: {outer_pool_size} Parallel Tasks x {inner_pool_size}{reminder_str} Cores-per-Task", 'i')

            # Configure the TaskConfig for the workers
            # Each worker sees 'num_processes' = inner_pool_size
            ctx_standard = self.ctx.update(num_processes=inner_pool_size)
            ctx_remainder = self.ctx.update(num_processes=inner_pool_size + 1)

            # Prepare Batch
            batch_args = []
            for i, payload in enumerate(current_tasks):
                # Distribute remainder cores to the first 'reminder' tasks
                if i < reminder:
                    ctx_to_use = ctx_remainder
                else:
                    ctx_to_use = ctx_standard
                batch_args.append((
                    payload,  
                    ctx_to_use,
                    perm_tcf, 
                    self.ctx.verbosity,
                    iter_labeller(payload[2])
                ))

            # EXECUTE BATCH
            # Dispatch workers (Workers do not have access to flow_mgr)
            # Use NoDaemonPool to allow the workers to spawn their own internal pools
            # if inner_pool_size > 1.
            try:
                with NoDaemonPool(processes=outer_pool_size) as pool:
                    # starmap blocks until all tasks in this batch are done
                    batch_results = pool.starmap(task_worker, batch_args)
            # Assimilate all worker logs from this batch to ensure the worker's traceback makes it into the main log
            except Exception:
                self.flow_logger.log("A worker process crashed: assimilating batch logs to retrieve traceback...", 'e', kill_on_error=False)
                for args in batch_args:
                    task_id = args[0][2]
                    task_str = f"{task_id[0]}.{task_id[1]}"
                    worker_log_path = self.ctx.root_dir / task_str / "output" / f"{perm_tcf.run_prefix}.log"
                    self.flow_logger.assimilate(worker_log_path)
                raise # Re-raise to trigger main process exit

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
            # Sort batch results by task_id to ensure deterministic logging
            batch_results.sort(key=lambda x: x[0]) # Sort by task_id
            for task_id, res, _, log_inheritance in batch_results:
                if not res: continue

                min_score, min_idx, min_mult = res.unpacked_min_mt
                min_mult_str = min_mult.to_marked_str()

                # Assimilate the worker's log into the main logger
                self.flow_logger.assimilate(log_inheritance.log_file, warnings=log_inheritance.warnings)
                iter_logger = GranLogger(None, self.ctx.verbosity, self.ctx.debug, parent_logger=self.flow_logger, inheritance=log_inheritance)
                # Logic to determine branching vs termination moved to flow_mgr
                # Note: We reconstruct path here to avoid passing it back from workers
                task_str = f"{task_id[0]}.{task_id[1]}"
                extracts = self.flow_mgr.extract_subproblems(
                        task_id, res,
                        iter_out = self.ctx.root_dir / task_str / "output",
                        iter_logger = iter_logger
                )

                iter_logger.end_report(min_score, min_idx, min_mult_str)


                if extracts is None:
                    continue # No events found, no new tasks
                else:
                    next_tasks.extend(extracts)
                    # Even if list is empty, we found a parsimonious event
                    events_found += 1

            # Sort new tasks by num of species tree leaves [largest first - more cores to bigger problems]
            # Done after the loop guarantees a SmrtTree object for sorting!
            next_tasks.sort(key=lambda x: len(x[0].ete_tree), reverse=True)
            # Debug: print task IDs and sizes
            self.flow_logger.log(f"Generated {len(next_tasks)} tasks for Depth {depth + 1}.", 'd')
            for t in next_tasks:
                self.flow_logger.log(f"  Task ID: {t[2]}, Number of Species Leaves: {len(t[0].ete_tree)}", 'd')

            current_tasks = next_tasks
            depth += 1

        final_tree = self.flow_mgr.glue_split_results(root_id=root_task_id)

        if self.ctx.plot: self.flow_mgr.plot()

        self.flow_logger.title_banner("Binary Split Mode Finished")
        return final_tree

def main(args_list: Optional[List[str]] = None, return_results: bool = False) -> Optional[dict]:
    """
    Main entry point for GRANDMA (CLI and API).
    
    :param args_list: List of string arguments. If None, uses sys.argv (CLI mode).
    :param return_results: If True, returns internal Python objects instead of exiting.
    :return: A dictionary of results if return_results is True, else None.
    """
    ctx, tcf = InitParser().parse(args_list)
    engine = Engine(ctx, tcf)
    final_results = engine.run()
    if return_results:
        # --- API PATH ---
        print("Returning results to Python API.")
        return final_results
    else:
        # --- CLI PATH ---
        sys.exit(0)

if __name__ == "__main__":
    main()