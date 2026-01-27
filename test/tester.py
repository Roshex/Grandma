# tester code to time 2 .py tools when call like: python tool1.py -g -s -o and other args

import time
import subprocess
import sys
from pathlib import Path

def time_tool(command: str, tool_name: str):
    start_time = time.time()
    print(f"Running {tool_name} with command: {command}")
    result = subprocess.run(command, shell=True)
    end_time = time.time()
    
    if result.returncode != 0:
        print(f"\t!!! {tool_name} failed with return code {result.returncode}")
    else:
        elapsed = end_time - start_time
        print(f"\t!!! {tool_name} completed in {elapsed:.2f} seconds")

if __name__ == "__main__":

    datasets = [
#        ("ex_no_w", "manual_gene_trees_no_w.txt", "manual_species_tree_no_w.tre"),
#        ("ex_w", "manual_gene_trees_w.txt", "manual_species_tree_w.tre"),
#        ("ex_w_snested", "manual_gene_trees_w_sn.txt", "manual_species_tree_w_sn.tre"),
        ("ex_k_back", "backbone.txt", "astral.tre"),
    ]

    for ds_name, g_file, s_file in datasets:

        path_manual = Path(__file__).parent / ds_name
        g = path_manual / g_file
        s = path_manual / s_file

        other_args = ''

        #old_tool = "C:\\Users\\psfsitaymay\\Downloads\\Grampa_RT_etc\\grandma\\grampa.py"
        old_tool = "E:\\Synced\\Studies\\Grampa_RT_etc\\grandma\\grampa.py"

        new_tool = path_manual.parent.parent / "gran.py"

        o_old = path_manual / 'o_old/'
        o_new = path_manual / 'o_new'

        #time_tool(f"python {new_tool} -g {g} -s {s} -o {o_new}/ {other_args} --debug --plot -v 3", "new") # --debug --plot
        time_tool(f"python {new_tool} -g {g} -s {s} -o {o_new}_splt/ {other_args} -m split --debug --plot -v 3 -p 5 -i 2", "new") #--plot --start 1
        #time_tool(f"python {new_tool} -g {g} -s {s} -o {o_new}_deep/ {other_args} -m full --debug --plot -v 3", "new")

        #time_tool(f"python {old_tool} -g {g} -s {s} -o {o_old}_v0 --overwrite {other_args} --maps -v 0", "old")
        #time_tool(f"python {old_tool} -g {g} -s {s} -o {o_old}_v1 --overwrite {other_args} --maps -v 1", "old")
        #time_tool(f"python {old_tool} -g {g} -s {s} -o {o_old}_v2 --overwrite {other_args} --maps -v 2", "old")
        #time_tool(f"python {old_tool} -g {g} -s {s} -o {o_old}_v3 --overwrite {other_args} -v 3", "old")

        #time_tool(f"python {Path(__file__).parent / 'exe_compare.py'} -log {path_manual / 'compare_log.txt'} -old {o_old} -new {o_new}", "compare")




    '''
    # Initial File Selection
    st_file = Path(args.spec_tree).resolve()
    gt_file = Path(args.gene_tree).resolve() if args.gene_tree else None

    # For Full mode only: if resuming, point to previous iteration's output
    if mode == "full" and i > 0:
        prev_st = out_dir / str(i-1) / 'multree.tre'
        prev_gt = out_dir / str(i-1) / 'genetrees.txt'
        if prev_st.exists():
            st_file = prev_st
            gt_file = prev_gt



        # Sanitize inputs to SmrtTree objects
   
    # 3. Determine Initial Inputs for TaskSpec
    st_file = Path(args.spec_tree).resolve()
    gt_file = Path(args.gene_tree).resolve() if args.gene_tree else None
        
    '''

    '''# Print setup info
    print(f'\nSetup:')
    print(f'Iterations: {max_iter} (provided start-point {start})')
    print(f'Cutoff: {mp_cutoff}')
    print(f'Handle Nested Hybridizations: {not ignore_nesting}')
    print(f'Preprocessing Config: {prep_config if prep_config else "None"}')
    print(f'Output Directory: {base_output_dir}')
    print(f'Plotting Enabled: {args.plot}')
    print(f'Debug Mode: {debug}')
    print(f'Other Args: {unknown_args}')


        # Determine initial file paths based on iteration/resume point
    if i == 0:
        st_file = Path(args.spec_tree).resolve()
        gt_file = Path(args.gene_tree).resolve() if args.gene_tree else ""
    else:
        # Load the processed files from the PREVIOUS successful iteration
        st_file = out_dir / str(i-1) / 'multree.tre'
        gt_file = out_dir / str(i-1) / 'genetrees.txt'



    @staticmethod
    def _schedule_batches(tasks: list, total_cores: int) -> List[List[Tuple[Any, int]]]:
        """
        Implements the "Normalized Smallest-First" heuristic.
        Returns a list of batches. 
        Each batch is a list of tuples: (TaskPayload, NumCores).
        """
        if not tasks: return []

        # 1. Calculate Weights
        def get_weight(t):
            if hasattr(t[0], 'ete_tree'): return len(t[0].ete_tree.get_leaves())
            return 99999 # Root/Path tasks act as massive weights
        
        # Store tuples of (OriginalTask, Weight)
        weighted_tasks = [(t, get_weight(t)) for t in tasks]
        
        # 2. Sort Smallest to Largest (Crucial for this heuristic)
        weighted_tasks.sort(key=lambda x: x[1])
        
        weights = [x[1] for x in weighted_tasks]
        min_w = weights[0] if weights else 1
        if min_w == 0: min_w = 1 # Safety

        # 3. Normalize
        norm_weights = [w / min_w for w in weights]

        batches = []
        current_batch_tasks = []    # List of (Task, Weight)
        current_norm_sum = 0

        # 4. Grouping (Bin Packing based on Normalized Weights)
        for i, norm_w in enumerate(norm_weights):
            
            # If adding this task exceeds core count...
            # (And the batch isn't empty - if it is empty, we must add the huge task anyway)
            if (current_norm_sum + norm_w > total_cores) and current_batch_tasks:
                # Close current batch
                batches.append(current_batch_tasks)
                # Start new
                current_batch_tasks = []
                current_norm_sum = 0
            
            current_batch_tasks.append(weighted_tasks[i])
            current_norm_sum += norm_w

        if current_batch_tasks:
            batches.append(current_batch_tasks)

        # 5. Allocate Cores within Batches
        final_schedule = []

        for batch in batches:
            # batch is list of (Task, Weight)
            batch_orig_weights = [x[1] for x in batch]
            batch_total_w = sum(batch_orig_weights)
            
            # Calculate raw shares
            if batch_total_w == 0:
                # All 0 weights? Even split.
                raw_shares = [total_cores / len(batch)] * len(batch)
            else:
                raw_shares = [(w / batch_total_w) * total_cores for w in batch_orig_weights]
            
            # Rounding logic: max(1, round)
            core_counts = [max(1, int(round(s))) for s in raw_shares]
            
            # Fix Rounding Errors (Sum must equal Total Cores)
            current_sum = sum(core_counts)
            diff = total_cores - current_sum
            
            if diff != 0:
                # Apply difference to the LARGEST task in the batch
                # (The big task absorbs the floating point variance)
                max_idx = batch_orig_weights.index(max(batch_orig_weights))
                
                # Ensure we don't reduce a task to < 1 core
                if diff < 0 and core_counts[max_idx] + diff < 1:
                    # Edge case: fallback to simply adding/subtracting from end
                    for i in range(abs(diff)):
                        idx = i % len(batch)
                        if diff > 0: core_counts[idx] += 1
                        else: core_counts[idx] = max(1, core_counts[idx] - 1)
                else:
                    core_counts[max_idx] += diff

            # Pack into result structure
            batch_result = []
            for i, (task, _) in enumerate(batch):
                batch_result.append((task, core_counts[i]))
            
            final_schedule.append(batch_result)

        return final_schedule

            """# --- SETUP PHASE ---
            # Sort Tasks: Heaviest First
            def get_complexity(t):
                if hasattr(t[0], 'ete_tree'): return len(t[0].ete_tree.get_leaves())
                return 99999
            
            current_tasks.sort(key=get_complexity, reverse=True)
            self.flow_logger.report_step(f"Depth {depth}", f"Processing {len(current_tasks)} tasks", start=True)

            # --- PHASE 1: EXECUTION (The Queue Loop) ---

            # --- 1. CALL THE HEURISTIC ---
            # Returns: [[(TaskA, 3), (TaskB, 5)], [(TaskC, 8)]]
            batches = self._schedule_batches(current_tasks, total_procs)

            # --- 2. EXECUTE BATCHES ---
            depth_raw_results = []
            
            for b_idx, batch in enumerate(batches):
                
                # Logging
                cores_map = [x[1] for x in batch]
                self.flow_logger.log(f"  Batch {b_idx+1}/{len(batches)}: Running {len(batch)} tasks. Cores: {cores_map}", 'i')

                # Prepare Arguments
                batch_args = []
                for payload, p_count in batch:
                    # Update Configs per Task
                    w_ctx = self.ctx.update(num_processes=p_count)
                    batch_args.append((payload, w_ctx, perm_tcf, self.ctx.verbosity))

                # EXECUTE BATCH
                # Dispatch workers (Workers do not have access to flow_mgr)
                # Use NoDaemonPool to allow the workers to spawn their own internal pools
                # if inner_pool_size > 1.
                with NoDaemonPool(processes=len(batch)) as pool:
                    # starmap blocks until all tasks in this batch are done
                    batch_results = pool.starmap(task_worker, batch_args)"""


    '''


#    def __post_init__(self):
#        """Debug logging if enabled."""
#        if self.debug:
#            # We initialize a temporary logger for debug output
#            self.output_dir.mkdir(parents=True, exist_ok=True)
#            logger = GrandmaLogger(log_path=self.log_path, verbosity=self.verbosity, clear_log=False)
#            '''logger.write("GRANDMA Configuration Initialized:", level=0)
#            for field_name in self.__slots__:
#                logger.write(f"  {field_name}: {getattr(self, field_name)}", level=0)'''

