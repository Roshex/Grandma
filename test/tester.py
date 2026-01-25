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
        ("ex_no_w", "manual_gene_trees_no_w.txt", "manual_species_tree_no_w.tre"),
        ("ex_w", "manual_gene_trees_w.txt", "manual_species_tree_w.tre"),
        ("ex_w_snested", "manual_gene_trees_w_sn.txt", "manual_species_tree_w_sn.tre"),
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

        time_tool(f"python {new_tool} -g {g} -s {s} -o {o_new}/ {other_args} --debug", "new") # --debug --plot
        time_tool(f"python {new_tool} -g {g} -s {s} -o {o_new}_splt/ {other_args} -m split --debug", "new") #--plot --start 1
        time_tool(f"python {new_tool} -g {g} -s {s} -o {o_new}_deep/ {other_args} -m full --debug", "new")

        #time_tool(f"python {old_tool} -g {g} -s {s} -o {o_old} --overwrite {other_args} --maps", "old")

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

