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

    path_manual = Path(__file__).parent / "ex_k_back"
    g = path_manual / 'backbone.txt'
    s = path_manual / 'astral.tre'

    #path_manual = Path(__file__).parent / "ex_no_w"
    #g = path_manual / 'manual_gene_trees_no_w.txt'
    #s = path_manual / 'manual_species_tree_no_w.tre'

    other_args = ''

    #old_tool = "C:\\Users\\psfsitaymay\\Downloads\\Grampa_RT_etc\\grandma\\grampa.py"
    old_tool = "E:\\Synced\\Studies\\Grampa_RT_etc\\grandma\\grampa.py"

    new_tool = path_manual.parent.parent / "gran.py"

    o_old = path_manual / 'o_old/'
    o_new = path_manual / 'o_new'

    time_tool(f"python {new_tool} -g {g} -s {s} -o {o_new}/ {other_args}", "new") # --debug --plot
    #time_tool(f"python {new_tool} -g {g} -s {s} -o {o_new}_splt/ {other_args} -m split --debug --plot", "new")
    #time_tool(f"python {new_tool} -g {g} -s {s} -o {o_new}_deep/ {other_args} -m full", "new")
    
    time_tool(f"python {old_tool} -g {g} -s {s} -o {o_old} --overwrite {other_args} --maps", "old")

    time_tool(f"python {Path(__file__).parent / 'exe_compare.py'} -log {path_manual / 'compare_log.txt'} -old {o_old} -new {o_new}", "compare")