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
#        ("ex_k_back", "backbone.txt", "astral.tre"),
#        ("ex_bend", "grampa_trees.tre", "species.tre"),
        ("ex_diaz", "grampa_trees.tre", "species.tre"),
#        ("ex_kall", "kall.nw", "kall.treefile"),

#        ("ding2023", "grampa_trees.tre", "grampa_species_tree.tre"),
#        ("dp2018", "genetrees.txt", "multree.tre"),
#        ("koe2020", "genetrees.txt", "multree.tre"),
#        ("ren2024", "genetrees.txt", "multree.tre"),
#        ("zhao2021", "grampa_trees.tre", "grampa_species_tree.tre"),

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

        #time_tool(f"python {new_tool} -g {g} -s {s} -o {o_new}/ {other_args} --debug --plot -v 3 -p 10", "new") # --debug --plot -w 1 50
        #time_tool(f"python {new_tool} -g {g} -s {s} -o {o_new}_nm/ {other_args} --debug --plot -v 3 -p 10 --nestedness model", "new") # --debug --plot
        #time_tool(f"python {new_tool} -g {g} -s {s} -o {o_new}_op/ {other_args} --debug --plot -v 3 -p 10 --optim", "new")
        #time_tool(f"python {new_tool} -g {g} -s {s} -o {o_new}_np/ {other_args} --debug --plot -v 3 -p 10 --optim --nestedness model", "new") # --debug --plot
        time_tool(f"python {new_tool} -g {g} -s {s} -o {o_new}_splt/ {other_args} -m split --debug --plot -v 3 -p 10 --start auto -x {path_manual/'ploidies.txt'}", "new") #--plot --start 1 -i 2
        #time_tool(f"python {new_tool} -g {g} -s {s} -o {o_new}_deep/ {other_args} -m full --debug --plot -v 3 -p 10 -x {path_manual/'ploidies.txt'}", "new")

        #time_tool(f"python {old_tool} -g {g} -s {s} -o {o_old}_v0 --overwrite {other_args} --maps -v 0", "old")
        #time_tool(f"python {old_tool} -g {g} -s {s} -o {o_old}_v1 --overwrite {other_args} --maps -v 1", "old")
        #time_tool(f"python {old_tool} -g {g} -s {s} -o {o_old}_v2 --overwrite {other_args} --maps -v 2", "old")
        #time_tool(f"python {old_tool} -g {g} -s {s} -o {o_old} --overwrite {other_args} -v 3 -p 10", "old")#_v3

        #time_tool(f"python {Path(__file__).parent / 'exe_compare.py'} -log {path_manual / 'compare_log.txt'} -old {o_old} -new {o_new}", "compare")


