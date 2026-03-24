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

    datasets = {
        'manual': [
#            ("none", "genetrees.txt", "spectree.tre"),
#            ("no_w", "genetrees.txt", "spectree.tre"),
#            ("w", "genetrees.txt", "spectree.tre"),
##             ("w_ils", "genetrees.txt", "spectree.tre"),
##             ("diaz", "grampa_trees.tre", "species.tre"),
##             ("k_back", "backbone.txt", "astral.tre"),
##             ("bend", "grampa_trees.tre", "species.tre"),
##             ("kall", "kall.nw", "kall.treefile"),
#            ("w_sn", "genetrees.txt", "spectree.tre"),
#            ("w_sn_H", "genetrees.txt", "spectree.tre"),
        ],
        'generator': [
             ("ex", "to_generate.json", "spectree.tre"),
        ],

        'empirical': [
#        ("ex_k_back", "backbone.txt", "astral.tre"),
#        ("ex_bend", "grampa_trees.tre", "species.tre"),
#        ("ex_diaz", "grampa_trees.tre", "species.tre"),
#        ("ex_kall", "kall.nw", "kall.treefile"),

#        ("ding2023", "grampa_trees.tre", "grampa_species_tree.tre"),
#        ("dp2018", "genetrees.txt", "multree.tre"),
#        ("koe2020", "genetrees.txt", "multree.tre"),
#        ("ren2024", "genetrees.txt", "multree.tre"),
#        ("zhao2021", "grampa_trees.tre", "grampa_species_tree.tre"),
        ],

    }

    curr_path = Path(__file__).parent

    #old_tool = "C:\\Users\\psfsitaymay\\Downloads\\Grampa_RT_etc\\grandma\\grampa.py"
    old_tool = "E:\\Synced\\Studies\\Grampa_RT_etc\\grandma\\grampa.py"
    new_tool = curr_path.parent / "gran.py"

    for kind, dataset in datasets.items():
        for ds_name, g_file, s_file in dataset:

            path_manual = curr_path / "test_data" / kind / ds_name

            g = path_manual / g_file
            s = path_manual / s_file
            x = path_manual / "ploidies.txt"

            o_old = path_manual / 'o_old'
            o_new = path_manual / 'o_new'

            run_str_new = f"python {new_tool} -g {g} -s {s} --debug --plot -v 3 -p 10 -x E:\\Repos\\Grandma\\test\\test_data\\generator\\ex\\o_new2\\M12\\ploidies.txt --strict_constraint -o"#
            run_str_old = f"python {old_tool} -g {g} -s {s} --overwrite -o"

            #time_tool(f"{run_str_new} {o_new}/", "new") # --debug --plot -w 1 50
            #time_tool(f"{run_str_new} {o_new}_nm/ --nestedness model", "new") # --debug --plot
            #time_tool(f"{run_str_new} {o_new}_op/ --optim", "new")
            #time_tool(f"{run_str_new} {o_new}_np/ --optim --nestedness model", "new") # --debug --plot
            #time_tool(f"{run_str_new} {o_new}_spltnew/ -m split --start auto -x {x} --maps 6 --min_gt_lvs 1", "new") #--plot --start 1 -i 2 --repair
            
            time_tool(f"{run_str_new} {o_new}2/ --generate {g} --bench", "new")
            
            #time_tool(f"{run_str_old} {o_old}_v0 --maps -v 0", "old")
            #time_tool(f"{run_str_old} {o_old}_v1 --maps -v 1", "old")
            #time_tool(f"{run_str_old} {o_old}_v2 --maps -v 2", "old")
            #time_tool(f"{run_str_old} {o_old}_iter/ -v 3 -p 10 -i 0", "old") #_v3

            #time_tool(f"python {curr_path / 'exe_compare.py'} -log {path_manual / 'compare_log.txt'} -old {o_old} -new {o_new}", "compare") #_splt\\0\\output


