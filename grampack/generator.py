import json
import random
from pathlib import Path
from typing import List, Tuple, Optional, Union, Any
from ete3 import Tree, TreeNode

from .models import SmrtTree
from .ops import TreeLoader
from Reticulate_Tree.reticulate_tree import ReticulateTree

class DatasetGenerator:
    def __init__(self, 
                 args: Optional[Any],
                 output_dir: Union[str, Path], 
                 logger: Optional[Any]):
        
        self.base_st_text = Path(args.spec_input).read_text().strip()
        self.output_dir = Path(output_dir)
        self.args = args
        self.logger = logger
        self.is_execute = False
        self.is_generate = False
        
        # Load JSON Configuration
        with open(args.generate, 'r') as f:
            self.config = json.load(f)

        task = self.config.get("task", "generate")
        if not task in {"generate", "g", "execute", "e", "g+e", "e+g"}:
            self.logger.log(f"Invalid task '{task}' in config. Must be one of '(g)enerate', '(e)xecute', or 'g+e'.", 'e')
        else:
            if task in {"execute", "e", "g+e", "e+g"}:
                self.is_execute = True
            if task in {"generate", "g", "g+e", "e+g"}:
                self.is_generate = True
            
        self.n_gts = self.config.get("n", 50)
        
        # Parse rates (can be float or dict)
        rates = self.config.get("rates", 0.1)
        if isinstance(rates, (float, int)):
            self.h_err = float(rates)
            self.dup_err = float(rates)
            self.loss_err = float(rates)
        else:
            self.h_err = rates.get("h_err", 0.1)
            self.dup_err = rates.get("dup_err", 0.1)
            self.loss_err = rates.get("loss_err", 0.1)

        self.executions = self.config.get("executions", [""])

    def run(self):
        datasets = self.config.get("datasets", {})
        if self.is_generate:
            self.logger.log(f"\n# --- Generating Histories for {len(datasets)} Datasets ---\n#", 'i')

        ds_dirs = []
        for name, events in datasets.items():
            if self.is_generate:
                self.logger.log(f"Generating dataset: {name} with {len(events)} event(s).", 'i')
                self._generate_dataset(name, events)
            ds_dirs.append(self.output_dir / name)

        # --- EXECUTE GRANDMA ON GENERATED DATASETS ---
        if self.is_execute:

            import sys
            import shlex
            import subprocess

            self.logger.log(f"\n# --- Starting Batch Execution on {len(ds_dirs)} Datasets ---\n#", 'i')
            
            # 1. Construct Base Command
            # We filter out the --generate flag and any existing input/output flags
            # to avoid conflicts when we append the new ones.
            base_cmd = [sys.executable, sys.argv[0]]
            
            skip_next = False
            # Flags we must remove so we can override them
            override_flags = {'-s', '--species-tree', '-g', '--gene-trees', '-o', '--output-dir', '--generate', '-m', '--mode', '--nesting'}
            
            for arg in sys.argv[1:]:
                if skip_next:
                    skip_next = False
                    continue
                
                # Handle flag=value syntax
                if any(arg.startswith(f"{f}=") for f in override_flags):
                    continue
                
                if arg in override_flags:
                    skip_next = True 
                    continue
                
                base_cmd.append(arg)

            # 2. Loop Datasets and Executions
            for ds_dir in ds_dirs:
                self.logger.log(f">> Running GRANDMA on {ds_dir.name}...", 'i')
                
                for exec_args_str in self.executions:
                    cmd = base_cmd.copy()
                    
                    # Parse the string into safe CLI arguments
                    exec_args = shlex.split(exec_args_str) if exec_args_str else []
                    
                    # Build a clean directory name from the arguments (e.g. "-m full --nesting r" -> "out_m_full_nesting_r")
                    if exec_args:
                        clean_parts = [p.strip("-") for p in exec_args]
                        mode_str = f"out3_{'_'.join(clean_parts)}"
                    else:
                        mode_str = "out3_default"
                    
                    # Inject paths and execution args
                    cmd.extend(['-s', str(ds_dir / "species_tree.tre")])
                    cmd.extend(['-g', str(ds_dir / "gene_trees.tre")])
                    cmd.extend(['-o', str(ds_dir / mode_str)])
                    cmd.extend(exec_args)
                    
                    self.logger.log(f"  Command: {' '.join(cmd)}", 'i')
                    
                    try:
                        # Run in subprocess to ensure clean memory/state for every run
                        subprocess.check_call(cmd)
                    except subprocess.CalledProcessError as e:
                        self.logger.log(f"Failed to run on {ds_dir.name} with args '{exec_args_str}'. Error code: {e.returncode}", 'w')
                    except Exception as e:
                        self.logger.log(f"Critical error executing {ds_dir.name} with args '{exec_args_str}': {e}", 'w')
                        
            self.logger.log(f"\n--- Batch Execution Completed ---\n", 'i')

    def _generate_dataset(self, name: str, events: List[Tuple[str, str]]):
        ds_dir = self.output_dir / name
        ds_dir.mkdir(parents=True, exist_ok=True)

        # 1. Build Ground Truth ST (Cumulative)
        # Parse with format=1 to support internal node names
        rt = ReticulateTree(self.base_st_text, is_multree=True)
        if rt.retnodes:
            self.logger.log(f"Species tree used by the generator must be singly-labelled. Found reticulation nodes: {rt.retnodes}", 'e')
        base_tree_obj = Tree(self.base_st_text, format=1)
        st_wrapper = SmrtTree(tree_obj=base_tree_obj)

        # Apply all events to ST to get the Ground Truth
        if events:
            for k, (h1_name, h2_name) in enumerate(events):
                # 1. Find Source (Single Lineage)
                h1_node = st_wrapper.get_node(h1_name)
                
                # 2. Find Targets (All matching copies)
                # Resolve pure name to find all instances (e.g., SpeciesA, SpeciesA*)
                # If h2_name is a specific node name (e.g. <3>), get its pure form
                temp_node = st_wrapper.get_node(h2_name)
                targets = st_wrapper.match(temp_node.pure)
                
                if not h1_node or not targets:
                    self.logger.log(f"  Dataset {name}: Skipping event {k} ({h1_name}->{h2_name}) in ground truth: source or target nodes not found.", 'w')
                    continue
                
                # Apply Graft to ALL targets
                for i, target in enumerate(targets):
                    # We copy H1 for each target
                    h1_copy = SmrtTree.copy_lineage(h1_node, tag=f"|{k+1}.{i}")
                    graft_name = f"<P{k+1}|{k+1}.{i}>"
                    st_wrapper.ete_tree = SmrtTree.graft_subtree(st_wrapper.ete_tree, target, h1_copy, graft_name)
                
                st_wrapper.refresh()

        final_true_st = st_wrapper.to_str(internals=True)
        
        # 2. Generate Gene Trees
        all_simulated_gts = []
        generated_count = 0
        attempts = 0
        max_attempts = self.n_gts * 50

        while generated_count < self.n_gts and attempts < max_attempts:
            attempts += 1
            
            # Start fresh from base
            gt_base_obj = Tree(self.base_st_text, format=1)
            gt_wrapper = SmrtTree(tree_obj=gt_base_obj)
            gt_wrapper._index_nodes()

            # Apply Events with Perturbation
            gt_success = True
            for k, (h1_name, h2_name) in enumerate(events):
                # Source is usually a specific lineage in the base
                p_h1_node = self._perturb_location(gt_wrapper, h1_name)
                
                # Targets: Find all matching candidates in current GT structure
                # We must find matches FIRST, then perturb each one individually
                temp_node = gt_wrapper.get_node(h2_name)
                raw_targets = gt_wrapper.match(temp_node.pure)
                
                if not p_h1_node or not raw_targets:
                    self.logger.log(f"  Dataset {name}: Skipping event {k} ({h1_name}->{h2_name}) in gene tree: source or target nodes not found.", 'w')
                    gt_success = False; break

                tag = f"" 
                
                # Graft onto ALL perturbed targets
                for target_node in raw_targets:
                    # Perturb this specific target location
                    p_target = self._perturb_location(gt_wrapper, target_node.name)
                    
                    # If perturbation moved off tree or node lost, try to stick to original
                    if not p_target: p_target = target_node
                    
                    gt_h1_copy = SmrtTree.copy_lineage(p_h1_node, tag=tag) 
                    gt_wrapper.ete_tree = SmrtTree.graft_subtree(gt_wrapper.ete_tree, p_target, gt_h1_copy, "<Graft>")
                
                gt_wrapper.refresh()

            if gt_success:
                # 3. Apply Noise (Duplication Pass)
                # This modifies the tree in-place by expanding nodes
                self._apply_duplication_recursive(gt_wrapper.ete_tree)
                
                # 4. Apply Noise (Loss Pass)
                # Identify survivors and prune everything else
                survivors = self._select_survivors(gt_wrapper.ete_tree)
                
                if len(survivors) >= 3: # Minimal tree size check
                    try:
                        gt_wrapper.ete_tree.prune(survivors, preserve_branch_length=True)
                        
                        # 5. Repair Tips
                        TreeLoader._check_and_fix_names(gt_wrapper.ete_tree)
                        all_simulated_gts.append(gt_wrapper.ete_tree.write(format=9))
                        generated_count += 1
                    except Exception:
                        pass # Prune failed (e.g. root issues), skip

        # 3. Save Files
        st_out = ds_dir / "species_tree.tre"
        with open(st_out, 'w') as f:
            f.write(Tree(self.base_st_text, format=1).write(format=9) + "\n") 
            
        true_st_out = ds_dir / "true_mul_tree.tre"
        with open(true_st_out, 'w') as f:
            f.write(final_true_st + "\n")

        gt_out = ds_dir / "gene_trees.tre"
        with open(gt_out, 'w') as f:
            for gt_str in all_simulated_gts:
                f.write(gt_str + "\n")
                
        self.logger.log(f"  Saved {len(all_simulated_gts)} gene trees to {ds_dir}", 'i')

    def _perturb_location(self, tree: SmrtTree, start_name: str) -> Optional[TreeNode]:
        """Randomly walks up or down from start_node based on h_err."""
        curr = tree.get_node(start_name)
        if not curr: return None
        
        # Simple geometric walk
        while random.random() < self.h_err:
            if random.random() < 0.5:
                if curr.up: curr = curr.up
            else:
                if not curr.is_leaf(): curr = random.choice(curr.children)
        return curr

    def _apply_duplication_recursive(self, node: TreeNode):
        """
        Pass 1: Structurally expands nodes based on dup_err.
        Modifies tree in-place.
        """
        # Recurse first (bottom-up prevents infinite recursion on new nodes)
        for child in node.children:
            self._apply_duplication_recursive(child)
            
        if random.random() < self.dup_err:
            # Create Dup Event: Node -> (Node_Left, Node_Right)
            # We copy the *children* to the new nodes, effectively duplicating the subtree structure
            # But since we are bottom-up, the children are already processed/expanded.
            
            # Detach current children
            children = [c.detach() for c in node.children]
            
            # Create two copies of the branch point
            left = TreeNode(); left.dist = 0; left.name = node.name
            right = TreeNode(); right.dist = 0; right.name = node.name
            
            # Clone children to both
            for c in children:
                left.add_child(c.copy())
                right.add_child(c) # Original objects go to right
            
            node.add_child(left)
            node.add_child(right)
            node.name = f"Dup_{node.name}" if node.name else "Dup"

    def _select_survivors(self, node: TreeNode) -> List[TreeNode]:
        """
        Pass 2: Selects surviving leaves based on loss_err.
        Returns list of leaf nodes to KEEP.
        """
        survivors = []
        for leaf in node.iter_leaves():
            # Roll for survival (1 - loss_err)
            if random.random() >= self.loss_err:
                survivors.append(leaf)
        return survivors