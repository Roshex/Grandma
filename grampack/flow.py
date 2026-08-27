import json
import random
import shutil
from typing import Tuple, List, Optional, Dict, Set, Callable, Any
from collections import defaultdict
from pathlib import Path
from functools import partial
from dataclasses import dataclass, field

from .config import GlobalContext
from .models import Tree, TreeNode, SmrtTree, TaskResult, MulTree, Map, ConcurrTask, GraftRecord, ProtectedDict, HistoryType
from .ops import CommonOps
from .logger import GranLogger

class BeamTracker:
    """Manages the Beam Search Tree as a JSON-backed Adjacency List for O(1) tracking."""
    def __init__(self, filepath: Path):
        self.filepath = filepath
        # node_id -> {"mt_idx": int, "task": "i.j", "parent": parent_id, "children": [child_ids]}
        initial_state = {"S0": {"mt_idx": 0, "task": "0.0", "parent": None, "children": []}}
        self.state = {
            "next_id": 1,
            "nodes": ProtectedDict(initial_state)
        }
        self.fake_to_real = {}
        self.symlink_anchors = set()
        self.load()

    def load(self):
        if self.filepath.exists():
            with open(self.filepath, 'r') as f:
                data = json.load(f)
                self.state['next_id'] = data['next_id']
                # Load JSON and wrap nodes in protection
                self.state['nodes'] = ProtectedDict(data['nodes'])

    def save(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.state, f, indent=4)

    def spawn_child(self, parent_id: str, task: str, mt_idx: int) -> str:
        """Registers a new branch in the search space and returns its unique ID (e.g. 'S42')."""
        child_id = f"S{self.state['next_id']}"
        self.state['next_id'] += 1
        
        self.state['nodes'][child_id] = {
            "mt_idx": mt_idx,
            "task": task, # strictly string "i.j"
            "parent": parent_id,
            "children": []
        }
        if parent_id in self.state['nodes']:
            self.state['nodes'][parent_id]['children'].append(child_id)
            
        self.save()
        return child_id

    def get_descendants(self, node_id: str) -> List[str]:
        """O(V) Fast iterative DFS to fetch all downstream search branches for pruning."""
        descendants = []
        stack = list(self.state['nodes'][node_id]["children"])
        while stack:
            curr = stack.pop()
            descendants.append(curr)
            stack.extend(self.state['nodes'][curr]["children"])
        return descendants

    def get_leaves(self, node_id: str) -> List[str]:
        """Returns all terminal leaf nodes for a given search branch."""
        if node_id not in self.state['nodes']: return []
        leaves = []
        stack = [node_id]
        while stack:
            curr = stack.pop()
            children = self.state['nodes'][curr]["children"]
            if not children:
                leaves.append(curr)
            else:
                stack.extend(children)
        return leaves
    
    def block_branch(self, node_id: str) -> Set[str]:
        """Marks node as blocked natively in the BeamTracker."""
        nodes = self.state['nodes']
        if node_id in nodes:
            nodes[node_id]['blocked'] = True
            self.save()
            
        bad_descendants = set(self.get_descendants(node_id))
        bad_descendants.add(node_id)
        return bad_descendants

    def build_virtual_graph(self, history: HistoryType):
        """Cross-references History to generate missing symmetric nodes for downstream DP math."""
        self.fake_to_real.clear()
        self.symlink_anchors.clear()
        nodes = self.state['nodes']
        
        # Restore mapping from previously generated & saved fake nodes
        for nid, ndata in list(nodes.items()):
            if ndata.get('is_copy'):
                p = ndata['parent']
                if p and not nodes[p].get('is_copy'):
                    self.symlink_anchors.add(p)
                if 'real_source' in ndata:
                    self.fake_to_real[nid] = ndata['real_source']
                    
        # Find all symlinks recorded in history
        symlinks = []
        for task_tuple, parent_dict in history.items():
            for p_id, events in parent_dict.items():
                if isinstance(events, dict) and "__symlink__" in events:
                    symlinks.append((task_tuple, p_id, events["__symlink__"]))
                    self.symlink_anchors.add(p_id)

        # Sort from Highest Depth (Deepest) to Lowest Depth (Root)
        # x[0] is the task_tuple (depth, idx). x[0][0] is the depth.
        symlinks.sort(key=lambda x: x[0][0], reverse=True)

        def clone_subtree(real_node_id, new_parent_id):
            fake_id = f"S{self.state['next_id']}"
            self.state['next_id'] += 1

            # Resolve the ultimate real source for chained copies
            # If the node we are copying is already a copy, grab its original source
            ultimate_source = nodes[real_node_id].get('real_source', real_node_id)
            
            self.fake_to_real[fake_id] = ultimate_source
            
            real_node = nodes[real_node_id]
            fake_node = real_node.copy()
            fake_node['parent'] = new_parent_id
            fake_node['children'] = []
            fake_node['is_copy'] = True
            fake_node['real_source'] = ultimate_source 
            nodes[fake_id] = fake_node
            
            if new_parent_id in nodes:
                nodes[new_parent_id]['children'].append(fake_id)
                
            for child_id in real_node.get('children', []):
                clone_subtree(child_id, fake_id)

        # Generate missing fake nodes dynamically
        graph_modified = False
        for task_tuple, sym_parent, src_parent in symlinks:
            sym_parent_node = nodes[sym_parent]
            
            task_str = f"{task_tuple[0]}.{task_tuple[1]}"
            
            already_generated = any(
                nodes[cid].get('is_copy') and nodes[cid]['task'] == task_str
                for cid in sym_parent_node['children']
            )
            if already_generated: continue

            self.symlink_anchors.add(sym_parent)
            src_node = nodes[src_parent]
            
            for child_id in src_node['children']:
                child_node = nodes[child_id]
                if child_node['task'] == task_str:
                    clone_subtree(child_id, sym_parent)
                    graph_modified = True
                        
        if graph_modified:
            self.save()

class PathNavigator:
    """Handles all Dynamic Programming, Greedy Pathing, and Graph Visualizations."""
    def __init__(self, tracker: BeamTracker, history: HistoryType, out_dir: Path, switch: str, logger: GranLogger, max_combs: int = 500):
        self.tracker = tracker
        self.history = history
        self.out_dir = out_dir
        self.switch = switch
        self.logger = logger
        self.max_combs = max_combs
        # Prep happens strictly here
        self.tracker.build_virtual_graph(self.history)

    def _parse_task(self, task_str: str) -> Tuple[int, int]:
        """Helper to standardize task string formatting."""
        return tuple(map(int, task_str.split('.')))

    def _get_hist_node(self, task_tuple: Tuple[int, int], parent_id: str, node_id: str) -> dict:
        """Safely fetches history data natively from HistoryType."""
        if task_tuple not in self.history: return {}
        
        real_parent = self.tracker.fake_to_real.get(parent_id, parent_id)
        real_node = self.tracker.fake_to_real.get(node_id, node_id)
        
        parent_data = self.history[task_tuple].get(real_parent, {})
        
        if "__symlink__" in parent_data:
            src_parent = parent_data["__symlink__"]
            return self.history[task_tuple][src_parent][real_node]
            
        return parent_data.get(real_node, {})

    def compute_all_universes(self, start_node: str = "S0", max_depth: Optional[int] = None) -> Tuple[List[Tuple[float, Set[str], str]], Dict[str, float]]:
        """
        Computes all combinatorial paths optimally using Delta Scoring (Parsimony Savings).
        Returns a sorted list of: (Absolute Score, Set of Node IDs, Formatted Lineage String)
        """
        from itertools import product
        from collections import defaultdict
        import heapq
        
        nodes = self.tracker.state['nodes']
        memo = {}
        
        def traverse(node_id: str) -> List[Tuple[float, Set[str], str]]:
            node = nodes[node_id]
            
            task = self._parse_task(node['task'])
            parent_id = node['parent']
            current_depth = task[0]
            
            # --- Calculate Local Delta (Parsimony Savings) ---
            passed = True
            blocked = node.get('blocked', False)
            local_delta = 0.0
            
            if parent_id:
                ev = self._get_hist_node(task, parent_id, node_id)
                in_data = self._get_hist_node(task, parent_id, "In")
                in_score = in_data['score']
                passed = ev.get('passed', True)
                
                if passed:
                    out_score = ev['score']
                    # Delta is negative if parsimony improves (score drops)
                    local_delta = out_score - in_score
                else:
                    local_delta = 0.0

            # --- Base Case: Forced Leaf ---
            if not passed or blocked or (max_depth is not None and current_depth >= max_depth):
                memo[node_id] = local_delta
                return [(local_delta, {node_id}, node_id)]

            # --- Identify Surviving Children in the Graph ---
            children = node['children']
            groups = defaultdict(list)
            for cid in children:
                c_node = nodes[cid]
                groups[c_node['task']].append(cid)

            # --- Identify Expected Tasks from History ---
            expected_tasks = []
            for t_tuple, parent_dict in self.history.items():
                real_node = self.tracker.fake_to_real.get(node_id, node_id)
                if real_node in parent_dict or node_id in parent_dict:
                    t_str = f"{t_tuple[0]}.{t_tuple[1]}"
                    expected_tasks.append((t_tuple, t_str))

            # --- Base Case: Natural Leaf ---
            if not expected_tasks and not children:
                memo[node_id] = local_delta
                return [(local_delta, {node_id}, node_id)]

            # --- Build Options for ALL Expected Branches ---
            task_options = []
            for t_tuple, t_str in expected_tasks:
                if t_str in groups and groups[t_str]:
                    # Branch survived: Traverse its children
                    options = []
                    for cid in groups[t_str]:
                        options.extend(traverse(cid))
                    task_options.append(options)
                else:
                    # Branch failed entirely: Delta is exactly 0.0
                    task_options.append([(0.0, set(), f"FAIL({t_str})")])

            # --- Dynamic Yielding (Memory Optimization) ---
            def combo_generator():
                for combo in product(*task_options):
                    # Universe Delta = My Delta + Sum of Children Deltas
                    delta_sum = local_delta + sum(c[0] for c in combo)
                    
                    combined_nodes = {node_id}
                    for c in combo:
                        combined_nodes.update(c[1])
                    
                    if len(combo) == 1:
                        lineage_str = f"{node_id} -> {combo[0][2]}"
                    else:
                        branches = " & ".join(f"({c[2]})" for c in combo)
                        lineage_str = f"{node_id} -> [{branches}]"
                        
                    yield (delta_sum, combined_nodes, lineage_str)
            
            # Keep Top max_combs combinations based on best (lowest/most negative) delta
            pruned = heapq.nsmallest(self.max_combs, combo_generator(), key=lambda x: x[0])
            
            memo[node_id] = pruned[0][0] if pruned else float('inf')
            return pruned

        # Run traversal to get delta-based universes
        delta_universes = traverse(start_node)
        
        # --- Convert Deltas back to Absolute Scores ---
        final_universes = []
        for delta, nodes_set, lineage in delta_universes:
            final_universes.append((delta, nodes_set, lineage))

        return final_universes, memo
        
    def _get_greedy_universe(self, start_node: str = "S0", max_depth: Optional[int] = None) -> Set[str]:
        """Fast, top-down extraction of the purely greedy path, ignoring DP lookahead."""
        nodes = self.tracker.state['nodes']
        greedy_states = set()
        active = [start_node]
        
        while active:
            next_active = []
            for node_id in active:
                greedy_states.add(node_id)
                node = nodes[node_id]
                
                task = self._parse_task(node['task'])
                if max_depth is not None and task[0] >= max_depth:
                    continue
                
                children = node['children']
                if not children: continue
                
                # Group children by task (AND logic for splits)
                groups = defaultdict(list)
                for cid in children:
                    c_node = nodes[cid]
                    groups[c_node['task']].append(cid)
                        
                # For each required sub-task, greedily pick the lowest *immediate* score
                for t_str, cids in groups.items():
                    best_score = float('inf')
                    best_cid = None
                    for cid in cids:
                        c_node = nodes[cid]
                        c_task = self._parse_task(c_node['task'])
                        
                        ev = self._get_hist_node(c_task, node_id, cid)
                        passed = ev.get('passed', True)
                        
                        if passed:
                            s = ev['score']
                        else:
                            in_data = self._get_hist_node(c_task, node_id, "In")
                            s = in_data['score']
                            
                        if s < best_score:
                            best_score = s
                            best_cid = cid
                            
                    if best_cid:
                        next_active.append(best_cid)
            active = next_active
            
        return greedy_states

    def _flatten_history(self, state_ids: Set[str], nodes: Dict[str, Dict]) -> Dict[Tuple[int, int], Dict]:
        """Extracts and formats the flat history sequence from a set of state IDs."""
        flat_hist = {}
        
        for state_id in state_ids:
            if state_id == "S0": continue
            node = nodes.get(state_id)
            if not node: continue
            
            parent_id = node['parent']
            task = self._parse_task(node['task'])
            
            ev = self._get_hist_node(task, parent_id, state_id).copy()
            if not ev: continue
                
            in_data = self._get_hist_node(task, parent_id, "In")
            if in_data:
                ev['in_score'] = in_data.get('score')
                ev['in_tree']  = in_data.get('sp_tree')
                ev['num_gts']  = in_data.get('num_gts')
                
            flat_hist[task] = ev

        return dict(sorted(flat_hist.items()))

    def get_bulk_histories(self, target_states: Set[str]) -> Dict[str, Tuple[Dict, str]]:
        """Computes DP math exactly ONCE, then extracts flat histories for multiple target states."""
        self.tracker.build_virtual_graph(self.history)
        all_universes, _ = self.compute_all_universes(start_node="S0")
        nodes = self.tracker.state['nodes']
        
        results = {}
        
        for s_id in target_states:
            # Instantly filter the pre-computed math
            valid_universes = [u for u in all_universes if s_id in u[1]]
            if not valid_universes: continue
                
            # Extract the nodes for this specific target
            _, best_states, lineage_str = valid_universes[0]
            
            flat_hist = self._flatten_history(best_states, nodes)
                
            results[s_id] = (flat_hist, lineage_str)
            
        return results

    def get_multiverse_history(self, target_state_id: Optional[str] = None, start_node: str = "S0", max_depth: Optional[int] = None) -> Tuple[Dict[Tuple[int, int], Dict], str]:

        self.tracker.build_virtual_graph(self.history)
        all_universes, memo = self.compute_all_universes(start_node=start_node, max_depth=max_depth)
        
        valid_universes = all_universes
        if target_state_id:
            valid_universes = [u for u in all_universes if target_state_id in u[1]]
            
        if not valid_universes:
            self.logger.log(f"No valid universes found (Target: {target_state_id})", 'w')
            return {}, ""
            
        # Unpack the new 3-tuple
        root_score, best_universe_states, lineage_str = valid_universes[0]
        
        self.logger.log("--- Universe Evaluation ---", 'i')
        self.logger.log(f"Total Universe Score: {root_score}", 'i')
        self.logger.log(f"Selected Lineage: {lineage_str}", 'd')

        nodes = self.tracker.state['nodes']
        flat_hist = self._flatten_history(best_universe_states, nodes)

        greedy_paths = []

        for opt_state in best_universe_states:
            # Get the greedy suffix starting from this optimal node
            greedy_suffix = self._get_greedy_universe(start_node=opt_state, max_depth=max_depth)
            
            # Build the full root-to-leaf path for the score plot highlighting
            prefix = set()
            curr = opt_state
            while curr:
                prefix.add(curr)
                curr = nodes[curr]['parent']
                
            greedy_paths.append(prefix | greedy_suffix)
        
        debug_file = self.out_dir / f"best_history.json"
        try:
            import json
            with open(debug_file, 'w') as f:
                json.dump({str(k): v for k, v in flat_hist.items()}, f, indent=4)
        except Exception as e:
            self.logger.log(f"Failed to save Universe History: {e}", 'w')

        self.plot_multiverse_evolution(best_universe_states, greedy_paths, memo)
        self.plot_multiverse_rankings(all_universes, best_universe_states, greedy_paths)
                
        return flat_hist, lineage_str

    def plot_multiverse_evolution(self, best_universe_states: Set[str], greedy_paths: List[Set[str]], memo: Dict[str, float], show_copies: bool = False):
        """
        Plots the multiverse search graph using a hierarchical, crossing-free layout with color-coded nodes and edges.
        Displayed scores represent the optimal cumulative downstream parsimony (from the leaves up to that node), rather than isolated root-to-leaf
        path totals. As a result, only the states comprising the globally optimal universe will display all constituent scores of that universe.
        """
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D
            
            nodes = self.tracker.state['nodes']
            G = nx.DiGraph()
            labels, node_colors = {}, []
            switch = self.switch
            
            # Helper to definitively identify branch intention based on active mode
            def get_branch_type(task_tuple) -> str:
                i, j = task_tuple
                if i < switch:
                    # Full mode regime
                    return 'nested' if j > 0 else 'standard'
                else:
                    # Split mode regime
                    return 'inner' if j % 2 == 1 else 'outer'

            # Build the graph and assign edge colors dynamically
            for n_id, n_data in nodes.items():
                if not show_copies and n_data.get('is_copy'): continue
                
                G.add_node(n_id)
                parent = n_data['parent']
                if parent:
                    if not show_copies and nodes[parent].get('is_copy'): continue
                    
                    task = self._parse_task(n_data['task'])
                    b_type = get_branch_type(task)
                    
                    if b_type == 'inner': e_color = 'dodgerblue'
                    elif b_type == 'outer': e_color = 'saddlebrown'
                    elif b_type == 'nested': e_color = 'dimgray'
                    else: e_color = 'gray'
                        
                    G.add_edge(parent, n_id, color=e_color)
                    
            # Determine Topological Depth (Y-axis)
            depths = {}
            def get_depth(node):
                if node not in depths:
                    parent = nodes[node]['parent']
                    # Safer parent check since we might have hidden the parent
                    if parent and parent in G.nodes():
                        depths[node] = get_depth(parent) + 1
                    else:
                        depths[node] = 0
                return depths[node]
                
            for n_id in G.nodes(): get_depth(n_id)

            # Determine Horizontal Position (X-axis) via DFS
            x_positions = {}
            leaf_counter = 0

            def assign_x(node):
                nonlocal leaf_counter
                children = list(G.successors(node))
                
                # Sort numerically (e.g., S1 before S2) for predictable left-to-right ordering
                children.sort(key=lambda x: int(x.replace('S', '')) if x.startswith('S') and x[1:].isdigit() else x)
                
                if not children:
                    x_positions[node] = leaf_counter
                    leaf_counter += 1
                else:
                    for child in children:
                        assign_x(child)
                    x_positions[node] = sum(x_positions[c] for c in children) / len(children)

            roots = [n for n, d in G.in_degree() if d == 0]
            if roots:
                assign_x(roots[0])
            else:
                for n in G.nodes():
                    if n not in x_positions: assign_x(n)

            pos = {node: (x_positions[node] * 2, -depths[node]) for node in G.nodes()}
            
            # Build visuals
            greedy_universe_states = set().union(*greedy_paths) if greedy_paths else set()
            for n_id in G.nodes():

                score = memo.get(n_id, None)
                if score is not None:
                    score_str = f"{score:.0f}"
                else:
                    # Differentiate between actual parsimony failures and blocked searches
                    passed = True
                    p_id = nodes[n_id]['parent']
                    if p_id:
                        task = self._parse_task(nodes[n_id]['task'])
                        ev = self._get_hist_node(task, p_id, n_id)
                        passed = ev.get('passed', True)
                    score_str = "0" if not passed else "BLOCK"

                labels[n_id] = f"{n_id}\n({score_str})"

                if getattr(self.tracker, 'symlink_anchors', None) and n_id in self.tracker.symlink_anchors:
                    node_colors.append('orange') # It's a symlink anchor
                elif n_id in best_universe_states:
                    node_colors.append('lightgreen')
                elif n_id in greedy_universe_states: 
                    node_colors.append('lightcoral')
                else:
                    node_colors.append('lightgrey')

            edge_colors = [G[u][v]['color'] for u, v in G.edges()]
                    
            plt.figure(figsize=(16, 10))
            nx.draw(G, pos, with_labels=True, labels=labels, node_color=node_colors, 
                    node_size=2400, font_size=8, font_weight="bold", 
                    edge_color=edge_colors, width=1.5, arrows=True)
            
            legend_elements = [
                Line2D([0], [0], color='w', marker='o', markerfacecolor='lightgreen', markersize=12, label='Optimal Universe'),
                Line2D([0], [0], color='w', marker='o', markerfacecolor='orange', markersize=12, label='Deduplication Nodes'),
                Line2D([0], [0], color='w', marker='o', markerfacecolor='lightcoral', markersize=12, label='Greedy Path Divergence'),
                Line2D([0], [0], color='gray', lw=2, label='Sequential'),
                Line2D([0], [0], color='saddlebrown', lw=2, label='Outer Split'),
                Line2D([0], [0], color='dodgerblue', lw=2, label='Inner Split')
            ]
            plt.legend(handles=legend_elements, loc='upper left', fontsize=10, 
                       title="Branch Types", title_fontsize='11', framealpha=0.9)
            
            plt.title("Multiverse Evolution:\nHierarchical Layout of All Search Paths", fontsize=14)
            out_file = self.out_dir / "multiverse_evolution.png"
            plt.savefig(out_file, dpi=600, bbox_inches='tight')
            plt.close()
            self.logger.log(f"Saved Multiverse Evolution plot to {out_file}", 'd')
            
        except ImportError:
            self.logger.log("NetworkX or Matplotlib not installed. Skipping multiverse evolution plot.", 'w')

    def plot_multiverse_rankings(self, all_universes: List[Tuple[float, Set[str], str]], best_universe_states: Set[str], greedy_paths: List[Set[str]]):
        """
        Plots a comparison ranking of all complete root-to-leaf universes evaluated.
        Scores are the absolute total parsimony savings (lower is better): each bar is an entire universe.
        Colors are green for the optimal universe, red for any universe that shares nodes with any of the greedy paths, and blue for others.
        """
        try:
            import matplotlib.pyplot as plt
            
            valid_universes = [u for u in all_universes if u[0] != float('inf')]
            len_universes = len(valid_universes)
            if not valid_universes or len_universes < 2: return
            if len_universes > 30:
                self.logger.log(f"Too many universes to plot ({len_universes}). Top 30 will be shown.", 'i')
                valid_universes = valid_universes[:30]
            
            u_labels = []
            s_vals = []
            colors = []
            
            for rank, (score, combined_nodes, lineage_str) in enumerate(valid_universes):
                # Truncate string if it gets too long for plotting
                if len(lineage_str) > 35:
                    label = lineage_str[:32] + "..."
                else:
                    label = lineage_str
                    
                label = f"{label} (#{rank+1})"
                u_labels.append(label)
                s_vals.append(score)

                # Highlight logic (Winner takes priority)
                is_winner = combined_nodes == best_universe_states
                is_greedy = any(combined_nodes.issubset(g_path) for g_path in greedy_paths)
                
                if is_winner:
                    colors.append('lightgreen')
                elif is_greedy:
                    colors.append('lightcoral')
                else:
                    colors.append('skyblue')

            fig_width = max(12, len(u_labels) * 0.35)
            plt.figure(figsize=(fig_width, 7))
            
            bars = plt.bar(u_labels, s_vals, color=colors)
            
            plt.ylabel("Total Score-Loss (Lower is better)")
            plt.xlabel("Universe Traversal Path (Root -> Leaves)")
            plt.title(f"All Complete Universes Ranked by Score Improvement (Top {len(valid_universes)})")
            
            plt.xticks(rotation=45, ha='right', fontsize=8)
            
            if len(bars) <= 60:
                for bar in bars:
                    plt.text(bar.get_x() + bar.get_width()/2.0, bar.get_height(), 
                             f'{bar.get_height():.1f}', va='bottom', ha='center', 
                             fontsize=8, rotation=90)
                
            plt.tight_layout()
            out_file = self.out_dir / "multiverse_rankings.png"
            plt.savefig(out_file, dpi=600)
            plt.close()
            self.logger.log(f"Saved Multiverse Ranking plot to {out_file}", 'd')
            
        except ImportError:
            self.logger.log("Matplotlib not installed. Skipping multiverse scores plot.", 'w')

class FlowManager:
    def __init__(self, ctx: GlobalContext, mode: str, logger: GranLogger):
        self.ctx = ctx
        self.mode = mode
        self.sample = self.set_sampling_func(ctx.sample)
        self.logger = logger
        self.tracker = BeamTracker(self.ctx.beam_file)
        self.navigator = PathNavigator(self.tracker, self.ctx.history, self.ctx.root_dir, self._infer_switch(), self.logger)
        self.best_history = None
       
    # --- Init Methods ---

    def set_sampling_func(self, n: int) -> callable:
        if self.ctx.debug:
            if self.ctx.seed:
                def _random_spacing(iterable, n):
                    """Returns n random values from iterable using fixed seed."""
                    idxs = sorted(random.sample(range(len(iterable)), min(n, len(iterable))))
                    return [iterable[i] for i in idxs]
                return partial(_random_spacing, n=n)
            else:
                def _equal_spacing(iterable, n):
                    """Returns n evenly spaced values from iterable."""
                    length = len(iterable)
                    if n >= length:
                        return list(range(length))
                    step = length / n
                    return [iterable[int(i * step)] for i in range(n)]
                return partial(_equal_spacing, n=n)
        def _noop(iterable):
            return []
        return _noop

    # --- Overlapping Methods for Full and Split Modes ---

    def _check_if_passed__(self, i: int, j: int, scores: dict, search_state_id: str, valleys: Tuple[float, float, float]) -> bool:
        """Returns True if the event should be accepted based on the cutoff rules."""
        ref_type, diff_func, offset = self.ctx.cutoff
        switch = self._infer_switch()

        if ref_type == 'none': return True
        if i < switch and j > 0: return True # Nested-fix always passes

        # Determine the Base Reference Score
        if ref_type == 'input':
            if i >= switch or i == 0: # Split regime or first iteration of full mode
                comp_score = scores['input']
            else: # Full mode after first iteration: look back to parent event's non-input score
                comp_score = self._get_prev_score(i, j, search_state_id)
        elif ref_type == 'fvall': comp_score = valleys[0]
        elif ref_type == 'lvall': comp_score = valleys[1]
        elif ref_type == 'rvall': comp_score = valleys[2]
        else:                     self.logger.log(f"Unknown cutoff reference type: {ref_type}.", 'e')
            
        # Apply the cutoff
        if diff_func == 'rel': offset *= comp_score
        return offset < (comp_score - scores['own'])

    def _get_prev_score__(self, i: int, j: int, search_state_id: str) -> float:
        """
        Retrieves the previous event's own score from history.
        Uses the PathNavigator to safely resolve any symlink paths.
        """
        parent_id = self.tracker.state['nodes'][search_state_id]['parent']
        parent_node = self.tracker.state['nodes'][parent_id]
        p_task = self.navigator._parse_task(parent_node['task'])
        
        # Safely fetch the historical event, correctly resolving virtual/symlinked nodes
        ev = self.navigator._get_hist_node(p_task, parent_id, search_state_id)

        return ev['score']        

    def _get_nonin_rank(self, res: TaskResult) -> int:
        """Returns the index of the best non-input MulTree."""
        # If input is best (idx 0), return idx of the second-best tree (rank 1)
        # Else, return idx of the best tree (rank 0)
        nonin_rank = 1 if res.mt_idx() == 0 else 0
        if nonin_rank not in res.mul_trees:
            nonin_rank = None
        return nonin_rank
    
    def save_events(self, i: int, j: int,
                     res: TaskResult,
                     search_state_id: str,
                     transform: Optional[Callable] = None) -> List[Tuple[int, MulTree, str, Optional[Any], float]]:
        
        self.ctx.history[(i, j)][search_state_id]["In"] = {
            'num_gts': len(res.gene_trees),
            'sp_tree': res.mul_trees[0].mt.to_str(internals=True) if 0 in res.mul_trees else "NA",
            'score': res.input_score
        }

        passed_events = []
        
        # Kept scores are in res.passed_events
        kept_idxs = [idx for idx in res.passed_events.keys() if idx != 0]
        kept_scores = [(idx, score) for idx, score in res.sorted_scores if idx in kept_idxs]

        if not kept_scores or search_state_id is None:
            self.ctx.history.save()
            return []

        for nonin_idx, nonin_score in kept_scores:
            nonin_mt = res.mul_trees[nonin_idx]
            passed = res.passed_events[nonin_idx]

            h_nodes = [nonin_mt.h1_node] + nonin_mt.hx_nodes
            transform_result = transform(nonin_mt, i, j) if transform else None
            sister_nodes = [nonin_mt.mt.get_sis(n) for n in h_nodes]
            new_state_id = self.tracker.spawn_child(search_state_id, f"{i}.{j}", nonin_idx)

            self.ctx.history[(i, j)][search_state_id][new_state_id] = {
                'mt_idx': nonin_idx,
                'mt_tree': nonin_mt.mt.to_str(internals=True),
                'h_name': nonin_mt.h1_node.name,
                'h_locs': [n.name if n is not None else 'None' for n in sister_nodes],
                'h_leaves': nonin_mt.h1_node.get_leaf_names(),
                'score': nonin_score,
                'passed': passed,
            }

            if passed:
                passed_events.append((nonin_idx, nonin_mt, new_state_id, transform_result, nonin_score))

        self.ctx.history.save()
        return passed_events
        
    def judge_events__(self, i: int, j: int,
                     res: TaskResult,
                     search_state_id: str,
                     transform: Optional[Callable] = None) -> List[Tuple[int, MulTree, str, Optional[Any]]]:
        """
        Judges whether the current events pass the parsimony cutoff and prepares data for the next iteration.
        Logs the input and best, or, if input is best, input and second-best MulTree data for history tracking.
        """
        step = "Assessing events parsimony"
        self.logger.report_step(step, "In progress...")

        best_idx = res.mt_idx()
        best_mt_str = res.mul_trees[best_idx].mt.to_str(internals=True)
        input_score = res.input_score
        valleys = res.valleys

        # Safely init nested dictionaries with the first entry refering to common event-level data
        '''self.ctx.history.setdefault((i, j), {}).setdefault(search_state_id, {
            "In": {
                'num_gts': len(res.gene_trees),
                'sp_tree': res.mul_trees[0].mt.to_str(internals=True),
                'score': input_score
            }
        })'''

        # NEW: Automatically creates (i, j) and search_state_id if they don't exist.
        # If "In" already exists, this WILL raise a KeyError, protecting your data!
        self.ctx.history[(i, j)][search_state_id]["In"] = {
            'num_gts': len(res.gene_trees),
            'sp_tree': res.mul_trees[0].mt.to_str(internals=True),
            'score': input_score
        }

        passed_events = []
        
        # Kept maps dictate which MTs were evaluated completely (already bounded by max_select)
        kept_idxs = [idx for idx in res.kept_mul_maps.keys() if idx != 0]
        # Scores are already sorted in ascending order to process best trees first
        kept_scores = [(idx, score) for idx, score in res.sorted_scores if idx in kept_idxs]

        if not kept_scores:
            self.logger.report_step(step, "Failed: no non-input trees to assess")
            """self.ctx.history[(i, j)][search_state_id][-1] = {
                'best_mt': best_mt_str,
                #'num_gts': len(res.gene_trees),
                'input_score': input_score,
                'nonin_score': 'N/A',
                'passed': False,
            }"""
            self.ctx.history.save()
            return []

        for nonin_idx, nonin_score in kept_scores:
            nonin_mt = res.mul_trees[nonin_idx]
            
            curr_event = {'input': input_score, 'own': nonin_score}
            passed = self._check_if_passed(i, j, curr_event, search_state_id, valleys)

            h_nodes = [nonin_mt.h1_node] + nonin_mt.hx_nodes

            # Apply operations which must take place BEFORE saving history!
            transform_result = transform(nonin_mt, i, j) if transform else None

            sister_nodes = [nonin_mt.mt.get_sis(n) for n in h_nodes]

            new_state_id = self.tracker.spawn_child(search_state_id, f"{i}.{j}", nonin_idx)

            self.ctx.history[(i, j)][search_state_id][new_state_id] = {
                'mt_idx': nonin_idx,
                'mt_tree': nonin_mt.mt.to_str(internals=True),
                'h_name': nonin_mt.h1_node.name,
                'h_locs': [n.name if n is not None else 'None' for n in sister_nodes],
                'h_leaves': nonin_mt.h1_node.get_leaf_names(),
                #'num_gts': len(res.gene_trees),
                #'input_score': input_score,
                'score': nonin_score,
                'passed': passed,
            }

            if passed:
                passed_events.append((nonin_idx, nonin_mt, new_state_id, transform_result))

        self.ctx.history.save()

        if passed_events:
            self.logger.report_step(step, f"Success: {len(passed_events)} events accepted")
        else:
            self.logger.report_step(step, "Failed: parsimony cutoff not met")

        return passed_events

    # --- Handlers for the Full mode ---

    def _relabel_species_tree(self, best_mt: MulTree, i: int, j: int) -> Dict[str, Set[str]]:
        """
        Renames the best MulTree's hybrid lineages for the next iteration.
        Format: |{i}.{j~copy_idx} (e.g., Species|1.0, <Internal|1.1>)
        best_mt is modified in place!
        
        Returns: 
            suffix_name_map: A mapping from suffix to the set of original names that were suffixed.
        """
        step = "Relabeling top non-input species tree"
        self.logger.report_step(step, "In progress...")

        mt_wrapper = best_mt.mt

        suffix_name_map = best_mt.rename_marked_nodes(i, j)
        
        self._debug_tree("Renamed MT:", mt_wrapper.ete_tree, other_attr=['H', 'pure'])

        self.logger.report_step(step, "Success")
        return suffix_name_map # suffix -> set of original names (for syncing GTs)

    def _relabel_gene_trees(self, res: TaskResult, best_mt_idx: int, suffix_name_map: Dict[str, Set[str]], copy_gts: bool) -> Dict[int, SmrtTree]:
        """
        Renames Gene Trees to match the Species Tree renaming logic.
        Uses the mapping generated by _rename_best_mt.
        Returns gt dict, but - GTs are modified in place!
        """
        step = "Relabeling gene trees"
        self.logger.report_step(step, "In progress...")

        if copy_gts: gts = {k: v.copy() for k, v in res.gene_trees.items()}
        else: gts = res.gene_trees

        min_maps = res.kept_mul_maps[best_mt_idx]
        
        debug_sample = self.sample(list(min_maps.keys()))

        for g_idx, map_obj in min_maps.items():
            gt_ete = gts[g_idx].ete_tree

            if g_idx in debug_sample:
                self._debug_tree(f"Original GT {g_idx}:", gt_ete)

            gts[g_idx].rename_leaves_from_mapping(map_obj, suffix_name_map)

            if g_idx in debug_sample:
                self._debug_tree(f"Renamed GT {g_idx}:", gt_ete, other_attr=['H', 'pure'])
        
        self.logger.report_step(step, "Success")
        return gts

    def find_missing_targets(self, multree: MulTree) -> Set[str]:
        if self.ctx.nesting in {"ignore", "model"}: return set() # Extra safety

        h1_node = multree.h1_node
        matches = multree.mt.match(h1_node.pure)
        get_sis = multree.mt.get_sis

        self.logger.log(f"Nested Fix: Found {[l.name for l in matches]} matches for H1 node in the MT.", 'd')
        all_h2_locs_names = set()
        already_populated = set()
            
        for h_node in matches:

            h_sis = get_sis(h_node)

            h2_locs = multree.mt.get_targets(h_sis)

            # New internality check logic
            skip_match = False
            if self.ctx.nesting == "strict_rectify":
                # Select one other sister as checking candidate for internality
                # We check the other sister, because h_node's P node may be sister's .up...
                other_sis = None
                for sis in h2_locs:
                    if get_sis(sis).pure != h_node.pure:
                        other_sis = sis
                        break
                if other_sis:
                    internal_count = len(h2_locs)
                    # If the parent of one of the other sisters has a count that's different from the sister's copy count,
                    # Then, the sister's lineage is rooted at the sister. Meaning, it is "external" and we should skip it.
                    parental_count = len(multree.mt.match(other_sis.up.pure))
                    if parental_count != internal_count:
                        # Not internal, so skip [strict_rectify only corrects internal nested cases]
                        skip_match = True

            if skip_match:
                h2_locs_names = set()
            else:
                h2_locs_names = set(n.name for n in h2_locs)

            self.logger.log(f"Nested Fix: For sister '{h_sis.name}', found H2 locations {h2_locs_names} after filtering.", 'd')
            all_h2_locs_names.update(h2_locs_names)
            
            already_populated.add(h_sis.name)

        # Targets must appear once, and exclude already populated ones
        # It is not safe to remove within the loop - the next match might find another copy of the same lineage.
        # We must track separately and remove at the end.
        return all_h2_locs_names - already_populated

    ### Old logic:
    # Works, but I found an easier way to check this, and earlier - during target search.
    # Thus it is currently unused.
    def _check_internality(self, t: Tree, node: TreeNode) -> bool:
        """
        Loads all the events from history that contain this node in their H1.
        Out of these events, finds the smallest one (most recent) and checks if the node is the root of the H1 subtree in that event.
        [A single species lineage is root by default]
        """
        containing_events = []
        # FIX 1: We must include the node itself in the check set.
        # If 'node' IS the H1 root, it must match 'h_node'. 
        # iter_ancestors() only gives strict parents.
        lineage_names = {p.name for p in node.iter_ancestors()}
        lineage_names.add(node.name)

        for event_data in self.ctx.history.values():
            locs = event_data.get('h_locs', [])
            for loc in locs:
                # FIX 2: Safety check. History might contain old names not in 't'
                loc_node = t.get_node(loc)
                if not loc_node: 
                    continue
                
                sisters = loc_node.get_sisters()
                if not sisters: 
                    continue

                h_node = sisters[0] # The H1 root is the sister of the graft location
                
                # Check if this H1 root is the node itself or one of its ancestors
                if h_node and h_node.name in lineage_names:
                    # Found a containing event
                    containing_events.append((h_node, len(h_node)))
                    # Optimization: No need to check other locs for the same event
                    break 

        if not containing_events:
            return False

        # FIX 3: Logic Confirmation
        # We find the "smallest" H1 (fewest leaves). In nested scenarios, 
        # the smaller H1 is the more recent nested event inside a larger older H1.
        smallest_event_root, _ = min(containing_events, key=lambda x: x[1])
        
        # Return True only if this node IS the root of that most recent event
        return smallest_event_root.name == node.name

    def relabel_problem(
            self, i: int, res: TaskResult, event: Any,
            iter_out: Path,
            iter_logger: GranLogger,
            j: int = 0,
            targets: Optional[List[str]] = None,
            copy_gts: bool = False
        ) -> Tuple[Optional[MulTree], Optional[Dict[int, SmrtTree]], Dict[str, Set[str]]]:
        """
        Handles the end of a 'Full' mode iteration using tip renaming for both the best MT and GTs.
        Returns: (next_st, next_gts) or None if stopping.
        """
        self.logger = iter_logger

        mt_idx, select_mt, _, _, _ = event

        suffix_name_map = self._relabel_species_tree(select_mt, i, j)

        # Rename Trees for Next Iteration
        next_gts = self._relabel_gene_trees(res, mt_idx, suffix_name_map, copy_gts)

        if self.ctx.nesting in {"rectify", "strict_rectify"}:
            step = "Checking for nested events"
            self.logger.report_step(step, "In progress...")
            if j == 0:
                # Check for Nested Hybridization
                # This encapsulates the while-loop for recursive sub-fixes
                targets = self.find_missing_targets(select_mt)
                # Sort targets to ensure deterministic order (important for testing/debugging)
                # We could sort by clade size to avoid needing to update future targets as we rename,
                # But this is less intuitive and not always correct.
                targets = sorted(list(targets))
            else:
                # Update targets
                # Update future targets in the list if they were renamed
                suffix = f"{iter}.{j}"
                for k in range(j, len(targets)):
                    future_target = targets[k]
                    # In the rectify case, there's only one Hx node
                    if future_target in suffix_name_map[suffix]:

                        if future_target.startswith("<P*"):
                            new_name = f"{future_target}{iter}|{suffix}"
                        else:
                            new_name = f"{future_target}|{suffix}"

                        targets[k] = new_name
                        self.flow_logger.log(f"Nested Fix: Updated pending target '{future_target}' to '{new_name}'", 'd')
            if targets:
                if j == 0:
                    self.logger.report_step(step, f"Success: found {len(targets)} targets to fix")
                else:
                    self.logger.report_step(step, f"Success: updated targets, {len(targets)-j} left")
            else:
                self.logger.report_step(step, f"Success: no targets detected")
        else:
            targets = []

        step = "Writing handoff files"
        self.logger.report_step(step, "In progress...")

        # Write handoff files for resume support
        CommonOps.export_tree_files(iter_out.parent, select_mt.mt.ete_tree, [gt.ete_tree for gt in next_gts.values()])

        if self.ctx.nesting in {"rectify", "strict_rectify"} and targets:
            success_msg = f"ready for task {i}.{j+1}"
        else:
            success_msg = f"ready for task {i+1}"

        self.logger.report_step(step, f"Success: {success_msg}")
        
        return select_mt, next_gts, targets
    
    # --- Handlers for the Split mode ---
    
    def _partition_gene_trees(self,
                              res: TaskResult, best_mt_idx: int,
                              select_mt: MulTree, copy_gts: bool) -> Tuple[Dict[int, SmrtTree], Dict[int, SmrtTree], Dict[int, List[int]]]:
        """Splits GTs into backbone (Outer) and hybrid clades (Inner)."""
        step = "Partitioning gene trees"
        self.logger.report_step(step, "In progress...")

        if copy_gts: gts = {k: v.copy() for k, v in res.gene_trees.items()}
        else: gts = res.gene_trees

        min_maps = res.kept_mul_maps[best_mt_idx]
        h_copy_map = select_mt.build_h_copy_map() # mt_node -> copy_idx (0 for H1, 1 for H2, etc.)

        outer_gts, inner_gts = {}, {}
        gt_split_dict = defaultdict(list)
        outie_below, innie_below, innie_counter = 0, 0, 0 # Below min leaf count

        debug_sample = self.sample(list(gts.keys()))

        for g_idx, gt_wrapper in gts.items():
            maps = min_maps.get(g_idx)
            if not maps: continue
            source_gt = gt_wrapper.ete_tree

            if g_idx in debug_sample:
                self._debug_tree(f"Pre-split GT {g_idx}:", source_gt)
                self.logger.log(f"Map for GT {g_idx}: {maps.cor}", 'd')

            # --- Map inner/outer leaves by homoeologous copy ---
            gt_leaf_to_copy_idx = {} # Inner leaves: gt_leaf -> copy_idx (0 for H1, 1 for H2, etc.)

            for mt_node, gt_nodes in maps.rev.items():
                if mt_node in h_copy_map:
                    copy_idx = h_copy_map[mt_node]
                    # Flatten the mapping so every GT leaf knows its exact state
                    for gt_n in gt_nodes:
                        gt_leaf_to_copy_idx[gt_n] = copy_idx
            
            # --- Bottom-up purity caching ---
            node_copy_state = {}
            for node in source_gt.traverse("postorder"):
                if node.is_leaf():
                    node_copy_state[node] = gt_leaf_to_copy_idx.get(node.name)
                else:
                    child_states = [node_copy_state[c] for c in node.children]
                    # Internal node is pure ONLY if ALL children map to the EXACT SAME copy
                    if child_states and all(s is not None for s in child_states) and len(set(child_states)) == 1:
                        node_copy_state[node] = child_states[0]
                    else:
                        node_copy_state[node] = None

            # --- Top-down extraction: manual stack allows 'skipping' subtrees ---
            stack = [source_gt]
            h_lineages = []
            while stack:
                node = stack.pop()
                # If copy_idx is an integer (0, 1, 2...), this node is pure for that specific copy
                if node_copy_state.get(node) is not None:
                    if len(node) < self.ctx.min_gt_lvs:
                        innie_below += 1
                    # SUCCESS: We found the largest pure clade for this homoeolog
                    # Nodes are guaranteed to have at least one leaf, and no overlaps
                    # Do NOT add children to the stack; this skips the entire subtree
                    # No need to copy() - pure nodes' children aren't added to stack, so no unsafe nested detach()s
                    h_lineages.append(node.name)
                else:
                    # Node is mixed (e.g., contains H1 and H2, or outer leaves), so we must check its children
                    stack.extend(node.children)

            is_outer, _, extracts = gt_wrapper.trim_lineages(h_lineages, retain=True)

            # --- Outer GTs ---
            if is_outer:
                if len(gt_wrapper) < self.ctx.min_gt_lvs:
                    outie_below += 1
                outer_gts[g_idx] = gt_wrapper
                if g_idx in debug_sample:
                    self._debug_tree(f"Pruned Outer GT {g_idx}:", gt_wrapper.ete_tree)
            else:
                outie_below += 1 # No leaves retained in outer GT since it's empty
                if g_idx in debug_sample:
                    self.logger.log(f"Outer GT {g_idx} is empty after trimming ", 'd')

            # --- Inner GTs ---
            for extracted_gt in extracts:
                inner_gts[innie_counter] = extracted_gt

                if g_idx in debug_sample:
                    self._debug_tree(f"Extracted Inner as Lineage {innie_counter}:", extracted_gt.ete_tree)

                gt_split_dict[g_idx].append(innie_counter)
                innie_counter += 1

        # Apply global cutoff
        # If all GTs of a given subproblem are below the minimum leaf cutoff, we discard them.
        # Previously, we discarded individual GTs based on this, but it biases the reconciliation score because changes to signal balance.
        if innie_below >= len(inner_gts): inner_gts = {}
        if outie_below >= len(outer_gts): outer_gts = {}

        self.logger.report_step(step, f"Success: got {len(outer_gts)} out. & {len(inner_gts)} in. gts")
        return inner_gts, outer_gts, gt_split_dict

    def _partition_species_tree(self, select_mt: MulTree) -> Tuple[SmrtTree, Optional[SmrtTree]]:
        """Safely creates the Inner and Outer Species Trees."""
        step = "Partitioning species tree"
        self.logger.report_step(step, "In progress...")
            
        names_to_trim = [select_mt.h1_node.name] + [n.name for n in select_mt.hx_nodes]
        outer_wrapper, inner_wrapper, _ = select_mt.partition('h1')

        self._debug_tree("Inner Species Tree (Hybrid Clade):", inner_wrapper.ete_tree)

        if outer_wrapper is None:
            self.logger.log(f"Trimming hybrid clades {names_to_trim} from Species Tree resulted in no outer tree.", 'd')
        else:
            self._debug_tree(f"Outer Species Tree (Backbone) after hybrid clades {names_to_trim} trimming:", outer_wrapper.ete_tree)

        self.logger.report_step(step, f"Success: got {len(outer_wrapper) if outer_wrapper else 0} out. & {len(inner_wrapper)} in. st sizes")
        return inner_wrapper, outer_wrapper

    def extract_subproblems(
            self, bin_id: Tuple[int, int], res: TaskResult, events: list,
            iter_out: Path, iter_logger: GranLogger
        ) -> Optional[List[ConcurrTask]]:
        """
        Processes a split worker result using ETE3-safe surgery and O(N) GT extraction.
        1. Inner: Extracts independent 'pure' subtrees for each hybrid lineage.
        2. Outer: Backbone with H1 clade collapsed to a placeholder leaf.
        Returns: List of new sub-tasks or None.
        """
        backup_logger = self.logger
        self.logger = iter_logger

        depth, idx = bin_id
        next_tasks = []
        gt_split_dict_all = {}
        last_mt_idx = events[-1][0]

        # Tracker for Execution Deduplication
        task_signatures = {}

        for mt_idx, branch_mt, new_state_id, _, _ in events:

            copy_gts = (mt_idx != last_mt_idx) # Only deepcopy if this is a new MT to avoid unnecessary copying

            # Partition the Gene Trees into Inner and Outer sets
            inner_gts, outer_gts, gt_split_dict_all[new_state_id] = self._partition_gene_trees(res, mt_idx, branch_mt, copy_gts)

            # Perform topological surgery on the Species Tree
            inner_wrapper, outer_wrapper = self._partition_species_tree(branch_mt)

            step_extract = f"Extracting inferred event subproblems ({new_state_id})"
            self.logger.report_step(step_extract, "In progress...")

            # Define subproblems cleanly: (Label, Wrapper, GTs, Task_Offset)
            subproblems = [
                ("Outer", outer_wrapper, outer_gts, 0),
                ("Inner", inner_wrapper, inner_gts, 1)
            ]

            # Process both Outer and Inner subproblems without redundant code
            for label, wrapper, gts, offset in subproblems:
                if wrapper and len(wrapper) >= self.ctx.min_st_lvs and len(gts) > 0:
                    
                    # Signature: Taxa content + Gene Tree count
                    sig_str = '|'.join(sorted(wrapper.ete_tree.iter_leaf_names())) + '|' + str(len(gts))
                    sig = hash(sig_str)
                    self.logger.log(f"{sig}: {sig_str}", 'd') # Log the signature for debugging

                    task_id = (depth + 1, idx * 2 + offset)
                    
                    if sig in task_signatures:
                        source_state_id = task_signatures[sig]
                        
                        # Create the Virtual Clone in History
                        self.ctx.history[task_id][new_state_id] = {"__symlink__": source_state_id}
                        self.logger.log(f"Optimization: {label} task for {new_state_id} symlinked to identical subproblem {source_state_id}", 'i')
                    else:
                        task_signatures[sig] = new_state_id
                        next_tasks.append((wrapper, gts, task_id, new_state_id))

            self.logger.report_step(step_extract, f"Success: extracted {len(next_tasks)} valid subproblems")

        step_write = "Writing handoff files"
        self.logger.report_step(step_write, "In progress...")

        # Write combined splits
        with open(iter_out.parent / f"gt_splits.json", 'w') as f:
            json.dump(gt_split_dict_all, f, indent=4)

        # Write handoff files for resume support
        task_strs = []
        # Unpack the 4-tuple, ensuring we write to the isolated new_state_id
        for task_st, task_gts, task_id, new_state_id in next_tasks:
            task_str = f"{task_id[0]}.{task_id[1]}"
            task_str_ = f"{new_state_id}->{task_str}"
            task_strs.append(task_str_)
            
            # Isolated directory pathing
            task_out = iter_out.parent / task_str / new_state_id
            task_out.mkdir(parents=True, exist_ok=True)
            CommonOps.export_tree_files(task_out, task_st.ete_tree, [gt.ete_tree for gt in task_gts.values()])

        if task_strs:
            self.logger.report_step(step_write, f"Success: ready for {len(task_strs)} tasks")
            self.logger.log(f"INFO: Tasks generated: {', '.join(task_strs)}", 'i')
        else:
            self.logger.report_step(step_write, "Success")
            self.logger.log("INFO: No tasks generated.", 'i')
            
        self.logger = backup_logger
        return next_tasks

    def process_depth_batch(self, batch_results: list, depth: int, max_select: int) -> List[ConcurrTask]:
        """
        Assimilates a complete batch of worker results, extracts subproblems, 
        and applies Lookahead and Breadth filters highly efficiently.
        """
        next_tasks = []

        def sort_key(item):
            task_id, _, _, _, search_state_id = item
            # Primary Sort (External): Parse "S42" to 42 so S2 sorts before S10
            s_num = int(search_state_id.replace('S', ''))
            # Secondary Sort (Internal): task_id tuple (which naturally sorts the 'j' index)
            return (s_num, task_id)
            
        batch_results.sort(key=sort_key)
        all_results = []
        current_active_states = []
        
        # Save Events
        for task_id, res, _, log_inheritance, state_id in batch_results:
            if not res: continue
            
            min_mt_repr = res.repr_min_mt

            iter_out = self.ctx.get_task_dir(task_id, state_id, is_beam= max_select != 1)
            
            passed_events = self.save_events(task_id[0], task_id[1], res, state_id)
            
            if passed_events:
                for ev in passed_events:
                    current_active_states.append(ev[2]) # new_state_id
            
            all_results.append((task_id, res, iter_out, log_inheritance, passed_events, min_mt_repr))

        # Global Filtration (Lookahead & Breadth)
        self.apply_universe_filters(depth, current_active_states)

        # Log Assimilation and Preparation of Next Tasks
        tracker_nodes = self.tracker.state['nodes']
        for task_id, res, iter_out, log_inheritance, passed_events, min_mt_repr in all_results:

            # Assimilate the worker's log into the main logger
            self.logger.assimilate(log_inheritance.log_file, warnings=log_inheritance.warnings)
            iter_logger = GranLogger(None, self.ctx.verbosity, self.ctx.debug, parent_logger=self.logger, inheritance=log_inheritance)

            if passed_events:
                unblocked_events = [ev for ev in passed_events if not tracker_nodes[ev[2]].get('blocked', False)]
                if unblocked_events:
                    extracts = self.extract_subproblems(task_id, res, unblocked_events, iter_out, iter_logger)
                    if extracts:
                        next_tasks.extend(extracts)

            iter_logger.end_report(*min_mt_repr)

        return next_tasks

    def apply_universe_filters(self, depth: int, current_active_states: List[str]) -> None:

        if not current_active_states: return

        l_val = self.ctx.lookahead
        b_val = self.ctx.breadth_max

        if l_val <= 0 and b_val <= 0: return

        # Optimization: only query the global universes once to execute both filters
        graph_mutated = False
        all_universes = []
        blocked_str = lambda lst: f"branches {', '.join(lst)}" if len(lst) <= 5 else f"branch {len(lst)} states"

        # Lookahead Filtration
        if l_val > 0 and depth >= l_val:
            all_universes, _ = self.navigator.compute_all_universes(start_node="S0")
            valid_universes = [u for u in all_universes if u[0] != float('inf')]
            
            # If no valid universes, quit this filter
            target_prune_depth = depth - l_val if valid_universes else 0
            
            if target_prune_depth > 0:
                nodes = self.tracker.state['nodes']
                prune_candidates = defaultdict(list)
                blocked = []
                
                # Group nodes at target prune depth by their parent
                for nid, ndata in nodes.items():
                    task = self.navigator._parse_task(ndata['task'])
                    if task[0] == target_prune_depth:
                        p_id = ndata['parent']
                        if p_id: prune_candidates[p_id].append(nid)

                # Pre-compute the best score for every node in O(Universes) time
                node_best_scores = defaultdict(lambda: float('inf'))
                for score, nodes_set, _ in valid_universes:
                    for node in nodes_set:
                        if score < node_best_scores[node]:
                            node_best_scores[node] = score
                            
                for p_id, children in prune_candidates.items():
                    if len(children) <= 1: continue # No sibling choices to prune

                    # O(Children) lookup instead of O(Children * Universes)
                    best_child = min(children, key=lambda c: node_best_scores[c])
                    
                    if node_best_scores[best_child] != float('inf'):
                        for child_id in children:
                            if child_id != best_child:
                                bad_descendants = self.tracker.block_branch(child_id)
                                blocked.append(child_id)

                if blocked:
                    graph_mutated = True
                    self.logger.log(f"Lookahead ({l_val}): Blocked sub-optimal {blocked_str(blocked)} originating from depth {target_prune_depth}", 'i')

        # Max Breadth Filtration
        if b_val > 0:
            if graph_mutated or not all_universes:
                all_universes, _ = self.navigator.compute_all_universes(start_node="S0")
                
            valid_universes = [u for u in all_universes if u[0] != float('inf')]
            
            if len(valid_universes) > b_val:
                top_universes = valid_universes[:b_val]
                allowed_active_nodes = set()
                for score, nodes_set, _ in top_universes:
                    allowed_active_nodes.update(nodes_set)
                blocked = []

                for state_id in current_active_states:
                    if state_id not in allowed_active_nodes:
                        bad_descendants = self.tracker.block_branch(state_id)
                        blocked.append(state_id)
                        
                if blocked:
                    graph_mutated = True
                    self.logger.log(f"Breadth ({b_val}): Blocked {blocked_str(blocked)} from depth {depth+1} (capping at {b_val} top branches)", 'i')

    # -- Recombination Logic for the Split mode ---

    def get_bulk_state_caches(self, root_task_id: Tuple[int, int], unique_states: Set[str]) -> Dict[str, SmrtTree]:
        """Silently glues state caches in bulk using pre-computed math."""
        histories = self.navigator.get_bulk_histories(unique_states)
        trees = {}
        
        # Suspend logger output for the entire batch to keep the console clean
        with self.logger.silenced(True):
            for s_id, (univ_hist, desc) in histories.items():
                trees[s_id] = self._iterative_glue(root_task_id, univ_hist)
                
        return trees

    def glue_split_results(self, root_id: Tuple[int, int] = (0, 0), is_silent: bool = False) -> SmrtTree:
        """
        Recombines results by recursively diving to the innermost subproblems.
        """
        with self.logger.silenced(is_silent):
            self.logger.title_banner("Recombining Split Results")
            self.logger.log("Merging subproblem trees...", 'i')

            self.best_history, _ = self.navigator.get_multiverse_history()

            ft_wrapper = self._iterative_glue(root_id)
            
            self.logger.log("Success: All subproblems merged successfully.", 's')
            return ft_wrapper

    def _iterative_glue(self, root_task_id: Tuple[int, int], history: Dict = None) -> SmrtTree:
        """
        Recombines split results using history 'trackers' to identify graft targets.
        Returns:
            the final merged tree for the given root task ID, or;
            the original input tree if the root task was rejected.
        """
        # Stack stores tuples: (task_id, children_visited_flag)
        stack = [(root_task_id, False)]
        results = {}

        history = self.best_history if history is None else history

        while stack:
            task_id, visited = stack.pop()
            
            # Base Case: If this task was never run or didn't pass, 
            # we return None or the input tree.
            if task_id not in history:
                self.logger.log(f"Glue {task_id}: Task {task_id} not found in history.", 'd')
                results[task_id] = None
                continue

            event = history[task_id]

            # Check Pass/Fail
            if not event['passed']:
                self.logger.log(f"Glue {task_id}: Task {task_id} did not pass cutoff.", 'd')
                results[task_id] = None
                continue

            # --- Dive to children (Post-order traversal) ---
            depth, idx = task_id
            outer_id = (depth + 1, idx * 2)
            inner_id = (depth + 1, idx * 2 + 1)
            uid = 2**depth + idx - 1
            first_uid_of_depth = 2**depth - 1

            if not visited:
                # Push back current node marked as visited
                # Push children. Inner pushed first so Outer is processed first (LIFO order matches original)
                stack.extend([(task_id, True), (inner_id, False), (outer_id, False)])
                continue

            self.logger.log(f"--- Processing Task {task_id} ---", 'd')

            # --- Load Current Tree and Graft Locations for Terminal Subproblem ---

            # Uses the 'best_mt' with format=1
            current_mt = MulTree.from_history_event(event)
            locs = event['h_locs']

            # --- Initialize Graft Records ---
            all_h_nodes = [current_mt.h1_node] + current_mt.hx_nodes
            records: List[GraftRecord] = []
            for i, h_node in enumerate(all_h_nodes):
                up = h_node.up
                up_up = up.up if up else None
                sisters = up.get_sisters() if up else []
                records.append(GraftRecord(
                    copy_id=i,
                    original=locs[i],
                    corrected=locs[i],
                    parent=up.name if up else '<root>',
                    grandp=up_up.name if up_up else '<root>',
                    aunt=sisters[0].name if sisters else '<none>'
                ))
            self.logger.log(f"Glue {task_id} Initial Records: {[str(r) for r in records]}", 'd')

            # --- Retrieve the results of the children or infer them if missing ---

            # Post-traversal evaluation for current node & cleanup memory for children
            outer_wrapper = results.pop(outer_id, None)
            inner_wrapper = results.pop(inner_id, None)

            if not outer_wrapper and not inner_wrapper:
                self.logger.log(f"Glue {task_id}: No Outer and Inner results from {outer_id} and {inner_id}. Returning Current tree.", 'd')
                current_mt.rename_marked_nodes(uid, skip_p_tag=False)
                results[task_id] = current_mt.mt
                continue

            # Partition, get actual trimmed names, and consume (invalidate) the MulTree obj
            outer_wrapper_curr, inner_wrapper_curr, trimmed_names = current_mt.partition('h1')
            self.logger.log(f"Glue {task_id}: Partitioned current tree. Trimmed names: {trimmed_names}", 'd')

            if not inner_wrapper:
                # Infer inner tree from current tree (best_mt) if missing, since we know the hybrid clade is there
                inner_tree = inner_wrapper_curr.ete_tree
                self.logger.log(f"Glue {task_id}: No Inner results from {inner_id}. Retrieved from Current tree by partitioning H1_node.", 'd')
            else:
                inner_tree = inner_wrapper.ete_tree

            self._debug_tree(f"Inner Result Tree for Task {task_id}:", inner_tree, other_attr=['H', 'pure'])

            if not outer_wrapper:
                outer_wrapper = outer_wrapper_curr
                self.logger.log(f"Glue {task_id}: No Outer results from {outer_id}. Retrieved from Current tree by removing all H clades.", 'd')
            else:
                outer_wrapper_curr.destroy()
            self._debug_tree(f"Outer Result Tree for Task {task_id}:", outer_wrapper.ete_tree, other_attr=['H', 'pure'])

            self.logger.log(f"Glue {task_id}: Subproblems retrieved.", 'd')

            # --- Location Expansion and Grafting Logic using Records ---
            records = self._prepare_graft_records(task_id, outer_wrapper, records, first_uid_of_depth)

            # --- Execute Grafts ---
            outer_wrapper.graft_records(inner_tree, records, uid)
            self._debug_tree(f"Post-Graft Tree for Task {task_id}:", outer_wrapper.ete_tree, other_attr=['H', 'pure'])
            results[task_id] = outer_wrapper

        if results.get(root_task_id) is None:
            # Fallback to original ST
            self.logger.log("No valid recombination found. Returning the original ST.", 'i')
            # Not finding root in history SHOULD raise an error!
            return SmrtTree(tree_obj=Tree(history[root_task_id]['in_tree'], format=1))
        return results[root_task_id]

    def _prepare_graft_records(self, task_id: Tuple[int, int], outer_wrapper: SmrtTree, records: List[GraftRecord], first_uid_of_depth: int) -> List[GraftRecord]:
        """
        Targets need to be corrected for some special cases (e.g., source_wgd, target_wgd, source_ils, target_ils), and also expended to previously split events.
        We can't account for WGD in the source ahead of fetching the subtrees, because of modes like 'split+model from mt' or 'mixed+model',
        where the event may contain some WGD and some "appearant" allopolyploidy, due the the full mode logic.
        """
        
        # --- Case: source_wgd ---

        # if p_i == p_j -> loc_i = p_i.sis, p_i = p_i.up (i.e., autopolyploidy - we graft h1 to the sister of the orignal parent,
        # and then, h2 to h1, which is now in the tree as well)
        p_groups = defaultdict(list)
        for rec in records:
            p_groups[rec.parent].append(rec)
            
        for p_name, group in p_groups.items():
            if len(group) > 1:
                # Find the primary locus safely
                self.logger.log(f"Glue {task_id}: Found duplicate parent '{p_name}', indicating source WGD", 'd')
                assert len(group) == 2, f"Expected exactly 2 records for parent '{p_name}' in source WGD case, found {len(group)}"
                # H1 (clean_rec) has loc marked with *
                clean_rec = next((r for r in group if '*' in r.original), None)
                clean_rec.parent = clean_rec.grandp
                clean_rec.original = clean_rec.aunt
                clean_rec.corrected = clean_rec.aunt

        # --- Case: source_ils ---

        # if p_i == loc_j -> loc_j = loc_i (i.e., we graft both p_i and p_j to the same location loc - the original one)
        # Must be done after source_wgd correction, because WGD correction can create new parent matches that indicate ILS
        for rec in records:
            ils_matches = [r for r in records if r.parent == rec.corrected]
            if ils_matches:
                self.logger.log(f"Glue {task_id}: Found parent match for loc '{rec.corrected}', indicating source ILS", 'd')
                rec.corrected = ils_matches[0].original
            # Invalidate grandp and aunt just in case
            rec.grandp = '<invalid>'
            rec.aunt = '<invalid>'

        # --- Case: target_wgd & target_ils Crawling ---
        def _crawl_up(loc_node: TreeNode, query_name: str) -> TreeNode:
            """
            Crawls up the tree from a specific node to handle target WGD and ILS.
            Returns the highest valid parent TreeNode.
            """
            # E.g., if y is a loc in (y.1, y.2); or ((a, y.1), y.2);
            # In both cases, we want to graft only above all y's, because if more than one y is in the tree, it's either:
            # [a.] an event in parallel [split mode], so it happened **after** the current event (in which case, we crawl up)
            # [b.] an event from the original input or full mode [in mixed] (here, we mustn't crawl, becasue we want to expand them later)
            # Thus, [a.] has to be done before expansion, because then we won't get redundant locations!
            # Because, we only crawl in [a.], we can distinguish it from [b.] by checking the p depth of y.2, hence these 2 main conditions:
            # [1.] loc.up is an old p (depth < first_uid_of_depth) [distinguish]
            # [2.] loc.up's other child is either == loc [case t_wgd] or == one of loc's children [case t_ils], by .pure
            # Then, we crawl up. This needs to be done until conditions are not met!
            # We check against first_uid_of_depth, because depths have the form, e.g.: [full] 0, 1, 2, [switch to split] 3-6, 7-14...
            # and we break the crawl if the depth is below current
            current_node = loc_node
            current_name = query_name
            
            while True:
                parent = current_node.up
                if not parent: break # Nowhere to crawl up to - root reached
                
                parent_name = parent.name
                left_side = parent_name.split('>', 1)[0].split('|', 1)[0]
                try: left_side = left_side.split('<P')[1]
                except IndexError: break # Not a parent tag
                
                if int(left_side) < first_uid_of_depth:
                    break # Reached historical depth, stop crawling
                
                opts = [current_node.pure] # target_wgd case
                opts += [c.pure for c in current_node.children if not current_node.is_leaf()] # target_ils case
                loc_sis = outer_wrapper.get_sis(current_node).pure
                
                if loc_sis in opts:
                    self.logger.log(f"Glue {task_id}: Target WGD/ILS found for '{current_name}' by parent's sister '{loc_sis}'. Crawling to '{parent_name}'.", 'd')
                    current_node = parent
                    current_name = parent_name
                else:
                    break # Correction complete
                    
            return current_node

        # --- Expand Locations ---
        for rec in records:
            # Important: we look for the node name, not node.pure!
            # If a lookup of node.pure is needed, it means there's some bug in SmrtTree.graft_records() or downstream from it
            loc_node = outer_wrapper.get_node(rec.corrected)
            if loc_node:
                pure_loc = loc_node.pure
                corrected = set()
                potential = outer_wrapper.match(pure_loc)
                for exp in potential:
                    corrected.add( _crawl_up(exp, exp.name) )
                rec.expanded_targets = list(corrected)
                rec.corrected = pure_loc
                self.logger.log(f"Glue {task_id}: Expanded corrected location '{rec.corrected}' to {[n.name for n in rec.expanded_targets]}.", 'd')
            else:
                # Mark as delayed WGD if missing from outer tree
                rec.expanded_targets = []
                self.logger.log(f"Glue {task_id}: Corrected location '{rec.corrected}' not found in Outer tree & marked as delayed WGD expansion.", 'd')

        self.logger.log(f"Glue {task_id}: Resolved graft locations with corrections: {[str(r) for r in records]}", 'd')

        return records

    # --- Fast-forwarding logic for Split mode ---

    def fast_forward_split(self, current_tasks: List[ConcurrTask]) -> List[ConcurrTask]:
        """
        Reconstructs the task queue from disk state based on history.
        Uses BFS to traverse solved nodes and identify the frontier.
        """

        self.logger.log("Fast-forwarding Split tasks based on history...", 'i')
        self.logger.log(f"Current tasks: {current_tasks}, output_dir: {self.ctx.root_dir}", 'd')

        # queue contains Tuple[Current_Task_ID_Tuple, Parent_Task_ID_Tuple]
        # Start with Root ID (0, 0) and no parent
        queue = [(current_tasks[0][2], None)]
        real_tasks = []
        
        while queue:
            q = queue.pop(0)
            nid = q[0]
            depth, idx = nid
            nid_str = f"{depth}.{idx}"
            
            # Check if this task is already solved in history
            if nid in self.ctx.history:
                # Task done. Check for its children directories on disk
                c1 = (depth + 1, idx * 2)
                c2 = (depth + 1, idx * 2 + 1)

                c1_str = f"{c1[0]}.{c1[1]}"
                c2_str = f"{c2[0]}.{c2[1]}"

                self.logger.log(f"Task {nid_str} done. Checking children: {c1_str}, {c2_str} in {self.ctx.root_dir / nid_str}", 'd')
                
                # If child dir exists, add to traversal queue
                if (self.ctx.root_dir / nid_str / c1_str / "multree.tre").exists():
                    queue.append((c1, nid))
                if (self.ctx.root_dir / nid_str / c2_str / "multree.tre").exists():
                    queue.append((c2, nid))
            else:
                # Task NOT in history -> It is a frontier task to run.
                # Load inputs from disk
                p_nid = q[1]
                p_nid_str = f"{p_nid[0]}.{p_nid[1]}" if p_nid else "None"
                self.logger.log(f"Queueing frontier task: {nid_str} (child of {p_nid_str})", 'd')

                st_path = self.ctx.root_dir / p_nid_str / nid_str / "multree.tre"
                gt_path = self.ctx.root_dir / p_nid_str / nid_str / "genetrees.txt"
                
                self.logger.log(f"Looking for ST at {st_path}, GTs at {gt_path}", 'd')

                if st_path.exists() and gt_path.exists():
                    real_tasks.append((st_path, gt_path, nid))
                    self.logger.log(f"Resuming sub-problem: {nid_str}", 's')
                else:
                    # Fallback: if root is missing inputs but passed in current_tasks
                    if nid == str(current_tasks[0][2]):
                        self.logger.log(f"Using provided inputs for root task {nid_str}, {current_tasks}", 's')
                        real_tasks.extend(current_tasks)
                    else:
                        self.logger.log(f"Missing inputs for resume task {nid_str}. Skipping.", 'w')

        self.logger.log(f"Fast-forwarded tasks: {real_tasks}", 'd')

        return real_tasks
    
    def fast_forward_split_ng(self, current_tasks: List[ConcurrTask]) -> List[ConcurrTask]:
        nid, depth, idx = None, None, None
        while 1:
        # ... [keep top part] ...
            # Check if this task is already solved in history
            if nid in self.ctx.history:
                passed = False
                for state_id, events in self.ctx.history[nid].items():
                    for child_id, event_data in events.items():
                        if child_id != "In" and event_data.get('passed', False):
                            passed = True
                            break
                    if passed: break
                    
                if passed:
                    # Task done. Check for its children directories on disk
                    c1 = (depth + 1, idx * 2)
                    c2 = (depth + 1, idx * 2 + 1)
        # ... [keep the rest] ...

    # --- Output methods ---

    def _infer_switch(self):
        mixed_switch = self.ctx.mixed_switch
        if mixed_switch == 0:
            if self.mode == 'split':
                mixed_switch = 0
            # Other iterative modes shouldn't have mixed_switch == 0
            elif self.mode == 'full':
                mixed_switch = float('inf')
            elif self.mode == 'mixed':
                self.logger.log(f'Unexpected mixed_switch value of 0 for mode {self.mode}.', 'e')
        return mixed_switch

    def create_problem_tree_ascii(self) -> str:
        """
        Reconstructs the execution history into a visual ETE3 ASCII tree.
        Shows pass/fail status and correctly roots orphaned tasks.
        """
        if not self.ctx.history:
            return "No tasks recorded."
        
        if self.best_history is None:
            self.best_history, _ = self.navigator.get_multiverse_history()

        # Create a virtual super-root to hold the (0,0) start
        super_root = TreeNode(name="<RUN_START>")
        nodes = {}
        last_task_at_depth = {}

        mixed_switch = self._infer_switch()

        sorted_tasks = sorted(self.best_history.keys(), key=lambda x: (x[0], x[1]))

        for task_id in sorted_tasks:
            depth, idx = task_id

            passed = self.best_history[task_id].get('passed', False)
            status = "PASS" if passed else "FAIL"
            
            node = TreeNode(name=f"({depth},{idx})[{status}]")
            nodes[task_id] = node

            if depth == 0 and idx == 0:
                super_root.add_child(node)
            else:
                # --- Regime Transition Logic ---
                if depth < mixed_switch:
                    # Full Mode Regime (Including Nested Fixes)
                    if idx > 0:
                        # Nested fix: Attach to the immediately preceding task at the SAME depth
                        p_id = last_task_at_depth.get(depth)
                    else:
                        # First task at depth: Attach to the final task of the PREVIOUS depth
                        p_id = last_task_at_depth.get(depth - 1)
                        
                elif depth == mixed_switch:
                    # Regime Transition: The Split binary fan-out originates from the final Full mode task
                    p_id = last_task_at_depth.get(depth - 1)
                    
                else:
                    # Pure Split Mode Regime (Binary Branching)
                    p_id = (depth - 1, idx // 2)
                    
                # Safely attach to parent
                if p_id in nodes:
                    nodes[p_id].add_child(node)
                else:
                    super_root.add_child(node) # Fallback for malformed history

            # Update the tracker so the next task knows exactly what just finished!
            last_task_at_depth[depth] = task_id

        # Apply stylistic formatting to internal nodes
        for node in super_root.iter_descendants():
            if not node.is_leaf():
                if len(node.children) == 1:
                    node.name = f"-{node.name}-"
                else:
                    node.name = f"-{node.name}- "
        node_zero = super_root.children[0]
        if not node_zero.is_leaf():
            node_zero.name = node_zero.name[1:]

        # Return string without the first '\n' padding character
        return super_root.get_ascii(show_internal=True)[1:]

    def plot_metrics(self):
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator, FixedLocator
        from matplotlib.lines import Line2D
        import numpy as np

        # --- 1. Data Parsing & Grouping ---
        def get_taxa_count(tree_str):
            try:
                t_str = CommonOps._fix_semicolon(tree_str)
                return len(Tree(t_str, format=1))
            except:
                return 0

        # Data container: grouped_history[i] = list of (j, data_dict)
        # Split Mode: i=depth, j=index
        # Full Mode: i=iteration, j=nested_id
        grouped_history = defaultdict(list)

        if not self.best_history:
            self.best_history, _ = self.navigator.get_multiverse_history()
        
        for key, val in self.best_history.items():
            i, j = key
            
            # --- Metrics Calculation ---
            out_score = val['score']
            if out_score == 'N/A': continue # Skip if no non-input tree existed
            in_score = val['in_score']

            h_lvs = len(val['h_leaves'])
            h_copies = len(val['h_locs'])
            out_taxa = get_taxa_count(val['mt_tree'])
            in_taxa = max(0, out_taxa - h_lvs*(h_copies-1 if h_copies > 0 else 0)) # Adjust for hybrid leaves that don't add taxa
            assert in_taxa == get_taxa_count(val['in_tree']), f"Taxa count mismatch for input tree at {key}"
            
            out_norm = out_score / out_taxa if out_taxa > 0 else 0
            in_norm = in_score / in_taxa if in_taxa > 0 else 0

            entry = {
                'key': key,
                'in':  {'score': in_score,  'taxa': in_taxa,  'norm': in_norm},
                'out': {'score': out_score, 'taxa': out_taxa, 'norm': out_norm}
            }
            grouped_history[i].append((j, entry))

        if not grouped_history:
            self.logger.log("No history data available to plot.", 'w')
            return

        sorted_iters = sorted(grouped_history.keys())

        # --- 2. Setup Plot ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        metrics = [
            ('score', 'MP Score', 'MP Score (Input vs Inferred)'), 
            ('taxa', 'Taxa Count', 'Taxa Count'), 
            ('norm', 'MP Score / Taxa', 'Normalized Score')
        ]

        # --- 3. Drawing Logic ---
        for ax, (m_key, y_label, title) in zip(axes, metrics):
            
            # Coordinate Cache for connections: Key -> {'in': (x,y), 'out': (x,y)}
            # Key is (depth, idx) for split, or (iter, sub) for full
            coord_map = {}
            
            # Track previous bin's last output for Full mode sequential connections
            last_bin_coords = None 

            for i in sorted_iters:
                events = sorted(grouped_history[i], key=lambda x: x[0])
                n_events = len(events)
                
                # Intra-bin connection tracker
                prev_sub_x, prev_sub_y = None, None
                
                # Bin Logic
                bin_width = 0.8

                for k, (j, data) in enumerate(events):
                    # X Coordinates
                    if self.mode == 'split':
                        # Split Mode: No spreading. All events at this depth share the same center.
                        # The "2 points" (slope) width is fixed.
                        slot_center = i
                        slot_width = bin_width
                    else:
                        # Full Mode: Spread events across the bin width to show nesting order
                        slot_width = bin_width / n_events
                        slot_center = (i - bin_width/2) + (slot_width * (k + 0.5))
                    
                    half_line = (slot_width * 0.8) / 2
                    x_in = slot_center - half_line
                    x_out = slot_center + half_line
                    
                    y_in = data['in'][m_key]
                    y_out = data['out'][m_key]
                    
                    # Store exact coords for later connection logic
                    coord_map[(i, j)] = {'in': (x_in, y_in), 'out': (x_out, y_out)}

                    # Marker Style
                    is_nested = (self.mode != 'split' and j > 0)
                    marker_shape = 'D' if is_nested else 'o'
                    marker_size = 40 if not is_nested else 30

                    # Plot Slope & Points
                    ax.plot([x_in, x_out], [y_in, y_out], color='black', linewidth=1, zorder=3)
                    ax.scatter(x_in, y_in, facecolors='white', edgecolors='black', marker=marker_shape, s=marker_size, zorder=4)
                    ax.scatter(x_out, y_out, color='black', marker=marker_shape, s=marker_size, zorder=4)

                    # --- Connections (Immediate) ---
                    if self.mode != 'split':
                        # Full Mode Intra-bin: Connect Nested to previous in same bin
                        if prev_sub_x is not None:
                            ax.plot([prev_sub_x, x_in], [prev_sub_y, y_in], color='black', linestyle=':', linewidth=1, zorder=2)
                        # Full Mode Inter-bin: Connect 1st event to last event of prev bin
                        elif last_bin_coords is not None:
                            ax.plot([last_bin_coords[0], x_in], [last_bin_coords[1], y_in], color='black', linestyle=':', linewidth=1, zorder=1)

                    prev_sub_x, prev_sub_y = x_out, y_out

                # End of Bin Loop
                last_bin_coords = (prev_sub_x, prev_sub_y)

            # --- Connections (Post-Plotting for Split Mode) ---
            if self.mode == 'split':
                for (depth, idx), curr in coord_map.items():
                    if depth > 0:
                        # Parent Key Logic: Depth-1, Index//2
                        parent_key = (depth - 1, idx // 2)
                        if parent_key in coord_map:
                            prev = coord_map[parent_key]
                            # Connect Parent OUT to Child IN
                            ax.plot([prev['out'][0], curr['in'][0]], 
                                    [prev['out'][1], curr['in'][1]], 
                                    color='black', linestyle=':', linewidth=1, zorder=2)

            # --- 4. Styling & Grid ---
            ax.set_title(title)
            ax.set_ylabel(y_label)
            if self.mode == 'split': ax.set_xlabel('Recursion Depth')
            else: ax.set_xlabel('Iteration Number')

            x_min, x_max = ax.get_xlim()
            x_integers = np.arange(np.floor(x_min), np.ceil(x_max) + 1, 1)
            ax.xaxis.set_major_locator(FixedLocator(x_integers))
            ax.xaxis.set_minor_locator(FixedLocator(x_integers + 0.5))
            
            ax.grid(visible=True, which='minor', axis='x', linestyle='-', alpha=0.5)
            ax.grid(visible=True, which='major', axis='y', linestyle='-', alpha=0.3)
            ax.grid(visible=False, which='major', axis='x')
            
            # --- 5. Legend ---
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label='Input', markerfacecolor='w', markeredgecolor='black', markersize=6),
                Line2D([0], [0], marker='o', color='w', label='Inferred', markerfacecolor='black', markeredgecolor='black', markersize=6),
            ]
            if self.mode != 'split':
                legend_elements.append(
                    Line2D([0], [0], marker='D', color='w', label='Nested fix', markerfacecolor='grey', markeredgecolor='grey', markersize=4)
                )
            
            if m_key == 'score': 
                ax.legend(handles=legend_elements, loc='best', fontsize='small')

        plt.tight_layout()
        output_file = self.ctx.root_dir / 'metrics_plot.png'
        plt.savefig(output_file, dpi=600)
        #self.logger.log(f'Plot saved to {output_file}', 'i')
        plt.close()

    def _debug_tree(self, title: str, ete_tree: Tree, key='d', other_attr=[]) -> None:
        self.logger.log(f"{title}", key)
        self.logger.log(ete_tree.get_ascii(show_internal=True, attributes=['name'] + other_attr), key)
