import webbrowser
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from ete3 import Tree, TreeNode
from pyvis.network import Network
from collections import defaultdict, Counter

# Convension for this class:
# _func() for private methods, may or may not be static
# func() for public methods, therefore we should try to make them static or classmethods [not to modify the instance state]

class ReticulateTree:

    def __init__(self, input_data, is_multree=False, threshold=None, normalize=None):
        self.tree = None
        self.tree_str = None
        self.dag = None
        self.retnodes = None
        self.threshold = threshold
        self.normalize = normalize

        input_type = self._determine_input_type(input_data)

        if input_type == 'str:newick':
            self.tree_str = input_data
            self.tree = Tree(input_data, format=1)
            self._init_from_tree(is_multree)

        elif input_type == 'ete3tree':
            self.tree = input_data
            self.tree_str = input_data.write(format=1)
            self._init_from_tree(is_multree)

        elif input_type == 'str:enewick':
            self._init_from_dag(
                self.__class__.enewick_to_dag(input_data)
            )

        elif input_type.endswith('nexml'):
            raise NotImplementedError('NeXML input is not yet supported for ReticulateTree initialization.')
            #dag = self._parse_nexml_to_dag(input_data, input_type.split(':')[0])
            #self._init_from_dag(dag)

        elif input_type == 'nxdag':
            self._init_from_dag(input_data)

        # XOR None check to ensure correct param tracking
        if (self.threshold is None) ^ (self.normalize is None):
            print('Warning: Under the current settings, MUL-tree folding parameters are not used.')
            self.threshold = None
            self.normalize = None
        
        self.retnodes = self.get_reticulation_nodes()

    def _init_from_tree(self, is_multree):
        if is_multree or self.check_duplicated():
            if self.threshold is not None or self.normalize is not None:
                # Polyphest's multree folding defaults
                if self.threshold is None:
                    self.threshold = 0.2
                if self.normalize is None:
                    self.normalize = True
                print("Note: Polyphest's multree folding does NOT guarantee DAG back-translatability to the original (MUL-)tree!")
                self.dag = self.__class__.polyphest_multree_to_dag(self.tree, threshold=self.threshold, normalize=self.normalize)
            else:
                self.dag = self.__class__.multree_to_dag(self.tree, strict=True)
        else:
            self.dag = self.tree_to_dag(self.tree)

    def _init_from_dag(self, dag):
        self.dag = dag
        self.tree = self.__class__.dag_to_multree(self.dag, match_dag=True)
        self.tree_str = self.tree.write(format=1)
        
    def check_duplicated(self):
        ''' Check if the tree has duplicated labels (i.e., multiple leaves with the same name). '''
        return any(v > 1 for v in self.get_leaf_counts().values())
  
    @staticmethod
    def _determine_input_type(input_data):
        '''
        Determine the type of input data provided to ReticulateTree.
        Returns a string indicating the type:
        - 'str:newick' for Newick strings
        - 'str:enewick' for enriched Newick strings with #H markers
        - 'str:nexml' for NeXML strings
        - 'ete3tree' for ETE3 Tree objects
        - 'nxdag' for NetworkX directed acyclic graphs (DAGs)
        - 'file:nexml' for file-like objects (assumed to be NeXML)
        '''
        input_type = None
        if isinstance(input_data, str):
            input_str = input_data.strip()
            # Can be Newick, enriched Newick, or NeXML - check format
            if input_str.endswith('nexml>'):
                input_type = 'str:nexml'
                print('Warning: NeXML input is only supported for networks!')
            else:
                if not input_str.endswith(';'):
                    raise ValueError('String format unrecognized. Must end with ";" for Newick / enriched Newick, or "nexml>" for NeXML.')
                else:
                    if '#H' in input_str or '//' in input_str:
                        input_type = 'str:enewick'
                    else:
                        input_type = 'str:newick'
        elif isinstance(input_data, Tree):
            input_type = 'ete3tree'
        elif isinstance(input_data, nx.DiGraph):
            input_type = 'nxdag'
        elif hasattr(input_data, 'read'):
            # Assume file-like object
            input_type = 'file:nexml'  # Default to NeXML for file-like objects
            print('Warning: NeXML input is only supported for networks!')
        else:
            raise ValueError(f'Unsupported input type: {type(input_data)}')
        return input_type

    @staticmethod
    def get_dag_root(G):
        ''' Get the root of a directed acyclic graph (DAG). '''
        roots = [n for n, d in G.in_degree() if d == 0]
        if len(roots) != 1:
            raise ValueError('Expected exactly one root.')
        return roots[0]

    @staticmethod
    def compute_depths(G):
        ''' Compute DAG depths to mimic layered layout. '''
        root = ReticulateTree.get_dag_root(G)
        depths = {node: float('-inf') for node in G.nodes}
        depths[root] = 0
        # Topological sort guarantees that all parents are visited before children
        for node in nx.topological_sort(G):
            for child in G.successors(node):
                proposed_depth = depths[node] + 1
                if proposed_depth > depths[child]:
                    depths[child] = proposed_depth
        return depths

    @staticmethod
    def is_reticulation_node(G, node):
        ''' Check if a DAG node is a reticulation node. '''
        return G.in_degree(node) > 1

    # ───── MODULE: From simple tree to network ─────

    @staticmethod
    def tree_to_dag(t):
        '''
        Convert ete3 singly-labeled tree to networkx DAG (no reticulations)
        '''
        G = nx.DiGraph()
        for node in t.traverse('postorder'):
            nid = id(node)
            G.add_node(
                nid,
                ete=node,         # original ete3 node
                label=node.name,  # display label # node.name if node.is_leaf() else str(nid)
            )
            if node.up:
                if hasattr(node, 'dist'):
                    G.add_edge(id(node.up), nid, length=node.dist)
                else:
                    G.add_edge(id(node.up), nid)
        return G

    # ───── MODULE: From MUL-tree to network ─────

    @staticmethod
    def _compute_height_map(root):
        height_map = defaultdict(list)
        heights = {}  # only used inside this loop to compute parent's height

        for node in root.traverse('postorder'):
            if node.is_leaf():
                h = 0
            else:
                h = max(heights[child] for child in node.children) + 1
            heights[node] = h
            height_map[h].append(node)

        return height_map
    
    @staticmethod
    def _compute_sorted_multisets(root):
        '''
        Returns a dict: {node: tuple(sorted multiset items)} for each node.
        '''
        sorted_multisets = {}
        multisets = {}

        for node in root.traverse('postorder'):
            if node.is_leaf():
                mset = Counter([node.name])
            else:
                mset = Counter()
                for child in node.children:
                    mset += multisets[child]

            # Store the Counter for children use, but also store sorted tuple for output
            multisets[node] = mset
            sorted_multisets[node] = tuple(sorted(mset.items()))

        return sorted_multisets

    @staticmethod
    def _compute_canonical_forms(root):
        canonical_map = {}

        def recurse(node):
            '''
            Recursively compute a canonical string representation of a subtree rooted at `node`,
            such that isomorphic subtrees (with possibly duplicated leaves) get the same string.
            '''
            if node.is_leaf():
                canon = node.name
            else:
                child_encodings = sorted([recurse(c) for c in node.children])
                canon = '(' + ','.join(child_encodings) + ')'
            canonical_map[node] = canon
            return canon

        recurse(root)
        return canonical_map
    
    @staticmethod
    def _insert_reticulation(G, src, tgt):
        u = f'retic_{src}_{tgt}'
        G.add_node(u, label='Ret.')
        # Keep the length attribute if present
        old_edge = G[src][tgt]
        G.remove_edge(src, tgt)
        if 'length' in old_edge:
            G.add_edge(src, u, length=old_edge['length'])
        else:
            G.add_edge(src, u)
        G.add_edge(u, tgt)
        return u

    @staticmethod
    def _simplify_redundant_reticulations(G):
        to_remove = []
        for node in list(G.nodes):
            preds = list(G.predecessors(node))
            succs = list(G.successors(node))

            # Here we can't use `is_reticulation_node` because it checks in-degree > 1,
            # which is still not the case for these temporary reticulation nodes (to be removed).
            if 'ete' not in G.nodes[node] and len(preds) == 1 and len(succs) == 1:
                parent, child = preds[0], succs[0]
                if not G.has_edge(parent, child):
                    G.add_edge(parent, child)
                to_remove.append(node)

        G.remove_nodes_from(to_remove)

    @staticmethod
    def _get_single_child(G, retnode):
        children = list(G.successors(retnode))
        x = len(children)
        if x != 1:
            raise ValueError(f'Reticulation node {retnode} should have exactly one child, found {x}')
        return children[0]

    @classmethod
    def multree_to_dag(cls, tree, strict=True):
        '''
        An implementation of the HOLM2006 algorithm to convert an ete3 multree to a networkx DAG phylogenetic network.
        This method should NOT modify the original tree.

        Optimizations:
        - Group nodes at each height by canonical form (or multiset) instead of doing O(k^2) pairwise comparisons.
        - Cache sorted multiset representations used for ordering.
        '''
        # Compute heights and multisets (same as before)
        height_map = cls._compute_height_map(tree)

        # Create the initial networkx graph
        G = cls.tree_to_dag(tree)

        # Prepare canonical map or use multisets, depending on strict flag
        if strict:
            # Expect compute_canonical_forms to return a dict: {node: canon_str}
            canonical_map = cls._compute_canonical_forms(tree)
            def key_fn(n):
                return canonical_map[n]
        else:
            # Precompute a stable, hashable representation for each node's multiset
            sorted_multisets = cls._compute_sorted_multisets(tree)
            def key_fn(n):
                return sorted_multisets[n]

        # Process heights from root down to leaves (same order as before)
        for h in sorted(height_map.keys(), reverse=True):
            nodes_at_height = height_map[h]
            # Sort nodes in a stable way using the same key as original code used for ordering.
            # (Original sorted by sorted(multisets[n].items()); for strict True canonical_map ordering is used implicitly.)
            nodes_at_height = sorted(nodes_at_height, key=lambda n: key_fn(n))

            # Group nodes by their key (canonical string or sorted multiset tuple)
            groups = defaultdict(list)
            for n in nodes_at_height:
                groups[key_fn(n)].append(n)

            # For each group with >1 node, pick the first as the retained node and treat the rest as isomorphs
            for group_nodes in groups.values():
                if len(group_nodes) <= 1:
                    continue

                leader = group_nodes[0]  # the node we keep and attach the reticulation above
                # If leader has no parent, we cannot insert a reticulation above it (mirrors previous behavior)
                parent_leader = leader.up
                if not parent_leader:
                    # This mirrors previous code that only inserts reticulation when parent exists.
                    # Skip group if there's no valid parent for leader.
                    continue

                pid = id(parent_leader)
                nid_leader = id(leader)

                # Insert reticulation node above leader in G
                retic_node = cls._insert_reticulation(G, pid, nid_leader)

                # For every other isomorphic node in the group, reroute its parent's edge(s) to the retic node
                for iso in group_nodes[1:]:
                    iso_parent = iso.up
                    iso_pid = id(iso_parent) if iso_parent else None

                    # If the DAG contains the edge we expect, replace it so the parent's edge now goes to the retic node
                    if iso_pid and G.has_edge(iso_pid, id(iso)):
                        G.remove_edge(iso_pid, id(iso))
                        G.add_edge(iso_pid, retic_node)

                    # Remove the isomorphic subtree nodes from the DAG (exactly as before)
                    to_delete = [id(n) for n in iso.traverse()]
                    G.remove_nodes_from(to_delete)

        # Remove reticulation nodes that have exactly one predecessor and one successor
        cls._simplify_redundant_reticulations(G)
        return G

    @classmethod
    def polyphest_multree_to_dag(cls, tree, threshold=0.2, normalize=True):
        '''
        Polyphest's implementation of the HOLM2006 algorithm:

        Converts an ETE3 MUL-tree to a DAG by merging near-isomorphic subtrees
        based on graph edit distance <= threshold.

        This follows the structure of polyphest's folding.convert_multree_to_network(),
        but uses ETE3 + NetworkX instead of PhyNetPy + Polyphest's helper functions.
        '''
        # Height map {height: [nodes at that height]}
        height_map = cls._compute_height_map(tree)

        # DAG copy of the tree (node IDs are id(ete3_node))
        G = cls.tree_to_dag(tree)

        # Helper: get set of leaf names under node
        def get_leaf_set(n):
            return frozenset(leaf.name for leaf in n.iter_leaves())

        # Near-isomorphic check (polyphest-style)
        def is_near_isomorphic(n1, n2):
            # Quick leaf-set filter
            if get_leaf_set(n1) != get_leaf_set(n2):
                return False

            H1 = cls.tree_to_dag(n1)
            H2 = cls.tree_to_dag(n2)
            dist = nx.graph_edit_distance(
                H1, H2,
                node_match=lambda u, v: u.get('label') == v.get('label')
            )
            if normalize:
                dist /= (H1.number_of_nodes() + H2.number_of_nodes())
            return dist <= threshold

        # Process from top (root) to bottom (leaves)
        for h in sorted(height_map.keys(), reverse=True):
            nodes_to_check = list(height_map[h])
            while nodes_to_check:
                node = nodes_to_check.pop(0)
                isomorphic_trees = [node]
                remaining = []

                for other in nodes_to_check:
                    if is_near_isomorphic(node, other):
                        isomorphic_trees.append(other)
                    else:
                        remaining.append(other)

                nodes_to_check = remaining

                if len(isomorphic_trees) > 1:
                    leader = isomorphic_trees[0]
                    parent_leader = leader.up
                    if not parent_leader:
                        continue  # Can't insert reticulation above root

                    pid = id(parent_leader)
                    nid_leader = id(leader)
                    retic_node = cls._insert_reticulation(G, pid, nid_leader)

                    for iso in isomorphic_trees[1:]:
                        iso_parent = iso.up
                        if iso_parent and G.has_edge(id(iso_parent), id(iso)):
                            G.remove_edge(id(iso_parent), id(iso))
                            G.add_edge(id(iso_parent), retic_node)
                        G.remove_nodes_from([id(n) for n in iso.traverse()])

        cls._simplify_redundant_reticulations(G)
        return G

    # ───── MODULE: From exact network to (MUL-)tree ─────

    @classmethod
    def dag_to_multree(cls, G, match_dag=False):
        '''
        Convert a DAG (from multree or similar) back into an ete3 Tree.
        This handles multi-labeled trees and reticulations.
        '''
        def clone_node(orig_node_data):
            node = TreeNode()
            label = orig_node_data.get('name') or orig_node_data.get('label') or None
            if label:
                node.name = label
            return node

        # Identify reticulation nodes by having multiple parents
        retic_nodes = {n for n in G.nodes if cls.is_reticulation_node(G, n)}

        # Identify root (node with in-degree 0)
        root_id = cls.get_dag_root(G)
        # Root must not be a reticulation node
        if root_id in retic_nodes:
            raise ValueError('Root node cannot be a reticulation node.')

        def build_tree(node_id):
            node_data = G.nodes[node_id]

            # Skip reticulation nodes and attach their single child directly
            if node_id in retic_nodes:
                return build_tree( ReticulateTree._get_single_child(G, node_id) )

            # Normal node
            curr_node = clone_node(node_data)

            # Attach to DAG node if requested
            if match_dag:
                G.nodes[node_id]['ete'] = curr_node
                G.nodes[node_id]['label'] = curr_node.name

            for child_id in G.successors(node_id):
                child_subtree = build_tree(child_id)
                curr_node.add_child(child_subtree)

            return curr_node

        # Post-process: tag reticulation nodes (only if match_dag)
        if match_dag:
            for node in retic_nodes:
                G.nodes[node].pop('ete', None)
                G.nodes[node]['label'] = 'Ret.'

        return build_tree(root_id)

    # ───── MODULE: From H-enriched newick to network ─────

    @staticmethod
    def _find_reticulation_defs(G):
        defs = {}
        for n, data in G.nodes(data=True):
            name = data.get('label', '')
            if name.startswith('#H') and G.out_degree(n) > 0:
                hid = name.split('#H')[1]
                if hid in defs:
                    raise ValueError(f"Duplicate definition for H id '{hid}' found.")
                defs[hid] = n
        return defs

    @staticmethod
    def _link_and_track_references(G, defs):
        renamed_targets = set()
        for n, data in list(G.nodes(data=True)):
            name = data.get('label', '')
            if name.startswith('#H') and G.out_degree(n) == 0:  # leaf reference
                hid = name.split('#H')[1]
                if hid not in defs:
                    continue

                parent = next(G.predecessors(n), None)
                if parent is None:
                    continue

                edata = dict(G[parent][n])  # preserve edge data (branch length)
                G.remove_node(n)  # remove leaf reference

                target = defs[hid]
                if G.out_degree(target) > 1:  # cladistic case
                    G.add_edge(parent, target, **edata)
                    new_node = hid + '_ref'
                    G.add_node(new_node)
                    for child in list(G.successors(target)):
                        G.add_edge(new_node, child, **G[target][child])
                        G.remove_edge(target, child)
                    G.add_edge(target, new_node)  # structural only
                else:  # direct case
                    G.add_edge(parent, target, **edata)

                renamed_targets.add(hid)

        return renamed_targets

    @classmethod
    def enewick_to_dag(cls, enewick_str):
        '''
        Convert enriched Newick with #H reticulation markers into a DAG.
        - Branch lengths from #H reference leaves are preserved on the edge to the reticulation node.
        - Reticulation -> new internal node edges have no branch length (structural only).
        '''
        # Step 1: parse as-is into DAG
        tree = Tree(enewick_str, format=1) # format for Newick TBD !!!
        G = cls.tree_to_dag(tree)

        # Step 2: find definition nodes (internal nodes ending in #H)
        defs = cls._find_reticulation_defs(G)

        # Step 3: link references and track retnodes
        renamed_targets = cls._link_and_track_references(G, defs)

        # Step 4: rename all reticulation targets
        for hid in renamed_targets:
            target = defs[hid]
            new_tgt_key = 'retic_' + hid
            nx.relabel_nodes(G, {target: new_tgt_key}, copy=False)
            G.nodes[new_tgt_key]['label'] = 'Ret.'  # mark as reticulation

        return G

    def to_enewick(self, filename=None):
        G = self.dag
        retnodes = set(self.retnodes)  # reticulation nodes (no numbering)
        
        # Assign new #H IDs to reticulation nodes
        hid_map = {node: str(i) for i, node in enumerate(sorted(retnodes))}
        
        # Find root
        root = self.get_dag_root(G)

        defined = set()  # keep track of reticulations already defined

        def build(node, parent=None):
            # If reticulation node
            if node in retnodes:
                hid = hid_map[node]
                if node not in defined and (parent is None or parent not in retnodes):
                    # First encounter from main parent → definition
                    defined.add(node)
                    children = list(G.successors(node))
                    if not children:
                        # Reticulation leaf alone
                        return f"#H{hid}"
                    inner = ",".join(build(c, node) for c in children)
                    return f"({inner})#H{hid}"
                else:
                    # Reference form (from other parents)
                    return f"#H{hid}"

            # Non-reticulation node
            children = list(G.successors(node))
            if not children:
                # leaf: output label or node id
                label = G.nodes[node].get("label", str(node))
                return label
            
            label = G.nodes[node].get("label", "")
            inner = ",".join(build(c, node) for c in children)
            return f"({inner}){label}"

        newick_str = build(root) + ";"

        if filename:
            with open(filename, "w") as f:
                f.write(newick_str)
        return newick_str


    # ───── MODULE: Visualization ─────

    def __str__(self):
        return self.tree_str
    
    def visualize(self, filename=None, uid_labeled=False):
        G = self.dag
        depths = self.compute_depths(G)

        # Annotate graph with depth-based 'layer' for multipartite layout
        for node, depth in depths.items():
            G.nodes[node]['layer'] = depth

        pos = nx.multipartite_layout(G, subset_key='layer')

        # Color and shape encoding
        node_colors = []
        node_shapes = {}
        for node in G.nodes():
            if self.is_reticulation_node(G, node):
                node_shapes[node] = 'D'
                node_colors.append('orange')
            elif G.out_degree(node) == 0:
                node_shapes[node] = 'o'
                node_colors.append('lightgreen')
            else:
                node_shapes[node] = 'o'
                node_colors.append('lightblue')

        for shape in set(node_shapes.values()):
            shaped_nodes = [n for n in G.nodes if node_shapes[n] == shape]
            nx.draw_networkx_nodes(
                G, pos, nodelist=shaped_nodes,
                node_shape=shape,
                node_color=[node_colors[list(G.nodes).index(n)] for n in shaped_nodes],
                node_size=1000
            )

        nx.draw_networkx_edges(G, pos, arrows=True)

        # Labels (with uid fallback)
        labels = {}
        for n, d in G.nodes(data=True):
            label = d.get('label', None)
            labels[n] = str(n) if uid_labeled and label == '' else label
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)

        plt.axis('off')
        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=690)
            print(f'Graph saved to {filename}')
        plt.show()

    def interact(self, filename, uid_labeled=False, launch=True):
        '''
        Interactive DAG visualization using pyvis.
        Emulates the structure and appearance of the matplotlib-based layout.
        '''
        filename = Path(filename)
        net = Network(height='800px', width='100%', directed=True, layout=True)
        net.toggle_physics(False)  # Static layout for hierarchy-like appearance

        G = self.dag
        depths = self.compute_depths(G)

        # Add nodes
        for node, data in G.nodes(data=True):
            label = data.get('label', '')
            if not uid_labeled and label == '':
                label = ' '

            node_type = 'internal'
            shape = 'ellipse'
            color = 'lightblue'

            if self.is_reticulation_node(G, node):
                node_type = 'reticulation'
                shape = 'diamond'
                color = 'orange'
            elif G.out_degree(node) == 0:
                node_type = 'leaf'
                color = 'lightgreen'

            level = depths.get(node, 0)

            net.add_node(
                str(node),
                label=label,
                title=f'{node_type} node: {node}',
                level=level,
                shape=shape,
                color=color,
                size=20
            )

        # Add edges
        for u, v in G.edges():
            net.add_edge(str(u), str(v), width=1)

        # Save and optionally open
        net.save_graph(str(filename))
        print(f'Interactive graph saved to {filename}.')
        if launch:
            try:
                net.show(str(filename))
            except Exception as e:
                print(f'Failed to display HTML. Reason: {e}\nTrying to open in browser...')
                try:
                    webbrowser.open(str(filename))
                    # Wait for the user to come back
                    input('Press Enter to continue...')
                except:
                    print('Failed to open in browser. Please open the file manually.')
                    pass

    # ───── MODULE: Comparative features ─────

    def get_reticulation_nodes(self):
        '''
        Get all reticulation nodes in the DAG.
        '''
        return [n for n in self.dag.nodes() if self.is_reticulation_node(self.dag, n)]
    
    def get_reticulation_count(self):
        return len(self.retnodes)
    
    def get_reticulation_leaves(self):
        '''
        Get all leaves that are descendants of reticulation nodes.
        '''
        leaves = {}
        for ret in self.retnodes:
            desc = nx.descendants(self.dag, ret)
            leaf_labels = [
                self.dag.nodes[n].get('label', str(n))
                for n in desc
                if self.dag.out_degree(n) == 0
            ]
            leaves[ret] = leaf_labels
        return leaves
    
    def get_reticulation_sisters(self):
        '''
        For each reticulation node, return its two sister clades (one from each parent).
        Each clade is a list of ete3 TreeNodes (from the 'ete' attribute in the DAG).
        
        Returns:
            dict: {ret_node_id: [clade1_nodes, clade2_nodes]}
        '''
        def collect_subtree_nodes(node):
            '''Return node as ete3 TreeNodes under the given DAG node.'''
            data = self.dag.nodes[node]
            # Check if n is a ret node, if so, take its only child

            if self.is_reticulation_node(self.dag, node):
                data = self.dag.nodes[ self._get_single_child(self.dag, node) ]
            try:
                return data['ete']
            except KeyError:
                raise ValueError(f'No ete3 node found for DAG node {node}')
            
        sisters = {}
        for ret in self.retnodes:
            parents = list(self.dag.predecessors(ret))
            if len(parents) != 2:
                # Not a proper binary reticulation
                #raise ValueError(f'Reticulation node {ret} does not have exactly two parents: {parents}')
                print(f'Warning: Reticulation node {ret} does not have exactly two parents: {parents}')
                print('This indicates that a Strict MUL-to-Net mode was used on a malformed inference.')

            clades = []
            for parent in parents:
                # Identify the specific child of the parent that is the reticulation node
                # We want the 'sibling' subtrees, i.e., other children of the parent
                sibling_children = [child for child in self.dag.successors(parent) if child != ret]
                # Collect all ete3 nodes in the sibling clades
                clade_nodes = []
                for child in sibling_children:
                    clade_nodes.extend(collect_subtree_nodes(child))
                clades.append(clade_nodes)

            sisters[ret] = clades

        # Convert to a more user-friendly format: list of sets of names
        sisters = { ret:
            [set([n.name for n in side]) for side in sister_pair]
            for ret, sister_pair in sisters.items()
        }

        return sisters
    
    def get_leaf_counts(self):
        return dict(Counter(leaf.name for leaf in self.tree.iter_leaves()))

    def measure(self, printout=False):
        measurements = {
            'reticulation_count': self.get_reticulation_count(),
            'reticulation_leaves': self.get_reticulation_leaves(),
            'reticulation_sisters': self.get_reticulation_sisters(),
            'leaf_counts': self.get_leaf_counts(),
        }
        if printout:
            for key, value in measurements.items():
                k_str = key.replace('_', ' ').title()
                print(f'{k_str}: {value}')
        return measurements

    def get_edit_distance(self, other: 'ReticulateTree', normalize=True) -> float:
        '''
        Compute the graph edit distance between two ReticulateTree instances.
        This is a placeholder for future implementation.
        '''
        distance = next(nx.optimize_graph_edit_distance(
            self.dag, other.dag,
            node_match=lambda u, v: u.get('label') == v.get('label')
        ))
        if normalize:
            normalization = max(
                len(self.dag.nodes) + len(self.dag.edges),
                len(other.dag.nodes) + len(other.dag.edges)
            )
            distance /= normalization
        return distance
    
    def __sub__(self, other: 'ReticulateTree') -> float:
        '''
        Overriding the minus operator to compute the edit distance between two ReticulateTree instances.
        '''
        if not isinstance(other, ReticulateTree):
            raise TypeError('Subtraction is only supported between ReticulateTree instances.')
        return self.get_edit_distance(other)
        
