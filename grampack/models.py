import math
import array
from ete3 import Tree, TreeNode
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass, field

class NameRegistry:
    """
    Global bidirectional map for Species Names <-> Integer IDs.
    Ensures O(1) comparisons and compact storage.
    """
    def __init__(self):
        self._str_to_int: Dict[str, int] = {}
        self._int_to_str: List[str] = []
    
    def get_id(self, name: str) -> int:
        if name not in self._str_to_int:
            self._str_to_int[name] = len(self._int_to_str)
            self._int_to_str.append(name)
        return self._str_to_int[name]
    
    def get_name(self, idx: int) -> str:
        return self._int_to_str[idx]
    
    def size(self) -> int:
        return len(self._int_to_str)

    # --- Persistence Methods ---
    def get_state(self) -> Tuple[Dict[str, int], List[str]]:
        """Returns the internal state for pickling."""
        return self._str_to_int, self._int_to_str

    def set_state(self, state: Tuple[Dict[str, int], List[str]]):
        """Restores the internal state from pickle."""
        self._str_to_int, self._int_to_str = state

@dataclass
class FlatTree:
    """
    Linearized 'Struct of Arrays' representation of a tree.
    Optimized for pickling and fast array-based traversal.
    """
    # Topology & Data (Indexed by Node ID 0..N)
    num_nodes: int
    root_id: int
    
    # Structure
    parents: array.array      # parents[u] = p
    children_start: array.array # CSR-like indexing for children
    children_flat: array.array  # Flattened children list
    
    # Traversal Orders
    postorder: array.array    # Sequence of Node IDs in postorder
    
    # Data
    node_to_name_id: array.array # Mapping to NameRegistry ID (for leaves/internal)
    
    # LCA Precomputation (Euler Tour + Sparse Table)
    euler_tour: array.array   # Sequence of nodes visited in DFS
    depths: array.array       # Depth of nodes in Euler Tour
    first_visit: array.array  # Index in euler_tour where node u is first seen
    
    # --- Fields with defaults must come LAST ---
    name_id_to_node_id: Dict[int, int] = field(default_factory=dict)
    node_id_to_name_id: Dict[int, int] = field(default_factory=dict)
    rmq_table: List[array.array] = field(default_factory=list) # Sparse table for RMQ

    def get_lca(self, u: int, v: int) -> int:
        """O(1) LCA query using Sparse Table RMQ."""
        if u == v: return u
        
        # 1. Find range in Euler tour
        first = self.first_visit[u]
        last = self.first_visit[v]
        if first > last:
            first, last = last, first
            
        # 2. Query RMQ for index with min depth
        span = last - first + 1
        k = int(math.log2(span))
        
        # Compare depths of the two candidates covering the range
        idx1 = self.rmq_table[k][first]
        # The second interval starts at `last - (1<<k) + 1`
        idx2 = self.rmq_table[k][last - (1 << k) + 1]
        
        if self.depths[idx1] < self.depths[idx2]:
            return self.euler_tour[idx1]
        else:
            return self.euler_tour[idx2]

class TreeLinearizer:
    """Helper to convert ETE3/SmrtTree objects to FlatTree."""
    
    @staticmethod
    def linearize(smrt_tree: 'SmrtTree', registry: NameRegistry) -> FlatTree:
        ete_tree = smrt_tree.ete_tree
        
        # Capture nodes in list. Index is the ID.
        # This avoids the O(N log N) sort of the dictionary keys later.
        nodes_in_order = list(ete_tree.traverse("preorder"))
        num_nodes = len(nodes_in_order)
        
        # Map object -> ID for parent lookups
        node_to_id = {node: i for i, node in enumerate(nodes_in_order)}
        
        # Pre-allocate arrays (O(1) append vs O(N) resize overhead)
        parents = array.array('i', [-1] * num_nodes)
        children_start = array.array('i', [0] * (num_nodes + 1))
        node_to_name_id = array.array('i', [-1] * num_nodes)
        
        # We still need dynamic append for flat children as we don't know total count of edges upfront 
        # (though for trees, edges = N-1, so we technically could pre-allocate, but append is fine here)
        children_flat = array.array('i')
        
        name_id_to_node_id = {}
        node_id_to_name_id = {}

        cursor = 0
        
        # Iterate the list directly
        for nid, node in enumerate(nodes_in_order):
            
            # --- Name Logic ---
            raw_name = str(node.name) if node.name else ""
            name_idx = -1
            
            if node.is_leaf():
                # Extract species name. 
                # For GT: "Gene_Species" -> "Species"
                # For ST (MUL): "Species*" -> "Species*" (Preserves distinction)
                # A clean name is needed to find matching species in ST
                sp_name = raw_name.split("_", 1)[-1] if "_" in raw_name else raw_name # raw_name.split("_")[-1] 
                name_idx = registry.get_id(sp_name)
                
                # Also index the full leaf name for Group lookup (e.g. "Gene1_Species")
                full_name_idx = registry.get_id(raw_name)
                name_id_to_node_id[full_name_idx] = nid
                node_id_to_name_id[nid] = full_name_idx
                
            elif raw_name:
                # Internal nodes (e.g. "H1")
                name_idx = registry.get_id(raw_name)
                name_id_to_node_id[name_idx] = nid
                node_id_to_name_id[nid] = name_idx

            node_to_name_id[nid] = name_idx

            # --- Topology Logic ---
            # Parent
            if node.up:
                parents[nid] = node_to_id[node.up]
            
            # Children
            children_start[nid] = cursor
            for child in node.children:
                # Direct dict lookup is fast
                children_flat.append(node_to_id[child])
                cursor += 1
        
        children_start[num_nodes] = cursor

        # 2. Postorder Traversal
        # Map the objects to IDs directly
        postorder = array.array('i', [node_to_id[n] for n in ete_tree.traverse("postorder")])

        # 3. Euler Tour & RMQ (Stack-Based DFS)
        return TreeLinearizer._build_flat_tree(
            num_nodes, node_to_id[ete_tree.get_tree_root()],
            parents, children_start, children_flat, postorder,
            node_to_name_id, name_id_to_node_id, node_id_to_name_id
        )

    @staticmethod
    def _build_flat_tree(num_nodes, root_id, parents, children_start, children_flat, postorder,
                         node_to_name_id, name_id_to_node_id, node_id_to_name_id):
        """
        Pure integer array processing. Separated from ETE3 logic.
        """
        euler_nodes = array.array('i')
        euler_depths = array.array('i')
        first_visit = array.array('i', [-1] * num_nodes)
        
        # Optimized Stack: (node_id, depth, child_ptr_start, child_ptr_end)
        # Storing 'end' in the stack avoids array lookups inside the loop
        stack = [[root_id, 0, children_start[root_id], children_start[root_id+1]]]
        
        # Pre-visit root
        first_visit[root_id] = 0
        euler_nodes.append(root_id)
        euler_depths.append(0)
        
        while stack:
            # Peek at top (don't unpack yet to keep reference mutable if needed, though lists are ref)
            frame = stack[-1]
            u, d, ptr, end = frame
            
            if ptr < end:
                # Visit next child
                v = children_flat[ptr]
                frame[2] += 1 # Advance pointer in current frame
                
                # Push child
                stack.append([v, d + 1, children_start[v], children_start[v+1]])
                
                # Pre-visit child
                first_visit[v] = len(euler_nodes)
                euler_nodes.append(v)
                euler_depths.append(d + 1)
            else:
                # Backtrack
                stack.pop()
                if stack:
                    parent = stack[-1]
                    euler_nodes.append(parent[0])
                    euler_depths.append(parent[1])

        # Sparse Table Logic
        L = len(euler_nodes)
        rmq = []
        if L > 0:
            k_max = int(math.log2(L)) + 1
            rmq = [array.array('i', [0] * L) for _ in range(k_max)]
            
            for i in range(L): rmq[0][i] = i
            
            for j in range(1, k_max):
                half = 1 << (j-1)
                for i in range(L - (1 << j) + 1):
                    idx1 = rmq[j-1][i]
                    idx2 = rmq[j-1][i + half]
                    if euler_depths[idx1] < euler_depths[idx2]:
                        rmq[j][i] = idx1
                    else:
                        rmq[j][i] = idx2

        return FlatTree(
            num_nodes=num_nodes,
            root_id=root_id,
            parents=parents,
            children_start=children_start,
            children_flat=children_flat,
            postorder=postorder,
            node_to_name_id=node_to_name_id,
            euler_tour=euler_nodes,
            depths=euler_depths,
            first_visit=first_visit,
            # Defaults last
            name_id_to_node_id=name_id_to_node_id,
            node_id_to_name_id=node_id_to_name_id,
            rmq_table=rmq
        )

class SmrtTree:
    """Wrapper around ETE3 Tree to provide GRAMPA-specific functionality."""

    __slots__ = ['ete_tree', 'node_map', 'match_map', 'flat_tree']

    def __init__(self, newick: str = None, tree_obj: Tree = None):
        if tree_obj:
            self.ete_tree = tree_obj
        else:
            if newick and not newick.strip().endswith(";"):
                newick += ";"
            self.ete_tree = Tree(newick, format=0) 
            
        self.node_map: Dict[str, TreeNode] = {}
        self.match_map: Dict[str, List[TreeNode]] = {}
        self.flat_tree: Optional[FlatTree] = None
        
        self._index_nodes()

    """def _index_nodes(self):
        self.node_map = {} 
        i = 1
        for node in self.ete_tree.traverse("postorder"):
            if not node.is_leaf():
                if not node.name:
                    node.name = f"<{i}>"
                    i += 1
            else:
                node.name = str(node.name).strip()
            
            self.node_map[node.name] = node"""

    def _index_nodes(self):
        """
        All nodes must have unique names after this, and pure attribute set (none-unique).
        Leaves mustn't be in <> format, as these are reserved for internal nodes.
        There should really be no use for the safe counter initialization (except on init), but it's here just in case!
        """
        self.node_map = {} 
        
        # Safe Counter Initialization:
        # scan existing names to find the max index used so far
        # cache nodes to avoid multiple traversals
        to_name = []
        max_idx = 0
        for node in self.ete_tree.traverse("postorder"):
            if node.is_leaf():
                node.name = str(node.name).strip()
            elif node.name:
                if node.name.startswith("<"):
                    try:
                        val = int(node.name[1:].split(">")[0])
                        if val > max_idx: max_idx = val
                    except ValueError:
                        pass
            else:
                to_name.append(node)
                continue
            self.node_map[node.name] = node
        i = max_idx + 1

        for node in to_name:
            node.name = f"<{i}>"
            i += 1
            self.node_map[node.name] = node

        for node in self.ete_tree.traverse("postorder"):
            if not hasattr(node, 'pure'):
                node.add_feature('pure', node.name.replace("*", ""))

    def _index_nodes_2(self, repair: bool = False, suffixed: bool = False):
        """
        Node attrutes:
        name = original name, must be **unique** (hence, node_map), serves for outputs and
            getting the node (also unique but doesn't pickle well)
        spec = cleaned species name - not unique, no affixes, serves as the "biological" unit
        reco = species name used for reconciliation, not unique, may be later suffixed:
            original gt may have two copies of "x" (=spec/reco)
            ofter first conciliation, they may become "x.1" and "x.2" (while spec remains "x")
            after second conciliation, they may become "x.1.1" and "x.2.2", for example
        r_id = registry ID for reco
        u_id = registry ID for name (may be same as r_id before suffixing is applied)
        """
        import re

        leaf_counts = {}
        self.node_map = {}
        i = 1
        for n in self.ete_tree.traverse("postorder"):
            if not n.is_leaf():
                name = f"<{i}>"
                i += 1
                if not n.name:
                    n.name = name
                n.add_feature("reco", n.name)
                n.add_feature("r_id", self.registry.get_id(n.name))
                n.add_feature("u_id", self.registry.get_id(n.name)) # not sure if needed
            else:
                n.name = str(n.name).strip()
                if "<" in n.name and ">" in n.name:
                    raise ValueError(f"Leaf name {n.name} cannot contain '<' or '>' characters.")

                name = n.name

                if repair:
                    # Ensure unique names
                    # Names are prefixed with count only if duplicates exist
                    # Zeroth copy gets a prefix in a retroactive manner
                    if name in leaf_counts:
                        dup_count = leaf_counts[name]
                        if dup_count == 0:
                            first_node = self.node_map.pop(name)
                            first_node.name = f"0_{name}"
                            self.node_map[first_node.name] = first_node
                        leaf_counts[name] += 1
                        n.name = f"{leaf_counts[name]}_{name}"
                    else:
                        leaf_counts[name] = 0
                else:
                    # Assume names are unique!
                    pass
                
                reco = n.name.split("_", 1)[1] if "_" in n.name else n.name
                n.add_feature("reco", reco)
                n.add_feature("r_id", self.registry.get_id(reco))
                n.add_feature("u_id", self.registry.get_id(n.name))

                spec = re.sub(r'(\.\d+)+$', '', reco) if suffixed else reco
                n.add_feature("spec", spec)
                
                if name in self.node_map:
                    raise ValueError(f"Duplicate leaf name detected after processing: {name}")

            self.node_map[n.name] = n

    def refresh(self):
        self._index_nodes()
        self.match_map = {}
        self.flat_tree = None # Invalidate flat tree on structural change

    def make_flat(self, registry: NameRegistry):
        """Generates the FlatTree bundle for optimized processing."""
        self.flat_tree = TreeLinearizer.linearize(self, registry)

    """def _cache_depths(self):
        for node in self.ete_tree.traverse("preorder"):
            if node.is_root():
                node.add_feature("fast_depth", 0)
            else:
                node.add_feature("fast_depth", node.up.fast_depth + 1)
            
            clean = node.name.replace("*", "") if node.is_leaf() else "" """

    def get_node(self, name: str) -> Optional[TreeNode]:
        return self.node_map.get(name)
    
    def match(self, name: str) -> List[TreeNode]:
        """
        Assumes nodes have a 'pure' attribute with the cleaned name (e.g. "Species" without "*").
        Returns all nodes matching the cleaned name, which is necessary for MUL trees.
        """
        if not self.match_map:
            for node in self.ete_tree.traverse():
                self.match_map.setdefault(node.pure, []).append(node)
        return self.match_map[name]

    @staticmethod
    def graft_subtree(tree: TreeNode, target: TreeNode, graft: TreeNode, name: str = None) -> TreeNode:
        """
        Grafts `graft` on the branch leading to `p_node`.
        Modifies the tree in place, but return is needed because it might create a new root
        """
        p_parent = target.up
        if p_parent is None:
            new_root = TreeNode(name=name)
            new_root.add_child(target.detach())
            new_root.add_child(graft)
            tree = new_root
            if name:
                new_root.add_feature("pure", name.replace("*", ""))
        else:
            new_internal = TreeNode(name=name)
            p_parent.add_child(new_internal)
            new_internal.add_child(target.detach())
            new_internal.add_child(graft)
            if name:
                new_internal.add_feature("pure", name.replace("*", ""))
        return tree
    
    def rename_marked_nodes(self, target_nodes: List[Set[TreeNode]], suffixes: List[str]) -> List[Set[str]]:
        """
        Suffix is added to the name of all nodes in target_nodes.
        This is used to mark the Hx lineages after grafting.
        """
        marked_names = []
        for inner_targets, suffix in zip(target_nodes, suffixes):
            marked_set = set()
            for node in inner_targets:
                old_name = node.name
                marked_set.add(old_name)

                if old_name == '<P>':
                    # P nodes of different iterations shouldn't have the same pure name! (hence no separator '|')
                    self.rename_node('<P>', f'<P{suffix}>')

                else:
                    # Generate New Name
                    # Leaves: "Species" -> "Species|1.0"
                    # Internals: "<1>" -> "<1|1.0>"
                    # Replace existing '*' with the new suffix
                    # Handle cases if | is already present - we want to replace it.
                    
                    '''if '|' in old_name:
                        old_sf = old_name.split('|')[1].split('*')[0]
                        new_name = old_name.replace(f'|{old_sf}*', f"|{suffix}")
                    else:
                        new_name = old_name.replace('*', f"|{suffix}")'''
                    
                    # REPLACE WITH:
                    new_name = old_name.replace('*', f"|{suffix}")

                    # Apply Rename using Wrapper (handles node_map updates)
                    self.rename_node(old_name, new_name)

            marked_names.append(marked_set)

        # Must refresh after name modification
        self.refresh()
        return marked_names

    def rename_leaves_based_on_map_targets(self, map: 'Map', targets: List[Set[str]], suffixes: List[str]) -> None:#targets: List[str], suffix: str) -> None:
        
        rev_map = map.rev # map_name -> List[names]

        for inner_targets, suffix in zip(targets, suffixes):
            for target in inner_targets:
                node_name_to_modify = rev_map.get(target, [])
                for node_name in node_name_to_modify:
                    n = self.get_node(node_name)
                    if n.is_leaf():

                        '''
                        if '|' in n.name: n.name = n.name.split('|')[0]
                        n.name = f"{n.name}|{suffix}"'''

                        # REPLACE WITH:
                        n.name = f"{n.name}|{suffix}"

                        # node.pure stays the same, as it's used for matching and should not be suffixed
                    # No need to modify internal nodes - Reconcile only works on lvs

        '''
        t = self.ete_tree
        # Iterate leaves
        for l in t.iter_leaves():
            # Find to where this leaf is mapped in the map
            # map.cor[leaf_name] returns List[node_names]
            # We take the first mapping (usually only one for optimal recon)
            target_name = map.cor[l.name][0]
            if target_name in targets:
                if '|' in l.name: l.name = l.name.split('|')[0]
                l.name = f"{l.name}|{suffix}"
                # l.pure stays the same!
                '''

        # Must refresh after name modification
        self.refresh()
    
    def rename_node(self, old_name: str, new_name: str, tagged=False) -> None:
        """
        New name should be in pure form,
        Old name is the current unique name in the tree, and should be tagged if tagged=True.
        If old_name not found, does nothing.
        """
        node = self.get_node(old_name)
        if not node:
            return
        
        tag = ""
        if tagged:
            tag = new_name[-1]
            if tag == '>':
                tag = new_name[-2:-1]
        if tag.isalnum():
            raise ValueError(f"Tagged rename requires a non-alphanumeric tag at the end of the new name. Got '{tag}' in '{new_name}'.")

        node.name = new_name[:-1]+tag+'>' if new_name.endswith(">") else new_name+tag

        if not hasattr(node, 'pure'):
            node.add_feature('pure', new_name.split('|')[0])
            if not node.is_leaf() and not node.pure.endswith('>'):
                node.pure += '>'
        else:
            node.pure = new_name.split('|')[0]
            if not node.is_leaf() and not node.pure.endswith('>'):
                node.pure += '>'
        # doesn't work because p!=p

        # Update node_map
        del self.node_map[old_name]
        self.node_map[new_name] = node
        # Clear caches
        self.match_map = {}
        self.flat_tree = None
    
    @staticmethod
    def copy_lineage(subtree: TreeNode, tag: str = '') -> TreeNode:
        subtree = subtree.copy()
        # Pure name should be already set due to copy!
        for n in subtree.traverse():
            # n.add_feature('pure', n.name)
            if n.is_leaf():
                n.name = f"{n.name}{tag}"
            elif n.name and n.name.startswith("<") and n.name.endswith(">"):
                # Wipe internal node names which will be made unique during indexing
                #n.name = None
                n.name = n.name.replace(">", f"{tag}>")
        return subtree

    def to_mul_tree(self, h_node_label: str, p_node_label: str) -> Optional[Tuple['SmrtTree', TreeNode, TreeNode]]:
        # [Existing to_mul_tree code remains identical]
        # ... copy, graft logic ...
        # Just ensure the returned SmrtTree is fresh
        new_tree_obj = self.ete_tree.copy()
        
        h_matches = new_tree_obj.search_nodes(name=h_node_label)
        p_matches = new_tree_obj.search_nodes(name=p_node_label)
        
        if not h_matches or not p_matches: return None, None, None
        h1_node = h_matches[0]
        p_node = p_matches[0]

        if p_node in h1_node.iter_descendants(): return None, None, None

        h2_subtree = SmrtTree.copy_lineage(h1_node, '*')

        new_tree_obj = SmrtTree.graft_subtree(new_tree_obj, p_node, h2_subtree, '<P>')

        new_smrt = SmrtTree(tree_obj=new_tree_obj)
        return new_smrt, h1_node, h2_subtree # Note: returns objects in the new tree context



    def to_mul_tree_multi(self, h1_name: str, hx_names: List[str]) -> Optional[Tuple['SmrtTree', TreeNode, List[TreeNode]]]:
        """
        Grafts multiple H-lineages (H2, H3...) onto the H1 branch.
        Used for 'Model' mode to capture all nested copies at once.
        """
        new_tree_obj = self.ete_tree.copy()
        
        # 1. Find H1 (The Stock)
        h1_matches = new_tree_obj.search_nodes(name=h1_name)
        if not h1_matches: return None, None, []
        h1_node = h1_matches[0]

        # 2. Process Hx targets (The Scions)
        hx_nodes_final = []
        
        # We use a tag generator: *, **, ***, ...
        tags = ["*" * i for i in range(1, len(hx_names) + 1)]

        for i, (h_name, tag) in enumerate(zip(hx_names, tags), start=2):
            # We must search by name in the *current* state of new_tree_obj
            # (Note: grafting changes the tree structure, but names persist)
            p_matches = new_tree_obj.search_nodes(name=h_name)
            if not p_matches: continue
            p_node = p_matches[0]

            # Nesting check: Cannot graft a parent into a child
            if p_node in h1_node.iter_descendants(): continue

            # Create the copy
            # Note: We copy from H1 (the source of the introgression)
            h_copy = SmrtTree.copy_lineage(h1_node, tag)

            # Graft
            # We use a unique internal name for the graft point to avoid confusion
            graft_name = f"<P{i}>"
            new_tree_obj = SmrtTree.graft_subtree(new_tree_obj, p_node, h_copy, graft_name)
            
            hx_nodes_final.append(h_copy)

        new_smrt = SmrtTree(tree_obj=new_tree_obj)
        
        # We must re-find H1 because the root might have changed during grafting
        h1_final = new_smrt.get_node(h1_node.name)
        
        return new_smrt, h1_final, hx_nodes_final


    # --- I/O and Pickling ---

    def to_mult_str(self, internals=True) -> str:
        name_to_pure = {n.name: n.pure for n in self.ete_tree.traverse()}
        return self.to_str(internals=internals, name_formatter=lambda name: name_to_pure.get(name, 'Error'))

    def to_mult(self, internals=True) -> Tree:
        mult_str = self.to_mult_str(internals=internals)
        return Tree(mult_str, format=8 if internals else 9)

    def to_marked_str(self, node_to_mark: TreeNode) -> str:
        """
        Node_to_mark is the node in the ete_tree that should be marked with a "+" in the string output.
        Generally it is the H1 node after a new inference.
        Returns a string representation of the tree with +/* marked nodes.
        """
        marked_nodes = [n for n in node_to_mark.traverse()] if node_to_mark else []
        # Rename, create str, undo rename
        for n in marked_nodes:
            n.name = n.name + "+" if n.is_leaf() else n.name [:-1] + '+>'
        marked_str = self.to_str(internals=True)
        for n in marked_nodes:
            n.name = n.name.replace("+", "")
        return marked_str

    def to_str(self, internals=True, name_formatter=None) -> str:
        root_name = str(self.ete_tree.name) if internals else ""
        return self.ete_tree.write(format=8 if internals else 9, name_formatter=name_formatter)[:-1]+root_name+";"
    
    def __getstate__(self):
        return self.ete_tree
    
    def __setstate__(self, state):
        self.ete_tree = state
        self.refresh()

@dataclass(slots=True, frozen=True)
class Map:
    n_dups: int
    n_losses: int
    # gt_node_name -> list[sp_node_names]
    cor: Dict[str, List[str]] 
    # Lazy loaded reverse map: sp_node_name -> list[gt_node_names]
    _rev: Dict[str, List[str]] = field(default=None, init=False)
    # Node to Node mapping is avoided to prevent pickling issues

    dups: Dict[str, int] = field(default_factory=dict) # gt node label -> num dups
    losses: Dict[str, int] = field(default_factory=dict) # gt node label -> num losses

    @property
    def rev(self) -> Dict[str, List[str]]:
        if self._rev is None: # Still not built (i.e., empty dict)
            # Bypass frozen constraint for lazy loading
            rev_map = {}
            for gt_node, sp_nodes in self.cor.items():
                for sp in sp_nodes:
                    rev_map.setdefault(sp, []).append(gt_node)
            object.__setattr__(self, '_rev', rev_map)
        return self._rev
    
    def __getitem__(self, key: str) -> List[str]:
        return self.cor[key]
    
@dataclass(slots=True, frozen=True)
class ReconResult:
    score: int
    maps: List[Map]

@dataclass(slots=True, frozen=True)
class GroupData:
    # Changed from List[str] to List[int] (IDs from NameRegistry)
    ambiguous_groups: List[List[int]]
    #fixed_groups: List[Tuple[List[int], str]]
    fixed_groups: List[Tuple[List[int], int]] # Changed from str to int

@dataclass(slots=True, frozen=True)
class MulTree:
    mt: SmrtTree
    h_clade: List[str] = field(default_factory=list)
    # Storing OBJECTS optimizes the Reconciler (no lookups)
    # These are safe to pickle TO workers, but shouldn't be used to map back TO main.
    h1_node: Optional[TreeNode] = None 
    # Replaced single h2_node with hx_nodes list
    hx_nodes: List[TreeNode] = field(default_factory=list)
    # For mode compatibility, h2_node can return the first element or None
    @property
    def h2_node(self) -> Optional[TreeNode]:
        return self.hx_nodes[0] if self.hx_nodes else None

@dataclass(slots=True, frozen=True)
class TaskResult:
    """Return payload from a GRAMPA step."""
    
    # List of (MulTreeID, Score) tuples, sorted by Score ASC
    sorted_scores: List[Tuple[int, int]]
    
    # Map: MulTreeID -> MulTree Object
    mul_trees: Dict[int, MulTree]
    
    # Map: MulTreeID -> (GeneTreeID -> MapObject)
    # Only contains maps for the top N trees (usually just the best one)
    kept_mul_maps: Dict[int, Dict[int, Map]]
    
    # All gene trees used in this step
    gene_trees: Dict[int, SmrtTree]

    # Internal cache for self_score lookup (Rank of Species Tree ID 0)
    _input_rank: int = field(init=False, default=-1)

    def __post_init__(self):
        """Retrieves onces at O(N) the rank of the input species tree (always Index 0)."""
        rank = -1
        # O(N) iteration done ONCE during creation
        for i, (idx, _) in enumerate(self.sorted_scores):
            if idx == 0:
                rank = i
                break
        # Bypass frozen constraint
        object.__setattr__(self, '_input_rank', rank)

    def mt_idx(self, rank=0) -> int:
        """Returns the ID of the best performing mt."""
        return self.sorted_scores[rank][0] # Tuple is (Index, Score)

    def mt_score(self, rank=0) -> int:
        """Returns the score of the best performing mt."""
        return self.sorted_scores[rank][1]

    @property
    def input_score(self) -> int:
        """O(1) retrieval of the input species tree score."""
        if self._input_rank == -1: return float('inf')
        # Return tuple item [1] (score), as [0] (index) is always 0
        return self.sorted_scores[self._input_rank][1]


# --- Type aliases ---

HistoryType = Dict[Tuple[int, int], Dict[str, Any]]
ConcurrTask = Tuple[SmrtTree, Dict[int, SmrtTree], str]

