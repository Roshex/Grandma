import math
import array
from pathlib import Path
from functools import partial
from ete3 import Tree, TreeNode
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Any, Set, Union
from dataclasses import dataclass, field

from Reticulate_Tree.reticulate_tree import ReticulateTree

# Grampa does:                  raw.rsplit("_", 1)[-1]  // split by last underscore
# Grandma will eventually do:   raw.split("_", 1)[-1]   // split by first underscore, to preserve species names with underscores
splitSpec = lambda raw: raw.rsplit("_", 1)[-1] if "_" in raw else raw

class NameRegistry:
    """
    Global bidirectional map for Species Names <-> Integer IDs.
    Ensures O(1) comparisons and compact storage.
    """
    def __init__(self):
        self._str_to_int: Dict[str, int] = {}
        self._int_to_str: List[str] = []
    
    def get_id(self, name: str) -> int:
        try:
            return self._str_to_int[name]
        except KeyError:
            idx = len(self._int_to_str)
            self._str_to_int[name] = idx
            self._int_to_str.append(name)
            return idx
    
    def get_ids(self, names: List[str]) -> List[int]:
        return [self.get_id(name) for name in names]
    
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
    node_depths: array.array
    
    # --- Fields with defaults must come LAST ---
    name_id_to_node_id: Dict[int, int] = field(default_factory=dict)
    node_id_to_name_id: Dict[int, int] = field(default_factory=dict)
    rmq_table: List[array.array] = field(default_factory=list) # Sparse table for RMQ

    def get_lca(self, u: int, v: int) -> int:
        """O(1) LCA query using Sparse Table RMQ."""
        if u == v: return u
        
        # Find range in Euler tour
        first = self.first_visit[u]
        last = self.first_visit[v]
        if first > last:
            first, last = last, first
            
        # Query RMQ for index with min depth
        span = last - first + 1
        k = span.bit_length() - 1 # same as int(math.log2(span)) for any span > 0, and much faster
        
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
                sp_name = splitSpec(raw_name) 
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
        node_depths = array.array('i', [0] * num_nodes)
        
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
                node_depths[v] = d + 1
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
            node_depths=node_depths,
            # Defaults last
            name_id_to_node_id=name_id_to_node_id,
            node_id_to_name_id=node_id_to_name_id,
            rmq_table=rmq
        )

@dataclass(slots=True)
class GraftRecord:
    """Encapsulates the location and metadata for a single H lineage graft."""
    copy_id: int
    original: str
    corrected: str
    parent: str
    grandp: str
    aunt: str
    expanded_targets: Optional[List[TreeNode]] = None

    def __repr__(self):
        if self.expanded_targets is not None:
            if self.expanded_targets:
                return f"GraftRec(id={self.copy_id}, orig='{self.original}', fixed='{self.corrected}', p='{self.parent}', expanded={[n.name for n in self.expanded_targets]})"
            else:
                return f"GraftRec(id={self.copy_id}, orig='{self.original}', fixed='{self.corrected}', p='{self.parent}', delayed={not bool(self.expanded_targets)})"
        return f"GraftRec(id={self.copy_id}, orig='{self.original}', p='{self.parent}', g={self.grandp})"

@dataclass(slots=True, frozen=True)
class GroupData:
    # ints are IDs from NameRegistry
    ambiguous_groups: List[List[int]]
    fixed_groups: List[Tuple[List[int], int]]

class SmrtTree:
    """Wrapper around ETE3 Tree to provide GRAMPA-specific functionality."""

    __slots__ = ['ete_tree', 'node_map', 'match_map', 'flat_tree', 'Q']

    def __init__(self, tree_obj: Tree = None, newick: str = None, frmt: int = 0, kw_root_attrs: dict = {}):
        if tree_obj:
            self.ete_tree = tree_obj
        else:
            if newick and not newick.strip().endswith(";"):
                newick += ";"
            self.ete_tree = Tree(newick, format=frmt, **kw_root_attrs)
            
        self.node_map: Dict[str, TreeNode] = {}
        self.match_map: Dict[str, List[TreeNode]] = {}
        self.flat_tree: Optional[FlatTree] = None
        self.Q: float = 1.0
        
        self._index_nodes()

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
            self.add_pure(node)

    def get_node(self, name: str) -> Optional[TreeNode]:
        return self.node_map.get(name)
    
    def match(self, name: str) -> List[TreeNode]:
        """
        Assumes nodes have a 'pure' attribute with the cleaned name (e.g. "Species" without "*").
        Returns all nodes matching the cleaned name, which is necessary for MUL trees.
        In list() to not give access to the original & to prevent iterator mutation during looping over matches.
        """
        if not self.match_map:
            for node in self.ete_tree.traverse():
                self.match_map.setdefault(node.pure, []).append(node)
        return list(self.match_map[name])

    def make_flat(self, registry: NameRegistry):
        """Generates the FlatTree bundle for optimized processing."""
        self.flat_tree = TreeLinearizer.linearize(self, registry)

    def refresh(self):
        self._index_nodes()
        self.match_map = {}
        self.flat_tree = None # Invalidate flat tree on structural change
        self.Q = 1.0 # Reset Q on structural change

    def calculate_Q(self, total_species: int) -> Tuple[float, float, float, bool]:
        """Calculates harmonic mean Q of topology occupancy (O) and resolved support (R)."""
        leaves = list(self.ete_tree.iter_leaves())
        s = len(set(leaf.pure for leaf in leaves))
        l = len(leaves)
        O = s / total_species if total_species > 0 else 0
        
        # Safely exclude root and leaves
        internal_nodes = [node for node in self.ete_tree.traverse() if not node.is_leaf() and not node.is_root()]
        
        c_sum = 0.0
        normalizer = 1.0  # Normalize 0-100 to 0.0-1.0
        has_support = False
        for node in internal_nodes:
            val = getattr(node, 'support', 1.0)
            if val != 1.0 and val != 100.0: has_support = True
            if val > 1.0: normalizer = 100.0
            c_sum += val
        c_sum /= normalizer
            
        B_max = l - 2
        R = c_sum / B_max if B_max > 0 else (1.0 if l <= 2 else 0.0)
        self.Q = 2 * O * R / (O + R) if (O + R) > 0 else 0.0
        return O, R, self.Q, has_support

    """
    def calculate_Q(t: Tree, N: int) -> float:
        # O = s/N := taxon occupancy
        # s := number of species in the GT; N := total number of species [species, not leaves!]
        # R = Sum(c_i)/B_max := resolved support
        # c_i := support value of each internal node (in %); B_max := maximum possible number of internal branches for that GT (i.e., fully bifurcating)
        # B_max = l - 2 for rooted trees, l - 3 for unrooted trees (but we assume rooted due to repair), where l is the number of leaves in the GT.
        # Returns Q = 2*O*R/(O+R), the harmonic mean of O and R.

        # LOGIC TO BE CHECKED !!! #
        leaves = t.get_leaves()
        s = len(set(leaf.pure for leaf in leaves))
        l = len(leaves)
        O = s / N if N > 0 else 0
        c_sum = sum(node.support/100 for node in t.traverse() if not node.is_leaf())
        B_max = l - 2 if t.is_rooted else l - 3
        R = c_sum / B_max if B_max > 0 else 0
        Q = 2 * O * R / (O + R) if (O + R) > 0 else 0
        return Q, O, R
    """

    def are_all_nodes_unique(self) -> bool:
        """Checks if all nodes have unique names (not pure)."""
        seen = set()
        for node in self.ete_tree.traverse():
            if node.name in seen:
                return False
            seen.add(node.name)
        return True
    
    def contains(self, original_tree: Tree) -> bool:
        """
        Verifies that the original tree's topology and exact node names 
        are perfectly preserved within this tree's structure.
        """
        # Verify all original names (internal and leaf) are still in the tree
        orig_names = {n.name for n in original_tree.traverse() if n.name}
        self_names = {n.name for n in self.ete_tree.traverse() if n.name}

        if not orig_names.issubset(self_names):
            return False

        # Prune a copy of self down to only the original leaves
        orig_leaves = original_tree.get_leaf_names()
        tree_copy = self.ete_tree.copy()
        
        try:
            tree_copy.prune(orig_leaves, preserve_branch_length=False)
        except Exception:
            return False

        # Compare topologies using Robinson-Foulds distance
        # unrooted_trees=False ensures the rooted topology is compared correctly
        rf, rf_max, common_attrs, edges_t1, edges_t2, dis_edges_t1, dis_edges_t2 = tree_copy.robinson_foulds(original_tree, unrooted_trees=False)
        
        return rf == 0

    def _clear_dead(self, affected_nodes: Dict[str, Set[str]]) -> None:
        """
        Removes affected nodes from caches after structural modifications.
        affected_nodes: Dict[pure_name, Set[node_names]]
        """
        self.flat_tree = None
        for pure, names in affected_nodes.items():
            for k in names:
                self.node_map.pop(k, None)
            if pure in self.match_map:
                self.match_map[pure] = [n for n in self.match_map[pure] if n.name not in names]
                if not self.match_map[pure]:
                    del self.match_map[pure]

    def rename_leaves_from_mapping(self, recon_map: 'Map', suffix_name_map: Dict[str, Set[str]]) -> None:
        
        rev_map = recon_map.rev # map_name -> List[names]

        for suffix, target_set in suffix_name_map.items():
            for target in target_set:
                node_name_to_modify = rev_map.get(target, [])
                for node_name in node_name_to_modify:
                    n = self.get_node(node_name)
                    if n.is_leaf():
                        n.name = f"{n.name}|{suffix}"
                        # node.pure stays the same, as it's used for matching and should not be suffixed
                    # No need to modify internal nodes - Reconcile only works on lvs

        # Must refresh after name modification
        self.refresh()
    
    def _rename_node_no_reindex(self, old_name: str, new_name: str) -> None:
        node = self.get_node(old_name)
        node.name = new_name

    def rename_node(self, old_name: str, new_name: str) -> None:
        node = self.get_node(old_name)
        self._rename_node_no_reindex(old_name, new_name)

        # Update safely, by invalidating incorrect fields
        del self.node_map[old_name]
        self.node_map[new_name] = node
        self.match_map = {}
        self.flat_tree = None

    @staticmethod
    def add_pure(node: TreeNode):
        """
        Adds a 'pure' attribute to new nodes.
        Does not modify existing pure attributes - only <P> nodes should modify pure names, and do it manually.
        """
        if not hasattr(node, 'pure'):
            pure = node.name.replace("*", "").split('|', 1)[0]
            if not node.is_leaf() and not pure.endswith('>'):
                pure += '>'
            node.add_feature('pure', pure)

    @staticmethod
    def pure_str(name: str) -> str:
        if not name: return name
        base = name.replace("*", "").split('|', 1)[0]
        if name.endswith('>') and not base.endswith('>'):
            base += '>'
        return base
    
    @staticmethod
    def get_sis(node: Optional[Tree]) -> Optional[Tree]:
        """Returns the sister node of the given node, or None if root."""
        if not node or node.is_root():
            return None
        sisters = node.get_sisters()
        if len(sisters) != 1:
            raise ValueError("Tree structure invalid for sister retrieval.")
        return sisters[0]

    def get_targets(self, primary: Union[str, Tree]) -> List[Tree]:
        '''
        Returns the list of all target nodes matching the pure name of a primary target.
        '''
        # if primary is a string, find the node first
        if isinstance(primary, str):
            primary = self.get_node(primary)

        prim_name = primary.name
        if '|' in prim_name:
            pure_name = prim_name.split('|')[0]
            if not primary.is_leaf():
                pure_name += '>'
        else:
            pure_name = prim_name

        if pure_name != primary.pure:
            raise ValueError(f"Primary target's pure attribute '{primary.pure}' does not match expected pure name '{pure_name}'.")
        
        return self.match(pure_name)

    @property
    def desc_pure_cache(self) -> Dict[Tree, Set[str]]:
        pure_desc_cache = {}
        for node in self.ete_tree.traverse("postorder"):
            desc_set: Set[str] = set()
            for child in node.children:
                desc_set.add(child.pure)
                # Union with child's descendants
                desc_set.update(pure_desc_cache.get(child, set()))
            pure_desc_cache[node] = desc_set
        return pure_desc_cache

    @property
    def clade_pure_counts(self) -> Dict[str, Dict[str, int]]:
        """
        O(N) bottom-up traversal. Returns the exact count of each pure species 
        under every node. Format: Dict[node_name, Dict[pure_sp, count]].
        """
        counts_cache = {}
        for node in self.ete_tree.traverse("postorder"):
            if node.is_leaf():
                counts_cache[node.name] = {node.pure: 1}
            else:
                merged = {}
                for child in node.children:
                    for sp, count in counts_cache.get(child.name, {}).items():
                        merged[sp] = merged.get(sp, 0) + count
                counts_cache[node.name] = merged
        return counts_cache

    @property
    def node_order(self) -> List[str]:
        """
        Returns nodes in legacy GRAMPA order: 
        All leaves first (left-to-right), followed by all internal nodes (postorder).
        Executes in a single O(N) pass and caches the result.
        """
        #if not hasattr(self, '_node_order'):
        leaves = []
        internals = []
        # A single postorder traversal naturally visits leaves left-to-right!
        for node in self.ete_tree.traverse("postorder"):
            if node.is_leaf():
                leaves.append(node.name)
            else:
                internals.append(node.name)
        #self._node_order = leaves + internals
        #return self._node_order
        return leaves + internals

    # --------------------------------------------------------------------------
    # Structural Tree Manipulation Logic
    # --------------------------------------------------------------------------

    @staticmethod
    def graft_subtree(tree: TreeNode, target: TreeNode, graft: TreeNode, name: str, purify: bool = False) -> TreeNode:
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
            if purify:
                SmrtTree.add_pure(new_root)
        else:
            new_internal = TreeNode(name=name)
            p_parent.add_child(new_internal)
            new_internal.add_child(target.detach())
            new_internal.add_child(graft)
            if purify:
                SmrtTree.add_pure(new_internal)
        return tree
    
    @staticmethod
    def copy_lineage(subtree: TreeNode, tag: str = '') -> TreeNode:
        # detach() clears the root (t.up == None)
        subtree = subtree.copy().detach()
        # Make sure the node is now a root
        # assert subtree.up is None, "Subtree copy should be a new root."
        if str:
            # Pure name should be already set due to copy!
            for n in subtree.traverse():
                if n.is_leaf():
                    n.name = f"{n.name}{tag}"
                elif n.name and n.name.startswith("<") and n.name.endswith(">"):
                    n.name = n.name.replace(">", f"{tag}>")
        return subtree

    def _tag_and_graft(self, inner_tree: Tree, target: Tree, parent_tag: str, uid: int, copy_id: int, skip_tagging: bool=False) -> Tuple[Tree, Tree]:
    
        if skip_tagging:
            suffix = ''
        # Extract the surrounding suffix if present (returns empty string if root or '|' is missing)
        else:
            suffix = target.up.name if target.up else ''
            suffix = suffix.partition('|')[2].rstrip('>')
            suffix = '|' + suffix if suffix else ''

        # For copies other than the originally named, update the parent tag, and preppend the new copy ID to the suffix.
        if parent_tag.startswith('<P*'):
            parent_tag = f'<P{uid}>'
            suffix = f"|{uid}.{copy_id}{suffix}"

        # This is an internal node for sure
        new_name = parent_tag[:-1] + suffix + '>'
        graft = SmrtTree.copy_lineage(inner_tree, suffix)
        return SmrtTree.graft_subtree(self.ete_tree, target, graft, name=new_name, purify=True), graft
    
    def _synch_graft(self, graft: Tree):
        # Synchronize the new graft into the wrapper (.pure should be already set!)
        for n in graft.traverse():
            if n.name not in self.node_map:
                self.node_map[n.name] = n
                self.match_map.setdefault(n.pure, []).append(n)
        p_node = graft.up
        self.node_map[p_node.name] = p_node
        self.match_map.setdefault(p_node.pure, []).append(p_node)

    def graft_records(self, inner_tree: Tree, records: List[GraftRecord], uid: int):
        outer_tree = self.ete_tree
        
        # Standard Grafts
        for rec in records:
            if not rec.expanded_targets: continue
            targets = rec.expanded_targets
            targets.sort(key=lambda x: len(x.name))
            for i, target in enumerate(targets):
                # Check if non-P-node parent was already added with its graft (i.e., the original copy was grafted)
                is_first_occurrence = (rec.copy_id == 0 and self.get_node(rec.parent) is None)
                if is_first_occurrence:
                    assert (i == 0), f"The original copy must be untagged and grafted first (raised for target: {target.name} with graft name {rec.parent})."
                outer_tree, graft = self._tag_and_graft(inner_tree, target, rec.parent, uid, rec.copy_id, skip_tagging=is_first_occurrence)
                self.ete_tree = outer_tree
                self._synch_graft(graft)
        
        # Delayed Inner Autopolyploidy Grafts
        for rec in records:
            if rec.expanded_targets: continue
            loc_node = self.get_node(rec.corrected)
            targets = self.match(loc_node.pure)
            targets.sort(key=lambda x: len(x.name))
            for target in targets:
                outer_tree, graft = self._tag_and_graft(inner_tree, target, rec.parent, uid, rec.copy_id)
                self.ete_tree = outer_tree
                self._synch_graft(graft)

    def trim_lineages(self, node_names: List[str], retain: bool = False) -> Tuple[bool, List[Optional[str]], List['SmrtTree']]:
        """
        Safely removes specified clades from the tree sequentially while updating topology.

        This method modifies the underlying ETE3 tree in place. It automatically deletes 
        the resulting "knuckle" (single-child) parent nodes to preserve a bifurcating structure, 
        and safely handles root removal by promoting the remaining child.
        
        Notes:
            - The target nodes are assumed to exist and must be topologically disjoint (non-overlapping).
            - If the root node itself is detached, the ETE3 `detach()` simply passes the root through. 
              The `is_outer` flag is used to track this edge case, indicating whether the current 
              instance still represents the main "outer" tree.

        Return:
            - is_outer (bool): True if this instance remains the main outer tree. False if this instance
                was entirely detached (i.e., the root itself was trimmed), and is now in detached_subtrees.
            - trimmed_up_names (List[Optional[str]]): The names of each deleted parent node from which a 
              lineage was severed. A `None` value indicates the root was trimmed.
            - detached_subtrees (List['SmrtTree']): A list of the detached subtrees wrapped 
              in new `SmrtTree` instances. Empty if `retain` is False.
        """

        """def add_trackers(n: TreeNode, trackers: List[str]):
            if trackers:
                if not hasattr(n, 'H'): n.add_feature('H', [])
                n.H.extend(trackers)"""

        # Track nodes permanently removed from the tree
        dead_keys: defaultdict[str, set] = defaultdict(set) # pure_name -> set of node names removed with that pure name
        trimmed_up_names: List[Optional[str]] = []
        detached_subtrees: List[TreeNode] = [] # Store detached subtrees for potential reattachment if needed (e.g. for root case)
        is_outer: bool = True
        
        for name in node_names:
            n = self.get_node(name)
            assert n is not None, f"Node with name '{name}' not found."
            n_up = n.up
            trimmed_up_names.append(n_up.name if n_up else None)
            n = n.detach()
            detached_subtrees.append(n)

            if n_up is None:
                is_outer = False
                # break
                continue

            # Track all descendants of the detached clade for dictionary removal
            # We don't clear in the None case, because then it is self...
            for d in n.traverse():
                dead_keys[d.pure].add(d.name)

            # Parent is either bypassed (if root) or deleted
            dead_keys[n_up.pure].add(n_up.name)
            
            """trackers = getattr(n_up, 'H', [])""" 

            # Handle the child promotion if the parent is the root, otherwise just delete the parent
            if n_up.up is None:
                children = n_up.get_children()
                assert len(children) == 1, "Unexpected structure when trimming a tree."
                child = children[0]
                self.ete_tree = child.detach() # Ensures root pointer is None
                """add_trackers(self.ete_tree, trackers)"""
            else:
                """add_trackers(n_up.up, trackers)"""
                # ETE3 automatically splices child to grandparent
                # prevent_nondicotomic == True by default - recursively deletes orphans (except the deleted node)
                # we may want to put False here to not risk it, but the rest of the tree should be dichotomic anyway!
                n_up.delete()
                        
        # Clear caches appropriately after all modifications are done
        self._clear_dead(dead_keys)

        # Check cleaning vs trimming consistency
        self.assert_len

        if not retain:
            if not is_outer: self.destroy() # Clean-up
            return is_outer, trimmed_up_names, []
            
        detached_subtrees = [
            SmrtTree(tree_obj=n) if up_name else self 
            for n, up_name in zip(detached_subtrees, trimmed_up_names)
        ]
        return is_outer, trimmed_up_names, detached_subtrees
    
    def to_multi_mul_tree(self, h1_name: str, hx_names: List[str]) -> Optional[Tuple['SmrtTree', TreeNode, List[TreeNode]]]:
        """
        Grafts multiple H-lineages (H2, H3...) onto the H1 branch.
        Used for 'Model' mode to capture all nested copies at once.
        """
        new_tree_obj = self.ete_tree.copy()
        
        # Find H1 (The Stock)
        h1_matches = new_tree_obj.search_nodes(name=h1_name)
        if not h1_matches: return None, None, []
        h1_node = h1_matches[0]

        # Process Hx targets (The Scions)
        hx_nodes_final = []
        
        # Tag generator: *, **, ***, ...
        tags = ["*" * i for i in range(1, len(hx_names) + 1)]

        for h_name, tag in zip(hx_names, tags):
            # We must search by name in the *current* state of new_tree_obj
            # (Note: grafting changes the tree structure, but names persist)
            p_matches = new_tree_obj.search_nodes(name=h_name)
            if not p_matches: raise ValueError(f"Target node '{h_name}' not found in the tree for grafting.") # was continue, but shouldn't it err?!
            p_node = p_matches[0]

            # Nesting check: Cannot graft a parent into a child
            if p_node in h1_node.iter_descendants(): continue

            # Create the copy from H1 (the source of the introgression)
            h_copy = SmrtTree.copy_lineage(h1_node, tag)

            # Graft with a unique internal name for the graft point
            graft_name = f"<P{tag}>"
            new_tree_obj = SmrtTree.graft_subtree(new_tree_obj, p_node, h_copy, graft_name)
            
            hx_nodes_final.append(h_copy)

        # Wrap the modified tree in a new SmrtTree object to re-index and refresh
        new_smrt = SmrtTree(tree_obj=new_tree_obj)
        
        # We must re-find H1 because the root might have changed during grafting
        h1_final = new_smrt.get_node(h1_node.name)
        
        return new_smrt, h1_final, hx_nodes_final

    # --------------------------------------------------------------------------
    # I/O and String Conversion (ETE3-based, with optional name formatting)
    # --------------------------------------------------------------------------

    def write_forms(self, output_dir: Path):

        out_silt = output_dir / "final_single_label_form.tre"
        out_mult = output_dir / "final_multree.tre"
        
        with open(out_silt, 'w') as f:
            f.write(self.ete_tree.write(format=8))

        with open(out_mult, 'w') as f:
            f.write(self.to_mult_str(internals=False))

    def to_mult_str(self, internals: bool=True) -> str:
        name_to_pure = {n.name: n.pure for n in self.ete_tree.traverse()}
        return self.to_str(internals=internals, name_formatter=lambda name: name_to_pure.get(name, 'Error'))

    def to_mult(self, internals: bool=True) -> Tree:
        mult_str = self.to_mult_str(internals=internals)
        return Tree(mult_str, format=8 if internals else 9)

    def mark_node_to_str(self, node_to_mark: TreeNode, symbol: str = "+", internals: bool=True) -> str:
        """
        Node_to_mark is the node in the ete_tree that should be marked with a "+" in the string output.
        Generally it is the H1 node after a new inference.
        Returns a string representation of the tree with +/* marked nodes.
        """
        marked_nodes = [n for n in node_to_mark.traverse()] if node_to_mark else []
        # Rename, create str, undo rename
        for n in marked_nodes:
            n.name = n.name + symbol if n.is_leaf() else n.name [:-1] + symbol + '>'
        marked_str = self.to_str(internals=internals)
        for n in marked_nodes:
            n.name = n.name.replace(symbol, "")
        return marked_str
    
    def to_str(self, internals: bool=True, name_formatter=None, **kwargs) -> str:
        """
        Wrapper for _to_str to work on self.
        """
        return self._to_str(self.ete_tree, internals=internals, name_formatter=name_formatter, **kwargs)

    @staticmethod
    def _to_str(ete_tree, internals: bool=True, name_formatter=None, **kwargs) -> str:

        # Determine the ETE3 format number based on your parameter
        fmt_num = 8 if internals else 9
        # Checking children > 1 prevents printing single-leaf tree's label twice
        root_name = str(ete_tree.name) if internals and len(ete_tree) > 1 else ""
        
        if name_formatter:
            # Bind extra arguments (like maps=, dups=) to the formatter
            if kwargs:
                name_formatter = partial(name_formatter, **kwargs)
                
            # Save original names and apply formatting in-place
            original_names = {}
            for node in ete_tree.traverse():
                original_names[node] = node.name
                raw_name = str(node.name) if node.name else ""
                fmt_name = str(name_formatter(raw_name)) if raw_name else ""
                node.name = fmt_name
            # Do the same for the root name if needed
            if root_name:
                root_name = str(name_formatter(root_name))
                # No need to revert root name as it's not part of the tree object
                
            base_str = ete_tree.write(format=fmt_num)
            
            # Revert the names to the original to avoid side effects on the tree object
            for node, orig_name in original_names.items():
                node.name = orig_name
            
        else:
            # Standard ETE3 writing (No formatter provided)
            base_str = ete_tree.write(format=fmt_num)
        
        return base_str[:-1] + root_name + ";"

    def to_rt(self) -> ReticulateTree:
        tree_copy = self.ete_tree.copy()
        for n in tree_copy.traverse():
            n.name = n.pure
        return ReticulateTree(tree_copy)

    # --------------------------------------------------------------------------
    # GROUP COLLAPSING LOGIC (Object-based, run once per iter)
    # --------------------------------------------------------------------------

    def compute_groups(self, mul_data: 'MulTree', registry: NameRegistry, 
                       h1_sisters: Set[str] = None, hx_sisters_list: List[Set[str]] = None) -> GroupData:
        """
        Registry-Optimized O(N) implementation.
        Uses integer IDs for Set operations (Union/IsSubset) to achieve significant speedup.
        """
        h1_target_ids = {registry.get_id(name) for name in mul_data.h_clade}
        # Cache: node -> (species_id_set, leaf_names_list, active_roots)
        # species_id_set: Set[int] - much faster than Set[str]
        groups, singles, node_info = {}, {}, {}

        # Raw ete3 traversal - Maximum speed!
        for node in self.ete_tree.traverse("postorder"):
            if node.is_leaf():
                # Extract name and convert to ID
                sp_name = splitSpec(node.name)
                sp_id = registry.get_id(sp_name)
                is_h1 = sp_id in h1_target_ids
                
                s_set, l_list = {sp_id}, [node.name]
                if is_h1:
                    singles[node.name] = [] 
                    a_roots = [node.name]
                else:
                    a_roots = []
                node_info[node] = (s_set, l_list, a_roots)
                
            else:
                u_s_set, u_l_list, u_a_roots = set(), [], []
                all_h1_descendants, total_species_count = True, 0
                
                for child in node.children:
                    c_s_set, c_l_list, c_a_roots = node_info[child]
                    u_s_set.update(c_s_set)
                    u_l_list.extend(c_l_list)
                    u_a_roots.extend(c_a_roots)
                    total_species_count += len(c_s_set)
                    
                    # Integer set subset check is highly optimized
                    if not c_s_set.issubset(h1_target_ids):
                        all_h1_descendants = False
                
                if all_h1_descendants and (len(u_s_set) == total_species_count) and len(node.children) > 1:
                    # Valid Group
                    for r in u_a_roots:
                        groups.pop(r, None)
                        singles.pop(r, None)
                    groups[node.name] = [u_l_list, []]
                    u_a_roots = [node.name]
                
                node_info[node] = (u_s_set, u_l_list, u_a_roots)

        # --- Post-Processing ---
        
        def fill_anc_leaves(n_name, target_dict):
            n_obj = self.get_node(n_name)
            if not n_obj or not n_obj.up: return
            p_obj = n_obj.up
            if p_obj in node_info and n_obj in node_info:
                p_leaves = node_info[p_obj][1]
                n_leaves_set = set(node_info[n_obj][1])
                anc_list = [l for l in p_leaves if l not in n_leaves_set]
                if target_dict is groups:
                    target_dict[n_name][1] = anc_list
                else: # is singles
                    target_dict[n_name] = anc_list

        for g_name in groups: fill_anc_leaves(g_name, groups)
        for s_name in singles: fill_anc_leaves(s_name, singles)

        # --- Fixes Logic ---

        # List of List[int], List of (List[int], int)
        final_ambiguous, final_fixed = [], [] 

        # Sister checking
        if mul_data.h1_node and (h1_sisters is None):
            h1_sisters, hx_sisters_list = mul_data.get_sister_clades()

        def check_fix(unit_nodes, anc_leaves):
            # Convert to Set for fast subset math
            group_sis_specs = {splitSpec(n) for n in anc_leaves}
            if group_sis_specs:
                if h1_sisters and group_sis_specs.issubset(h1_sisters):
                    # Index 0 corresponds to the Base/H1 target
                    final_fixed.append((registry.get_ids(unit_nodes), 0))
                    return
                for idx, hx_sisters in enumerate(hx_sisters_list):
                    if hx_sisters and group_sis_specs.issubset(hx_sisters):
                        # Index idx + 1 corresponds to H2 (1), H3 (2), etc.
                        final_fixed.append((registry.get_ids(unit_nodes), idx + 1))
                        return
            final_ambiguous.append(registry.get_ids(unit_nodes))

        for g_leaves, anc_leaves in groups.values(): check_fix(g_leaves, anc_leaves)
        for s_name, anc_leaves in singles.items(): check_fix([s_name], anc_leaves)

        return GroupData(final_ambiguous, final_fixed)

    # --------------------------------------------------------------------------
    # Utilities and Pickling
    # --------------------------------------------------------------------------

    def copy(self) -> 'SmrtTree':
        # The faster newick-extended method should work too as it preserves attributes added as features (e.g. pure),
        # and it is faster, but I trust the cPickle method more (and it's already faster than deepcopy), so I'll keep it for now.
        return SmrtTree(tree_obj=self.ete_tree.copy(method="cpickle"))

    def destroy(self) -> None:

        # Drop the root pointer
        tree = self.ete_tree
        self.ete_tree = None

        # Sever circular references (primary GC bottleneck)
        if tree is not None:
            for node in tree.traverse("postorder"):
                node.up = None
                node.children = []
            del tree

        # Clear caches
        self.node_map.clear()
        self.match_map.clear()
        self.flat_tree = None

    @property
    def assert_len(self):
        assert len(self) == len(self.ete_tree), f"Length mismatch: {len(self)} by names vs {len(self.ete_tree)} by tree leaves"

    @property
    def assert_topology(self):
        """Checks bifurcation and root legality"""
        assert self.ete_tree.is_root(), "Tree is not rooted to None"
        assert not any(len(n.children) not in (0, 2) for n in self.ete_tree.traverse()), "Tree has non-bifurcating nodes"

    def __len__(self):
        # Returns the number of leaves
        # Assume bifurcating tree with unique names, so num_nodes = 2 * num_leaves - 1
        return (len(self.node_map) + 1) // 2

    def __getstate__(self):
        return self.ete_tree
    
    def __setstate__(self, state):
        self.ete_tree = state
        self.refresh()

@dataclass(slots=True)
class TreeCache:
    """O(1) Data structure holding the global tree and precomputed global state for ploidy constraints. Lazy loaded on demand."""
    st: SmrtTree
    populated: bool = field(default=False, init=False)
    ploidy_stats: Optional[Dict[str, Tuple[int, int]]] = None
    clade_counts: Optional[Dict[str, Dict[str, int]]] = None

    def populate(self, ploidies: Dict[str, int], is_strict: bool) -> None:
        if not self.populated:
            self.ploidy_stats = self.compute_ploidy_stats(self.st, ploidies, is_strict)
            self.clade_counts = self.st.clade_pure_counts
            self.populated = True

    @staticmethod
    def _count_effective_lineages(tree: Tree) -> Dict[str, Tuple[int, int]]:
        """
        Counts effective lineages for each species in the current ST.
        Populates self.ploidy_stats: {species: (number_of_pure_groups, max_size_of_pure_group)}
        Logic:
        1. Pure Groups: A clade where all descendants are the same species.
        2. Polytomies: Siblings of the same pure species at a mixed node are aggregated 
           into a single 'group' (e.g., (x,x,y) counts as one x-group of size 2).
        3. Nested Pure Groups: Only the maximal pure group is counted (e.g., ((x,x),x) is 1 group of size 3).
        """
        counts: Dict[str, List[int]] = {}
        node_states = {} # Cache: node -> (pure_species, size) or None
        
        for node in tree.traverse("postorder"):
            if node.is_leaf():
                node_states[node] = (node.pure, 1)
                continue
                
            child_states = [node_states[c] for c in node.children]
            first_sp = child_states[0][0] if child_states and child_states[0] else None
            
            all_same = True
            total_size = 0
            
            for state in child_states:
                if state is None or state[0] != first_sp:
                    all_same = False
                if state is not None:
                    total_size += state[1]
                    
            if all_same and first_sp is not None:
                node_states[node] = (first_sp, total_size)
            else:
                # Mixed clade: finalize pure children
                current_level_groups: Dict[str, int] = {}
                for state in child_states:
                    if state is not None:
                        sp, size = state
                        current_level_groups[sp] = current_level_groups.get(sp, 0) + size
                
                # Update global counts
                for sp, size in current_level_groups.items():
                    if sp not in counts: counts[sp] = [0, 0]
                    counts[sp][0] += 1
                    if size > counts[sp][1]: counts[sp][1] = size
                
                node_states[node] = None

        # Handle Root State
        root_state = node_states.get(tree)
        if root_state is not None:
            sp, size = root_state
            if sp not in counts: counts[sp] = [0, 0]
            counts[sp][0] += 1
            if size > counts[sp][1]: counts[sp][1] = size

        return {k: tuple(v) for k, v in counts.items()}
        
    @staticmethod
    def compute_ploidy_stats(st: SmrtTree, ploidies: Dict[str, int], is_strict: bool) -> Dict[str, Tuple[int, int]]:
        if is_strict:
            # Strict Mode
            ploidy_stats = {}
            for sp in ploidies.keys():
                count = len(st.match(sp))
                ploidy_stats[sp] = (count, 1 if count > 0 else 0)
            return ploidy_stats
        else:
            # Lineage-Based Mode
            return TreeCache._count_effective_lineages(st.ete_tree)
        
@dataclass(slots=True, frozen=True)
class MulTree:
    mt: SmrtTree
    h_clade: List[str] = field(default_factory=list)
    # Storing OBJECTS optimizes the Reconciler (no lookups)
    # These are safe to pickle TO workers, but shouldn't be used to map back TO main.
    h1_node: Optional[TreeNode] = None 
    # Replaced single h2_node with hx_nodes list
    hx_nodes: List[TreeNode] = field(default_factory=list)

    # Safety controls to not abuse stale wrappers
    _stale_stars: bool = field(default=False, init=False)

    # For mode compatibility, h2_node can return the first element or None
    @property
    def h2_node(self) -> Optional[TreeNode]:
        return self.hx_nodes[0] if self.hx_nodes else None
    
    @property
    def hx_sisters(self) -> List[Optional[TreeNode]]:
        sisters = []
        for hx_node in self.hx_nodes:
            sis = self.mt.get_sis(hx_node)
            sisters.append(sis)
        return sisters
    
    @property
    def h1_sister(self) -> Optional[TreeNode]:
        if self.h1_node is None: return None
        return self.mt.get_sis(self.h1_node)
    
    def to_marked_str(self, internals: bool=True) -> str:
        return self.mt.mark_node_to_str(self.h1_node, internals=internals)
    
    # A routine to init from a history event
    @classmethod
    def from_history_event(cls, event: Dict[str, Any]) -> 'MulTree':
        tree = Tree(event['best_mt'], format=1)
        tree_wrapper = SmrtTree(tree_obj=tree)
        h_name = event['h_name']
        return cls(
            mt=tree_wrapper,
            h_clade=event['h_leaves'],
            h1_node=tree_wrapper.get_node(h_name),
            hx_nodes=cls._get_starred_hx_nodes(tree_wrapper, event['h_locs'], h_name)
        )

    @staticmethod
    def _get_starred_hx_nodes(tree_wrapper: SmrtTree, h_locs: List[str], h_name: str) -> List[TreeNode]:
        hx_nodes = []
        for i in range(1, len(h_locs)):
            stars = '*' * i
            if h_name.endswith('>'):
                to_match = f"{h_name[:-1]}{stars}>"
            else:
                to_match = f"{h_name}{stars}"
            h_node = tree_wrapper.get_node(to_match)
            assert h_node == tree_wrapper.get_sis(tree_wrapper.get_node(h_locs[i])), f"Expected {to_match} to be sister of {h_locs[i]}"
            hx_nodes.append(h_node)
        return hx_nodes
    
    def build_h_copy_map(self) -> Dict[str, int]:
        """Creates an O(1) lookup mapping a leaf name to its homoeologous copy index."""
        mt_node_to_copy_idx = {}
        for ln in self.h1_node.iter_leaf_names():
            mt_node_to_copy_idx[ln] = 0
        for x, hx_node in enumerate(self.hx_nodes, 1):
            for ln in hx_node.iter_leaf_names():
                mt_node_to_copy_idx[ln] = x
        return mt_node_to_copy_idx

    def rename_marked_nodes(self, depth: int, copy_offset: int=0, skip_p_tag: bool=False) -> Dict[str, Set[str]]:
        """
        A suffix is added to the name of all *nodes* in a set of the mapping.
        This is used to mark the Hx lineages after grafting.
            Leaves:     Species*        -> Species|1.0
                        Species|1.0**   -> Species|1.0|2.1
            Internals:  <1*>            -> <1|1.0>
                        <1|1.0*>        -> <1|1.0|2.0>
            P nodes:    <P**>           -> <Pi|i.1> (e.g. <P1|1.1>)
                        <P1|1.1*>       -> <P1|1.1|2.0>
        Returns: A mapping from suffix to the set of original *names* that were suffixed.
        """
        # Note: in the current implementation, the sets are disjoint
        # If this is no longer true, we should first build a renaming map,
        # then, apply it in a second pass to avoid conflicts.

        suffix_name_map = defaultdict(set)
        for k, hx_node in enumerate(self.hx_nodes, 1):

            parent = hx_node.up
            x = parent.name.count('*')
            stars = '*' * x
            assert x == k, f"Expected {k} stars in parent name {parent.name} for hx_node {hx_node.name}, but found {x}."
            assert parent.name.startswith(f"<P*"), f"Expected parent name of the form <P[*]>, but got {parent.name}"

            # Generate suffixes for the multiple Hx case
            # H2 gets i.1, H3 gets i.2, etc.
            # This means that all modes have the same suffixing scheme
            suffix = f"{depth}.{copy_offset+x}"

            marked_set = set()

            # Mark the <P> node
            old_name = parent.name
            marked_set.add(old_name)
            # P nodes of different iterations can't have the same pure name!
            # But, they also must have the same [i] if coming from the same iteration!
            parent.pure = f"<P{depth}>" # No need to append or prepend, because <P[*]> is a fresh P node always!
            if skip_p_tag:
                parent.name = parent.pure
            else:
                parent.name = parent.pure[:-1] + f"|{suffix}>"

            # Mark all other nodes in this H lineage copy
            # Includes hx_node and descendants
            for node in hx_node.traverse('postorder'):
                old_name = node.name
                marked_set.add(old_name)

                new_name = old_name.replace(stars, f"|{suffix}")

                # Apply Rename using Wrapper (handles node_map updates)
                self.mt._rename_node_no_reindex(old_name, new_name)

            suffix_name_map[suffix] = marked_set

        # Must refresh after name modification
        self.mt.refresh()
        self._stale()
        return suffix_name_map
  
    def partition(self, get_inners: str = 'h1') -> Tuple[Optional[SmrtTree], Union[Optional[SmrtTree], List[SmrtTree]], List[Optional[str]]]:
        """
        Partitions the Mul Tree into outer and inner components.
        get_inners: 'h1' to retain only the H1 lineage as inner, 'all' to retain all H lineages as inner, 'none' to not extract any inner lineages.

        Return:
            - outer_wrapper: SmrtTree representing the outer tree (None if the root was trimmed)
            - retain: None, SmrtTree, or List[SmrtTree] representing the retained inner lineage(s) based on get_inners parameter
            - parent_names: List of names of the parent nodes from which lineages were trimmed (None for root)
        """
        # Putting h1 last ensure it gets the original "real" up_name, not a <P[*]>, in cases of autopolyploidy
        inner_lineages = [n.name for n in self.hx_nodes] + [self.h1_node.name]

        is_outer, parent_names, inner_wrappers = self.mt.trim_lineages(
            inner_lineages, retain=(get_inners != 'none')
        )

        inner_wrappers = inner_wrappers[-1:] + inner_wrappers[:-1]
        parent_names = parent_names[-1:] + parent_names[:-1]

        if not is_outer:
            outer_wrapper = None
        else:
            outer_wrapper = self.mt

        # Prevent destroying the outer wrapper
        self.destroy(disconnect=True)

        retained = None
        if get_inners == 'h1':
            retained = inner_wrappers[0]
        elif get_inners == 'all':
            retained = inner_wrappers
        elif get_inners != 'none':
            raise ValueError(f"Invalid get_inners value: {get_inners}. Expected 'h1', 'all', or 'none'.")

        return outer_wrapper, retained, parent_names

    # --- Sister Clade Logic ---
    # The static methods below only run on MTs / STs, that's why they are in MulTree and not SmrtTree, to avoid confusion about applicability.
    # For speed and pickling reasons, they are static and operate on the SmrtTree wrapper and use node names for lookups, rather than TreeNode objects which may become stale after grafting.

    @staticmethod
    def _find_node_by_clade(tree: SmrtTree, target_leaves: Set[str]) -> Any:
        # Use Dictionary Lookup O(1) instead of search_nodes O(N)
        leaf_nodes = []
        for t in target_leaves:
            # Use get_node from the wrapper
            node = tree.get_node(t)
            if node: leaf_nodes.append(node)
        if not leaf_nodes: return None
        lca = tree.ete_tree.get_common_ancestor(leaf_nodes)
        lca_leaves = set(lca.iter_leaf_names())
        if lca_leaves == target_leaves: return lca
        return None
    
    @staticmethod
    def _get_sister_clade_labels(node_obj: Any) -> List[str]:
        """
        Given a node object, returns the set of leaf labels in its sister clades.
        Applicable only to the species tree, so labels are expected to be the raw leaf names (e.g. "Species" or "Species*") without GeneID.
        """
        if not node_obj or not node_obj.up: return []
        # Must be l.name (not l.pure) to preserve the '*' for disjoint checks!
        return [
            l_name 
            for sis in node_obj.up.children if sis != node_obj 
            for l_name in sis.iter_leaf_names()
        ]

    def get_sister_clades(self) -> Tuple[Set[str], List[Set[str]]]:
        """
        Calculates sister clades entirely internally using SmrtTree helpers.
        Returns:
          1. h1_sisters: Set of names indicating H1 (Base) placement.
          2. hx_sisters_list: List[Set[str]] where index 0 -> H2, 1 -> H3.
        """
        if self._stale_stars:
            raise RuntimeError("Cannot compute sister clades after star-based renaming.")
        if self.h1_node is None: return set(), []
            
        h1_target = set(self.h_clade)
        mt = self.mt
        n1_obj = MulTree._find_node_by_clade(mt, h1_target)
        h1_sisters = MulTree._get_sister_clade_labels(n1_obj) if n1_obj else []

        hx_sisters_list = []
        # Targets (mul_data.hx_nodes) is guaranteed to be a list (either empty or populated)
        # Can't use the hx_node object directly due to risk of stale references, so we find the node by clade each time.
        for idx, _ in enumerate(self.hx_nodes):
            # Dynamically find the node in the current tree topology
            # to avoid stale node references yielding incorrect sister clades.
            target_suffix = "*" * (idx + 1)
            hx_target = {f"{x}{target_suffix}" for x in self.h_clade}
            n_obj = MulTree._find_node_by_clade(mt, hx_target)
            
            if n_obj:
                sisters = MulTree._get_sister_clade_labels(n_obj)
                if not set(h1_sisters).isdisjoint(set(n_obj.iter_leaf_names())): 
                    h1_sisters = []
                if n1_obj and not set(sisters).isdisjoint(set(n1_obj.iter_leaf_names())): 
                    sisters = []
                # Append Set of clean names directly to the list
                hx_sisters_list.append({s.replace("*", "") for s in sisters})
            else:
                hx_sisters_list.append(set())

        h1_sisters = {x.replace("*", "") for x in h1_sisters}
        return h1_sisters, hx_sisters_list
    
    def destroy(self, disconnect: bool = True):
        """Marks the MulTree as consumed to prevent further use of it."""
        if self.mt is not None and not disconnect:
            self.mt.destroy()
        object.__setattr__(self, 'mt', None)
        object.__setattr__(self, 'h_clade', [])
        # Drop ETE3 node pointers so they don't prevent Python from freeing the memory
        object.__setattr__(self, 'h1_node', None)
        object.__setattr__(self, 'hx_nodes', [])
        self._stale()

    def _stale(self):
        """Marks the MulTree as having stale node references, which prevents any future star-based lookups."""
        object.__setattr__(self, '_stale_stars', True)

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
            rev_map = defaultdict(list)
            for gt_node, sp_nodes in self.cor.items():
                for sp in sp_nodes:
                    rev_map[sp].append(gt_node)
            object.__setattr__(self, '_rev', dict(rev_map))
        return self._rev
    
    def __getitem__(self, key: str) -> List[str]:
        return self.cor[key]
    
@dataclass(slots=True, frozen=True)
class ReconResult:
    score: Union[int, float]
    maps: List[Map]

@dataclass(slots=True, frozen=True)
class TaskResult:
    """Return payload from a GRAMPA step."""
    
    # List of (MulTreeID, Score) tuples, sorted by Score ASC
    sorted_scores: List[Tuple[int, Union[int, float]]]
    
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

    def mt_score(self, rank=0) -> Union[int, float]:
        """Returns the score of the best performing mt."""
        return self.sorted_scores[rank][1]

    @property
    def input_score(self) -> Union[int, float]:
        """O(1) retrieval of the input species tree score."""
        if self._input_rank == -1: return float('inf')
        # Return tuple item [1] (score), as [0] (index) is always 0
        return self.sorted_scores[self._input_rank][1]
    
    @property
    def unpacked_min_mt(self) -> Tuple[Union[int, float], int, MulTree]:
        """Returns a tuple of (score, mt_id, mt_object) for easy unpacking of the best performing mt (incl. input)."""
        min_score = self.mt_score()
        min_idx = self.mt_idx()
        min_mult = self.mul_trees[min_idx]
        return min_score, min_idx, min_mult

# --- Type aliases ---

HistoryType = Dict[Tuple[int, int], Dict[str, Any]]
ConcurrTask = Tuple[SmrtTree, Dict[int, SmrtTree], Tuple[int, Optional[int]]]

