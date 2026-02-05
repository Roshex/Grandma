import math
import array
from ete3 import Tree, TreeNode
from typing import List, Dict, Optional, Tuple, Any
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
        
        # 1. Assign Integer IDs to all nodes (Preorder to keep IDs usually topological)
        node_to_id = {node: i for i, node in enumerate(ete_tree.traverse("preorder"))}
        num_nodes = len(node_to_id)
        
        # Arrays
        parents = array.array('i', [-1] * num_nodes)
        children_flat = array.array('i')
        children_start = array.array('i', [0] * (num_nodes + 1))
        node_to_name_id = array.array('i', [-1] * num_nodes)
        
        # Fill Topology
        cursor = 0
        sorted_nodes = sorted(node_to_id.keys(), key=lambda n: node_to_id[n])
        
        name_id_to_node_id = {}
        node_id_to_name_id = {}

        for node in sorted_nodes:
            nid = node_to_id[node]
            
            # Needs to be the unique raw name (with * for MUL trees) 
            raw_name = str(node.name) if node.name else ""
            
            name_idx = -1
            if node.is_leaf():
                # Extract species name. 
                # For GT: "Gene_Species" -> "Species"
                # For ST (MUL): "Species*" -> "Species*" (Preserves distinction)
                # A clean name is needed to find matching species in ST
                sp_name = raw_name.split("_")[-1] 
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

            # Parent
            if node.up:
                parents[nid] = node_to_id[node.up]
            
            # Children
            children_start[nid] = cursor
            for child in node.children:
                cid = node_to_id[child]
                children_flat.append(cid)
                cursor += 1
        
        children_start[num_nodes] = cursor

        # 2. Postorder Traversal (Int IDs)
        postorder = array.array('i', [node_to_id[n] for n in ete_tree.traverse("postorder")])

        # 3. Euler Tour & RMQ Construction
        euler_nodes = array.array('i')
        euler_depths = array.array('i')
        first_visit = array.array('i', [-1] * num_nodes)
        
        def dfs(u, d):
            first_visit[u] = len(euler_nodes)
            euler_nodes.append(u)
            euler_depths.append(d)
            
            # Iterate children using CSR
            start = children_start[u]
            end = children_start[u+1]
            for i in range(start, end):
                v = children_flat[i]
                dfs(v, d + 1)
                euler_nodes.append(u)
                euler_depths.append(d)

        root = ete_tree.get_tree_root()
        dfs(node_to_id[root], 0)
        
        # Build Sparse Table
        L = len(euler_nodes)
        if L > 0:
            k_max = int(math.log2(L)) + 1
            rmq = [array.array('i', [0] * L) for _ in range(k_max)]
            
            # Initialize M[0]
            for i in range(L):
                rmq[0][i] = i
            
            # Compute remaining
            for j in range(1, k_max):
                for i in range(L - (1 << j) + 1):
                    idx1 = rmq[j-1][i]
                    idx2 = rmq[j-1][i + (1 << (j-1))]
                    if euler_depths[idx1] < euler_depths[idx2]:
                        rmq[j][i] = idx1
                    else:
                        rmq[j][i] = idx2
        else:
            rmq = []

        return FlatTree(
            num_nodes=num_nodes,
            root_id=node_to_id[root],
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

    __slots__ = ['ete_tree', 'node_map', 'lca_cache', 'flat_tree']

    def __init__(self, newick: str = None, tree_obj: Tree = None):
        if tree_obj:
            self.ete_tree = tree_obj
        else:
            if newick and not newick.strip().endswith(";"):
                newick += ";"
            self.ete_tree = Tree(newick, format=0) 
            
        self.node_map: Dict[str, TreeNode] = {}
        self.lca_cache: Dict[Tuple[int, ...], TreeNode] = {}
        self.flat_tree: Optional[FlatTree] = None
        
        self._index_nodes()
        #self._cache_depths()

    def _index_nodes(self):
        self.node_map = {} 
        i = 1
        for node in self.ete_tree.traverse("postorder"):
            if not node.is_leaf():
                if not node.name:
                    node.name = f"<{i}>"
                    i += 1
            else:
                node.name = str(node.name).strip()
            
            self.node_map[node.name] = node

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
        else:
            new_internal = TreeNode(name=name)
            p_parent.add_child(new_internal)
            new_internal.add_child(target.detach())
            new_internal.add_child(graft)
        return tree

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

        h2_subtree = h1_node.copy()
        for leaf in h2_subtree.iter_leaves():
            leaf.name = f"{leaf.name}*"

        new_tree_obj = SmrtTree.graft_subtree(new_tree_obj, p_node, h2_subtree)

        for n in new_tree_obj.traverse():
            if n.name and n.name.startswith("<") and n.name.endswith(">"):
                n.name = None
        
        new_smrt = SmrtTree(tree_obj=new_tree_obj)
        return new_smrt, h1_node, h2_subtree # Note: returns objects in the new tree context

    def to_string(self, internal_labels=True) -> str:
        root_name = str(self.ete_tree.name) if internal_labels else ""
        return self.ete_tree.write(format=8 if internal_labels else 9)[:-1]+root_name+";"
    
    def __getstate__(self):
        return self.ete_tree
    
    def __setstate__(self, state):
        self.ete_tree = state
        self.refresh()

    # Legacy

    def report_labels(self):
        """Prints the species tree with internal nodes labeled."""
        print("\n# The input species tree with internal nodes labeled:")
        print(self.to_string(internal_labels=True) + "\n")

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
    fixed_groups: List[Tuple[List[int], str]]

@dataclass(slots=True, frozen=True)
class MulTree:
    mt: SmrtTree
    h_clade: List[str] = field(default_factory=list)
    # Storing OBJECTS optimizes the Reconciler (no lookups)
    # These are safe to pickle TO workers, but shouldn't be used to map back TO main.
    h1_node: Optional[TreeNode] = None 
    h2_node: Optional[TreeNode] = None

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

