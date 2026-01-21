from ete3 import Tree, TreeNode
from typing import List, Set, Dict, Optional, Tuple
from dataclasses import dataclass, field

# replacing GrandmaTree, just the name is different
class SmrtTree:
    """Wrapper around ETE3 Tree to provide GRAMPA-specific functionality."""

    __slots__ = ['ete_tree', 'node_map', 'lca_cache']

    def __init__(self, newick: str = None, tree_obj: Tree = None):
        if tree_obj:
            self.ete_tree = tree_obj
        else:
            # add ; if missing
            if newick and not newick.strip().endswith(";"):
                newick += ";"
            # Format 1 allows internal node names
            self.ete_tree = Tree(newick, format=0) 
            
        self.node_map: Dict[str, TreeNode] = {} # Map name -> TreeNode
        
        # OPTIMIZATION: Cache for LCA lookups to avoid tree traversal overhead
        self.lca_cache: Dict[Tuple[int, ...], TreeNode] = {}
        
        self._index_nodes()
        self._cache_depths() # FIXED: This must be enabled!

    def _index_nodes(self):
        """
        Assigns <x> labels to internal nodes if missing.
        Uses postorder to match legacy GRAMPA bottom-up labeling.
        """
        self.node_map = {} # Reset
        i = 1
        for node in self.ete_tree.traverse("postorder"):
            if not node.is_leaf():
                if not node.name:
                    node.name = f"<{i}>"
                    i += 1
            else:
                node.name = str(node.name).strip()
            
            self.node_map[node.name] = node

    def refresh(self):
        """
        Rebuilds indices and caches. Call this after unpickling or structural changes.
        """
        self._index_nodes()
        self._cache_depths()
        self.build_lca_cache()
        
    def _cache_depths(self):
        """
        Calculates node depth and attaches it DIRECTLY to the ete3 node object.
        Allows O(1) attribute access (node.fast_depth).
        """
        # Traverse preorder: parent is always processed before child
        for node in self.ete_tree.traverse("preorder"):
            if node.is_root():
                node.add_feature("fast_depth", 0)
            else:
                node.add_feature("fast_depth", node.up.fast_depth + 1)
            
            # Optimization for MUL-trees: Cache the "clean" name to avoid .split()/.replace() later
            clean = node.name.replace("*", "") if node.is_leaf() else ""
            node.add_feature("clean_name", clean)

    def build_lca_cache(self):
        """
        Pre-computes or prepares the tree for heavy LCA queries.
        This is called before the heavy permutation loops.
        """
        self.lca_cache = {}
        # We could pre-fill, but lazy caching in get_lca is usually better 
        # to avoid storing N^2 pairs if not all are needed.
        pass

    '''def build_lca_cache(self):
        """Prepares the tree for heavy LCA queries."""
        # Clear existing cache to avoid stale references
        self.lca_cache = {}
        # Ensure depths are cached (fast_depth feature)
        self._cache_depths()'''

    def get_node(self, name: str) -> Optional[TreeNode]:
        return self.node_map.get(name)

    def get_lca(self, species_list: List[str]) -> Optional[TreeNode]:
        """String-based LCA lookup (Legacy support)"""
        nodes = [self.node_map[name] for name in species_list]
        if not nodes: return None
        if len(nodes) == 1: return nodes[0]
        return self.ete_tree.get_common_ancestor(nodes)

    def get_lca_obj(self, nodes: List[TreeNode]) -> Optional[TreeNode]:
        """
        OPTIMIZATION: Object-based LCA lookup.
        Uses caching to avoid tree traversal.
        """
        if not nodes: return None
        if len(nodes) == 1: return nodes[0]
        
        # Sort by memory address (id) to create a consistent key for the pair/set
        # Tuple creation is very fast in Python
        key = tuple(sorted(id(n) for n in nodes))
        
        if key in self.lca_cache:
            return self.lca_cache[key]
        
        lca = self.ete_tree.get_common_ancestor(nodes)
        self.lca_cache[key] = lca
        return lca

    def get_clade_leaves(self, node_name: str) -> Set[str]:
        node = self.get_node(node_name)
        if not node: return set()
        return {leaf.name for leaf in node.iter_leaves()}
    
    def to_mul_tree(self, h_node_label: str, p_node_label: str) -> Optional[Tuple['SmrtTree', TreeNode, TreeNode]]:
        """
        Generates a MUL-tree by copying h_node subtree to p_node location.
        Returns: (NewSmrtTree, New_H1_Node, New_H2_Node)
        """
        # Work on a deep copy
        new_tree_obj = self.ete_tree.copy()
        
        # 1. Locate H1 (original) and H2 (target parent) in the new tree
        h_matches = new_tree_obj.search_nodes(name=h_node_label)
        p_matches = new_tree_obj.search_nodes(name=p_node_label)
        
        if not h_matches or not p_matches: return None, None, None
        h1_node = h_matches[0]
        p_node = p_matches[0]

        # Prevent creating loops
        if p_node in h1_node.iter_descendants(): return None, None, None

        # 2. Create the H2 clade (Copy of H1)
        h2_subtree = h1_node.copy()
        for leaf in h2_subtree.iter_leaves():
            leaf.name = f"{leaf.name}*"
            
        # 3. Graft H2 at P_Node location
        p_parent = p_node.up
        
        if p_parent is None:
            # P is root: Create new root above P and H2
            new_root = TreeNode()
            new_root.add_child(p_node.detach())
            new_root.add_child(h2_subtree)
            new_tree_obj = new_root
        else:
            # Standard graft: Create new internal node between Parent and P
            new_internal = TreeNode()
            p_parent.add_child(new_internal)
            p_node.detach()
            new_internal.add_child(p_node)
            new_internal.add_child(h2_subtree)

        # 4. Cleanup old labels
        for n in new_tree_obj.traverse():
            if n.name and n.name.startswith("<") and n.name.endswith(">"):
                n.name = None
        
        # 5. Return wrapped tree and the KEY NODES
        # The constructor will handle re-indexing and caching
        new_smrt = SmrtTree(tree_obj=new_tree_obj)
        
        # We must re-find the nodes in the finalized object or use the references we have.
        # Since new_smrt wraps new_tree_obj, h1_node is valid. 
        # h2_subtree is the root of the grafted clade (the H2 node).
        return new_smrt, h1_node, h2_subtree

    '''def to_mul_tree(self, h_node_label: str, p_node_label: str) -> Optional['SmrtTree']:
        """Generates a MUL-tree by copying h_node subtree to p_node location."""
        new_tree_obj = self.ete_tree.copy()
        
        h_matches = new_tree_obj.search_nodes(name=h_node_label)
        p_matches = new_tree_obj.search_nodes(name=p_node_label)
        
        if not h_matches or not p_matches: return None
        h_node = h_matches[0]
        p_node = p_matches[0]

        if p_node in h_node.iter_descendants(): return None

        h_subtree_copy = h_node.copy()
        for leaf in h_subtree_copy.iter_leaves():
            leaf.name = f"{leaf.name}*"
            
        p_parent = p_node.up
        if p_parent is None:
            new_root = TreeNode()
            new_root.add_child(p_node)
            # new_root.add_child(p_node.detach()) ???
            new_root.add_child(h_subtree_copy)
            new_tree_obj = new_root
        else:
            new_internal = TreeNode()
            p_parent.add_child(new_internal)
            p_node.detach()
            new_internal.add_child(p_node)
            new_internal.add_child(h_subtree_copy)

        # Clean up old internal labels before re-indexing
        for n in new_tree_obj.traverse():
            if n.name and n.name.startswith("<") and n.name.endswith(">"):
                n.name = None
        
        # Constructor will call _cache_depths() and re-index
        return SmrtTree(tree_obj=new_tree_obj)'''
    
    def to_string(self, internal_labels=True) -> str:
        root_name = str(self.ete_tree.name) if internal_labels else ""
        return self.ete_tree.write(format=8 if internal_labels else 9)[:-1]+root_name+";"
    
    # New: Add this for pickling safety!
    def __getstate__(self):
        return self.ete_tree
    
    def __setstate__(self, state):
        self.ete_tree = state
        self.refresh() # Auto-rebuild maps on unpickle

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
    ambiguous_groups: List[List[str]]
    fixed_groups: List[Tuple[List[str], str]]

@dataclass(slots=True, frozen=True)
class MulTree:
    mt: SmrtTree
    h_clade: List[str] = field(default_factory=list)
    # Storing OBJECTS optimizes the Reconciler (no lookups)
    # These are safe to pickle TO workers, but shouldn't be used to map back TO main.
    h1_node: Optional[TreeNode] = None 
    h2_node: Optional[TreeNode] = None

@dataclass(slots=True, frozen=True)
class StepResult:
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
    def self_score(self) -> int:
        """O(1) retrieval of the input species tree score."""
        if self._input_rank == -1: return float('inf')
        # Return tuple item [1] (score), as [0] (index) is always 0
        return self.sorted_scores[self._input_rank][1]

'''
@dataclass(slots=True)
class HEvent:
    """Represents a confirmed hybridization event in the flow.
    Stores data that can be extracted from written outputs, to enable resuming flows.
    """
    prev_tree: Tree #GrandmaTree?
    prev_score: int
    curr_tree: Tree #GrandmaTree?
    curr_score: int
    self_score: int
    gt_num: int

    h1_node: TreeNode # redundant - just store the MulData?
    h2_node: TreeNode # redundant - just store the MulData?
    score: float
    score_tuple: Tuple[float, float]
    other_tree: Tree #GrandmaTree?
'''

