'''
This is the Critical Shim. It replaces recontree.py.
It uses ETE3 but adds methods to make the tree look and behave like the old dictionary-based trees for the algorithms.
'''

from ete3 import Tree, TreeNode
from typing import List, Set, Dict, Optional, Tuple
from dataclasses import dataclass, field

class GrandmaTree:
    """
    Wrapper around ETE3 Tree to provide GRAMPA-specific functionality.
    """
    def __init__(self, newick: str = None, tree_obj: Tree = None):
        if tree_obj:
            self.ete_tree = tree_obj
        else:
            # add ; if missing
            if newick and not newick.strip().endswith(";"):
                newick += ";"
            # Format 1 allows internal node names
            self.ete_tree = Tree(newick, format=0) 
            
        self.node_map = {} # Map name -> TreeNode
        
        # OPTIMIZATION: Cache for LCA lookups to avoid tree traversal overhead
        self.lca_cache: Dict[Tuple[int, ...], TreeNode] = {}
        
        self._index_nodes()
        self._cache_depths() # FIXED: This must be enabled!

    def _index_nodes(self):
        """
        Assigns <x> labels to internal nodes if missing.
        Uses postorder to match legacy GRAMPA bottom-up labeling.
        """
        i = 1
        for node in self.ete_tree.traverse("postorder"):
            if not node.is_leaf():
                if not node.name:
                    node.name = f"<{i}>"
                    i += 1
            else:
                node.name = str(node.name).strip()
            
            self.node_map[node.name] = node

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

    def get_node(self, name: str) -> TreeNode:
        return self.node_map.get(name)

    def get_lca(self, species_list: List[str]) -> TreeNode:
        """String-based LCA lookup (Legacy support)"""
        nodes = [self.node_map[name] for name in species_list]
        if not nodes: return None
        if len(nodes) == 1: return nodes[0]
        return self.ete_tree.get_common_ancestor(nodes)

    def get_lca_obj(self, nodes: List[TreeNode]) -> TreeNode:
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

    def to_mul_tree(self, h_node_label: str, p_node_label: str) -> 'GrandmaTree':
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
            new_root.add_child(h_subtree_copy)
            new_tree_obj = new_root
        else:
            new_internal = TreeNode()
            p_parent.add_child(new_internal)
            p_node.detach()
            new_internal.add_child(p_node)
            new_internal.add_child(h_subtree_copy)

        for n in new_tree_obj.traverse():
            if n.name.startswith("<") and n.name.endswith(">"):
                n.name = None
        
        # Constructor will call _cache_depths() and re-index
        return GrandmaTree(tree_obj=new_tree_obj)

    def to_string(self, internal_labels=True) -> str:
        root_name = str(self.ete_tree.name) if internal_labels else ""
        return self.ete_tree.write(format=8 if internal_labels else 9)[:-1]+root_name+";"

@dataclass(slots=True, frozen=True)
class MulData:
    mt: GrandmaTree
    h_clade: list = field(default_factory=list)
    h1_node: str = "NA"
    h2_node: str = "NA"
    