'''
This is the Critical Shim. It replaces recontree.py.
It uses ETE3 but adds methods to make the tree look and behave like the old dictionary-based trees for the algorithms.
'''

from ete3 import Tree, TreeNode
from typing import List, Set, Dict, Optional
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
            #print('starting to load tree:', newick)
            # Format 1 allows internal node names
            self.ete_tree = Tree(newick, format=0) # ETE format 0: standard newick: flexible names and blengths + support values
            #print(self.ete_tree)
            
        self.node_map = {} # Map name -> TreeNode
        self._index_nodes()
        self._cache_depths()

    def _index_nodes(self):
        """
        Assigns <x> labels to internal nodes if missing.
        CHANGED: Uses postorder to match legacy GRAMPA bottom-up labeling (Root = High ID).
        """
        i = 1
        # Change "levelorder" to "postorder"
        for node in self.ete_tree.traverse("postorder"):
            if not node.is_leaf():
                if not node.name:
                    node.name = f"<{i}>"
                    i += 1
                elif not node.name.startswith("<"):
                    # Preserve existing names if they aren't auto-generated IDs
                    pass
            else:
                # Clean leaf names
                node.name = str(node.name).strip()
            
            self.node_map[node.name] = node

    def _cache_depths(self):
        """
        Calculates node depth (distance to root in number of nodes).
        Legacy GRAMPA defines root depth as 0.
        """
        self.depths = {}
        for node in self.ete_tree.traverse("preorder"):
            depth = 0
            curr = node
            while curr.up:
                depth += 1
                curr = curr.up
            self.depths[node.name] = depth

    def get_node_depth(self, node_name: str) -> int:
        return self.depths.get(node_name, 0)

    def get_node(self, name: str) -> TreeNode:
        return self.node_map.get(name)

    def get_lca(self, species_list: List[str]) -> TreeNode:
        """Returns the LCA node object for a list of species names."""
        nodes = [self.node_map[name] for name in species_list]
        if not nodes: return None
        if len(nodes) == 1: return nodes[0]
        return self.ete_tree.get_common_ancestor(nodes)

    def get_clade_leaves(self, node_name: str) -> Set[str]:
        node = self.get_node(node_name)
        if not node:
            return set()
        return {leaf.name for leaf in node.iter_leaves()}

    def to_mul_tree(self, h_node_label: str, p_node_label: str) -> 'GrandmaTree':
        """Generates a MUL-tree by copying h_node subtree to p_node location."""
        new_tree_obj = self.ete_tree.copy()
        
        # 1. Search nodes
        h_matches = new_tree_obj.search_nodes(name=h_node_label)
        p_matches = new_tree_obj.search_nodes(name=p_node_label)
        
        if not h_matches or not p_matches:
            return None
            
        h_node = h_matches[0]
        p_node = p_matches[0]

        # 2. Validity Check
        # No need to check p_node == h_node: otherwise we block autopolyploidy
        if p_node in h_node.iter_descendants():
            return None

        # 3. Copy subtree
        h_subtree_copy = h_node.copy()
        
        # 4. Relabel tips in copy
        for leaf in h_subtree_copy.iter_leaves():
            leaf.name = f"{leaf.name}*"
            
        # 5. Attach
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

        # 6. Remove the <x> labels in the new tree to re-index later (when calling constructor)
        for n in new_tree_obj.traverse():
            if n.name.startswith("<") and n.name.endswith(">"):
                n.name = None
        
        return GrandmaTree(tree_obj=new_tree_obj)

    def to_string(self, internal_labels=True) -> str:
        # Format 8 includes all names, 1 includes internal
        fmt = 1 if internal_labels else 5 # 5 is internal only? ETE formats are tricky.
        # Format 8: all names, no branch lengths (closest to what we need)
        # However, legacy code output standard newick.
        root_name = str(self.ete_tree.name) if internal_labels else ""
        return self.ete_tree.write(format=8 if internal_labels else 9)[:-1]+root_name+";"

@dataclass(slots=True, frozen=True)
class MulData:
    mt: GrandmaTree
    h_clade: list = field(default_factory=list)
    h1_node: str = "NA"
    h2_node: str = "NA"

