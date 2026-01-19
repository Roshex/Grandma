from ete3 import Tree
import networkx as nx
from collections import Counter
from event_compare import ReticulateTree


class Multiset:
    def __init__(self):
        self.counter = Counter()
        self.concatenated = ""

    def add(self, item):
        if isinstance(item, Multiset):
            self.counter.update(item.counter)
        else:
            self.counter[item] += 1

    def addSorted(self, item):
        self.add(item)

    def setConcatenatedElements(self, label=None):
        if label is not None:
            self.concatenated = label
        else:
            flat = sorted(self.counter.elements())
            self.concatenated = "|".join(flat)

    def __repr__(self):
        return f"{dict(self.counter)}"
    
    def __eq__(self, other):
        if not isinstance(other, Multiset):
            return False
        return self.counter == other.counter and self.concatenated == other.concatenated

    def __hash__(self):
        return hash((frozenset(self.counter.items()), self.concatenated))


class HeightList:
    def __init__(self):
        self.entries = []  # list of tuples: (node, multiset)

    def addSorted(self, node, multiset):
        self.entries.append((node, multiset))
        self.entries.sort(key=lambda x: repr(x[1]))  # Sort by multiset string

    def remove(self, node):
        self.entries = [(n, m) for n, m in self.entries if n != node]

    def __getitem__(self, idx):
        return self.entries[idx][0]

    def __len__(self):
        return len(self.entries)

    def isEmpty(self):
        return len(self.entries) == 0

    def size(self):
        return len(self.entries)


class MultilabeledTree:
    def __init__(self, newick=None):
        self.tree = Tree(newick, format=1) if newick else Tree()
        self.all_multisets = {}
        self.all_node_heights = {}
        self.all_multisets = self.collect_all_multisets()
        self.all_node_heights = self.collect_all_node_heights()

    def get_multisets(self, node):
        cluster_set = []
        self._get_multisets_rec(node, cluster_set)
        return cluster_set

    def _get_multisets_rec(self, node, cluster_set):
        if node.is_leaf():
            cluster = Multiset()
            cluster.setConcatenatedElements(node.name)
            cluster.add(node.name)
            cluster_set.append(cluster)
            return cluster

        if len(node.children) == 1 and (node.up and len(node.up.children) > 1):
            return self._get_multisets_rec(node.children[0], cluster_set)

        cluster = Multiset()
        for child in node.children:
            cluster.addSorted(self._get_multisets_rec(child, cluster_set))
        cluster.setConcatenatedElements()
        cluster_set.append(cluster)
        return cluster

    def collect_all_multisets(self):
        multisets = {}
        self._collect_all_multisets_rec(self.tree, multisets)
        return multisets

    def _collect_all_multisets_rec(self, node, multisets):
        if node.is_leaf():
            cluster = Multiset()
            cluster.setConcatenatedElements(node.name)
            cluster.add(node.name)
            multisets[id(node)] = cluster
            return cluster

        cluster = Multiset()
        for child in node.children:
            cluster.addSorted(self._collect_all_multisets_rec(child, multisets))
        cluster.setConcatenatedElements()
        multisets[id(node)] = cluster
        return cluster

    def collect_all_node_heights(self):
        heights = {}
        self._collect_node_heights_rec(self.tree, heights)
        return heights

    def _collect_node_heights_rec(self, node, heights):
        if node.is_leaf():
            heights[id(node)] = 0
            return 0
        max_height = 0
        for child in node.children:
            h = 1 + self._collect_node_heights_rec(child, heights)
            max_height = max(max_height, h)
        heights[id(node)] = max_height
        return max_height

    def get_multiset(self, node):
        return self.all_multisets.get(id(node))

    def get_height(self, node):
        return self.all_node_heights.get(id(node), 0)

    def get_all_taxa(self):
        return {leaf.name for leaf in self.tree.iter_leaves()}

    def get_descending_nodes(self, node):
        return [node] + list(node.iter_descendants())

    def get_ascending_nodes(self, node):
        asc = []
        while node.up:
            asc.append(node.up)
            node = node.up
        return asc

    def compute_all_canonical_forms(self, root):
        self._canonical_map = {}

        def recurse(node):
            """
            Recursively compute a canonical string representation of a subtree rooted at `node`,
            such that isomorphic subtrees (with possibly duplicated leaves) get the same string.
            """
            if node.is_leaf():
                canon = node.name
            else:
                child_encodings = sorted([recurse(c) for c in node.children])
                canon = '(' + ','.join(child_encodings) + ')'
            self._canonical_map[node] = canon
            return canon

        recurse(root)
    
    def are_strictly_isomorphic(self, node1, node2):
        """
        Strict topological isomorphism, allowing for duplicated taxa.
        """
        #print(node1.get_ascii(show_internal=True))
        #print(node2.get_ascii(show_internal=True))
        return self._canonical_map[node1] == self._canonical_map[node2]

    def apply_exact_network(self, strict=True):
        t = MultilabeledTree(self.tree.write(format=1))  # fresh copy
        root = t.tree
        h_max = {}
        rootHeight = t.get_height(root)
        h = HeightList()
        h.addSorted(root, t.get_multiset(root))
        h_max[rootHeight] = h

        if strict:
            t.compute_all_canonical_forms(root)
            iso_cond = lambda a, b: t.are_strictly_isomorphic(a, b)
        else:
            iso_cond = lambda a, b: t.get_multiset(a) == t.get_multiset(b)

        for i in range(rootHeight, -1, -1):
            l_h = h_max.get(i)
            if not l_h:
                continue
            while not l_h.isEmpty():
                t_max = l_h[0]
                for child in t_max.children:
                    h_j = t.get_height(child)
                    h_list = h_max.get(h_j)
                    if h_list is None:
                        h_list = HeightList()
                        h_max[h_j] = h_list
                    h_list.addSorted(child, t.get_multiset(child))

                isomorphs = []
                for j in range(1, l_h.size()):
                    w = l_h[j]
                    if iso_cond(t_max, w):
                        isomorphs.append(w)
                    else:
                        break

                if isomorphs:
                    if not t_max.up:
                        continue  # Skip if root

                    parent = t_max.up
                    u = parent.add_child(name=None)
                    parent.remove_child(t_max)
                    u.add_child(t_max)
                    t.set_reticulate_edge(parent, u)
                    t.set_edge_weight(parent, u, 0)

                    for w in isomorphs:
                        if not w.up:
                            continue
                        w_parent = w.up
                        w.detach()
                        w_parent.add_child(u)
                        t.set_reticulate_edge(w_parent, u)
                        t.set_edge_weight(w_parent, u, 0)

                    for iso in isomorphs:
                        l_h.remove(iso)
                l_h.remove(t_max)

        print("number of reticulations:", t.get_number_reticulate_edges() // 2)
        return t

    def set_reticulate_edge(self, parent, child):
        if not hasattr(self, "_reticulate_edges"):
            self._reticulate_edges = set()
        self._reticulate_edges.add((id(parent), id(child)))

    def set_edge_weight(self, parent, child, weight):
        if not hasattr(self, "_edge_weights"):
            self._edge_weights = {}
        self._edge_weights[(id(parent), id(child))] = weight

    def is_reticulate_edge(self, parent, child):
        return (id(parent), id(child)) in getattr(self, "_reticulate_edges", set())

    def get_number_reticulate_edges(self):
        return len(getattr(self, "_reticulate_edges", []))
    
    def to_networkx(self):
        G = nx.DiGraph()

        # mapping ete3 node id to display name
        def get_node_label(node):
            return node.name if node.name else str(id(node))

        # add nodes
        for node in self.tree.traverse("preorder"):
            G.add_node(id(node), label=get_node_label(node))

        # add edges
        for node in self.tree.traverse("preorder"):
            for child in node.children:
                G.add_edge(id(node), id(child), reticulate=self.is_reticulate_edge(node, child))

        return G

if __name__ == "__main__":
    newick = "((((c,w),d),(x,((z,y),w))),((a,(x,((y,z),w))),b));"
    mlt = MultilabeledTree(newick)

    print("All Taxa:", mlt.get_all_taxa())
    print("Special Nodes:", mlt.get_number_of_special_nodes())

    for node in mlt.tree.traverse():
        print(f"Node: {node.name or 'internal'}, Height: {mlt.get_height(node)}, Multiset: {mlt.get_multiset(node)}")

    exact_net = mlt.apply_exact_network()

    print(mlt.get_multisets(mlt.tree.get_tree_root()))
    print(mlt.get_height(mlt.tree.get_tree_root()))

    dag = exact_net.to_networkx()

    new_obj = ReticulateTree(dag)
    new_obj.visualize()
