"""
algo.py - the mathematics of GRANDMA: grouping, the target sweep, and pairwise
duplication-loss reconciliation. Everything here is computation over FlatTrees;
orchestration, I/O and multiprocessing live in ops.py / reconcile.py.

======================================================================================
1. GROUPING - what states a mapping unit may take
======================================================================================
A unit is a monophyletic clade of gene copies whose species all lie in the donor clade.
GRAMPA collapses each unit and tries only the c CONSTANT assignments (all of it into
copy 0, ..., all into copy c-1). That is not loss-free: consolidating a unit forces
M(root(U)) strictly inside one copy and therefore FORBIDS M(root(U)) = rho, the node
where the two sub-genomes join - precisely the node representing the polyploidy event.
Minimal counterexample: S=(A,(B,C)), G=((g_A,g_B),g_C), H1=H2=root; the collapsed search
returns 5, the true minimum is 3.

  Lemma (unit locality). Everything outside subtree(root(U)) depends on U only through
  M(root(U)), so
      cost = sum_i local_i(assignment of U_i) + ext(M(root(U_1)), ..., M(root(U_g)))
  exactly, whenever the units are DISJOINT CLADES all of whose leaves are donor-clade
  copies - which is what compute_groups produces.

  Corollary. Enumerating, per unit, one minimal-local-cost representative per achievable
  value of M(root(U)) is EQUIVALENT to enumerating all c^|U| assignments. The achievable
  images are the c copy images plus lca{h_k : k in K} over the subsets K actually used,
  and those are the internal nodes of the topology induced on the c copy roots - at most
  c-1. So a unit has at most 2c-1 states whatever its size: 3 for two copies, 5 for
  three, 7 for four.

exact=True repairs this without giving up collapsing. Every FREE multi-leaf unit gains a
third state, "mixed", carried as one canonical split plus a per-target correction

    corr_U[t] = local_min_U(mixed, t) - local_U(canonical split, t)

which is legitimate because every split of U gives M(root(U)) = rho, so the rest of the
tree cannot tell two splits apart (unit locality) and only the LOCAL term differs. The
correction comes from a 3-state DP over the unit's clade, vectorised over targets:
inside a unit every image depth is one of

    backbone : d_S(x_v) + A(t)          A(t) = 1 iff t is an ancestor-or-self of h
    graft    : d_S(x_v) + D(t)+1-d(h)   D(t) = d_S(t)
    rho      : R(t)                     R(t) = d_S(lca_S(t, h))

and every duplication indicator inside a unit is target-INDEPENDENT, because a unit's
images live in subtree(h), in its copy, or at rho, and rho is never inside subtree(h).

`unit_states` computes exactly that with an O(|U|*(2c-1)^2) DP (standard engine);
`TargetSweep._mixed_correction` computes the same thing for the sweep, where the copies
move with the target. The unit RULE therefore affects only speed, never the answer.
A PINNED unit gets no mixed state: pinning already places its gene-tree sisters beside
one copy, so a split loses on the internal depths, on the duplications inside the unit
and at its parent.

======================================================================================
2. THE TARGET SWEEP - every placement of a donor clade in one pass
======================================================================================
Each candidate is T(h, t) = S with a copy C of subtree(h) grafted above target t.
Instead of building and reconciling each of the O(N) candidates, the cost is written as
a function of t and evaluated for all t at once.

  Lemma (graft collapse). In T(h, t), with r the new node above t, for c in C and any
  backbone node v:  lca_T(c, v) = r if v <= t, else lca_S(t, v); and in both cases
  d_T(lca_T(c, v)) = d_S(lca_S(t, v)).
  Proof. If v <= t then lca_S(t,v) = t and d_T(r) = d_S(t), since r takes t's old depth.
  Otherwise lca_S(t,v) is a proper ancestor of t, hence outside subtree(t), so the
  insertion leaves its depth unchanged. []

From the outside the graft behaves as the single point r. With the closed form
    cost = (dup + 2*loss)*D + loss*(sum_leaves d(M) - sum_internal d(M) - 2(n-1))
and each gene-tree node classified pure-C / pure-B / mixed:

    cost(sigma, t) = const(sigma)
                   + loss * m(sigma) * (d_S(t)+1)      m = # maximal pure-C clades
                   + loss * Omega_sigma(t)             subtree sum over S
                   - loss * F_sigma(t)                 ancestral prefix sum over S
                   + (dup + 2*loss) * (D_pure(sigma) + D_mix(sigma, t))

    Omega(t) = #{pure-B LEAF images inside subtree(t)} - #{pure-B INTERNAL images ...}
    F(t)     = sum over mixed nodes u of d_S(lca_S(t, w_u)), w_u = lca of u's backbone
               images (a constant per node), which expands to a root-path sum:
                   F(t) = sum over a in anc(t), a != root, of c(a)
                   c(a) = #{mixed u : w_u in subtree(a)}

A pure-B node has only pure-B children and a pure-C node only pure-C children, so
D_pure is target-independent; only MIXED nodes move, and each is an O(1) region of S
(_mixed_dup_region). Every target-dependent term is a subtree sum (a preorder interval)
or a root-path sum (a difference array over intervals), so all resolve with cumulative
sums and the per-target loop disappears.

Scope: exactly TWO copies (single-target candidates), a singly labelled bifurcating
species tree. Multi-target candidates go through PairwiseRecon.

Speed dials that never change the answer: `pin` (sister-based pinning; targets grouped
by pinning pattern, each group swept with its free units), `gray` (incremental Gray-code
enumeration, one unit per step, walk stopping at the first dominating ancestor that
keeps its class and image) and `batch` (assignments resolved per vectorised pass).
gray=False rebuilds per combination and is kept as the reference oracle.

Complexity: O(P*(n_G+N)) for ALL N targets per gene tree, P = prod of per-unit state
counts, against O(N*2^g*n_G) for building and reconciling each candidate. Exact grouping
costs (3/2)^(#free multi-leaf units) - the same factor unit_states costs the engine.

======================================================================================
3. PAIRWISE RECONCILIATION - Zmasek-Eddy / Durand duplication-loss
======================================================================================
Scores a gene tree against a species or MUL-tree given a leaf mapping: a duplication
where a node's image equals a child's, losses from the depth gaps. One postorder pass
over integer arrays, O(1) LCA via an Euler tour and a sparse table. PairwiseRecon owns
reconcile_sl (singly labelled) and reconcile_permutation (MUL-trees, enumerating unit
states with the same Gray-code machinery the sweep uses).
"""

import array
import itertools
import numpy as np
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union, Iterable

from .models import SmrtTree, MulTree, GroupData, Map, NameRegistry, FlatTree, splitSpec

_INF = float('inf')

def _prod(it):
    r = 1
    for x in it:
        r *= x
    return r

# node classes
C, B, X = 0, 1, 2                 # pure-C (graft), pure-B (backbone), mixed

# --------------------------------------------------------------------------
# Grouping
# --------------------------------------------------------------------------

def unit_clade(gt_flat: FlatTree, unit_leaves: Sequence[int]) -> Tuple[int, List[int]]:
    """(postorder of the unit's clade, backbone image x_v of every node)."""
    cs, cf, parents = gt_flat.children_start, gt_flat.children_flat, gt_flat.parents
    target = set(unit_leaves)
    # find the root of the unit's clade
    root = unit_leaves[0]
    while True:
        p = parents[root]
        if p < 0:
            break
        stack, sub = [p], set()
        while stack:
            v = stack.pop()
            a, b = cs[v], cs[v + 1]
            if a == b:
                sub.add(v)
            else:
                stack.extend(cf[a:b])
        if sub <= target:
            root = p
        else:
            break
    # postorder of the unit's clade
    order, stack = [], [(root, False)]
    while stack:
        v, done = stack.pop()
        if done:
            order.append(v)
            continue
        stack.append((v, True))
        stack.extend((c, False) for c in cf[cs[v]:cs[v + 1]])
    return root, order

def unit_states(gt_flat, unit_leaves: Sequence[int],
                leaf_targets: Dict[int, Sequence[int]],
                mul_flat, dup_cost: int, loss_cost: int,
                exact: bool = True) -> List[Tuple[int, ...]]:
    """
    The assignment patterns a unit may take, aligned with `unit_leaves`.

    leaf_targets[leaf] -> the MUL-tree node ids of that species' copies, in copy order.

    exact=False  : the c CONSTANT patterns only, i.e. GRAMPA's behaviour.
    exact=True   : one minimal-local-cost representative per achievable image of the
                   unit's clade root (<= 2c-1 of them), which by the corollary above is
                   equivalent to enumerating every assignment of the unit.

    Returned patterns are tuples of MUL-tree NODE IDS, one per leaf of `unit_leaves`, so
    the caller can apply them without knowing anything about the DP.
    """
    n_copies = max((len(leaf_targets[x]) for x in unit_leaves), default=0)
    if n_copies == 0:
        return []

    # --- constant patterns (always present, and the only ones when exact=False) ---
    constants: List[Tuple[int, ...]] = []
    for k in range(n_copies):
        constants.append(tuple(leaf_targets[x][k if k < len(leaf_targets[x]) else 0]
                               for x in unit_leaves))
    if not exact or len(unit_leaves) == 1:
        # A single-leaf unit cannot be split, so its states ARE the constants.
        seen, out = set(), []
        for pat in constants:
            if pat not in seen:
                seen.add(pat)
                out.append(pat)
        return out

    # --- DP over the unit's clade ---------------------------------------------
    root, order = unit_clade(gt_flat, unit_leaves)
    cs, cf = gt_flat.children_start, gt_flat.children_flat
    nd = mul_flat.node_depths
    lca = mul_flat.get_lca

    pos = {x: i for i, x in enumerate(unit_leaves)}
    # table[v] : image -> (cost, assignment as a dict leaf -> node)
    table: Dict[int, Dict[int, Tuple[float, dict]]] = {}

    for v in order:
        s, e = cs[v], cs[v + 1]
        if s == e:
            if v not in pos:
                raise ValueError("unit is not a clade: an outside leaf was reached")
            table[v] = {t: (0, {v: t}) for t in leaf_targets[v]}
            continue

        c1, c2 = cf[s], cf[s + 1]
        cur: Dict[int, Tuple[float, dict]] = {}
        for i1, (k1, a1) in table[c1].items():
            d1 = nd[i1]
            for i2, (k2, a2) in table[c2].items():
                m = lca(i1, i2)
                dm = nd[m]
                if m == i1 or m == i2:
                    c = dup_cost
                    l1, l2 = d1 - dm, nd[i2] - dm
                else:
                    c = 0
                    l1, l2 = d1 - dm - 1, nd[i2] - dm - 1
                if l1 > 0:
                    c += loss_cost * l1
                if l2 > 0:
                    c += loss_cost * l2
                tot = k1 + k2 + c
                prev = cur.get(m)
                if prev is None or tot < prev[0]:
                    merged = dict(a1)
                    merged.update(a2)
                    cur[m] = (tot, merged)
        table[v] = cur
        # free the children: the DP only ever needs the frontier
        table.pop(c1, None)
        table.pop(c2, None)

    out, seen = [], set()
    for _img, (_cost, assign) in sorted(table[root].items()):
        pat = tuple(assign[x] for x in unit_leaves)
        if pat not in seen:
            seen.add(pat)
            out.append(pat)
    # the constants are always achievable images, but keep them explicitly in case a
    # degenerate tree makes the DP miss one (it cannot, but the guard is free)
    for pat in constants:
        if pat not in seen:
            seen.add(pat)
            out.append(pat)
    return out

# --------------------------------------------------------------------------
# Gene-tree UNIT RULE - two implementations of the same semantics
# --------------------------------------------------------------------------
# compute_groups walks ETE trees during cull; compute_units walks FlatTrees during the
# sweep. They MUST produce the same partition: a clade is collapsed iff all its leaves
# are donor-clade copies and its two CHILDREN's species sets are disjoint (a check made
# at that node only - a species may therefore repeat inside one group). If you change
# one, change the other, and re-run the agreement test.

def compute_units(gt_flat: FlatTree, clade_species: Set[int], rule: bool = True) -> List[List[int]]:
    """
    The ambiguous units: clades of G whose leaves are all donor-clade gene copies.

    rule=True (default) reproduces models.compute_groups EXACTLY (based on GRAMPA's non-
    exact grouping), including its duplicate test: a clade is collapsed when the species
    sets of its TWO CHILDREN are disjoint - a check made only at that node, not over the
    whole clade. A species may therefore appear several times inside one engine group
    (e.g. ((x,x),y) merges with a sibling carrying neither x nor y). Use this whenever
    the sweep must be comparable with the standard pathway.

    rule=False requires the whole clade to be duplicate-free, which is what the
    GRAMPA paper describes ("only if there is not more than one copy of a species among
    the clades can they be grouped"). It yields strictly finer units, hence a larger
    assignment space and a score that is <= the engine's, and closer to the true MP.
    No pinning is applied either way: pinning is target-dependent and is handled by
    TargetSweep.pin_states.
    """
    cs, cf, post = gt_flat.children_start, gt_flat.children_flat, gt_flat.postorder
    names = gt_flat.node_to_name_id

    if rule:
        groups: Dict[int, List[int]] = {}
        singles: Dict[int, bool] = {}
        info: Dict[int, tuple] = {}
        for u in post:
            s, e = cs[u], cs[u + 1]
            if s == e:
                sp = names[u]
                is_h1 = sp in clade_species
                info[u] = ({sp}, [u], [u] if is_h1 else [])
                if is_h1:
                    singles[u] = True
                continue
            u_s, u_l, u_a, all_h1, total = set(), [], [], True, 0
            for c in cf[s:e]:
                c_s, c_l, c_a = info[c]
                u_s |= c_s
                u_l += c_l
                u_a += c_a
                total += len(c_s)
                if not c_s <= clade_species:
                    all_h1 = False
            if all_h1 and len(u_s) == total and (e - s) > 1:
                for r in u_a:
                    groups.pop(r, None)
                    singles.pop(r, None)
                groups[u] = u_l
                u_a = [u]
            info[u] = (u_s, u_l, u_a)
        return [list(v) for v in groups.values()] + [[k] for k in singles]

    # --- strict (transitively duplicate-free) variant --------------------------
    leaves_below: Dict[int, List[int]] = {}
    species_below: Dict[int, set] = {}
    all_in: Dict[int, bool] = {}
    dup_free: Dict[int, bool] = {}
    unit_root: Dict[int, Optional[int]] = {}

    for u in post:
        s, e = cs[u], cs[u + 1]
        if s == e:
            sp = names[u]
            leaves_below[u] = [u]
            species_below[u] = {sp}
            all_in[u] = sp in clade_species
            dup_free[u] = True
            unit_root[u] = u if all_in[u] else None
        else:
            c1, c2 = cf[s], cf[s + 1]
            leaves_below[u] = leaves_below[c1] + leaves_below[c2]
            species_below[u] = species_below[c1] | species_below[c2]
            all_in[u] = all_in[c1] and all_in[c2]
            dup_free[u] = (dup_free[c1] and dup_free[c2]
                           and not (species_below[c1] & species_below[c2]))
            unit_root[u] = u if (all_in[u] and dup_free[u]) else None

    units, taken = [], set()
    for u in reversed(post):
        if unit_root[u] is None:
            continue
        blk = leaves_below[u]
        if any(x in taken for x in blk):
            continue
        units.append(blk)
        taken.update(blk)
    return units

def compute_groups(gt: SmrtTree, mul_data: MulTree, registry: NameRegistry, 
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
    for node in gt.ete_tree.traverse("postorder"):
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
        n_obj = gt.get_node(n_name)
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
# Common utilities for sweep and pairwise reconciliation
# --------------------------------------------------------------------------

def _mixed_radix_gray(radices: Sequence[int]):
    """
    Reflected mixed-radix Gray code. Yields (position, new_digit) for every step
    after the initial all-zero tuple, visiting each tuple exactly once and changing
    exactly ONE coordinate per step.
    """
    n = len(radices)
    if n == 0 or any(r <= 0 for r in radices):
        return
    digits = [0] * n
    direction = [1] * n
    while True:
        i = n - 1
        while i >= 0:
            nxt = digits[i] + direction[i]
            if 0 <= nxt < radices[i]:
                digits[i] = nxt
                yield i, nxt
                break
            direction[i] = -direction[i]
            i -= 1
        else:
            return

def _group_ancestors(gt_flat: FlatTree, group_leaves: List[List[int]]
                        ) -> Tuple[List[List[int]], List[int]]:
    """
    For every group, the ancestors of its leaves in postorder, plus the index at
    which those ancestors start dominating the WHOLE group.

    Beyond that index the ancestors form a chain, so once such a node's LCA image
    is unchanged, no node above it can change either and the update can stop.
    """
    n = gt_flat.num_nodes
    parents, post = gt_flat.parents, gt_flat.postorder
    post_idx = [0] * n
    for i, u in enumerate(post):
        post_idx[u] = i

    anc_lists: List[List[int]] = []
    chain_starts: List[int] = []
    for grp in group_leaves:
        counts: Dict[int, int] = {}
        for leaf in grp:
            p = parents[leaf]
            while p != -1:
                counts[p] = counts.get(p, 0) + 1
                p = parents[p]
        anc = sorted(counts, key=post_idx.__getitem__)
        full = len(grp)
        start = len(anc)
        for i, u in enumerate(anc):
            if counts[u] == full:
                start = i
                break
        anc_lists.append(anc)
        chain_starts.append(start)
    return anc_lists, chain_starts

# --------------------------------------------------------------------------
# Species tree indexing
# --------------------------------------------------------------------------

class SpeciesIndex:
    """Precomputed, target-independent structure of the species tree."""

    __slots__ = ('n', 'parent', 'depth', 'preorder', 'postorder', 'children',
                 'tin', 'tout', 'root', 'sp_of_leaf', 'node_of_species',
                 'tin_np', 'tout_np', 'depth_np')

    def __init__(self, st_flat):
        n = self.n = st_flat.num_nodes
        cs, cf = st_flat.children_start, st_flat.children_flat
        self.parent = list(st_flat.parents)
        self.children = [list(cf[cs[v]:cs[v + 1]]) for v in range(n)]
        self.root = st_flat.root_id
        self.depth = list(st_flat.node_depths)

        # preorder / postorder and subtree intervals, one DFS
        tin = [0] * n
        tout = [0] * n
        pre, post = [], []
        clock = 0
        stack = [(self.root, False)]
        while stack:
            v, done = stack.pop()
            if done:
                post.append(v)
                tout[v] = clock - 1          # inclusive: last preorder index in subtree(v)
                continue
            tin[v] = clock
            clock += 1
            pre.append(v)
            stack.append((v, True))
            for c in reversed(self.children[v]):
                stack.append((c, False))
        self.tin, self.tout, self.preorder, self.postorder = tin, tout, pre, post

        # Vectorised views. Every target-dependent quantity in the sweep is either a
        # subtree sum (an interval in preorder space) or a root-path sum (a difference
        # array over subtree intervals), so both become cumsums and the per-target loop
        # disappears entirely.
        self.tin_np = np.asarray(tin, dtype=np.int64)
        self.tout_np = np.asarray(tout, dtype=np.int64)
        self.depth_np = np.asarray(self.depth, dtype=np.int64)

        # species name id -> backbone node (S is singly labelled)
        self.node_of_species: Dict[int, int] = {}
        for v in range(n):
            if cs[v] == cs[v + 1]:
                self.node_of_species[st_flat.node_to_name_id[v]] = v

    # ---- structural queries ------------------------------------------------
    def is_desc(self, v: int, t: int) -> bool:
        """v is a descendant-or-self of t."""
        return self.tin[t] <= self.tin[v] <= self.tout[t]

    def child_toward(self, w: int, x: int) -> int:
        """The child of w that is an ancestor-or-self of x (x must be strictly below w)."""
        for c in self.children[w]:
            if self.is_desc(x, c):
                return c
        raise ValueError("child_toward: x is not below w")

    def other_child(self, w: int, c: int) -> int:
        for k in self.children[w]:
            if k != c:
                return k
        raise ValueError("other_child: w is not bifurcating")

    # ---- the two linear resolvers -----------------------------------------
    def subtree_sum(self, marks: Sequence[float]) -> List[float]:
        """out[t] = sum of marks over subtree(t). One postorder pass."""
        out = list(marks)
        for v in self.postorder:
            p = self.parent[v]
            if p >= 0:
                out[p] += out[v]
        return out

    def rootpath_sum(self, marks: Sequence[float]) -> List[float]:
        """out[t] = sum of marks over the ancestors-or-self of t. One preorder pass."""
        out = list(marks)
        for v in self.preorder:
            p = self.parent[v]
            if p >= 0:
                out[v] += out[p]
        return out

# --------------------------------------------------------------------------
# The Sweep
# --------------------------------------------------------------------------

class TargetSweep:
    """Scores one gene tree against every placement of one donor clade."""

    def __init__(self, sidx: SpeciesIndex, dup_cost: int = 1, loss_cost: int = 1):
        self.S = sidx
        self.dup_cost = dup_cost
        self.loss_cost = loss_cost

    # ----------------------------------------------------------------------
    # PINNING REGIONS
    # ----------------------------------------------------------------------

    def _unit_sisters(self, gt_flat: FlatTree, units: List[List[int]]):
        """
        For every unit, the species of its SISTER clade in the gene tree - exactly the
        `anc_leaves` that compute_groups/check_fix look at (the leaves of the unit root's
        parent, minus the unit's own leaves; empty when the unit root is the gene-tree
        root). Returned as a set of species name-ids.
        """
        cs, cf, post, names = (gt_flat.children_start, gt_flat.children_flat,
                               gt_flat.postorder, gt_flat.node_to_name_id)
        parent = gt_flat.parents
        below = {}
        for u in post:
            s, e = cs[u], cs[u + 1]
            below[u] = {names[u]} if s == e else (below[cf[s]] | below[cf[s + 1]])

        leaf_set = {}
        for i, grp in enumerate(units):
            leaf_set[i] = set(grp)

        out = []
        for grp in units:
            # the unit root: the deepest node whose leaf set is exactly this unit
            root = grp[0]
            target = set(grp)
            while True:
                p = parent[root]
                if p < 0:
                    break
                s, e = cs[p], cs[p + 1]
                sub = set()
                stack = [p]
                while stack:
                    v = stack.pop()
                    a, b = cs[v], cs[v + 1]
                    if a == b:
                        sub.add(v)
                    else:
                        stack.extend(cf[a:b])
                if sub <= target:
                    root = p
                else:
                    break
            p = parent[root]
            if p < 0:
                out.append(set())
                continue
            s, e = cs[p], cs[p + 1]
            sib = cf[s + 1] if cf[s] == root else cf[s]
            out.append(set(below[sib]))
        return out

    def pin_states(self, gt_flat: FlatTree, h, units, st_flat: FlatTree, valid_targets=None):
        """
        The pinned copy of every unit, as a function of the target.

        Mirrors MulTree.get_sister_clades + GroupData.check_fix:
          * the BACKBONE copy's sister set is blanked when the graft lands inside h's
            sister subtree (or above h itself), because then that sister clade contains
            copies of the donor species;
          * the GRAFT's sister set is clade(t), blanked when t is an ancestor of h;
          * a unit is pinned to the backbone if its gene-tree sister species are
            contained in the (unblanked) backbone sister set - a t-INDEPENDENT test -
            and otherwise to the graft if they are contained in clade(t), i.e. iff
            t is an ancestor-or-self of lca_S(sisters(U)). Backbone takes precedence,
            exactly as check_fix tests h1_sisters before hx_sisters.

        Returns (states, free_counts) where states[t] is a tuple over units with entries
        0 (backbone), 1 (graft) or None (free), and free_counts[t] is the number of free
        units, i.e. len(GroupData.ambiguous_groups) as the engine would count it.
        """
        S = self.S
        n = S.n
        sis_sets = self._unit_sisters(gt_flat, units)

        # h's sister clade in S, and the species it contains
        parent = S.parent[h]
        b_sis_species = set()
        sis_node = -1
        if parent >= 0:
            sis_node = S.other_child(parent, h)
            stack = [sis_node]
            while stack:
                v = stack.pop()
                if not S.children[v]:
                    b_sis_species.add(st_flat.node_to_name_id[v])
                else:
                    stack.extend(S.children[v])

        pin0 = [bool(sset) and sset <= b_sis_species for sset in sis_sets]
        z = []
        for sset in sis_sets:
            if not sset:
                z.append(-1)
                continue
            nodes = [S.node_of_species[sp] for sp in sset if sp in S.node_of_species]
            if len(nodes) != len(sset):
                z.append(-1)                    # a species outside S: never graft-pinned
                continue
            cur = nodes[0]
            lca = self._lca_factory()
            for v in nodes[1:]:
                cur = lca(cur, v)
            z.append(cur)

        targets = valid_targets if valid_targets is not None else range(n)
        states, free_counts = {}, [0] * n
        for t in targets:
            backbone_ok = (sis_node >= 0 and t != h and not S.is_desc(t, sis_node))
            graft_ok = not S.is_desc(h, t)
            st = []
            free = 0
            for i in range(len(units)):
                if backbone_ok and pin0[i]:
                    st.append(0)
                elif graft_ok and z[i] >= 0 and S.is_desc(z[i], t):
                    st.append(1)
                else:
                    st.append(None)
                    free += 1
            states[t] = tuple(st)
            free_counts[t] = free
        return states, free_counts

    # ----------------------------------------------------------------------
    def score_all_targets(self, gt_flat: FlatTree, st_flat: FlatTree, h: int,
                          units: Optional[List[List[int]]] = None,
                          valid_targets: Optional[Sequence[int]] = None,
                          pin: bool = True, exact: bool = True, mixed_on_pinned: bool = True,
                          rule: bool = True, gray: bool = True, batch: int = 64):
        """
        costs[t] = MP(G, T(h, t)) for every node t of S, as a NumPy array; entries for
        illegal targets (strict descendants of h) are +inf.
 
        GROUPING - the only switches that change the answer:
          exact=True (default)  keeps the COLLAPSED units and gives every FREE multi-leaf
              unit a third state, "mixed", carried as a canonical split plus a per-target
              correction (_mixed_correction). Exact, at (3/2)^(#free multi-leaf units)
              the work - the same factor grouping.unit_states costs the standard engine.
              A PINNED unit gets no mixed state: pinning already places the unit's
              gene-tree sisters beside one copy, so a split loses on the internal depths,
              on the duplications inside the unit and at its parent. The standard engine
              assumes the same for its fixed groups.
          exact=False           reproduces GRAMPA's collapsing: constant states only.
              rule=True (default) is GRAMPA's sibling-disjoint collapse rule;
              rule=False the transitive one (research knob; matches neither engine).
          units                 overrides the decomposition; only with exact=False.
 
        SPEED DIALS - never change the answer:
          pin    True: sister-based pinning; targets are grouped by pinning pattern and
                 each group swept with its free units only. Falls back to one unpinned
                 sweep when the decomposition would cost more.
          mixed_on_pinned
                 True: pinned units are never mixed - slower and less similar to the
                 standard engine's approach.
          gray   True: incremental Gray-code enumeration (one unit changes per step, and
                 the walk up its ancestors stops at the first dominating node that keeps
                 its class and image). False: rebuild from scratch per combination -
                 the reference implementation, kept as an oracle.
          batch  how many assignments are resolved in one vectorised pass.
        """
        S = self.S
        cs = gt_flat.children_start
        names = gt_flat.node_to_name_id
        n_leaves = sum(1 for u in range(gt_flat.num_nodes) if cs[u] == cs[u + 1])

        # backbone image of every gene leaf
        b_of_leaf: Dict[int, int] = {}
        for u in range(gt_flat.num_nodes):
            if cs[u] == cs[u + 1]:
                sp = names[u]
                if sp not in S.node_of_species:
                    raise RuntimeError(f"gene-tree species id {sp} is absent from the "
                                       f"species tree")
                b_of_leaf[u] = S.node_of_species[sp]

        if st_flat is None:
            raise RuntimeError("score_all_targets needs st_flat to read species ids")
        clade_species = {st_flat.node_to_name_id[v]
                         for v in self._clade_leaves(st_flat, h)}

        if exact:
            if units is not None or not rule:
                raise ValueError("exact=True cannot be used with `units` or `rule=False`.")
        if units is None:
            # If units are still not, then the path is not_exact, and no exogenous units were given,
            # so we need to compute the default units based on the engine rule or not.
            units = compute_units(gt_flat, clade_species, rule=rule)
        g = len(units)

        best = np.full(S.n, np.inf)
        if g == 0:
            costs = self._cost_vector(gt_flat, h, b_of_leaf, set(), n_leaves)
            best[:] = costs
        else:
            self._cur_units = units
            anc_lists, chain_starts = _group_ancestors(gt_flat, units)
            anc_cache = list(zip(anc_lists, chain_starts))
 
            # Per unit: the two CONSTANT states. The third, "mixed", is added only
            # where the unit is FREE - a pinned unit provably prefers its pinned
            # constant to any split (pinning already implies the unit's gene-tree
            # sisters sit beside that copy, so mixing loses on the internal depths, on
            # the duplications inside the unit, AND at its parent), which is also what
            # the standard engine assumes for its fixed groups. Corrections are built
            # lazily: a unit pinned in every region never pays for one.
            const_states = [[(frozenset(), None), (frozenset(u), None)] for u in units]
            mixed_cache: Dict[int, object] = {}
 
            def mixed_state(i):
                if not exact or len(units[i]) < 2:
                    return None
                if i not in mixed_cache:
                    canon = frozenset(units[i][:1])
                    corr = self._mixed_correction(gt_flat, h, units[i], b_of_leaf, canon)
                    mixed_cache[i] = None if corr is None else (canon, corr)
                return mixed_cache[i]
 
            # ---- pinning regions ------------------------------------------
            # Pinning removes only assignments that are provably not the minimum, so it
            # cannot change these scores - it only shrinks the enumeration. Targets are
            # grouped by their pinning pattern and each group is swept with its free
            # units. If that decomposition would cost more than one unpinned sweep, fall
            # back: identical numbers either way.
            regions = None
            if pin:
                st_map, _free = self.pin_states(gt_flat, h, units, st_flat, valid_targets)
                by_pattern: Dict[tuple, List[int]] = {}
                for t, pat in st_map.items():
                    by_pattern.setdefault(pat, []).append(t)
                n_states = [2 + (1 if (exact and len(u) > 1) else 0) for u in units]
 
                def radix(pat):
                    r = 1
                    for i, x in enumerate(pat):
                        r *= n_states[i] if x is None else 1
                    return r
                if sum(radix(p) for p in by_pattern) < _prod(n_states):
                    regions = by_pattern
 
            if regions is None:
                all_states = []
                for i in range(g):
                    sl = list(const_states[i])
                    ms = mixed_state(i)
                    if ms is not None:
                        sl.append(ms)
                    all_states.append(sl)
                mn = self._min_over_states(gt_flat, h, b_of_leaf, units, all_states,
                                           n_leaves, anc_cache, gray, batch)
                if valid_targets is None:
                    np.minimum(best, mn, out=best)
                else:
                    idx = np.asarray(list(valid_targets), dtype=np.int64)
                    best[idx] = np.minimum(best[idx], mn[idx])
            else:
                for pat, tlist in regions.items():
                    sub = []
                    for i, x in enumerate(pat):
                        if x is None:
                            sl = list(const_states[i])
                            ms = mixed_state(i)
                            if ms is not None:
                                sl.append(ms)
                            sub.append(sl)
                        else:
                            sub.append([const_states[i][x]])
                    mn = self._min_over_states(gt_flat, h, b_of_leaf, units, sub,
                                               n_leaves, anc_cache, gray, batch)
                    idx = np.asarray(tlist, dtype=np.int64)
                    best[idx] = np.minimum(best[idx], mn[idx])
 
        for v in range(S.n):
            if v != h and S.is_desc(v, h):
                best[v] = _INF
        return best

    # ----------------------------------------------------------------------
    def _cost_vector(self, gt_flat: FlatTree, h: int, b_of_leaf: Dict[int, int],
                     to_graft: set, n_leaves: int):
        """
        cost(sigma, t) for every t, as a numpy array indexed by species-tree node id.

        The gene-tree pass is O(n_G) Python; everything that depends on the target is
        emitted as a handful of marks and resolved with three cumulative sums:

            subtree sum   Omega(t) = sum over subtree(t)      -> interval in preorder
            root-path sum F(t)     = sum over ancestors of t  -> difference array
            region sums   D_mix(t)                            -> difference array

        so the former `for t in range(n)` assembly loop is now one vectorised expression.
        """
        S = self.S
        cs, cf, post = gt_flat.children_start, gt_flat.children_flat, gt_flat.postorder
        d = S.depth
        dh = d[h]
        n = S.n
        tin, tout = S.tin, S.tout

        cls: Dict[int, int] = {}
        cimg: Dict[int, int] = {}
        bimg: Dict[int, int] = {}

        const = 0
        m = 0
        d_pure = 0
        all_cnt = 0

        # sparse marks, resolved below with np.bincount
        om_i: List[int] = []; om_v: List[int] = []      # omega, at preorder positions
        xw_i: List[int] = []                            # w_u of mixed nodes
        sb_lo: List[int] = []; sb_hi: List[int] = []; sb_v: List[int] = []   # SUB regions
        pt_i: List[int] = []; pt_v: List[int] = []      # POINT regions

        lca = self._lca_factory()

        for u in post:
            s, e = cs[u], cs[u + 1]
            if s == e:
                if u in to_graft:
                    cls[u] = C
                    x = b_of_leaf[u]
                    cimg[u] = x
                    const += d[x] - dh
                    m += 1
                else:
                    cls[u] = B
                    v = b_of_leaf[u]
                    bimg[u] = v
                    const += d[v]
                    om_i.append(tin[v]); om_v.append(1)
                continue

            c1, c2 = cf[s], cf[s + 1]
            k1, k2 = cls[c1], cls[c2]

            if k1 == C and k2 == C:
                cls[u] = C
                x = lca(cimg[c1], cimg[c2])
                cimg[u] = x
                const -= d[x] - dh
                m -= 1
                if x == cimg[c1] or x == cimg[c2]:
                    d_pure += 1

            elif k1 == B and k2 == B:
                cls[u] = B
                v = lca(bimg[c1], bimg[c2])
                bimg[u] = v
                const -= d[v]
                om_i.append(tin[v]); om_v.append(-1)
                if v == bimg[c1] or v == bimg[c2]:
                    d_pure += 1

            else:
                cls[u] = X
                bs = [bimg[c] for c in (c1, c2) if cls[c] != C]
                w = bs[0] if len(bs) == 1 else lca(bs[0], bs[1])
                bimg[u] = w
                xw_i.append(tin[w])

                a, sb, pt = self._mixed_dup_region(c1, c2, k1, k2, w, bimg)
                all_cnt += a
                for idx, val in sb:
                    sb_lo.append(tin[idx]); sb_hi.append(tout[idx] + 1); sb_v.append(val)
                for idx, val in pt:
                    pt_i.append(tin[idx]); pt_v.append(val)

        # ---- resolve, vectorised ---------------------------------------------
        tin_np, tout_np, depth_np = S.tin_np, S.tout_np, S.depth_np

        def _pre(idx, val=None):
            if not idx:
                return np.zeros(n, dtype=np.float64)
            return np.bincount(np.asarray(idx, dtype=np.int64),
                               weights=None if val is None else np.asarray(val, dtype=np.float64),
                               minlength=n)[:n]

        # Omega(t): subtree sum of the omega marks
        om = _pre(om_i, om_v)
        om_c = np.concatenate((np.zeros(1), np.cumsum(om)))
        Omega = om_c[tout_np + 1] - om_c[tin_np]

        # F(t): root-path sum of c(a), c(a) itself a subtree count of the mixed w_u
        xw = _pre(xw_i)
        xw_c = np.concatenate((np.zeros(1), np.cumsum(xw)))
        c_arr = xw_c[tout_np + 1] - xw_c[tin_np]                # per node
        c_root = c_arr[S.root]
        diff = (np.bincount(tin_np, weights=c_arr, minlength=n + 1)
                - np.bincount(tout_np + 1, weights=c_arr, minlength=n + 1))
        F = np.cumsum(diff[:n + 1])[:n][tin_np]

        # D_mix(t): SUB regions (difference array) + POINT marks
        if sb_lo:
            dsub = (np.bincount(np.asarray(sb_lo, dtype=np.int64),
                                weights=np.asarray(sb_v, dtype=np.float64), minlength=n + 1)
                    - np.bincount(np.asarray(sb_hi, dtype=np.int64),
                                  weights=np.asarray(sb_v, dtype=np.float64), minlength=n + 1))
            Dsub = np.cumsum(dsub[:n + 1])[:n][tin_np]
        else:
            Dsub = np.zeros(n)
        Dpt = _pre(pt_i, pt_v)[tin_np] if pt_i else np.zeros(n)

        dup_cost, loss_cost = self.dup_cost, self.loss_cost
        w_dup = dup_cost + 2 * loss_cost
        base = const - 2 * (n_leaves - 1)

        return (w_dup * (d_pure + all_cnt + Dsub + Dpt)
                + loss_cost * (base + m * (depth_np + 1) + Omega - (F - c_root)))
    
    # ----------------------------------------------------------------------
    # EXACT GROUPING WITHOUT GIVING UP COLLAPSING
    # ----------------------------------------------------------------------
    # A unit may take three states: all-backbone, all-graft, or MIXED. Every mixed
    # assignment gives M(root(U)) = rho, so the rest of the tree cannot tell them apart
    # (unit locality); only the LOCAL cost inside the unit differs. So the enumeration
    # can carry ONE canonical split as the representative of "mixed" and add a per-target
    # correction
    #       corr_U[t] = local_min_U(mixed, t) - local_U(canonical split, t)
    # which restores exactness while keeping the collapsed units. The correction is a
    # 3-state DP over the unit's clade, fully vectorised over targets, because inside a
    # unit every image depth is one of
    #       backbone : d_S(x_v) + A(t)        A(t) = 1 iff t is an ancestor-or-self of h
    #       graft    : d_S(x_v) + D(t)+1-d(h) D(t) = d_S(t)
    #       rho      : R(t)                   R(t) = d_S(lca_S(t,h))
    # and every duplication indicator inside the unit is target-INDEPENDENT: the images
    # of a unit's nodes live in subtree(h), in its copy, or at rho, and rho is never
    # inside subtree(h), so the only way a node equals a child is by S-identity (classes
    # 0/0 and 1/1) or by having a rho child.

    def _target_scalars(self, h: int):
        """(A, D, R) as vectors over targets; cached per donor clade."""
        cache = getattr(self, '_scalar_cache', None)
        if cache is None:
            cache = self._scalar_cache = {}
        if h in cache:
            return cache[h]
        S = self.S
        n = S.n
        tin, tout = S.tin_np, S.tout_np
        mark = np.zeros(n)
        mark[S.tin[h]] = 1.0
        c = np.concatenate((np.zeros(1), np.cumsum(mark)))
        A = c[tout + 1] - c[tin]                       # 1 iff h is inside subtree(t)
        diff = (np.bincount(tin, weights=A, minlength=n + 1)
                - np.bincount(tout + 1, weights=A, minlength=n + 1))
        R = np.cumsum(diff[:n + 1])[:n][tin] - 1.0     # d_S(lca_S(t, h))
        D = S.depth_np.astype(np.float64)
        cache[h] = (A, D, R)
        return cache[h]

    def _mixed_correction(self, gt_flat: FlatTree, h, unit, b_of_leaf, canonical_graft):
        """
        corr[t] = local_min(mixed, t) - local(canonical split, t), as a vector.

        Both terms come from the same 3-state DP; only the second is evaluated along a
        fixed class assignment. Returns None for a unit that cannot be split.
        """
        if len(unit) < 2:
            return None
        S = self.S
        cs, cf = gt_flat.children_start, gt_flat.children_flat
        dS, dh = S.depth, S.depth[h]
        A, D, R = self._target_scalars(h)
        base1 = D + (1.0 - dh)
        n = S.n
        lca = self._lca_factory()
        dup_cost, loss_cost = self.dup_cost, self.loss_cost

        root, order = unit_clade(gt_flat, unit)

        x = {}                                        # backbone image of every clade node
        for v in order:
            a, b = cs[v], cs[v + 1]
            x[v] = b_of_leaf[v] if a == b else lca(x[cf[a]], x[cf[a + 1]])

        def dep(cls_, v):
            if cls_ == 0:
                return dS[x[v]] + A
            if cls_ == 1:
                return base1 + dS[x[v]]
            return R

        INF = np.full(n, np.inf)
        best = {}                                     # v -> [3 vectors]
        fixed = {}                                    # v -> (class, cost vector) canonical
        for v in order:
            a, b = cs[v], cs[v + 1]
            if a == b:
                best[v] = [np.zeros(n), np.zeros(n), INF]
                fixed[v] = (1 if v in canonical_graft else 0, np.zeros(n))
                continue
            c1, c2 = cf[a], cf[a + 1]
            same = (x[v] == x[c1] or x[v] == x[c2])   # S-identity: target-independent
            cur = [INF.copy(), INF.copy(), INF.copy()]
            for ka in range(3):
                ba = best[c1][ka]
                if not np.isfinite(ba).any():
                    continue
                for kb in range(3):
                    bb = best[c2][kb]
                    if not np.isfinite(bb).any():
                        continue
                    m = 0 if (ka == 0 and kb == 0) else (1 if (ka == 1 and kb == 1) else 2)
                    dup = (1 if same else 0) if m != 2 else (1 if (ka == 2 or kb == 2) else 0)
                    dm = dep(m, v)
                    cost = (dup_cost * dup
                            + loss_cost * ((dep(ka, c1) - dm - 1 + dup)
                                           + (dep(kb, c2) - dm - 1 + dup)))
                    cand = ba + bb + cost
                    cur[m] = np.minimum(cur[m], cand)
            best[v] = cur

            ka, ca = fixed[c1]
            kb, cb = fixed[c2]
            m = 0 if (ka == 0 and kb == 0) else (1 if (ka == 1 and kb == 1) else 2)
            dup = (1 if same else 0) if m != 2 else (1 if (ka == 2 or kb == 2) else 0)
            dm = dep(m, v)
            fixed[v] = (m, ca + cb + dup_cost * dup
                        + loss_cost * ((dep(ka, c1) - dm - 1 + dup)
                                       + (dep(kb, c2) - dm - 1 + dup)))

        assert fixed[root][0] == 2, "the canonical split must leave the unit root mixed"
        return best[root][2] - fixed[root][1]

    # ----------------------------------------------------------------------
    # INCREMENTAL (GRAY-CODE) + BATCHED EVALUATION
    # ----------------------------------------------------------------------
    # The per-sigma work splits in two halves that are attacked independently:
    #   * the O(n_G) gene-tree classification pass  -> Gray code: consecutive sigmas
    #     differ in ONE unit, so only that unit's ancestors can change class or image;
    #     the accumulators are additive, so their contributions are simply subtracted
    #     and re-added, and the walk stops at the first dominating ancestor that keeps
    #     its class AND image.
    #   * the O(N) resolve                          -> batching: the marks for several
    #     sigmas are stacked into one (B, N) matrix and the cumulative sums run along
    #     axis 1, so B resolves cost a handful of NumPy calls instead of 4B.

    def _node_record(self, cls_u, img_u, c1, c2, cls_arr, img_arr, d, dh, tin, tout):
        """The additive contribution of ONE gene-tree node, as an undoable record."""
        const = m = dpure = allc = 0
        om = []; xw = []; sb = []; pt = []
        if c1 is None:                                   # leaf
            if cls_u == C:
                const += d[img_u] - dh
                m += 1
            else:
                const += d[img_u]
                om.append((tin[img_u], 1))
        elif cls_u == C:
            const -= d[img_u] - dh
            m -= 1
            if img_u == img_arr[c1] or img_u == img_arr[c2]:
                dpure += 1
        elif cls_u == B:
            const -= d[img_u]
            om.append((tin[img_u], -1))
            if img_u == img_arr[c1] or img_u == img_arr[c2]:
                dpure += 1
        else:                                            # mixed
            xw.append(tin[img_u])
            a, sbr, ptr = self._mixed_dup_region(c1, c2, cls_arr[c1], cls_arr[c2],
                                                 img_u, img_arr)
            allc += a
            for idx, val in sbr:
                sb.append((tin[idx], tout[idx] + 1, val))
            for idx, val in ptr:
                pt.append((tin[idx], val))
        return (const, m, dpure, allc, om, xw, sb, pt)

    def _apply(self, st, rec, sign):
        st['const'] += sign * rec[0]
        st['m'] += sign * rec[1]
        st['dpure'] += sign * rec[2]
        st['allc'] += sign * rec[3]
        om, xw, sb, pt = st['om'], st['xw'], st['dsub'], st['pt']
        for i, v in rec[4]:
            om[i] += sign * v
        for i in rec[5]:
            xw[i] += sign
        for lo, hi, v in rec[6]:
            sb[lo] += sign * v
            sb[hi] -= sign * v
        for i, v in rec[7]:
            pt[i] += sign * v

    def _init_state(self, gt_flat: FlatTree, h, b_of_leaf, to_graft):
        """Full classification pass; returns the mutable incremental state."""
        S = self.S
        cs, cf, post = gt_flat.children_start, gt_flat.children_flat, gt_flat.postorder
        d, dh, tin, tout = S.depth, S.depth[h], S.tin, S.tout
        n = S.n
        nn = gt_flat.num_nodes

        st = {'const': 0, 'm': 0, 'dpure': 0, 'allc': 0,
              'om': np.zeros(n), 'xw': np.zeros(n),
              'dsub': np.zeros(n + 1), 'pt': np.zeros(n),
              'cls': [0] * nn, 'img': [0] * nn, 'rec': [None] * nn}
        cls_arr, img_arr = st['cls'], st['img']
        lca = self._lca_factory()

        for u in post:
            a, b = cs[u], cs[u + 1]
            if a == b:
                cls_arr[u] = C if u in to_graft else B
                img_arr[u] = b_of_leaf[u]
                rec = self._node_record(cls_arr[u], img_arr[u], None, None,
                                        cls_arr, img_arr, d, dh, tin, tout)
            else:
                c1, c2 = cf[a], cf[a + 1]
                k1, k2 = cls_arr[c1], cls_arr[c2]
                if k1 == C and k2 == C:
                    cls_arr[u] = C; img_arr[u] = lca(img_arr[c1], img_arr[c2])
                elif k1 == B and k2 == B:
                    cls_arr[u] = B; img_arr[u] = lca(img_arr[c1], img_arr[c2])
                else:
                    cls_arr[u] = X
                    bs = [img_arr[c] for c in (c1, c2) if cls_arr[c] != C]
                    img_arr[u] = bs[0] if len(bs) == 1 else lca(bs[0], bs[1])
                rec = self._node_record(cls_arr[u], img_arr[u], c1, c2,
                                        cls_arr, img_arr, d, dh, tin, tout)
            st['rec'][u] = rec
            self._apply(st, rec, +1)
        return st

    def _flip(self, st, gt_flat: FlatTree, h, b_of_leaf, unit, anc, chain_start, graft_sub):
        """Re-assign one unit to an arbitrary subset `graft_sub` of its leaves,
        updating the accumulators in place."""
        S = self.S
        cs, cf = gt_flat.children_start, gt_flat.children_flat
        d, dh, tin, tout = S.depth, S.depth[h], S.tin, S.tout
        cls_arr, img_arr, recs = st['cls'], st['img'], st['rec']
        lca = self._lca_factory()

        for leaf in unit:
            new_cls = C if leaf in graft_sub else B
            if cls_arr[leaf] == new_cls:
                continue
            self._apply(st, recs[leaf], -1)
            cls_arr[leaf] = new_cls
            img_arr[leaf] = b_of_leaf[leaf]
            recs[leaf] = self._node_record(cls_arr[leaf], img_arr[leaf], None, None,
                                           cls_arr, img_arr, d, dh, tin, tout)
            self._apply(st, recs[leaf], +1)

        for i, u in enumerate(anc):
            a = cs[u]
            c1, c2 = cf[a], cf[a + 1]
            k1, k2 = cls_arr[c1], cls_arr[c2]
            if k1 == C and k2 == C:
                nk = C; ni = lca(img_arr[c1], img_arr[c2])
            elif k1 == B and k2 == B:
                nk = B; ni = lca(img_arr[c1], img_arr[c2])
            else:
                nk = X
                bs = [img_arr[c] for c in (c1, c2) if cls_arr[c] != C]
                ni = bs[0] if len(bs) == 1 else lca(bs[0], bs[1])

            old_k, old_i = cls_arr[u], img_arr[u]
            self._apply(st, recs[u], -1)
            cls_arr[u] = nk; img_arr[u] = ni
            recs[u] = self._node_record(nk, ni, c1, c2, cls_arr, img_arr,
                                        d, dh, tin, tout)
            self._apply(st, recs[u], +1)

            if nk == old_k and ni == old_i and i >= chain_start:
                break                     # nothing above this node can change

    def _snapshot(self, st, n_leaves, corr=None):
        n = self.S.n
        return (st['om'].copy(), st['xw'].copy(), st['dsub'][:n + 1].copy(),
                st['pt'].copy(),
                st['const'] - 2 * (n_leaves - 1), st['m'], st['dpure'] + st['allc'],
                None if corr is None else corr.copy())

    def _resolve_batch(self, rows):
        """Resolve B stacked mark-sets at once; returns a (B, N) cost matrix."""
        S = self.S
        n = S.n
        tin, tout, depth = S.tin_np, S.tout_np, S.depth_np
        B = len(rows)
        om = np.stack([r[0] for r in rows])
        xw = np.stack([r[1] for r in rows])
        ds = np.stack([r[2] for r in rows])
        pt = np.stack([r[3] for r in rows])
        base = np.asarray([r[4] for r in rows], dtype=np.float64)[:, None]
        mm = np.asarray([r[5] for r in rows], dtype=np.float64)[:, None]
        dp = np.asarray([r[6] for r in rows], dtype=np.float64)[:, None]

        zc = np.zeros((B, 1))
        om_c = np.concatenate((zc, np.cumsum(om, axis=1)), axis=1)
        Omega = om_c[:, tout + 1] - om_c[:, tin]

        xw_c = np.concatenate((zc, np.cumsum(xw, axis=1)), axis=1)
        c_arr = xw_c[:, tout + 1] - xw_c[:, tin]                 # (B, N) per node
        c_root = c_arr[:, S.root][:, None]
        diff = np.zeros((B, n + 1))
        np.add.at(diff, (slice(None), tin), c_arr)
        np.add.at(diff, (slice(None), tout + 1), -c_arr)
        F = np.cumsum(diff, axis=1)[:, :n][:, tin]

        Dsub = np.cumsum(ds, axis=1)[:, :n][:, tin]
        Dpt = pt[:, tin]

        w_dup = self.dup_cost + 2 * self.loss_cost
        out = (w_dup * (dp + Dsub + Dpt)
               + self.loss_cost * (base + mm * (depth + 1) + Omega - (F - c_root)))
        if rows[0][7] is not None:
            out = out + np.stack([r[7] for r in rows])     # mixed-state corrections
        return out

    def _min_over_states(self, gt_flat: FlatTree, h, b_of_leaf, units, unit_states, n_leaves,
                         anc_cache, gray: bool = True, batch: int = 64):
        """
        Elementwise minimum over every combination of per-unit STATES.

        unit_states[i] is a list of (graft_subset, correction_or_None) for unit i. With
        two states per unit this is the classic all-backbone / all-graft enumeration;
        with a third it also covers "mixed", represented by a canonical split plus its
        correction vector, which is exact by unit locality.
        """
        if not gray:
            return self._min_over_states_product(gt_flat, h, b_of_leaf,
                                                 unit_states, n_leaves, batch)
        cur = [0] * len(unit_states)
        subs = [unit_states[i][0][0] for i in range(len(unit_states))]
        corrs = [unit_states[i][0][1] for i in range(len(unit_states))]

        init = set().union(*subs) if subs else set()
        st = self._init_state(gt_flat, h, b_of_leaf, init)

        any_corr = any(c is not None for sl in unit_states for _, c in sl)
        run = np.zeros(self.S.n) if any_corr else None
        if run is not None:
            for c in corrs:
                if c is not None:
                    run += c

        best = None
        rows = [self._snapshot(st, n_leaves, run)]

        def flush():
            nonlocal best, rows
            if not rows:
                return
            mn = self._resolve_batch(rows).min(axis=0)
            best = mn if best is None else np.minimum(best, mn)
            rows = []

        radices = [len(sl) for sl in unit_states]
        for gi, choice in _mixed_radix_gray(radices):
            new_sub, new_corr = unit_states[gi][choice]
            anc, chain = anc_cache[gi]
            self._flip(st, gt_flat, h, b_of_leaf, units[gi], anc, chain, new_sub)
            if run is not None:
                if corrs[gi] is not None:
                    run -= corrs[gi]
                if new_corr is not None:
                    run += new_corr
            subs[gi], corrs[gi], cur[gi] = new_sub, new_corr, choice
            rows.append(self._snapshot(st, n_leaves, run))
            if len(rows) >= batch:
                flush()
        flush()
        return best
    
    def _min_over_states_product(self, gt_flat: FlatTree, h, b_of_leaf, unit_states,
                                 n_leaves: int, batch: int = 64):
        """Reference enumeration: rebuild the state from scratch for every combination.
        Same numbers as the incremental path; kept as an oracle and for --optim without
        the Gray-code bit."""
        best = None
        rows = []
        def flush():
            nonlocal best, rows
            if not rows:
                return
            mn = self._resolve_batch(rows).min(axis=0)
            best = mn if best is None else np.minimum(best, mn)
            rows = []
        any_corr = any(c is not None for sl in unit_states for _, c in sl)
        for combo in itertools.product(*[range(len(sl)) for sl in unit_states]):
            to_graft = set()
            run = np.zeros(self.S.n) if any_corr else None
            for i, k in enumerate(combo):
                sub, corr = unit_states[i][k]
                to_graft |= sub
                if corr is not None:
                    run += corr
            st = self._init_state(gt_flat, h, b_of_leaf, to_graft)
            rows.append(self._snapshot(st, n_leaves, run))
            if len(rows) >= batch:
                flush()
        flush()
        return best

    # ----------------------------------------------------------------------
    def _mixed_dup_region(self, c1, c2, k1, k2, w, bimg):
        """
        The set of targets for which the MIXED node u is a duplication, as
        (all_count, [(node, delta_SUB)], [(node, delta_POINT)]).

        Writing ID_t(x) for the identity of x's image in T(h,t), and using the Lemma
        (everything in the graft collapses to r):

          ID_t(u) = r                      if w is a descendant-or-self of t
                  = lca_S(t, w)            otherwise

        Case {C, X}:  the graft-side child's image already lies at or above r, so
                      lca_T(C, M(x)) = M(x): u is ALWAYS a duplication.
        Case {X, X}:  with w = lca(w1, w2), for t at-or-above w both children are r; for
                      t strictly below w one child's image is exactly w (S is binary, so
                      t misses at least one of the two child-subtrees); for t elsewhere
                      all three images coincide. ALWAYS a duplication.
        Case {C, B}:  here w equals the pure-B child's image v, so ID(u) = ('S', w) iff
                      t is strictly inside subtree(w) -> region SUB(w) \\ {w}.
        Case {X, B}:  partition the targets into
                        (a) ancestors-or-self of w        -> both are r, duplication
                        (c) outside subtree(w), not an ancestor
                                                          -> all lca's coincide, duplication
                        (b) strictly inside subtree(w)    -> duplication iff the mixed
                            child's image is w itself, or the pure-B child's image is w,
                            or t sits in the child-subtree of w that does NOT contain w_x
                      i.e. ALL - SUB(w) + POINT(w), plus the qualifying part of (b).
        Note lca_S(t, w) is always an ancestor-or-self of w, so a pure-B child whose
        image is strictly below w can never match: that is why only v == w matters.

        `bimg` need only be defined for the B- and X-class children; a pure-C child's
        image can never equal ID(u), which is r or a backbone node.
        """
        S = self.S
        if k1 == C and k2 == C:
            raise AssertionError("a node with two pure-C children is pure-C")

        # {C, X} and {X, X}: always a duplication
        if (k1 == C and k2 == X) or (k2 == C and k1 == X) or (k1 == X and k2 == X):
            return 1, (), ()

        # {C, B}: w == the B child's image
        if (k1 == C and k2 == B) or (k2 == C and k1 == B):
            return 0, ((w, 1),), ((w, -1),)

        # {X, B}
        x_child = c1 if k1 == X else c2
        b_child = c2 if k1 == X else c1
        w_x = bimg[x_child]
        v = bimg[b_child]

        all_cnt = 1
        sb = [(w, -1)]
        pt = [(w, 1)]
        if v == w or w_x == w:
            sb.append((w, 1))
            pt.append((w, -1))
        else:
            other = S.other_child(w, S.child_toward(w, w_x))
            sb.append((other, 1))
        return all_cnt, tuple(sb), tuple(pt)

    # ----------------------------------------------------------------------
    def _lca_factory(self):
        S = self.S
        tin, tout, parent, depth = S.tin, S.tout, S.parent, S.depth

        def lca(a: int, b: int) -> int:
            if a == b:
                return a
            if tin[a] <= tin[b] <= tout[a]:
                return a
            if tin[b] <= tin[a] <= tout[b]:
                return b
            x = a
            while not (tin[x] <= tin[b] <= tout[x]):
                x = parent[x]
            return x
        return lca

    # ----------------------------------------------------------------------
    @staticmethod
    def _clade_leaves(st_flat: FlatTree, h: int) -> List[int]:
        cs, cf = st_flat.children_start, st_flat.children_flat
        out, stack = [], [h]
        while stack:
            v = stack.pop()
            s, e = cs[v], cs[v + 1]
            if s == e:
                out.append(v)
            else:
                stack.extend(cf[s:e])
        return out

# --------------------------------------------------------------------------
# Traditioal pairwise reconciliation
# --------------------------------------------------------------------------

class PairwiseRecon:
    """A class for performing pairwise reconciliation between gene trees and species trees."""

    def __init__(self, dup_cost: int = 1, loss_cost: int = 1, strict_targets: bool = True):
        self.dup_cost = dup_cost
        self.loss_cost = loss_cost
        self.strict_targets = strict_targets

    # --------------------------------------------------------------------------
    # COMMON LOGIC
    # --------------------------------------------------------------------------

    @staticmethod
    def build_target_map(mul_flat: FlatTree, registry: NameRegistry) -> Dict[int, List[int]]:
        """
        Calculates the target map ONCE per MUL-tree to avoid redundant string parsing.
        Maps base Species IDs to a list of available MUL-tree node indices.
        The list index is the sub-genome tag count ('*'), so it is the index that
        GroupData.fixed_groups refers to; do NOT reorder or compact it here.
        """
        target_map: Dict[int, List[int]] = {}
        cs = mul_flat.children_start
        names = mul_flat.node_to_name_id
        for i in range(mul_flat.num_nodes):
            if cs[i] != cs[i + 1]:
                continue
            name_id = names[i]
            if name_id == -1:
                continue

            sp_name = registry.get_name(name_id)
            base_name = sp_name.replace("*", "")
            base_id = registry.get_id(base_name)

            targets = target_map.get(base_id)
            if targets is None:
                targets = target_map[base_id] = []

            tag_count = sp_name.count("*")
            while len(targets) <= tag_count:
                targets.append(-1)
            targets[tag_count] = i

        # Fill holes so that a fixed t_idx always resolves to a real node.
        for targets in target_map.values():
            if not targets:
                continue
            valid_target = next((t for t in targets if t != -1), -1)
            for k in range(len(targets)):
                if targets[k] == -1:
                    targets[k] = valid_target

        return target_map

    @staticmethod
    def translate_groups_to_ids(gt_flat: FlatTree, group_data: GroupData
                                ) -> Tuple[List[List[int]], List[Tuple[List[int], int]]]:
        """Map the registry name-IDs stored in GroupData to node IDs of this gene tree."""
        lookup = gt_flat.name_id_to_node_id
        ambig_groups_ids = []
        # grp_ids is List[int]
        for grp_ids in group_data.ambiguous_groups:
            valid_ids = [lookup[nid] for nid in grp_ids if nid in lookup]
            if valid_ids:
                ambig_groups_ids.append(valid_ids)

        fixed_groups_ids = []
        for grp_ids, target_idx in group_data.fixed_groups:
            valid_ids = [lookup[nid] for nid in grp_ids if nid in lookup]
            if valid_ids:
                fixed_groups_ids.append((valid_ids, target_idx))

        return ambig_groups_ids, fixed_groups_ids

    # --------------------------------------------------------------------------
    # CORE SCORING (one implementation, three call patterns)
    # --------------------------------------------------------------------------

    @staticmethod
    def _scan(gt: FlatTree, st: FlatTree, dup_cost: int, loss_cost: int,
              nodes: Iterable[int], lca_maps: array.array,
              contrib: Optional[array.array] = None,
              add_root: bool = True, base: int = 0) -> int:
        """
        Score every node of `nodes` (which MUST be a subsequence of gt.postorder) and
        write its LCA image into `lca_maps`. Optionally records the per-node cost in
        `contrib`. Returns base + the summed cost.

        This is the single implementation of the Zmasek-Eddy / Durand duplication-loss
        cost; reconcile_sl, the per-combination loop and the incremental updater
        all route through it. The LCA query is inlined (rather than calling
        st.get_lca) because it sits in the innermost loop.
        """
        cs, cf = gt.children_start, gt.children_flat
        nd, fv, dep, eul, rmq = (st.node_depths, st.first_visit, st.depths,
                                 st.euler_tour, st.rmq_table)
        score = base

        # --- iterate postorder ---
        for u in nodes:
            s = cs[u]

            # --- skip leaves (they are already mapped) ---
            if s == cs[u + 1]:
                continue

            # --- internal nodes ---
            m1 = lca_maps[cf[s]]
            m2 = lca_maps[cf[s + 1]]

            # --- inlined O(1) LCA (Euler tour + sparse table) ---
            if m1 == m2:
                m = m1
            else:
                f = fv[m1]
                l = fv[m2]
                if f > l:
                    f, l = l, f
                k = (l - f + 1).bit_length() - 1
                a = rmq[k][f]
                b = rmq[k][l - (1 << k) + 1]
                m = eul[a] if dep[a] < dep[b] else eul[b]

            lca_maps[u] = m

            d = nd[m]
            if m == m1 or m == m2:
                c = dup_cost
                l1 = nd[m1] - d
                l2 = nd[m2] - d
            else:
                c = 0
                l1 = nd[m1] - d - 1
                l2 = nd[m2] - d - 1
            if l1 > 0:
                c += loss_cost * l1
            if l2 > 0:
                c += loss_cost * l2

            if contrib is not None:
                contrib[u] = c
            score += c

        # --- add root penalty if requested ---
        if add_root:
            root_depth = st.node_depths[lca_maps[gt.postorder[-1]]]
            if root_depth > 0:
                score += loss_cost * root_depth

        return score

    @staticmethod
    def _init_leaf_maps(gt: FlatTree, target_map: Dict[int, List[int]],
                        n: int, strict_targets: bool = True) -> Tuple[array.array, List[int]]:
        """Allocate the node->node map and set every leaf to its first available target.
        Returns (lca_maps, leaf_ids)."""
        cs, names = gt.children_start, gt.node_to_name_id
        lca_maps = array.array('i', [-1] * n)
        leaves = []
        for i in range(n):
            if cs[i] != cs[i + 1]:
                continue
            leaves.append(i)
            targets = target_map.get(names[i])
            if not targets:
                if strict_targets:
                    raise RuntimeError(
                        f"Gene-tree leaf {i} maps to species id {names[i]}, which has no "
                        f"counterpart in the species/MUL-tree. Refusing to fall back to "
                        f"the tree root (that silently inflates the loss count).")
                targets = [0]
            lca_maps[i] = targets[0]
        return lca_maps, leaves

    @staticmethod
    def _build_map(gt: FlatTree, st: FlatTree, lca_maps: Sequence[int],
                   registry: NameRegistry) -> Map:
        """
        Rebuild the per-node duplication / loss annotation from a finished map.
        Split out of the scoring loop: dups and losses are pure functions of lca_maps,
        so computing them in a second pass keeps three `if retmap` tests out of the
        innermost loop without duplicating the cost logic.
        Semantics are identical to the original in-loop bookkeeping:
          node_dups[v]   = 1 iff v is a duplication node
          node_losses[v] = losses on the branch ABOVE v (root: losses above the root)
        """
        if registry is None:
            raise ValueError("Registry required for returning maps in flat mode")

        cs, cf, post = gt.children_start, gt.children_flat, gt.postorder
        nd = st.node_depths
        node_dups: Dict[int, int] = {}
        node_losses: Dict[int, int] = {}

        for u in post:
            s = cs[u]
            if s == cs[u + 1]:
                node_dups[u] = 0
                node_losses[u] = 0
                continue
            c1 = cf[s]
            c2 = cf[s + 1]
            m1, m2, m = lca_maps[c1], lca_maps[c2], lca_maps[u]
            is_dup = 1 if (m == m1 or m == m2) else 0
            node_dups[u] = is_dup
            d = nd[m]
            l1 = nd[m1] - d - 1 + is_dup
            l2 = nd[m2] - d - 1 + is_dup
            node_losses[c1] = l1 if l1 > 0 else 0
            node_losses[c2] = l2 if l2 > 0 else 0
            node_losses[u] = 0                     # overwritten by u's parent, if any

        root_id = post[-1]
        root_depth = nd[lca_maps[root_id]]
        if root_depth > 0:
            node_losses[root_id] += root_depth

        final_maps_str: Dict[str, List[str]] = {}
        final_dups_str: Dict[str, int] = {}
        final_losses_str: Dict[str, int] = {}
        id_to_name = gt.node_id_to_name_id
        get_name = registry.get_name
        st_names = st.node_to_name_id

        for u in range(gt.num_nodes):
            u_full_id = id_to_name.get(u)
            if u_full_id is None:
                continue                            # unnamed internal node: nothing to report
            u_full_name = get_name(u_full_id)
            final_maps_str[u_full_name] = [get_name(st_names[lca_maps[u]])]
            if u in node_dups:
                final_dups_str[u_full_name] = node_dups[u]
            if u in node_losses:
                final_losses_str[u_full_name] = node_losses[u]

        return Map(n_dups=sum(final_dups_str.values()),
                   n_losses=sum(final_losses_str.values()),
                   cor=final_maps_str,
                   dups=final_dups_str,
                   losses=final_losses_str)

    def reconcile_sl(self, gt: FlatTree, st: FlatTree,
                       registry: NameRegistry = None,
                       precalc_map: Union[Dict[int, int], List[int], array.array, None] = None,
                       retmap: bool = False) -> Tuple[int, Optional[List[Map]]]:
        """O(1)-LCA integer-array reconciliation of a gene tree to a singly-labeled or a disambiguated-labeled species tree.
        Returns (score, [map]) if retmap else (score, None)."""
        n = gt.num_nodes
        strict_targets = self.strict_targets

        if precalc_map is None:
            st_leaf_map = {}
            cs_st, names_st = st.children_start, st.node_to_name_id
            for i in range(st.num_nodes):
                if cs_st[i] == cs_st[i + 1]:
                    st_leaf_map[names_st[i]] = i

            lca_maps = array.array('i', [-1] * n)
            cs, names = gt.children_start, gt.node_to_name_id
            for i in range(n):
                if cs[i] != cs[i + 1]:
                    continue
                target = st_leaf_map.get(names[i], -1)
                if target == -1:
                    if strict_targets:
                        raise RuntimeError(
                            f"Gene-tree leaf {i} (species id {names[i]}) is absent from the "
                            f"species tree; reconciliation would be undefined.")
                    target = 0
                lca_maps[i] = target
        elif isinstance(precalc_map, dict):
            lca_maps = array.array('i', [-1] * n)
            for k, v in precalc_map.items():
                lca_maps[k] = v
        else:
            lca_maps = array.array('i', precalc_map)

        score = PairwiseRecon._scan(gt, st, self.dup_cost, self.loss_cost, gt.postorder, lca_maps)

        if retmap:
            return score, [PairwiseRecon._build_map(gt, st, lca_maps, registry)]
        return score, None

    # --------------------------------------------------------------------------
    # PERMUTATION LOGIC
    # --------------------------------------------------------------------------

    def _prepare_groups(self, gt_flat: FlatTree, target_map: Dict[int, List[int]],
                        ambig_groups: List[List[int]], fixed_groups: List[Tuple[List[int], int]],
                        lca_maps: array.array, mul_flat, use_exact
                        ) -> Tuple[List[List[int]], List[List[Tuple[int, ...]]]]:
        """
        Resolve, for every ambiguous group, the tuple of MUL-tree nodes its leaves take
        under each choice index, and apply the initial assignment (fixed groups -> their
        pinned copy, ambiguous groups -> choice 0).

        A group may span SEVERAL species of the hybrid clade (a duplicate-free clade is
        collapsed as a unit), and each of its leaves is mapped through ITS OWN target
        list at the group's shared choice index - exactly as the shipped implementation
        does. The number of choices is taken from the first leaf's list, and an index
        beyond a given leaf's list falls back to 0.

        Two choice indices that resolve to the same node FOR EVERY LEAF of the group are
        redundant (build_target_map pads missing sub-genome slots by repetition); only
        the first of them is enumerated. Deduplicating on the whole tuple - rather than
        on the first leaf's list - keeps this exact for multi-species groups.
        """
        names = gt_flat.node_to_name_id
        group_leaves: List[List[int]] = []
        group_assign: List[List[Tuple[int, ...]]] = []

        def _targets(nid: int) -> List[int]:
            targets = target_map.get(names[nid])
            if not targets:
                if self.strict_targets:
                    raise RuntimeError(
                        f"Gene-tree leaf {nid} maps to species id {names[nid]}, which is "
                        f"absent from the MUL-tree.")
                return [0]
            return targets

        for grp in ambig_groups:
            leaf_targets = {nid: _targets(nid) for nid in grp}
            assigns = unit_states(gt_flat, grp, leaf_targets, mul_flat,
                                  self.dup_cost, self.loss_cost, exact=use_exact)
            group_leaves.append(grp)
            group_assign.append(assigns)
            for nid, node in zip(grp, assigns[0]):
                lca_maps[nid] = node

        for grp, t_idx in fixed_groups:
            for nid in grp:
                tl = _targets(nid)
                lca_maps[nid] = tl[t_idx] if 0 <= t_idx < len(tl) else tl[0]

        return group_leaves, group_assign

    def reconcile_permutation(self, gt_flat: FlatTree, mul_flat: FlatTree,
                              registry: NameRegistry, group_data: GroupData,
                              target_map: Dict[int, List[int]],
                              retmap: bool = False, use_gray: bool = True,
                              use_exact: bool = False,
                              ) -> Tuple[Union[int, float], Optional[List[Map]]]:
        """
        Minimise the reconciliation cost over all allele maps of this gene tree.

        use_gray=True  : Gray-code enumeration with incremental rescoring. Consecutive
                         combinations differ in exactly one group, so only that group's
                         ancestors can change; the walk stops as soon as an ancestor that
                         dominates the group keeps its image. Exact - the score is a sum
                         of per-node contributions and only nodes whose children's images
                         changed are recomputed.
        use_gray=False : the reference implementation - one full postorder scan per
                         combination. Kept as a correctness oracle.
        """
        dup_cost, loss_cost = self.dup_cost, self.loss_cost
        ambig_groups, fixed_groups = PairwiseRecon.translate_groups_to_ids(gt_flat, group_data)

        n = gt_flat.num_nodes
        lca_maps, _leaves = PairwiseRecon._init_leaf_maps(gt_flat, target_map, n, strict_targets=self.strict_targets)
        group_leaves, group_assign = self._prepare_groups(
            gt_flat, target_map, ambig_groups, fixed_groups, lca_maps,
            mul_flat=mul_flat, use_exact=use_exact
            )

        radices = [len(a) for a in group_assign]
        best_score: Union[int, float] = _INF
        best_maps: List[List[int]] = []          # snapshots of lca_maps, resolved later

        def _record(score: Union[int, float], snapshot: Optional[array.array]) -> None:
            nonlocal best_score, best_maps
            if score < best_score:
                best_score = score
                best_maps = [snapshot] if snapshot is not None else []
            elif retmap and score == best_score and snapshot is not None:
                best_maps.append(snapshot)

        # ---------------- reference path -------------------------------------
        if not use_gray:
            for combo in itertools.product(*(range(r) for r in radices)):
                for gi, choice in enumerate(combo):
                    for nid, node in zip(group_leaves[gi], group_assign[gi][choice]):
                        lca_maps[nid] = node
                score = PairwiseRecon._scan(gt_flat, mul_flat, dup_cost, loss_cost,
                                         gt_flat.postorder, lca_maps)
                _record(score, lca_maps[:] if retmap else None)

        # ---------------- incremental path -----------------------------------
        else:
            contrib = array.array('i', [0] * n)
            total = PairwiseRecon._scan(gt_flat, mul_flat, dup_cost, loss_cost,
                                     gt_flat.postorder, lca_maps, contrib=contrib,
                                     add_root=False)
            root_id = gt_flat.postorder[-1]
            nd = mul_flat.node_depths
            root_pen = loss_cost * nd[lca_maps[root_id]]
            if root_pen < 0:
                root_pen = 0
            total += root_pen
            _record(total, lca_maps[:] if retmap else None)

            if radices:
                anc_lists, chain_starts = _group_ancestors(gt_flat, group_leaves)
                cs, cf = gt_flat.children_start, gt_flat.children_flat
                fv, dep, eul, rmq = (mul_flat.first_visit, mul_flat.depths,
                                     mul_flat.euler_tour, mul_flat.rmq_table)
                root_is_leaf = cs[root_id] == cs[root_id + 1]

                for gi, choice in _mixed_radix_gray(radices):
                    for nid, node in zip(group_leaves[gi], group_assign[gi][choice]):
                        lca_maps[nid] = node

                    if root_is_leaf:                      # single-leaf gene tree
                        new_pen = loss_cost * nd[lca_maps[root_id]]
                        total += new_pen - root_pen
                        root_pen = new_pen

                    anc = anc_lists[gi]
                    cstart = chain_starts[gi]
                    for i in range(len(anc)):
                        u = anc[i]
                        s = cs[u]
                        m1 = lca_maps[cf[s]]
                        m2 = lca_maps[cf[s + 1]]
                        if m1 == m2:
                            m = m1
                        else:
                            f = fv[m1]
                            l = fv[m2]
                            if f > l:
                                f, l = l, f
                            k = (l - f + 1).bit_length() - 1
                            a = rmq[k][f]
                            b = rmq[k][l - (1 << k) + 1]
                            m = eul[a] if dep[a] < dep[b] else eul[b]

                        d = nd[m]
                        if m == m1 or m == m2:
                            c = dup_cost
                            l1 = nd[m1] - d
                            l2 = nd[m2] - d
                        else:
                            c = 0
                            l1 = nd[m1] - d - 1
                            l2 = nd[m2] - d - 1
                        if l1 > 0:
                            c += loss_cost * l1
                        if l2 > 0:
                            c += loss_cost * l2

                        total += c - contrib[u]
                        contrib[u] = c

                        old = lca_maps[u]
                        if m != old:
                            lca_maps[u] = m
                            if u == root_id:
                                new_pen = loss_cost * d
                                total += new_pen - root_pen
                                root_pen = new_pen
                        elif i >= cstart:
                            break                        # image unchanged at a full
                                                         # ancestor -> nothing above moves

                    _record(total, lca_maps[:] if retmap else None)

        #if best_score == _INF:                            # no combination at all
        #    raise RuntimeError("reconcile_permutation produced no candidate mapping.")

        if retmap:
            return best_score, [PairwiseRecon._build_map(gt_flat, mul_flat, snap, registry)
                                for snap in best_maps]
        return best_score, None

