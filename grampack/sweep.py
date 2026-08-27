"""
sweep.py - score EVERY placement of a donor clade in one sweep, instead of one
reconciliation per candidate MUL-tree.

Status: EXPERIMENTAL / research prototype. It does not replace anything; it is meant to
be run alongside the normal engine and differentially tested against it.

--------------------------------------------------------------------------------------
The idea
--------------------------------------------------------------------------------------
Every candidate MUL-tree is T(h, t) = S + a copy C of subtree(h) grafted above a target
t. Instead of building T(h, t) and reconciling for each of the O(N) targets, we compute,
for each allele assignment sigma, the cost as an explicit FUNCTION of t, and evaluate it
for all t with two linear passes over S.

The structural fact that makes this possible:

    Lemma (graft collapse). In T(h, t) with new node r above t, for any c in C and any
    backbone node v:
        lca_T(c, v) = r                if v is a descendant-or-self of t
                    = lca_S(t, v)      otherwise
    and in BOTH cases  d_T(lca_T(c,v)) = d_S(lca_S(t,v)).

    Proof. If v <= t then lca_S(t,v) = t and d_T(r) = d_S(t) (r takes t's old depth).
    Otherwise lca_S(t,v) is a proper ancestor of t, hence outside subtree(t), so the
    insertion leaves its depth unchanged. []

So the whole graft behaves, from the outside, like the single point r: all coupling with
the backbone runs through d_S(t) and lca_S(t, .).

Combined with the closed form of the reconciliation cost

    cost = (dup + 2*loss) * D + loss * ( sum_leaves d(M) - sum_internal d(M) - 2(n-1) )

and the classification of every gene-tree node as

    pure-C  (all leaves assigned to the graft)
    pure-B  (all leaves on the backbone)
    mixed   (both)

we get, for a FIXED sigma:

    cost(sigma, t) =  const(sigma)
                    + loss * m(sigma) * (d_S(t) + 1)        m = # maximal pure-C clades
                    + loss * Omega_sigma(t)                 subtree sum over S
                    - loss * F_sigma(t)                     ancestral prefix sum over S
                    + (dup + 2*loss) * ( D_pure(sigma) + D_mix(sigma, t) )

    Omega(t) = #{ pure-B LEAF images inside subtree(t) } - #{ pure-B INTERNAL images ... }
    F(t)     = sum over mixed nodes u of d_S( lca_S(t, w_u) ),  w_u = lca of u's backbone
               images (a constant per node), which expands to a root-path sum:
                   F(t) = sum over a in anc(t), a != root, of c(a),
                   c(a) = #{ mixed u : w_u in subtree(a) }

A pure-B node has only pure-B children and a pure-C node only pure-C children (a mixed
child forces its parent to be mixed), so D_pure does not depend on t at all. Only the
duplication status of MIXED nodes moves with the target, and each of those is an O(1)
region of S (see _mixed_dup_region).

--------------------------------------------------------------------------------------
Scope of this prototype
--------------------------------------------------------------------------------------
* Exactly TWO copies: the backbone original and one grafted copy (single-target MUL-
  trees). That covers every MUL-tree at depth 0 and most of the others.
* The species tree S must be singly labelled and bifurcating.
* Group collapsing is OFF by default (one unit per movable gene copy), which is exact.
  Collapsing IS available, but building this prototype turned up cases where collapsing
  a duplicate-free clade loses the optimum - most visibly for the autopolyploid
  placement t == h, where splitting a unit across the two sibling copies IS the
  autopolyploid signal. So collapse=True is an approximation. To reproduce the engine
  bit-for-bit, pass the engine's own units via `units=`.
* Sister-based PINNING is not used: it depends on the target, whereas the unit structure
  does not. Dropping it is exact - the pinned assignment is provably the optimal one, so
  min over all sigma equals min over the pinned ones - but it costs 2^g instead of
  2^(g - pinned). Groups (collapsing) ARE used: they are target-independent.

Complexity: O(2^g * (n_G + N)) for ALL N targets, versus O(N * 2^g * n_G) today.
"""

import itertools
from typing import Dict, List, Optional, Sequence, Tuple

_INF = float('inf')

# node classes
C, B, X = 0, 1, 2                 # pure-C (graft), pure-B (backbone), mixed


class SpeciesIndex:
    """Precomputed, target-independent structure of the species tree."""

    __slots__ = ('n', 'parent', 'depth', 'preorder', 'postorder', 'children',
                 'tin', 'tout', 'root', 'sp_of_leaf', 'node_of_species')

    def __init__(self, st_flat, registry=None):
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
# Gene-tree side
# --------------------------------------------------------------------------

def default_units(gt_flat, sidx: SpeciesIndex, clade_species: set,
                  engine_rule: bool = True) -> List[List[int]]:
    """
    The ambiguous units: clades of G whose leaves are all donor-clade gene copies.

    engine_rule=True (default) reproduces models.compute_groups EXACTLY, including its
    duplicate test: a clade is collapsed when the species sets of its TWO CHILDREN are
    disjoint - a check made only at that node, not over the whole clade. A species may
    therefore appear several times inside one engine group (e.g. ((x,x),y) merges with a
    sibling carrying neither x nor y). Use this whenever the sweep must be comparable
    with the standard pathway.

    engine_rule=False requires the whole clade to be duplicate-free, which is what the
    GRAMPA paper describes ("only if there is not more than one copy of a species among
    the clades can they be grouped"). It yields strictly finer units, hence a larger
    assignment space and a score that is <= the engine's, and closer to the true MP.
    No pinning is applied either way: pinning is target-dependent and is handled by
    TargetSweep.pin_states.
    """
    cs, cf, post = gt_flat.children_start, gt_flat.children_flat, gt_flat.postorder
    names = gt_flat.node_to_name_id

    if engine_rule:
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


# --------------------------------------------------------------------------
# The sweep
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

    def _unit_sisters(self, gt_flat, units):
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

    def pin_states(self, gt_flat, h, units, st_flat, valid_targets=None):
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
    def score_all_targets(self, gt_flat, h: int, units: Optional[List[List[int]]] = None,
                          st_flat=None, collapse: bool = False, pin: bool = False,
                          valid_targets: Optional[Sequence[int]] = None) -> List[float]:
        """
        Returns costs[t] = MP(G, T(h, t)) for every node t of S; entries for illegal
        targets (strict descendants of h) are +inf.

        `units` are the ambiguous units as gene-tree LEAF ids. Pass GRANDMA's own
        GroupData (translated to node ids) to mirror the engine exactly.

        collapse=False (default) enumerates one unit per movable LEAF: EXACT, 2^(#movable).
        collapse=True groups duplicate-free donor-clade clades. That is much faster but
        it is an APPROXIMATION - see the module docstring. For a faithful comparison with
        the engine, pass the engine's own units instead.
        """
        S = self.S
        cs, cf, post = gt_flat.children_start, gt_flat.children_flat, gt_flat.postorder
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

        if units is None:
            units = (default_units(gt_flat, S, clade_species, engine_rule=True)
                     if collapse else
                     [[x] for x in sorted(b_of_leaf)
                      if gt_flat.node_to_name_id[x] in clade_species])

        # leaves that may move; everything else is pinned to the backbone
        movable = [x for grp in units for x in grp]
        in_unit = {x: i for i, grp in enumerate(units) for x in grp}
        g = len(units)

        best = [_INF] * S.n

        # ---- pinning regions -------------------------------------------------
        # Sister-based pinning removes assignments that are provably not the minimum
        # among the constant ones (the pinned copy is optimal), so it CANNOT change the
        # scores this function returns - it only shrinks the enumeration. Targets are
        # therefore grouped by their pinning pattern and each group is swept with its
        # free units only. If that decomposition would cost more than one unpinned
        # sweep (many patterns, few pins), we fall back - same numbers either way.
        regions = None
        if pin and g:
            states, _free = self.pin_states(gt_flat, h, units, st_flat, valid_targets)
            by_pattern: Dict[tuple, List[int]] = {}
            for t, pat in states.items():
                by_pattern.setdefault(pat, []).append(t)
            cost_pinned = sum(1 << sum(1 for x in pat if x is None) for pat in by_pattern)
            if cost_pinned < (1 << g):
                regions = by_pattern

        if regions is None:
            tset = None if valid_targets is None else set(valid_targets)
            for combo in itertools.product((0, 1), repeat=g):
                to_graft = set()
                for i, choice in enumerate(combo):
                    if choice:
                        to_graft.update(units[i])
                costs = self._cost_vector(gt_flat, h, b_of_leaf, to_graft, n_leaves)
                for t in range(S.n):
                    if (tset is None or t in tset) and costs[t] < best[t]:
                        best[t] = costs[t]
        else:
            for pat, tlist in regions.items():
                free_idx = [i for i, x in enumerate(pat) if x is None]
                fixed_graft = set()
                for i, x in enumerate(pat):
                    if x == 1:
                        fixed_graft.update(units[i])
                for combo in itertools.product((0, 1), repeat=len(free_idx)):
                    to_graft = set(fixed_graft)
                    for k, choice in enumerate(combo):
                        if choice:
                            to_graft.update(units[free_idx[k]])
                    costs = self._cost_vector(gt_flat, h, b_of_leaf, to_graft, n_leaves)
                    for t in tlist:
                        if costs[t] < best[t]:
                            best[t] = costs[t]

        # illegal targets: strictly inside the donor clade
        for v in range(S.n):
            if v != h and S.is_desc(v, h):
                best[v] = _INF
        return best

    # ----------------------------------------------------------------------
    def _cost_vector(self, gt_flat, h: int, b_of_leaf: Dict[int, int],
                     to_graft: set, n_leaves: int) -> List[float]:
        """cost(sigma, t) for every t, in O(n_G + N)."""
        S = self.S
        cs, cf, post = gt_flat.children_start, gt_flat.children_flat, gt_flat.postorder
        d = S.depth
        dh = d[h]
        n = S.n

        cls: Dict[int, int] = {}
        cimg: Dict[int, int] = {}              # image inside subtree(h)
        bimg: Dict[int, int] = {}              # backbone image / w_u

        const = 0                              # t-free part of (sum_leaves - sum_int)
        m = 0                                  # coefficient of (d(t)+1)
        omega = [0] * n                        # marks resolved by subtree_sum
        xw = [0] * n                           # marks for c(a) = #{mixed u : w_u <= a}
        n_mixed = 0
        d_pure = 0

        # duplication regions for mixed nodes, as signed primitives
        all_cnt = 0
        sub_mark = [0] * n                     # SUB(v)  -> resolved by rootpath_sum
        pt_mark = [0] * n                      # POINT(v)

        lca = self._lca_factory()

        for u in post:
            s, e = cs[u], cs[u + 1]
            if s == e:                                            # ---- leaf
                if u in to_graft:
                    cls[u] = C
                    cimg[u] = b_of_leaf[u]
                    const += d[cimg[u]] - dh
                    m += 1                                        # provisional, see below
                else:
                    cls[u] = B
                    bimg[u] = b_of_leaf[u]
                    const += d[bimg[u]]
                    omega[bimg[u]] += 1
                continue

            c1, c2 = cf[s], cf[s + 1]
            k1, k2 = cls[c1], cls[c2]

            if k1 == C and k2 == C:                               # ---- pure C
                cls[u] = C
                x = lca(cimg[c1], cimg[c2])
                cimg[u] = x
                const -= d[x] - dh
                m -= 1                                            # merges two blocks
                if x == cimg[c1] or x == cimg[c2]:
                    d_pure += 1

            elif k1 == B and k2 == B:                             # ---- pure B
                cls[u] = B
                v = lca(bimg[c1], bimg[c2])
                bimg[u] = v
                const -= d[v]
                omega[v] -= 1
                if v == bimg[c1] or v == bimg[c2]:
                    d_pure += 1

            else:                                                 # ---- mixed
                cls[u] = X
                bs = [bimg[c] for c in (c1, c2) if cls[c] != C]
                w = bs[0] if len(bs) == 1 else lca(bs[0], bs[1])
                bimg[u] = w
                n_mixed += 1
                xw[w] += 1                                        # feeds F(t)

                a, sb, pt = self._mixed_dup_region(u, c1, c2, k1, k2, w, bimg, cimg)
                all_cnt += a
                for idx, val in sb:
                    sub_mark[idx] += val
                for idx, val in pt:
                    pt_mark[idx] += val

        # ---- resolve the target-dependent parts, two linear passes each -----
        Omega = S.subtree_sum(omega)
        c_arr = S.subtree_sum(xw)
        F = S.rootpath_sum(c_arr)                 # includes the root; subtract c(root)
        c_root = c_arr[S.root]
        Dsub = S.rootpath_sum(sub_mark)

        dup_cost, loss_cost = self.dup_cost, self.loss_cost
        w_dup = dup_cost + 2 * loss_cost
        base = const - 2 * (n_leaves - 1)

        out = [0.0] * n
        for t in range(n):
            d_mix = all_cnt + Dsub[t] + pt_mark[t]
            depth_part = base + m * (d[t] + 1) + Omega[t] - (F[t] - c_root)
            out[t] = w_dup * (d_pure + d_mix) + loss_cost * depth_part
        return out

    # ----------------------------------------------------------------------
    def _mixed_dup_region(self, u, c1, c2, k1, k2, w, bimg, cimg):
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
    def _clade_leaves(st_flat, h: int) -> List[int]:
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

