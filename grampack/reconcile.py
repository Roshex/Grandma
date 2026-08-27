import pickle
import itertools
import multiprocessing as mp
import array
from tqdm import tqdm
from pathlib import Path
from functools import partial
from typing import List, Dict, Tuple, Union, Sequence, Iterable, Optional

from .config import TaskConfig
from .logger import GranLogger
from .models import SmrtTree, MulTree, GroupData, Map, ReconResult, TaskResult, FlatTree, NameRegistry, decode_optim
from .ops import GeneTreeManager

# NOTE ON IMPORTS: numpy / scipy / matplotlib are imported lazily inside the functions
# that need them. Under the 'spawn' start method (Windows) every worker re-imports this
# module, so keeping heavy third-party imports out of module scope directly reduces
# worker start-up time.

# --------------------------------------------------------------------------
# MODULE SWITCHES
# --------------------------------------------------------------------------

# If a gene-tree species has no counterpart in the MUL-tree we raise instead of
# silently mapping it to node 0 (the root), which used to inflate losses invisibly.
# Should always be True!
STRICT_TARGETS = True

# 'first valley' definition, shared by the KDE and the Hanning fallback so that the
# --cutoff fvall / --nbest -1 semantics do not depend on whether SciPy is installed.
#   True  -> first density minimum to the RIGHT of the leading mode
#   False -> left-most density minimum (legacy KDE behaviour)
FIRST_VALLEY_AFTER_PEAK = False

_INF = float('inf')

def _worker_reconcile_single(
    mul_item: Tuple[int, FlatTree],
    flat_gts: Dict[int, FlatTree],
    dup_cost: int,
    loss_cost: int,
    registry: NameRegistry, 
    pickle_dir: str, 
    run_prefix: str,
    gt_weights: Dict[int, float],
    retmap: bool = False,
    use_gray: bool = False,
    rep_of: Optional[Dict[int, int]] = None
) -> Tuple[int, Union[int, float], Optional[Dict[int, ReconResult]]]:
    
    mul_idx, flat_mul = mul_item
    total_score: Union[int, float] = 0.0
    gt_results = {} if retmap else None
        
    # Case A: Input tree (index 0) 
    if mul_idx == 0:

        for g_num, gt_flat in flat_gts.items():
            res_score, res_maps = Reconciler.reconcile_sl(
                gt_flat, flat_mul, dup_cost, loss_cost,
                registry=registry if retmap else None, retmap=retmap)
            weight = gt_weights.get(g_num, 1.0)
            scaled_score = res_score * weight if weight != 1.0 else res_score
            total_score += scaled_score
            if retmap:
                gt_results[g_num] = ReconResult(scaled_score, res_maps)

    # Case B: a MUL-Tree
    else:

        p_path = Path(pickle_dir) / f"{run_prefix}_{mul_idx}_groups.pickle"
        try:
            with open(p_path, 'rb') as f:
                groups_data = pickle.load(f)
        except OSError as e:
            # A missing/unreadable pickle used to yield a score of 0, which then WON the
            # minimisation. Never degrade silently here.
            raise RuntimeError(f"MUL-tree {mul_idx}: cannot read group data '{p_path}': {e}") from e

        missing = [g for g in flat_gts if g not in groups_data]
        if missing:
            raise RuntimeError(
                f"MUL-tree {mul_idx}: no group data for gene tree(s) {missing[:5]}"
                f"{'...' if len(missing) > 5 else ''}. Scores would not be comparable with "
                f"the input tree, which is always scored over every gene tree.")

        target_map = Reconciler.build_target_map(flat_mul, registry)

        score_cache: Dict[int, Union[int, float]] = {}
        for g_num in flat_gts:
            rep = rep_of.get(g_num, g_num) if not retmap else g_num
            res_score = score_cache.get(rep)
            res_maps = None
            # Only compute the score for the representative gene tree; all identical copies share that score (unless retmap is requested)
            if res_score is None or retmap:
                res_score, res_maps = Reconciler.reconcile_permutation(
                    flat_gts[rep], flat_mul, dup_cost, loss_cost, registry,
                    groups_data[rep], target_map, retmap=retmap, use_gray=use_gray)
                score_cache[rep] = res_score
            weight = gt_weights.get(g_num, 1.0)
            scaled_score = res_score * weight if weight != 1.0 else res_score
            total_score += scaled_score
            if retmap:
                gt_results[g_num] = ReconResult(scaled_score, res_maps)

    return mul_idx, round(total_score, 3), gt_results

class Reconciler:
    def __init__(self, config: TaskConfig, logger: GranLogger, num_processes: int = 1, pickle_action: str = 'archive', to_map: bool = False, optim: int = 0):
        self.tcf = config
        self.logger = logger
        self.n_procs = num_processes
        self.pickle_action = pickle_action
        self.to_map = to_map
        self.dedup_gts, self.use_gray, self.use_sweep = decode_optim(optim)

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
                        n: int) -> Tuple[array.array, List[int]]:
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
                if STRICT_TARGETS:
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

    @staticmethod
    def reconcile_sl(gt: FlatTree, st: FlatTree,
                       dup_cost: int, loss_cost: int,
                       registry: NameRegistry = None,
                       precalc_map: Union[Dict[int, int], List[int], array.array, None] = None,
                       retmap: bool = False) -> Tuple[int, Optional[List[Map]]]:
        """O(1)-LCA integer-array reconciliation of a gene tree to a singly-labeled or a disambiguated-labeled species tree.
        Returns (score, [map]) if retmap else (score, None)."""
        n = gt.num_nodes

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
                    if STRICT_TARGETS:
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

        score = Reconciler._scan(gt, st, dup_cost, loss_cost, gt.postorder, lca_maps)

        if retmap:
            return score, [Reconciler._build_map(gt, st, lca_maps, registry)]
        return score, None

    # --------------------------------------------------------------------------
    # PERMUTATION LOGIC
    # --------------------------------------------------------------------------

    @staticmethod
    def _prepare_groups(gt_flat: FlatTree, target_map: Dict[int, List[int]],
                        ambig_groups: List[List[int]], fixed_groups: List[Tuple[List[int], int]],
                        lca_maps: array.array) -> Tuple[List[List[int]], List[List[Tuple[int, ...]]]]:
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
                if STRICT_TARGETS:
                    raise RuntimeError(
                        f"Gene-tree leaf {nid} maps to species id {names[nid]}, which is "
                        f"absent from the MUL-tree.")
                return [0]
            return targets

        for grp in ambig_groups:
            leaf_targets = [_targets(nid) for nid in grp]
            n_choices = len(leaf_targets[0])
            assigns: List[Tuple[int, ...]] = []
            seen = set()
            for k in range(n_choices):
                combo = tuple(tl[k] if k < len(tl) else tl[0] for tl in leaf_targets)
                if combo in seen:
                    continue
                seen.add(combo)
                assigns.append(combo)
            group_leaves.append(grp)
            group_assign.append(assigns)
            for nid, node in zip(grp, assigns[0]):
                lca_maps[nid] = node

        for grp, t_idx in fixed_groups:
            for nid in grp:
                tl = _targets(nid)
                lca_maps[nid] = tl[t_idx] if 0 <= t_idx < len(tl) else tl[0]

        return group_leaves, group_assign

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def reconcile_permutation(gt_flat: FlatTree, mul_flat: FlatTree, dup_cost: int, loss_cost: int,
                              registry: NameRegistry, group_data: GroupData,
                              target_map: Dict[int, List[int]],
                              retmap: bool = False, use_gray: bool = True
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
        ambig_groups, fixed_groups = Reconciler.translate_groups_to_ids(gt_flat, group_data)

        n = gt_flat.num_nodes
        lca_maps, _leaves = Reconciler._init_leaf_maps(gt_flat, target_map, n)
        group_leaves, group_assign = Reconciler._prepare_groups(
            gt_flat, target_map, ambig_groups, fixed_groups, lca_maps)

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
                score = Reconciler._scan(gt_flat, mul_flat, dup_cost, loss_cost,
                                         gt_flat.postorder, lca_maps)
                _record(score, lca_maps[:] if retmap else None)

        # ---------------- incremental path -----------------------------------
        else:
            contrib = array.array('i', [0] * n)
            total = Reconciler._scan(gt_flat, mul_flat, dup_cost, loss_cost,
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
                anc_lists, chain_starts = Reconciler._group_ancestors(gt_flat, group_leaves)
                cs, cf = gt_flat.children_start, gt_flat.children_flat
                fv, dep, eul, rmq = (mul_flat.first_visit, mul_flat.depths,
                                     mul_flat.euler_tour, mul_flat.rmq_table)
                root_is_leaf = cs[root_id] == cs[root_id + 1]

                for gi, choice in Reconciler._mixed_radix_gray(radices):
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
            return best_score, [Reconciler._build_map(gt_flat, mul_flat, snap, registry)
                                for snap in best_maps]
        return best_score, None

    # --------------------------------------------------------------------------
    # DRIVER
    # --------------------------------------------------------------------------

    def _warn_if_multi_labelled(self, st_flat: FlatTree, registry: NameRegistry) -> None:
        """Index 0 is scored without allele-map enumeration (identity assignment).
        That is correct for a relabelled multi-MT, whose gene trees are locked to a
        sub-genome, but NOT for a raw multi-labelled input whose gene trees are not."""
        seen = set()
        cs = st_flat.children_start
        for i in range(st_flat.num_nodes):
            if cs[i] != cs[i + 1]:
                continue
            base = registry.get_name(st_flat.node_to_name_id[i]).replace("*", "")
            if base in seen:
                self.logger.log(
                    "Input tree (index 0) is multi-labelled: it is scored with the identity "
                    "leaf assignment, which is only comparable to the MUL-tree scores if the "
                    "gene trees are already labelled per sub-genome.", 'w')
                return
            seen.add(base)

    def _dispatch(self, tasks: List[Tuple[int, FlatTree]], worker_func: callable,
                  desc: str) -> Tuple[Dict[int, Union[int, float]], Dict[int, Dict[int, ReconResult]]]:
        """Run _worker_reconcile_single over `tasks`, in parallel when worthwhile."""
        retmap = worker_func.keywords['retmap']

        all_scores: Dict[int, Union[int, float]] = {}
        detailed_res: Dict[int, Dict[int, ReconResult]] = {} if retmap else None

        # Disable multiprocessing to prevent massive IPC overhead on few tasks
        procs = self.n_procs
        if len(tasks) <= max(2, self.n_procs // 2):
            procs = 1

        if procs > 1:
            with mp.Pool(processes=procs) as pool:
                flat_tasks = [(k, mt.mt.flat_tree) for k, mt in tasks]
                iterator = pool.imap_unordered(worker_func, flat_tasks)
                for idx, score, gt_res in tqdm(iterator, total=len(tasks), desc=desc, unit="st", disable=self.logger.disable_tqdm, ncols=177):
                    all_scores[idx] = score
                    if retmap:
                        detailed_res[idx] = gt_res
        else:
            for k, v in tqdm(tasks, total=len(tasks), desc=desc, unit="st", disable=self.logger.disable_tqdm, ncols=177):
                item = (k, v.mt.flat_tree)
                idx, score, gt_res = worker_func(item)
                all_scores[idx] = score
                if retmap:
                    detailed_res[idx] = gt_res

        return all_scores, detailed_res

    def _get_worker_func(self, gene_trees: Dict[int, SmrtTree], registry: NameRegistry, retmap: bool = False) -> callable:
        flat_gts = {k: v.flat_tree for k, v in gene_trees.items()}
        dup_cost, loss_cost = self.tcf.weights
        gt_weights = {idx: (gt.Q if self.tcf.quota_gts == 'harmonic' else 1.0) for idx, gt in gene_trees.items()}

        rep_of: Dict[int, int] = {}
        if self.dedup_gts and not retmap:
            first_seen: Dict[bytes, int] = {}
            for idx, flat in flat_gts.items():
                sig = flat.signature
                rep = first_seen.setdefault(sig, idx)
                if rep != idx:
                    rep_of[idx] = rep
            if rep_of:
                self.logger.log(
                    f"Using Gene-tree de-duplication: {len(flat_gts)} trees -> "
                    f"{len(flat_gts) - len(rep_of)} distinct labelled-topologies.", 'i')

        return partial(
            _worker_reconcile_single, 
            flat_gts=flat_gts,
            dup_cost=dup_cost,
            loss_cost=loss_cost,
            registry=registry, 
            pickle_dir=str(self.tcf.pickle_dir), 
            run_prefix=self.tcf.run_prefix,
            gt_weights=gt_weights,
            retmap=retmap, 
            use_gray=self.use_gray,
            rep_of=rep_of
        )

    def recon_all(self, mul_trees: Dict[int, MulTree], gene_trees: Dict[int, SmrtTree],
                  registry: NameRegistry, retmap: bool = False) -> Tuple[List[Tuple[int, int]], Dict[int, Dict[int, ReconResult]]]:
        
        step = "Reconciliation"
        self.logger.report_step(step, "In progress...")
        
        # Flatten everything. Anything still valid is reused: the caches are dropped by
        # SmrtTree._invalidate_caches on every structural or naming change, so a gene tree
        # relabelled between iterations is rebuilt automatically.
        # Gene trees are only traversed -> no Euler tour / sparse table.
        # MUL-trees are the reconciliation TARGET -> they need O(1) LCA.
        for gt in gene_trees.values():   gt.make_flat(registry)
        for mdata in mul_trees.values(): mdata.mt.make_lca(registry)

        if 0 in mul_trees:
            self._warn_if_multi_labelled(mul_trees[0].mt.flat_tree, registry)

        tasks = list(mul_trees.items())
        worker_func = self._get_worker_func(gene_trees, registry, retmap)
        all_scores, detailed_res = self._dispatch(tasks, worker_func, "# Scoring   ")
        
        self.logger.report_step(step, "Success", full_update=True)
        return sorted(all_scores.items(), key=lambda kv: (kv[1], kv[0])), detailed_res # Tie-break on the index
    
    def recon_lowest_maps(self, target_idxs: List[int], mul_trees: Dict[int, MulTree], gene_trees: Dict[int, SmrtTree],
                        registry: NameRegistry) -> Dict[int, Dict[int, ReconResult]]:
        
        step = "Getting maps for relevant trees"
        self.logger.report_step(step, "In progress...")

        targets = [(i, mul_trees[i]) for i in target_idxs]
        worker_func = self._get_worker_func(gene_trees, registry, retmap=True)

        _scores, detailed_res = self._dispatch(targets, worker_func, "# Mapping   ")
            
        self.logger.report_step(step, f"Success: got {len(target_idxs)}/{len(mul_trees)} maps", full_update=True)
        return detailed_res
    
    @staticmethod
    def _distribute_scores(sorted_scores: List[Tuple[int, int]]) -> Tuple[List[int], int, Tuple[float, float, float], str]:
        """
        Finds frequency valleys in the score distribution using density smoothing.
        Returns: (scores, input_score, (first_valley, left_valley, right_valley), method)
        """
        scores = [score for _, score in sorted_scores]
        input_score = next((score for idx, score in sorted_scores if idx == 0), None)

        if not scores:
            return scores, (_INF if input_score is None else input_score), (_INF, -_INF, _INF), ''
        if len(scores) < 3:
            mx = float(max(scores))
            return scores, (_INF if input_score is None else input_score), (mx, -_INF, _INF), ''
            
        import numpy as np
        valleys: List[float] = []
        peak_pos = -_INF
        method = ''
        
        try:
            from scipy.stats import gaussian_kde
            from scipy.signal import find_peaks
            
            # Fit a continuous probability density function (KDE)
            kde = gaussian_kde(scores)
            
            # Evaluate across a smooth grid of 100 points
            x_grid = np.linspace(min(scores), max(scores), 100)
            pdf = kde(x_grid)
            
            # Find valleys by identifying the peaks of the inverted PDF
            valley_idxs, _ = find_peaks(-pdf)
            valleys = [float(x_grid[idx]) for idx in valley_idxs]
            peak_pos = float(x_grid[int(np.argmax(pdf))])

            method = 'KDE smoothed'
                
        except Exception:
            # Catches ImportError AND numpy.linalg.LinAlgError, which gaussian_kde raises
            # on a singular sample (every MUL-tree scoring the same) - previously fatal.
            counts, bin_edges = np.histogram(scores, bins='auto')
            
            window_len = max(3, len(counts) // 5)
            if window_len % 2 == 0: window_len += 1 # Ensure odd length
            
            w = np.hanning(window_len)
            smoothed = np.convolve(counts, w / w.sum(), mode='same')
            
            peak_pos = float(bin_edges[int(np.argmax(smoothed)) + 1])
            
            # Find all valid valleys
            for i in range(1, len(smoothed) - 1):
                if smoothed[i] < smoothed[i - 1] and smoothed[i] <= smoothed[i + 1]:
                    valleys.append(float(bin_edges[i + 1]))

            method = 'Hanning smoothed'

        # Single definition of 'first valley' for both back-ends.            
        first_valley = None
        if valleys:
            if FIRST_VALLEY_AFTER_PEAK:
                first_valley = next((v for v in valleys if v > peak_pos), None)
            if first_valley is None:
                first_valley = valleys[0]
                        
        if first_valley is None: first_valley = float(max(scores))

        if input_score is None: input_score = _INF
            
        # Find the closest valleys to the left and right of the input score
        left_valleys = [v for v in valleys if v < input_score]
        right_valleys = [v for v in valleys if v > input_score]
        
        # If no valley exists on a given side, fallback to infinity bounds
        left_valley = max(left_valleys) if left_valleys else -_INF
        right_valley = min(right_valleys) if right_valleys else _INF
        
        return scores, input_score, (first_valley, left_valley, right_valley), method

    def _check_if_passed(self, mt_score: float, in_score: float, valleys: Tuple[float, float, float], prev_score: Optional[float] = None) -> bool:
        ref_type, diff_func, offset = self.tcf.cutoff

        if ref_type == 'none': return True

        if ref_type == 'input':
            comp_score = prev_score if prev_score is not None else in_score
        elif ref_type == 'fvall': comp_score = valleys[0]
        elif ref_type == 'lvall': comp_score = valleys[1]
        elif ref_type == 'rvall': comp_score = valleys[2]
        else:
            self.logger.log(f"Unknown cutoff reference type: {ref_type}.", 'e')
            
        if diff_func == 'rel': offset *= comp_score
        return offset < (comp_score - mt_score)

    def select_mts(self, local_n_best: int, sorted_scores: List[Tuple[int, int]], input_idx: int, input_score: float, valleys: Tuple[float, float, float]) -> Tuple[List[int], Dict[int, bool]]:

        step = "Selecting best MUL-trees"
        self.logger.report_step(step, "In progress...")

        target_idxs = []
        passed_events = {}
        in_mode = self.tcf.mode

        hard_break = _INF
        if local_n_best == 0:    hard_break = input_score
        elif local_n_best == -1: hard_break = valleys[0]
        elif local_n_best == -2: hard_break = valleys[1]
        elif local_n_best == -3: hard_break = valleys[2]

        if in_mode == "st-only":
            if input_idx is None:
                self.logger.log("Mode is 'st-only' but no ST (index 0) found among reconciled trees.", 'e')
            target_idxs.append(input_idx)
            passed_events[input_idx] = True
                
        else:
            if in_mode != 'no-st':
                if input_idx is None:
                    self.logger.log("Input tree (index 0) not found among reconciled trees despite mode setting (not 'no-st').", 'e')
                target_idxs.append(input_idx)
                passed_events[input_idx] = True

            count = 0
            for idx, score in sorted_scores:
                if idx == 0: continue # Skip input tree in ranking (already handled above)

                if score > hard_break: break # Enforce hard break for n_best types based on input/valleys
                if local_n_best > 0 and count >= local_n_best: break # Enforce n_best limit if it's a positive integer
                
                target_idxs.append(idx)
                passed_events[idx] = self._check_if_passed(score, input_score, valleys, self.tcf.prev_score)
                count += 1
                    
            # Ensure at least 1 non-input MT is present in output even if passed=False
            if not target_idxs or target_idxs == [0]:
                best_tuple = next(((idx, score) for idx, score in sorted_scores if idx != 0), None)
                if best_tuple is not None:
                    mt_idx, mt_score = best_tuple
                    target_idxs.append(mt_idx)
                    passed_events[mt_idx] = self._check_if_passed(mt_score, input_score, valleys, self.tcf.prev_score)

        if not target_idxs or (len(target_idxs) == 1 and target_idxs[0] == 0):
            self.logger.report_step(step, "Success: no new events to assess")
        else:
            total_passed = sum(passed_events.values()) - (1 if input_idx in passed_events else 0)
            if total_passed:
                self.logger.report_step(step, f"Success: {total_passed} events accepted")
            else:
                self.logger.report_step(step, "Success: no new events passed cutoff")

        return target_idxs, passed_events

    def run(self, mul_trees: dict, gene_trees: dict, registry: NameRegistry) -> TaskResult:

        if registry is None: registry = NameRegistry()

        if self.use_gray:
            self.logger.log("Using Gray-code enumeration for the reconciliation step.", 'i')

        n_best = self.tcf.n_best
        in_mode = self.tcf.mode
        high_demand = False
        num_mts = len(mul_trees)
        input_idx = 0 if 0 in mul_trees else None

        # 0 : everything better than input
        # -1 : everything better than first valley
        # -2 : everything better than valley left of the input
        # -3 : everything better than valley right of the input
        # -4 or less : everything (no cutoff)

        if in_mode == 'no-st' and (n_best in [0, -2, -3]):
            self.logger.log("Mode is 'no-st' but MT selection is set to be relative to input ranking. Adjusting to n_best=-1 to be based on the first valley.", 'w')
            n_best = -1
        if in_mode == 'st-only':
            n_best = 0

        # All maps requested
        if n_best <= -4 or n_best > num_mts: n_best = num_mts

        # Only one recon of the ST, or High Map Demand Threshold: 10%
        if in_mode == 'st-only' or n_best > num_mts * 0.1: high_demand = True

        # Run reconciliation
        try:
            if high_demand:
                if n_best > 0:
                    self.logger.log(f"High map demand detected ({n_best}/{num_mts}). Generating maps directly during scoring.", 'i')
                sorted_scores, detailed_res = self.recon_all(mul_trees, gene_trees, registry, retmap=True)
            else:
                sorted_scores, _ = self.recon_all(mul_trees, gene_trees, registry, retmap=False)
        except Exception:
            self.logger.log("Reconcile failed catastrophically: handling pickle files before exiting...", 'e', kill_on_error=False)
            try: GeneTreeManager(self.tcf, self.logger, self.n_procs, self.pickle_action).handle_pickles()
            # Don't re-raise errors from cleanup to avoid masking main results
            except Exception as e: self.logger.log(f"Failed to clean up pickle files during crash recovery: {e}", 'w')
            # Re-raise original exception for logger handling
            raise

        scores, input_score, valleys, method = self._distribute_scores(sorted_scores)

        # Evaluate Selection
        target_idxs, passed_events = self.select_mts(n_best, sorted_scores, input_idx, input_score, valleys)

        # Retrieve/Trim Maps for targets
        if high_demand:
            detailed_res = {k: detailed_res[k] for k in target_idxs if k in detailed_res}
        else:
            detailed_res = self.recon_lowest_maps(target_idxs, mul_trees, gene_trees, registry)

        # Write outputs
        self.write_detailed(detailed_res, gene_trees)
        self.write_scores_and_counts(sorted_scores, mul_trees, detailed_res)
        self.plot_score_distribution(scores, input_score, valleys, method)

        # Instead of keeping ReconResult, keep Maps[0] (Dict[int, Dict[int, Map]] vs Dict[int, Dict[int, ReconResult]] in StepResult)
        detailed_kept = {mul_idx: {g_idx: res.maps[0] for g_idx, res in detailed_res[mul_idx].items()} for mul_idx in detailed_res}

        return TaskResult(sorted_scores, mul_trees, detailed_kept, gene_trees, passed_events)
        
    # ----------------------------------------------------------------------
    # TARGET SWEEP PIPELINE  (--optim 4 / 5 / 6 / 7)
    # ----------------------------------------------------------------------

    def _sweep_guard(self, st_flat: FlatTree, registry: NameRegistry) -> None:
        """The sweep models exactly TWO copies of the donor clade grafted at ONE target,
        which requires a singly labelled species tree. Refuse anything else loudly rather
        than returning quietly wrong scores."""
        seen = set()
        cs = st_flat.children_start
        for i in range(st_flat.num_nodes):
            if cs[i] != cs[i + 1]:
                continue
            sp = registry.get_name(st_flat.node_to_name_id[i]).replace("*", "")
            if sp in seen:
                self.logger.log(
                    "The target sweep requires a singly labelled species tree (it models "
                    "one graft producing two copies). This input is multi-labelled - run "
                    "without --optim 4 for this task.", 'e')
                raise RuntimeError("sweep: multi-labelled species tree")
            seen.add(sp)

    def run_sweep(self, valid_pairings: Dict[str, List[List['Tree']]],
                           h1_resolved: List[str], st_wrapper: SmrtTree,
                           gene_trees: Dict[int, SmrtTree], registry: NameRegistry,
                           gene_mgr) -> Optional[TaskResult]:
        """
        Scores every candidate placement without building its MUL-tree.

        Single-target candidates are scored by TargetSweep: one O(n_G + N) pass per
        (donor clade, gene tree, allele assignment) yields the score for EVERY target.
        Multi-target candidates (nesting='model' with a duplicated recipient) are outside
        the sweep's model, so they are built and reconciled by the normal engine; the two
        score sets share one index space.

        MUL-trees are then materialised only for the selected candidates.
        """
        from .sweep import SpeciesIndex, TargetSweep, default_units

        if registry is None:
            registry = NameRegistry()

        step = "Indexing MUL-tree components"
        self.logger.report_step(step, "In progress...")

        # ---------------- 0. species tree ---------------------------------
        st_wrapper.make_lca(registry)            # sweep needs depths; reconcile_sl needs RMQ
        st_flat = st_wrapper.flat_tree
        self._sweep_guard(st_flat, registry)

        sidx = SpeciesIndex(st_flat, registry=registry)
        dup_cost, loss_cost = self.tcf.weights
        sw = TargetSweep(sidx, dup_cost, loss_cost)

        n2id = st_flat.name_id_to_node_id           # keyed by FULL name id, not pure
        def st_id(name: str) -> Optional[int]:
            return n2id.get(registry.get_id(name))

        # ---------------- 1. candidate index space -------------------------
        # One stable index per candidate, shared by the sweep, the eager engine, the
        # group pickles and the output files.
        meta: Dict[int, Tuple[str, List]] = {}
        single: List[int] = []
        multi: List[int] = []
        idx = 1
        for h1_name in h1_resolved:
            for matches in valid_pairings.get(h1_name, []):
                meta[idx] = (h1_name, matches)
                (single if len(matches) == 1 else multi).append(idx)
                idx += 1
        if not meta:
            self.logger.log("No candidate placements to evaluate.", 'w')
            return None

        self.logger.report_step(step, f"Success: {len(single)} single-target, {len(multi)} multi-target candidates")

        # ---------------- 2. gene trees, cap, de-duplication ---------------
        step = "Filtering gene trees over group cap"
        self.logger.report_step(step, "In progress...")

        for gt in gene_trees.values():
            gt.make_flat(registry)          # Traversed only: no Euler tour / or RMQ

        # The engine caps on units remaining AFTER sister-pinning, which is
        # target-dependent; pin_states gives that same count for every target, so the
        # filter now removes exactly the gene trees the standard path removes.
        cap = self.tcf.group_cap
        clade_ids: Dict[str, set] = {}
        valid_t: Dict[str, List[int]] = {}
        for h1_name in h1_resolved:
            h = st_id(h1_name)
            if h is None:
                continue
            clade_ids[h1_name] = {st_flat.node_to_name_id[v]
                                  for v in TargetSweep._clade_leaves(st_flat, h)}
            ts = [st_id(m[0].name) for m in valid_pairings.get(h1_name, [])]
            valid_t[h1_name] = [t for t in ts if t is not None]

        over_cap: Dict[int, int] = {}
        for g_idx, gt in gene_trees.items():
            for h1_name, sp_ids in clade_ids.items():
                units = default_units(gt.flat_tree, sidx, sp_ids)
                if not units:
                    continue
                _st, free = sw.pin_states(gt.flat_tree, st_id(h1_name), units, st_flat,
                                          valid_t[h1_name])
                if any(free[t] > cap for t in valid_t[h1_name]):
                    over_cap[g_idx] = over_cap.get(g_idx, 0) + 1            
        for g_idx in sorted(over_cap):
            self.logger.log(f"Gene tree on line {g_idx+1} is over the group cap for "
                            f"{over_cap[g_idx]} donor clades and will be filtered.", 'w')
            del gene_trees[g_idx]
        if not gene_trees:
            self.logger.log("Every gene tree was filtered by the group cap.", 'e')

        gt_weights = {i: (gt.Q if self.tcf.quota_gts == 'harmonic' else 1.0)
                      for i, gt in gene_trees.items()}

        unique_gts: Dict[int, FlatTree] = {}
        weight_of_rep: Dict[int, float] = {}
        if self.dedup_gts:
            first_seen: Dict[bytes, int] = {}
            for g_idx, gt in gene_trees.items():
                rep = first_seen.setdefault(gt.flat_tree.signature, g_idx)
                weight_of_rep[rep] = weight_of_rep.get(rep, 0.0) + gt_weights[g_idx]
                if rep == g_idx:
                    unique_gts[g_idx] = gt.flat_tree
            self.logger.log(f"Sweep de-duplication: {len(gene_trees)} gene trees -> "
                            f"{len(unique_gts)} distinct labelled topologies.", 'i')
        else:
            unique_gts = {i: gt.flat_tree for i, gt in gene_trees.items()}
            weight_of_rep = dict(gt_weights)

        self.logger.report_step(step, f"Success: {len(over_cap)} gts over cap")#, full_update=True)

        # ---------------- 3. eager set: the input tree + multi-target MTs ---
        mul_trees: Dict[int, MulTree] = {0: MulTree(mt=st_wrapper)}
        for m_idx in multi:
            h1_name, matches = meta[m_idx]
            h1_st_node = st_wrapper.get_node(h1_name)
            mt_wrapper, h1_obj, hx_objs = st_wrapper.to_multi_mul_tree(
                h1_name, sorted(n.name for n in matches))
            if mt_wrapper:
                mul_trees[m_idx] = MulTree(mt_wrapper, h1_st_node.get_leaf_names(),
                                           h1_obj, hx_nodes=hx_objs)

        if len(mul_trees) > 1:
            if not gene_mgr.cull(mul_trees, gene_trees, registry):
                return None                       # check-nums mode
            eager_scores, _ = self.recon_all(mul_trees, gene_trees, registry, retmap=False)
        else:
            # Only the input tree: score it directly, no pool, no pickles.
            st_score = 0.0
            for g_idx, gt_flat in unique_gts.items():
                s, _ = self.reconcile_sl(gt_flat, st_flat, dup_cost, loss_cost,
                                         registry=registry)
                st_score += s * weight_of_rep.get(g_idx, 1.0)
            eager_scores = [(0, round(st_score, 3))]

        all_scores: Dict[int, float] = dict(eager_scores)

        # ---------------- 4. the sweep -------------------------------------
        step = "Sweeping candidate placements"
        self.logger.report_step(step, "In progress...")

        by_h1: Dict[str, List[int]] = {}
        for c_idx in single:
            by_h1.setdefault(meta[c_idx][0], []).append(c_idx)

        for h1_name, c_idxs in tqdm(by_h1.items(), total=len(by_h1), desc="# Sweeping  ",
                                    unit="h1", disable=self.logger.disable_tqdm, ncols=177):
            h = st_id(h1_name)
            if h is None:
                self.logger.log(f"Donor clade '{h1_name}' not found in the flattened "
                                f"species tree; its candidates are skipped.", 'w')
                continue
            sp_ids = clade_ids[h1_name]

            totals = [0.0] * sidx.n
            for g_idx, gt_flat in unique_gts.items():
                units = default_units(gt_flat, sidx, sp_ids)
                vec = sw.score_all_targets(gt_flat, h, units=units, st_flat=st_flat,
                                           pin=True, valid_targets=valid_t[h1_name])
                w = weight_of_rep.get(g_idx, 1.0)
                for t in range(sidx.n):
                    v = vec[t]
                    if v != float('inf'):
                        totals[t] += v * w

            for c_idx in c_idxs:
                t_id = st_id(meta[c_idx][1][0].name)
                if t_id is None:
                    continue
                all_scores[c_idx] = round(totals[t_id], 3)

        sorted_scores = sorted(all_scores.items(), key=lambda kv: (kv[1], kv[0]))
        self.logger.report_step(step, f"Success: scored {len(sorted_scores)-1} candidates")

        # ---------------- 5. selection (same policy as run()) ---------------
        n_best = self.tcf.n_best
        in_mode = self.tcf.mode
        num_mts = len(sorted_scores)
        input_idx = 0 if 0 in all_scores else None
        if in_mode == 'no-st' and n_best in (0, -2, -3):
            self.logger.log("Mode is 'no-st' but MT selection is relative to the input "
                            "ranking. Adjusting to n_best=-1.", 'w')
            n_best = -1
        if in_mode == 'st-only':
            n_best = 0
        if n_best <= -4 or n_best > num_mts:
            n_best = num_mts

        scores, input_score, valleys, method = self._distribute_scores(sorted_scores)
        target_idxs, passed_events = self.select_mts(n_best, sorted_scores, input_idx,
                                                    input_score, valleys)

        # ---------------- 6. materialise only the winners --------------------
        build_step = "Building selected MUL-trees"
        self.logger.report_step(build_step, "In progress...")
        newly_built = 0
        for m_idx in target_idxs:
            if m_idx in mul_trees or m_idx not in meta:
                continue
            h1_name, matches = meta[m_idx]
            h1_st_node = st_wrapper.get_node(h1_name)
            mt_wrapper, h1_obj, hx_objs = st_wrapper.to_multi_mul_tree(
                h1_name, sorted(n.name for n in matches))
            if mt_wrapper:
                mul_trees[m_idx] = MulTree(mt_wrapper, h1_st_node.get_leaf_names(),
                                           h1_obj, hx_nodes=hx_objs)
                newly_built += 1
        self.logger.report_step(build_step, f"Success: built {newly_built} MUL-trees")

        target_idxs = [i for i in target_idxs if i in mul_trees]

        # Groups exist only for the eager MTs; collapse for the newly built winners.
        # collapse_groups skips index 0 and reuses any pickle that already exists, so
        # this neither rebuilds the eager ones nor filters gene trees a second time.
        if newly_built:
            cull_step = "Collapsing groups for selected candidates"
            self.logger.report_step(cull_step, "In progress...")
            gene_mgr.collapse_groups({i: mul_trees[i] for i in target_idxs if i != 0},
                                     gene_trees, registry)
            self.logger.report_step(cull_step, "Success")

        # Flatten: recon_all runs this earlier, which is false in the sweep pipeline,
        # where the selected MUL-trees are built only after scoring. make_flat/make_lca
        # reuse anything still valid, so this is a no-op on the standard path.
        targets = []
        for i in target_idxs:
            mdata = mul_trees.get(i)
            if mdata is None:
                self.logger.log(f"Selected MUL-tree {i} is not available; skipping its maps.", 'w')
                continue
            mdata.mt.make_lca(registry)          # reconciliation TARGET -> needs O(1) LCA
            targets.append((i, mdata))
        if not targets:
            self.logger.log("INFO: no targets to map", 'i')
            return {}

        detailed_res = self.recon_lowest_maps(target_idxs, mul_trees, gene_trees, registry)

        # ---------------- 7. outputs -----------------------------------------
        self.write_detailed(detailed_res, gene_trees)
        self.write_scores_and_counts(sorted_scores, mul_trees, detailed_res, meta=meta)
        self.plot_score_distribution(scores, input_score, valleys, method)

        detailed_kept = {m: {g: r.maps[0] for g, r in detailed_res[m].items() if r.maps}
                         for m in detailed_res}
        return TaskResult(sorted_scores, mul_trees, detailed_kept, gene_trees, passed_events)

    # --------------------------------------------------------------------------
    # WRITER LOGIC
    # --------------------------------------------------------------------------

    @staticmethod
    def score_to_str(score) -> str:
        if isinstance(score, float):
            return f"{score:.3f}"
        return str(score)
    
    def plot_score_distribution(self, scores: List[int], input_score: int, valleys: Tuple[float, float, float], method: str = ''):

        import matplotlib
        matplotlib.use('Agg') # headless: safe on HPC nodes and in workers
        import matplotlib.pyplot as plt

        step = "Plotting score distribution"
        self.logger.report_step(step, "In progress...")

        plt.figure(figsize=(10, 6))
        plt.hist(scores, bins=30, color='lightgray', edgecolor='black')
        first_v, left_v, right_v = valleys

        # Plot vertical lines for input tree score and the valleys if present
        if input_score is not None:
            plt.axvline(x=input_score, color='red', linestyle='--', label='Input Tree Score')
            plt.axvline(x=left_v, color='green', linestyle='--')
            plt.axvline(x=right_v, color='green', linestyle='--')
        if method:
            plt.axvline(x=first_v, color='green', linestyle='--', label=f'Valleys ({method})')

        plt.legend()
        plt.title('Score Distribution of MUL-trees')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        
        p = Path(self.tcf.output_dir) / f"{self.tcf.run_prefix}-score_distribution.png"
        plt.savefig(p)
        plt.close()
        
        self.logger.report_step(step, "Success")

    def write_detailed(self, detailed_res: dict, gene_trees: dict):

        def map_formatter(name: str, maps: dict, dups: dict) -> str:
            """
            Dynamically injects [Map-Dup] labels into the Newick string 
            using a formatter, completely avoiding slow tree copying/mutation.
            """
            if name in maps:
                cur_map = maps[name][0]
                # Append '+' if it mapped to the H1 (Base) copy
                if "*" not in cur_map: cur_map += "+"
                # Format: Node<|Map-Dups|> -> Node[Map-Dups]
                return f"{name}<|{cur_map}-{dups.get(name, 0)}|>"
            return name
        
        step = "Writing detailed output file"
        self.logger.report_step(step, "In progress...")
        
        p = Path(self.tcf.output_dir) / f"{self.tcf.run_prefix}-detailed.txt"
        with open(p, 'w') as f:
            
            header = "mul.tree\tgene.tree\tdups\tlosses\ttotal.score"
            header += "\tmaps\n" if self.to_map != 0 else "\n"
            f.write(header)

            for mul_idx, res_dict in detailed_res.items():

                f.write(f"# MUL-tree {mul_idx}\n")
                for gene_idx, res in res_dict.items():

                    # Handle multiple maps if present
                    if (maps_len := len(res.maps)) > 1:
                        f.write(f"# GT-{gene_idx+1} to MT-{mul_idx}\t{maps_len} maps found!\n")
                        
                    gt_obj = gene_trees[gene_idx]
                    score_str = self.score_to_str(res.score)
                    for map_obj in res.maps:
                        if self.to_map:
                            map_str = gt_obj.to_str(
                                internals=True,
                                name_formatter=map_formatter,
                                maps=map_obj.cor,
                                dups=map_obj.dups
                            )
                            map_str = '\t' + map_str.replace("<|", "[").replace("|>", "]") # Avoid Newick issues with angle brackets
                        else:
                            map_str = ''
                        f.write(f"{mul_idx}\t{gene_idx+1}\t{map_obj.n_dups}\t{map_obj.n_losses}\t{score_str}{map_str}\n")
                                 
        self.logger.report_step(step, f"Success: recorded {len(detailed_res)} trees{' with maps' if self.to_map else ''}")

    def write_scores_and_counts(self, sorted_scores: list, mul_trees: dict, detailed_res: dict,
                                meta: dict = None):
        step = "Writing main output files"
        self.logger.report_step(step, "In progress...")

        p = Path(self.tcf.output_dir) / f"{self.tcf.run_prefix}-scores.txt"
        with open(p, 'w') as f:
            f.write("mul.tree\th1.node\thx.nodes\tscore\tlabeled.tree\n")
            for idx, score in sorted_scores:
                mul_data = mul_trees.get(idx)
                if mul_data is not None:
                    tree_str = mul_data.to_marked_str()
                    h1_name = mul_data.h1_node.name if mul_data.h1_node else "NA"
                    hx_names = ",".join(n.name for n in mul_data.hx_sisters) \
                        if mul_data.hx_sisters else "NA"
                elif meta and idx in meta:
                    # Swept candidate that was never built: report it from its metadata.
                    h1_name, matches = meta[idx]
                    hx_names = ",".join(n.name for n in matches)
                    tree_str = "NOT_BUILT"
                else:
                    raise RuntimeError(f"Index {idx} not found in mul_trees or meta; cannot write scores.")
                f.write(f"{idx}\t{h1_name}\t{hx_names}\t{self.score_to_str(score)}\t{tree_str}\n")

        self.write_dup_loss(detailed_res, mul_trees)

        self.logger.report_step(step, "Success")

    def write_dup_loss(self, detailed_res: dict, mul_trees: dict):
        p_dup = Path(self.tcf.output_dir) / f"{self.tcf.run_prefix}-dup-counts.txt"
        with open(p_dup, 'w') as f:
            f.write("mul.tree\tnode\tdups\tlosses\n")
            for mul_idx, res_dict in detailed_res.items():
                mul_data = mul_trees[mul_idx]
                hybrid_clade = mul_data.h_clade
                ordered_nodes = mul_data.mt.node_order

                # Pre-fill dictionary with 0s to guarantee NO missing rows
                main_dups = {node: 0 for node in ordered_nodes}
                main_losses = main_dups.copy()

                # Accumulate counts
                for _g_idx, res in res_dict.items():
                    first_map = res.maps[0]
                    cor_maps = first_map.cor
                    for gt_node, count in first_map.dups.items():
                        if count > 0:
                            map_node = cor_maps[gt_node][0]
                            main_dups[map_node] += count
                    for gt_node, count in first_map.losses.items():
                        if count > 0:
                            map_node = cor_maps[gt_node][0]
                            main_losses[map_node] += count

                # Write ordered output
                for node in ordered_nodes:
                    out_node = node + "+" if node in hybrid_clade else node
                    f.write(f"{mul_idx}\t{out_node}\t{main_dups[node]}\t{main_losses[node]}\n")