import pickle
import multiprocessing as mp
from tqdm import tqdm
from pathlib import Path
from functools import partial
from typing import List, Dict, Tuple, Union, Optional

from .config import TaskConfig
from .logger import GranLogger
from .models import SmrtTree, MulTree, ReconResult, TaskResult, FlatTree, NameRegistry, decode_optim
from .ops import GeneTreeManager, MulTreeManager, CommonOps

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

# One SpeciesIndex/TargetSweep per worker process: both are derived from st_flat alone,
# so rebuilding them locally is cheaper than shipping them with every task (and keeps
# TargetSweep's per-donor scalar cache process-local).
_SWEEP_ENGINE = {}   # single-slot cache: one species tree per worker per run

def _sweep_engine(st_flat, dup_cost, loss_cost):
    eng = _SWEEP_ENGINE.get('eng')
    if (eng is None or _SWEEP_ENGINE['nodes'] != st_flat.num_nodes
            or eng.dup_cost != dup_cost or eng.loss_cost != loss_cost):
        from .algo import SpeciesIndex, TargetSweep
        eng = TargetSweep(SpeciesIndex(st_flat), dup_cost, loss_cost)
        _SWEEP_ENGINE.update(eng=eng, nodes=st_flat.num_nodes)
    return eng

def _worker_sweep_single(
    h_item: Tuple[str, int],
    st_flat: FlatTree,
    flat_gts: Dict[int, FlatTree],
    weights: Dict[int, float],
    dup_cost: int,
    loss_cost: int,
    valid_t: Dict[str, List[int]],
    use_exact: bool = False,
    use_gray: bool = True,
):
    """Score every target of ONE donor clade, summed over the (de-duplicated) gene trees.
    Returns only the valid-target entries, so the payload back is a few dozen floats."""
    import numpy as np
    h1_name, h = h_item
    sw = _sweep_engine(st_flat, dup_cost, loss_cost)
    targets = valid_t[h1_name]
    vt = np.asarray(targets, dtype=np.int64)
    totals = np.zeros(len(vt))

    for g_idx, gt_flat in flat_gts.items():
        vec = sw.score_all_targets(gt_flat, st_flat, h, valid_targets=targets,
                                   exact=use_exact, rule=True, gray=use_gray)
        sub = vec[vt]
        if not np.isfinite(sub).all():
            raise RuntimeError(f"sweep returned inf for a valid target of '{h1_name}'")
        totals += sub * weights.get(g_idx, 1.0)

    return h1_name, vt, totals

def _worker_pairwise_single(
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
    rep_of: Optional[Dict[int, int]] = None,
    use_exact: bool = False
) -> Tuple[int, Union[int, float], Optional[Dict[int, ReconResult]]]:
    from .algo import PairwiseRecon
    pr = PairwiseRecon(dup_cost, loss_cost, STRICT_TARGETS)
    
    mul_idx, flat_mul = mul_item
    total_score: Union[int, float] = 0.0
    gt_results = {} if retmap else None
        
    # Case A: Input tree (index 0) 
    if mul_idx == 0:

        for g_num, gt_flat in flat_gts.items():
            res_score, res_maps = pr.reconcile_sl(
                gt_flat, flat_mul, registry=registry if retmap else None, retmap=retmap)
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

        target_map = pr.build_target_map(flat_mul, registry)

        use_dedup = bool(rep_of)
        score_cache: Dict[int, Union[int, float]] = {}
        for g_num, gt_flat in flat_gts.items():
            if use_dedup:
                rep = rep_of.get(g_num, g_num)
                res_score = score_cache.get(rep)
                if res_score is None:
                    res_score, res_maps = pr.reconcile_permutation(
                        flat_gts[rep], flat_mul, registry,
                        groups_data[rep], target_map, retmap=False, use_gray=use_gray, use_exact=use_exact)
                    score_cache[rep] = res_score
                else:
                    res_maps = None
            else:
                res_score, res_maps = pr.reconcile_permutation(
                    gt_flat, flat_mul, registry,
                    groups_data[g_num], target_map, retmap=retmap, use_gray=use_gray, use_exact=use_exact)

            weight = gt_weights.get(g_num, 1.0)
            scaled_score = res_score * weight if weight != 1.0 else res_score
            total_score += scaled_score
            if retmap:
                gt_results[g_num] = ReconResult(scaled_score, res_maps)

    return mul_idx, round(total_score, 3), gt_results

class Reconciler:
    def __init__(self, config: TaskConfig, logger: GranLogger, num_processes: int = 1,
                 pickle_action: str = 'archive', to_map: bool = False, optim: int = 0,
                 rep_of: Optional[Dict[int, int]] = None):
        self.tcf = config
        self.logger = logger
        self.n_procs = num_processes
        self.pickle_action = pickle_action
        self.to_map = to_map
        self.dedup_gts, self.use_gray, self.use_sweep, self.use_exact = decode_optim(optim)
        self.dedup_threshold = getattr(config, 'disable_dedup_below', 0.05)
        self.rep_of = rep_of                    # None -> decide here (sweep path, st-only)

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

    def _dispatch_sweep(self, donors: List[Tuple[str, int]], worker_func: callable,
                        desc: str = "# Sweeping  ") -> Dict[str, Tuple]:
        """Run _worker_sweep_single over the donor clades, in parallel when worthwhile."""
        out: Dict[str, Tuple] = {}

        procs = self.n_procs
        if len(donors) <= max(2, self.n_procs // 2):
            procs = 1            # IPC would cost more than the work

        bar = dict(total=len(donors), desc=desc, unit="h1",
                   disable=self.logger.disable_tqdm, ncols=177)
        if procs > 1:
            with mp.Pool(processes=procs) as pool:
                for h1_name, vt, totals in tqdm(
                        pool.imap_unordered(worker_func, donors), **bar):
                    out[h1_name] = (vt, totals)
        else:
            for item in tqdm(donors, **bar):
                h1_name, vt, totals = worker_func(item)
                out[h1_name] = (vt, totals)
        return out

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

        '''rep_of: Dict[int, int] = {}
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
                    f"{len(flat_gts) - len(rep_of)} distinct labelled-topologies.", 'i')'''
                
        """if retmap:
            rep_of = {}                         # maps are per gene tree, never shared
        elif self.rep_of is not None:
            rep_of = self.rep_of                # decided once, in cull
        else:
            rep_of, _ = CommonOps.plan_dedup(gene_trees, self.dedup_threshold,
                                   enabled=self.dedup_gts,
                                   latched_off=getattr(self.tcf, 'dedup_latch', False),
                                   logger=self.logger, label="[recon]")
            
        if rep_of:
            self.logger.log(
                f"Using Gene-tree de-duplication: {len(flat_gts)} trees -> "
                f"{len(flat_gts) - len(rep_of)} distinct labelled-topologies.", 'i')"""

        rep_of, unique_gts, rep_weight = self._dedup_views(gene_trees, retmap)

        return partial(
            _worker_pairwise_single, 
            flat_gts=flat_gts,
            dup_cost=dup_cost,
            loss_cost=loss_cost,
            registry=registry, 
            pickle_dir=str(self.tcf.pickle_dir), 
            run_prefix=self.tcf.run_prefix,
            gt_weights=gt_weights,
            retmap=retmap, 
            use_gray=self.use_gray,
            rep_of=rep_of,
            use_exact=self.use_exact
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

    def _resolve_n_best(self, num_mts: int) -> int:
        """Resolve the number of best MUL-trees to select.
         0 : everything better than input
        -1 : everything better than first valley
        -2 : everything better than valley left of the input
        -3 : everything better than valley right of the input
        -4 or less : everything (no cutoff)"""
        n_best, in_mode = self.tcf.n_best, self.tcf.mode

        if in_mode == 'no-st' and n_best in (0, -2, -3):
            self.logger.log("Mode is 'no-st' but MT selection is set to be relative to input "
                            "ranking. Adjusting to n_best=-1 to be based on the first valley.", 'w')
            n_best = -1
        if in_mode == 'st-only':
            n_best = 0
        if n_best <= -4 or n_best > num_mts: # All maps requested
            n_best = num_mts

        return n_best

    def run(self, mul_trees: dict, gene_trees: dict, registry: NameRegistry) -> TaskResult:

        if registry is None: registry = NameRegistry()

        if self.use_gray:
            self.logger.log("Using Gray-code enumeration for the reconciliation step.", 'i')

        num_mts = len(mul_trees)
        n_best = self._resolve_n_best(num_mts)
        input_idx = 0 if 0 in mul_trees else None

        # Only one recon of the ST, or High Map Demand Threshold: 10%
        high_demand = False
        if self.tcf.mode == 'st-only' or n_best > num_mts * 0.1: high_demand = True

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

        return self._finalise(sorted_scores, mul_trees, gene_trees, detailed_res, passed_events, scores, input_score, valleys, method, meta=None)

    def _finalise(self, sorted_scores, mul_trees, gene_trees, detailed_res, passed_events,
                  scores, input_score, valleys, method, meta=None) -> TaskResult:

        # Write outputs
        self.write_detailed(detailed_res, gene_trees)
        self.write_scores_and_counts(sorted_scores, mul_trees, detailed_res, meta=meta)
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

    def _dedup_views(self, gene_trees: Dict[int, SmrtTree], retmap: bool = False, label: str = "[recon]") -> Tuple[Dict[int, int], Dict[int, FlatTree], Dict[int, float]]:
        """
        (rep_of, unique_gts, weight_of_rep) - three views of ONE de-duplication decision.

        rep_of        gene tree -> representative of its class (empty when not worth it)
        unique_gts    representative -> flat tree; what the sweep actually scores
        weight_of_rep representative -> summed weight of its whole class

        The decision itself is taken once, in cull (self.rep_of), so the threshold and
        the serial latch apply identically to the engine and to the sweep. Maps are
        per gene tree and are never shared, hence retmap forces the identity partition.
        """
        gt_weights = {i: (gt.Q if self.tcf.quota_gts == 'harmonic' else 1.0)
                      for i, gt in gene_trees.items()}
        if retmap:
            rep_of = {}
        elif self.rep_of is not None:
            rep_of = {g: r for g, r in self.rep_of.items() if g in gene_trees}
        else:
            rep_of, _ = CommonOps.plan_dedup(gene_trees, self.dedup_threshold,
                                   enabled=self.dedup_gts,
                                   latched_off=getattr(self.tcf, 'dedup_latch', False),
                                   logger=self.logger, label=label)

        unique_gts, weight_of_rep = {}, {}
        for g_idx, gt in gene_trees.items():
            rep = rep_of.get(g_idx, g_idx)
            weight_of_rep[rep] = weight_of_rep.get(rep, 0.0) + gt_weights[g_idx]
            if rep == g_idx:
                unique_gts[g_idx] = gt.flat_tree
        return rep_of, unique_gts, weight_of_rep

    def _sweep_index(self, meta, single, st_flat, n2id, registry) -> Tuple[Dict[str, int], Dict[str, set], Dict[str, List[int]], Dict[int, int]]:

        from .algo import TargetSweep
        def st_id(name: str) -> Optional[int]:
            nid = registry.find_id(name)
            return None if nid is None else n2id.get(nid)

        # The engine caps on units remaining AFTER sister-pinning, which is
        # target-dependent; pin_states gives that same count for every target, so the
        # filter now removes exactly the gene trees the standard path removes.
        clade_ids: Dict[str, set] = {}
        valid_t: Dict[str, List[int]] = {}
        h_id: Dict[str, int] = {}                    # donor clade name -> species-tree node
        for c_idx, (h1_name, matches) in meta.items():
            if h1_name not in h_id: # not seen yet
                h = st_id(h1_name)
                if h is None:
                    self.logger.log(f"Donor clade '{h1_name}' is absent from the flattened "
                                    f"species tree; the pairings and the ST are out of sync.", 'e')
                h_id[h1_name] = h
                valid_t[h1_name] = []
                clade_ids[h1_name] = {st_flat.node_to_name_id[v]
                                    for v in TargetSweep._clade_leaves(st_flat, h)}
            t = st_id(matches[0].name)
            if t is not None:
                # Make sure cap filtering is applied to the same set of targets as
                # the engine would see (here, single-target candidates only)
                if c_idx in single:
                    valid_t[h1_name].append(t)

        # candidate -> its (single) target node, resolved once
        t_id_of: Dict[int, int] = {}
        for c_idx in single:
            t = st_id(meta[c_idx][1][0].name)
            if t is not None:
                t_id_of[c_idx] = t

        return h_id, clade_ids, valid_t, t_id_of

    def run_sweep(self, mt_space: Tuple[Dict[int, list], List[int], List[int]],
                           st_wrapper: SmrtTree, gene_trees: Dict[int, SmrtTree], registry: NameRegistry,
                           gene_mgr: GeneTreeManager, mul_mgr: MulTreeManager) -> Optional[TaskResult]:
        """
        Scores every candidate placement without building its MUL-tree.

        Single-target candidates are scored by TargetSweep: one O(n_G + N) pass per
        (donor clade, gene tree, allele assignment) yields the score for EVERY target.
        Multi-target candidates (nesting='model' with a duplicated recipient) are outside
        the sweep's model, so they are built and reconciled by the normal engine; the two
        score sets share one index space.

        MUL-trees are then materialised only for the selected candidates.
        """
        from .algo import SpeciesIndex, TargetSweep

        if registry is None:
            registry = NameRegistry()

        step = "Indexing MUL-tree components"
        self.logger.report_step(step, "In progress...")
        
        # ---------------- 0. candidate index space -------------------------
        # One stable index per candidate, shared by the sweep, the eager engine, the
        # group pickles and the output files.
        meta, single, multi = mt_space
        if not meta:
            self.logger.log("No candidate placements to evaluate.", 'w')
            return None

        # ---------------- 1. species tree ---------------------------------
        st_wrapper.make_lca(registry)            # sweep needs depths; reconcile_sl needs RMQ
        st_flat = st_wrapper.flat_tree
        self._sweep_guard(st_flat, registry)

        dup_cost, loss_cost = self.tcf.weights
        sw = TargetSweep(SpeciesIndex(st_flat), dup_cost, loss_cost)

        n2id = st_flat.name_id_to_node_id           # keyed by FULL name id, not pure
        h_id, clade_ids, valid_t, t_id_of = self._sweep_index(meta, single, st_flat, n2id, registry)

        self.logger.report_step(step, f"Success: {len(single)} single- & {len(multi)} multi-MTs")

        # ---------------- 2. gene trees, cap ---------------
        gene_mgr.filter_by_sweep_cap(gene_trees, st_flat, sw, clade_ids, h_id, valid_t, registry)
        if not gene_trees:
            return None

        # ---------------- 3. eager set: the input tree + multi-target MTs ---
        mul_trees: Dict[int, MulTree] = {0: MulTree(mt=st_wrapper)}
        if multi: mul_mgr.build(meta, mul_trees=mul_trees, keep=multi, label='eager ')

        if len(mul_trees) > 1:
            if not gene_mgr.cull(mul_trees, gene_trees, registry):
                return None                       # check-nums mode
            eager_scores, _ = self.recon_all(mul_trees, gene_trees, registry, retmap=False)

        # De-dup must be done after the eager block, because it calls cull()
        rep_of, unique_gts, weight_of_rep = self._dedup_views(gene_trees, label="[sweep]")

        # Score eager multi-MTs
        if len(mul_trees) == 1:
            from .algo import PairwiseRecon
            pr = PairwiseRecon(dup_cost, loss_cost, STRICT_TARGETS)
            # For now, only the input tree: score it directly, no pool, no pickles.
            st_score = 0.0
            for g_idx, gt_flat in unique_gts.items():
                s, _ = pr.reconcile_sl(gt_flat, st_flat, registry=registry)
                st_score += s * weight_of_rep.get(g_idx, 1.0)
            eager_scores = [(0, round(st_score, 3))]

        all_scores: Dict[int, float] = dict(eager_scores)

        # ---------------- 4. the sweep -------------------------------------
        step = "Sweeping candidate placements"
        self.logger.report_step(step, "In progress...")

        by_h1: Dict[str, List[int]] = {}
        for c_idx in single:
            by_h1.setdefault(meta[c_idx][0], []).append(c_idx)

        # TODO(WorkPool)
        worker = partial(_worker_sweep_single,
                        st_flat=st_flat, flat_gts=unique_gts, weights=weight_of_rep,
                        dup_cost=dup_cost, loss_cost=loss_cost, valid_t=valid_t,
                        use_exact=self.use_exact, use_gray=self.use_gray)
        donors = [(name, h_id[name]) for name in by_h1 if name in h_id]
        results = self._dispatch_sweep(donors, worker)

        for h1_name, c_idxs in by_h1.items():
            got = results.get(h1_name)
            if got is None:
                continue
            vt, totals = got
            pos = {int(t): k for k, t in enumerate(vt)}
            for c_idx in c_idxs:
                t_id = t_id_of.get(c_idx)
                if t_id is not None and t_id in pos:
                    all_scores[c_idx] = round(float(totals[pos[t_id]]), 3)

        self.logger.report_step(step, f"Success: scored {len(all_scores)-1} candidates")

        sorted_scores = sorted(all_scores.items(), key=lambda kv: (kv[1], kv[0]))

        # ---------------- 5. selection (same policy as run()) ---------------
        n_best = self._resolve_n_best(len(sorted_scores))
        input_idx = 0 if 0 in mul_trees else None

        scores, input_score, valleys, method = self._distribute_scores(sorted_scores)
        target_idxs, passed_events = self.select_mts(n_best, sorted_scores, input_idx,
                                                    input_score, valleys)

        # ---------------- 6. materialise & map the winners only --------------------
        target_idxs_not_built = [i for i in target_idxs if i not in mul_trees]
        mul_mgr.build(meta, mul_trees=mul_trees, keep=target_idxs_not_built, label='selected ')

        target_idxs = [i for i in target_idxs if i in mul_trees]

        # Groups exist only for the eager MTs; collapse for the newly built winners.
        # collapse_groups skips index 0 and reuses any pickle that already exists, so
        # this neither rebuilds the eager ones nor filters gene trees a second time.
        if target_idxs_not_built: gene_mgr.collapse_groups(
            {i: mul_trees[i] for i in target_idxs if i != 0}, gene_trees, registry, label='selected ')

        # Flatten: recon_all runs this earlier, which is false in the sweep pipeline.
        for i in target_idxs: mul_trees[i].mt.make_lca(registry)

        detailed_res = self.recon_lowest_maps(target_idxs, mul_trees, gene_trees, registry)

        return self._finalise(sorted_scores, mul_trees, gene_trees, detailed_res, passed_events,
                              scores, input_score, valleys, method, meta=meta)

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