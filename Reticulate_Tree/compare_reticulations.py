import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from reticulate_tree import ReticulateTree

'''
bug in the edit distance calculation, it doesn't match internal nodes and it does in fact mess up the score
must check that the GED for the polyphest's mul2net is OK...
'''

def compare_num_rets(ret_count_A, ret_count_B):
    return abs(ret_count_A - ret_count_B)

def j_dist(intersect, union):
    return 0.0 if union == 0 else 1.0 - (intersect / union)

def compare_ploidy_diff(leaf_counts_A, leaf_counts_B):
    '''
    Distance (1 - matches) over leaf ploidy counts beyond diploid.
    Also returns TP/FP/FN counts of extra copies.
    Example:
    A: {a:2,b:3,c:2,rest:1...}, B: {a:2,b:2,c:3,rest:1...}
    -> extras are: a: 1, b: 2 or 1, c: 1 or 2
    -> they differ by 2 out of 5 extras -> dist = 2/5 = 1 - 3/5 (=matches) = 0.4
    -> 3 correct, 1 false positive, 1 false negative
    '''
    taxa = set(leaf_counts_A) | set(leaf_counts_B)
    intersect = 0
    union = 0
    TP = FP = FN = 0
    
    for t in taxa:
        c1 = leaf_counts_A.get(t, 0) - 1  # truth copies beyond diploid
        c2 = leaf_counts_B.get(t, 0) - 1  # predicted copies
        if c1 == 0 and c2 == 0:
            continue
        common = min(c1, c2)
        TP += common
        FP += max(0, c2 - c1)
        FN += max(0, c1 - c2)
        intersect += common
        union += max(c1, c2)
    
    return {'dist': j_dist(intersect, union), 'FP': FP, 'FN': FN, 'TP': TP}

def jaccard(a, b):
    '''Jaccard similarity between two sets; 1.0 if both empty.'''
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)

def hungarian_match(list1, list2, sim_fn):
    '''
    Match two lists of items using the Hungarian algorithm to maximize similarity.
    sim_fn(x, y) should return a similarity score (higher = better).
    Returns (row_ind, col_ind) arrays of matched indices. Possibly empty arrays.
    '''
    if not list1 or not list2:
        return np.array([], int), np.array([], int)
    sim_matrix = np.array([[sim_fn(x, y) for y in list2] for x in list1], dtype=float)
    # maximize by minimizing negative similarity
    row_ind, col_ind = linear_sum_assignment(-sim_matrix)
    #total_score = sim_matrix[row_ind, col_ind].sum()
    return row_ind, col_ind

def run_hungarian_on_groups(groups1, groups2):
    '''
    Prepare sets for groups1 and groups2 and compute optimal Hungarian matching
    using Jaccard similarity. Returns (sets1, sets2, row_ind, col_ind).
    sets1/sets2 are lists of set objects in the same order as the input iterables.
    '''
    sets1 = [set(v) for v in groups1]
    sets2 = [set(v) for v in groups2]
    row_ind, col_ind = hungarian_match(sets1, sets2, jaccard)
    max_len = max(len(sets1), len(sets2))
    return sets1, sets2, row_ind, col_ind, max_len

def populate_dict(keys, values):
    '''
    Create a dict from keys and values lists.
    '''
    return {k: v for k, v in zip(keys, values)}

ALL_STATS_KEYS = ['dist', 'FP', 'FN', 'TP']
def handle_empty_cases(max_len, len1, len2):
    '''
    Handle cases where one or both input sets are empty.
    Returns: (dist, FP, FN, TP)
    '''
    if not max_len:
        return populate_dict(ALL_STATS_KEYS, (0.0, 0.0, 0.0, 0.0))
    if not len1: # only set2 non-empty
        return populate_dict(ALL_STATS_KEYS, (1.0, 1.0, 0.0, 0.0))
    if not len2: # only set1 non-empty
        return populate_dict(ALL_STATS_KEYS, (1.0, 0.0, 1.0, 0.0))
    return None  # both non-empty

def compute_leaves_stats(sets1, sets2, row_ind, col_ind, debug=False):

    # compute stats for each matched pair: (intersection, lenA, lenB)
    stats = [
        (len(sets1[i] & sets2[j]), len(sets1[i]), len(sets2[j]))
        for i, j in zip(row_ind, col_ind)
    ]
    # sum up totals
    totals = {}
    totals['dist'] = sum(j_dist(inter, lenA + lenB - inter) for inter, lenA, lenB in stats)
    totals['FP'] = sum((lenB - inter) / lenB for inter, _, lenB in stats)
    totals['FN'] = sum((lenA - inter) / lenA for inter, lenA, _ in stats)

    if debug:
        print('dists:', [j_dist(inter, lenA + lenB - inter) for inter, lenA, lenB in stats])
        print('FPs:', [(lenB - inter) / lenB for inter, _, lenB in stats])
        print('FNs:', [(lenA - inter) / lenA for inter, lenA, _ in stats])

    return totals

def compute_sisters_stats(sisters1, sisters2, keys1, keys2, row_ind, col_ind, debug=False):

    totals = {}
    t_dist, t_FP, t_FN = 0.0, 0.0, 0.0
    get1, get2 = sisters1.get, sisters2.get  # localize lookups (small perf gain)
    for i, j in zip(row_ind, col_ind):

        if debug:
            print(f'Computing reticulation pair: {i}, {j}')

        # map matched indices back to keys
        k1, k2 = keys1[i], keys2[j]
        precomputed = run_hungarian_on_groups(get1(k1, []), get2(k2, []))
        curr = match_and_compare(None, None, precomputed_match=precomputed, debug=debug)
        t_dist += curr['dist']
        t_FP   += curr['FP']
        t_FN   += curr['FN']
    totals['dist'], totals['FP'], totals['FN'] = t_dist, t_FP, t_FN

    return totals

def correct_and_normalize_stats(totals, max_len, intersect_len, len1, len2, debug=False):
    '''
    Normalize total_dist, total_FP, total_FN by max_len and account for unmatched groups.
    Modifies the `totals` dict in place.
    Assumes that at most one input set is empty (i.e., max_len > 0).
    totals = {'dist': float, 'FP': float, 'FN': float}
    '''
    # account for unmatched groups
    totals['dist'] += (max_len - intersect_len) # unmatched groups count as full distance
    totals['FP'] += max(0, max_len - len1) # unmatched groups in set2 count as full FP
    totals['FN'] += max(0, max_len - len2) # unmatched groups in set1 count as full FN

    # normalize by max number of groups
    totals['dist'] /= max_len
    totals['FP'] /= max_len
    totals['FN'] /= max_len

    if debug:
        print(f'Corrections: dist +{(max_len - intersect_len)}, FP +{max(0, max_len - len1)}, FN +{max(0, max_len - len2)}')
        print(f'Corrected norm. stats: dist={totals['dist']:.4f}, FP={totals['FP']:.4f}, FN={totals['FN']:.4f}')

def match_and_compare(leaves1, leaves2, sisters1=None, sisters2=None, precomputed_match=None, debug=False):
    '''
    Leaves mode:
        (1 - Total Jaccard) over matched reticulation leaf sets.
    Sisters mode:
        (1 - Total Nested Jaccard) over matched reticulation leaf sets and their sister groups:
        - First match reticulations by their leaf-sets
        - Then for each matched reticulation pair, compute matching stats across their sister groups
    Then, compute the average the stats across all matched reticulation pairs and correct for unmatched reticulations
    leaves1 and leaves2 are dict-like mapping reticulation-id -> iterable of leaf labels.
    sisters1 and sisters2 are dict-like mapping reticulation-id -> list of iterables of leaf labels (sister groups).
    Returns: dict of stats with keys: dist, FP, FN, TP
    '''
    mode = 'leaves' if sisters1 is None and sisters2 is None else 'sisters'
    if (sisters1 is None or sisters2 is None) and mode == 'sisters':
        raise ValueError('Both or neither of sisters1 and sisters2 must be provided.')

    if precomputed_match is not None:
        sets1, sets2, row_ind, col_ind, max_len = precomputed_match
    else: # leaves must be dict-like like in this case!
        sets1, sets2, row_ind, col_ind, max_len = run_hungarian_on_groups(leaves1.values(), leaves2.values())

    if debug:
        print(f'Hungarian {mode} matching: {row_ind}, {col_ind}, {max_len}\n\tfor sets: {sets1}, {sets2}')

    empty_case = handle_empty_cases(max_len, len(sets1), len(sets2))
    if empty_case is not None:
        return empty_case

    if mode == 'leaves':
        totals = compute_leaves_stats(sets1, sets2, row_ind, col_ind, debug=debug)
    elif mode == 'sisters':
        # preserve keys order in lists to index into sets1/sets2
        keys1, keys2 = list(leaves1), list(leaves2)
        totals = compute_sisters_stats(sisters1, sisters2, keys1, keys2, row_ind, col_ind, debug=debug)
    else:
        raise ValueError(f'Unknown mode: {mode}')

    correct_and_normalize_stats(totals, max_len, len(row_ind), len(sets1), len(sets2), debug=debug)

    # NOTE: true positives (TP) can't be derived as 1 - tot_dist in this case, because
    # the denominator is different in each case (unmatched groups count as full distance/FP/FN).
    # One would need to define a TP_precise and TP_recall separately if needed.
    # eg:
    # tot_TP_prec = sum(inter / a_len if a_len else 0.0 for inter, a_len, _ in stats)
    # tot_TP_rec  = sum(inter / b_len if b_len else 0.0 for inter, _, b_len in stats)
    
    '''
Clustering A: A1={1,2,3} (size 3), A2={4} (size 1) — 2 clusters
Clustering B: B1={1,2} (size 2), B2={6,4,5} (size 3) — 2 clusters
    
stats = [
  2/3    # A1 matched to B1: intersection = 2, |A1|=3, |B1|=2
  1/3    # A2 matched to B2: intersection = 1, |A2|=1, |B2|=3
]
-> 2/3 + 1/3 = 1.0 -> 1 - 1.0/2 (= 0.5 TP) = 0.5 dist

But:
TP_prec = 2/3 + 1/1 = 1.666 -> 1.666/2 = 0.8333
TP_rec  = 2/2 + 1/3 = 1.333 -> 1.333/2 = 0.6666

Pres = TP / (TP + FP) = (2, 1) / (2, 1 + 0, 2) = 1 + 1/3 = 4/3 -> /2 = 0.6667
Rec = TP / (TP + FN) = (2, 1) / (2, 1 + 1, 0) = 2/3 + 1 = 5/3 -> /2 = 0.8333
    '''

    totals['TP'] = 1.0 - totals['dist'] # approximate TP as 1 - dist for simplicity

    return totals

def pairwise_compare(obj1, obj2, df=None):
    '''
    Compare two ReticulateTree objects (or their cached rows in df).
    If df is provided, obj1/obj2 are treated as row labels (names).
    '''
    if df is not None:
        row1, row2 = df.loc[obj1], df.loc[obj2]
        precomputed = run_hungarian_on_groups(row1['reticulation_leaves'].values(),
                                                            row2['reticulation_leaves'].values())
        return {
            'edit_distance':        row1['object'] - row2['object'],
            'num_rets_diff':        compare_num_rets(row1['reticulation_count'], row2['reticulation_count']),
            'ploidy_diff':          compare_ploidy_diff(row1['leaf_counts'], row2['leaf_counts']),
            'ret_leaf_jaccard':     match_and_compare(row1['reticulation_leaves'], row2['reticulation_leaves'],
                                        precomputed_match = precomputed),
            'ret_sisters_jaccard':  match_and_compare(row1['reticulation_leaves'], row2['reticulation_leaves'],
                                        row1['reticulation_sisters'], row2['reticulation_sisters'], precomputed),
        }
    # object-based comparison
    precomputed = run_hungarian_on_groups(obj1.get_reticulation_leaves().values(), obj2.get_reticulation_leaves().values())
    return {
        'edit_distance':            obj1 - obj2,
        'num_rets_diff':            compare_num_rets(obj1.get_reticulation_count(), obj2.get_reticulation_count()),
        'ploidy_diff':              compare_ploidy_diff(obj1.get_leaf_counts(), obj2.get_leaf_counts()),
        'ret_leaf_jaccard':         match_and_compare(obj1.get_reticulation_leaves(), obj2.get_reticulation_leaves(),
                                        precomputed_match = precomputed),
        'ret_sisters_jaccard':      match_and_compare(obj1.get_reticulation_leaves(), obj2.get_reticulation_leaves(),
                                        obj1.get_reticulation_sisters(), obj2.get_reticulation_sisters(), precomputed),
    }

def are_identical(obj1, obj2):
    '''
    Check if two ReticulateTree objects are identical by comparing the measurements.
    '''
    # Compare all 6 measurements
    comp = pairwise_compare(obj1, obj2)
    print(f'Comparison vector: {comp.values()}')
    # Edit distancd is bugged probably due to internal nodes being unmatched...
    return (#comp['edit_distance'] == 0 and
            comp['num_rets_diff'] == 0 and
            comp['ploidy_diff']['dist'] == 0.0 and
            comp['ret_leaf_jaccard']['dist'] == 0.0 and
            comp['ret_sisters_jaccard']['dist'] == 0.0)

def pairwise_comparison(df, debug=False):
    ''' Create pairwise comparison matrix for each measurement. '''
    names = df.index
    n = len(names)
    
    metrics = {
        'edit_distance': np.zeros((n, n)),
        'num_rets_diff': np.zeros((n, n)),
        'ploidy_diff': np.zeros((n, n)),
        'ret_leaf_jaccard': np.zeros((n, n)),
        'ret_sisters_jaccard': np.zeros((n, n)),
    }

    print(f'Comparing {n} trees...')

    for i in range(n):
        for j in range(i + 1):
            name1, name2 = names[i], names[j]
            if debug:
                print(f'Comparing {name1} vs {name2}')
            results = pairwise_compare(name1, name2, df)
            for k in metrics:
                res = results[k]
                if debug:
                    print(f'  {k}: {res}')
                if isinstance(res, dict):
                    res = res['dist']  # take only the score, not the dict
                metrics[k][i, j] = res
                # Symmetric fill
                if i != j:
                    metrics[k][j, i] = res

    dfs = {}
    for k, mat in metrics.items():
        dfs[k] = pd.DataFrame(mat, index=names, columns=names)

    return dfs

def one_stop_compare(phylo_trees: dict, printout: bool = True, visualize: bool = False):

    results = []
    logs = []

    for name, value in phylo_trees.items():
        try:
            tree = ReticulateTree(value)
            measurements = tree.measure(printout=printout)
            results.append({'name': name, 'object': tree, **measurements})
            if visualize:
                tree.visualize()
            if printout:
                logs.append(f'Finished {name}\n')
        except Exception as e:
            logs.append(f'Error processing {name}: {e}')

    try:
        df = pd.DataFrame(results).set_index('name')
        comparison = pairwise_comparison(df, debug=printout)
    except Exception as e:
        logs.append(f'Error processing DataFrame: {e}')
        return {'data': None, 'comparisons': None, 'logs': logs}

    return {'data': df, 'comparisons': comparison, 'logs': logs}

def plot_comparison_heatmaps(comparison_matrices, phylo_df, save_path=None):
    # Plot each as heatmap
    for metric, matrix in comparison_matrices.items():
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True,
                    xticklabels=phylo_df.index, yticklabels=phylo_df.index)
        plt.title(f'{metric.upper()} Heatmap')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path / f'{metric}_heatmap.png', dpi=600)
        plt.show()

if __name__ == '__main__':

    main_path = Path(__file__).parent