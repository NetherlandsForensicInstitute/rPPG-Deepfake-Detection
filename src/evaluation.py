import numpy as np
import scipy.stats
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split


# Metrics to calculate
METRICS = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'auc']


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """Compute metrics for a single set of sample predictions and labels."""
    only_one_class = len(np.unique(y_true)) < 2
    return {
        'accuracy':  accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall':    recall_score(y_true, y_pred, average='binary', zero_division=0),
        'f1':        f1_score(y_true, y_pred, average='binary', zero_division=0),
        'auc':       roc_auc_score(y_true, y_prob) if not only_one_class else float('nan'),
    }


def ci(values, confidence=0.95):
    """Returns (mean, std, lower, upper) using t-distribution CI of the mean."""
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    mean = arr.mean() if n > 0 else float('nan')
    if n < 2:
        return mean, float('nan'), float('nan'), float('nan')
    std = arr.std(ddof=1)
    se = std / np.sqrt(n)
    t = scipy.stats.t.ppf((1 + confidence) / 2, df=n - 1)
    return mean, std, mean - t * se, mean + t * se


def bootstrap_ci(y_true, y_pred, y_prob, n_boot=1000, confidence=0.95, seed=0) -> dict:
    """
    Per-metric 95% CI for a single test set, via resampling rows with replacement.
    """
    # Coerce to arrays so integer-based resampling works
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    n = len(y_true)

    rng = np.random.default_rng(seed)
    alpha = (1 - confidence) / 2

    # Recompute every metric on n_boot resamples of the test set
    boot_values = {m: [] for m in METRICS}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        resampled = compute_metrics(y_true[idx], y_pred[idx], y_prob[idx])
        for m in METRICS:
            boot_values[m].append(resampled[m])

    # Percentile CI per metric
    result = {}
    for m in METRICS:
        vals = np.array(boot_values[m], dtype=float)
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            result[m] = (float('nan'), float('nan'))
        else:
            lo, hi = np.percentile(vals, [100 * alpha, 100 * (1 - alpha)])
            result[m] = (float(lo), float(hi))
    return result


def find_threshold(y_true, y_prob, metric='f1'):
    """Return the threshold in [0.05, 0.95] that maximizes the given metric."""
    best_thresh, best_score = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        y_pred = (y_prob >= t).astype(int)
        if metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        elif metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        else:
            score = accuracy_score(y_true, y_pred)
        if score > best_score:
            best_score, best_thresh = score, float(t)
    return best_thresh


def safe_split(df, test_size, random_state, stratify_col='Type'):
    """Split a DataFrame into train/test, stratified by a column if provided."""
    try:
        return train_test_split(df, test_size=test_size, random_state=random_state,
                                stratify=df[stratify_col])
    except (ValueError, KeyError):
        return train_test_split(df, test_size=test_size, random_state=random_state)


def aggregate_subset(records, group_col) -> pd.DataFrame:
    """Compute mean +/- CI across folds for each subset value (e.g. per-Type or per-group)."""
    df = pd.DataFrame(records)
    rows = []
    for key, grp in df.groupby(group_col):
        row = {group_col: key}
        for m in METRICS:
            mean, std, lo, hi = ci(grp[m].values)
            row[m] = mean
            row[f'{m}_std']      = std
            row[f'{m}_ci_lower'] = lo
            row[f'{m}_ci_upper'] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def print_summary_table(title, rows_df, group_col):
    """Print a summary table of metrics for each subset value (e.g. per-Type or per-group)."""
    print(f"\n{title}")
    print('-' * 65)
    header = f"  {group_col:<28} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}"
    print(header)
    for _, row in rows_df.iterrows():
        print(f"  {str(row[group_col]):<28} "
              f"{row['accuracy']:>7.4f} {row['precision']:>7.4f} "
              f"{row['recall']:>7.4f} {row['f1']:>7.4f} {row['auc']:>7.4f}")