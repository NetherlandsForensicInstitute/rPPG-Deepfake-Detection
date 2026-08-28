"""Plot per-condition metrics from cv_train per_type_fold_results.csv.

Participants 1-6, Paper, Screen and Mannequin are merged into a single
"Experiments" bar (averaged within each fold before CI is computed).

When --annotations/--sample-predictions are given, the source-condition plot and the
head-movement/lighting plots are combined into a single side-by-side figure per metric
(width ratios 3:1:1, matching analyze_predictions.py's accuracy_per_condition.svg layout)
instead of separate single-panel files.

Usage
-----
python analysis/plot_cv_results.py --results results_cv/per_type_fold_results.csv
python analysis/plot_cv_results.py --results results_cv/per_type_fold_results.csv --metric recall
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)

METRICS = ['accuracy', 'precision', 'recall', 'f1', 'auc']

EXPERIMENTS_TYPES = (
    {f'Participant {i}' for i in range(1, 7)}
    | {'Paper', 'Screen', 'Mannequin'}
)

AUTH_COLORS = {
    'Real':         '#4878CF',
    'Text2Video': '#D65F5F',
    'Faceswap':     '#E6A817',
}

# Side panels in the combined figure, in display order
CONDITION_COLS = ('head_movement', 'lighting')


def _ci(values, confidence=0.95):
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


def _authenticity(type_name):
    if type_name == 'Experiments':
        return 'Real'
    if type_name.endswith('SYN'):
        return 'Text2Video'
    if type_name.endswith('FS'):
        return 'Faceswap'
    return 'Real'


def build_plot_data(df, metric):
    df = df.copy()
    df['plot_type'] = df['type'].apply(
        lambda t: 'Experiments' if t in EXPERIMENTS_TYPES else t
    )

    # Average within each fold so multi-type groups become one observation per fold
    fold_means = (
        df.groupby(['fold', 'plot_type'])[metric]
        .mean()
        .reset_index()
    )

    rows = []
    for type_name, grp in fold_means.groupby('plot_type'):
        mean, _, lo, hi = _ci(grp[metric].values)
        rows.append({
            'type':         type_name,
            'label':        type_name.split('_')[0],
            'mean':         mean,
            'ci_lower':     lo,
            'ci_upper':     hi,
            'Authenticity': _authenticity(type_name),
        })

    agg = pd.DataFrame(rows)
    agg = agg.sort_values(
        ['Authenticity', 'mean'], ascending=[True, False]
    ).reset_index(drop=True)
    agg['label'] = pd.Categorical(agg['label'], categories=agg['label'], ordered=True)
    return agg


def _draw_source_bars(ax, agg, metric):
    """Draw the by-source (Authenticity-colored) bar chart onto an existing axes."""
    ci_lo = (agg['mean'] - agg['ci_lower']).clip(lower=0).values
    ci_hi = (agg['ci_upper'] - agg['mean']).clip(lower=0).values
    colors = [AUTH_COLORS.get(a, '#888888') for a in agg['Authenticity']]

    x = np.arange(len(agg))
    ax.bar(x, agg['mean'], color=colors, zorder=3)
    ax.errorbar(x, agg['mean'], yerr=[ci_lo, ci_hi],
                fmt='none', color='black', capsize=4, linewidth=1.2, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(agg['label'], rotation=45, ha='right', fontsize=13)
    ax.set_ylabel(metric.capitalize(), fontsize=14)
    ax.set_xlabel('Source', fontsize=14, labelpad=10)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', zorder=0)
    ax.tick_params(axis='y', labelsize=13)
    sns.despine(right=True, top=True, ax=ax)

    present = agg['Authenticity'].unique()
    handles = [Patch(color=c, label=l) for l, c in AUTH_COLORS.items() if l in present]
    ax.legend(handles=handles, fontsize=12,
              loc='upper left', bbox_to_anchor=(-0.35, 1), borderaxespad=0)


def make_plot(agg, metric, output_dir):
    """Standalone single-panel source plot (used when no annotation data is available)."""
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, ax = plt.subplots(figsize=(max(10, len(agg) * 0.85), 6))

    _draw_source_bars(ax, agg, metric)

    plt.tight_layout()
    path = os.path.join(output_dir, f'{metric}_per_condition_cv.svg')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'Saved {path}')


def _load_annotations(annot_path, sample_df):
    """Join sample_predictions with a video-level annotations JSON.

    The annotations JSON has columns keyed as "TypePrefix,filename" with rows
    for gender, skin_color, head_movement, lighting (same format as analyze_predictions.py).
    """
    annots = pd.read_json(annot_path)
    for col in ['head_movement', 'lighting']:
        sample_df[col] = None
    for idx, row in sample_df.iterrows():
        key = str(row['Type']).split('_')[0] + ',' + os.path.basename(str(row['Filename']))
        if key in annots.columns:
            for col in ['head_movement', 'lighting']:
                if col in annots.index:
                    sample_df.at[idx, col] = annots.loc[col, key]
    return sample_df


def build_annotation_plot_data(sample_df, condition_col, metric):
    """Aggregate sample predictions by annotation condition, CI across folds."""
    df = sample_df.dropna(subset=[condition_col]).copy()

    def _metrics_row(grp):
        yt = grp['label'].values
        yp = grp['y_pred'].values
        yb = grp['y_prob'].values
        if metric == 'accuracy':
            return accuracy_score(yt, yp)
        if metric == 'precision':
            return precision_score(yt, yp, average='binary', zero_division=0)
        if metric == 'recall':
            return recall_score(yt, yp, average='binary', zero_division=0)
        if metric == 'f1':
            return f1_score(yt, yp, average='binary', zero_division=0)
        if metric == 'auc':
            return roc_auc_score(yt, yb) if len(np.unique(yt)) > 1 else float('nan')
        return float('nan')

    fold_means = (
        df.groupby(['fold', condition_col])
        .apply(_metrics_row)
        .reset_index(name=metric)
    )

    rows = []
    for val, grp in fold_means.groupby(condition_col):
        mean, _, lo, hi = _ci(grp[metric].values)
        rows.append({'condition': val, 'mean': mean, 'ci_lower': lo, 'ci_upper': hi})

    agg = pd.DataFrame(rows).sort_values('mean', ascending=False).reset_index(drop=True)
    agg['condition'] = pd.Categorical(agg['condition'], categories=agg['condition'], ordered=True)
    return agg


def _draw_condition_bars(ax, agg, condition_col, metric):
    """Draw a plain grey by-condition bar chart onto an existing axes."""
    ci_lo = (agg['mean'] - agg['ci_lower']).clip(lower=0).values
    ci_hi = (agg['ci_upper'] - agg['mean']).clip(lower=0).values

    x = np.arange(len(agg))
    ax.bar(x, agg['mean'], color='lightgray', zorder=3)
    ax.errorbar(x, agg['mean'], yerr=[ci_lo, ci_hi],
                fmt='none', color='black', capsize=4, linewidth=1.2, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(agg['condition'], rotation=0, fontsize=13)
    ax.set_ylabel(metric.capitalize(), fontsize=14)
    ax.set_xlabel(condition_col.replace('_', ' ').capitalize(), fontsize=14, labelpad=10)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', zorder=0)
    ax.tick_params(axis='y', labelsize=13)
    sns.despine(right=True, top=True, ax=ax)


def make_annotation_plot(agg, condition_col, metric, output_dir):
    """Standalone single-panel condition plot (used when no combined figure applies)."""
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, ax = plt.subplots(figsize=(max(6, len(agg) * 1.2), 6))

    _draw_condition_bars(ax, agg, condition_col, metric)

    plt.tight_layout()
    path = os.path.join(output_dir, f'{metric}_by_{condition_col}_cv.svg')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'Saved {path}')


def make_combined_plot(agg_main, side_panels, metric, output_dir):
    """Source plot + condition side panels in one figure (width ratios 3:1:1:...).

    side_panels: list of (condition_col, agg) pairs, drawn left to right after the
    main source panel, matching analyze_predictions.py's accuracy_per_condition.svg.
    """
    plt.rcParams['font.family'] = 'Times New Roman'
    width_ratios = [3] + [1] * len(side_panels)
    fig, axs = plt.subplots(1, 1 + len(side_panels), figsize=(15, 6),
                            gridspec_kw={'width_ratios': width_ratios})

    _draw_source_bars(axs[0], agg_main, metric)

    for ax, (condition_col, agg) in zip(axs[1:], side_panels):
        _draw_condition_bars(ax, agg, condition_col, metric)

    plt.tight_layout()
    path = os.path.join(output_dir, f'{metric}_per_condition_cv.svg')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'Saved {path}')


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--results', required=True,
                        help='Path to per_type_fold_results.csv produced by cv_train.py.')
    parser.add_argument('--sample-predictions', default=None,
                        help='Path to sample_predictions.csv produced by cv_train.py '
                             '(required for --annotations plots).')
    parser.add_argument('--annotations', default=None,
                        help='Path to video-level annotations JSON '
                             '(same format as analyze_predictions.py).')
    parser.add_argument('--output-dir', default=None,
                        help='Directory for output SVGs (default: same directory as --results).')
    parser.add_argument('--metric', choices=METRICS, default=None,
                        help='Single metric to plot. Omit to plot all metrics.')
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.results))
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(args.results)
    metrics = [args.metric] if args.metric else METRICS

    # Load annotation-based sample predictions up front, if given, so each metric's
    # source plot can be combined with its condition side panels in one figure
    sample_df = None
    available_conditions = []
    if args.annotations:
        if not args.sample_predictions:
            print('Warning: --annotations requires --sample-predictions. Ignoring --annotations.')
        else:
            sample_df = pd.read_csv(args.sample_predictions)
            sample_df = _load_annotations(args.annotations, sample_df)
            available_conditions = [
                c for c in CONDITION_COLS if sample_df[c].notna().sum() > 0
            ]
            for c in CONDITION_COLS:
                if c not in available_conditions:
                    print(f'No annotation data for "{c}", skipping.')

    for metric in metrics:
        if metric not in df.columns:
            print(f'Warning: metric "{metric}" not found in CSV, skipping.')
            continue
        agg_main = build_plot_data(df, metric)

        if not available_conditions:
            # No annotation data at all: plain single-panel source plot
            make_plot(agg_main, metric, output_dir)
            continue

        # Build each available side panel and combine with the source plot
        side_panels = []
        for condition_col in available_conditions:
            agg_side = build_annotation_plot_data(sample_df, condition_col, metric)
            if not agg_side.empty:
                side_panels.append((condition_col, agg_side))

        if side_panels:
            make_combined_plot(agg_main, side_panels, metric, output_dir)
        else:
            make_plot(agg_main, metric, output_dir)


if __name__ == '__main__':
    main()