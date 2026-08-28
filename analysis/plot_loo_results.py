import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

METRICS_FOLD    = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'auc']
METRICS_PERTYPE = ['accuracy', 'f1', 'auc']

AUTH_COLORS = {
    'Real':         '#4878CF',
    'Text2Video': '#D65F5F',
    'Faceswap':     '#E6A817',
}


def _authenticity(type_name):
    if type_name.endswith('SYN'):
        return 'Text2Video'
    if type_name.endswith('FS'):
        return 'Faceswap'
    return 'Real'


def _short(type_name):
    return type_name.split('_')[0]


def _make_bar_plot(labels, values, colors, authenticities, ylabel, output_path):
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 6))

    x = np.arange(len(labels))
    ax.bar(x, values, color=colors, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=13)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlabel('Left-out generator', fontsize=14, labelpad=10)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', zorder=0)
    ax.tick_params(axis='y', labelsize=13)
    sns.despine(right=True, top=True, ax=ax)

    present_auth = sorted(set(authenticities))
    legend_handles = [
        Patch(color=AUTH_COLORS.get(a, '#888888'), label=a)
        for a in present_auth if a in AUTH_COLORS
    ]
    ax.legend(handles=legend_handles, fontsize=12,
              loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f'Saved {output_path}')


def plot_overall(fold_df, metric, output_dir):
    """Bar chart of overall test-set performance per left-out generator."""
    df = fold_df.sort_values(metric, ascending=False).reset_index(drop=True)

    orig_types     = df['left_out_type'].tolist()
    labels         = [_short(t) for t in orig_types]
    values         = df[metric].values
    authenticities = [_authenticity(t) for t in orig_types]
    colors         = [AUTH_COLORS.get(a, '#888888') for a in authenticities]

    _make_bar_plot(
        labels, values, colors, authenticities,
        ylabel=metric,
        output_path=os.path.join(output_dir, f'{metric}_per_generator_loo.svg'),
    )


def plot_generalisation(per_type_df, metric, output_dir):
    """Bar chart of performance specifically on the unseen (left-out) samples."""
    if metric.lower() not in per_type_df.columns:
        print(f'Warning: metric "{metric}" not in per_type_results, skipping generalisation plot.')
        return

    df = per_type_df[per_type_df['is_left_out']].copy()
    df = df.sort_values(metric.lower(), ascending=False).reset_index(drop=True)

    orig_types     = df['left_out_type'].tolist()
    labels         = [_short(t) for t in orig_types]
    values         = df[metric.lower()].values
    authenticities = [_authenticity(t) for t in orig_types]
    colors         = [AUTH_COLORS.get(a, '#888888') for a in authenticities]

    _make_bar_plot(
        labels, values, colors, authenticities,
        ylabel=metric,
        output_path=os.path.join(output_dir, f'{metric}_generalisation_loo.svg'),
    )


def plot_accuracy_precision(fold_df, output_dir):
    """Grouped bar chart: Accuracy and Precision side by side for each left-out generator."""
    for col in ('accuracy', 'precision'):
        if col not in fold_df.columns:
            print(f'Warning: "{col}" not in fold_results.csv, skipping combined plot.')
            return

    df = fold_df.sort_values('accuracy', ascending=False).reset_index(drop=True)
    orig_types     = df['left_out_type'].tolist()
    labels         = [_short(t) for t in orig_types]
    authenticities = [_authenticity(t) for t in orig_types]
    auth_colors    = [AUTH_COLORS.get(a, '#888888') for a in authenticities]

    plt.rcParams['font.family'] = 'Times New Roman'
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.0), 6))

    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, df['accuracy'].values,  width, color=auth_colors,
           label='Accuracy',  zorder=3)
    ax.bar(x + width / 2, df['precision'].values, width, color=auth_colors,
           alpha=0.5, label='Precision', zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=13)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_xlabel('Left-out generator', fontsize=14, labelpad=10)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', zorder=0)
    ax.tick_params(axis='y', labelsize=13)
    sns.despine(right=True, top=True, ax=ax)

    # Authenticity colours legend
    present_auth = sorted(set(authenticities))
    auth_handles = [
        Patch(color=AUTH_COLORS.get(a, '#888888'), label=a)
        for a in present_auth if a in AUTH_COLORS
    ]
    # Metric pattern legend (solid = Accuracy, faded = Precision)
    metric_handles = [
        Patch(facecolor='grey', label='Accuracy'),
        Patch(facecolor='grey', alpha=0.5, label='Precision'),
    ]
    first_legend = ax.legend(handles=auth_handles, fontsize=12,
                             loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.add_artist(first_legend)
    ax.legend(handles=metric_handles, fontsize=12,
              loc='upper left', bbox_to_anchor=(1.01, 0.6), borderaxespad=0)

    plt.tight_layout()
    path = os.path.join(output_dir, 'accuracy_precision_per_generator_loo.svg')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'Saved {path}')


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--results', required=True,
                        help='Path to fold_results.csv produced by loo_generalization.py.')
    parser.add_argument('--per-type', default=None,
                        help='Path to per_type_results.csv (for generalisation plot).')
    parser.add_argument('--output-dir', default=None,
                        help='Directory for output SVGs (default: same dir as --results).')
    parser.add_argument('--metric', choices=METRICS_FOLD, default=None,
                        help='Single metric to plot. Omit to plot all metrics.')
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.results))
    os.makedirs(output_dir, exist_ok=True)

    fold_df = pd.read_csv(args.results)
    metrics = [args.metric] if args.metric else METRICS_FOLD

    per_type_df = pd.read_csv(args.per_type) if args.per_type else None

    for metric in metrics:
        if metric not in fold_df.columns:
            print(f'Warning: "{metric}" not found in fold_results.csv, skipping.')
            continue
        plot_overall(fold_df, metric, output_dir)
        if per_type_df is not None:
            plot_generalisation(per_type_df, metric, output_dir)

    plot_accuracy_precision(fold_df, output_dir)


if __name__ == '__main__':
    main()
