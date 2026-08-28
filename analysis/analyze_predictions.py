import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    f1_score, precision_score, recall_score, roc_auc_score,
)

AUTH_COLORS = {
    'Real':         '#4878CF',
    'Text2Video': '#D65F5F',
    'Faceswap':     '#E6A817',
}


def _ci(values, confidence=0.95):
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    mean = arr.mean() if n > 0 else float('nan')
    if n < 2:
        return mean, float('nan'), float('nan')
    se = arr.std(ddof=1) / np.sqrt(n)
    t = scipy.stats.t.ppf((1 + confidence) / 2, df=n - 1)
    return mean, mean - t * se, mean + t * se


def evaluate_results(subset_df):
    y_true = subset_df['label']
    y_pred = subset_df['predicted_label']
    y_score = subset_df['predicted_prob']
    return {
        'AUC': roc_auc_score(y_true, y_score),
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='binary'),
        'Recall': recall_score(y_true, y_pred, average='binary'),
        'F1 Score': f1_score(y_true, y_pred, average='binary'),
        'Confusion Matrix': confusion_matrix(y_true, y_pred).tolist(),
    }


def calculate_metrics_for_column(df, column, fold_col=None):
    """Compute accuracy per subclass with 95% CI across folds when available."""
    subset = df.dropna(subset=[column])

    if fold_col and fold_col in subset.columns:
        fold_rows = (
            subset.groupby([fold_col, column])
            .apply(lambda g: accuracy_score(g['label'], g['predicted_label']))
            .reset_index(name='accuracy')
        )
        rows = []
        for subclass, grp in fold_rows.groupby(column):
            mean, lo, hi = _ci(grp['accuracy'].values)
            rows.append({'subclass': subclass, 'accuracy': mean,
                         'ci_lower': lo, 'ci_upper': hi})
        return pd.DataFrame(rows)

    rows = []
    for subclass, group in subset.groupby(column):
        tn, fp, fn, tp = confusion_matrix(
            group['label'], group['predicted_label'], labels=[0, 1]
        ).ravel()
        total = tp + tn + fp + fn
        rows.append({
            'subclass':  subclass,
            'accuracy':  (tp + tn) / total if total > 0 else 0,
            'ci_lower':  float('nan'),
            'ci_upper':  float('nan'),
        })
    return pd.DataFrame(rows)


def _bar_with_ci(ax, x, means, ci_lower, ci_upper, colors, **bar_kwargs):
    ax.bar(x, means, color=colors, zorder=3, **bar_kwargs)
    valid = ~np.isnan(ci_lower) & ~np.isnan(ci_upper)
    if valid.any():
        err_lo = np.clip(means[valid] - ci_lower[valid], 0, None)
        err_hi = np.clip(ci_upper[valid] - means[valid], 0, None)
        ax.errorbar(x[valid], means[valid], yerr=[err_lo, err_hi],
                    fmt='none', color='black', capsize=4, linewidth=1.2, zorder=4)


def main():
    parser = argparse.ArgumentParser(
        description='Analyse deepfake detection predictions produced by train.py or cv_train.py.'
    )
    parser.add_argument('--predictions', type=str, default=None,
                        help='Path to predicted_labels.csv produced by train.py.')
    parser.add_argument('--sample-predictions', type=str, default=None,
                        help='Path to sample_predictions.csv produced by cv_train.py.')
    parser.add_argument('--annotations', type=str, default=None,
                        help='Path to annotations JSON for per-condition analysis (optional).')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Directory to save output figures (default: current dir).')
    args = parser.parse_args()

    if not args.predictions and not args.sample_predictions:
        parser.error('Provide --predictions or --sample-predictions.')

    os.makedirs(args.output_dir, exist_ok=True)

    csv_path = args.sample_predictions or args.predictions
    df = pd.read_csv(csv_path)

    # Normalise column names from sample_predictions.csv format
    df = df.rename(columns={'y_pred': 'predicted_label', 'y_prob': 'predicted_prob'})

    fold_col = 'fold' if 'fold' in df.columns else None

    # Optionally join video-level annotations (gender, skin colour, head movement, lighting)
    if args.annotations:
        annots = pd.read_json(args.annotations)
        for key, row in df.iterrows():
            col_key = row['Type'].split('_')[0] + ',' + row['Filename'].split('/')[-1]
            if col_key in annots.columns:
                df.loc[key, ['gender', 'skin_color', 'head_movement', 'lighting']] = annots[col_key]

    # Focus on test split only (predicted_labels.csv has a 'set' column; sample_predictions.csv does not)
    if 'set' in df.columns:
        df = df[df['set'] == 'test'].copy()

    # Authenticity labels
    df['Authenticity'] = 'Real'
    df.loc[df['Type'].str.endswith('SYN'), 'Authenticity'] = 'Generated'
    df.loc[df['Type'].str.endswith('FS'), 'Authenticity'] = 'Faceswap'

    # Per-authenticity metrics and confusion matrices
    metrics = []
    for subclass, group in df.groupby('Authenticity'):
        acc  = accuracy_score(group['label'], group['predicted_label'])
        prec = precision_score(group['label'], group['predicted_label'], zero_division=0)
        rec  = recall_score(group['label'], group['predicted_label'], zero_division=0)
        f1   = f1_score(group['label'], group['predicted_label'], zero_division=0)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix(group['label'], group['predicted_label']),
            display_labels=[0, 1],
        )
        disp.plot()
        plt.title(subclass)
        plt.savefig(os.path.join(args.output_dir, f'confusion_matrix_{subclass}.svg'))
        plt.close()
        metrics.append({'type': subclass, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1})

    print(pd.DataFrame(metrics).to_string(index=False))
    print(evaluate_results(df))

    # Overall confusion matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(df['label'], df['predicted_label']),
        display_labels=[0, 1],
    )
    disp.plot()
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix_overall.svg'))
    plt.close()

    # Per-source accuracy plot
    no_pulse_types = {'Screen', 'Mannequin', 'Paper'}
    experiment_types = {f'Participant {i}' for i in range(1, 7)}
    replace_map = {t: 'Experiments' for t in experiment_types}
    replace_map.update({t: 'No Pulse' for t in no_pulse_types})
    df['Type'] = df['Type'].replace(replace_map)
    df = df[df['Type'] != 'No Pulse']

    generator_model_metrics = calculate_metrics_for_column(df, 'Type', fold_col=fold_col)
    generator_model_metrics['Authenticity'] = 'Real'
    generator_model_metrics.loc[
        generator_model_metrics['subclass'].str.endswith('SYN'), 'Authenticity'
    ] = 'Text2Video'
    generator_model_metrics.loc[
        generator_model_metrics['subclass'].str.endswith('FS'), 'Authenticity'
    ] = 'Faceswap'
    generator_model_metrics['subclass'] = generator_model_metrics['subclass'].apply(
        lambda x: x.split('_')[0]
    )
    generator_model_metrics = generator_model_metrics.sort_values(
        by=['Authenticity', 'accuracy'], ascending=[True, False]
    ).reset_index(drop=True)
    generator_model_metrics['subclass'] = pd.Categorical(
        generator_model_metrics['subclass'],
        categories=generator_model_metrics['subclass'],
        ordered=True,
    )

    print(generator_model_metrics)

    plt.rcParams['font.family'] = 'Times New Roman'
    fig, axs = plt.subplots(1, 3, figsize=(15, 6), gridspec_kw={'width_ratios': [3, 1, 1]})

    x0 = np.arange(len(generator_model_metrics))
    colors0 = [AUTH_COLORS.get(a, '#888888') for a in generator_model_metrics['Authenticity']]
    _bar_with_ci(
        axs[0], x0,
        generator_model_metrics['accuracy'].values,
        generator_model_metrics['ci_lower'].values,
        generator_model_metrics['ci_upper'].values,
        colors=colors0,
    )
    axs[0].set_xticks(x0)
    axs[0].set_xticklabels(generator_model_metrics['subclass'], rotation=45, ha='right', fontsize=13)
    present = generator_model_metrics['Authenticity'].unique()
    handles = [Patch(color=c, label=l) for l, c in AUTH_COLORS.items() if l in present]
    axs[0].legend(handles=handles, loc='upper right', bbox_to_anchor=(-0.1, 1), fontsize=13)
    axs[0].set_xlabel('Source', fontsize=15, labelpad=10)
    axs[0].set_ylabel('Accuracy', fontsize=15)
    axs[0].grid(axis='y', zorder=0)
    axs[0].tick_params(axis='y', labelsize=13)
    axs[0].set_ylim(0, 1)
    sns.despine(right=True, top=True, ax=axs[0])

    if 'head_movement' in df.columns and df['head_movement'].notna().any():
        hm = calculate_metrics_for_column(df, 'head_movement', fold_col=fold_col)
        hm = hm.sort_values('subclass').reset_index(drop=True)
        x1 = np.arange(len(hm))
        _bar_with_ci(axs[1], x1, hm['accuracy'].values,
                     hm['ci_lower'].values, hm['ci_upper'].values,
                     colors=['darkgrey'] * len(hm))
        axs[1].set_xticks(x1)
        axs[1].set_xticklabels(hm['subclass'], rotation=0, fontsize=13)
        axs[1].set_xlabel('Head Movement', fontsize=15, labelpad=10)
        axs[1].set_ylabel('Accuracy', fontsize=15)
        axs[1].grid(axis='y', zorder=0)
        axs[1].tick_params(axis='y', labelsize=13)
        axs[1].set_ylim(0, 1)
        sns.despine(right=True, top=True, ax=axs[1])

    if 'lighting' in df.columns and df['lighting'].notna().any():
        li = calculate_metrics_for_column(df, 'lighting', fold_col=fold_col)
        li = li.sort_values('subclass').reset_index(drop=True)
        x2 = np.arange(len(li))
        _bar_with_ci(axs[2], x2, li['accuracy'].values,
                     li['ci_lower'].values, li['ci_upper'].values,
                     colors=['darkgrey'] * len(li))
        axs[2].set_xticks(x2)
        axs[2].set_xticklabels(li['subclass'], rotation=0, fontsize=13)
        axs[2].set_xlabel('Lighting', fontsize=15, labelpad=10)
        axs[2].set_ylabel('Accuracy', fontsize=15)
        axs[2].grid(axis='y', zorder=0)
        axs[2].tick_params(axis='y', labelsize=13)
        axs[2].set_ylim(0, 1)
        sns.despine(right=True, top=True, ax=axs[2])

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'accuracy_per_condition.svg'))
    plt.close()

    if 'skin_color' in df.columns and 'gender' in df.columns:
        fig, axs = plt.subplots(1, 2, figsize=(8, 6))

        sc = calculate_metrics_for_column(df, 'skin_color', fold_col=fold_col)
        sc = sc.sort_values('subclass').reset_index(drop=True)
        xsc = np.arange(len(sc))
        _bar_with_ci(axs[0], xsc, sc['accuracy'].values,
                     sc['ci_lower'].values, sc['ci_upper'].values,
                     colors=['steelblue'] * len(sc))
        axs[0].set_xticks(xsc)
        axs[0].set_xticklabels(sc['subclass'], rotation=0, fontsize=13)
        axs[0].set_xlabel('Skin Color', fontsize=15)
        axs[0].set_ylabel('Accuracy', fontsize=15)
        axs[0].set_ylim(0, 1)
        axs[0].tick_params(axis='y', labelsize=13)
        sns.despine(right=True, top=True, ax=axs[0])

        gn = calculate_metrics_for_column(df, 'gender', fold_col=fold_col)
        gn = gn.sort_values('subclass').reset_index(drop=True)
        xgn = np.arange(len(gn))
        _bar_with_ci(axs[1], xgn, gn['accuracy'].values,
                     gn['ci_lower'].values, gn['ci_upper'].values,
                     colors=['steelblue'] * len(gn))
        axs[1].set_xticks(xgn)
        axs[1].set_xticklabels(gn['subclass'], rotation=0, fontsize=13)
        axs[1].set_xlabel('Gender', fontsize=15)
        axs[1].set_ylabel('Accuracy', fontsize=15)
        axs[1].set_ylim(0, 1)
        axs[1].tick_params(axis='y', labelsize=13)
        sns.despine(right=True, top=True, ax=axs[1])

        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'accuracy_per_population.svg'))
        plt.close()


if __name__ == '__main__':
    main()
