import argparse
import os

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd

from src.evaluation import (
    METRICS, aggregate_subset, ci, compute_metrics, find_threshold,
    print_summary_table, safe_split,
)
from src.features import _FEATURE_FUNCS
from train import load_and_label


_ID_COLS = ('Filename', 'Type', 'Authenticity', 'label')


def _build_feature_table(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """
    Return a DataFrame with 'Filename', 'Type', 'label' + the numeric feature
    columns produced by the row-feature functions registered for feature_set.
    """
    # Row-feature functions for the selected baseline (e.g. compression, bpm, snr, ...)
    funcs = _FEATURE_FUNCS[feature_set]

    # Apply each row-feature function per sample and merge into one dict per row
    def _row_features(row):
        feats = {}
        for func in funcs:
            feats.update(func(row))
        return pd.Series(feats)

    feature_cols = df.apply(_row_features, axis=1)

    # Get ID (metadata) columns
    id_cols = [c for c in ('Filename', 'Type', 'label') if c in df.columns]

    # Return a single dataframe with all features
    return pd.concat([df[id_cols].reset_index(drop=True),
                      feature_cols.reset_index(drop=True)], axis=1)


def _make_pipeline(class_weight, C, seed) -> Pipeline:
    """
    Creates a LogisticRegression pipeline with a SimpleImputer, StandardScaler, and LogisticRegression.

    Arguments:
    - class_weight: Class weights for imbalanced datasets.
    - C: Regularization parameter for LogisticRegression.
    - seed: Random seed for reproducibility.
    """
    return Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('clf', LogisticRegression(C=C, class_weight=class_weight, max_iter=2000, random_state=seed)),
    ])


def _run_fold(fold_name, train_df, val_df, test_df, feature_cols, args):
    """
    Fit one fold's LogisticRegression baseline.

    Returns a tuple containing:
    - overall: overall performance metrics,
    - per_type: 1-row per type,
    - result: the full test set predictions, with 'y_pred' and 'y_prob' columns.
    """
    print(f"\n{'='*65}")
    print(f"Fold {fold_name}  —  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    print(f"{'='*65}")

    # Create data splits
    x_train, y_train = train_df[feature_cols].values, train_df['label'].values
    x_val, y_val = val_df[feature_cols].values, val_df['label'].values
    x_test, y_test = test_df[feature_cols].values, test_df['label'].values

    # Make pipeline and fit model
    class_weight = None if args.no_class_weights else 'balanced'
    pipeline = _make_pipeline(class_weight, args.C, args.seed + int(fold_name) - 1)
    pipeline.fit(x_train, y_train)

    # Run predictions on val set
    y_val_prob = pipeline.predict_proba(x_val)[:, 1]
    threshold = find_threshold(y_val, y_val_prob, metric=args.threshold_metric)
    print(f"  Threshold ({args.threshold_metric} on val): {threshold:.2f}")
    y_prob = pipeline.predict_proba(x_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # Compute metrics
    overall = compute_metrics(y_test, y_pred, y_prob)
    overall['fold'] = fold_name
    overall['threshold'] = threshold
    print("  Overall: " + "  ".join(f"{k}={v:.4f}" for k, v in overall.items() if k != 'fold'))

    base_cols = ['Filename', 'Type', 'label'] if 'Filename' in test_df.columns else ['Type', 'label']
    result = test_df[base_cols].copy().reset_index(drop=True)
    result['fold']   = fold_name
    result['y_pred'] = y_pred
    result['y_prob'] = y_prob

    # Compute per-type metrics
    per_type = []
    for t, grp in result.groupby('Type'):
        m = compute_metrics(grp['label'].values, grp['y_pred'].values, grp['y_prob'].values)
        per_type.append({'fold': fold_name, 'type': t, 'n': len(grp), **m})

    return overall, per_type, result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=str, required=True,
                        help='Path to the NDJSON signals/features file.')
    parser.add_argument('--feature-set', choices=_FEATURE_FUNCS, required=True,
                        help='Which baseline feature set to use.')
    parser.add_argument('--output-dir', type=str, default='results_baseline',
                        help='Directory for results (default: results_baseline/).')
    parser.add_argument('--mode', choices=['k-fold', 'repeated'], default='repeated',
                        help='CV strategy (default: repeated).')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--test-size', type=float, default=0.25)
    parser.add_argument('--val-size', type=float, default=0.15)
    parser.add_argument('--C', type=float, default=1.0,
                        help='Inverse regularization strength for LogisticRegression (default: 1.0).')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--real-types', type=str, required=True,
                        help='Comma-separated Type values treated as real (label=0).')
    parser.add_argument('--no-class-weights', action='store_true')
    parser.add_argument('--threshold-metric', choices=['f1', 'recall', 'accuracy'], default='f1')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Get real types and create dataframe
    real_types = [t.strip() for t in args.real_types.split(',')]
    df = load_and_label(args.data, real_types)
    df = df.reset_index(drop=True)
    df['label'] = df['label'].astype(int)

    print(f"Loaded {len(df)} samples  "f"(real={int((df['label']==0).sum())}  fake={int((df['label']==1).sum())})")

    # Create feature dataframe
    feature_df = _build_feature_table(df, args.feature_set)
    feature_cols = [c for c in feature_df.columns if c not in ('Filename', 'Type', 'label')]
    print(f"Feature set '{args.feature_set}': {len(feature_cols)} columns -> {feature_cols}")

    # Create lists to store results
    all_overall, all_per_type, all_sample_preds = [], [], []

    # Loop over folds and run each fold
    if args.mode == 'k-fold':
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(feature_df, feature_df['Type'])):
            train_val_df = feature_df.iloc[train_val_idx].reset_index(drop=True)
            test_df      = feature_df.iloc[test_idx].reset_index(drop=True)
            train_df, val_df = safe_split(train_val_df, args.val_size, args.seed + fold_idx)
            overall, per_type, samples = _run_fold(
                fold_idx + 1, train_df, val_df, test_df, feature_cols, args)
            all_overall.append(overall)
            all_per_type.extend(per_type)
            all_sample_preds.append(samples)
    else:
        for rep in range(args.repeats):
            seed_i = args.seed + rep
            train_val_df, test_df = safe_split(feature_df, args.test_size, seed_i)
            train_df, val_df = safe_split(train_val_df, args.val_size, seed_i)
            overall, per_type, samples = _run_fold(
                rep + 1, train_df, val_df, test_df, feature_cols, args)
            all_overall.append(overall)
            all_per_type.extend(per_type)
            all_sample_preds.append(samples)

    # Collect all predictions
    pd.concat(all_sample_preds, ignore_index=True).to_csv(
        os.path.join(args.output_dir, 'sample_predictions.csv'), index=False)
    overall_df = pd.DataFrame(all_overall)
    overall_df.to_csv(os.path.join(args.output_dir, 'fold_results.csv'), index=False)

    # Calculate summary statistics and print
    n_folds = len(overall_df)
    print(f"\n{'='*65}")
    print(f"Overall summary — {args.feature_set} ({len(feature_cols)} features), "
          f"{args.mode}  ({n_folds} folds/repeats)")
    print(f"{'='*65}")
    header = f"{'Metric':<12}  {'Mean':>8}  {'Std':>8}  {'95% CI':>22}"
    print(header)
    print('-' * len(header))

    # Build table with summary statistics
    summary_rows = []
    for metric in METRICS:
        values = overall_df[metric].dropna().values
        mean, std, lo, hi = ci(values)
        print(f"{metric:<12}  {mean:>8.4f}  {std:>8.4f}  [{lo:>8.4f}, {hi:>8.4f}]")
        summary_rows.append({'feature_set': args.feature_set, 'n_features': len(feature_cols),
                              'metric': metric, 'mean': mean, 'std': std,
                              'ci_lower': lo, 'ci_upper': hi})

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(args.output_dir, 'summary.csv'), index=False)

    per_type_raw = pd.DataFrame(all_per_type)
    per_type_raw.to_csv(os.path.join(args.output_dir, 'per_type_fold_results.csv'), index=False)
    type_summary = aggregate_subset(all_per_type, 'type')
    type_summary.to_csv(os.path.join(args.output_dir, 'per_type_summary.csv'), index=False)
    print_summary_table(f"Per-type summary (mean across {n_folds} folds)", type_summary, 'type')

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == '__main__':
    main()