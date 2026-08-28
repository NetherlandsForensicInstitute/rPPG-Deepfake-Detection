import argparse
import os

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model

from src.evaluation import (
    METRICS, aggregate_subset as _aggregate_subset, ci as _ci,
    compute_metrics as _compute_metrics, find_threshold as _find_threshold,
    print_summary_table as _print_summary_table, safe_split as _safe_split,
)
from src.model import build_model, reshape_bvp, set_seed
from train import load_and_label

def _select_train_source(rows_df, alt_df):
    """Replace rows_df's samples with matching rows from alt_df (joined on Filename).

    Samples in rows_df whose Filename is absent from alt_df are dropped.
    """
    if 'Filename' not in rows_df.columns or 'Filename' not in alt_df.columns:
        raise ValueError(
            "--train-data requires a 'Filename' column in both NDJSON files "
            "to match samples across folds."
        )
    matched = alt_df[alt_df['Filename'].isin(rows_df['Filename'])].reset_index(drop=True)
    missing = len(rows_df) - len(matched)
    if missing > 0:
        print(f"  [train-data] {missing}/{len(rows_df)} samples not found in "
              f"--train-data; using {len(matched)} for training.")
    return matched


def _prepare_train_val(train_val_df, alt_df, val_size, random_state):
    if alt_df is not None:
        train_val_df = _select_train_source(train_val_df, alt_df)
    return _safe_split(train_val_df, val_size, random_state)


def _run_fold(fold_name, train_df, val_df, test_df, args):
    """Train one fold. Returns (overall_dict, per_type_list, result_df)."""
    print(f"\n{'='*65}")
    print(f"Fold {fold_name}  —  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    print(f"{'='*65}")

    set_seed(args.seed + int(fold_name) - 1)

    x_train, mask_train = reshape_bvp(train_df['BVPS'], args.num_windows, args.num_frames)
    x_val,   mask_val   = reshape_bvp(val_df['BVPS'],   args.num_windows, args.num_frames)
    x_test,  mask_test  = reshape_bvp(test_df['BVPS'],  args.num_windows, args.num_frames)
    y_train = np.array(train_df['label'])
    y_val   = np.array(val_df['label'])
    y_test  = np.array(test_df['label'])

    if not args.no_class_weights:
        weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight = {int(c): float(w) for c, w in zip(np.unique(y_train), weights)}
    else:
        class_weight = None

    timesteps, num_frames, num_features = x_train.shape[1], x_train.shape[2], x_train.shape[3]
    model = build_model(
        timesteps, num_frames, num_features,
        args.lstm_layers, args.lstm_units,
        dropout=args.dropout, learning_rate=args.learning_rate,
    )

    ckpt_path = os.path.join(args.output_dir, f'best_fold_{fold_name}.keras')
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
        ModelCheckpoint(ckpt_path, monitor='val_loss', mode='min', save_best_only=True, verbose=0),
    ]
    if args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStopping(monitor='val_loss', mode='min', patience=args.early_stopping_patience,
                          min_delta=args.early_stopping_min_delta, restore_best_weights=True)
        )

    model.fit(
        (x_train, mask_train), y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=((x_val, mask_val), y_val),
        callbacks=callbacks,
        class_weight=class_weight,
        shuffle=True,
        verbose=1,
    )

    # ── Predictions ───────────────────────────────────────────────────────
    model = load_model(ckpt_path)
    os.remove(ckpt_path)
    y_val_prob = model.predict((x_val, mask_val), verbose=0).flatten()
    threshold = _find_threshold(y_val, y_val_prob, metric=args.threshold_metric)
    print(f"  Threshold ({args.threshold_metric} on val): {threshold:.2f}")
    y_prob = model.predict((x_test, mask_test), verbose=0).flatten()
    y_pred = (y_prob >= threshold).astype(int)

    overall = _compute_metrics(y_test, y_pred, y_prob)
    overall['fold'] = fold_name
    overall['threshold'] = threshold
    print("  Overall: " + "  ".join(f"{k}={v:.4f}" for k, v in overall.items() if k != 'fold'))

    # Attach predictions to test frame for subset breakdown
    base_cols = ['Filename', 'Type', 'label'] if 'Filename' in test_df.columns else ['Type', 'label']
    result = test_df[base_cols].copy().reset_index(drop=True)
    result['fold']   = fold_name
    result['y_pred'] = y_pred
    result['y_prob'] = y_prob

    # Per-type
    per_type = []
    for t, grp in result.groupby('Type'):
        m = _compute_metrics(grp['label'].values, grp['y_pred'].values, grp['y_prob'].values)
        per_type.append({'fold': fold_name, 'type': t, 'n': len(grp), **m})

    return overall, per_type, result


def main():
    """
    Train and evaluate the model with cross-validation or repeated random splits..

    Supports two strategies:
      k-fold    Stratified K-Fold (default k=5). Each fold uses 1/k samples as the
                test set; the remainder is split into train/val for early stopping.
      repeated  Repeated random train/val/test splits with different seeds.

    Per-fold metrics (accuracy, precision, recall, F1, AUC) are aggregated into
    mean ± std and 95% confidence intervals (t-distribution).  A per-type
    breakdown is included in the output.

    Usage
    -----
    # 5-fold CV with defaults:
    python cv_train.py --data signals.ndjson

    # 10-fold CV:
    python cv_train.py --data signals.ndjson --mode k-fold --folds 10

    # 20 repeated random splits:
    python cv_train.py --data signals.ndjson --mode repeated --repeats 20

    # Train on filtered signals, evaluate on the corresponding unfiltered test folds:
    python cv_train.py --data signals.ndjson --train-data signals_filtered.ndjson
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=str, required=True,
                        help='Path to the NDJSON signals file.')
    parser.add_argument('--output-dir', type=str, default='results_cv',
                        help='Directory for results (default: results_cv/).')
    parser.add_argument('--mode', choices=['k-fold', 'repeated'], default='repeated',
                        help='CV strategy (default: repeated).')
    # Help text default corrected to match the actual argparse default (5 folds = 20% test)
    parser.add_argument('--folds', type=int, default=5,
                        help='Number of folds for k-fold CV (default: 5, giving 20%% test per fold).')
    parser.add_argument('--repeats', type=int, default=10,
                        help='Number of repeats for repeated random splits (default: 10).')
    parser.add_argument('--test-size', type=float, default=0.23,
                        help='Test fraction for repeated splits (default: 0.23).')
    parser.add_argument('--val-size', type=float, default=12 / (65 + 12),
                        help='Validation fraction of the non-test remainder (default: 12/77, '
                             '~0.1558).')
    # Model / training
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lstm-units', type=int, default=32)
    parser.add_argument('--lstm-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--num-frames', type=int, default=180)
    parser.add_argument('--num-windows', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--real-types', type=str, required=True,
                        help='Comma-separated Type values treated as real (label=0).')
    parser.add_argument('--train-data', type=str, default=None,
                        help="Optional NDJSON file supplying alternate train/val signals "
                             "(e.g. frequency-filtered BVPs). Fold and test-set assignment "
                             "is still driven by --data; for each fold's train/val portion, "
                             "matching samples (joined on 'Filename') are pulled from this "
                             "file instead. Samples missing from this file are dropped from "
                             "training. Test folds always use --data.")
    parser.add_argument('--early-stopping-patience', type=int, default=10)
    parser.add_argument('--early-stopping-min-delta', type=float, default=0.0)
    parser.add_argument('--no-class-weights', action='store_true')
    parser.add_argument('--threshold-metric', choices=['f1', 'recall', 'accuracy'], default='f1',
                        help='Metric to maximise when selecting the decision threshold on the '
                             'validation set (default: f1).')

    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    real_types = [t.strip() for t in args.real_types.split(',')]
    df = load_and_label(args.data, real_types)
    df = df.reset_index(drop=True)
    df['label'] = df['label'].astype(int)

    print(f"Loaded {len(df)} samples  "
          f"(real={int((df['label']==0).sum())}  fake={int((df['label']==1).sum())})")

    train_alt_df = None
    if args.train_data:
        train_alt_df = load_and_label(args.train_data, real_types)
        train_alt_df = train_alt_df.reset_index(drop=True)
        train_alt_df['label'] = train_alt_df['label'].astype(int)
        print(f"Using {args.train_data} for train/val signals "
              f"({len(train_alt_df)} samples available).")

    all_overall      = []
    all_per_type     = []
    all_sample_preds = []

    if args.mode == 'k-fold':
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        # val_size is treated as a fraction of the train_val portion (matching train.py)
        rel_val = args.val_size

        for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(df, df['Type'])):
            train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
            test_df = df.iloc[test_idx].reset_index(drop=True)
            train_df, val_df = _prepare_train_val(train_val_df, train_alt_df, rel_val, args.seed + fold_idx)
            overall, per_type, samples = _run_fold(fold_idx + 1, train_df, val_df, test_df, args)
            all_overall.append(overall)
            all_per_type.extend(per_type)
            all_sample_preds.append(samples)

    else:  # repeated random splits — two sequential splits matching train.py exactly
        for rep in range(args.repeats):
            seed_i = args.seed + rep
            train_val_df, test_df = _safe_split(df, args.test_size, seed_i)
            train_df, val_df = _prepare_train_val(train_val_df, train_alt_df, args.val_size, seed_i)
            overall, per_type, samples = _run_fold(rep + 1, train_df, val_df, test_df, args)
            all_overall.append(overall)
            all_per_type.extend(per_type)
            all_sample_preds.append(samples)

    # ── Sample-level predictions (for annotation-based analysis) ──────
    pd.concat(all_sample_preds, ignore_index=True).to_csv(
        os.path.join(args.output_dir, 'sample_predictions.csv'), index=False)

    # ── Overall summary ────────────────────────────────────────────────
    overall_df = pd.DataFrame(all_overall)
    overall_df.to_csv(os.path.join(args.output_dir, 'fold_results.csv'), index=False)

    n_folds = len(overall_df)
    print(f"\n{'='*65}")
    print(f"Overall summary — {args.mode}  ({n_folds} folds/repeats)")
    print(f"{'='*65}")
    header = f"{'Metric':<12}  {'Mean':>8}  {'Std':>8}  {'95% CI':>22}"
    print(header)
    print('-' * len(header))

    summary_rows = []
    for metric in METRICS:
        values = overall_df[metric].dropna().values
        mean, std, lo, hi = _ci(values)
        print(f"{metric:<12}  {mean:>8.4f}  {std:>8.4f}  [{lo:>8.4f}, {hi:>8.4f}]")
        summary_rows.append({'metric': metric, 'mean': mean, 'std': std,
                              'ci_lower': lo, 'ci_upper': hi})

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(args.output_dir, 'summary.csv'), index=False)

    # ── Per-type summary ───────────────────────────────────────────────
    per_type_raw = pd.DataFrame(all_per_type)
    per_type_raw.to_csv(os.path.join(args.output_dir, 'per_type_fold_results.csv'), index=False)

    type_summary = _aggregate_subset(all_per_type, 'type')
    type_summary.to_csv(os.path.join(args.output_dir, 'per_type_summary.csv'), index=False)
    _print_summary_table(
        f"Per-type summary (mean across {n_folds} folds)", type_summary, 'type')

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == '__main__':
    main()