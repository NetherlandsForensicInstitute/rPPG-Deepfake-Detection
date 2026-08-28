import argparse
import os
from functools import partial

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model

from src.evaluation import METRICS, bootstrap_ci as _bootstrap_ci, ci as _ci
from src.evaluation import compute_metrics as _compute_metrics
from src.evaluation import find_threshold as _find_threshold, safe_split as _safe_split_base
from src.model import build_model, reshape_bvp, set_seed
from train import load_and_label

_safe_split = partial(_safe_split_base, stratify_col='label')


def main():
    """
    Runs a leave-one-generator-out (LOGO) evaluation.

    For each deepfake generator G:
      - Train on real samples + all fake samples from every type EXCEPT G.
      - Evaluate on a held-out test set that contains G's fakes and real samples.

    Real samples are split once (fixed across all folds) so per-fold results are
    directly comparable.  Per-fold results and a summary with 95% CIs are saved.
    The 95% CIs are computed using bootstrap resampling.

    Usage
    -----
    python loo_generalization.py --data signals.ndjson

    # Custom real-type labels:
    python loo_generalization.py --data signals.ndjson --real-types youtube
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=str, required=True,
                        help='Path to the NDJSON signals file.')
    parser.add_argument('--output-dir', type=str, default='results_loo',
                        help='Directory for results (default: results_loo/).')
    parser.add_argument('--real-test-size', type=float, default=0.15,
                        help='Fraction of real samples reserved for the test set (default: 0.15).')
    parser.add_argument('--real-val-size', type=float, default=0.15,
                        help='Fraction of real samples used for validation (default: 0.15).')
    parser.add_argument('--fake-val-size', type=float, default=0.15,
                        help='Fraction of seen-fake samples used for validation (default: 0.15).')
    parser.add_argument('--min-samples', type=int, default=5,
                        help='Skip a generator type if it has fewer than this many samples.')
    # Model / training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lstm-units', type=int, default=32)
    parser.add_argument('--lstm-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--num-frames', type=int, default=180)
    parser.add_argument('--num-windows', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--real-types', type=str, required=True,
                        help='Comma-separated Type values treated as real (label=0).')
    parser.add_argument('--early-stopping-patience', type=int, default=10)
    parser.add_argument('--early-stopping-min-delta', type=float, default=0.0)
    parser.add_argument('--no-class-weights', action='store_true')
    parser.add_argument('--threshold-metric', choices=['f1', 'recall', 'accuracy'], default='f1',
                        help='Metric to maximise when selecting the decision threshold on the '
                             'validation set (default: f1).')
    parser.add_argument('--save-models', action='store_true',
                        help='Save the trained .keras model for each fold.')
    parser.add_argument('--n-boot', type=int, default=1000,
                        help='Bootstrap resamples for the per-fold 95%% CI (default: 1000). '
                             'Each LOO fold trains once, so this CI resamples that fold\'s '
                             'test set rather than aggregating repeated runs.')
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    real_types_list = [t.strip() for t in args.real_types.split(',')]
    df = load_and_label(args.data, real_types_list)
    df = df.reset_index(drop=True)
    df['label'] = df['label'].astype(int)

    real_df = df[df['label'] == 0].copy()
    fake_df = df[df['label'] == 1].copy()
    fake_types = sorted(fake_df['Type'].unique())

    print(f"Real samples : {len(real_df)}")
    print(f"Fake samples : {len(fake_df)}")
    print(f"Fake types   : {fake_types}\n")

    # Split real samples once — held fixed across all folds.
    real_trainval, real_test = train_test_split(
        real_df, test_size=args.real_test_size, random_state=args.seed,
        stratify=real_df['Type'] if real_df['Type'].nunique() > 1 else None,
    )
    rel_val = args.real_val_size / (1.0 - args.real_test_size)
    real_train, real_val = train_test_split(
        real_trainval, test_size=rel_val, random_state=args.seed,
    )
    print(f"Real split — train: {len(real_train)}  val: {len(real_val)}  test: {len(real_test)}\n")

    fold_results = []
    sample_results = []

    for fold_idx, left_out_type in enumerate(fake_types):
        left_out = fake_df[fake_df['Type'] == left_out_type]
        seen_fakes = fake_df[fake_df['Type'] != left_out_type]

        if len(left_out) < args.min_samples:
            print(f"Skipping '{left_out_type}': only {len(left_out)} samples.")
            continue

        print(f"\n{'='*65}")
        print(f"Fold {fold_idx+1}/{len(fake_types)} — left out: '{left_out_type}' "
              f"({len(left_out)} samples)")
        print(f"  Seen fakes: {len(seen_fakes)} from "
              f"{sorted(seen_fakes['Type'].unique())}")
        print(f"{'='*65}")

        # Split seen fakes into train / val
        fake_train, fake_val = _safe_split(seen_fakes, args.fake_val_size, args.seed + fold_idx)

        train_set = pd.concat([real_train, fake_train], ignore_index=True)
        val_set   = pd.concat([real_val,   fake_val],   ignore_index=True)
        # Test: held-out real + the left-out generator's samples
        test_set  = pd.concat([real_test,  left_out],   ignore_index=True)

        print(f"  train={len(train_set)}  val={len(val_set)}  test={len(test_set)}")

        set_seed(args.seed + fold_idx)

        x_train, mask_train = reshape_bvp(train_set['BVPS'], args.num_windows, args.num_frames)
        x_val,   mask_val   = reshape_bvp(val_set['BVPS'],   args.num_windows, args.num_frames)
        x_test,  mask_test  = reshape_bvp(test_set['BVPS'],  args.num_windows, args.num_frames)
        y_train = np.array(train_set['label'])
        y_val   = np.array(val_set['label'])
        y_test  = np.array(test_set['label'])

        if not args.no_class_weights:
            weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            class_weight = {int(c): float(w) for c, w in zip(np.unique(y_train), weights)}
        else:
            class_weight = None

        timesteps, num_frames, num_features = (
            x_train.shape[1], x_train.shape[2], x_train.shape[3]
        )
        model = build_model(
            timesteps, num_frames, num_features,
            args.lstm_layers, args.lstm_units,
            dropout=args.dropout, learning_rate=args.learning_rate,
        )

        ckpt_path = os.path.join(args.output_dir, f'_ckpt_{left_out_type}.keras')
        callbacks = [
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
            ModelCheckpoint(ckpt_path, monitor='val_loss', mode='min',
                            save_best_only=True, verbose=0),
        ]
        if args.early_stopping_patience > 0:
            callbacks.append(
                EarlyStopping(monitor='val_loss', patience=args.early_stopping_patience,
                              min_delta=args.early_stopping_min_delta,
                              restore_best_weights=True)
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

        model = load_model(ckpt_path)
        os.remove(ckpt_path)

        if args.save_models:
            model_path = os.path.join(args.output_dir, f'model_loo_{left_out_type}.keras')
            model.save(model_path)

        # Evaluate
        y_val_prob = model.predict((x_val, mask_val), verbose=0).flatten()
        threshold = _find_threshold(y_val, y_val_prob, metric=args.threshold_metric)
        print(f"  Threshold ({args.threshold_metric} on val): {threshold:.2f}")
        y_prob = model.predict((x_test, mask_test), verbose=0).flatten()
        y_pred = (y_prob >= threshold).astype(int)

        overall = _compute_metrics(y_test, y_pred, y_prob)

        # Each LOO fold trains once, so bootstrap the fold's own test set for a CI
        # instead of aggregating across (nonexistent) repeats of this fold
        fold_ci = _bootstrap_ci(y_test, y_pred, y_prob, n_boot=args.n_boot,
                                seed=args.seed + fold_idx)

        # Per-type breakdown on the test set
        test_set = test_set.copy()
        test_set['y_pred'] = y_pred
        test_set['y_prob'] = y_prob

        print(f"\n  {'Type':<30} {'n':>5} {'Acc':>6} {'F1':>6} {'AUC':>6}")
        print(f"  {'-'*30} {'-'*5} {'-'*6} {'-'*6} {'-'*6}")

        for t, grp in test_set.groupby('Type'):
            yt = grp['label'].values
            yp = grp['y_pred'].values
            yb = grp['y_prob'].values
            only_one = len(np.unique(yt)) < 2
            acc = accuracy_score(yt, yp)
            f1  = f1_score(yt, yp, average='binary', zero_division=0)
            auc = roc_auc_score(yt, yb) if not only_one else float('nan')
            tag = ' ← left out' if t == left_out_type else ''
            print(f"  {t:<30} {len(grp):>5} {acc:>6.3f} {f1:>6.3f} {auc:>6.3f}{tag}")
            sample_results.append({
                'left_out_type': left_out_type,
                'test_type':     t,
                'is_left_out':   t == left_out_type,
                'n':             len(grp),
                'accuracy': acc, 'f1': f1, 'auc': auc,
            })

        print(f"\n  Overall (n={len(test_set)}) — " + "  ".join(
            f"{k}={v:.4f} [{fold_ci[k][0]:.4f}, {fold_ci[k][1]:.4f}]"
            for k, v in overall.items()
        ))

        # Fold row: point estimates + bootstrap CI + sample counts
        row = {
            'left_out_type': left_out_type,
            'threshold':     threshold,
            'n_test':        len(test_set),
            'n_real_test':   len(real_test),
            'n_left_out':    len(left_out),
            **overall,
        }
        for m in METRICS:
            row[f'{m}_ci_lower'], row[f'{m}_ci_upper'] = fold_ci[m]
        fold_results.append(row)

    # Print summary
    if not fold_results:
        print("No folds completed.")
        return

    fold_df   = pd.DataFrame(fold_results)
    sample_df = pd.DataFrame(sample_results)

    fold_df.to_csv(os.path.join(args.output_dir, 'fold_results.csv'), index=False)
    sample_df.to_csv(os.path.join(args.output_dir, 'per_type_results.csv'), index=False)

    n = len(fold_df)
    print(f"\n{'='*65}")
    print(f"Summary — LOGO evaluation  ({n} folds)")
    print(f"{'='*65}")
    header = f"{'Metric':<12}  {'Mean':>8}  {'Std':>8}  {'95% CI':>22}"
    print(header)
    print('-' * len(header))

    # Total test samples across folds, for context alongside the fold count above
    total_test_samples = int(fold_df['n_test'].sum())
    print(f"({total_test_samples} total test samples across all folds)")

    summary_rows = []
    for metric in METRICS:
        values = fold_df[metric].dropna().values
        mean, std, lo, hi = _ci(values)
        print(f"{metric:<12}  {mean:>8.4f}  {std:>8.4f}  [{lo:>8.4f}, {hi:>8.4f}]")
        summary_rows.append({'metric': metric, 'mean': mean, 'std': std,
                              'ci_lower': lo, 'ci_upper': hi,
                              'n_folds': n, 'total_test_samples': total_test_samples})

    # Per-generator pivot: point estimate + bootstrap CI + sample counts.
    # CI here is per-fold (bootstrapped over that fold's own test set), not a
    # between-fold CI -- each generator is only left out once.
    ci_cols = [f'{m}_ci_lower' for m in METRICS] + [f'{m}_ci_upper' for m in METRICS]
    left_out_summary = fold_df[
        ['left_out_type', 'n_test', 'n_left_out'] + METRICS + ci_cols
    ].set_index('left_out_type')

    # Compact "mean [lo, hi]" display copy for the printed table (CSV keeps raw columns)
    display_summary = left_out_summary[['n_test', 'n_left_out']].copy()
    for m in METRICS:
        display_summary[m] = [
            f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"
            for mean, lo, hi in zip(left_out_summary[m],
                                    left_out_summary[f'{m}_ci_lower'],
                                    left_out_summary[f'{m}_ci_upper'])
        ]

    print(f"\n{'='*65}")
    print("Per-generator LOGO performance (model never trained on this type)")
    print("Each CI is bootstrapped over that fold's own test set (n_test).")
    print(f"{'='*65}")
    print(display_summary.to_string())

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(args.output_dir, 'summary.csv'), index=False)
    left_out_summary.to_csv(os.path.join(args.output_dir, 'per_generator_summary.csv'))

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == '__main__':
    main()