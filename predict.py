import argparse
import os

from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from train import load_and_label, reshape_bvp


def _subset_metrics(df: pd.DataFrame) -> dict:
    y_true = df['true_label'].values
    y_pred = df['predicted_label'].values
    y_prob = df['score'].values
    only_one_class = len(np.unique(y_true)) < 2
    return {
        'n':         len(df),
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall':    recall_score(y_true, y_pred, average='binary', zero_division=0),
        'f1':        f1_score(y_true, y_pred, average='binary', zero_division=0),
        'auc':       roc_auc_score(y_true, y_prob) if not only_one_class else float('nan'),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the trained .keras model file.')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to the NDJSON signals file to run predictions on.')
    parser.add_argument('--output-dir', type=str, default='predictions',
                        help='Directory to save predictions and plots (default: predictions/).')
    parser.add_argument('--num-frames', type=int, default=180,
                        help='Number of frames per BVP window (must match the trained model).')
    parser.add_argument('--num-windows', type=int, default=10,
                        help='Number of temporal windows per sample (must match the trained model).')
    parser.add_argument('--evaluate', action='store_true',
                        help='Compute classification metrics using ground-truth labels from the data file.')
    parser.add_argument('--real-types', type=str, default=None,
                        help='Comma-separated Type values that are real (label=0). '
                             'Required when --evaluate is set.')
    args = parser.parse_args()

    if args.evaluate and not args.real_types:
        parser.error('--real-types is required when --evaluate is set.')

    os.makedirs(args.output_dir, exist_ok=True)

    real_types = [t.strip() for t in args.real_types.split(',')] if args.real_types else None

    # Load model
    print(f"Loading model from {args.model} ...")
    model = tf.keras.models.load_model(args.model, compile=False)
    model.summary()

    # Load data
    if args.evaluate:
        df = load_and_label(args.data, real_types)
    else:
        df = pd.read_json(args.data, lines=True)

    x, mask = reshape_bvp(df['BVPS'], args.num_windows, args.num_frames)
    print(f"Input shape: {x.shape}")

    # Run inference
    output = model.predict((x, mask), verbose=1)
    scores = output.flatten()
    predicted_labels = (output > 0.5).astype(int).flatten()

    # Save predictions CSV
    result_df = df[['Filename', 'Type']].copy() if 'Filename' in df.columns else df[['Type']].copy()
    result_df['predicted_label'] = predicted_labels
    result_df['score'] = scores
    if args.evaluate and 'label' in df.columns:
        result_df['true_label'] = df['label'].values

    out_csv = os.path.join(args.output_dir, 'predictions.csv')
    result_df.to_csv(out_csv, index=False)
    print(f"Predictions saved to {out_csv}")

    # Score distribution plot
    plt.figure()
    sns.histplot(scores, bins=40, kde=True)
    plt.xlabel('Predicted fake probability')
    plt.title('Score distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'score_distribution.jpg'))
    plt.close()

    if not args.evaluate:
        return

    # Evaluation
    y_true = df['label'].values
    print("\n" + classification_report(y_true, predicted_labels,
                                       target_names=['Real', 'Fake']))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, predicted_labels))

    only_one_class = len(np.unique(y_true)) < 2
    auc = None
    if not only_one_class:
        auc = roc_auc_score(y_true, scores)
        print(f"AUC: {auc:.4f}")

        fpr, tpr, _ = roc_curve(y_true, scores)
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'roc_curve.jpg'))
        plt.close()

    # Per-type breakdown (full metrics)
    per_type_rows = [{'type': t} | _subset_metrics(grp)
                     for t, grp in result_df.groupby('Type')]
    per_type_df = pd.DataFrame(per_type_rows)
    print("\nPer-type metrics:")
    print(per_type_df.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    per_type_df.to_csv(os.path.join(args.output_dir, 'per_type_metrics.csv'), index=False)


if __name__ == '__main__':
    main()