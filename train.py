import argparse
import os

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from keras_tuner.src.backend.io import tf
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.model import (
    MetricsCallback, build_model, evaluate_results,
    reshape_bvp, set_seed, tune_model,
)


# Types with no physiological pulse signal (static screen/mannequin/paper controls);
# these must not be counted as 'Real' living-person footage.
_NO_PULSE_TYPES = {'Screen', 'Mannequin', 'Paper'}

# 65% train / 12% val / 23% test split
_TEST_FRACTION = 0.23
_VAL_FRACTION_OF_TRAINVAL = 12 / (65 + 12)


def load_and_label(path: str, real_types: list) -> pd.DataFrame:
    """Load an NDJSON signals file and add binary deepfake labels.

    Any sample whose Type is in real_types gets label 0 (real); every other
    sample gets label 1 (fake). No-pulse control types (see _NO_PULSE_TYPES)
    are always dropped, since they have no living person and can't count as
    either class.
    """
    # Read the NDJSON signals file into a DataFrame
    df = pd.read_json(path, lines=True)

    # Drop no-pulse controls before labelling
    df = df[~df['Type'].isin(_NO_PULSE_TYPES)].copy()

    # Caller lists which Type values are real; everything else is fake
    df['label'] = (~df['Type'].isin(real_types)).astype(int)

    # Drop any rows that ended up without a label
    df.dropna(subset=['label'], inplace=True)
    return df



def plot_training_history(history, metrics_callback, output_dir):
    h = history.history

    plt.figure()
    plt.plot(h['loss'], label='Training Loss')
    plt.plot(h['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_over_epochs.jpg'))
    plt.close()

    plt.figure()
    plt.plot(h['accuracy'], label='Training Accuracy')
    plt.plot(h['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'accuracy_over_epochs.jpg'))
    plt.close()

    plt.figure()
    plt.plot(metrics_callback.precision, label='Validation Precision')
    plt.plot(metrics_callback.recall, label='Validation Recall')
    plt.plot(metrics_callback.f1, label='Validation F1 Score')
    plt.title('Validation Precision, Recall, and F1 Score Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'precision_recall_f1_over_epochs.jpg'))
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Train an LSTM classifier on rPPG signals for deepfake detection.'
    )
    parser.add_argument('--data', type=str, required=True,
                        help='Path to the NDJSON file used for training (and test if --eval-data is omitted).')
    parser.add_argument('--eval-data', type=str, default=None,
                        help='Optional separate NDJSON file used as the test set. '
                             'When provided, --data is split into train/val only.')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save the model, plots, and metrics (default: results/).')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lstm-units', type=int, default=32)
    parser.add_argument('--lstm-layers', type=int, default=2)
    # Dropout default corrected to match the actual argparse default below
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate after each LSTM layer (default: 0.1).')

    # Learning-rate default corrected to match the actual argparse default below
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Initial Adam learning rate (default: 1e-4).')
    parser.add_argument('--num-frames', type=int, default=180,
                        help='Number of frames per BVP window to use as features.')
    parser.add_argument('--num-windows', type=int, default=10,
                        help='Number of temporal windows per sample.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--real-types', type=str, required=True,
                        help='Comma-separated list of Type values that are real (label=0). '
                             'All other types are treated as fake (label=1).')
    parser.add_argument('--tune', action='store_true',
                        help='Run Keras Tuner hyperparameter search instead of fixed architecture.')
    parser.add_argument('--early-stopping-patience', type=int, default=10,
                        help='Number of epochs with no improvement before stopping (default: 10). '
                             'Set to 0 to disable early stopping.')
    parser.add_argument('--early-stopping-min-delta', type=float, default=0.0,
                        help='Minimum change in val_loss to qualify as improvement (default: 0.0).')
    parser.add_argument('--metrics-freq', type=int, default=1,
                        help='Compute precision/recall/F1 every N epochs (default: every epoch).')
    parser.add_argument('--no-class-weights', action='store_true',
                        help='Disable automatic class weighting (weights are used by default to '
                             'handle class imbalance without discarding samples).')
    args = parser.parse_args()

    real_types = [t.strip() for t in args.real_types.split(',')]

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load and label training data
    df = load_and_label(args.data, real_types)

    if args.eval_data:
        # External test set: use all of --data for train/val
        train_set, val_set = train_test_split(
            df, random_state=args.seed, test_size=_VAL_FRACTION_OF_TRAINVAL, stratify=df['Type']
        )
        test_set = load_and_label(args.eval_data, real_types)
    else:
        # Single-file mode: three-way internal split
        train_set, test_set = train_test_split(
            df, random_state=args.seed, test_size=_TEST_FRACTION, stratify=df['Type']
        )
        train_set, val_set = train_test_split(
            train_set, random_state=args.seed, test_size=_VAL_FRACTION_OF_TRAINVAL, stratify=train_set['Type']
        )

    print(f"Train: {train_set.shape}, Val: {val_set.shape}, Test: {test_set.shape}")

    # Feature extraction
    x_train, mask_train = reshape_bvp(train_set['BVPS'], args.num_windows, args.num_frames)
    x_val,   mask_val   = reshape_bvp(val_set['BVPS'],   args.num_windows, args.num_frames)
    x_test,  mask_test  = reshape_bvp(test_set['BVPS'],  args.num_windows, args.num_frames)
    y_train = np.array(train_set['label'])
    y_val   = np.array(val_set['label'])
    y_test  = np.array(test_set['label'])

    if not args.no_class_weights:
        weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight = {int(c): float(w) for c, w in zip(np.unique(y_train), weights)}
        print(f"Class weights: {class_weight}")
    else:
        class_weight = None

    timesteps, num_frames, num_features = x_train.shape[1], x_train.shape[2], x_train.shape[3]

    # Build model
    if args.tune:
        model = tune_model(x_train, mask_train, y_train, x_val, mask_val, y_val,
                           timesteps, num_frames, num_features)
    else:
        model = build_model(timesteps, num_frames, num_features, args.lstm_layers, args.lstm_units,
                            dropout=args.dropout, learning_rate=args.learning_rate)

    print(model.summary())

    metrics_callback = MetricsCallback((x_val, mask_val), y_val, freq=args.metrics_freq)
    early_stop = (
        EarlyStopping(monitor='val_loss', patience=args.early_stopping_patience,
                      min_delta=args.early_stopping_min_delta, restore_best_weights=True)
        if args.early_stopping_patience > 0 else None
    )
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath='best_model.keras',  # Or whatever filepath your script loads at the end
        monitor='val_loss',
        save_best_only=True,
        mode='min'
    )

    history = model.fit(
        (x_train, mask_train), y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=((x_val, mask_val), y_val),
        callbacks=[c for c in [metrics_callback, early_stop, reduce_lr, checkpoint_cb] if c is not None],
        class_weight=class_weight,
        shuffle=True,
    )

    # Evaluation
    test_loss, test_acc = model.evaluate((x_test, mask_test), y_test)
    print(f"Test accuracy: {test_acc:.4f}")

    y_pred_probs_test = model.predict((x_test, mask_test))
    y_pred_test = (y_pred_probs_test > 0.5).astype(int).flatten()
    print(classification_report(y_test, y_pred_test))
    print(confusion_matrix(y_test, y_pred_test))

    # Attach predictions to each split (predict once per split)
    y_pred_probs_val   = model.predict((x_val, mask_val))
    y_pred_probs_train = model.predict((x_train, mask_train))

    test_set  = test_set.copy()
    val_set   = val_set.copy()
    train_set = train_set.copy()

    test_set['predicted_label']  = y_pred_test
    test_set['predicted_prob']   = y_pred_probs_test.flatten()
    val_set['predicted_label']   = (y_pred_probs_val > 0.5).astype(int).flatten()
    val_set['predicted_prob']    = y_pred_probs_val.flatten()
    train_set['predicted_label'] = (y_pred_probs_train > 0.5).astype(int).flatten()
    train_set['predicted_prob']  = y_pred_probs_train.flatten()

    test_set['set']  = 'test'
    val_set['set']   = 'val'
    train_set['set'] = 'train'

    predicted_labels = pd.concat([test_set, val_set, train_set], ignore_index=True)
    signal_cols = ['BVPS', 'BPMES', 'BPM', 'Uncertainty', 'timesES']
    save_cols = [c for c in predicted_labels.columns if c not in signal_cols]
    predicted_labels[save_cols].to_csv(os.path.join(args.output_dir, 'predicted_labels.csv'), index=False)

    final_metrics = evaluate_results(test_set)
    print(final_metrics)

    # Save training metrics
    h = history.history
    np.savez(
        os.path.join(args.output_dir, 'training_metrics.npz'),
        train_loss=h['loss'],
        val_loss=h['val_loss'],
        train_acc=h['accuracy'],
        val_acc=h['val_accuracy'],
        precision=metrics_callback.precision,
        recall=metrics_callback.recall,
        f1=metrics_callback.f1,
    )

    plot_training_history(history, metrics_callback, args.output_dir)


if __name__ == '__main__':
    main()