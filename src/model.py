import os
import random
from typing import Tuple

import keras_tuner as kt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    BatchNormalization, Bidirectional, Conv1D, Dense, Dropout,
    GlobalAveragePooling1D, LSTM, SpatialDropout1D, TimeDistributed,
)
from tensorflow.keras.metrics import Recall
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class MetricsCallback(tf.keras.callbacks.Callback):
    """
    Callback to compute and log precision, recall, and F1 score during training.
    """
    def __init__(self, x_val, y_val, freq: int = 1):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.freq = freq
        self.precision = []
        self.recall = []
        self.f1 = []

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.freq != 0:
            return
        logs = logs or {}
        val_ds = tf.data.Dataset.from_tensor_slices(self.x_val).batch(256)
        y_pred_probs = self.model.predict(val_ds, verbose=0)
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()
        prec = precision_score(self.y_val, y_pred, average='binary', zero_division=0)
        rec = recall_score(self.y_val, y_pred, average='binary', zero_division=0)
        f1 = f1_score(self.y_val, y_pred, average='binary', zero_division=0)
        self.precision.append(prec)
        self.recall.append(rec)
        self.f1.append(f1)
        print(f"Epoch {epoch + 1}: Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")


def set_seed(seed: int):
    """Set random seeds for Python, NumPy, and TensorFlow."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def reshape_bvp(bvp_series, num_windows, num_frames) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reshape BVP series to (n_samples, num_windows, num_frames, num_patches),
    plus a boolean window-validity mask of shape (n_samples, num_windows).

    Windows and frames are truncated/padded to num_windows and num_frames respectively.
    Windows added by padding (i.e. samples with fewer than num_windows real windows) are
    marked False in the mask, so the model can be told to ignore them (see build_model).

    Arguments:
        bvp_series: A list of BVP samples, each of which is a list of BVP patches of shape [#windows, #patches, #frames].
        num_windows: Number of windows each sample is truncated/padded to.
        num_frames: Number of frames per window each sample is truncated/padded to.

    Returns:
        x: A numpy array of shape (n_samples, num_windows, num_frames, num_patches).
        mask: A numpy array of shape (n_samples, num_windows) containing boolean values indicating whether each window
              is should be used by the model during training to update weights.

    """
    def format_sample(bvp_sample):
        array = np.array(bvp_sample)[0:num_windows, :, 0:num_frames]
        n_real_windows = array.shape[0]
        pad_width = [
            (0, max(0, num_windows - array.shape[0])),
            (0, 0),
            (0, max(0, num_frames - array.shape[2])),
        ]
        padded = np.pad(array, pad_width, mode='constant')
        mask = np.zeros(num_windows, dtype=bool)
        mask[:n_real_windows] = True
        return np.transpose(padded, (0, 2, 1)), mask  # (num_windows, num_frames, num_patches), (num_windows,)

    formatted = [format_sample(sample) for sample in bvp_series]
    x = np.stack([f[0] for f in formatted])
    mask = np.stack([f[1] for f in formatted])
    return x, mask


def build_model(timesteps, num_frames, num_features, lstm_layers, lstm_units, dropout, learning_rate) -> Model:
    """
    Function that builds the model.

    Arguments:
        timesteps: The number of timesteps in the input sequence.
        num_frames: The number of frames in each timestep.
        num_features: The number of features in each frame.
        lstm_layers: The number of LSTM layers in the model.
        lstm_units: The number of LSTM units in each layer.
        dropout: The dropout rate for the model.
        learning_rate: The learning rate for the model.

    Returns the model.
    """
    # Input layers
    inputs = Input(shape=(timesteps, num_frames, num_features), name='bvp')
    mask_input = Input(shape=(timesteps,), dtype='bool', name='window_mask')

    # Convolutional layers across time
    x = TimeDistributed(Conv1D(filters=32, kernel_size=19, activation='relu', padding='same'))(inputs)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Dropout(dropout))(x)

    x = TimeDistributed(Conv1D(filters=64, kernel_size=7, activation='relu', padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Dropout(dropout))(x)

    x = TimeDistributed(GlobalAveragePooling1D())(x)

    # Prevents LSTMs from overfitting to individual feature channels
    x = SpatialDropout1D(dropout)(x)

    # The mask is passed explicitly to every LSTM layer
    for i in range(lstm_layers):
        return_sequences = (i < lstm_layers - 1)
        x = Bidirectional(LSTM(lstm_units, return_sequences=return_sequences))(x, mask=mask_input)
        x = Dropout(dropout)(x)

    # Linear output layer
    outputs = Dense(1, activation='sigmoid')(x)

    # Combine all layers into a model
    model = Model(inputs=[inputs, mask_input], outputs=outputs)

    # Use Adam optimizer with binary cross-entropy loss
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', Recall(name='recall')]
    )
    return model


def build_model_tuning(hp, timesteps, num_frames, num_features):
    """
    Function to build a model for hyperparameter tuning.

    Matches the paper's tuning scope: only the number of LSTM layers and the number
    of units per layer are searched; the conv architecture, dropout, and learning
    rate are held fixed at the same values as build_model().
    """
    lstm_layers = hp.Int('lstm_layers', min_value=2, max_value=10, step=1)
    lstm_units  = hp.Int('lstm_units',  min_value=32, max_value=128, step=32)
    dropout       = 0.1
    learning_rate = 1e-4

    inputs = Input(shape=(timesteps, num_frames, num_features), name='bvp')
    mask_input = Input(shape=(timesteps,), dtype='bool', name='window_mask')

    x = TimeDistributed(Conv1D(filters=32, kernel_size=19, activation='relu', padding='same'))(inputs)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Dropout(dropout))(x)

    x = TimeDistributed(Conv1D(filters=64, kernel_size=7, activation='relu', padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Dropout(dropout))(x)

    x = TimeDistributed(GlobalAveragePooling1D())(x)
    x = SpatialDropout1D(dropout)(x)

    for i in range(lstm_layers):
        x = Bidirectional(LSTM(lstm_units, return_sequences=(i < lstm_layers - 1)))(x, mask=mask_input)
        x = Dropout(dropout)(x)

    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=[inputs, mask_input], outputs=outputs)
    model.compile(loss='binary_crossentropy', metrics=['accuracy'],
                  optimizer=Adam(learning_rate=learning_rate))
    return model


def tune_model(x_train, mask_train, y_train, x_val, mask_val, y_val, timesteps, num_frames, num_features):
    """Run Keras Tuner random search and return the best model."""
    tuner = kt.RandomSearch(
        lambda hp: build_model_tuning(hp, timesteps, num_frames, num_features),
        objective='val_accuracy',
        max_trials=20,
        executions_per_trial=1,
        directory='tuning_results',
        project_name='lstm_tuning',
        overwrite=True,
    )
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    tuner.search(
        (x_train, mask_train), y_train,
        epochs=100,
        validation_data=((x_val, mask_val), y_val),
        batch_size=tuner.oracle.hyperparameters.Choice('batch_size', values=[32, 64, 128, 256]),
        callbacks=[early_stop],
    )
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Best lstm_layers:   {best_hps.get('lstm_layers')}")
    print(f"Best lstm_units:    {best_hps.get('lstm_units')}")
    print(f"Best batch_size:    {best_hps.get('batch_size')}")
    return tuner.hypermodel.build(best_hps)


def evaluate_results(subset_df) -> dict:
    """
    Evaluate the results of a subset of data.

    Arguments:
        subset_df: DataFrame containing true and predicted labels for evaluation.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    y_true = subset_df['label']
    y_pred = subset_df['predicted_label']
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='binary'),
        'Recall': recall_score(y_true, y_pred, average='binary'),
        'F1 Score': f1_score(y_true, y_pred, average='binary'),
        'Confusion Matrix': confusion_matrix(y_true, y_pred).tolist(),
    }