import argparse

import numpy as np
import pandas as pd
import tensorflow as tf

from src.model import reshape_bvp

parser = argparse.ArgumentParser(
    description='Find no-pulse control videos (Mannequin/Screen/Paper) the model misclassifies as real.'
)
parser.add_argument('--model', type=str, required=True, help='Path to trained Keras model.')
parser.add_argument('--data', type=str, required=True, help='Path to signals NDJSON file.')
args = parser.parse_args()

# Load model
model = tf.keras.models.load_model(args.model)

df = pd.read_json(args.data, lines=True)

df = df[df['Type'].isin(['Mannequin', 'Screen', 'Paper'])]

# Make predictions
predictions = []
labels = []

X, mask = reshape_bvp(df['BVPS'], num_windows=10, num_frames=180)
types = df['Type']
for i in range(len(X)):
    output = model((np.array([X[i]]), np.array([mask[i]])))
 
    # Single sigmoid score; threshold at 0.5 (argmax on a 1-unit output is always 0)
    predictions.append(int(output[0][0] > 0.5))

print(predictions)
print(types)
for i in df['Filename']:
    print(i)