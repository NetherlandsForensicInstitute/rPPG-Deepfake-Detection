import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from extraction.load_dataset import load_dataset


def get_duration(path: str) -> float | None:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps <= 0:
        return None
    return frames / fps


def main():
    parser = argparse.ArgumentParser(
        description="Plot video duration distribution for a FileVideoDataset."
    )
    parser.add_argument("--dataset", required=True, help="Dataset name (passed to load_dataset).")
    parser.add_argument("--dataset-dir", required=True, help="Root directory of the dataset.")
    parser.add_argument("--output", default=None, help="Path to save the plot (default: show interactively).")
    parser.add_argument("--bins", type=int, default=30)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, args.dataset_dir)

    records = []
    failed = []
    for instance in tqdm(dataset, desc="Reading video metadata"):
        d = get_duration(instance.path)
        if d is not None:
            records.append({'subfolder': instance.path.parent.name, 'duration': d})
        else:
            failed.append(instance.path)

    if failed:
        print(f"\nWarning: could not read {len(failed)} video(s).")

    import pandas as pd
    df = pd.DataFrame(records)
    durations = df['duration'].values
    print(f"\nVideos read: {len(durations)}")
    print(f"Min:    {durations.min():.2f}s")
    print(f"Max:    {durations.max():.2f}s")
    print(f"Mean:   {durations.mean():.2f}s")
    print(f"Median: {np.median(durations):.2f}s")
    print(f"Std:    {durations.std():.2f}s")

    print("\nAverage duration per subfolder:")
    per_folder = (
        df.groupby('subfolder')['duration']
        .agg(count='count', mean='mean', median='median')
        .sort_values('mean', ascending=False)
    )
    print(per_folder.to_string(float_format=lambda x: f'{x:.2f}'))

    plt.figure(figsize=(10, 5))
    plt.hist(durations, bins=args.bins, edgecolor="black")
    plt.xlabel("Duration (s)")
    plt.ylabel("Count")
    plt.title(f"Video duration distribution — {args.dataset} (n={len(durations)})")
    plt.tight_layout()

    if args.output:
        plt.savefig(args.output)
        print(f"Plot saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
