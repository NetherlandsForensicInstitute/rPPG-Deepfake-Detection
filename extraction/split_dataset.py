import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sklearn.model_selection import train_test_split

from train import load_and_label, set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=str, required=True,
                        help='Path to the NDJSON signals file to split.')
    parser.add_argument('--output-dir', type=str, default='splits',
                        help='Directory to write the output NDJSON files (default: splits/).')
    # Defaults match train.py's 65% train / 12% val / 23% test split.
    parser.add_argument('--test-size', type=float, default=0.23,
                        help='Fraction of samples for the test set (default: 0.23, matching train.py).')
    parser.add_argument('--split-val', action='store_true',
                        help='Also produce a separate val.ndjson. '
                             'Without this flag only train.ndjson and test.ndjson are written. '
                             'When train.py receives --eval-data it handles the val split internally.')
    parser.add_argument('--val-size', type=float, default=12 / (65 + 12),
                        help='Fraction of the non-test samples to use for validation when '
                             '--split-val is set (default: 12/77, ~0.1558, matching train.py).')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default: 0, matching train.py).')
    parser.add_argument('--real-types', type=str, required=True,
                        help='Comma-separated Type values that are real (label=0).')
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    real_types = [t.strip() for t in args.real_types.split(',')]
    df = load_and_label(args.data, real_types)
    print(f"Loaded {len(df)} samples  (label distribution: {df['label'].value_counts().to_dict()})")

    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df['Type'],
    )

    if args.split_val:
        train_df, val_df = train_test_split(
            train_df,
            test_size=args.val_size,
            random_state=args.seed,
            stratify=train_df['Type'],
        )

    basename = os.path.splitext(os.path.basename(args.data))[0]

    def save(split_df, name):
        path = os.path.join(args.output_dir, f'{basename}_{name}.ndjson')
        split_df.to_json(path, orient='records', lines=True)
        label_dist = split_df['label'].value_counts().to_dict()
        print(f"  {name:6s}  {len(split_df):>6} samples  labels={label_dist}  →  {path}")

    print(f"\nSplit summary (seed={args.seed}, test_size={args.test_size}):")
    save(train_df, 'train')
    if args.split_val:
        save(val_df, 'val')
    save(test_df, 'test')
    print()


if __name__ == '__main__':
    main()
