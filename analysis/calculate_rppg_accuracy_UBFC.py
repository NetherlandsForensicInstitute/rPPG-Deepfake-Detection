import argparse
import logging
import os

import pandas as pd
from tqdm import tqdm

from src.utils import (
    construct_rppg_df, df_from_ubfc_ds1, df_from_ubfc_ds2,
    merge_2_df_rppg_timestamp, plot_signals, stats_on_merged_df,
)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate rPPG accuracy on the UBFC-rPPG benchmark dataset.'
    )
    parser.add_argument('--signals-dir', type=str, required=True,
                        help='Directory containing extracted UBFC rPPG signal NDJSON files.')
    parser.add_argument('--dataset-dir', type=str, required=True,
                        help='Root directory of the UBFC dataset (DATASET_1 or DATASET_2).')
    parser.add_argument('--dataset-name', type=str, required=True, choices=['UBFC1', 'UBFC2'],
                        help='Dataset variant.')
    parser.add_argument('--plots-dir', type=str, default=None,
                        help='Directory to save per-participant signal plots (optional).')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path (default: <dataset_name>_stats.csv).')
    args = parser.parse_args()

    output_path = args.output or f"{args.dataset_name}_stats.csv"
    signal_files = [
        f for f in os.listdir(args.signals_dir)
        if args.dataset_name in f and f.endswith('.json')
    ]
    results_df = pd.DataFrame()

    for signals_file in tqdm(signal_files):
        method_name = signals_file.split('_')[-1].split('.')[0]
        rppg = pd.read_json(os.path.join(args.signals_dir, signals_file), lines=True)

        if rppg.empty:
            logging.warning(f"No signals extracted for method '{method_name}'; skipping.")
            continue

        for participant in os.listdir(args.dataset_dir):
            participant_path = os.path.join(args.dataset_dir, participant)
            rppg_person = rppg[rppg['Type'] == participant]
            if rppg_person.empty:
                continue
            rppg_person = construct_rppg_df(rppg_person.iloc[0], "")

            logging.info(f"Processing {method_name}, {participant}")

            if args.dataset_name == 'UBFC1':
                ground_truth = df_from_ubfc_ds1(os.path.join(participant_path, 'gtdump.xmp'))
            else:
                ground_truth = df_from_ubfc_ds2(os.path.join(participant_path, 'ground_truth.txt'))

            merged_df = merge_2_df_rppg_timestamp(ground_truth, rppg_person)

            if args.plots_dir:
                plot_signals(args.plots_dir, merged_df, args.dataset_name, method_name, participant)

            stats_df = stats_on_merged_df(
                merged_df,
                rppg_col_name='BPM_rPPG ',
                ground_truth_cols=['Ground truth BPM'],
            )
            stats_df['Extraction Method'] = method_name
            stats_df['Participant'] = participant
            results_df = pd.concat([results_df, stats_df], ignore_index=True)

    results_df.to_csv(output_path, index=False)
    logging.info(f"Saved results to {output_path}")


if __name__ == '__main__':
    main()