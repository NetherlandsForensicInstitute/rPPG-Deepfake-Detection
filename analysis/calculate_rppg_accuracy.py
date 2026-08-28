import argparse
import logging
import os

import pandas as pd
from tqdm import tqdm

from src.utils import (
    construct_rppg_df, df_from_garmin_csv, df_from_movesense_json,
    merge_3_df_rppg_timestamp, merge_df_all_timestamps, plot_signals_nfi,
    stats_on_merged_df,
)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate rPPG accuracy against ground-truth HR sensors.'
    )
    parser.add_argument('--experiments-dir', type=str, required=True,
                        help='Root directory containing per-participant ground-truth data.')
    parser.add_argument('--signals-dir', type=str, required=True,
                        help='Directory containing extracted rPPG signal NDJSON files.')
    parser.add_argument('--output', type=str, default='gt_experiment_stats.csv',
                        help='Output CSV path for aggregated statistics.')
    parser.add_argument('--save-plots', action='store_true',
                        help='Save per-experiment signal comparison plots.')
    args = parser.parse_args()

    experiments = ['base', 'move', 'dark']
    participant_dirs = [f'Participant {i}' for i in range(1, 7)]
    results_df = pd.DataFrame()

    for signals_file in tqdm(os.listdir(args.signals_dir)):
        method_name = signals_file.split('_')[-1].split('.')[0]
        rppg = pd.read_json(os.path.join(args.signals_dir, signals_file), lines=True)

        for count, participant_dir in enumerate(participant_dirs):
            participant_path = os.path.join(args.experiments_dir, participant_dir)
            rppg_person = rppg[rppg['Type'] == participant_dir]
            participant_nr = str(count + 1)

            for experiment in experiments:
                logging.info(f"Processing {method_name}, participant {participant_nr}, {experiment}")

                movesense = df_from_movesense_json(
                    os.path.join(participant_path, f"{participant_nr}_{experiment}_HR.json")
                )
                garmin = df_from_garmin_csv(
                    os.path.join(participant_path, f"{participant_nr}_{experiment}_garmin.csv")
                )
                rppg_exp = rppg_person[
                    rppg_person['Filename'].str.contains(experiment, case=False, na=False)
                ]

                webcam_rows = rppg_exp[
                    rppg_exp['Filename'].str.contains('webcam', case=False, na=False)
                ].reset_index(drop=True)
                if webcam_rows.empty:
                    continue
                rppg_webcam = construct_rppg_df(webcam_rows.iloc[0], 'webcam')

                phone_rows = rppg_exp[
                    rppg_exp['Filename'].str.contains('phone', case=False, na=False)
                ].reset_index(drop=True)
                rppg_phone = (
                    construct_rppg_df(phone_rows.iloc[0], 'phone')
                    if not phone_rows.empty else pd.DataFrame()
                )

                if args.save_plots:
                    plotting_df = merge_df_all_timestamps(garmin, movesense, rppg_phone, rppg_webcam)
                    plot_signals_nfi(
                        args.experiments_dir, plotting_df, method_name,
                        participant_nr, experiment, phone=not rppg_phone.empty,
                    )

                if not rppg_phone.empty:
                    stats_df = stats_on_merged_df(
                        merge_3_df_rppg_timestamp(garmin, movesense, rppg_phone),
                        'BPM_rPPG phone', ['BPM_Garmin', 'BPM_Movesense'],
                    )
                    stats_df['Extraction Method'] = method_name
                    stats_df['Experiment'] = experiment
                    stats_df['Device'] = 'phone'
                    stats_df['Participant'] = participant_nr
                    results_df = pd.concat([results_df, stats_df], ignore_index=True)

                stats_df = stats_on_merged_df(
                    merge_3_df_rppg_timestamp(garmin, movesense, rppg_webcam),
                    'BPM_rPPG webcam', ['BPM_Garmin', 'BPM_Movesense'],
                )
                stats_df['Extraction Method'] = method_name
                stats_df['Experiment'] = experiment
                stats_df['Device'] = 'webcam'
                stats_df['Participant'] = participant_nr
                results_df = pd.concat([results_df, stats_df], ignore_index=True)

    results_df.to_csv(args.output, index=False)
    logging.info(f"Saved results to {args.output}")


if __name__ == '__main__':
    main()