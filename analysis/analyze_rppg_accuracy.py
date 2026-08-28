import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    parser = argparse.ArgumentParser(
        description='Analyse rPPG accuracy statistics produced by calculate_rppg_accuracy.py.'
    )
    parser.add_argument('--stats', type=str, required=True,
                        help='Path to the gt_experiment_stats.csv file.')
    args = parser.parse_args()

    stats = pd.read_csv(args.stats)
    stats['MAE_avg']  = (stats['MAE_rppg_BPM_Garmin']  + stats['MAE_rppg_BPM_Movesense'])  / 2
    stats['RMSE_avg'] = (stats['RMSE_rppg_BPM_Garmin'] + stats['RMSE_rppg_BPM_Movesense']) / 2

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    sns.boxplot(stats, hue='Experiment', x='Extraction Method', y='MAE_avg', ax=axs[0])
    sns.boxplot(stats, hue='Experiment', x='Extraction Method', y='RMSE_avg', ax=axs[1])
    plt.tight_layout()
    plt.show()

    print(stats.groupby('Experiment')[
        ['MAE_rppg_BPM_Garmin', 'MAE_rppg_BPM_Movesense', 'MAE_avg']
    ].mean())
    print("\nAll experiments:")
    print(stats.groupby('Extraction Method')[['RMSE_avg', 'MAE_avg']].mean())
    print("\nBase experiment only:")
    print(stats[stats['Experiment'] == 'base'].groupby('Extraction Method')[['RMSE_avg', 'MAE_avg']].mean())
    print("\nMove experiment only:")
    print(stats[stats['Experiment'] == 'move'].groupby('Extraction Method')[['RMSE_avg', 'MAE_avg']].mean())
    print("\nDark experiment only:")
    print(stats[stats['Experiment'] == 'dark'].groupby('Extraction Method')[['RMSE_avg', 'MAE_avg']].mean())
    print("\nPhone only:")
    print(stats[stats['Device'] == 'phone'].groupby('Extraction Method')[['RMSE_avg', 'MAE_avg']].mean())
    print("\nWebcam only:")
    print(stats[stats['Device'] == 'webcam'].groupby('Extraction Method')[['RMSE_avg', 'MAE_avg']].mean())

    sns.swarmplot(stats[stats['Experiment'] == 'base'], x='Participant', y='MAE_avg')
    plt.show()

    sns.swarmplot(stats[stats['Experiment'] == 'base'], x='Participant',
                  hue='Extraction Method', y='MAE_rppg_BPM_Garmin', palette='Set2')
    plt.show()


if __name__ == '__main__':
    main()