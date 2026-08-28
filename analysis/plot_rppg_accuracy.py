import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    description='Plot rPPG BPM accuracy (RMSE vs. ground truth) across datasets, devices and conditions.'
)
parser.add_argument('--gt-stats', type=str, required=True, help='Path to gt_experiment_stats.csv.')
parser.add_argument('--ubfc1-stats', type=str, required=True, help='Path to UBFC1_stats.csv.')
parser.add_argument('--ubfc2-stats', type=str, required=True, help='Path to UBFC2_stats.csv.')
parser.add_argument('--predictions', type=str, required=True, help='Path to predicted_labels.csv.')
args = parser.parse_args()

gt_df = pd.read_csv(args.gt_stats)
ubfc1_df = pd.read_csv(args.ubfc1_stats)
ubfc2_df = pd.read_csv(args.ubfc2_stats)
model_predictions = pd.read_csv(args.predictions)

# Add model predictions
for key, row in model_predictions.iterrows():
    participant = row['Filename'].split('/')[-2]
    if participant in ['Participant 1', 'Participant 2', 'Participant 3', 'Participant 4', 'Participant 5', 'Participant 6']:
        experiment, device = row['Filename'].split('.')[0].split('_')[-2:]
        gt_df.loc[(gt_df['Participant'] == int(participant.split(' ')[-1])) & (gt_df['Experiment'] == experiment) & (gt_df['Device'] == device), 'predicted_label'] = row['predicted_label']
        gt_df.loc[(gt_df['Participant'] == int(participant.split(' ')[-1])) & (gt_df['Experiment'] == experiment) & (gt_df['Device'] == device), 'predicted_prob'] = row['predicted_prob']

# Combine datasets
gt_df['dataset'] = 'Experiments'
gt_df['RMSE_GT'] = (gt_df['RMSE_rppg_BPM_Garmin'] + gt_df['RMSE_rppg_BPM_Movesense']) / 2
gt_df.drop(columns=['MAE_rppg_BPM_Garmin', 'RMSE_rppg_BPM_Garmin', 'RMSE_rppg_BPM_Movesense', 'MAE_BPM_Garmin_BPM_Movesense', 'RMSE_BPM_Garmin_BPM_Movesense', 'MAE_rppg_BPM_Movesense'], inplace=True)
ubfc1_df['dataset'] = 'UBFC1'
ubfc1_df.rename(columns={'RMSE_rppg_Ground truth BPM': 'RMSE_GT'}, inplace=True)
ubfc1_df.drop(columns=['MAE_rppg_Ground truth BPM'], inplace=True)
ubfc2_df['dataset'] = 'UBFC2'
ubfc2_df.rename(columns={'RMSE_rppg_Ground truth BPM': 'RMSE_GT'}, inplace=True)
ubfc2_df.drop(columns=['MAE_rppg_Ground truth BPM'], inplace=True)
df = pd.concat([gt_df, ubfc1_df, ubfc2_df]).reindex()

# Create figure
fig, axs = plt.subplots(1,3, figsize=(15, 5))

# Show RMSE per dataset
sns.barplot(data=df, x='dataset', y='RMSE_GT', hue='Extraction Method', ax=axs[0], legend=False)
axs[0].set_ylabel("RMSE")
axs[0].set_ylim(0, 40)

# Show RMSE per Camera type
sns.barplot(data=df.dropna(subset=['Device']), x='Device', y='RMSE_GT', hue='Extraction Method', ax=axs[1], legend=False)
axs[1].set_ylabel("RMSE")
axs[1].set_ylim(0, 40)

# Show RMSE per Experimental Condition
sns.barplot(data=df.dropna(subset=['Experiment']), x='Experiment', y='RMSE_GT', hue='Extraction Method', ax=axs[2])
axs[2].set_ylabel("RMSE")
axs[2].set_ylim(0, 40)

plt.tight_layout()
plt.legend(loc='upper right')
plt.show()

sns.scatterplot(gt_df[gt_df['Extraction Method'] == 'pos'], x='RMSE_GT', y='predicted_prob')
plt.show()

sns.heatmap(gt_df.corr(numeric_only=True), annot=True)
plt.show()
