import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import matplotlib.pyplot as plt
import matplotlib as mpl
import logging
import json
import os


def construct_rppg_df(row, label):
    df = pd.DataFrame({
        f"BPM_rPPG {label}": row['BPM'],
        "Timestamp": row['timesES'],
        f"Uncertainty {label}": row['Uncertainty']
    })
    df['Timestamp'] = df['Timestamp'].astype(float)
    return df

def merge_df_all_timestamps(garmin, movesense, rppg_phone, rppg_webcam):
    max_ts = min(
        garmin['Timestamp'].max(),
        movesense['Timestamp'].max(),
        rppg_webcam['Timestamp'].max(),
        rppg_phone['Timestamp'].max() if not rppg_phone.empty else np.inf
    )
    all_ts = pd.concat([
        garmin[['Timestamp']],
        movesense[['Timestamp']],
        rppg_webcam[['Timestamp']],
        rppg_phone[['Timestamp']] if not rppg_phone.empty else pd.DataFrame()
    ]).drop_duplicates().sort_values('Timestamp')
    all_ts = all_ts[all_ts['Timestamp'] <= max_ts]

    merged = pd.merge_asof(all_ts, garmin.sort_values('Timestamp'), on='Timestamp')
    merged = pd.merge_asof(merged, movesense.sort_values('Timestamp'), on='Timestamp')
    if not rppg_phone.empty:
        merged = pd.merge_asof(merged, rppg_phone.sort_values('Timestamp'), on='Timestamp')
    merged = pd.merge_asof(merged, rppg_webcam.sort_values('Timestamp'), on='Timestamp')
    return merged

def merge_2_df_rppg_timestamp(df1, rppg):
    rppg_sorted = rppg.sort_values('Timestamp')
    merged = pd.merge_asof(rppg_sorted, df1.sort_values('Timestamp'), on='Timestamp')
    return merged

def merge_3_df_rppg_timestamp(df1, df2, rppg):
    rppg_sorted = rppg.sort_values('Timestamp')
    merged = pd.merge_asof(rppg_sorted, df1.sort_values('Timestamp'), on='Timestamp')
    merged = pd.merge_asof(merged, df2.sort_values('Timestamp'), on='Timestamp')
    return merged

def stats_on_merged_df(df, rppg_col_name, ground_truth_cols):
    rppg = df[rppg_col_name].to_numpy()

    stats_df = pd.DataFrame({})

    for gt_col in ground_truth_cols:
        gt = df[gt_col].to_numpy()
        stats_df['MAE_rppg_' + gt_col] = [mean_absolute_error(gt, rppg)]
        stats_df['RMSE_rppg_' + gt_col] = [root_mean_squared_error(gt, rppg)]

    for gt_01, gt_02 in itertools.combinations(ground_truth_cols, 2):
        stats_df['MAE_' + gt_01 + '_' + gt_02] = [mean_absolute_error(df[gt_01].to_numpy(), df[gt_02].to_numpy())]
        stats_df['RMSE_' + gt_01 + '_' + gt_02] = [root_mean_squared_error(df[gt_01].to_numpy(), df[gt_02].to_numpy())]

    return stats_df

def df_from_movesense_json(filepath):
    try:
        with open(filepath, 'r') as file:
            data = json.load(file)
        heart_rate_data = data['data']
        flattened_data = [{
            'BPM_Movesense': entry['heartRate']['average'],
            'rrData': entry['heartRate']['rrData'][0]
        } for entry in heart_rate_data]

        df = pd.DataFrame(flattened_data)
        df['Timestamp'] = df['rrData'].cumsum() / 1000
        return df.drop(columns=['rrData'])
    except Exception as e:
        logging.error(f"Failed to load Movesense JSON: {filepath} - {e}")
        return pd.DataFrame()

def df_from_garmin_csv(filepath):
    try:
        df = pd.read_csv(filepath)
        df = df[df['Message'] == 'record'][['Value 1', 'Value 4']].iloc[1:].reset_index(drop=True)
        df.columns = ['Timestamp', 'BPM_Garmin']
        starttime = int(df.loc[0, 'Timestamp'])
        df['Timestamp'] = df['Timestamp'].astype(float) - starttime
        return df
    except Exception as e:
        logging.error(f"Failed to load Garmin CSV: {filepath} - {e}")
        return pd.DataFrame()

def df_from_ubfc_ds1(filepath):
    with open(filepath, 'r') as file:
        rows = [l.split(',')[:2] for l in file.readlines()]
    df = pd.DataFrame(rows, columns=['Timestep (ms)', 'Ground truth BPM'])
    df['Ground truth BPM'] = df['Ground truth BPM'].astype(float)
    df['Timestamp'] = df['Timestep (ms)'].astype(float) / 1000
    df.drop(columns=['Timestep (ms)'], inplace=True)
    return df

def df_from_ubfc_ds2(filepath):
    with open(filepath, 'r') as file:
        rows = file.readlines()
    rows = np.array([[float(i.strip(' ')) for i in l.split(' ') if i.strip(' ') != ''] for l in rows]).swapaxes(0, 1)
    df = pd.DataFrame(rows, columns=['PPG signal', 'Ground truth BPM', 'Timestamp'])
    df['Ground truth BPM'] = df['Ground truth BPM'].astype(float)
    df.drop(columns=['PPG signal'], inplace=True)
    return df

def plot_signals_nfi(path, df, method, name, experiment, phone):
    title = experiment.replace("Dark", "Low Light").replace("Move", "Motion").replace("Base", "Baseline")
    plt.figure(figsize=(13, 7))
    plt.rcParams.update({'font.size': 20})
    mpl.rcParams['axes.labelsize'] = 25

    plt.plot(df['Timestamp'], df['BPM_Garmin'], label='Smartwatch', color='purple')
    plt.plot(df['Timestamp'], df['BPM_Movesense'], label='ECG sensor', color='red')
    plt.plot(df['Timestamp'], df['BPM_rPPG webcam'], label='rPPG webcam', color='orange')
    plt.fill_between(df['Timestamp'], df['BPM_rPPG webcam'] - df['Uncertainty webcam'],
                     df['BPM_rPPG webcam'] + df['Uncertainty webcam'], color='orange', alpha=0.2)
    if phone:
        plt.plot(df['Timestamp'], df['BPM_rPPG phone'], label='BPM_rPPG phone', color='blue')
        plt.fill_between(df['Timestamp'], df['BPM_rPPG phone'] - df['Uncertainty phone'],
                         df['BPM_rPPG phone'] + df['Uncertainty phone'], color='blue', alpha=0.2)

    plt.xlabel('Time (s)')
    plt.ylabel('Heart Rate (BPM)')
    plt.title(f'Comparison of heart rate measurements ({name}, {title})', fontsize=30)
    plt.legend(loc='lower left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)

    figdir = os.path.join(path, f"signal_plots/{method}")
    os.makedirs(figdir, exist_ok=True)
    plt.savefig(os.path.join(figdir, f"{name}_{experiment}.png"), bbox_inches='tight')
    plt.close()

def plot_signals(path, df, dataset, method, participant):
    plt.figure(figsize=(13, 7))
    plt.rcParams.update({'font.size': 20})
    mpl.rcParams['axes.labelsize'] = 25

    plt.plot(df['Timestamp'], df['Ground truth BPM'], label='Ground truth', color='red')
    plt.plot(df['Timestamp'], df['BPM_rPPG '], label='rPPG', color='orange')
    plt.fill_between(df['Timestamp'], df['BPM_rPPG '] - df['Uncertainty '],
                     df['BPM_rPPG '] + df['Uncertainty '], color='orange', alpha=0.2)
    plt.xlabel('Time (s)')
    plt.ylabel('Heart Rate (BPM)')
    plt.title(f'Comparison of heart rate measurements ({method})', fontsize=30)
    plt.legend(loc='lower left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.ylim(40, 140)
    plt.tight_layout()

    figdir = os.path.join(path, f"{method}")
    os.makedirs(figdir, exist_ok=True)
    plt.savefig(os.path.join(figdir, f"{dataset}_{method}_{participant}.png"), bbox_inches='tight')
    plt.close()

def flatten(value) -> np.ndarray:
    """
    Flatten a possibly nested/ragged structure (list of arrays, dict of per-region lists, scalars, None)
    into a 1-D array of finite floats.
    """
    if value is None:
        return np.array([], dtype=float)
    if isinstance(value, dict):
        parts = [flatten(v) for v in value.values()]
        arr = np.concatenate(parts) if parts else np.array([], dtype=float)
    elif isinstance(value, (list, tuple, np.ndarray)):
        parts = [flatten(v) for v in value]
        arr = np.concatenate(parts) if parts else np.array([], dtype=float)
    else:
        try:
            arr = np.array([float(value)], dtype=float)
        except (TypeError, ValueError):
            arr = np.array([], dtype=float)
    return arr[np.isfinite(arr)]


def _summary_stats(arr: np.ndarray, prefix: str) -> dict:
    """Mean/std of a 1-D array of finite floats."""
    if arr.size == 0:
        return {f'{prefix}_mean': np.nan, f'{prefix}_std': np.nan}
    return {f'{prefix}_mean': float(arr.mean()), f'{prefix}_std': float(arr.std())}


def average_patch_axis(bvp_sample) -> np.ndarray:
    """
    Transform a bvp signal of shape (num_winows, num_patches, num_frames) into an array of
    shape (num_windows, num_frames), averaging over the patch axis if present.
    """
    arr = np.array(bvp_sample, dtype=float)
    if arr.ndim == 3:  # (num_windows, num_patches, num_frames)
        arr = arr.mean(axis=1)
    return arr