import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ── Load Data (same as Phase 1) ───────────────────────────
DATA_PATH = r'C:\Users\HP\Desktop\study\Forensics\project\forensic final project\signature_authenticator\Task2\Task2'
COLUMNS = ['x', 'y', 'timestamp', 'button', 'azimuth', 'altitude', 'pressure']

all_data = []
for filename in os.listdir(DATA_PATH):
    filepath = os.path.join(DATA_PATH, filename)
    filename_upper = filename.upper().replace('.TXT', '')
    if not filename_upper.startswith('U'):
        continue
    try:
        u_part = filename_upper.split('S')[0]
        s_part = filename_upper.split('S')[1]
        user_id = int(u_part[1:])
        sig_id  = int(s_part)
    except:
        continue
    label = 0 if sig_id <= 20 else 1
    df = pd.read_csv(filepath, sep=' ', skiprows=1, header=None, names=COLUMNS)
    df['user_id']      = user_id
    df['signature_id'] = sig_id
    df['label']        = label
    all_data.append(df)

full_df = pd.concat(all_data, ignore_index=True)
print("Data loaded! Shape:", full_df.shape)

# ── Feature Extraction ────────────────────────────────────
def extract_features(sig_df):
    sig_df = sig_df.sort_values('timestamp').reset_index(drop=True)
    
    # Time features
    total_time = sig_df['timestamp'].max() - sig_df['timestamp'].min()
    
    # Pressure features
    mean_pressure     = sig_df['pressure'].mean()
    pressure_variance = sig_df['pressure'].var()
    max_pressure      = sig_df['pressure'].max()
    min_pressure      = sig_df['pressure'].min()
    
    # Speed features (distance / time between consecutive points)
    dx = sig_df['x'].diff()
    dy = sig_df['y'].diff()
    dt = sig_df['timestamp'].diff().replace(0, np.nan)
    
    dist   = np.sqrt(dx**2 + dy**2)
    speed  = dist / dt
    
    mean_speed = speed.mean()
    max_speed  = speed.max()
    
    # Path length
    path_length = dist.sum()
    
    # Stroke features (button=0 means pen lifted)
    pen_down = sig_df['button']
    # Count number of strokes (transitions from 0 to 1)
    num_strokes = ((pen_down == 1) & (pen_down.shift(1) == 0)).sum()
    
    # Pause duration (total time when button = 0)
    pause_mask     = sig_df['button'] == 0
    pause_duration = sig_df.loc[pause_mask, 'timestamp'].diff().sum()
    
    # Altitude & Azimuth averages
    mean_altitude = sig_df['altitude'].mean()
    mean_azimuth  = sig_df['azimuth'].mean()
    
    return {
        'total_time'       : total_time,
        'mean_pressure'    : mean_pressure,
        'pressure_variance': pressure_variance,
        'max_pressure'     : max_pressure,
        'min_pressure'     : min_pressure,
        'mean_speed'       : mean_speed,
        'max_speed'        : max_speed,
        'path_length'      : path_length,
        'num_strokes'      : num_strokes,
        'pause_duration'   : pause_duration,
        'mean_altitude'    : mean_altitude,
        'mean_azimuth'     : mean_azimuth,
    }

# ── Apply to every signature ──────────────────────────────
features_list = []

grouped = full_df.groupby(['user_id', 'signature_id', 'label'])
total   = len(grouped)

for i, ((user_id, sig_id, label), group) in enumerate(grouped):
    feats = extract_features(group)
    feats['user_id']      = user_id
    feats['signature_id'] = sig_id
    feats['label']        = label
    features_list.append(feats)
    
    if (i+1) % 100 == 0:
        print(f"  Processed {i+1}/{total} signatures...")

features_df = pd.DataFrame(features_list)

print("\n=== FEATURE EXTRACTION COMPLETE ===")
print("Shape:", features_df.shape)
print("\nSample features:")
print(features_df.head())

print("\nGenuine vs Forged — Mean Pressure:")
print(features_df.groupby('label')['mean_pressure'].mean())

print("\nGenuine vs Forged — Mean Speed:")
print(features_df.groupby('label')['mean_speed'].mean())

# ── Save features to CSV ──────────────────────────────────
features_df.to_csv('features.csv', index=False)
print("\nFeatures saved to features.csv!")
print("\nPhase 2 Complete! ✓")



