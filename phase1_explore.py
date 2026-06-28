import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ── CORRECT PATH ──────────────────────────────────────────
DATA_PATH = r'C:\Users\HP\Desktop\study\Forensics\project\forensic final project\signature_authenticator\Task2\Task2'

COLUMNS = ['x', 'y', 'timestamp', 'button', 'azimuth', 'altitude', 'pressure']

# ── Load ALL signature files into one DataFrame ───────────
all_data = []

for filename in os.listdir(DATA_PATH):
    filepath = os.path.join(DATA_PATH, filename)
    
    # Parse user and signature number from filename e.g. U1S1
    filename_upper = filename.upper().replace('.TXT', '')
    if not filename_upper.startswith('U'):
        continue
    
    try:
        u_part = filename_upper.split('S')[0]   # 'U1'
        s_part = filename_upper.split('S')[1]   # '1'
        user_id = int(u_part[1:])
        sig_id  = int(s_part)
    except:
        continue
    
    # S1-S20 = Genuine (0), S21-S40 = Forged (1)
    label = 0 if sig_id <= 20 else 1
    
    # Read file (skip first line — it's just the point count)
    df = pd.read_csv(filepath, sep=' ', skiprows=1, header=None, names=COLUMNS)
    df['user_id']      = user_id
    df['signature_id'] = sig_id
    df['label']        = label
    
    all_data.append(df)

# Combine everything
full_df = pd.concat(all_data, ignore_index=True)

# ── Basic Info ────────────────────────────────────────────
print("=== FULL DATASET ===")
print("Shape:", full_df.shape)
print("\nColumns:", full_df.columns.tolist())
print("\nFirst 5 rows:")
print(full_df.head())

print("\n=== LABEL DISTRIBUTION ===")
print(full_df['label'].value_counts())
print("0 = Genuine  |  1 = Forged")

print("\n=== USERS ===")
print("Total users:", full_df['user_id'].nunique())

print("\n=== MISSING VALUES ===")
print(full_df.isnull().sum())

# ── Plot genuine vs forged for User 1 ────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

user1 = full_df[full_df['user_id'] == 1]

genuine = user1[(user1['label'] == 0) & (user1['signature_id'] == 1)]
forged  = user1[(user1['label'] == 1) & (user1['signature_id'] == 21)]

axes[0].plot(genuine['x'], genuine['y'], 'b-', linewidth=1)
axes[0].set_title('Genuine Signature - User 1')
axes[0].invert_yaxis()

axes[1].plot(forged['x'], forged['y'], 'r-', linewidth=1)
axes[1].set_title('Forged Signature - User 1')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('phase1_output.png')
plt.show()

print("\nPhase 1 Complete! Plot saved.")




