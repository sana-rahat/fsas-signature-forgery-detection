import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  FORENSIC SIGNATURE FORGERY DETECTION SYSTEM")
print("  SVC2004 Task 2 Dataset | Anomaly Detection")
print("=" * 60)

# ═══════════════════════════════════════════════════════════
# STEP 1 — LOAD RAW DATA
# ═══════════════════════════════════════════════════════════
DATA_PATH = r'C:\Users\HP\Desktop\study\Forensics\project\forensic final project\signature_authenticator\Task2\Task2'
COLUMNS   = ['x', 'y', 'timestamp', 'button', 'azimuth', 'altitude', 'pressure']

print("\n[1/5] Loading raw signature data...")

all_data = []
for filename in os.listdir(DATA_PATH):
    filepath = os.path.join(DATA_PATH, filename)
    fn = filename.upper().replace('.TXT', '')
    if not fn.startswith('U'):
        continue
    try:
        user_id = int(fn.split('S')[0][1:])
        sig_id  = int(fn.split('S')[1])
    except:
        continue
    label = 0 if sig_id <= 20 else 1
    df = pd.read_csv(filepath, sep=' ', skiprows=1, header=None, names=COLUMNS)
    df['user_id']      = user_id
    df['signature_id'] = sig_id
    df['label']        = label
    all_data.append(df)

full_df = pd.concat(all_data, ignore_index=True)
print(f"    ✓ Loaded {full_df['user_id'].nunique()} users | {len(full_df):,} data points")
print(f"    ✓ Genuine rows: {(full_df['label']==0).sum():,} | Forged rows: {(full_df['label']==1).sum():,}")

# ═══════════════════════════════════════════════════════════
# STEP 2 — FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════
print("\n[2/5] Extracting features per signature...")

def extract_features(sig_df):
    sig_df = sig_df.sort_values('timestamp').reset_index(drop=True)
    total_time        = sig_df['timestamp'].max() - sig_df['timestamp'].min()
    mean_pressure     = sig_df['pressure'].mean()
    pressure_variance = sig_df['pressure'].var()
    max_pressure      = sig_df['pressure'].max()
    min_pressure      = sig_df['pressure'].min()
    dx    = sig_df['x'].diff()
    dy    = sig_df['y'].diff()
    dt    = sig_df['timestamp'].diff().replace(0, np.nan)
    dist  = np.sqrt(dx**2 + dy**2)
    speed = dist / dt
    mean_speed    = speed.mean()
    max_speed     = speed.max()
    path_length   = dist.sum()
    pen_down      = sig_df['button']
    num_strokes   = ((pen_down == 1) & (pen_down.shift(1) == 0)).sum()
    pause_mask    = sig_df['button'] == 0
    pause_duration = sig_df.loc[pause_mask, 'timestamp'].diff().sum()
    mean_altitude = sig_df['altitude'].mean()
    mean_azimuth  = sig_df['azimuth'].mean()
    return {
        'total_time': total_time, 'mean_pressure': mean_pressure,
        'pressure_variance': pressure_variance, 'max_pressure': max_pressure,
        'min_pressure': min_pressure, 'mean_speed': mean_speed,
        'max_speed': max_speed, 'path_length': path_length,
        'num_strokes': num_strokes, 'pause_duration': pause_duration,
        'mean_altitude': mean_altitude, 'mean_azimuth': mean_azimuth,
    }

features_list = []
for (user_id, sig_id, label), group in full_df.groupby(['user_id', 'signature_id', 'label']):
    feats = extract_features(group)
    feats['user_id'] = user_id
    feats['signature_id'] = sig_id
    feats['label'] = label
    features_list.append(feats)

features_df = pd.DataFrame(features_list)
print(f"    ✓ {len(features_df)} signatures | {len([c for c in features_df.columns if c not in ['user_id','signature_id','label']])} features each")

FEATURE_COLS = [
    'total_time', 'mean_pressure', 'pressure_variance',
    'max_pressure', 'min_pressure', 'mean_speed', 'max_speed',
    'path_length', 'num_strokes', 'pause_duration',
    'mean_altitude', 'mean_azimuth'
]

# ═══════════════════════════════════════════════════════════
# STEP 3 — TRAIN MODELS
# ═══════════════════════════════════════════════════════════
print("\n[3/5] Training Isolation Forest & One-Class SVM per user...")

all_results    = []
iso_scores_all = []
svm_scores_all = []
y_true_all     = []

for user_id in sorted(features_df['user_id'].unique()):
    user_data = features_df[features_df['user_id'] == user_id]
    genuine   = user_data[user_data['label'] == 0]

    X_train = genuine[FEATURE_COLS].values
    X_test  = user_data[FEATURE_COLS].values
    y_true  = user_data['label'].values

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    iso = IsolationForest(contamination=0.3, random_state=42)
    iso.fit(X_train)
    iso_pred = np.where(iso.predict(X_test) == 1, 0, 1)
    iso_scores_all.extend(-iso.score_samples(X_test))

    svm = OneClassSVM(kernel='rbf', nu=0.3, gamma='scale')
    svm.fit(X_train)
    svm_pred = np.where(svm.predict(X_test) == 1, 0, 1)
    svm_scores_all.extend(-svm.score_samples(X_test))

    y_true_all.extend(y_true)

    def calc_metrics(yt, yp):
        if len(np.unique(yp)) < 2:
            return 0, 0, sum(yt == yp) / len(yt)
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0,1]).ravel()
        FAR = fp / (fp + tn) if (fp + tn) > 0 else 0
        FRR = fn / (fn + tp) if (fn + tp) > 0 else 0
        ACC = (tn + tp) / len(yt)
        return round(FAR,3), round(FRR,3), round(ACC,3)

    iF, iR, iA = calc_metrics(y_true, iso_pred)
    sF, sR, sA = calc_metrics(y_true, svm_pred)
    all_results.append({'user_id': user_id,
                        'iso_FAR': iF, 'iso_FRR': iR, 'iso_ACC': iA,
                        'svm_FAR': sF, 'svm_FRR': sR, 'svm_ACC': sA})

results_df     = pd.DataFrame(all_results)
y_true_all     = np.array(y_true_all)
iso_scores_all = np.array(iso_scores_all)
svm_scores_all = np.array(svm_scores_all)

fpr_iso, tpr_iso, _ = roc_curve(y_true_all, iso_scores_all)
fpr_svm, tpr_svm, _ = roc_curve(y_true_all, svm_scores_all)
auc_iso = auc(fpr_iso, tpr_iso)
auc_svm = auc(fpr_svm, tpr_svm)

print(f"    ✓ Isolation Forest → ACC: {results_df['iso_ACC'].mean():.3f} | FAR: {results_df['iso_FAR'].mean():.3f} | FRR: {results_df['iso_FRR'].mean():.3f} | AUC: {auc_iso:.3f}")
print(f"    ✓ One-Class SVM    → ACC: {results_df['svm_ACC'].mean():.3f} | FAR: {results_df['svm_FAR'].mean():.3f} | FRR: {results_df['svm_FRR'].mean():.3f} | AUC: {auc_svm:.3f}")

# ═══════════════════════════════════════════════════════════
# STEP 4 — GENERATE ALL PLOTS
# ═══════════════════════════════════════════════════════════
print("\n[4/5] Generating visualizations...")

x = results_df['user_id']

# Plot 1 — FAR/FRR per user
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('FAR & FRR per User — Model Comparison', fontsize=14, fontweight='bold')
for ax, model, far_col, frr_col, title in zip(
    axes,
    ['iso', 'svm'],
    ['iso_FAR', 'svm_FAR'],
    ['iso_FRR', 'svm_FRR'],
    ['Isolation Forest', 'One-Class SVM']
):
    ax.bar(x - 0.2, results_df[far_col], 0.4, label='FAR', color='tomato',    alpha=0.85)
    ax.bar(x + 0.2, results_df[frr_col], 0.4, label='FRR', color='steelblue', alpha=0.85)
    ax.axhline(results_df[far_col].mean(), color='red',  linestyle='--', linewidth=1.2, label=f"Avg FAR={results_df[far_col].mean():.3f}")
    ax.axhline(results_df[frr_col].mean(), color='blue', linestyle='--', linewidth=1.2, label=f"Avg FRR={results_df[frr_col].mean():.3f}")
    ax.set_title(title); ax.set_xlabel('User ID'); ax.set_ylabel('Rate')
    ax.legend(); ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig('plot1_FAR_FRR_per_user.png', dpi=150); plt.close()

# Plot 2 — Accuracy comparison
fig, ax = plt.subplots(figsize=(16, 5))
ax.bar(x - 0.2, results_df['iso_ACC'], 0.4, label='Isolation Forest', color='mediumseagreen', alpha=0.85)
ax.bar(x + 0.2, results_df['svm_ACC'], 0.4, label='One-Class SVM',    color='mediumpurple',   alpha=0.85)
ax.axhline(results_df['iso_ACC'].mean(), color='green',  linestyle='--', linewidth=1.2, label=f"IF Avg={results_df['iso_ACC'].mean():.3f}")
ax.axhline(results_df['svm_ACC'].mean(), color='purple', linestyle='--', linewidth=1.2, label=f"SVM Avg={results_df['svm_ACC'].mean():.3f}")
ax.set_title('Accuracy per User — Isolation Forest vs One-Class SVM', fontsize=13, fontweight='bold')
ax.set_xlabel('User ID'); ax.set_ylabel('Accuracy'); ax.set_ylim(0, 1); ax.legend()
plt.tight_layout()
plt.savefig('plot2_accuracy_comparison.png', dpi=150); plt.close()

# Plot 3 — ROC Curve
fig, ax = plt.subplots(figsize=(8, 7))
ax.plot(fpr_iso, tpr_iso, color='mediumseagreen', lw=2, label=f'Isolation Forest (AUC={auc_iso:.3f})')
ax.plot(fpr_svm, tpr_svm, color='mediumpurple',   lw=2, label=f'One-Class SVM (AUC={auc_svm:.3f})')
ax.plot([0,1],[0,1], 'k--', linewidth=1)
ax.fill_between(fpr_iso, tpr_iso, alpha=0.08, color='green')
ax.fill_between(fpr_svm, tpr_svm, alpha=0.08, color='purple')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve — Signature Forgery Detection', fontsize=13, fontweight='bold')
ax.legend(fontsize=11); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('plot3_ROC_curve.png', dpi=150); plt.close()

# Plot 4 — Pressure Heatmap
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Pressure Heatmap — User 1: Genuine vs Forged', fontsize=13, fontweight='bold')
for ax, lbl, sig_id, title, cmap in zip(
    axes,
    [0, 1], [1, 21],
    ['Genuine (S1)', 'Forged (S21)'],
    ['Blues', 'Reds']
):
    sig = full_df[(full_df['user_id'] == 1) & (full_df['signature_id'] == sig_id)]
    sc  = ax.scatter(sig['x'], sig['y'], c=sig['pressure'], cmap=cmap, s=4, alpha=0.7)
    plt.colorbar(sc, ax=ax, label='Pressure')
    ax.set_title(title); ax.invert_yaxis()
    ax.set_xlabel('X'); ax.set_ylabel('Y')
plt.tight_layout()
plt.savefig('plot4_pressure_heatmap.png', dpi=150); plt.close()

# Plot 5 — Feature Distributions
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Feature Distributions — Genuine vs Forged', fontsize=14, fontweight='bold')
genuine_df = features_df[features_df['label'] == 0]
forged_df  = features_df[features_df['label'] == 1]
for ax, feat in zip(axes.flatten(), ['mean_pressure','mean_speed','pressure_variance','path_length','num_strokes','total_time']):
    ax.hist(genuine_df[feat], bins=30, alpha=0.6, color='steelblue', label='Genuine', density=True)
    ax.hist(forged_df[feat],  bins=30, alpha=0.6, color='tomato',    label='Forged',  density=True)
    ax.set_title(feat.replace('_',' ').title()); ax.set_xlabel('Value')
    ax.set_ylabel('Density'); ax.legend()
plt.tight_layout()
plt.savefig('plot5_feature_distributions.png', dpi=150); plt.close()

# Plot 6 — Summary Table
fig, ax = plt.subplots(figsize=(8, 3))
ax.axis('off')
table_data = [
    ['Metric',             'Isolation Forest',                       'One-Class SVM'],
    ['Accuracy',           f"{results_df['iso_ACC'].mean():.3f}",    f"{results_df['svm_ACC'].mean():.3f}"],
    ['FAR (False Accept)', f"{results_df['iso_FAR'].mean():.3f}",    f"{results_df['svm_FAR'].mean():.3f}"],
    ['FRR (False Reject)', f"{results_df['iso_FRR'].mean():.3f}",    f"{results_df['svm_FRR'].mean():.3f}"],
    ['AUC (ROC)',          f"{auc_iso:.3f}",                         f"{auc_svm:.3f}"],
]
tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0], loc='center', cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(12); tbl.scale(1.5, 2)
for j in range(3):
    tbl[0, j].set_facecolor('#2c3e50')
    tbl[0, j].set_text_props(color='white', fontweight='bold')
ax.set_title('Model Performance Summary', fontsize=13, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('plot6_summary_table.png', dpi=150); plt.close()

print("    ✓ All 6 plots saved!")

# ═══════════════════════════════════════════════════════════
# STEP 5 — SAVE OUTPUTS
# ═══════════════════════════════════════════════════════════
print("\n[5/5] Saving output files...")
features_df.to_csv('features.csv', index=False)
results_df.to_csv('model_results.csv', index=False)
print("    ✓ features.csv saved")
print("    ✓ model_results.csv saved")

print("\n" + "=" * 60)
print("  PIPELINE COMPLETE!")
print("=" * 60)
print(f"\n  Isolation Forest → ACC: {results_df['iso_ACC'].mean():.3f} | FAR: {results_df['iso_FAR'].mean():.3f} | FRR: {results_df['iso_FRR'].mean():.3f} | AUC: {auc_iso:.3f}")
print(f"  One-Class SVM    → ACC: {results_df['svm_ACC'].mean():.3f} | FAR: {results_df['svm_FAR'].mean():.3f} | FRR: {results_df['svm_FRR'].mean():.3f} | AUC: {auc_svm:.3f}")
print(f"\n  Output files: features.csv, model_results.csv")
print(f"  Plots saved:  plot1 through plot6 (.png)")
print("=" * 60)




