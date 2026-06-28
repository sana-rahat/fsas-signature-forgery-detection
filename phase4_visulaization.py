import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ── Load data ─────────────────────────────────────────────
features_df = pd.read_csv('features.csv')
results_df  = pd.read_csv('model_results.csv')

FEATURE_COLS = [
    'total_time', 'mean_pressure', 'pressure_variance',
    'max_pressure', 'min_pressure', 'mean_speed', 'max_speed',
    'path_length', 'num_strokes', 'pause_duration',
    'mean_altitude', 'mean_azimuth'
]

print("Data loaded! Generating all plots...\n")

# ═══════════════════════════════════════════════════════════
# PLOT 1 — FAR / FRR Bar Chart (both models across all users)
# ═══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('FAR & FRR per User — Model Comparison', fontsize=14, fontweight='bold')

x = results_df['user_id']

# Isolation Forest
axes[0].bar(x - 0.2, results_df['iso_FAR'], 0.4, label='FAR', color='tomato',    alpha=0.85)
axes[0].bar(x + 0.2, results_df['iso_FRR'], 0.4, label='FRR', color='steelblue', alpha=0.85)
axes[0].axhline(results_df['iso_FAR'].mean(), color='red',  linestyle='--', linewidth=1.2, label=f"Avg FAR={results_df['iso_FAR'].mean():.3f}")
axes[0].axhline(results_df['iso_FRR'].mean(), color='blue', linestyle='--', linewidth=1.2, label=f"Avg FRR={results_df['iso_FRR'].mean():.3f}")
axes[0].set_title('Isolation Forest')
axes[0].set_xlabel('User ID')
axes[0].set_ylabel('Rate')
axes[0].legend()
axes[0].set_ylim(0, 1)

# One-Class SVM
axes[1].bar(x - 0.2, results_df['svm_FAR'], 0.4, label='FAR', color='tomato',    alpha=0.85)
axes[1].bar(x + 0.2, results_df['svm_FRR'], 0.4, label='FRR', color='steelblue', alpha=0.85)
axes[1].axhline(results_df['svm_FAR'].mean(), color='red',  linestyle='--', linewidth=1.2, label=f"Avg FAR={results_df['svm_FAR'].mean():.3f}")
axes[1].axhline(results_df['svm_FRR'].mean(), color='blue', linestyle='--', linewidth=1.2, label=f"Avg FRR={results_df['svm_FRR'].mean():.3f}")
axes[1].set_title('One-Class SVM')
axes[1].set_xlabel('User ID')
axes[1].set_ylabel('Rate')
axes[1].legend()
axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('plot1_FAR_FRR_per_user.png', dpi=150)
plt.show()
print("Plot 1 saved: FAR/FRR per user")

# ═══════════════════════════════════════════════════════════
# PLOT 2 — Accuracy Comparison Bar Chart
# ═══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(16, 5))

ax.bar(x - 0.2, results_df['iso_ACC'], 0.4, label='Isolation Forest', color='mediumseagreen', alpha=0.85)
ax.bar(x + 0.2, results_df['svm_ACC'], 0.4, label='One-Class SVM',    color='mediumpurple',   alpha=0.85)
ax.axhline(results_df['iso_ACC'].mean(), color='green',  linestyle='--', linewidth=1.2, label=f"IF Avg={results_df['iso_ACC'].mean():.3f}")
ax.axhline(results_df['svm_ACC'].mean(), color='purple', linestyle='--', linewidth=1.2, label=f"SVM Avg={results_df['svm_ACC'].mean():.3f}")
ax.set_title('Accuracy per User — Isolation Forest vs One-Class SVM', fontsize=13, fontweight='bold')
ax.set_xlabel('User ID')
ax.set_ylabel('Accuracy')
ax.set_ylim(0, 1)
ax.legend()

plt.tight_layout()
plt.savefig('plot2_accuracy_comparison.png', dpi=150)
plt.show()
print("Plot 2 saved: Accuracy comparison")

# ═══════════════════════════════════════════════════════════
# PLOT 3 — ROC Curve (aggregate across all users)
# ═══════════════════════════════════════════════════════════
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
    iso_scores = -iso.score_samples(X_test)   # higher = more anomalous

    svm = OneClassSVM(kernel='rbf', nu=0.3, gamma='scale')
    svm.fit(X_train)
    svm_scores = -svm.score_samples(X_test)

    iso_scores_all.extend(iso_scores)
    svm_scores_all.extend(svm_scores)
    y_true_all.extend(y_true)

y_true_all     = np.array(y_true_all)
iso_scores_all = np.array(iso_scores_all)
svm_scores_all = np.array(svm_scores_all)

fpr_iso, tpr_iso, _ = roc_curve(y_true_all, iso_scores_all)
fpr_svm, tpr_svm, _ = roc_curve(y_true_all, svm_scores_all)
auc_iso = auc(fpr_iso, tpr_iso)
auc_svm = auc(fpr_svm, tpr_svm)

fig, ax = plt.subplots(figsize=(8, 7))
ax.plot(fpr_iso, tpr_iso, color='mediumseagreen', lw=2, label=f'Isolation Forest (AUC = {auc_iso:.3f})')
ax.plot(fpr_svm, tpr_svm, color='mediumpurple',   lw=2, label=f'One-Class SVM (AUC = {auc_svm:.3f})')
ax.plot([0,1],[0,1], 'k--', linewidth=1, label='Random Classifier')
ax.fill_between(fpr_iso, tpr_iso, alpha=0.08, color='green')
ax.fill_between(fpr_svm, tpr_svm, alpha=0.08, color='purple')
ax.set_xlabel('False Positive Rate (FAR)', fontsize=12)
ax.set_ylabel('True Positive Rate (Forgery Detection Rate)', fontsize=12)
ax.set_title('ROC Curve — Signature Forgery Detection', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plot3_ROC_curve.png', dpi=150)
plt.show()
print(f"Plot 3 saved: ROC curve | IF AUC={auc_iso:.3f} | SVM AUC={auc_svm:.3f}")

# ═══════════════════════════════════════════════════════════
# PLOT 4 — Pressure Heatmap (Genuine vs Forged, User 1)
# ═══════════════════════════════════════════════════════════
DATA_PATH = r'C:\Users\HP\Desktop\study\Forensics\project\forensic final project\signature_authenticator\Task2\Task2'
COLUMNS   = ['x', 'y', 'timestamp', 'button', 'azimuth', 'altitude', 'pressure']

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
    df['user_id'] = user_id
    df['signature_id'] = sig_id
    df['label'] = label
    all_data.append(df)

full_df = pd.concat(all_data, ignore_index=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Pressure Heatmap — User 1: Genuine vs Forged', fontsize=13, fontweight='bold')

for ax, (lbl, title, cmap) in zip(axes, [(0, 'Genuine (S1)', 'Blues'), (1, 'Forged (S21)', 'Reds')]):
    sig_id = 1 if lbl == 0 else 21
    sig = full_df[(full_df['user_id'] == 1) & (full_df['signature_id'] == sig_id)]

    sc = ax.scatter(sig['x'], sig['y'],
                    c=sig['pressure'], cmap=cmap,
                    s=4, alpha=0.7)
    plt.colorbar(sc, ax=ax, label='Pressure')
    ax.set_title(title)
    ax.invert_yaxis()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

plt.tight_layout()
plt.savefig('plot4_pressure_heatmap.png', dpi=150)
plt.show()
print("Plot 4 saved: Pressure heatmap")

# ═══════════════════════════════════════════════════════════
# PLOT 5 — Feature Distribution: Genuine vs Forged
# ═══════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Feature Distributions — Genuine vs Forged', fontsize=14, fontweight='bold')

plot_features = ['mean_pressure', 'mean_speed', 'pressure_variance',
                 'path_length',   'num_strokes', 'total_time']

genuine_df = features_df[features_df['label'] == 0]
forged_df  = features_df[features_df['label'] == 1]

for ax, feat in zip(axes.flatten(), plot_features):
    ax.hist(genuine_df[feat], bins=30, alpha=0.6, color='steelblue', label='Genuine', density=True)
    ax.hist(forged_df[feat],  bins=30, alpha=0.6, color='tomato',    label='Forged',  density=True)
    ax.set_title(feat.replace('_', ' ').title())
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()

plt.tight_layout()
plt.savefig('plot5_feature_distributions.png', dpi=150)
plt.show()
print("Plot 5 saved: Feature distributions")

# ═══════════════════════════════════════════════════════════
# PLOT 6 — Summary Table (Model Comparison)
# ═══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 3))
ax.axis('off')

table_data = [
    ['Metric',            'Isolation Forest', 'One-Class SVM'],
    ['Accuracy',          f"{results_df['iso_ACC'].mean():.3f}", f"{results_df['svm_ACC'].mean():.3f}"],
    ['FAR (False Accept)', f"{results_df['iso_FAR'].mean():.3f}", f"{results_df['svm_FAR'].mean():.3f}"],
    ['FRR (False Reject)', f"{results_df['iso_FRR'].mean():.3f}", f"{results_df['svm_FRR'].mean():.3f}"],
    ['AUC (ROC)',          f"{auc_iso:.3f}",                      f"{auc_svm:.3f}"],
]

tbl = ax.table(cellText=table_data[1:],
               colLabels=table_data[0],
               loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
tbl.scale(1.5, 2)

# Color header row
for j in range(3):
    tbl[0, j].set_facecolor('#2c3e50')
    tbl[0, j].set_text_props(color='white', fontweight='bold')

ax.set_title('Model Performance Summary', fontsize=13, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('plot6_summary_table.png', dpi=150)
plt.show()
print("Plot 6 saved: Summary table")

print("\n" + "="*50)
print("PHASE 4 COMPLETE! All 6 plots saved ✓")
print("="*50)
print(f"\nFinal Summary:")
print(f"  Isolation Forest → ACC: {results_df['iso_ACC'].mean():.3f} | AUC: {auc_iso:.3f}")
print(f"  One-Class SVM    → ACC: {results_df['svm_ACC'].mean():.3f} | AUC: {auc_svm:.3f}")




