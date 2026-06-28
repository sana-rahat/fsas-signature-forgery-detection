import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ── Load features ─────────────────────────────────────────
features_df = pd.read_csv('features.csv')
print("Features loaded! Shape:", features_df.shape)

# ── Feature columns to use for ML ────────────────────────
FEATURE_COLS = [
    'total_time', 'mean_pressure', 'pressure_variance',
    'max_pressure', 'min_pressure', 'mean_speed', 'max_speed',
    'path_length', 'num_strokes', 'pause_duration',
    'mean_altitude', 'mean_azimuth'
]

# ── Train per-user models ─────────────────────────────────
all_results = []

users = features_df['user_id'].unique()
print(f"\nTraining models for {len(users)} users...\n")

for user_id in sorted(users):
    user_data = features_df[features_df['user_id'] == user_id]
    
    # Split genuine and forged
    genuine = user_data[user_data['label'] == 0]
    forged  = user_data[user_data['label'] == 1]
    
    # Train only on genuine signatures
    X_train = genuine[FEATURE_COLS].values
    
    # Test on both genuine + forged
    X_test  = user_data[FEATURE_COLS].values
    y_true  = user_data['label'].values  # 0=genuine, 1=forged
    
    # ── Scale features ────────────────────────────────────
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    
    # ── Model 1: Isolation Forest ─────────────────────────
    iso_forest = IsolationForest(contamination=0.3, random_state=42)
    iso_forest.fit(X_train)
    # IsolationForest: 1=normal(genuine), -1=anomaly(forged)
    iso_pred_raw = iso_forest.predict(X_test)
    iso_pred = np.where(iso_pred_raw == 1, 0, 1)  # convert to 0/1
    
    # ── Model 2: One-Class SVM ────────────────────────────
    oc_svm = OneClassSVM(kernel='rbf', nu=0.3, gamma='scale')
    oc_svm.fit(X_train)
    svm_pred_raw = oc_svm.predict(X_test)
    svm_pred = np.where(svm_pred_raw == 1, 0, 1)  # convert to 0/1
    
    # ── Calculate metrics per user ────────────────────────
    def calc_metrics(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0,1]
        ).ravel() if len(np.unique(y_pred)) > 1 else (0,0,0,0)
        
        FAR = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Acceptance Rate
        FRR = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Rejection Rate
        ACC = (tn + tp) / len(y_true)
        return FAR, FRR, ACC
    
    iso_FAR, iso_FRR, iso_ACC = calc_metrics(y_true, iso_pred)
    svm_FAR, svm_FRR, svm_ACC = calc_metrics(y_true, svm_pred)
    
    all_results.append({
        'user_id'    : user_id,
        'iso_FAR'    : round(iso_FAR, 3),
        'iso_FRR'    : round(iso_FRR, 3),
        'iso_ACC'    : round(iso_ACC, 3),
        'svm_FAR'    : round(svm_FAR, 3),
        'svm_FRR'    : round(svm_FRR, 3),
        'svm_ACC'    : round(svm_ACC, 3),
    })

results_df = pd.DataFrame(all_results)

# ── Print Results ─────────────────────────────────────────
print("=" * 60)
print("PER-USER RESULTS (first 10 users)")
print("=" * 60)
print(results_df.head(10).to_string(index=False))

print("\n" + "=" * 60)
print("OVERALL AVERAGE PERFORMANCE")
print("=" * 60)
print(f"Isolation Forest  →  ACC: {results_df['iso_ACC'].mean():.3f} | FAR: {results_df['iso_FAR'].mean():.3f} | FRR: {results_df['iso_FRR'].mean():.3f}")
print(f"One-Class SVM     →  ACC: {results_df['svm_ACC'].mean():.3f} | FAR: {results_df['svm_FAR'].mean():.3f} | FRR: {results_df['svm_FRR'].mean():.3f}")

# ── Save results ──────────────────────────────────────────
results_df.to_csv('model_results.csv', index=False)
print("\nResults saved to model_results.csv!")
print("\nPhase 3 Complete! ✓")



