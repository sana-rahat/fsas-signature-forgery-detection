from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import os, warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ── FIX: Registering custom zfill filter directly to Flask's Jinja environment ──
@app.template_filter('zfill')
def zfill_filter(value, width=2):
    """Pads string numbers with leading zeros to fix the Jinja rendering crash."""
    return str(value).zfill(width)

DATA_PATH = r'C:\Users\HP\Desktop\study\Forensics\project\forensic final project\signature_authenticator\Task2\Task2'
COLUMNS   = ['x', 'y', 'timestamp', 'button', 'azimuth', 'altitude', 'pressure']
FEATURE_COLS = [
    'total_time', 'mean_pressure', 'pressure_variance',
    'max_pressure', 'min_pressure', 'mean_speed', 'max_speed',
    'path_length', 'num_strokes', 'pause_duration',
    'mean_altitude', 'mean_azimuth'
]

def extract_features_df(df):
    df = df.sort_values('timestamp').reset_index(drop=True)
    total_time        = float(df['timestamp'].max() - df['timestamp'].min())
    mean_pressure     = float(df['pressure'].mean())
    pressure_variance = float(df['pressure'].var()) if df['pressure'].var() > 0 else 1.0
    max_pressure      = float(df['pressure'].max())
    min_pressure      = float(df['pressure'].min())
    dx   = df['x'].diff().fillna(0)
    dy   = df['y'].diff().fillna(0)
    dt   = df['timestamp'].diff().fillna(10).replace(0,10)
    dist = np.sqrt(dx**2 + dy**2)
    spd  = dist / dt
    mean_speed     = float(spd.mean())
    max_speed      = float(spd.max())
    path_length    = float(dist.sum())
    pen            = df['button']
    num_strokes    = int(((pen==1)&(pen.shift(1)==0)).sum())
    if num_strokes == 0: num_strokes = 1
    pause_rows     = df[df['button']==0]
    pause_duration = float(pause_rows['timestamp'].diff().sum()) if len(pause_rows)>1 else 0.0
    mean_altitude  = float(df['altitude'].mean())
    mean_azimuth   = float(df['azimuth'].mean())
    return {
        'total_time':total_time,'mean_pressure':mean_pressure,
        'pressure_variance':pressure_variance,'max_pressure':max_pressure,
        'min_pressure':min_pressure,'mean_speed':mean_speed,
        'max_speed':max_speed,'path_length':path_length,
        'num_strokes':num_strokes,'pause_duration':pause_duration,
        'mean_altitude':mean_altitude,'mean_azimuth':mean_azimuth,
    }

def extract_features_pts(points):
    df = pd.DataFrame(points)
    if 'pressure' not in df.columns: df['pressure'] = 500
    if 'button'   not in df.columns: df['button']   = 1
    if 'azimuth'  not in df.columns: df['azimuth']  = 1350
    if 'altitude' not in df.columns: df['altitude'] = 800
    return extract_features_df(df)

# ── Load dataset ──────────────────────────────────────────
print("Loading SVC2004...")
all_data = []
for filename in os.listdir(DATA_PATH):
    filepath = os.path.join(DATA_PATH, filename)
    fn = filename.upper().replace('.TXT','')
    if not fn.startswith('U'): continue
    try:
        user_id = int(fn.split('S')[0][1:])
        sig_id  = int(fn.split('S')[1])
    except: continue
    label = 0 if sig_id <= 20 else 1
    try:
        df = pd.read_csv(filepath,sep=' ',skiprows=1,header=None,names=COLUMNS)
        df['user_id']=user_id; df['signature_id']=sig_id; df['label']=label
        all_data.append(df)
    except: continue

full_df = pd.concat(all_data, ignore_index=True)

# Extract features
features_list = []
for (uid,sid,lbl),grp in full_df.groupby(['user_id','signature_id','label']):
    f = extract_features_df(grp)
    f['user_id']=uid; f['signature_id']=sid; f['label']=lbl
    features_list.append(f)
features_df = pd.DataFrame(features_list)

genuine_df = features_df[features_df['label']==0]
forged_df  = features_df[features_df['label']==1]

X_gen = genuine_df[FEATURE_COLS].values
X_for = forged_df[FEATURE_COLS].values

global_scaler = StandardScaler()
X_gen_sc = global_scaler.fit_transform(X_gen)
X_for_sc = global_scaler.transform(X_for)

global_iso = IsolationForest(contamination=0.2,random_state=42,n_estimators=200)
global_iso.fit(X_gen_sc)
global_svm = OneClassSVM(kernel='rbf',nu=0.2,gamma='scale')
global_svm.fit(X_gen_sc)

# Thresholds
iso_gen_scores = global_iso.score_samples(X_gen_sc)
iso_for_scores = global_iso.score_samples(X_for_sc)
iso_threshold  = float((np.percentile(iso_gen_scores,25) + np.percentile(iso_for_scores,75)) / 2)

genuine_stats = {col:{'mean':float(genuine_df[col].mean()),'std':float(genuine_df[col].std())+1e-9} for col in FEATURE_COLS}
forged_stats  = {col:{'mean':float(forged_df[col].mean()), 'std':float(forged_df[col].std())+1e-9}  for col in FEATURE_COLS}

# Per-user models
user_models = {}
for uid in sorted(features_df['user_id'].unique()):
    ud  = features_df[features_df['user_id']==uid]
    gen = ud[ud['label']==0]
    X   = gen[FEATURE_COLS].values
    sc  = StandardScaler(); Xs = sc.fit_transform(X)
    iso = IsolationForest(contamination=0.1,random_state=42,n_estimators=100); iso.fit(Xs)
    svm = OneClassSVM(kernel='rbf',nu=0.1,gamma='scale'); svm.fit(Xs)
    user_models[uid] = {'scaler':sc,'iso':iso,'svm':svm}

enrolled_users = {}
print(f"✓ Ready! {len(user_models)} user models trained.")

# ── Store raw sig data for dataset tab ───────────────────
sig_store = {}
for (uid,sid,lbl),grp in full_df.groupby(['user_id','signature_id','label']):
    key = f"{uid}_{sid}"
    sig_store[key] = {
        'x': grp['x'].tolist(), 'y': grp['y'].tolist(),
        'pressure': grp['pressure'].tolist(),
        'timestamp': grp['timestamp'].tolist(),
        'label': lbl
    }

@app.route('/')
def index():
    return render_template('index.html', users=list(range(1,41)))

# ── Tab 1: Dataset Verification ──────────────────────────
@app.route('/verify_dataset', methods=['POST'])
def verify_dataset():
    try:
        data   = request.get_json(force=True)
        uid    = int(data['user_id'])
        sig_id = int(data['sig_id'])

        key = f"{uid}_{sig_id}"
        if key not in sig_store:
            return jsonify({'error':'Signature not found'})

        sig  = sig_store[key]
        grp  = full_df[(full_df['user_id']==uid)&(full_df['signature_id']==sig_id)]
        f    = extract_features_df(grp)
        X    = np.array([[f[c] for c in FEATURE_COLS]])
        X_sc = user_models[uid]['scaler'].transform(X)

        iso_p = int(user_models[uid]['iso'].predict(X_sc)[0])
        svm_p = int(user_models[uid]['svm'].predict(X_sc)[0])
        iso_s = float(user_models[uid]['iso'].score_samples(X_sc)[0])
        svm_s = float(user_models[uid]['svm'].score_samples(X_sc)[0])

        true_label = sig['label']
        votes_genuine = sum([1 if iso_p==1 else 0, 1 if svm_p==1 else 0])

        if votes_genuine >= 2:
            verdict,color,icon = 'GENUINE','genuine','✅'
        elif votes_genuine == 0:
            verdict,color,icon = 'FORGED','forged','❌'
        else:
            verdict,color,icon = 'UNCERTAIN','uncertain','⚠️'

        correct = (verdict=='GENUINE' and true_label==0) or (verdict=='FORGED' and true_label==1)
        conf    = min(97, max(60, int(70 + iso_s*35)))

        analysis = {}
        for col in ['mean_speed','total_time','num_strokes','path_length','mean_pressure','pressure_variance']:
            gm = genuine_stats[col]['mean']
            gs = genuine_stats[col]['std']
            fm = forged_stats[col]['mean']
            z  = (f[col] - gm) / gs
            analysis[col] = {
                'value'        : round(f[col],2),
                'genuine_mean' : round(gm,2),
                'forged_mean'  : round(fm,2),
                'z_score'      : round(float(z),2),
                'closer_to'    : 'genuine' if abs(f[col]-gm) < abs(f[col]-fm) else 'forged'
            }

        return jsonify({
            'verdict':verdict,'color':color,'icon':icon,
            'confidence':conf,'true_label':true_label,
            'correct':correct,
            'iso_result':'Genuine' if iso_p==1 else 'Forged',
            'svm_result':'Genuine' if svm_p==1 else 'Forged',
            'iso_score':round(iso_s,4),
            'sig_data': sig, 'analysis': analysis,
            'features':{k:round(v,2) for k,v in f.items() if k in FEATURE_COLS}
        })
    except Exception as e:
        return jsonify({'error':str(e)})

# ── Tab 2: Pattern Detection (live draw) ─────────────────
@app.route('/detect_pattern', methods=['POST'])
def detect_pattern():
    try:
        data   = request.get_json(force=True)
        points = data.get('points',[])
        if len(points) < 20:
            return jsonify({'error':'Draw a longer signature!'})

        f    = extract_features_pts(points)
        X    = np.array([[f[c] for c in FEATURE_COLS]])
        X_sc = global_scaler.transform(X)

        iso_p = int(global_iso.predict(X_sc)[0])
        svm_p = int(global_svm.predict(X_sc)[0])
        iso_s = float(global_iso.score_samples(X_sc)[0])
        svm_s = float(global_svm.score_samples(X_sc)[0])

        suspicious = []
        analysis   = {}
        for col in ['mean_speed','total_time','num_strokes','path_length','mean_pressure','pressure_variance']:
            z = (f[col] - genuine_stats[col]['mean']) / genuine_stats[col]['std']
            flag = abs(z) > 1.8
            analysis[col] = {
                'value'   : round(f[col],2),
                'expected': round(genuine_stats[col]['mean'],2),
                'z_score' : round(float(z),2),
                'flag'    : bool(flag)
            }
            if flag: suspicious.append(col.replace('_',' ').title())

        votes_f = sum([
            1 if iso_p==-1 else 0,
            1 if svm_p==-1 else 0,
            1 if iso_s < iso_threshold else 0,
            1 if len(suspicious)>=2 else 0,
        ])

        if votes_f >= 2:
            verdict,color,icon = 'FORGED PATTERNS','forged','❌'
            reason = f"Suspicious behavioral patterns detected: {', '.join(suspicious) if suspicious else 'anomalous dynamics'}"
        else:
            verdict,color,icon = 'NATURAL WRITING','genuine','✅'
            reason = "Writing patterns are consistent with genuine signatures in the training dataset"

        conf = min(95, max(55, int(65 + iso_s*25)))

        return jsonify({
            'verdict':verdict,'color':color,'icon':icon,
            'reason':reason,'confidence':conf,
            'iso_result':'Natural' if iso_p==1 else 'Suspicious',
            'svm_result':'Natural' if svm_p==1 else 'Suspicious',
            'iso_score':round(iso_s,4),
            'suspicious':suspicious,'analysis':analysis,
        })
    except Exception as e:
        return jsonify({'error':str(e)})

# ── Tab 3: Personal Verification ─────────────────────────
@app.route('/enroll', methods=['POST'])
def enroll():
    try:
        data     = request.get_json(force=True)
        username = data.get('username','').strip()
        points   = data.get('points',[])
        if not username: return jsonify({'error':'Enter your name!'})
        if len(points)<20: return jsonify({'error':'Draw a longer signature!'})

        f = extract_features_pts(points)
        if username not in enrolled_users:
            enrolled_users[username] = {'samples':[],'iso':None,'svm':None,'scaler':None}
        enrolled_users[username]['samples'].append([f[c] for c in FEATURE_COLS])
        count = len(enrolled_users[username]['samples'])

        if count >= 5:
            X  = np.array(enrolled_users[username]['samples'])
            sc = StandardScaler(); Xs = sc.fit_transform(X)
            iso = IsolationForest(contamination=0.1,random_state=42); iso.fit(Xs)
            svm = OneClassSVM(kernel='rbf',nu=0.1,gamma='scale');     svm.fit(Xs)
            enrolled_users[username].update({'iso':iso,'svm':svm,'scaler':sc})
            return jsonify({'status':'trained','count':count,
                'message':f'✅ {username} enrolled! Model trained on {count} signatures.'})

        return jsonify({'status':'collecting','count':count,
            'message':f'Sample {count}/5 collected. {5-count} more needed.'})
    except Exception as e:
        return jsonify({'error':str(e)})

@app.route('/verify_personal', methods=['POST'])
def verify_personal():
    try:
        data     = request.get_json(force=True)
        username = data.get('username','')
        points   = data.get('points',[])
        if username not in enrolled_users: return jsonify({'error':'User not enrolled!'})
        if enrolled_users[username]['iso'] is None:
            n = len(enrolled_users[username]['samples'])
            return jsonify({'error':f'Need {5-n} more enrollment samples!'})
        if len(points)<20: return jsonify({'error':'Draw a longer signature!'})

        f    = extract_features_pts(points)
        X    = np.array([[f[c] for c in FEATURE_COLS]])
        X_sc = enrolled_users[username]['scaler'].transform(X)
        iso_p = int(enrolled_users[username]['iso'].predict(X_sc)[0])
        svm_p = int(enrolled_users[username]['svm'].predict(X_sc)[0])
        iso_s = float(enrolled_users[username]['iso'].score_samples(X_sc)[0])

        votes = sum([1 if iso_p==1 else 0, 1 if svm_p==1 else 0])
        if votes==2:   verdict,color,icon,reason = 'GENUINE','genuine','✅',f"Signature matches {username}'s biometric profile"
        elif votes==0: verdict,color,icon,reason = 'FORGED','forged','❌',f"Signature does NOT match {username}'s biometric profile"
        else:          verdict,color,icon,reason = 'UNCERTAIN','uncertain','⚠️',"Models disagree — partial match detected"

        conf = min(97,max(55,int(70+iso_s*30)))
        return jsonify({'verdict':verdict,'color':color,'icon':icon,
            'reason':reason,'confidence':conf,
            'iso_result':'Match' if iso_p==1 else 'No Match',
            'svm_result':'Match' if svm_p==1 else 'No Match'})
    except Exception as e:
        return jsonify({'error':str(e)})

@app.route('/get_users')
def get_users():
    return jsonify({'users':list(enrolled_users.keys())})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)