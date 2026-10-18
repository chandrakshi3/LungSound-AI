from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn.multioutput import MultiOutputClassifier
from .segmentation import read_segment
from .preprocessing import mfcc_features
def build_matrix(manifest,target_sr=22050,n_mfcc=20):
    X=[]; y=[]; good=[]; errors=[]
    for r in manifest.to_dict('records'):
        try:
            a,sr=read_segment(r,target_sr); X.append(mfcc_features(a,sr,n_mfcc)); y.append([int(r['crackles']),int(r['wheezes'])]); good.append(r)
        except Exception as e: errors.append({'recording_id':r.get('recording_id'),'error':str(e)})
    if not X: raise ValueError('No valid segments could be converted into features')
    return np.stack(X),np.asarray(y),pd.DataFrame(good),pd.DataFrame(errors)
def group_split(df,test_size=.2,seed=42):
    groups=df.patient_id.fillna(df.recording_id).astype(str).to_numpy()
    if len(set(groups))<2: raise ValueError('At least two patient/recording groups are needed')
    tr,te=next(GroupShuffleSplit(n_splits=1,test_size=test_size,random_state=seed).split(df,groups=groups))
    if set(groups[tr]) & set(groups[te]): raise RuntimeError('Group leakage detected')
    return tr,te
def train_baselines(X,y,rows,artifact_dir,seed=42,test_size=.2):
    tr,te=group_split(rows,test_size,seed); out=Path(artifact_dir);out.mkdir(parents=True,exist_ok=True)
    models={'random_forest':MultiOutputClassifier(RandomForestClassifier(n_estimators=250,class_weight='balanced',random_state=seed,n_jobs=-1)), 'svm':MultiOutputClassifier(make_pipeline(StandardScaler(),SVC(probability=True,class_weight='balanced',random_state=seed)))}
    reports=[]
    for name,m in models.items():
        m.fit(X[tr],y[tr]); pred=m.predict(X[te]);
        reports.append({'model':name,'test_segments':int(len(te)),'group_split':True,'crackles_f1':float(f1_score(y[te,0],pred[:,0],zero_division=0)),'wheezes_f1':float(f1_score(y[te,1],pred[:,1],zero_division=0)),'crackles_precision':float(precision_score(y[te,0],pred[:,0],zero_division=0)),'crackles_recall':float(recall_score(y[te,0],pred[:,0],zero_division=0)),'wheezes_precision':float(precision_score(y[te,1],pred[:,1],zero_division=0)),'wheezes_recall':float(recall_score(y[te,1],pred[:,1],zero_division=0))})
        joblib.dump({'name':name,'estimators':list(m.estimators_),'n_mfcc':20,'threshold':.5,'trained':True},out/f'{name}.joblib')
    splits=rows.copy(); splits['split']='train'; splits.iloc[te,splits.columns.get_loc('split')]='test'; splits.to_csv(out/'split_assignments.csv',index=False)
    (out/'metrics.json').write_text(json.dumps({'seed':seed,'split_method':'GroupShuffleSplit by patient ID, recording ID fallback','models':reports},indent=2),encoding='utf-8')
    return reports
