from pathlib import Path
import joblib,numpy as np
from .preprocessing import mfcc_features
from .aggregation import aggregate_scores
class ModelUnavailable(RuntimeError): pass
def load_baseline(path):
    p=Path(path)
    if not p.exists(): raise ModelUnavailable(f'No trained model at {p}. Run python scripts/train_all.py --baseline after preparing ICBHI data.')
    return joblib.load(p)
def predict_segments(model,segments,sr,threshold=.5):
    X=np.stack([mfcc_features(x,sr,model.get('n_mfcc',20)) for x in segments]); scores=[]
    for est in model['estimators']:
        if hasattr(est,'predict_proba'):
            p=est.predict_proba(X); classes=list(est.classes_); scores.append(p[:,classes.index(1)] if 1 in classes else np.zeros(len(X)))
        else: scores.append(est.predict(X).astype(float))
    arr=np.column_stack(scores); return arr,aggregate_scores(arr,threshold)
