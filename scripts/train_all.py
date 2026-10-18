import argparse,sys,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.train_baseline import build_matrix,train_baselines
p=argparse.ArgumentParser();p.add_argument('--baseline',action='store_true');p.add_argument('--data-root',default='data/raw');p.add_argument('--manifest',default='data/manifests/segments.csv');p.add_argument('--artifact-dir',default='artifacts/baseline');p.add_argument('--seed',type=int,default=42);a=p.parse_args()
if not a.baseline:p.error('Specify --baseline')
f=Path(a.manifest)
if not f.exists(): subprocess.run([sys.executable,'scripts/prepare_dataset.py','--data-root',a.data_root],check=True)
if not f.exists() or f.stat().st_size==0: raise SystemExit('No segment manifest. Download and place ICBHI dataset first.')
X,y,rows,errors=build_matrix(pd.read_csv(f));print(f'Feature rows: {len(X)}; excluded feature rows: {len(errors)}');print(pd.DataFrame(train_baselines(X,y,rows,a.artifact_dir,a.seed)).to_string(index=False))
