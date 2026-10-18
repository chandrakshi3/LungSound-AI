import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data_loader import discover_dataset
p=argparse.ArgumentParser();p.add_argument('--data-root',default='data/raw');a=p.parse_args();v,i,n=discover_dataset(a.data_root);print(f'Total WAVs: {n}\nValid: {len(v)}\nExcluded: {len(i)}');print(i.to_string(index=False) if len(i) else 'No validation issues detected.')
