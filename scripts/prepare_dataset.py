import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data_loader import discover_dataset
from src.segmentation import create_segments
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',default='data/raw');p.add_argument('--manifest-dir',default='data/manifests');a=p.parse_args();valid,issues,total=discover_dataset(a.data_root);out=Path(a.manifest_dir);out.mkdir(parents=True,exist_ok=True);valid.to_csv(out/'recordings.csv',index=False);issues.to_csv(out/'recording_issues.csv',index=False)
 print(f'WAV files found: {total}\nValid recordings: {len(valid)}\nExcluded/issues: {len(issues)}\nIdentifiable patients: {valid.patient_id.nunique() if len(valid) else 0}')
 if len(valid):
  seg,si=create_segments(valid);seg.to_csv(out/'segments.csv',index=False);si.to_csv(out/'segment_issues.csv',index=False);print(f'Annotated segments: {len(seg)}; segment issues: {len(si)}')
 else: print('No valid recordings. Add ICBHI WAV and annotation files; labels are never fabricated.')
if __name__=='__main__':main()
