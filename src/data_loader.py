from pathlib import Path
import pandas as pd
import soundfile as sf
COLS=['start','end','crackles','wheezes']
def parse_annotation(path):
    rows=[]
    with Path(path).open(encoding='utf-8-sig') as f:
        for no,line in enumerate(f,1):
            if not line.strip(): continue
            p=line.split()
            if len(p)!=4: raise ValueError(f'{path}:{no}: expected 4 columns')
            try: start,end=float(p[0]),float(p[1]); c,w=int(p[2]),int(p[3])
            except ValueError as e: raise ValueError(f'{path}:{no}: invalid values') from e
            if start<0 or end<=start or c not in (0,1) or w not in (0,1): raise ValueError(f'{path}:{no}: invalid boundary/label')
            rows.append((start,end,c,w))
    return pd.DataFrame(rows,columns=COLS)
def patient_id_from_name(path):
    stem=Path(path).stem
    return stem.split('_',1)[0] if '_' in stem else None
def discover_dataset(root):
    root=Path(root); ad=root/'audio'; nd=root/'annotations'
    if not ad.exists(): ad=root
    if not nd.exists(): nd=root
    valid=[]; issues=[]; wavs=sorted(ad.rglob('*.wav')) if ad.exists() else []
    for wav in wavs:
        ann=nd/(wav.stem+'.txt')
        if not ann.exists():
            matches=list(root.rglob(wav.stem+'.txt')); ann=matches[0] if matches else ann
        if not ann.exists(): issues.append({'recording_id':wav.stem,'status':'missing_annotation','detail':'No matching annotation'}); continue
        try:
            info=sf.info(str(wav)); labels=parse_annotation(ann)
            if labels.empty: raise ValueError('empty annotation file')
            if (labels.end>info.duration+.05).any(): raise ValueError('annotation exceeds audio duration')
            valid.append({'recording_id':wav.stem,'patient_id':patient_id_from_name(wav),'audio_path':str(wav.resolve()),'annotation_path':str(ann.resolve()),'duration':info.duration,'sample_rate':info.samplerate,'n_annotations':len(labels),'status':'valid'})
        except Exception as e: issues.append({'recording_id':wav.stem,'status':'invalid','detail':str(e)})
    return pd.DataFrame(valid),pd.DataFrame(issues),len(wavs)
