import pandas as pd
import soundfile as sf
from .data_loader import parse_annotation
def category(c,w): return 'Both' if c and w else 'Crackles' if c else 'Wheezes' if w else 'Normal'
def create_segments(recordings_df):
    rows=[]; issues=[]
    for r in recordings_df.to_dict('records'):
        try:
            ann=parse_annotation(r['annotation_path']); duration=sf.info(r['audio_path']).duration
            for i,a in ann.iterrows():
                if a.start<0 or a.end<=a.start or a.end>duration+.05: issues.append({'recording_id':r['recording_id'],'row':i,'status':'invalid_boundary'}); continue
                rows.append({'recording_id':r['recording_id'],'patient_id':r.get('patient_id'),'audio_path':r['audio_path'],'start':float(a.start),'end':float(a.end),'crackles':int(a.crackles),'wheezes':int(a.wheezes),'category':category(int(a.crackles),int(a.wheezes))})
        except Exception as e: issues.append({'recording_id':r.get('recording_id'),'status':'annotation_error','detail':str(e)})
    return pd.DataFrame(rows),pd.DataFrame(issues)
def read_segment(row,target_sr=22050):
    import librosa
    sr=sf.info(row['audio_path']).samplerate
    x,_=sf.read(row['audio_path'],start=int(row['start']*sr),stop=int(row['end']*sr),dtype='float32',always_2d=True)
    x=x.mean(axis=1)
    return (librosa.resample(x,orig_sr=sr,target_sr=target_sr) if sr!=target_sr else x),target_sr
