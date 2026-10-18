from dataclasses import dataclass
import io
import numpy as np
import librosa
import soundfile as sf
from .audio_quality import assess_audio_quality
@dataclass
class AudioRecord:
    audio: np.ndarray
    sample_rate: int
    duration: float
    quality: dict
def load_audio(source,target_sr=22050,min_duration=.25,max_duration=120):
    try:
        data,sr=sf.read(io.BytesIO(source) if isinstance(source,(bytes,bytearray)) else str(source),dtype='float32',always_2d=True)
    except Exception as e: raise ValueError(f'Could not read WAV audio: {e}') from e
    if data.size==0 or sr<=0: raise ValueError('Audio is empty or sample rate is invalid.')
    if not np.all(np.isfinite(data)): raise ValueError('Audio contains NaN or infinite samples.')
    audio=data.mean(axis=1); duration=len(audio)/sr
    if duration<min_duration: raise ValueError(f'Audio too short ({duration:.2f}s).')
    if duration>max_duration: raise ValueError(f'Audio too long ({duration:.1f}s).')
    if sr!=target_sr: audio=librosa.resample(audio,orig_sr=sr,target_sr=target_sr)
    audio=np.asarray(audio,dtype=np.float32); peak=float(np.max(np.abs(audio))) if audio.size else 0
    if peak>1: audio=audio/peak
    duration=len(audio)/target_sr
    return AudioRecord(audio,target_sr,duration,assess_audio_quality(audio,target_sr,duration))
def log_mel(audio,sr,n_mels=64,n_fft=1024,hop_length=256):
    s=librosa.feature.melspectrogram(y=np.asarray(audio,dtype=float),sr=sr,n_mels=n_mels,n_fft=n_fft,hop_length=hop_length,power=2)
    return librosa.power_to_db(s,ref=np.max).astype(np.float32)
def mfcc_features(audio,sr,n_mfcc=20):
    m=librosa.feature.mfcc(y=np.asarray(audio,dtype=float),sr=sr,n_mfcc=n_mfcc)
    d=librosa.feature.delta(m) if m.shape[1]>2 else np.zeros_like(m)
    return np.concatenate([m.mean(1),m.std(1),d.mean(1),d.std(1)]).astype(np.float32)
def window_audio(audio,sr,window_seconds=5,overlap=.2,min_seconds=.25):
    n=len(audio); win=max(1,int(window_seconds*sr)); step=max(1,int(win*(1-overlap)))
    if n<=win: return [(0.,n/sr,np.asarray(audio))] if n/sr>=min_seconds else []
    out=[]
    for start in range(0,n-win+1,step): out.append((start/sr,(start+win)/sr,np.asarray(audio[start:start+win])))
    if out and out[-1][1]<n/sr and n/sr-out[-1][1]>.5: out.append((out[-1][1],n/sr,np.asarray(audio[int(out[-1][1]*sr):])))
    return [(a,b,x) for a,b,x in out if len(x)/sr>=min_seconds]
