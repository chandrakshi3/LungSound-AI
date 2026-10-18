import io
import numpy as np
import soundfile as sf
import pytest
from src.preprocessing import load_audio,log_mel,mfcc_features,window_audio
def wav_bytes(x,sr=16000):
 b=io.BytesIO();sf.write(b,x,sr,format='WAV');return b.getvalue()
def test_load_mono_resample():
 x=np.zeros((16000,2),dtype=np.float32);x[:,0]=.1;r=load_audio(wav_bytes(x),22050);assert r.sample_rate==22050 and r.audio.ndim==1 and abs(r.duration-1)<.02
def test_invalid_audio():
 with pytest.raises(ValueError):load_audio(b'not wav')
def test_silence_warning():assert load_audio(wav_bytes(np.zeros(8000)),16000).quality['warnings']
def test_features():
 x=np.sin(2*np.pi*200*np.arange(16000)/16000).astype('float32');assert mfcc_features(x,16000).shape==(80,);assert log_mel(x,16000).ndim==2
def test_windows():assert len(window_audio(np.zeros(12000),1000,5,.2))>=2
