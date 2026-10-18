import matplotlib.pyplot as plt
import librosa,librosa.display
def audio_figures(audio,sr):
    f1,ax=plt.subplots(figsize=(10,2.5));librosa.display.waveshow(audio,sr=sr,ax=ax);ax.set_title('Waveform');ax.set_xlabel('Time (s)')
    f2,ax=plt.subplots(figsize=(10,3.5));s=librosa.feature.melspectrogram(y=audio,sr=sr,n_mels=64,n_fft=1024,hop_length=256);db=librosa.power_to_db(s,ref=s.max());im=librosa.display.specshow(db,sr=sr,hop_length=256,x_axis='time',y_axis='mel',ax=ax);f2.colorbar(im,ax=ax,format='%+2.0f dB');ax.set_title('Log-Mel spectrogram')
    return f1,f2
