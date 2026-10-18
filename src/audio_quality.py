import numpy as np
def assess_audio_quality(audio, sr, duration=None):
    x=np.asarray(audio,dtype=float).reshape(-1); warnings=[]
    if sr<=0: warnings.append('Invalid sample rate.')
    if not x.size: warnings.append('Audio is empty.')
    elif not np.all(np.isfinite(x)): warnings.append('Audio contains non-finite samples.')
    else:
        peak=float(np.max(np.abs(x))); rms=float(np.sqrt(np.mean(x*x)))
        if rms<1e-4: warnings.append('Audio is silent or extremely quiet; do not interpret as normal.')
        if peak>=.999: warnings.append('Audio may be clipped.')
        if duration is not None and duration<.25: warnings.append('Recording is unusually short.')
        if rms>0 and peak/rms>30: warnings.append('High peak-to-RMS ratio; inspect for noise (heuristic only).')
    return {'ok':not warnings,'warnings':warnings}
