# College demo guide
## Preparation
1. Install dependencies and obtain the official ICBHI dataset.
2. Put WAVs in `data/raw/audio` and matching annotation files in `data/raw/annotations`.
3. Run `python scripts/validate_dataset.py` then `python scripts/prepare_dataset.py`.
4. Train with `python scripts/train_all.py --baseline`; inspect the real report.
5. Run `pytest -q`, then `streamlit run app.py`.
## Demo sequence
Home page → dataset scan and exclusions → upload an actual WAV → view playback, waveform, log-Mel plot → select a trained baseline and analyze → explain crackles/wheezes scores and highlighted windows → show actual saved metrics.
## Viva questions
**Why two outputs?** Crackles and wheezes can occur independently or together.
**What is MFCC?** A compact representation of the spectral envelope of audio.
**Why group split?** Segments from one patient can be similar and cause leakage across train/test.
**Does Normal mean healthy?** No, only that both model labels fell below a threshold.
**Are scores medical probabilities?** Not unless calibrated and clinically validated.
**What does highlighting prove?** Only that model output crossed a threshold in that window.
