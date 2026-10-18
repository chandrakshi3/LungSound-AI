# LungSound AI — Explainable Respiratory Sound Classification
Educational research prototype analyzing WAV recordings for two labels: crackles and wheezes. It derives Normal / Crackles / Wheezes / Both. **Not a medical device; not a diagnosis or treatment tool.**

## Status
Project code, dataset ingestion/validation, preprocessing, baseline training, optional CNN trainer, dashboard, docs and tests are included. ICBHI data and trained model artifacts are not bundled. No evaluation results are fabricated. CNN inference is not yet connected to the dashboard; baseline inference is supported.

## Windows PowerShell install
```powershell
cd LungSound-AI
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If activation is blocked, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` for the current terminal.

## Dataset setup
Download the ICBHI 2017 Respiratory Sound Database from https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge and follow access terms. Extract WAV files into `data/raw/audio/` and matching `.txt` cycle annotations into `data/raw/annotations/`. Each annotation line is `start end crackles wheezes`.
```powershell
python scripts/validate_dataset.py --data-root data/raw
python scripts/prepare_dataset.py --data-root data/raw
```
Review `data/manifests/recordings.csv`, `recording_issues.csv`, `segments.csv`, and `segment_issues.csv`. Missing labels are not assumed to be normal.

## Train and evaluate
```powershell
python scripts/train_all.py --baseline
python scripts/evaluate_models.py
```
The script compares Random Forest and SVM using MFCC summary features and a group-held-out split. Patient IDs are inferred from recording names; recording ID is a fallback if needed. The split is a starter held-out evaluation, not a full train/validation/final-test protocol. Select thresholds using validation data before reporting research results. Metrics are saved in `artifacts/baseline/metrics.json` and assignments in `split_assignments.csv`.

## Run dashboard
```powershell
streamlit run app.py
```
Pages include overview, audio analysis, dataset explorer, model performance and project documentation. Uploads are processed in memory and not saved by default. Default limits: 25 MB and 120 seconds.

## Optional CNN
`src/train_cnn.py` provides CPU-compatible PyTorch CNN training on log-Mel spectrograms with patient/recording-group validation and early stopping. Use `train_cnn(rows, 'artifacts/cnn/lungsound_cnn.pt', epochs=10)` with rows from `data/manifests/segments.csv`. CNN inference is not wired into the Streamlit selector yet.

## Tests
```powershell
pytest -q
```
Tests use synthetic signals only to validate software behavior, not as respiratory data or benchmark results.

## Privacy and responsible use
Do not commit dataset files or patient-identifying information. Audio is not sent to external services. Model outputs do not confirm disease. Quality checks are heuristic only.
