from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.aggregation import category as derive_category
from src.config import load_config
from src.data_loader import discover_dataset
from src.predict import ModelUnavailable, load_baseline, predict_segments
from src.preprocessing import load_audio, log_mel, window_audio

CFG = load_config()
ARTIFACTS = ROOT / "artifacts" / "baseline"
MANIFESTS = ROOT / "data" / "manifests"

st.set_page_config(
    page_title="LungSound AI | Respiratory Sound Lab",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Streamlit-safe styling: fixed high-contrast palette plus restrained cards.
st.markdown(
    """
    <style>
    :root { color-scheme: light; }
    html, body, [data-testid="stAppViewContainer"], .stApp,
    [data-testid="stMain"], section[data-testid="stMain"] > div {
      background: #f4f8fb !important; color: #142b3b !important;
    }
    [data-testid="stHeader"] { background: rgba(244,248,251,.96) !important; }
    .block-container { max-width: 1380px; padding-top: 1.35rem; padding-bottom: 3rem; }
    .stApp, .stApp p, .stApp li, .stApp label, .stApp small,
    .stApp [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stMarkdownContainer"] p,
    .stApp [data-testid="stMarkdownContainer"] li,
    .stApp [data-testid="stWidgetLabel"] p,
    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stMetricLabel"],
    .stApp [data-testid="stMetricValue"], .stApp [data-testid="stMetricDelta"] {
      color: #193447 !important;
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 { color: #102d40 !important; letter-spacing: -.025em; }
    .stApp [data-testid="stCaptionContainer"] { color: #536b7b !important; }
    section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div {
      background: #eaf1f6 !important; color: #193447 !important;
      border-right: 1px solid #d5e1e9;
    }
    section[data-testid="stSidebar"] * { color: #193447 !important; }
    .stButton > button, .stDownloadButton > button {
      background: #087e8b !important; color: #fff !important;
      border: 1px solid #087e8b !important; border-radius: 10px !important;
      font-weight: 650 !important; min-height: 2.65rem;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
      background: #06636d !important; border-color: #06636d !important; color: #fff !important;
    }
    .stTextInput input, .stNumberInput input, .stTextArea textarea,
    .stSelectbox [data-baseweb="select"], .stMultiSelect [data-baseweb="select"] {
      background: #fff !important; color: #193447 !important; border-color: #bccdd8 !important;
    }
    .stApp [data-testid="stAlert"] { border-radius: 12px; }
    .stApp [data-testid="stAlert"] p { color: #193447 !important; }
    .stApp [data-testid="stMetric"] {
      background: #fff; border: 1px solid #d9e4eb; border-radius: 14px;
      padding: .9rem 1rem; box-shadow: 0 2px 8px rgba(26,57,77,.035);
    }
    .stApp [data-testid="stDataFrame"], .stApp [data-testid="stTable"] {
      border: 1px solid #d9e4eb; border-radius: 12px; overflow: hidden;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] { border-color: #d9e4eb !important; border-radius: 14px !important; }
    hr { border-color: #d9e4eb !important; }
    .ls-eyebrow { color:#087e8b; font-size:.76rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; }
    .ls-hero { background:linear-gradient(120deg,#10394b 0%,#087e8b 100%); padding:1.65rem 1.8rem;
      border-radius:20px; color:white; margin:.25rem 0 1.1rem 0; box-shadow:0 10px 28px rgba(8,126,139,.13); }
    .ls-hero h1 { color:white !important; margin:0; font-size:2.1rem; line-height:1.2; }
    .ls-hero p { color:#e1f7f8 !important; margin:.55rem 0 0 0; max-width:850px; }
    .ls-card { background:#fff; border:1px solid #d9e4eb; border-radius:15px; padding:1rem 1.1rem; height:100%; }
    .ls-card h3 { font-size:1.03rem; margin:.25rem 0 .45rem 0; }
    .ls-card p { color:#526b7b !important; font-size:.91rem; line-height:1.5; margin:0; }
    .ls-chip { display:inline-block; padding:.25rem .55rem; border-radius:999px; background:#e3f5f4; color:#076b72;
      font-size:.75rem; font-weight:800; letter-spacing:.025em; }
    .ls-muted { color:#526b7b !important; }
    .ls-footer { margin-top:2rem; padding-top:1rem; border-top:1px solid #d9e4eb; color:#607687; font-size:.82rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except Exception:
        return None


def available_models() -> dict[str, Path]:
    found = {}
    for label, filename in [("Random Forest", "random_forest.joblib"), ("SVM", "svm.joblib")]:
        path = ARTIFACTS / filename
        if path.exists():
            found[label] = path
    return found


def model_status_text() -> str:
    count = len(available_models())
    if count == 2:
        return "2 baseline models available"
    if count == 1:
        return "1 baseline model available"
    return "No trained baseline model found"


def hero(title: str, subtitle: str, eyebrow: str = "LUNGSOUND AI · RESEARCH LAB") -> None:
    st.markdown(
        f'<div class="ls-hero"><div class="ls-eyebrow" style="color:#a9eff0">{eyebrow}</div>'
        f'<h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def section_title(title: str, subtitle: str | None = None) -> None:
    st.subheader(title)
    if subtitle:
        st.caption(subtitle)


def category_counts(segments: pd.DataFrame) -> pd.DataFrame:
    if segments.empty:
        return pd.DataFrame(columns=["Category", "Segments"])
    df = segments.copy()
    if "category" not in df.columns and {"crackles", "wheezes"}.issubset(df.columns):
        df["category"] = [derive_category(bool(c), bool(w)) for c, w in zip(df.crackles, df.wheezes)]
    if "category" not in df.columns:
        return pd.DataFrame(columns=["Category", "Segments"])
    names = ["Normal", "Crackles", "Wheezes", "Both"]
    counts = df["category"].value_counts()
    return pd.DataFrame({"Category": names, "Segments": [int(counts.get(n, 0)) for n in names]})


def style_plot(fig: go.Figure, height: int = 330) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=38, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font=dict(family="Inter, Arial, sans-serif", color="#193447", size=12),
        title_font=dict(size=15, color="#193447"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(gridcolor="#edf2f5", zerolinecolor="#edf2f5"),
        yaxis=dict(gridcolor="#edf2f5", zerolinecolor="#edf2f5"),
    )
    return fig


def waveform_figure(audio: np.ndarray, sr: int) -> go.Figure:
    stride = max(1, int(math.ceil(len(audio) / 12000)))
    x = np.arange(0, len(audio), stride) / sr
    y = audio[::stride]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Waveform", line=dict(color="#087e8b", width=1)))
    fig.update_layout(title="Waveform · amplitude over time", xaxis_title="Time (seconds)", yaxis_title="Amplitude")
    return style_plot(fig, 270)


def spectrogram_figure(audio: np.ndarray, sr: int) -> go.Figure:
    mel = log_mel(audio, sr, n_mels=int(CFG["audio"].get("n_mels", 64)))
    stride = max(1, int(math.ceil(mel.shape[1] / 1400)))
    mel = mel[:, ::stride]
    x = np.arange(mel.shape[1]) * 256 * stride / sr
    y = np.arange(mel.shape[0])
    fig = go.Figure(data=go.Heatmap(
        z=mel, x=x, y=y, colorscale="Viridis", colorbar=dict(title="dB", thickness=12),
        hovertemplate="Time: %{x:.2f} s<br>Mel bin: %{y}<br>Level: %{z:.1f} dB<extra></extra>",
    ))
    fig.update_layout(title="Log-Mel spectrogram", xaxis_title="Time (seconds)", yaxis_title="Mel frequency bin")
    return style_plot(fig, 340)


def scores_figure(windows: list, scores: np.ndarray, threshold: float) -> go.Figure:
    starts = [float(w[0]) for w in windows]
    ends = [float(w[1]) for w in windows]
    centers = [(a + b) / 2 for a, b in zip(starts, ends)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=centers, y=scores[:, 0], mode="lines+markers", name="Crackles", line=dict(color="#0b8793", width=2), marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=centers, y=scores[:, 1], mode="lines+markers", name="Wheezes", line=dict(color="#8a63bf", width=2), marker=dict(size=6)))
    fig.add_hline(y=threshold, line_dash="dash", line_color="#da7b38", annotation_text=f"Threshold {threshold:.2f}")
    fig.update_yaxes(range=[0, 1], title="Model score")
    fig.update_xaxes(title="Window midpoint (seconds)")
    fig.update_layout(title="Window-level model scores")
    return style_plot(fig, 330)


def metrics_summary(metrics: dict | None) -> pd.DataFrame:
    if not metrics:
        return pd.DataFrame()
    models = metrics.get("models", [])
    if isinstance(models, list):
        return pd.DataFrame(models)
    if isinstance(models, dict):
        return pd.DataFrame([{"model": key, **value} for key, value in models.items() if isinstance(value, dict)])
    return pd.DataFrame()


recordings_df = read_csv(MANIFESTS / "recordings.csv")
segments_df = read_csv(MANIFESTS / "segments.csv")
metrics_json = read_json(ARTIFACTS / "metrics.json")
models_available = available_models()

with st.sidebar:
    st.markdown("<div style='font-size:2.4rem;line-height:1'>🫁</div>", unsafe_allow_html=True)
    st.markdown("## LungSound AI")
    st.caption("Respiratory sound research workbench")
    st.divider()
    page = st.radio(
        "WORKSPACE",
        ["Overview", "Analyze audio", "Dataset explorer", "Model performance", "Methodology & viva"],
        label_visibility="visible",
    )
    st.divider()
    st.markdown("**SYSTEM STATUS**")
    st.caption(f"Models · {len(models_available)}/2 available")
    st.caption(f"Dataset · {len(recordings_df)} recordings in manifest")
    if models_available:
        st.success("Model artifacts detected")
    else:
        st.warning("Train baseline models first")
    st.markdown(
        "<div class='ls-footer'>For education and research only. Not a medical device; outputs are not diagnoses.</div>",
        unsafe_allow_html=True,
    )

if page == "Overview":
    hero(
        "Listen to the signal. Explore the pattern.",
        "An explainable respiratory-audio workspace for inspecting WAV recordings, visualizing sound features, and reviewing model predictions for crackles and wheezes.",
    )
    n_recordings = len(recordings_df)
    n_segments = len(segments_df)
    n_patients = int(recordings_df["patient_id"].nunique()) if not recordings_df.empty and "patient_id" in recordings_df.columns else 0
    c = st.columns(4)
    c[0].metric("Recordings prepared", f"{n_recordings:,}")
    c[1].metric("Annotated cycles", f"{n_segments:,}")
    c[2].metric("Identifiable patients", f"{n_patients:,}")
    c[3].metric("Trained baseline models", f"{len(models_available)}/2")

    st.markdown("### Workspace")
    cards = st.columns(3)
    for col, icon, title, desc in [
        (cards[0], "01", "Analyze audio", "Upload a WAV recording, inspect waveform and spectrogram, then score overlapping windows with a trained baseline model."),
        (cards[1], "02", "Explore the dataset", "Review prepared recordings, annotation volume, category balance, and data-quality exclusions."),
        (cards[2], "03", "Review model performance", "See the evaluation report saved by the training pipeline. Metrics are shown only when an actual report exists."),
    ]:
        with col:
            st.markdown(f"<div class='ls-card'><span class='ls-chip'>{icon} · WORKSPACE</span><h3>{title}</h3><p>{desc}</p></div>", unsafe_allow_html=True)

    st.markdown("### Training data at a glance")
    if not segments_df.empty:
        counts = category_counts(segments_df)
        p1, p2 = st.columns([1.15, 1])
        with p1:
            fig = go.Figure(go.Bar(x=counts["Category"], y=counts["Segments"], marker_color=["#5c9b87", "#0b8793", "#8a63bf", "#da8c42"], text=counts["Segments"], textposition="outside"))
            fig.update_layout(title="Annotated cycle distribution", xaxis_title="Dataset label", yaxis_title="Segments")
            st.plotly_chart(style_plot(fig, 320), use_container_width=True)
        with p2:
            st.markdown("<div class='ls-card'><h3>How to read the four categories</h3><p><b>Normal</b> means neither sound label is present in the annotation.<br><br><b>Crackles</b> or <b>Wheezes</b> means only that label is present.<br><br><b>Both</b> means both labels are present.<br><br>These are sound annotations, not disease diagnoses.</p></div>", unsafe_allow_html=True)
    else:
        st.info("No prepared manifest was found. Add the WAV recordings and matching annotations, then run the dataset preparation script.")

    if metrics_json:
        df_metrics = metrics_summary(metrics_json)
        if not df_metrics.empty:
            st.markdown("### Last saved baseline evaluation")
            show_cols = [c for c in ["model", "test_segments", "crackles_f1", "wheezes_f1"] if c in df_metrics.columns]
            st.dataframe(df_metrics[show_cols], use_container_width=True, hide_index=True)
    st.markdown("<div class='ls-footer'>LungSound AI · Explainable Respiratory Sound Classification · Local processing only</div>", unsafe_allow_html=True)

elif page == "Analyze audio":
    hero("Analyze a respiratory recording", "Inspect audio quality and signal structure before viewing predictions. Predictions require an existing trained baseline artifact.", "LUNGSOUND AI · AUDIO ANALYSIS")
    left, right = st.columns([1.45, 1])
    with left:
        uploaded = st.file_uploader("Choose a WAV recording", type=["wav"], help="Accepted format: WAV. Files are processed in memory and are not permanently saved by this app.")
    with right:
        max_mb = float(CFG["audio"].get("max_upload_mb", 25))
        max_seconds = float(CFG["audio"].get("max_duration_seconds", 120))
        st.markdown(f"<div class='ls-card'><span class='ls-chip'>INPUT LIMITS</span><h3>Recording requirements</h3><p>WAV audio only<br>Maximum size: {max_mb:g} MB<br>Maximum duration: {max_seconds:g} seconds<br>Local analysis: no external audio upload</p></div>", unsafe_allow_html=True)

    if uploaded is None:
        st.markdown("### Analysis flow")
        cols = st.columns(3)
        for col, n, title, desc in [
            (cols[0], "01", "Load", "The file is checked for validity, duration, silence, and clipping."),
            (cols[1], "02", "Visualize", "Waveform and log-Mel spectrogram help inspect the acoustic signal."),
            (cols[2], "03", "Score", "A trained baseline scores time windows for crackles and wheezes."),
        ]:
            with col:
                st.markdown(f"<div class='ls-card'><span class='ls-chip'>{n}</span><h3>{title}</h3><p>{desc}</p></div>", unsafe_allow_html=True)
        st.info("Upload a WAV file to begin. Audio visualization works without a trained model; classification does not.")
    else:
        if uploaded.size > max_mb * 1024 * 1024:
            st.error(f"This file is larger than the configured {max_mb:g} MB upload limit.")
            st.stop()
        audio_bytes = uploaded.getvalue()
        upload_key = hashlib.sha256(audio_bytes).hexdigest()
        try:
            rec = load_audio(
                audio_bytes,
                CFG["audio"]["target_sample_rate"],
                CFG["audio"]["min_duration_seconds"],
                CFG["audio"]["max_duration_seconds"],
            )
        except Exception as exc:
            st.error(f"Could not process this audio file: {exc}")
            st.stop()

        st.audio(audio_bytes, format="audio/wav")
        meta = st.columns(4)
        meta[0].metric("Duration", f"{rec.duration:.2f} s")
        meta[1].metric("Analysis sample rate", f"{rec.sample_rate:,} Hz")
        meta[2].metric("Channels", "Mono after preprocessing")
        meta[3].metric("Audio quality", "Review needed" if rec.quality["warnings"] else "Basic checks passed")
        for warning in rec.quality["warnings"]:
            st.warning(warning)
        if not rec.quality["warnings"]:
            st.success("Basic signal checks passed. These heuristics do not establish clinical recording quality.")

        tab_wave, tab_spec = st.tabs(["Waveform", "Log-Mel spectrogram"])
        with tab_wave:
            st.plotly_chart(waveform_figure(rec.audio, rec.sample_rate), use_container_width=True)
        with tab_spec:
            st.plotly_chart(spectrogram_figure(rec.audio, rec.sample_rate), use_container_width=True)

        st.markdown("### Model analysis")
        if not models_available:
            st.warning("No trained baseline artifacts were found. Train the models before requesting classification.")
            st.code("python scripts\\train_all.py --baseline", language="powershell")
        else:
            choice = st.selectbox("Select a trained model", list(models_available.keys()))
            threshold = st.slider("Research threshold", min_value=0.10, max_value=0.90, value=float(CFG.get("model", {}).get("threshold", 0.5)), step=0.05, help="This is a configurable research threshold, not a clinically validated cutoff.")
            analyze = st.button("Analyze recording", type="primary", use_container_width=True)
            if analyze:
                try:
                    model = load_baseline(models_available[choice])
                    windows = window_audio(rec.audio, rec.sample_rate, 5, 0.2)
                    if not windows:
                        raise ValueError("No valid analysis windows could be created from this recording.")
                    scores, summary = predict_segments(model, [w[2] for w in windows], rec.sample_rate, threshold=threshold)
                    # predict_segments uses the threshold for its aggregated result; display and export all values.
                    summary["model"] = choice
                    summary["recording_name"] = uploaded.name
                    summary["threshold"] = float(threshold)
                    summary["window_count"] = int(len(windows))
                    rows = [{
                        "start_s": round(float(w[0]), 3),
                        "end_s": round(float(w[1]), 3),
                        "crackles_score": float(s[0]),
                        "wheezes_score": float(s[1]),
                        "elevated_crackles": bool(s[0] >= threshold),
                        "elevated_wheezes": bool(s[1] >= threshold),
                    } for w, s in zip(windows, scores)]
                    st.session_state["last_analysis"] = {"key": upload_key, "summary": summary, "rows": rows, "windows": windows, "scores": scores, "warnings": rec.quality["warnings"]}
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")
                    st.stop()

            result = st.session_state.get("last_analysis")
            if result and result.get("key") == upload_key:
                summary = result["summary"]
                st.markdown("#### Output summary")
                st.caption("Sound-pattern estimates from the selected model. They do not identify or confirm a disease.")
                out = st.columns(3)
                out[0].metric("Predicted sound category", summary["category"])
                out[1].metric("Crackles model score", f"{summary['crackles_score']:.3f}")
                out[2].metric("Wheezes model score", f"{summary['wheezes_score']:.3f}")
                st.plotly_chart(scores_figure(result["windows"], result["scores"], float(summary["threshold"])), use_container_width=True)
                df_rows = pd.DataFrame(result["rows"])
                st.markdown("#### Window-level details")
                st.dataframe(df_rows, use_container_width=True, hide_index=True)
                payload = {"summary": summary, "segments": result["rows"], "quality_warnings": result["warnings"], "disclaimer": "Educational research prototype; not a diagnosis."}
                dl1, dl2 = st.columns(2)
                dl1.download_button("Download JSON report", json.dumps(payload, indent=2), file_name="lungsound_analysis.json", mime="application/json", use_container_width=True)
                dl2.download_button("Download window scores (CSV)", df_rows.to_csv(index=False), file_name="lungsound_window_scores.csv", mime="text/csv", use_container_width=True)

elif page == "Dataset explorer":
    hero("Explore the research dataset", "Inspect what was loaded and prepared for training. Counts are read from the local manifests, not assumed.", "LUNGSOUND AI · DATASET")
    if st.button("Refresh dataset scan", type="primary"):
        try:
            valid, issues, total = discover_dataset(ROOT / "data" / "raw")
            MANIFESTS.mkdir(parents=True, exist_ok=True)
            valid.to_csv(MANIFESTS / "recordings.csv", index=False)
            issues.to_csv(MANIFESTS / "recording_issues.csv", index=False)
            st.session_state["dataset_scan_summary"] = {"total": int(total), "valid": int(len(valid)), "issues": int(len(issues))}
            st.rerun()
        except Exception as exc:
            st.error(f"Dataset scan failed: {exc}")

    recordings_df = read_csv(MANIFESTS / "recordings.csv")
    segments_df = read_csv(MANIFESTS / "segments.csv")
    issues_df = read_csv(MANIFESTS / "recording_issues.csv")
    patients = int(recordings_df["patient_id"].nunique()) if not recordings_df.empty and "patient_id" in recordings_df.columns else 0
    k = st.columns(4)
    k[0].metric("Valid recordings", f"{len(recordings_df):,}")
    k[1].metric("Annotated cycles", f"{len(segments_df):,}")
    k[2].metric("Identifiable patients", f"{patients:,}")
    k[3].metric("Recording issues", f"{len(issues_df):,}")

    if not segments_df.empty:
        counts = category_counts(segments_df)
        a, b = st.columns([1, 1.25])
        with a:
            fig = go.Figure(go.Pie(labels=counts["Category"], values=counts["Segments"], hole=0.62, marker=dict(colors=["#5c9b87", "#0b8793", "#8a63bf", "#da8c42"])))
            fig.update_layout(title="Annotation category mix", showlegend=True)
            st.plotly_chart(style_plot(fig, 340), use_container_width=True)
        with b:
            fig = go.Figure(go.Bar(x=counts["Category"], y=counts["Segments"], marker_color=["#5c9b87", "#0b8793", "#8a63bf", "#da8c42"], text=counts["Segments"], textposition="outside"))
            fig.update_layout(title="Annotated respiratory cycles", yaxis_title="Count")
            st.plotly_chart(style_plot(fig, 340), use_container_width=True)
        with st.expander("View prepared segment manifest"):
            st.dataframe(segments_df.head(1000), use_container_width=True, hide_index=True)
    else:
        st.info("No prepared segment manifest found. Run `python scripts/prepare_dataset.py --data-root data/raw` first.")
    with st.expander("View recording manifest"):
        st.dataframe(recordings_df, use_container_width=True, hide_index=True)
    if not issues_df.empty:
        st.warning("Some recordings were excluded or need review.")
        st.dataframe(issues_df, use_container_width=True, hide_index=True)
    else:
        st.success("No recording-level issues are present in the saved manifest." if not recordings_df.empty else "No dataset manifest is available yet.")

elif page == "Model performance":
    hero("Evaluate the models honestly", "Review the latest saved holdout metrics. This page never substitutes placeholder or illustrative scores for actual evaluation output.", "LUNGSOUND AI · EVALUATION")
    metrics_json = read_json(ARTIFACTS / "metrics.json")
    df_metrics = metrics_summary(metrics_json)
    if metrics_json and not df_metrics.empty:
        st.markdown("### Saved evaluation summary")
        st.caption(f"Split method: {metrics_json.get('split_method', 'See saved training report')}. Seed: {metrics_json.get('seed', 'not recorded')}.")
        st.dataframe(df_metrics, use_container_width=True, hide_index=True)
        numeric = [c for c in ["crackles_f1", "wheezes_f1"] if c in df_metrics.columns]
        if numeric:
            cols = st.columns(len(numeric))
            for col, metric in zip(cols, numeric):
                best = pd.to_numeric(df_metrics[metric], errors="coerce").max()
                col.metric(f"Best saved {metric.replace('_', ' ')}", "—" if pd.isna(best) else f"{best:.3f}")
        f1_cols = [c for c in ["crackles_f1", "wheezes_f1"] if c in df_metrics.columns]
        if f1_cols and "model" in df_metrics.columns:
            long = df_metrics.melt(id_vars="model", value_vars=f1_cols, var_name="Target", value_name="F1")
            long["Target"] = long["Target"].str.replace("_f1", "", regex=False).str.replace("_", " ").str.title()
            fig = go.Figure()
            for target, g in long.groupby("Target"):
                fig.add_trace(go.Bar(x=g["model"], y=g["F1"], name=target))
            fig.update_layout(title="Baseline F1-score comparison", barmode="group", yaxis=dict(range=[0, 1], title="F1-score"), xaxis_title="Model")
            st.plotly_chart(style_plot(fig, 340), use_container_width=True)
        st.download_button("Download saved metrics JSON", json.dumps(metrics_json, indent=2), file_name="lungsound_metrics.json", mime="application/json")
    else:
        st.info("No saved evaluation metrics were found. Train a baseline model to generate the metrics report.")
        st.code("python scripts\\train_all.py --baseline", language="powershell")

    st.markdown("### How to interpret the results")
    st.markdown(
        "- **Precision** describes how often positive predictions were correct in the evaluated sample.\n"
        "- **Recall** describes how many annotated positive examples were detected.\n"
        "- **F1-score** summarizes precision and recall; zero can mean the model did not correctly predict any positive examples for that target.\n"
        "- **Group split** aims to keep segments from the same inferred patient/recording group on one side of the split. Confirm the saved split assignments before making generalization claims."
    )
    st.warning("A good-looking dashboard does not make a weak model accurate. Use the saved test results as-is and report limitations transparently.")

elif page == "Methodology & viva":
    hero("Methodology, architecture & viva guide", "A concise guide to the pipeline, model outputs, data safeguards, and limitations behind the demonstration.", "LUNGSOUND AI · PROJECT GUIDE")
    with st.expander("1. Project objective", expanded=True):
        st.write("LungSound AI is an educational respiratory-audio classification prototype. It predicts two sound labels—crackles and wheezes—and derives Normal, Crackles, Wheezes, or Both from those two outputs. It is not designed to diagnose a disease.")
    with st.expander("2. Processing pipeline", expanded=True):
        st.markdown("1. Load a valid WAV recording.\n2. Convert to mono and resample to the configured sample rate.\n3. Apply basic quality heuristics for silence, clipping, and unusual signal levels.\n4. Split the signal into overlapping windows.\n5. Extract MFCC summary features.\n6. Run the selected trained baseline model for crackles and wheezes.\n7. Aggregate window scores and visualize elevated-score windows.")
    with st.expander("3. Dataset and leakage safeguards"):
        st.write("The training pipeline uses annotation cycle labels from the dataset and group-based splitting using patient IDs inferred from recording names where available. The amount and reliability of patient identifiers influence how confidently unseen-patient generalization can be claimed.")
    with st.expander("4. Likely viva questions"):
        qa = [
            ("Why use MFCC features?", "MFCCs compactly summarize the short-time spectral envelope and provide a practical baseline representation for audio classification."),
            ("Why predict crackles and wheezes separately?", "They are independent sound labels, so two binary outputs naturally represent all four display categories."),
            ("What does the spectrogram show?", "It shows how energy is distributed across time and Mel-frequency bins. It helps inspect a signal but is not itself proof of a diagnosis."),
            ("Why split by patient?", "Segments from one person can share recording and physiology characteristics. Keeping a patient's segments together reduces leakage between training and test data."),
            ("Can this diagnose pneumonia or asthma?", "No. The system classifies annotated sound patterns only and is not a validated diagnostic device."),
            ("What if F1-score is zero?", "Check the label distribution, test-set positives, preprocessing, model predictions, and saved split assignments. Do not hide or replace the result."),
        ]
        for q, a in qa:
            st.markdown(f"**Q. {q}**")
            st.write(a)
    st.markdown("### Architecture")
    st.code(
        "flowchart TD\n"
        "A[WAV recording + annotation data] --> B[Validation and segmentation]\n"
        "B --> C[Preprocessing and MFCC features]\n"
        "C --> D[Random Forest / SVM]\n"
        "D --> E[Crackles and wheezes scores]\n"
        "E --> F[Derived category + window timeline]\n"
        "F --> G[Streamlit research dashboard]",
        language="mermaid",
    )
    st.warning("All outputs are research estimates. Sound categories do not confirm a medical condition and must not be used to guide treatment.")

