import streamlit as st

st.set_page_config(
    page_title="Voice Based Concept Understanding Analyser",
    page_icon="🎤",
    layout="wide",
)

import os
import time
import logging
from utils.speech_to_text import transcribe_audio
from utils.semantic_eval import calculate_similarity
from utils.audio_utils import load_audio, extract_audio_features, get_scoring_features, save_waveform
from utils.scoring_engine import (
    calculate_filler_word_stats,
    evaluate_understanding,
)
from utils.report_generator import generate_pdf

# Configure logging for all modules
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom CSS — professional dark theme styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* ---- Global ---- */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1100px;
    }

    /* ---- Header ---- */
    .app-header {
        margin-bottom: 1.8rem;
    }
    .app-header h1 {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.15rem;
    }
    .app-header p {
        opacity: 0.6;
        font-size: 0.95rem;
        margin-top: 0;
    }

    /* ---- Cards ---- */
    .card {
        background: rgba(128,128,128,0.06);
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
    }
    .card h3 {
        font-size: 1.1rem;
        margin-top: 0;
        margin-bottom: 0.8rem;
    }

    /* ---- Info banner ---- */
    .info-banner {
        background: rgba(31,111,235,0.08);
        border: 1px solid rgba(31,111,235,0.35);
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        color: #1f6feb;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }

    /* ---- Understanding banner ---- */
    .level-banner {
        padding: 1.2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .level-banner h2 {
        color: #fff;
        margin: 0;
        font-size: 1.6rem;
    }
    .level-banner p {
        color: rgba(255,255,255,0.9);
        margin: 0.3rem 0 0;
        font-size: 1.1rem;
    }

    /* ---- Section headers ---- */
    .section-header {
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid rgba(128,128,128,0.2);
    }

    /* ---- Metric card ---- */
    .metric-card {
        background: rgba(128,128,128,0.06);
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .metric-card .label {
        opacity: 0.55;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-card .value {
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 0.2rem;
    }

    /* ---- Feature row ---- */
    .feature-row {
        display: flex;
        justify-content: space-between;
        padding: 0.45rem 0;
        border-bottom: 1px solid rgba(128,128,128,0.15);
        font-size: 0.9rem;
    }
    .feature-row:last-child { border-bottom: none; }
    .feature-row .feat-label { opacity: 0.6; }
    .feature-row .feat-value { font-weight: 600; }

    /* ---- Hide default Streamlit branding ---- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
DEFAULTS = {
    "transcript": None,
    "similarity_score": None,
    "features": None,
    "scoring_features": None,
    "filler_stats": None,
    "final_score": None,
    "understanding_level": None,
    "level_color": None,
    "waveform_img": None,
    "audio_file_path": None,
    "reference_answer": None,
    "timings": None,
    "analysis_done": False,
}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="app-header">'
    "<h1>Voice-Based Concept Understanding Analyser</h1>"
    "<p>Automated evaluation of spoken conceptual explanations using AI.</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Input section — two-column layout
# ---------------------------------------------------------------------------
col_upload, col_ref = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown("**Upload Student Audio (WAV)**")
    uploaded_file = st.file_uploader(
        "Upload Student Audio (WAV)",
        type=["wav", "mp3"],
        label_visibility="collapsed",
    )

    # Test helper checkbox
    use_sample = st.checkbox("Use Sample Audio (for testing)", value=False)

with col_ref:
    st.markdown("#### Reference")
    reference_answer = st.text_area(
        "Reference",
        placeholder="Enter your reference text here",
        height=160,
        label_visibility="collapsed",
    )

# ---------------------------------------------------------------------------
# Resolve audio source
# ---------------------------------------------------------------------------
file_path = None

if use_sample:
    file_path = "audio/sample.mp3"
    if not os.path.exists(file_path):
        st.error(f"Sample audio not found at `{file_path}`.")
        file_path = None
elif uploaded_file is not None:
    # --- Input validation: file size (200 MB limit) ---
    MAX_SIZE_MB = 200
    if uploaded_file.size > MAX_SIZE_MB * 1024 * 1024:
        st.error(f"❌ File exceeds {MAX_SIZE_MB} MB limit. Please upload a smaller file.")
    else:
        file_path = "temp_audio.wav"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

# ---------------------------------------------------------------------------
# Audio preview + waveform (shown immediately after upload, before analysis)
# ---------------------------------------------------------------------------
if file_path and os.path.exists(file_path):
    # Audio player
    with open(file_path, "rb") as af:
        st.audio(af.read(), format="audio/wav" if file_path.endswith(".wav") else "audio/mp3")

    # Waveform preview (validate audio file)
    try:
        waveform_img = save_waveform(file_path)
        st.image(waveform_img, use_column_width=True)
        st.session_state["waveform_img"] = waveform_img
    except Exception as e:
        st.error(f"❌ Could not read audio file. It may be corrupted or in an unsupported format.\n\nDetails: {e}")
        st.stop()

    # --- Analysis button ---
    analyze_clicked = st.button("🔬 Analyze Concept Understanding", use_container_width=True)

    if analyze_clicked:
        # Validate reference text
        if not reference_answer or not reference_answer.strip():
            st.warning("⚠️ Please enter a Reference to evaluate against.")
        else:
            with st.spinner("Processing and evaluating..."):
                try:
                    timings = {}
                    logger.info("=== Analysis pipeline started ===")

                    # Load audio once — reused by all audio functions
                    t0 = time.time()
                    y, sr = load_audio(file_path)
                    timings["Audio Loading"] = round(time.time() - t0, 3)

                    # Speech-to-text
                    t0 = time.time()
                    transcript = transcribe_audio(file_path)
                    timings["Transcription (Whisper)"] = round(time.time() - t0, 3)

                    # Semantic similarity
                    t0 = time.time()
                    similarity_score = calculate_similarity(reference_answer, transcript)
                    timings["Embedding & Similarity"] = round(time.time() - t0, 3)

                    # Audio features (pass pre-loaded y, sr)
                    t0 = time.time()
                    features = extract_audio_features(file_path, y=y, sr=sr)
                    scoring_features = get_scoring_features(file_path, y=y, sr=sr)
                    timings["Audio Feature Extraction"] = round(time.time() - t0, 3)

                    # Filler word statistics
                    t0 = time.time()
                    filler_stats = calculate_filler_word_stats(transcript)
                    timings["Filler Word Analysis"] = round(time.time() - t0, 3)

                    # Multi-factor scoring
                    t0 = time.time()
                    similarity_normalized = similarity_score / 100.0
                    final_score, understanding_level, level_color = evaluate_understanding(
                        similarity_normalized, filler_stats["filler_ratio"], scoring_features
                    )
                    timings["Scoring Engine"] = round(time.time() - t0, 3)

                    timings["Total Pipeline"] = round(
                        sum(v for v in timings.values()), 3
                    )
                    logger.info("=== Pipeline complete: %s ===", timings)

                    # Persist results in session state
                    st.session_state.update({
                        "transcript": transcript,
                        "similarity_score": similarity_score,
                        "features": features,
                        "scoring_features": scoring_features,
                        "filler_stats": filler_stats,
                        "final_score": final_score,
                        "understanding_level": understanding_level,
                        "level_color": level_color,
                        "audio_file_path": file_path,
                        "reference_answer": reference_answer,
                        "timings": timings,
                        "analysis_done": True,
                    })
                    st.rerun()

                except Exception as e:
                    logger.exception("Analysis pipeline failed")
                    st.error(
                        f"❌ Analysis failed. Please try again or use a different audio file.\n\n"
                        f"**Error:** {e}"
                    )
else:
    # No file yet — show info banner
    st.markdown(
        '<div class="info-banner">Upload an audio file to begin analysis.</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Results section (reads from session state)
# ---------------------------------------------------------------------------
if st.session_state["analysis_done"]:
    transcript = st.session_state["transcript"]
    similarity_score = st.session_state["similarity_score"]
    features = st.session_state["features"]
    scoring_features = st.session_state["scoring_features"]
    filler_stats = st.session_state["filler_stats"]
    final_score = st.session_state["final_score"]
    understanding_level = st.session_state["understanding_level"]
    level_color = st.session_state["level_color"]
    ref_answer = st.session_state.get("reference_answer", "")
    similarity_normalized = similarity_score / 100.0

    st.markdown("---")

    # ---- Analysis Completed banner ----
    st.success("Analysis Completed")

    # ---- Two-column: Transcribed Explanation + Final Evaluation ----
    res_left, res_right = st.columns([3, 2], gap="large")

    with res_left:
        st.markdown('<div class="section-header">Transcribed Explanation</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="card"><p style="line-height:1.7;margin:0;">{transcript}</p></div>',
            unsafe_allow_html=True,
        )

    with res_right:
        st.markdown('<div class="section-header">Final Evaluation</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="card" style="text-align:center;">'
            f'<p style="opacity:0.6;margin:0 0 0.2rem;font-size:0.85rem;">Understanding Score</p>'
            f'<p style="font-size:2.5rem;font-weight:800;margin:0;">{final_score}/100</p>'
            f'<p style="color:{level_color};font-size:1.3rem;font-weight:700;margin:0.5rem 0 0;">'
            f'{understanding_level}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ---- Three-column key metrics ----
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            f'<div class="metric-card"><div class="label">Semantic Similarity</div>'
            f'<div class="value">{similarity_normalized:.2f}</div></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="metric-card"><div class="label">Filler Word Ratio</div>'
            f'<div class="value">{filler_stats["filler_ratio"]:.2f}</div></div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f'<div class="metric-card"><div class="label">Confidence (Energy)</div>'
            f'<div class="value">{scoring_features["rms_energy"]}</div></div>',
            unsafe_allow_html=True,
        )

    # ---- Detailed breakdown (collapsible) ----
    with st.expander("📋 Detailed Score Breakdown & Audio Features"):
        det1, det2 = st.columns(2)
        with det1:
            st.markdown('<div class="section-header">Score Components</div>', unsafe_allow_html=True)
            sim_pts = 50 if similarity_normalized > 0.7 else 30 if similarity_normalized > 0.4 else 10
            filler_pts = 20 if filler_stats["filler_ratio"] < 0.05 else 10
            pause_pts = 15 if scoring_features["pause_ratio"] < 0.25 else 5
            energy_pts = 15 if scoring_features["rms_energy"] > 0.01 else 5
            for label, pts, max_pts in [
                ("Similarity", sim_pts, 50),
                ("Filler Words", filler_pts, 20),
                ("Pause Ratio", pause_pts, 15),
                ("RMS Energy", energy_pts, 15),
            ]:
                st.markdown(
                    f'<div class="feature-row"><span class="feat-label">{label}</span>'
                    f'<span class="feat-value">{pts}/{max_pts}</span></div>',
                    unsafe_allow_html=True,
                )
        with det2:
            st.markdown('<div class="section-header">Audio Features</div>', unsafe_allow_html=True)
            for key, val in features.items():
                st.markdown(
                    f'<div class="feature-row"><span class="feat-label">{key}</span>'
                    f'<span class="feat-value">{val}</span></div>',
                    unsafe_allow_html=True,
                )

    # ---- Performance Timing (collapsible) ----
    timings = st.session_state.get("timings", None)
    if timings:
        with st.expander("⏱ Performance Timing"):
            for stage, secs in timings.items():
                bar_pct = min(1.0, secs / max(timings["Total Pipeline"], 0.001))
                st.markdown(
                    f'<div class="feature-row">'
                    f'<span class="feat-label">{stage}</span>'
                    f'<span class="feat-value">{secs}s</span></div>',
                    unsafe_allow_html=True,
                )
            st.caption("Timings measured per-run. Model loading is cached across sessions.")

    # ---- PDF Report ----
    waveform_path = st.session_state.get("waveform_img", None)
    pdf_path = generate_pdf(
        transcript=transcript,
        similarity_score=similarity_score,
        final_score=final_score,
        understanding_level=understanding_level,
        features=features,
        filler_stats=filler_stats,
        reference_text=ref_answer,
        waveform_img=waveform_path,
        scoring_features=scoring_features,
    )

    with open(pdf_path, "rb") as pdf_file:
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_file,
            file_name="Voice_Report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
