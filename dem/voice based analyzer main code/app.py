import streamlit as st
import tempfile
import os
from audio_recorder_streamlit import audio_recorder

# Your custom modules
from speech_to_text import speech_to_text, filler_word_ratio
from semantic_eval import semantic_similarity, auto_detect_topic, get_gemini_feedback, get_reference, CONCEPT_REFERENCES
from audio_utils import extract_audio_features, save_waveform
from scoring_engine import evaluate_understanding, get_score_emoji, calculate_fluency_score
from report_generator import generate_pdf_report

st.set_page_config(
    page_title="Voice-Based Concept Understanding Analyser",
    page_icon="🎙️",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main { background-color: #0d1117; color: white; }
    h1, h2, h3 { color: white; }
    .metric-card {
        background-color: #1a1a2e;
        padding: 15px;
        border-radius: 10px;
        margin: 5px;
        text-align: center;
    }
    .score-card {
        background-color: #16213e;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎙️ Voice-Based Concept Understanding Analyser")
st.markdown("**Automated evaluation of spoken conceptual explanations using AI**")
st.markdown("---")

# Session State
if "history" not in st.session_state:
    st.session_state.history = []
if "result" not in st.session_state:
    st.session_state.result = None
if "recorded_path" not in st.session_state:
    st.session_state.recorded_path = None

# Sidebar Settings
with st.sidebar:
    st.header("⚙️ Settings")
    gemini_api_key = st.text_input("🔑 Gemini API Key", type="password")
    auto_detect = st.checkbox("🤖 Auto Detect Topic (Gemini)", value=False)
    
    if not auto_detect:
        topic = st.selectbox("📚 Select Concept Topic", list(CONCEPT_REFERENCES.keys()))
    else:
        topic = "Machine Learning"  # fallback
    
    language = st.radio("🌐 Language", ["English", "Telugu (Beta ⚠️)"])
    lang_code = "te" if "Telugu" in language else "en"
    
    if "Telugu" in language:
        st.warning("Telugu support is in Beta")
    
    st.markdown("---")
    st.markdown("### 📊 Attempt History")
    if st.session_state.history:
        for i, h in enumerate(st.session_state.history[-5:]):
            if st.button(f"Attempt {i+1}: {h['topic']} → {h['score']}/100", key=f"hist_{i}"):
                st.session_state.result = h['result']
                st.rerun()
    else:
        st.info("No attempts yet!")
    st.markdown("---")
    if st.button("🔄 Reset / New Analysis"):
        st.session_state.result = None
        st.session_state.recorded_path = None
        st.rerun()
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()
# Main Area
col1, col2 = st.columns([1.5, 1])

audio_file = None

with col1:
    st.subheader("🎤 Record or Upload Audio")
    tab1, tab2 = st.tabs(["Live Record", "Upload File"])

    with tab1:
        audio_bytes = audio_recorder()
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                f.write(audio_bytes)
                st.session_state.recorded_path = f.name

    with tab2:
        audio_file = st.file_uploader("Upload audio (WAV/MP3)", type=["wav", "mp3"])

with col2:
    if not auto_detect:
        st.subheader("📖 Concept Reference")
        ref_text = get_reference(topic)
        st.info(ref_text)

# Analyze Button
if st.button("🔍 Analyze Concept Understanding", type="primary"):
    with st.spinner("Transcribing and evaluating..."):
        audio_path = None
        try:
            # Get valid audio path
            if st.session_state.recorded_path and os.path.exists(st.session_state.recorded_path):
                audio_path = st.session_state.recorded_path
            elif audio_file:
                suffix = ".wav" if audio_file.type == "audio/wav" else ".mp3"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(audio_file.getvalue())
                    audio_path = tmp.name
            else:
                st.error("Please record or upload an audio file.")
                st.stop()

            transcript_result = speech_to_text(audio_path, lang_code)

            if not transcript_result.get("success"):
                st.error(f"Transcription failed: {transcript_result.get('error', 'Unknown error')}")
            else:
                transcript = transcript_result["text"]
                filler = filler_word_ratio(transcript)
                audio_features = extract_audio_features(audio_path)
                waveform_bytes = save_waveform(audio_path)

                # Auto-detect topic
                current_topic = topic
                if auto_detect and gemini_api_key:
                    current_topic = auto_detect_topic(transcript, gemini_api_key)

                ref_text = get_reference(current_topic)
                similarity = semantic_similarity(transcript, ref_text)

                score, level, color = evaluate_understanding(similarity, filler, audio_features)
                fluency = calculate_fluency_score(filler, audio_features.get("pause_ratio", 0))
                emoji = get_score_emoji(level)

                feedback = get_gemini_feedback(transcript, current_topic, score, gemini_api_key) if gemini_api_key else "Add Gemini API key for AI feedback."

                st.session_state.history.append({
    "topic": topic,
    "score": score,
    "level": level,
    "result": {
        "transcript": transcript,
        "topic": topic,
        "ref_text": ref_text,
        "similarity": similarity,
        "filler": filler,
        "audio_features": audio_features,
        "score": score,
        "level": level,
        "color": color,
        "emoji": emoji,
        "fluency": fluency,
        "feedback": feedback,
        "waveform_bytes": waveform_bytes
    }
})

                st.session_state.result = {
                    "transcript": transcript,
                    "topic": current_topic,
                    "ref_text": ref_text,
                    "similarity": similarity,
                    "filler": filler,
                    "audio_features": audio_features,
                    "score": score,
                    "level": level,
                    "color": color,
                    "emoji": emoji,
                    "fluency": fluency,
                    "feedback": feedback,
                    "waveform_bytes": waveform_bytes
                }
                st.success("✅ Analysis Completed!")

        except Exception as e:
            st.error(f"Error: {str(e)}")
        finally:
            # Cleanup uploaded files only
            if audio_path and audio_path != st.session_state.get("recorded_path"):
                if os.path.exists(audio_path):
                    try:
                        os.unlink(audio_path)
                    except:
                        pass

# ==================== RESULTS SECTION ====================
if st.session_state.result:
    r = st.session_state.result
    st.markdown("---")
    st.subheader("✅ Analysis Results")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Transcribed Explanation**")
        st.write(r["transcript"])

    with col_b:
        st.markdown("**Final Score**")
        st.markdown(f"<h1 style='color:{r['color']}; text-align:center;'>{r['score']}/100</h1>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align:center;'>{r['emoji']} {r['level']}</h3>", unsafe_allow_html=True)

    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Semantic Similarity", r["similarity"])
    with col2:
        st.metric("Filler Words", r["filler"])
    with col3:
        st.metric("Fluency Score", f"{r['fluency']['fluency_score']}/100")

    if r.get("waveform_bytes"):
        st.image(r["waveform_bytes"], use_column_width=True)

    # PDF Download
    pdf_bytes = generate_pdf_report(
        topic=r['topic'],
        reference_text=r['ref_text'],
        transcribed_text=r['transcript'],
        similarity=r['similarity'],
        filler_ratio=r['filler'],
        pause_ratio=r['audio_features'].get('pause_ratio', 0),
        rms_energy=r['audio_features'].get('rms_energy', 0),
        final_score=r['score'],
        understanding_level=r['level'],
        fluency_score=r['fluency']['fluency_score'],
        gemini_feedback=r['feedback'],
        waveform_bytes=r['waveform_bytes']
    )

    st.download_button(
        label="📄 Download Full PDF Report",
        data=pdf_bytes,
        file_name=f"VBCUA_Report_{r['topic'].replace(' ', '_')}.pdf",
        mime="application/pdf"
    )

    # Clear button
    if st.button("Clear Results"):
        st.session_state.result = None
        st.rerun()
