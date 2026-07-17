# ==========================================================
# Voice-Based Concept Understanding Analyser
# Main Application
#
# Developed by:
# T A Karthik
# B.Tech CSE (AI & ML)
# Mohan Babu University
# ==========================================================

# ==========================================================
# Import Libraries
# ==========================================================

import streamlit as st

from src.speech_to_text import SpeechToText
from src.semantic_analysis import SemanticAnalyzer
from src.audio_features import AudioFeatures
from src.scoring import ScoringEngine
from src.pdf_report import PDFReport
from src.audio_visualizer import AudioVisualizer

from src.ui import (
    show_header,
    show_sidebar,
    show_transcript,
    show_semantic,
    show_audio,
    show_overall,
    show_download,
    show_footer
)

from src.utils import (
    validate_audio,
    delete_temp_file
)

# ==========================================================
# Streamlit Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Voice-Based Concept Understanding Analyser",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# Sidebar
# ==========================================================

show_sidebar()

# ==========================================================
# Header
# ==========================================================

show_header()

# ==========================================================
# Upload Section
# ==========================================================

left, right = st.columns([2, 1])

with left:

    st.subheader("Upload Student Audio")

    audio = st.file_uploader(

        "Drag and drop your audio",

        type=["wav", "mp3", "m4a"],

        label_visibility="collapsed"

    )

with right:

    st.subheader("Concept Reference")

    if audio is None:

        st.write("""

Artificial Intelligence is a branch of computer science that develops intelligent systems capable of learning, reasoning, solving problems, understanding language and making decisions.

""")

    else:

        st.success("✅ Audio Uploaded")

        st.write("Ready for analysis.")
st.markdown("---")

status_box = st.empty()

if audio is None:

    status_box.info("📁 Upload an audio file to begin analysis.")

# else:

#     status_box.success("✅ Audio uploaded successfully.")

# ==========================================================
# Reference Concept (Show only before upload)
# ==========================================================

# if audio is None:

#     semantic = SemanticAnalyzer()

#     reference = semantic.get_reference("Artificial Intelligence")

#     with st.expander("📖 Reference Concept", expanded=True):

#         st.write(reference)

# ==========================================================
# Audio Preview
# ==========================================================

if audio is not None:

    # st.success("✅ Audio uploaded successfully.")

    st.audio(audio)

    # -----------------------------------------
    # Display Waveform
    # -----------------------------------------

    visualizer = AudioVisualizer()
    waveform = visualizer.create_waveform(audio)

    st.subheader("📈 Audio Waveform")

    

    st.pyplot(waveform)

    analyze_button = st.button(
        "🚀 Analyze Concept Understanding",
        type="primary",
        use_container_width=True
    )

    

else:

    analyze_button = False
# ==========================================================
# AI ANALYSIS PIPELINE
# ==========================================================

if analyze_button:

    try:

        # ----------------------------------------------
        # Validate Uploaded File
        # ----------------------------------------------

        validate_audio(audio)

        # ----------------------------------------------
        # Progress Indicator
        # ----------------------------------------------

        progress_bar = st.progress(0)

        status = st.empty()

        # ----------------------------------------------
        # Step 1 : Speech To Text
        # ----------------------------------------------

        status.info("🎙 Step 1/5 : Transcribing audio using OpenAI Whisper...")

        progress_bar.progress(10)

        stt = SpeechToText()

        transcript, audio_path = stt.transcribe_audio(audio)
        visualizer = AudioVisualizer()
        waveform_path = visualizer.save_waveform(
            audio_path
        )

        progress_bar.progress(30)

        # ----------------------------------------------
        # Step 2 : Semantic Analysis
        # ----------------------------------------------

        status.info("🧠 Step 2/5 : Performing semantic similarity analysis...")

        semantic = SemanticAnalyzer()

        semantic_result = semantic.analyze(
            "Artificial Intelligence",
            transcript
        )
        similarity_score = semantic_result["score"]
        feedback = semantic_result["feedback"]
        confidence = semantic_result["confidence"]
        strengths = semantic_result["strengths"]
        improvements = semantic_result["improvements"]
        recommendation = semantic_result["recommendation"]


        progress_bar.progress(55)

        # ----------------------------------------------
        # Step 3 : Audio Feature Extraction
        # ----------------------------------------------

        status.info("🎵 Step 3/5 : Extracting audio features...")

        audio_features = AudioFeatures()

        features = audio_features.analyze(
            audio_path,
            transcript
        )

        progress_bar.progress(75)

        # ----------------------------------------------
        # Step 4 : Overall AI Evaluation
        # ----------------------------------------------

        status.info("🏆 Step 4/5 : Calculating overall AI score...")

        scoring = ScoringEngine()

        overall_score, grade, recommendation = scoring.calculate(

            similarity_score,

            features["wpm"],

            features["pause_ratio"],

            features["filler_count"],

            features["energy"]

        )

        progress_bar.progress(90)

        # ----------------------------------------------
        # Step 5 : Generate PDF Report
        # ----------------------------------------------

        status.info("📄 Step 5/5 : Generating professional PDF report...")

        pdf = PDFReport()

        pdf_path = pdf.generate(

            transcript,

            similarity_score,

            feedback,
            confidence,
            strengths,
            improvements,

            features,

            overall_score,

            grade,

            recommendation,

            waveform_path

        )

        progress_bar.progress(100)

        
        # ==================================================
        # DISPLAY RESULTS
        # ==================================================

       # =====================================
       # Student Transcript
       # =====================================

        # show_transcript(transcript)

        # st.markdown("---")

        # show_semantic(
        #     semantic_result
        # )

        # st.markdown("---")

        # show_audio(
        #     features
        # )

        # st.markdown("---")

        # show_overall(

        #     overall_score,

        #     grade,

        #     recommendation

        # )

        # st.markdown("---")

        # show_download(
        #     pdf_path
        # )
        # ==================================================
        # RESULTS DASHBOARD
        # ==================================================

        st.success("✅ Analysis Completed Successfully!")

        left, right = st.columns([2, 1], gap="large")

        # ============================
        # LEFT SIDE
        # ============================

        with left:

            st.subheader("📝 Transcribed Explanation")

            st.write(transcript)

            st.markdown("---")

            st.subheader("🧠 Semantic Analysis")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Similarity", f"{similarity_score}%")

            with col2:
                st.metric("Confidence", confidence)

        # ============================
        # RIGHT SIDE
        # ============================

        with right:

            st.subheader("🏆 Final Evaluation")

            st.metric(
                "Overall Score",
                f"{overall_score}/100"
            )

            st.metric(
                "Grade",
                grade
            )

            st.success(feedback)
        st.markdown("---")

        st.subheader("💡 AI Suggestions")

        for item in improvements:
            st.write(f"✅ {item}")

        st.markdown("---")
        st.subheader("✅ Strengths")

        for strength in strengths:
            st.success(f"✔ {strength}")
        st.markdown("---")
        st.metric("Overall Score", f"{overall_score}/100")

        st.metric("Grade", grade)

        st.metric("Understanding", feedback)
        st.markdown("---")

        show_download(pdf_path)
    
    # ==========================================================
    # ERROR HANDLING
    # ==========================================================

    except Exception as e:

        st.error("❌ An error occurred during analysis.")

        st.exception(e)

    # ==========================================================
    # CLEANUP
    # ==========================================================

    finally:

        try:

            if "audio_path" in locals():

                delete_temp_file(audio_path)

        except Exception:

            pass


# ==========================================================
# FOOTER
# ==========================================================

show_footer()