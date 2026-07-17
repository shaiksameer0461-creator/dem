import streamlit as st


# =====================================================
# PAGE HEADER
# =====================================================

def show_header():

    st.title("Voice-Based Concept Understanding Analyser")

    st.caption(
        "Automated evaluation of spoken conceptual explanations using Artificial Intelligence."
    )

    st.divider()


# =====================================================
# SIDEBAR
# =====================================================

def show_sidebar():

    st.sidebar.markdown("# 📌 About Project")

    st.sidebar.info(
        """
### 🎤 Voice-Based Concept Understanding Analyser

This AI-powered application evaluates a student's conceptual understanding
using speech recognition and semantic analysis.

---

### 🚀 Technologies

- 🎙 OpenAI Whisper
- 🧠 Sentence Transformers
- 📊 Librosa
- 🏆 AI Scoring Engine
- 📄 ReportLab

---

### 👨‍💻 Developed By

**T A Karthik**

🎓 B.Tech CSE (AI & ML)

🏫 Mohan Babu University

Version 1.0

© 2026
"""
    )


# =====================================================
# TRANSCRIPT
# =====================================================

def show_transcript(transcript):

    st.subheader("📝 Transcript")

    st.success(transcript)


# =====================================================
# SEMANTIC ANALYSIS
# =====================================================

def show_semantic(result):

    st.subheader("🧠 Semantic Analysis")
    # st.subheader("🔥 THIS IS MY NEW SEMANTIC ANALYSIS")

    st.metric(
        "Similarity Score",
        f"{result['score']}%"
    )

    st.success(
        f"Understanding : {result['feedback']}"
    )

    st.info(
        f"Confidence : {result['confidence']}"
    )

    st.markdown("## ✅ Strengths")

    for strength in result["strengths"]:
        st.success(f"✔ {strength}")

    st.markdown("## 📈 Areas for Improvement")

    for item in result["improvements"]:
        st.warning(f"• {item}")

    st.markdown("## 💡 AI Recommendation")

    st.info(result["recommendation"])


# =====================================================
# AUDIO ANALYSIS
# =====================================================

def show_audio(features):

    st.subheader("🎙 Audio Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Duration",
            f"{features['duration']} sec"
        )

        st.metric(
            "Word Count",
            features["word_count"]
        )

    with col2:

        st.metric(
            "Words / Minute",
            features["wpm"]
        )

        st.metric(
            "Voice Energy",
            features["energy"]
        )

    with col3:

        st.metric(
            "Pause Ratio",
            f"{features['pause_ratio']} %"
        )

        st.metric(
            "Filler Words",
            features["filler_count"]
        )


# =====================================================
# OVERALL AI EVALUATION
# =====================================================

def show_overall(score, grade, recommendation):

    st.subheader("🏆 Overall AI Evaluation")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Overall Score",
            f"{score}/100"
        )

    with col2:

        st.metric(
            "Grade",
            grade
        )

    with col3:

        st.metric(
            "Recommendation",
            recommendation
        )


# =====================================================
# PDF DOWNLOAD
# =====================================================

def show_download(pdf_path):

    st.subheader("📄 Download Report")

    with open(pdf_path, "rb") as pdf_file:

        st.download_button(

            label="📥 Download PDF Report",

            data=pdf_file,

            file_name="Voice_Analysis_Report.pdf",

            mime="application/pdf"

        )


# =====================================================
# FOOTER
# =====================================================

def show_footer():

    st.markdown("---")

    st.markdown(

        """
<div style="text-align:center;color:gray">

Voice-Based Concept Understanding Analyser

Developed using Artificial Intelligence

© 2026

</div>
""",

        unsafe_allow_html=True
    )