# 🎙️ Voice-Based Concept Understanding Analyser (VBCUA)

An AI-powered web application that evaluates how effectively users understand and explain conceptual topics through spoken communication.

---

## 📌 Project Overview

VBCUA combines speech-to-text transcription, semantic similarity analysis, audio feature extraction, and intelligent scoring to assess:
- Quality of conceptual understanding
- Fluency of speech delivery
- Communication confidence

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Streamlit | Web UI Framework |
| OpenAI Whisper | Speech-to-Text Transcription |
| Sentence-BERT | Semantic Similarity Analysis |
| Librosa | Audio Feature Extraction |
| Google Gemini | AI Feedback Generation |
| ReportLab | PDF Report Generation |

---

## 📁 ProjectStructure ├── app.py
├── speech_to_text.py
├── semantic_eval.py
├── audio_utils.py
├── scoring_engine.py
├── report_generator.py
└── requirements.txt
---

## ⚙️ Installation & Setup

```bash
git clone https://github.com/Padminipalabathuni/Voice-Based-Concept-understanding-Analyser.git
python -m venv myenv
myenv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
🔑 API Key Required
Google Gemini API Key → https://makersuite.google.com/
Enter in sidebar when running app
✨ Features
🎤 Audio file upload (WAV, MP3)
📝 Whisper speech transcription
📊 Semantic similarity scoring
🔊 Audio waveform visualization
🤖 Gemini auto topic detection
💪 Confidence score meter
📈 Attempt history tracking
🌐 Telugu beta support
📄 PDF report download
👤 Developer
Palabathuni Padmini
🚀 How to Use
Run streamlit run app.py
Enter Gemini API key in sidebar
Upload audio file (WAV/MP3)
Click Analyze Concept Understanding 
View results and download PDF report
