import json
import os

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


import streamlit as st

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


class SemanticAnalyzer:

    def __init__(self):

        self.model = load_model()

        json_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "reference_concepts.json"
        )

        with open(json_path, "r", encoding="utf-8") as file:
            self.reference = json.load(file)
    def get_reference(self, topic):

        return self.reference.get(
            topic,
            "Reference concept not available."
        )
    def analyze(self, topic, transcript):

        reference_text = self.reference.get(topic)

        if reference_text is None:
            raise ValueError(
                f"No reference found for '{topic}'"
            )

        # --------------------------------------
        # Generate Embeddings
        # --------------------------------------

        reference_embedding = self.model.encode(
            reference_text,
            convert_to_tensor=True
        )

        transcript_embedding = self.model.encode(
            transcript,
            convert_to_tensor=True
        )

        similarity = cos_sim(
            reference_embedding,
            transcript_embedding
        )

        similarity_score = round(
            float(similarity.item()) * 100,
            2
        )

        # --------------------------------------
        # Understanding Level
        # --------------------------------------

        if similarity_score >= 90:

            feedback = "Excellent Understanding"

            confidence = "Very High"

            strengths = [
                "Excellent conceptual understanding.",
                "Covered the major concepts correctly.",
                "Explanation is technically accurate.",
                "Good logical flow."
            ]

            improvements = [
                "Include one real-world example.",
                "Mention advanced AI techniques.",
                "Explain practical applications in more detail."
            ]

            recommendation = (
                "Excellent work! Continue improving by "
                "adding practical examples and deeper technical details."
            )

        elif similarity_score >= 80:

            feedback = "Good Understanding"

            confidence = "High"

            strengths = [
                "Good conceptual understanding.",
                "Most important concepts are covered.",
                "Explanation is clear."
            ]

            improvements = [
                "Include more technical terminology.",
                "Provide additional examples.",
                "Improve explanation depth."
            ]

            recommendation = (
                "Good explanation. Try expanding your answer "
                "with more detailed concepts."
            )

        elif similarity_score >= 70:

            feedback = "Fair Understanding"

            confidence = "Moderate"

            strengths = [
                "Basic understanding demonstrated.",
                "Some key concepts explained."
            ]

            improvements = [
                "Cover more important concepts.",
                "Improve explanation sequence.",
                "Use technical vocabulary."
            ]

            recommendation = (
                "Your explanation is acceptable but needs "
                "better concept coverage."
            )

        elif similarity_score >= 60:

            feedback = "Basic Understanding"

            confidence = "Low"

            strengths = [
                "Shows initial understanding."
            ]

            improvements = [
                "Review the topic.",
                "Explain concepts step-by-step.",
                "Include definitions and examples."
            ]

            recommendation = (
                "Revise the topic and practice speaking again."
            )

        else:

            feedback = "Needs Improvement"

            confidence = "Very Low"

            strengths = [
                "Attempted the explanation."
            ]

            improvements = [
                "Study the topic thoroughly.",
                "Practice before recording.",
                "Cover the main definition first.",
                "Use simple examples."
            ]

            recommendation = (
                "Significant improvement is required. "
                "Review the topic and try again."
            )

        # --------------------------------------
        # Return Complete AI Analysis
        # --------------------------------------

        return {

            "score": similarity_score,

            "feedback": feedback,

            "confidence": confidence,

            "strengths": strengths,

            "improvements": improvements,

            "recommendation": recommendation

        }