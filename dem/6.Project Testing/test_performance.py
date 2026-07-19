import time
import sys
sys.path.append('../5. Project Development Phase')

from semantic_eval import semantic_similarity
from scoring_engine import evaluate_understanding, calculate_fluency_score
from speech_to_text import filler_word_ratio

def test_semantic_similarity_speed():
    text1 = "Machine learning is a subset of artificial intelligence"
    text2 = "ML enables systems to learn from data automatically"
    start = time.time()
    for _ in range(10):
        semantic_similarity(text1, text2)
    end = time.time()
    avg_time = (end - start) / 10
    assert avg_time < 2.0, f"Too slow: {avg_time:.3f}s"
    print(f"✅ Semantic similarity speed: {avg_time:.3f}s avg")

def test_scoring_speed():
    audio = {"pause_ratio": 0.2, "rms_energy": 0.05}
    start = time.time()
    for _ in range(100):
        evaluate_understanding(0.7, 0.03, audio)
    end = time.time()
    avg_time = (end - start) / 100
    assert avg_time < 0.01
    print(f"✅ Scoring speed: {avg_time:.6f}s avg")

def test_filler_detection_speed():
    text = "um like basically this is you know a test sentence uh right"
    start = time.time()
    for _ in range(100):
        filler_word_ratio(text)
    end = time.time()
    avg_time = (end - start) / 100
    assert avg_time < 0.01
    print(f"✅ Filler detection speed: {avg_time:.6f}s avg")

def test_fluency_calculation_speed():
    start = time.time()
    for _ in range(100):
        calculate_fluency_score(0.03, 0.15)
    end = time.time()
    avg_time = (end - start) / 100
    print(f"✅ Fluency calculation speed: {avg_time:.6f}s avg")

def test_stability():
    audio = {"pause_ratio": 0.2, "rms_energy": 0.05}
    scores = []
    for _ in range(10):
        score, _, _ = evaluate_understanding(0.7, 0.03, audio)
        scores.append(score)
    assert len(set(scores)) == 1
    print(f"✅ Stability test passed: consistent score {scores[0]}")

if __name__ == "__main__":
    print("Running performance tests...\n")
    test_semantic_similarity_speed()
    test_scoring_speed()
    test_filler_detection_speed()
    test_fluency_calculation_speed()
    test_stability()
    print("\n✅ All performance tests passed!")
