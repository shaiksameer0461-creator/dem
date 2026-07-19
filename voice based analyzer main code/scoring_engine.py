def evaluate_understanding(similarity: float, filler_ratio: float, audio: dict) -> tuple:
    score = 0
    
    # Semantic similarity score (50 points max)
    score += 50 if similarity > 0.7 else 30 if similarity > 0.4 else 10
    
    # Filler word score (20 points max)
    score += 20 if filler_ratio < 0.05 else 10
    
    # Pause ratio score (15 points max)
    score += 15 if audio["pause_ratio"] < 0.25 else 5
    
    # RMS energy score (15 points max)
    score += 15 if audio["rms_energy"] > 0.01 else 5
    
    # Classification
    if score >= 80:
        return score, "Strong Understanding", "#2ecc71"
    elif score >= 50:
        return score, "Moderate Understanding", "#f39c12"
    else:
        return score, "Poor Understanding", "#e74c3c"

def get_score_emoji(level: str) -> str:
    if level == "Strong Understanding":
        return "🟢"
    elif level == "Moderate Understanding":
        return "🟡"
    else:
        return "🔴"

def calculate_fluency_score(filler_ratio: float, pause_ratio: float) -> dict:
    fluency = 100
    fluency -= (filler_ratio * 200)
    fluency -= (pause_ratio * 100)
    fluency = max(0, min(100, round(fluency, 2)))
    
    if fluency >= 75:
        level = "Fluent"
    elif fluency >= 50:
        level = "Moderate"
    else:
        level = "Needs Improvement"
    
    return {
        "fluency_score": fluency,
        "fluency_level": level
    }
