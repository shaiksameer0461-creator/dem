class ScoringEngine:
    """
    Calculates the final AI evaluation score.
    """

    def calculate(
        self,
        semantic_score,
        wpm,
        pause_ratio,
        filler_count,
        energy
    ):

        # -------------------------
        # 1. Semantic Score (70%)
        # -------------------------
        semantic_marks = semantic_score * 0.70

        # -------------------------
        # 2. Words Per Minute (10%)
        # Ideal: 100–140
        # -------------------------
        if 100 <= wpm <= 140:
            wpm_marks = 10

        elif 90 <= wpm < 100 or 141 <= wpm <= 160:
            wpm_marks = 8

        elif 80 <= wpm < 90 or 161 <= wpm <= 180:
            wpm_marks = 6

        else:
            wpm_marks = 4

        # -------------------------
        # 3. Pause Ratio (8%)
        # -------------------------
        if pause_ratio <= 15:
            pause_marks = 8

        elif pause_ratio <= 25:
            pause_marks = 6

        elif pause_ratio <= 35:
            pause_marks = 4

        else:
            pause_marks = 2

        # -------------------------
        # 4. Voice Energy (7%)
        # -------------------------
        if energy >= 0.08:
            energy_marks = 7

        elif energy >= 0.05:
            energy_marks = 5

        else:
            energy_marks = 3

        # -------------------------
        # 5. Filler Words (5%)
        # -------------------------
        if filler_count == 0:
            filler_marks = 5

        elif filler_count <= 2:
            filler_marks = 4

        elif filler_count <= 5:
            filler_marks = 3

        else:
            filler_marks = 1

        # -------------------------
        # Final Score
        # -------------------------
        overall_score = round(
            semantic_marks
            + wpm_marks
            + pause_marks
            + energy_marks
            + filler_marks,
            2
        )

        # -------------------------
        # Grade
        # -------------------------
        if overall_score >= 90:
            grade = "A+"
            recommendation = "Excellent"

        elif overall_score >= 80:
            grade = "A"
            recommendation = "Very Good"

        elif overall_score >= 70:
            grade = "B"
            recommendation = "Good"

        elif overall_score >= 60:
            grade = "C"
            recommendation = "Fair"

        else:
            grade = "D"
            recommendation = "Needs Improvement"

        return (
            overall_score,
            grade,
            recommendation
        )