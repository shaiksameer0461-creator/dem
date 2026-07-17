import os
from datetime import datetime


# =====================================================
# FORMAT DATE
# =====================================================

def current_datetime():
    """
    Return current date and time.
    """

    return datetime.now().strftime("%d-%m-%Y %H:%M")


# =====================================================
# VALIDATE AUDIO FILE
# =====================================================

def validate_audio(uploaded_file):

    if uploaded_file is None:
        raise ValueError("Please upload an audio file.")

    allowed_extensions = [
        ".wav",
        ".mp3",
        ".m4a"
    ]

    extension = os.path.splitext(
        uploaded_file.name
    )[1].lower()

    if extension not in allowed_extensions:

        raise ValueError(
            "Unsupported audio format."
        )

    max_size = 20 * 1024 * 1024

    if uploaded_file.size > max_size:

        raise ValueError(
            "Audio file must be smaller than 20 MB."
        )

    return True


# =====================================================
# DELETE TEMP FILE
# =====================================================

def delete_temp_file(file_path):

    try:

        if os.path.exists(file_path):

            os.remove(file_path)

    except Exception:

        pass


# =====================================================
# FORMAT SCORE
# =====================================================

def format_score(score):

    return f"{score:.2f}%"


# =====================================================
# FORMAT ENERGY
# =====================================================

def format_energy(energy):

    return round(energy, 4)


# =====================================================
# GRADE COLOR
# =====================================================

def grade_color(score):

    if score >= 90:

        return "green"

    elif score >= 80:

        return "blue"

    elif score >= 70:

        return "orange"

    return "red"