import os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch

def generate_pdf(transcript, similarity_score, final_score, understanding_level,
                 features, filler_stats, reference_text=None, waveform_img=None,
                 scoring_features=None):
    """
    Generates a comprehensive PDF report at reports/report.pdf containing:
      - Reference Concept
      - Student Transcription
      - Audio Visualization (waveform image)
      - Evaluation Summary table with all metrics and qualitative feedback
    """
    os.makedirs("reports", exist_ok=True)
    pdf_path = "reports/report.pdf"

    doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                            leftMargin=40, rightMargin=40,
                            topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()

    elements = []

    # ---- Title ----
    elements.append(
        Paragraph(
            "<font size=20><b>Voice Based Concept Understanding Report</b></font>",
            styles["Title"],
        )
    )
    elements.append(Spacer(1, 20))

    # ---- Reference Concept ----
    if reference_text and reference_text.strip():
        elements.append(Paragraph("<b>Reference Concept</b>", styles["Heading2"]))
        elements.append(Paragraph(reference_text, styles["BodyText"]))
        elements.append(Spacer(1, 15))

    # ---- Student Transcription ----
    elements.append(Paragraph("<b>Student Transcription</b>", styles["Heading2"]))
    elements.append(Paragraph(transcript, styles["BodyText"]))
    elements.append(Spacer(1, 15))

    # ---- Audio Visualization (waveform image) ----
    if waveform_img and os.path.exists(waveform_img):
        elements.append(Paragraph("<b>Audio Visualization</b>", styles["Heading2"]))
        elements.append(Spacer(1, 5))
        img = Image(waveform_img)
        # Scale to fit page width while maintaining aspect ratio
        page_width = A4[0] - 80  # account for margins
        aspect = img.imageWidth / img.imageHeight
        img_width = min(page_width, 6.5 * inch)
        img_height = img_width / aspect
        img.drawWidth = img_width
        img.drawHeight = img_height
        elements.append(img)
        elements.append(Spacer(1, 20))

    # ---- Evaluation Summary (metric table) ----
    elements.append(Paragraph("<b>Evaluation Summary</b>", styles["Heading2"]))
    elements.append(Spacer(1, 8))

    similarity_normalized = similarity_score / 100.0 if similarity_score else 0.0
    filler_ratio = filler_stats.get("filler_ratio", 0.0) if isinstance(filler_stats, dict) else 0.0
    rms_energy = scoring_features.get("rms_energy", "N/A") if isinstance(scoring_features, dict) else "N/A"
    pause_ratio = scoring_features.get("pause_ratio", "N/A") if isinstance(scoring_features, dict) else "N/A"

    table_data = [
        ["Metric", "Value"],
        ["Semantic Similarity", f"{similarity_normalized:.2f}"],
        ["Filler Word Ratio", f"{filler_ratio:.2f}"],
        ["Pause Ratio", f"{pause_ratio}"],
        ["Confidence (Energy)", f"{rms_energy}"],
        ["Final Score", f"{final_score}/100"],
        ["Understanding Level", understanding_level],
    ]

    table = Table(table_data, colWidths=[3 * inch, 3 * inch])
    table.setStyle(TableStyle([
        # Header row
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 11),
        # Body rows
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 10),
        # Grid
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("LINEBELOW", (0, 0), (-1, 0), 1, colors.black),
        # Padding
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        # Alternating row colors
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 20))

    # ---- Audio Features detail ----
    if isinstance(features, dict) and features:
        elements.append(Paragraph("<b>Audio Features</b>", styles["Heading2"]))
        elements.append(Spacer(1, 5))
        feat_data = [["Feature", "Value"]]
        for key, value in features.items():
            feat_data.append([str(key), str(value)])

        feat_table = Table(feat_data, colWidths=[3 * inch, 3 * inch])
        feat_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 11),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 1), (-1, -1), 10),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("LINEBELOW", (0, 0), (-1, 0), 1, colors.black),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
        ]))
        elements.append(feat_table)

    doc.build(elements)

    return pdf_path
