from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.units import inch
import io
from datetime import datetime

def generate_pdf_report(
    topic: str,
    reference_text: str,
    transcribed_text: str,
    similarity: float,
    filler_ratio: float,
    pause_ratio: float,
    rms_energy: float,
    final_score: int,
    understanding_level: str,
    fluency_score: float,
    gemini_feedback: str,
    waveform_bytes: bytes = None
) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#1a1a2e'),
        spaceAfter=10,
        alignment=1
    )
    story.append(Paragraph("Voice-Based Concept Understanding Analyser", title_style))
    story.append(Paragraph("Evaluation Report", title_style))
    story.append(Spacer(1, 0.2*inch))

    # Date and topic
    info_style = ParagraphStyle('Info', parent=styles['Normal'], fontSize=11)
    story.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%d %B %Y %I:%M %p')}", info_style))
    story.append(Paragraph(f"<b>Topic Evaluated:</b> {topic}", info_style))
    story.append(Spacer(1, 0.2*inch))

    # Reference concept
    story.append(Paragraph("Reference Concept", styles['Heading2']))
    story.append(Paragraph(reference_text.strip(), styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

    # Student transcription
    story.append(Paragraph("Student Transcription", styles['Heading2']))
    story.append(Paragraph(transcribed_text.strip(), styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

    # Waveform image
    if waveform_bytes:
        story.append(Paragraph("Audio Visualization", styles['Heading2']))
        img = Image(io.BytesIO(waveform_bytes), width=6*inch, height=2*inch)
        story.append(img)
        story.append(Spacer(1, 0.2*inch))

    # Evaluation summary table
    story.append(Paragraph("Evaluation Summary", styles['Heading2']))
    table_data = [
        ["Metric", "Value"],
        ["Semantic Similarity", str(similarity)],
        ["Filler Word Ratio", str(filler_ratio)],
        ["Pause Ratio", str(pause_ratio)],
        ["Confidence (Energy)", str(rms_energy)],
        ["Fluency Score", f"{fluency_score}/100"],
        ["Final Score", f"{final_score}/100"],
        ["Understanding Level", understanding_level],
    ]
    table = Table(table_data, colWidths=[3*inch, 3*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a1a2e')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 11),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f0f0f0')]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.2*inch))

    # Gemini feedback
    story.append(Paragraph("AI Feedback", styles['Heading2']))
    story.append(Paragraph(gemini_feedback, styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
