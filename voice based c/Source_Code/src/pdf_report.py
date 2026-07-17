import os
from datetime import datetime
from reportlab.pdfgen import canvas
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak
)


class PDFReport:

    def generate(
    self,
    transcript,
    semantic_score,
    feedback,
    confidence,
    strengths,
    improvements,
    features,
    overall_score,
    grade,
    recommendation,
    waveform_path
   ):

        os.makedirs("reports", exist_ok=True)

        pdf_path = os.path.join(
            "reports",
            "Voice_Analysis_Report.pdf"
        )

        document = SimpleDocTemplate(pdf_path)

        styles = getSampleStyleSheet()

        elements = []

        # --------------------------------
        # Title
        # --------------------------------
        elements.append(
            Paragraph(
                "<font size=22><b>VOICE-BASED CONCEPT UNDERSTANDING ANALYSER</b></font>",
                styles["Title"]
            )
        )

        elements.append(
            Paragraph(
                "<b>AI Evaluation Report</b>",
                styles["Heading2"]
            )
        )

        elements.append(
            Paragraph(
                f"Generated on : {datetime.now().strftime('%d-%m-%Y %H:%M')}",
                styles["Normal"]
            )
        )
        report_id = datetime.now().strftime("AI-%Y%m%d-%H%M%S")

        elements.append(
            Paragraph(
                f"<b>Report ID:</b> {report_id}",
                styles["Normal"]
            )
        )

        elements.append(Spacer(1,20))
        # --------------------------------
        # Analysis Completed Banner
        # --------------------------------

        banner = Table([["✔ Analysis Completed Successfully"]])

        banner.setStyle(TableStyle([

            ("BACKGROUND",(0,0),(-1,-1),colors.darkgreen),

            ("TEXTCOLOR",(0,0),(-1,-1),colors.white),

            ("ALIGN",(0,0),(-1,-1),"CENTER"),

            ("FONTNAME",(0,0),(-1,-1),"Helvetica-Bold"),

            ("FONTSIZE",(0,0),(-1,-1),12),

            ("BOTTOMPADDING",(0,0),(-1,-1),10),

            ("TOPPADDING",(0,0),(-1,-1),10)

        ]))

        elements.append(banner)

        elements.append(Spacer(1,20))
        elements.append(
            Paragraph(
                "<b>Audio Waveform</b>",
                styles["Heading2"]
            )
        )

        if waveform_path and os.path.exists(waveform_path):
            elements.append(
                Image(
                    waveform_path,
                    width=450,
                    height=160
                )
            )
        else:
            elements.append(
                Paragraph(
                    "Waveform image not available.",
                    styles["Normal"]
                )
            )

        elements.append(Spacer(1,20))


        elements.append(
            Paragraph(
                "<b>Overall Performance</b>",
                styles["Heading2"]
            )
        )

        performance = Table([

            ["Overall Score", f"{overall_score}/100"],

            ["Grade", grade],

            ["Evaluation", recommendation]

        ])

        performance.setStyle(TableStyle([

            ("BACKGROUND",(0,0),(-1,0),colors.darkblue),
            ("TEXTCOLOR",(0,0),(-1,0),colors.white),
            ("GRID",(0,0),(-1,-1),1,colors.black),
            ("ALIGN",(0,0),(-1,-1),"CENTER"),
            ("BOTTOMPADDING",(0,0),(-1,-1),8)

        ]))

        elements.append(performance)

        elements.append(Spacer(1,20))
      
 

        # --------------------------------
        # Transcript
        # --------------------------------
        elements.append(
            Paragraph(
                "<b>Student Transcript</b>",
                styles["Heading2"]
            )
        )

        elements.append(
            Paragraph(
                escape(transcript),
                styles["BodyText"]
            )
        )

        elements.append(Spacer(1,15))

        # --------------------------------
        # Semantic Analysis
        # --------------------------------
        elements.append(
            Paragraph(
                "<b>Semantic Analysis</b>",
                styles["Heading2"]
            )
        )

        semantic_table = Table([

            ["Metric", "Result"],

            ["Similarity Score", f"{semantic_score}%"],

            ["Understanding", feedback],

            ["Confidence", confidence]

        ])

        semantic_table.setStyle(

            TableStyle([

                ("BACKGROUND",(0,0),(-1,0),colors.lightblue),

                ("GRID",(0,0),(-1,-1),1,colors.black),

                ("BACKGROUND",(0,1),(0,-1),colors.whitesmoke),

                ("BOTTOMPADDING",(0,0),(-1,-1),8),

            ])

        )

        elements.append(semantic_table)

        elements.append(Spacer(1,15))

        # --------------------------------
        # Audio Analysis
        # --------------------------------

        elements.append(
            Paragraph(
                "<b>Audio Analysis</b>",
                styles["Heading2"]
            )
        )

        audio_table = Table([

            ["Metric","Value"],

            ["Duration",f"{features.get("duration", "N/A")} sec"],

            ["Word Count",features["word_count"]],

            ["Words Per Minute",features["wpm"]],

            ["Pause Ratio",f"{features['pause_ratio']} %"],

            ["Voice Energy",features["energy"]],

            ["Filler Words",features["filler_count"]]

        ])
        

        audio_table.setStyle(

            TableStyle([

                ("BACKGROUND",(0,0),(-1,0),colors.lightgreen),

                ("GRID",(0,0),(-1,-1),1,colors.black),

                ("BACKGROUND",(0,1),(0,-1),colors.whitesmoke),

                ("BOTTOMPADDING",(0,0),(-1,-1),8)

            ])

        )
        

        elements.append(audio_table)

        elements.append(Spacer(1,20))
        elements.append(PageBreak())
        # --------------------------------
        # Overall Evaluation
        # --------------------------------

        elements.append(
            Paragraph(
                "<b>Overall AI Evaluation</b>",
                styles["Heading2"]
            )
        )
        if grade == "A+":
            grade_color = colors.green
        elif grade == "A":
            grade_color = colors.green
        elif grade == "B":
            grade_color = colors.orange
        else:
            grade_color = colors.red
        

        overall_table = Table([

            ["Overall Score",f"{overall_score}/100"],

            ["Grade",grade],

            ["Recommendation",recommendation]

        ])

        overall_table.setStyle(

            TableStyle([

                ("BACKGROUND",(0,0),(-1,0),colors.orange),

                ("GRID",(0,0),(-1,-1),1,colors.black),

                ("BACKGROUND",(0,1),(0,-1),colors.whitesmoke),

                ("BOTTOMPADDING",(0,0),(-1,-1),8)

            ])

        )
        overall_table.setStyle(TableStyle([

            ("TEXTCOLOR",(1,1),(1,1),grade_color)

        ]))

        elements.append(overall_table)

        elements.append(Spacer(1,20))


        elements.append(
            Paragraph(
                "<b>AI Summary</b>",
                styles["Heading2"]
            )
        )

        summary = f"""
        The student's explanation achieved a semantic similarity score of
        <b>{semantic_score}%</b> with an overall grade of <b>{grade}</b>.
        The explanation demonstrates <b>{feedback}</b> and the speaking quality
        was evaluated as <b>{recommendation}</b>.
        """

        elements.append(
            Paragraph(
                summary,
                styles["BodyText"]
            )
        )

        elements.append(Spacer(1,20))




        # --------------------------------
        # Strengths
        # --------------------------------

        elements.append(
            Paragraph(
                "<b>Strengths</b>",
                styles["Heading2"]
            )
        )

        for item in strengths:

            elements.append(
                Paragraph(
                    f"✔ {item}",
                    styles["Normal"]
                )
            )

        elements.append(Spacer(1,15))
        # --------------------------------
        # Areas for Improvement
        # --------------------------------

        elements.append(
            Paragraph(
                "<b>Areas for Improvement</b>",
                styles["Heading2"]
            )
        )

        for item in improvements:

            elements.append(
                Paragraph(
                    f"• {item}",
                    styles["Normal"]
                )
            )

        elements.append(Spacer(1,15))
        # --------------------------------
        # AI Recommendation
        # --------------------------------

        elements.append(
            Paragraph(
                "<b>AI Recommendation</b>",
                styles["Heading2"]
            )
        )

        recommend_table = Table([

            ["AI Recommendation"],

            [recommendation]

        ])

        recommend_table.setStyle(TableStyle([

            ("BACKGROUND",(0,0),(-1,0),colors.darkgreen),

            ("TEXTCOLOR",(0,0),(-1,0),colors.white),

            ("ALIGN",(0,0),(-1,-1),"CENTER"),

            ("GRID",(0,0),(-1,-1),1,colors.black),
        
            ("BOTTOMPADDING",(0,0),(-1,-1),10),

            ("TOPPADDING",(0,0),(-1,-1),10)

        ]))

        elements.append(recommend_table)

        elements.append(Spacer(1,20))

        # --------------------------------
        # Footer
        # --------------------------------

        elements.append(Spacer(1,25))

        elements.append(
            Paragraph(
                "<b>Voice-Based Concept Understanding Analyser</b>",
                styles["Heading3"]
            )
        )

        elements.append(
            Paragraph(
                "AI-Powered Student Concept Evaluation System",
                styles["Italic"]
            )
        )

        elements.append(
            Paragraph(
                "Generated Automatically using Artificial Intelligence",
                styles["Italic"]
            )
        )

        elements.append(
            Paragraph(
                "© 2026 All Rights Reserved",
                styles["Italic"]
            )
        )

        
        elements.append(Spacer(1,20))

        elements.append(
            Paragraph(
                "<b>Thank you for using the Voice-Based Concept Understanding Analyser.</b>",
                styles["Heading3"]
            )
        )

        elements.append(
            Paragraph(
                "This report was automatically generated by the AI evaluation engine.",
                styles["Italic"]
            )
        )
        document.build(
            elements,
            onFirstPage=PDFReport.add_page_number,
            onLaterPages=PDFReport.add_page_number
        )

        return pdf_path
    @staticmethod
    def add_page_number(canvas, doc):
        page = canvas.getPageNumber()

        canvas.setFont("Helvetica",9)

        canvas.drawRightString(
            560,
            20,
            f"Page {page}"
        )