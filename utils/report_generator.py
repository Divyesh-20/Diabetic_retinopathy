"""
utils/report_generator.py – PDF report generation using ReportLab
"""

import io
import os
import numpy as np
from datetime import datetime
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from config import DR_STAGES, DR_STAGE_DESCRIPTIONS, DR_STAGE_COLORS, MODELS, MAIN_MODEL


def _pil_to_rl_image(pil_img: Image.Image, width_cm: float, height_cm: float) -> RLImage:
    """Convert PIL image to ReportLab Image object via BytesIO."""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return RLImage(buf, width=width_cm * cm, height=height_cm * cm)


def generate_pdf_report(
    original_img: Image.Image,
    gradcam_img: Image.Image,
    predicted_stage: int,
    confidence_scores: np.ndarray,
    model_name: str = MAIN_MODEL,
) -> bytes:
    """
    Generate a styled PDF report.
    Returns bytes of the PDF file.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    story  = []

    # ── Title ─────────────────────────────────────────────────────────────────
    title_style = ParagraphStyle(
        "title",
        parent=styles["Title"],
        fontSize=22,
        textColor=colors.HexColor("#1a237e"),
        spaceAfter=6,
    )
    story.append(Paragraph("Diabetic Retinopathy Detection Report", title_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1a237e")))
    story.append(Spacer(1, 0.3 * cm))

    # ── Meta info ─────────────────────────────────────────────────────────────
    meta_style = ParagraphStyle("meta", parent=styles["Normal"], fontSize=10, textColor=colors.grey)
    now_str = datetime.now().strftime("%d %B %Y, %I:%M %p")
    model_display = MODELS.get(model_name, model_name)
    story.append(Paragraph(f"<b>Date:</b> {now_str} &nbsp;&nbsp; <b>Model:</b> {model_display}", meta_style))
    story.append(Spacer(1, 0.5 * cm))

    # ── Diagnosis Result ──────────────────────────────────────────────────────
    stage_label = DR_STAGES[predicted_stage]
    hex_color   = DR_STAGE_COLORS[predicted_stage].lstrip("#")
    rl_color    = colors.HexColor(f"#{hex_color}")

    result_style = ParagraphStyle(
        "result",
        parent=styles["Heading1"],
        fontSize=20,
        textColor=rl_color,
        spaceAfter=4,
    )
    story.append(Paragraph(f"Diagnosis: Stage {predicted_stage} – {stage_label}", result_style))

    desc_style = ParagraphStyle("desc", parent=styles["Normal"], fontSize=11, leading=16)
    story.append(Paragraph(DR_STAGE_DESCRIPTIONS[predicted_stage], desc_style))
    story.append(Spacer(1, 0.4 * cm))

    # ── Images Side by Side ───────────────────────────────────────────────────
    orig_rl = _pil_to_rl_image(original_img, 7, 7)
    grad_rl = _pil_to_rl_image(gradcam_img,  7, 7)
    caption_style = ParagraphStyle("cap", parent=styles["Normal"], fontSize=9,
                                   textColor=colors.grey, alignment=TA_CENTER)

    img_table = Table(
        [[orig_rl, grad_rl],
         [Paragraph("Original Fundus Image", caption_style),
          Paragraph("Grad-CAM Heatmap", caption_style)]],
        colWidths=[8 * cm, 8 * cm],
    )
    img_table.setStyle(TableStyle([
        ("ALIGN",    (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",   (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING",  (0, 0), (-1, -1), 4),
    ]))
    story.append(img_table)
    story.append(Spacer(1, 0.5 * cm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
    story.append(Spacer(1, 0.3 * cm))

    # ── Confidence Scores ─────────────────────────────────────────────────────
    story.append(Paragraph("Confidence Scores per Stage", styles["Heading2"]))
    conf_data = [["Stage", "Label", "Confidence (%)"]]
    for i in range(5):
        pct = f"{confidence_scores[i]*100:.2f}%"
        marker = " ◀ Predicted" if i == predicted_stage else ""
        conf_data.append([f"Stage {i}", DR_STAGES[i], pct + marker])

    conf_table = Table(conf_data, colWidths=[3 * cm, 6 * cm, 7 * cm])
    ts = TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#1a237e")),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0), 11),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#e8eaf6")]),
        ("FONTSIZE",    (0, 1), (-1, -1), 10),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",(0, 0), (-1, -1), 8),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1),   5),
        ("TEXTCOLOR",   (0, predicted_stage + 1), (-1, predicted_stage + 1), rl_color),
        ("FONTNAME",    (0, predicted_stage + 1), (-1, predicted_stage + 1), "Helvetica-Bold"),
    ])
    conf_table.setStyle(ts)
    story.append(conf_table)
    story.append(Spacer(1, 0.5 * cm))

    # ── Disclaimer ────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
    disc_style = ParagraphStyle("disc", parent=styles["Normal"], fontSize=8,
                                 textColor=colors.grey, leading=12)
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        "<b>Disclaimer:</b> This report is generated by an AI-assisted system and is intended for "
        "informational purposes only. It does not constitute medical advice. Please consult a "
        "qualified ophthalmologist for diagnosis and treatment decisions.",
        disc_style
    ))

    doc.build(story)
    return buf.getvalue()
