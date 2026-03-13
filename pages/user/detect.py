"""
pages/user/detect.py – User-facing detection page
"""

import streamlit as st
import numpy as np
from PIL import Image

from utils.auth import require_role
from utils.preprocessing import preprocess_for_model, load_image_pil
from utils.gradcam import generate_gradcam
from utils.report_generator import generate_pdf_report
from models.model_factory import load_model, list_available_models
from config import DR_STAGES, DR_STAGE_DESCRIPTIONS, DR_STAGE_COLORS, MAIN_MODEL, MODELS


def show_detect_page():
    require_role("user")

    st.markdown("## 🔬 Fundus Image Analysis")
    st.markdown("Upload a retinal fundus image to detect the stage of Diabetic Retinopathy.")
    st.divider()

    # ── Check model availability ──────────────────────────────────────────────
    available = list_available_models()
    if MAIN_MODEL not in available:
        st.warning(
            "⚠️ The main model (**InceptionResNetV2 + LSTM**) has not been trained yet. "
            "Please ask the admin to train it first."
        )
        st.info(
            "If no models are trained, you can still test the pipeline by having admin "
            "train any model from the **Admin → Train Model** page."
        )
        return

    # ── File Upload ───────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload Fundus Image (PNG / JPG / JPEG)",
        type=["png", "jpg", "jpeg"],
        key="user_upload",
    )

    if uploaded is None:
        st.info("👆 Please upload a fundus image to begin analysis.")
        # Show demo card
        with st.expander("ℹ️ About this tool"):
            st.markdown("""
            - **Model**: InceptionResNetV2 + LSTM hybrid
            - **Input**: Colour fundus photograph (any resolution)
            - **Output**: DR stage (0–4) + confidence scores + Grad-CAM heatmap + PDF report
            - **Stages**: No DR → Mild → Moderate → Severe → Proliferative DR
            """)
        return

    # ── Run Analysis ──────────────────────────────────────────────────────────
    with st.spinner("🔄 Loading model…"):
        try:
            model = load_model(MAIN_MODEL)
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return

    col_img, col_res = st.columns([1, 1])

    with col_img:
        st.markdown("#### Original Fundus Image")
        pil_img = load_image_pil(uploaded)
        st.image(pil_img, use_column_width=True)

    with st.spinner("🧠 Running inference + Grad-CAM…"):
        # Preprocess & predict
        img_batch = preprocess_for_model(uploaded, MAIN_MODEL)
        preds = model.predict(img_batch, verbose=0)[0]   # (5,)
        predicted_stage = int(np.argmax(preds))

        # Grad-CAM
        uploaded.seek(0)
        try:
            original_pil, gradcam_pil, _ = generate_gradcam(model, uploaded, MAIN_MODEL, predicted_stage)
        except Exception:
            gradcam_pil = pil_img   # fallback: show original if GradCAM fails

    # ── Display Results ───────────────────────────────────────────────────────
    stage_label = DR_STAGES[predicted_stage]
    hex_color   = DR_STAGE_COLORS[predicted_stage]
    confidence  = float(preds[predicted_stage]) * 100

    with col_res:
        st.markdown("#### Diagnosis Result")
        st.markdown(
            f"""<div style="background:{hex_color}22; border-left:5px solid {hex_color};
                           padding:16px; border-radius:8px; margin-bottom:12px;">
                <h2 style="color:{hex_color}; margin:0;">Stage {predicted_stage}: {stage_label}</h2>
                <p style="color:#fff; margin:6px 0 0 0; font-size:1rem;">
                    {DR_STAGE_DESCRIPTIONS[predicted_stage]}
                </p>
              </div>""",
            unsafe_allow_html=True,
        )
        st.markdown(f"**Confidence:** `{confidence:.1f}%`")

    # ── GradCAM ───────────────────────────────────────────────────────────────
    st.divider()
    col_gc1, col_gc2 = st.columns(2)
    with col_gc1:
        st.markdown("#### 🎯 Grad-CAM Heatmap")
        st.image(gradcam_pil, use_column_width=True, caption="Red regions = model's focus areas")

    # ── Confidence Bar Chart ──────────────────────────────────────────────────
    with col_gc2:
        st.markdown("#### 📊 Confidence Scores per Stage")
        import plotly.graph_objects as go
        stage_names = [f"Stage {i}: {DR_STAGES[i]}" for i in range(5)]
        bar_colors  = [DR_STAGE_COLORS[i] for i in range(5)]
        fig = go.Figure(go.Bar(
            x=[f"{p*100:.1f}%" for p in preds],
            y=stage_names,
            orientation="h",
            marker_color=bar_colors,
            text=[f"{p*100:.1f}%" for p in preds],
            textposition="auto",
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            xaxis=dict(showgrid=False, range=[0, 100]),
            yaxis=dict(showgrid=False),
            margin=dict(l=0, r=0, t=10, b=0),
            height=280,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── PDF Report Download ───────────────────────────────────────────────────
    st.divider()
    st.markdown("#### 📄 Download Report")
    try:
        uploaded.seek(0)
        pdf_bytes = generate_pdf_report(
            original_img=pil_img.resize((224, 224)),
            gradcam_img=gradcam_pil.resize((224, 224)),
            predicted_stage=predicted_stage,
            confidence_scores=preds,
            model_name=MAIN_MODEL,
        )
        st.download_button(
            label="⬇️ Download PDF Report",
            data=pdf_bytes,
            file_name="DR_Detection_Report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as e:
        st.error(f"PDF generation failed: {e}")
