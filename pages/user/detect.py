"""
User-facing detection page 
"""

import streamlit as st
import numpy as np
from PIL import Image

from utils.auth import require_role
from utils.preprocessing import preprocess_for_model, load_image_pil, validate_fundus_image
from utils.gradcam import generate_gradcam
from utils.report_generator import generate_pdf_report
from models.model_factory import load_model, list_available_models
from config import DR_STAGES, DR_STAGE_DESCRIPTIONS, DR_STAGE_COLORS, MAIN_MODEL


# ── Cached Model Loader ─────────────────────────────────────────────
@st.cache_resource
def get_cached_model(model_name):
    return load_model(model_name)


# ── Inference Logic (Separated) ─────────────────────────────────────
def run_inference(uploaded_file, model):
    img_batch = preprocess_for_model(uploaded_file, MAIN_MODEL)
    preds = model.predict(img_batch, verbose=0)[0]
    predicted_stage = int(np.argmax(preds))
    return preds, predicted_stage


# ── Page UI ─────────────────────────────────────────────────────────
def show_detect_page():
    require_role("user")

    st.markdown("## 🔬 Fundus Image Analysis")
    st.markdown("Upload a retinal fundus image to detect Diabetic Retinopathy stage.")
    st.divider()

    # ── Check Model Availability ─────────────────────────────────────
    available = list_available_models()
    if MAIN_MODEL not in available:
        st.warning(
            "⚠️ Main model not trained yet. Ask admin to train it."
        )
        return

    # ── Upload ───────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload Fundus Image (PNG / JPG / JPEG)",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded is None:
        st.info("Upload a fundus image to begin.")
        return

    # ── Load Model ───────────────────────────────────────────────────
    with st.spinner("Loading model..."):
        try:
            model = get_cached_model(MAIN_MODEL)
        except Exception as e:
            st.error(f"Model load failed: {e}")
            return

    # ── Display Image ─────────────────────────────────────────────────
    col_img, col_res = st.columns([1, 1])

    with col_img:
        st.markdown("#### Original Image")
        pil_img = load_image_pil(uploaded)
        st.image(pil_img, use_column_width=True)

    # ── Validate Image ────────────────────────────────────────────────
    with st.spinner("Validating image..."):
        img_array = np.array(pil_img.resize((224, 224)))
        is_valid, reason = validate_fundus_image(img_array)

    if not is_valid:
        st.error("❌ This does not appear to be a retinal fundus image")
        with st.expander("🔍 See why your image was rejected", expanded=True):
            st.markdown(f"""
**Validation result:** `{reason}`

**What is a fundus image?**  
A fundus (retinal) image is a specialized photograph of the back of the eye, taken with an ophthalmoscope or fundus camera. 
It typically looks like a **circular orange/red disc** on a **black background**, with visible **blood vessels** branching from a central bright spot (optic disc).

**Tips:**
- ✅ Use images from a fundus camera or eye-screening device  
- ✅ The image should have orange/red tones with a dark border  
- ❌ Regular photographs, screenshots, or microscopy images will be rejected  
- ❌ Grayscale, blue-tinted, or plain-colored images will be rejected  
""")
        return

    # ── Inference ────────────────────────────────────────────────────
    with st.spinner("Running analysis..."):
        preds, predicted_stage = run_inference(uploaded, model)

        # Confidence
        confidence = float(preds[predicted_stage]) * 100

        # Sanity check
        if np.max(preds) < 0.4:
            st.warning("⚠️ Low confidence prediction. Result may be unreliable.")

        # Grad-CAM
        uploaded.seek(0)
        try:
            _, gradcam_pil, _ = generate_gradcam(
                model, uploaded, MAIN_MODEL, predicted_stage
            )
        except Exception:
            gradcam_pil = pil_img

    # ── Display Result ───────────────────────────────────────────────
    stage_label = DR_STAGES[predicted_stage]
    hex_color = DR_STAGE_COLORS[predicted_stage]

    with col_res:
        st.markdown("#### Diagnosis Result")
        st.markdown(
            f"""
            <div style="background:{hex_color}22; border-left:5px solid {hex_color};
                        padding:16px; border-radius:8px;">
                <h2 style="color:{hex_color}; margin:0;">
                    Stage {predicted_stage}: {stage_label}
                </h2>
                <p style="color:#fff; margin-top:6px;">
                    {DR_STAGE_DESCRIPTIONS[predicted_stage]}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(f"**Confidence:** `{confidence:.1f}%`")

    # ── Grad-CAM + Chart ─────────────────────────────────────────────
    st.divider()
    col_gc1, col_gc2 = st.columns(2)

    with col_gc1:
        st.markdown("#### Grad-CAM")
        st.image(gradcam_pil, use_column_width=True)

    with col_gc2:
        st.markdown("#### Confidence Scores")

        import plotly.graph_objects as go

        stage_names = [f"Stage {i}: {DR_STAGES[i]}" for i in range(5)]
        bar_colors = [DR_STAGE_COLORS[i] for i in range(5)]

        fig = go.Figure(go.Bar(
            x=[f"{p*100:.1f}%" for p in preds],
            y=stage_names,
            orientation="h",
            marker_color=bar_colors,
            text=[f"{p*100:.1f}%" for p in preds],
            textposition="auto",
        ))

        fig.update_layout(
            xaxis=dict(range=[0, 100]),
            height=280,
        )

        st.plotly_chart(fig, use_container_width=True)

    # ── PDF Report ───────────────────────────────────────────────────
    st.divider()
    st.markdown("#### Download Report")

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
            label="Download PDF Report",
            data=pdf_bytes,
            file_name="DR_Report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    except Exception as e:
        st.error(f"PDF generation failed: {e}")