import streamlit as st
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from PIL import Image
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models import build_model
from src.predict import predict_image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SKIN_CHECKPOINTS = {
    "EfficientNet-B0": ("efficientnet", PROJECT_ROOT / "best_model_efficientnet.pth"),
    "ResNet50": ("resnet", PROJECT_ROOT / "best_model_resnet.pth"),
}
SKIN_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

XRAY_CHECKPOINTS = {
    "EfficientNet-B0": ("efficientnet", PROJECT_ROOT / "best_model_xray.pth"),
}
XRAY_CLASSES = ["fractured", "not fractured"]

# confidence threshold for triage routing — tune later
CONFIDENCE_THRESHOLD = 0.70


@st.cache_resource
def load_model(model_name, checkpoint_path, num_classes):
    model = build_model(model_name=model_name, num_classes=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    return model


def render_triage_banner(confidence):
    if confidence >= CONFIDENCE_THRESHOLD:
        st.success(f"✅ High confidence ({confidence*100:.1f}%) — provisional result. Routine follow-up recommended.")
    else:
        st.warning(f"⚠️ Low confidence ({confidence*100:.1f}%) — flagged for doctor review via telehealth.")


def run_prediction_ui(checkpoints, class_names, upload_key):
    selected_label = st.selectbox("Choose a model", list(checkpoints.keys()), key=f"{upload_key}_model")
    model_name, checkpoint_path = checkpoints[selected_label]

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key=upload_key)

    if uploaded_file is not None:
        temp_path = PROJECT_ROOT / "App" / f"temp_{upload_key}.jpg"
        image = Image.open(uploaded_file).convert("RGB")
        image.save(temp_path)

        if not checkpoint_path.exists():
            st.error(f"Checkpoint not found: {checkpoint_path}")
            return

        model = load_model(model_name, checkpoint_path, num_classes=len(class_names))

        with st.spinner("Running prediction..."):
            result = predict_image(model, str(temp_path), DEVICE, model_name=model_name, class_names=class_names)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        with col2:
            if result["heatmap"] is not None:
                st.image(result["heatmap"], caption="Grad-CAM Heatmap", use_container_width=True)
            else:
                st.info("Grad-CAM unavailable in this environment")

        st.subheader("Prediction")
        m1, m2 = st.columns(2)
        m1.metric(label="Class", value=result["prediction"])
        m2.metric(label="Confidence", value=f"{result['confidence']*100:.2f}%")

        render_triage_banner(result["confidence"])

        st.subheader("Class Probabilities")
        prob_dict = {cls: float(prob) for cls, prob in zip(class_names, result["probabilities"])}
        st.bar_chart(prob_dict)

        st.caption("⚠️ This is a screening aid, not a medical diagnosis. Always consult a qualified doctor.")


st.set_page_config(page_title="MediScan AI", layout="centered", page_icon="🩺")

# ---- lightweight mock login (UI only, no real auth) ----
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🩺 MediScan AI")
    st.caption("v1.0 — Demo Build")
    st.markdown("#### Sign in to continue")
    name = st.text_input("Clinic / Health worker name")
    if st.button("Sign in", type="primary"):
        if name.strip():
            st.session_state.logged_in = True
            st.session_state.user_name = name
            st.rerun()
        else:
            st.error("Enter a name to continue.")
    st.stop()

# ---- main app ----

col_title, col_badge = st.columns([4, 1])
with col_title:
    st.title("🩺 MediScan AI")
    st.caption(f"Signed in as {st.session_state.user_name}")
with col_badge:
    st.markdown("### `v1.0 demo`")

st.markdown("AI-powered rural diagnostic screening — upload an image to get a prediction, confidence score, and Grad-CAM explanation.")

tab_skin, tab_xray = st.tabs(["🧴 Skin Lesion Screening", "🦴 Bone Fracture Screening"])

with tab_skin:
    run_prediction_ui(SKIN_CHECKPOINTS, SKIN_CLASSES, upload_key="skin")

with tab_xray:
    run_prediction_ui(XRAY_CHECKPOINTS, XRAY_CLASSES, upload_key="xray")

st.divider()
st.caption("MediScan AI v1.0 · Screening demo for TECHVERSE'26 · Not a substitute for professional medical diagnosis.")
