import streamlit as st
import torch
from PIL import Image
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models import build_model
from src.predict import predict_image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINTS = {
    "EfficientNet-B0": ("efficientnet", PROJECT_ROOT / "best_model_efficientnet.pth"),
    "ResNet50": ("resnet", PROJECT_ROOT / "best_model_resnet.pth"),
}


@st.cache_resource
def load_model(model_name, checkpoint_path):
    model = build_model(model_name=model_name, num_classes=7).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    return model


st.set_page_config(page_title="SkinSense AI", layout="centered")
st.title("SkinSense AI — Skin Lesion Classifier")

st.markdown("Upload a dermoscopic image to get a prediction, confidence score, and Grad-CAM explanation.")

selected_label = st.selectbox("Choose a model", list(CHECKPOINTS.keys()))
model_name, checkpoint_path = CHECKPOINTS[selected_label]

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    temp_path = PROJECT_ROOT / "App" / "temp_upload.jpg"
    image = Image.open(uploaded_file).convert("RGB")
    image.save(temp_path)

    if not checkpoint_path.exists():
        st.error(f"Checkpoint not found: {checkpoint_path}")
    else:
        model = load_model(model_name, checkpoint_path)

        with st.spinner("Running prediction..."):
            result = predict_image(model, str(temp_path), DEVICE, model_name=model_name)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", width='stretch')
        with col2:
            st.image(result["heatmap"], caption="Grad-CAM Heatmap", use_container_width=True)

        st.subheader("Prediction")
        st.metric(label="Class", value=result["prediction"])
        st.metric(label="Confidence", value=f"{result['confidence']*100:.2f}%")

        st.subheader("Class Probabilities")
        prob_dict = {
            cls: float(prob)
            for cls, prob in zip(
                ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
                result["probabilities"]
            )
        }
        st.bar_chart(prob_dict)