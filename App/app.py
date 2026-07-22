import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import plotly.graph_objects as go

# ─── PAGE CONFIG ────────────────────────────────────────────────
st.set_page_config(
    page_title="SkinSense AI",
    page_icon="🔬",
    layout="centered"
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #f0f4ff;
}

.main-header {
    background: linear-gradient(135deg, #1a56db 0%, #0e3fa6 100%);
    border-radius: 20px;
    padding: 40px 32px;
    text-align: center;
    margin-bottom: 32px;
    box-shadow: 0 8px 32px rgba(26, 86, 219, 0.25);
}

.main-header h1 {
    font-family: 'DM Serif Display', serif;
    color: white;
    font-size: 2.8rem;
    margin: 0;
    letter-spacing: -0.5px;
}

.main-header p {
    color: rgba(255,255,255,0.85);
    font-size: 1rem;
    margin: 10px 0 0 0;
    font-weight: 300;
}

.upload-box {
    background: white;
    border: 2px dashed #c7d7fa;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    margin-bottom: 24px;
    transition: border-color 0.3s;
}

.result-card {
    background: white;
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 20px;
    box-shadow: 0 2px 16px rgba(26, 86, 219, 0.08);
    border-left: 5px solid #1a56db;
}

.disease-name {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: #1a56db;
    margin: 0;
}

.confidence-badge {
    display: inline-block;
    background: #e8f0fe;
    color: #1a56db;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 0.9rem;
    font-weight: 600;
    margin-top: 8px;
}

.info-card {
    background: white;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 2px 16px rgba(26, 86, 219, 0.08);
}

.info-card h4 {
    color: #1a56db;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 0 0 8px 0;
}

.info-card p {
    color: #374151;
    font-size: 0.95rem;
    margin: 0;
    line-height: 1.6;
}

.warning-box {
    background: #fff8e1;
    border: 1px solid #ffd54f;
    border-radius: 12px;
    padding: 16px 20px;
    margin-top: 24px;
    font-size: 0.85rem;
    color: #7c6200;
}

.stButton > button {
    background: linear-gradient(135deg, #1a56db, #0e3fa6);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 32px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 1rem;
    width: 100%;
    cursor: pointer;
    transition: opacity 0.2s;
}

.stButton > button:hover {
    opacity: 0.9;
}

footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ──────────────────────────────────────────────────
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

DISEASE_INFO = {
    'akiec': {
        'full_name': 'Actinic Keratosis',
        'description': 'A rough, scaly patch caused by years of sun exposure. Can develop into skin cancer if untreated.',
        'severity': 'Moderate Risk'
    },
    'bcc': {
        'full_name': 'Basal Cell Carcinoma',
        'description': 'The most common form of skin cancer. Rarely spreads but needs early treatment.',
        'severity': 'High Risk'
    },
    'bkl': {
        'full_name': 'Benign Keratosis',
        'description': 'A non-cancerous skin growth. Common in older adults, generally harmless.',
        'severity': 'Low Risk'
    },
    'df': {
        'full_name': 'Dermatofibroma',
        'description': 'A benign skin nodule, often found on the legs. Usually harmless and painless.',
        'severity': 'Low Risk'
    },
    'mel': {
        'full_name': 'Melanoma',
        'description': 'The most dangerous form of skin cancer. Early detection is critical for treatment.',
        'severity': '⚠️ Critical Risk'
    },
    'nv': {
        'full_name': 'Melanocytic Nevi',
        'description': 'Common moles caused by clusters of pigmented cells. Usually benign.',
        'severity': 'Low Risk'
    },
    'vasc': {
        'full_name': 'Vascular Lesion',
        'description': 'Abnormalities of blood vessels in the skin. Usually benign.',
        'severity': 'Low Risk'
    }
}

SEVERITY_COLORS = {
    'Low Risk': '#10b981',
    'Moderate Risk': '#f59e0b',
    'High Risk': '#ef4444',
    '⚠️ Critical Risk': '#dc2626'
}

# ─── MODEL LOADING ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(1280, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 7)
    )
    model.load_state_dict(torch.load('skin_disease_efficientnet.pth', map_location='cpu'))
    model.eval()
    return model

# ─── TRANSFORM ──────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ─── PREDICT ────────────────────────────────────────────────────
def predict(image, model):
    tensor = transform(image).unsqueeze(0)  # add batch dimension
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
    return probs.numpy()

# ─── UI ─────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔬 SkinSense AI</h1>
    <p>Upload a dermoscopy image for instant AI-powered skin disease classification</p>
</div>
""", unsafe_allow_html=True)

# load model
try:
    model = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Model not found. Make sure `skin_disease_efficientnet.pth` is in the same folder. Error: {e}")
    model_loaded = False

# upload
uploaded_file = st.file_uploader(
    "Upload a skin lesion image",
    type=["jpg", "jpeg", "png"],
    help="Upload a dermoscopy or clinical image of the skin lesion"
)

if uploaded_file and model_loaded:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Uploaded Image", width='stretch')

    with col2:
        with st.spinner("Analyzing..."):
            probs = predict(image, model)

        pred_idx = np.argmax(probs)
        pred_class = CLASS_NAMES[pred_idx]
        confidence = probs[pred_idx] * 100
        info = DISEASE_INFO[pred_class]
        severity_color = SEVERITY_COLORS[info['severity']]

        st.markdown(f"""
        <div class="result-card">
            <p class="disease-name">{info['full_name']}</p>
            <span class="confidence-badge">Confidence: {confidence:.1f}%</span>
            <p style="margin-top:12px; color:#6b7280; font-size:0.85rem;">
                <span style="color:{severity_color}; font-weight:600;">● {info['severity']}</span>
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-card">
            <h4>About this condition</h4>
            <p>{info['description']}</p>
        </div>
        """, unsafe_allow_html=True)

    # probability chart
    st.markdown("### Probability Distribution")
    full_names = [DISEASE_INFO[c]['full_name'] for c in CLASS_NAMES]
    colors = ['#1a56db' if i == pred_idx else '#c7d7fa' for i in range(7)]

    fig = go.Figure(go.Bar(
        x=[p * 100 for p in probs],
        y=full_names,
        orientation='h',
        marker_color=colors,
        text=[f"{p*100:.1f}%" for p in probs],
        textposition='outside'
    ))
    fig.update_layout(
        plot_bgcolor='#0f172a',
        paper_bgcolor='#0f172a',
        xaxis=dict(title=dict(text='Confidence (%)', font=dict(color='white')), range=[0, 110], showgrid=True, gridcolor='#1e293b', color='white'),
        yaxis=dict(autorange='reversed', color='white'),
        margin=dict(l=160, r=80, t=10, b=40),
        height=360,
        font=dict(family='DM Sans', size=13, color='white')
    )
    st.plotly_chart(fig, width='stretch')

    st.markdown("""
    <div class="warning-box">
        ⚕️ <strong>Medical Disclaimer:</strong> This tool is for educational purposes only and is not a substitute
        for professional medical diagnosis. Always consult a qualified dermatologist for medical advice.
    </div>
    """, unsafe_allow_html=True)

elif not uploaded_file:
    st.markdown("""
    <div class="upload-box">
        <p style="font-size:2rem;">🖼️</p>
        <p style="color:#6b7280; margin:0;">Drag and drop or click above to upload a skin lesion image</p>
        <p style="color:#9ca3af; font-size:0.85rem; margin-top:8px;">Supports JPG, JPEG, PNG</p>
    </div>
    """, unsafe_allow_html=True)