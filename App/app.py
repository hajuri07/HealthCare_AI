from fastapi import FastAPI, UploadFile, File
import torch
import io
from src.models import build_model
from src.transforms import val_transform
from src.llm_summary import generate_summary
from src.gradcam import generate_gradcam
import numpy as np
import base64

from io import BytesIO
from PIL import Image
import cv2
app = FastAPI()

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRIAGE = {
    "akiec": "Moderate concern — see a dermatologist",
    "bcc":   "High concern — see a dermatologist promptly",
    "bkl":   "Low concern — benign, monitor for changes",
    "df":    "Low concern — benign, monitor for changes",
    "mel":   "URGENT — see a dermatologist immediately",
    "nv":    "Low concern — benign, monitor for changes",
    "vasc":  "Low concern — benign, monitor for changes",
}

model = build_model(model_name="efficientnet", num_classes=7).to(DEVICE)
model.load_state_dict(torch.load("C:\Ibrahim\Projects\Healthcare\Pytorch-Skin-Diseases-classification\efficient_best_model.pth", map_location=DEVICE))
model.eval()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    tensor = val_transform(pil_image).unsqueeze(0).to(DEVICE)
    tensor.requires_grad_()  # grad-cam needs gradients flowing

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()

    pred_class = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx].item()

    # prep original image as float [0,1] RGB for grad-cam overlay
    resized = pil_image.resize((224, 224))
    original_np = np.array(resized).astype(np.float32) / 255.0

    cam_image = generate_gradcam(model, tensor, original_np, model_name="efficientnet")

    # encode heatmap as base64 PNG so Flutter can display it directly
    cam_pil = Image.fromarray(cam_image)
    buf = BytesIO()
    cam_pil.save(buf, format="PNG")
    cam_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    summary = generate_summary(pred_class, confidence, TRIAGE[pred_class])

    return {
        "prediction": pred_class,
        "confidence": round(confidence, 4),
        "triage_message": TRIAGE[pred_class],
        "ai_summary": summary,
        "gradcam_image": cam_base64,   # Flutter decodes this to show the heatmap
        "disclaimer": "This is not a medical diagnosis. Consult a healthcare professional."
    }