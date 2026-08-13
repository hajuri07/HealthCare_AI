import numpy as np
import torch

from PIL import Image

from src.gradcam import generate_gradcam
from src.transforms import val_transform


CLASS_NAMES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]


def predict_image(model, image_path, device,model_name):

    
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image_tensor = val_transform(image).unsqueeze(0).to(device)

    resized_image = image.resize((224, 224))
    original = np.array(resized_image).astype(np.float32) / 255.0
    with torch.no_grad():

        outputs = model(image_tensor)

        probs = torch.softmax(outputs, dim=1)

        confidence, pred = torch.max(probs, dim=1)

   
    heatmap = generate_gradcam(
        model,
        image_tensor,
        original,
        model_name
    )
    

    return {
        "prediction": CLASS_NAMES[pred.item()],
        "confidence": confidence.item(),
        "probabilities": probs.squeeze().cpu().numpy(),
        "heatmap": heatmap
    }
