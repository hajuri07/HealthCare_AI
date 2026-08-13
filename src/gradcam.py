import numpy as np
import torch

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def get_target_layer(model, model_name):
    model_name = model_name.lower().strip()
    if model_name == "efficientnet":
        return [model.features[-1]]
    elif model_name == "resnet":
        return [model.layer4[-1]]
    elif model_name == "convnext":
        return [model.features[-1]]
    elif model_name == "customcnn":
        return [model.features[-4]]
    else:
        raise ValueError(f"No Grad-CAM target layer defined for: {model_name}")


def generate_gradcam(model, image_tensor, original_image, model_name):
    model.eval()

    target_layers = get_target_layer(model, model_name)

    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=image_tensor)[0]

    visualization = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)

    return visualization
