# SkinSense AI

An explainable deep learning pipeline for skin lesion classification on the HAM10000 dataset — built with class-imbalance handling, a leakage-free evaluation split, multi-backbone comparison, and Grad-CAM explainability.

---

## Why this project

Skin lesion classification is a 7-class problem with two properties that make it harder than a typical benchmark:

- **Severe class imbalance** — benign nevi dominate the dataset while melanoma, the class where a false negative matters most, is comparatively rare.
- **The need for trust, not just a label** — a bare prediction isn't enough in a health-adjacent context; the model's reasoning needs to be inspectable.

This project treats both of those as first-class engineering problems, not afterthoughts.

---

## Results

| Model | Test Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| **EfficientNet-B0** | 82.2% | 0.7477 | 0.8294 |
| **ResNet50** | 69.8% | 0.5824 | 0.7229 |

Numbers above are computed on a **locked, held-out test set** (`test_split.csv`) that never touches training or model selection — see [Evaluation Methodology](#evaluation-methodology) below for why that matters.

---

## Project Structure

```
Pytorch-Skin-Diseases-classification/
│
├── App/
│   └── app.py                  # Streamlit inference app
│
├── Notebooks/
│   ├── train_runner.ipynb
│   └── Evaluate.ipynb
│
├── src/
│   ├── dataloader.py            # SkinDataset, class_to_idx
│   ├── transforms.py            # train/val augmentation pipelines
│   ├── models.py                # build_model() — EfficientNet-B0 / ResNet50 / ConvNeXt-Tiny / CustomCNN
│   ├── train.py                 # training loop, class weighting, MLflow logging
│   ├── evaluate.py               # accuracy / F1 / confusion matrix / per-class accuracy
│   ├── predict.py                # single-image inference + Grad-CAM
│   └── gradcam.py                # architecture-aware Grad-CAM target layers
│
├── HAM10000_images/              # (not tracked — see Data section)
├── HAM10000_metadata.csv
├── test_split.csv                # locked held-out split (generated on first train run)
├── best_model_efficientnet.pth
├── best_model_resnet.pth
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone <repo-url>
cd Pytorch-Skin-Diseases-classification
pip install -r requirements.txt
```

Download the HAM10000 dataset and place the images in `HAM10000_images/` and the metadata CSV in the project root as `HAM10000_metadata.csv`.

---

## Usage

### Train a model

```python
from src.train import train_model

train_model(
    model_name="efficientnet",   # or "resnet", "convnext", "customcnn"
    batch_size=32,
    epochs=20,
    learning_rate=1e-3
)
```

The first training run creates `test_split.csv` — a locked 15% test set that every subsequent model is evaluated against, regardless of when it's trained.

### Evaluate

Open `Notebooks/Evaluate.ipynb`, which loads each checkpoint, runs it against the locked test set, and reports accuracy, macro/weighted F1, a confusion matrix, and per-class accuracy.

### Run inference + Grad-CAM

```python
from src.predict import predict_image

result = predict_image(model, image_path, device, model_name="efficientnet")
# result["prediction"], result["confidence"], result["heatmap"]
```

### Launch the app

```bash
streamlit run App/app.py
```

Upload an image, pick a model from the dropdown (only trained checkpoints appear), and see the prediction, confidence, and Grad-CAM heatmap side by side.

---

## Evaluation Methodology

Early iterations of this project used a simple 80/20 train/validation split — but the validation set was also being used *during training* to pick the best checkpoint and drive the LR scheduler. That means the "best model" was implicitly selected to perform well on validation data, making any evaluation on that same set optimistic rather than a true measure of generalization.

This was fixed with a proper **70/15/15 train/validation/test split**, stratified by diagnosis. The test split is generated once and persisted to `test_split.csv` — it is never regenerated, so every model trained in this project (regardless of when) is evaluated against the exact same held-out data. This is what makes the EfficientNet vs. ResNet comparison above valid rather than coincidental.

---

## Handling Class Imbalance

Rather than architectural workarounds, imbalance is handled at the loss level with class-balanced weighting:

```python
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(7),
    y=train_df["dx"].map(class_to_idx)
)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
```

MixUp/CutMix and weighted random sampling were considered but deliberately deferred — layering multiple interventions at once makes it impossible to attribute any improvement to a specific cause.

---

## Explainability

Grad-CAM is integrated via `pytorch-grad-cam`, with the target convolutional layer resolved per architecture (EfficientNet's `.features[-1]`, ResNet's `.layer4[-1]`, etc.) — a single hardcoded layer would silently break or produce meaningless heatmaps on a different backbone. Verified on held-out test images that the highlighted region tracks the actual lesion rather than background artifacts (skin markers, rulers) present in some HAM10000 images.

---

## Experiment Tracking

All runs — parameters, per-epoch metrics, and model artifacts — are logged via MLflow:

```bash
mlflow ui
```

---

## Roadmap

- [x] Class imbalance handling
- [x] Locked train/val/test split
- [x] EfficientNet-B0 vs. ResNet50 comparison
- [x] Grad-CAM explainability
- [x] Streamlit app with model comparison
- [ ] Train ConvNeXt-Tiny and CustomCNN backbones
- [ ] Docker containerization
- [ ] Deployment

---

## License

MIT (or update to match your intended license).