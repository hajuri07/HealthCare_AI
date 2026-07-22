import torch
from sklearn.metrics import classification_report,confusion_matrix

def evaluate(model,test_loader,device):

    model.eval()

    preds = []
    labels = []

    with torch.no_grad():

        for images, target in test_loader:

            images = images.to(device)
            outputs = model(images)
            _, prediction = torch.max(outputs, 1)
            preds.extend(prediction.cpu().numpy())
            labels.extend(target.numpy())

    print(classification_report(labels, preds))
    print(confusion_matrix(labels, preds))