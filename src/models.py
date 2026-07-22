#Cusotm cnn Model
import torch
import torch.nn as nn
import torchvision.models as models

class CustomCNN(nn.Module):

    def __init__(self,num_classes=7):
        super ().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3,32,kernel_size =3 , padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(32,64,kernel_size =3 , padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64,128,kernel_size =3 , padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*28*28,512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,num_classes)
        )
    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def custom_classifier(in_features,num_classes):
    return nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes) 
        )
        

def build_model(model_name: str, num_classes: int)->nn.Module:
    model_name = model_name.lower().strip()

    if model_name == "customcnn":
        return CustomCNN(num_classes)

     
    elif model_name == "efficientnet":
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )
        
        
        in_features = model.classifier[1].in_features 
        
        
        model.classifier = custom_classifier(in_features=in_features,num_classes=num_classes)
        return model

    
    elif model_name == "resnet":
        model = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT
        )
        
        in_features = model.fc.in_features
        
        
        model.fc = custom_classifier(in_features=in_features,num_classes=num_classes)
        
        return model

    
    elif model_name == "convnext":
        model = models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.DEFAULT
        )
        
        in_features = model.classifier[2].in_features
        
        
        model.fc = custom_classifier(in_features=in_features,num_classes=num_classes)
        return model

    else:
        raise ValueError(f"Invalid model name: '{model_name}'")







