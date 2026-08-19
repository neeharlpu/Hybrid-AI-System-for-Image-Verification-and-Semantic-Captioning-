import torch
import torch.nn as nn
from torchvision import models

def load_model(device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    model.load_state_dict(torch.load("models/model.pth", map_location=device))
    model.to(device)
    model.eval()

    return model


def predict(image_tensor, model):
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    classes = ['FAKE', 'REAL']
    return classes[pred.item()], conf.item()