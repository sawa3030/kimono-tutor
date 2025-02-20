import os
import pickle

import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Lambda, ToTensor

from common import NeuralNetwork, transform

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth"))

classes = [
    "cross",
    "maki",
    "otaiko",
    "tare",
    "tesaki",
]

model.eval()

image_path = os.path.expanduser("~/kimono/train_data/cross/image.png")
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)


with open("class_to_idx.pickle.pkl", "rb") as f:
    class_to_idx = pickle.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    with torch.no_grad():
        id = model(image.to(device)).argmax(1)
        output = classes[id]
        print(output)
