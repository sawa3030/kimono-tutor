import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Lambda, Compose
from PIL import Image
import matplotlib.pyplot as plt
import os
from common import transform
from common import NeuralNetwork
import pickle

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

image_path = os.path.expanduser('~/kimono/train_data/cross/image.png')
image = Image.open(image_path).convert("RGB") 
image = transform(image).unsqueeze(0).to(device)


with open('class_to_idx.pickle.pkl', 'rb') as f:
    class_to_idx = pickle.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    with torch.no_grad():
        id = model(image.to(device)).argmax(1)
        output = classes[id]
        print(output)