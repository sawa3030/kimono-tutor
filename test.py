# 実際に試してみるパート

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

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28*3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 5),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

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

# image = ImageFolder(root=os.path.expanduser('~/kimono/train_data/cross/image.png'), transform=transform)
# print(image)


image_path = os.path.expanduser('~/kimono/train_data/cross/image.png')
image = Image.open(image_path).convert("RGB") 
image = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    # x, y = x.to(device), y.to(device)
    # pred = model(x.to(device))
    # predicted, actual = classes[pred[0].argmax(0)], classes[y]
    # print(f'Predicted: "{predicted}", Actual: "{actual}"')
    id = model(image.to(device)).argmax(1)
    output = classes[id]
    print(output)