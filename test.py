import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from ultralytics import YOLO
import os

save_dir = "output"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# Get the segmentation
model = YOLO("weights/best.pt")
results = model.predict("input.jpg", save=True, show=True)

def evaluate_pic(img_path):
    model = YOLO("weights/best.pt")
    results = model.predict(img_path)
    return "evaluate done"
