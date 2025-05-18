import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from ultralytics import YOLO
import os
from enum import Enum

class Label(Enum):
    KIMONO = 0
    ERI = 1
    OBI = 2
    HANERI = 3

save_dir = "output"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# Get the segmentation
model = YOLO("weights/best.pt")
results = model.predict("input.jpg", save=True)

# print("====results====")
# print(results[0])
# print("====results end====")

masks = results[0].masks.data
classes = results[0].boxes.cls

for c, mask in zip(classes, masks):
    mask = mask.int() * 255
    if c == Label.KIMONO.value:
        cv2.imwrite(os.path.join(save_dir,'kimono.jpg'), mask.cpu().numpy().astype(np.uint8))
    elif c == Label.ERI.value:
        cv2.imwrite(os.path.join(save_dir,'eri.jpg'), mask.cpu().numpy().astype(np.uint8))
    elif c == Label.OBI.value:
        cv2.imwrite(os.path.join(save_dir,'obi.jpg'), mask.cpu().numpy().astype(np.uint8))
    elif c == Label.HANERI.value:
        cv2.imwrite(os.path.join(save_dir,'haneri.jpg'), mask.cpu().numpy().astype(np.uint8))

# people_mask = masks[Label.ERI.value].int() * 255
# cv2.imwrite(os.path.join(save_dir,'merged_segs.jpg'), people_mask.cpu().numpy().astype(np.uint8))