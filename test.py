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

        image = mask.cpu().numpy().astype(np.uint8)
        contours, _ = cv2.findContours(mask.cpu().numpy().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            print("No contours found.")
            continue
        largest_contour = max(contours, key=cv2.contourArea)
        points = largest_contour.reshape(-1, 2)
        sum_pts = points[:, 0] + points[:, 1] * 2
        diff_pts = points[:, 0] - points[:, 1] * 2

        top_left = points[np.argmin(sum_pts)]
        bottom_right = points[np.argmax(sum_pts)]
        top_right = points[np.argmin(diff_pts)]
        bottom_left = points[np.argmax(diff_pts)]

        corner_points = {
            "top_left": tuple(top_left),
            "top_right": tuple(top_right),
            "bottom_left": tuple(bottom_left),
            "bottom_right": tuple(bottom_right),
        }

        img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for name, pt in corner_points.items():
            cv2.circle(img_color, pt, 6, (0, 0, 255), -1)
            cv2.putText(img_color, name, (pt[0] + 5, pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(os.path.join(save_dir,'obi.jpg'), img_color)

    elif c == Label.HANERI.value:
        cv2.imwrite(os.path.join(save_dir,'haneri.jpg'), mask.cpu().numpy().astype(np.uint8))
