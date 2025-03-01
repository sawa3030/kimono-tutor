from ultralytics import YOLO
import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch

# model = YOLO('./runs/train/weights/best.pt')
# print(model)
# results = model("data/train/images/DSC_3239.JPG")
# # results = model("./data/test/images/y17_jpg.rf.1134f8afd5cda5d2a789b0474d537baa.jpg")
# results[0].show()

model = YOLO("yolov8x-seg.pt")
results = model.predict("input.jpg", save=True, show=True)
print("-----results-----")

masks = results[0].masks.data
boxes = results[0].boxes.data
clss = boxes[:, 5]
conf = boxes[:, 4]
people_indices = torch.where(clss == 0)
people_id = torch.argmax(conf[people_indices])
people_mask = masks[people_id].int() * 255
cv2.imwrite('merged_segs.jpg', people_mask.cpu().numpy().astype(np.uint8))

image = cv2.imread("input.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, image = cv2.threshold(image, 177, 255, cv2.THRESH_BINARY)
cv2.imwrite("output.jpg", image)
