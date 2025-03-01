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
# for i in range(len(results)):
for result in results:
    masks = result.masks.data
    boxes = result.boxes.data
    # extract classes
    clss = boxes[:, 5]
    # get indices of results where class is 0 (people in COCO)
    people_indices = torch.where(clss == 0)
    # use these indices to extract the relevant masks
    people_masks = masks[people_indices]
    # scale for visualizing results
    people_mask = torch.any(people_masks, dim=0).int() * 255
    # save to file
    print(people_mask.cpu().numpy())
    cv2.imwrite('merged_segs.jpg', people_mask.cpu().numpy().astype(np.uint8))
    # print(type(results[i].masks.data))
    # print(results[i].masks.data.shape)
    # # cv2.imwrite("mask"+str(i)+".jpg", results[i].masks.cpu().numpy().masks[0])
    # cv2.imwrite("mask"+str(i)+".jpg", results[i].masks.data[0])

image = cv2.imread("input.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, image = cv2.threshold(image, 177, 255, cv2.THRESH_BINARY)
cv2.imwrite("output.jpg", image)
