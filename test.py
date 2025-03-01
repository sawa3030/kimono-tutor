import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from ultralytics import YOLO

# Get the segmentation
model = YOLO("yolov8x-seg.pt")
results = model.predict("input.jpg", save=True, show=True)
masks = results[0].masks.data
boxes = results[0].boxes.data
clss = boxes[:, 5]
conf = boxes[:, 4]
people_indices = torch.where(clss == 0)
people_id = torch.argmax(conf[people_indices])
people_mask = masks[people_id].int() * 255
cv2.imwrite('merged_segs.jpg', people_mask.cpu().numpy().astype(np.uint8))

# get the thresholding image
original_image = cv2.imread("input.jpg")
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
_, gray_image = cv2.threshold(gray_image, 177, 255, cv2.THRESH_BINARY)
cv2.imwrite("gray_output.jpg", gray_image)

# get the masked thresholding image
people_mask = cv2.resize(people_mask.cpu().numpy().astype(np.uint8), (1108, 1477))
masked_image = cv2.bitwise_and(gray_image, gray_image, mask=people_mask)
cv2.imwrite("masked_output.jpg", masked_image)

# get the contours and "erimoto"
contours, _ = cv2.findContours(masked_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    bottommost_point = max(largest_contour[:, 0], key=lambda p: p[1])  # p[1] はY座標
    x, y = bottommost_point
    print("(x,y) = ", x, y)
    cv2.circle(gray_image, (x, y), 10, (255, 0, 0))

contour_output = cv2.drawContours(original_image, contours, -1, (0, 255, 0))
# cv2.imwrite("contour_output.jpg", contour_output)
cv2.imwrite("point_output.jpg", gray_image)

model = YOLO("yolov8x-pose.pt")
results = model.predict("input.jpg")

keypoints = results[0].keypoints.xy.cpu().numpy()
confs = results[0].keypoints.conf.cpu().numpy()
print(confs)
id = np.argmax(confs)
print(id)
# for person in keypoints:
for x, y in keypoints[0]:
        cv2.circle(original_image, (int(x), int(y)), 5, (0, 255, 0), -1)
