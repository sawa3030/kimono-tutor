import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from ultralytics import YOLO
import os

from ultralytics import SAM

model = SAM("sam_b.pt")
model.info()
results = model("input.jpg", save=True)

# save_dir = "output"
# if not os.path.exists(save_dir):
#     os.mkdir(save_dir)

# # Get the segmentation
# model = YOLO("yolov8x-seg.pt")
# results = model.predict("input.jpg", save=True, show=True)
# masks = results[0].masks.data
# boxes = results[0].boxes.data
# clss = boxes[:, 5]
# conf = boxes[:, 4]
# people_indices = torch.where(clss == 0)
# people_id = torch.argmax(conf[people_indices])
# people_mask = masks[people_id].int() * 255
# cv2.imwrite(os.path.join(save_dir,'merged_segs.jpg'), people_mask.cpu().numpy().astype(np.uint8))

# # get the thresholding image
# original_image = cv2.imread("input.jpg")
# gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
# _, gray_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
# cv2.imwrite(os.path.join(save_dir,"gray_output.jpg"), gray_image)

# # get the masked thresholding image
# # print(gray_image.shape)
# # print(people_mask.shape)
# people_mask = cv2.resize(people_mask.cpu().numpy().astype(np.uint8), (gray_image.shape[1], gray_image.shape[0]))
# masked_image = cv2.bitwise_and(gray_image, gray_image, mask=people_mask)
# cv2.imwrite(os.path.join(save_dir,"masked_output.jpg"), masked_image)

# # get the contours and "erimoto"
# original_image = cv2.imread("input.jpg")
# contours, _ = cv2.findContours(masked_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# if contours:
#     print("found contours")
#     largest_contour = max(contours, key=cv2.contourArea)
#     # if cv2.contourArea(largest_contour)*0.8 > cv2.countNonZero(largest_contour):
#     #     contours.remove(largest_contour)
#     #     continue
#     # write largest contour
#     cv2.drawContours(original_image, [largest_contour], -1, (0, 255, 0), 3)

#     bottommost_point = max(largest_contour[:, 0], key=lambda p: p[1])
#     x, y = bottommost_point
#     print("(x,y) = ", x, y)

#     for up_point_y in reversed(range(0, y)):
#         print(x)
#         up_point = (x.item(), up_point_y)
#         print("up_point = ", up_point)
#         if cv2.pointPolygonTest(largest_contour, up_point, False) < 0:
#             cv2.circle(original_image, up_point, 10, (0, 255, 255), thickness=3)
#             break
#     cv2.circle(original_image, (x, y), 10, (0, 255, 255), thickness=3)
# cv2.imwrite(os.path.join(save_dir,"erimoto_output.jpg"), original_image)

# contour_output = cv2.drawContours(original_image, contours, -1, (0, 255, 0))
# # cv2.imwrite("contour_output.jpg", contour_output)
# cv2.imwrite(os.path.join(save_dir,"point_output.jpg"), gray_image)

# model = YOLO("yolov8x-pose.pt")
# results = model.predict("input.jpg")

# keypoints = results[0].keypoints.xy.cpu().numpy()
# for x, y in keypoints[0]:
#     cv2.circle(original_image, (int(x), int(y)), 10, (0, 255, 0), -1)
# cv2.imwrite(os.path.join(save_dir,"pose_output.jpg"), original_image)

# original_image = cv2.imread("input.jpg",cv2.IMREAD_GRAYSCALE)
# laplacian = cv2.Laplacian(original_image, cv2.CV_64F)
# cv2.imwrite(os.path.join(save_dir,"laplacian_output.jpg"), laplacian)

# sobel = cv2.Sobel(original_image, cv2.CV_8UC1, 0, 1, ksize=5)
# cv2.imwrite(os.path.join(save_dir,"sobel_output.jpg"), sobel)

# original_image = cv2.imread("input.jpg")
# canny = cv2.Canny(original_image, 100, 200)
# cv2.imwrite(os.path.join(save_dir,"canny_output.jpg"), canny)
# # cv2.cvtColor(sobel, cv2.COLOR_BGR2LAB)
# lines = cv2.HoughLines(sobel,1,np.pi/180,1000)
# print(len(lines))
# for line in lines:
#     for rho,theta in line:
#         a = np.cos(theta)
#         b = np.sin(theta)
#         x0 = a*rho
#         y0 = b*rho
#         x1 = int(x0 + 1000*(-b))
#         y1 = int(y0 + 1000*(a))
#         x2 = int(x0 - 1000*(-b))
#         y2 = int(y0 - 1000*(a))

#         cv2.line(original_image,(x1,y1),(x2,y2),(0,0,255),2)

# cv2.imwrite(os.path.join(save_dir,'houghlines3.jpg'),original_image)

