from ultralytics import YOLO

model = YOLO('./runs/train/weights/best.pt')
print(model)
results = model("data/train/images/DSC_3239.JPG")
# results = model("./data/test/images/y17_jpg.rf.1134f8afd5cda5d2a789b0474d537baa.jpg")
results[0].show()