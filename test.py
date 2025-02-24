from ultralytics import YOLO

model = YOLO('./runs/train/weights/best.pt')
print(model)
results = model("./data/test/images/y2_jpg.rf.946e6ccf974efc6a756f520f83ba9f81.jpg")
# results = model("./data/test/images/y17_jpg.rf.1134f8afd5cda5d2a789b0474d537baa.jpg")
results[0].show()