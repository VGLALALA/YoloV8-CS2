from ultralytics import YOLO

model = YOLO('/media/vglalala/File/ultralytics/runs/detect/train/weights/best.pt')
model.export(format='onnx', dynamic=True)