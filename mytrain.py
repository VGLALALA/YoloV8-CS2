from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO('/media/vglalala/File/ultralytics/ultralytics/cfg/models/v8/myyolov8.yaml')

# Load a pretrained YOLO model (recommended for training)
model = YOLO('/media/vglalala/File/ultralytics/yolov8n.pt')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(data='/media/vglalala/File/ultralytics/ultralytics/cfg/datasets/mycoco128.yaml', epochs=30)

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model
#results = model('https://ultralytics.com/images/bus.jpg')

# Export the model to ONNX format
#success = model.export(format='onnx')
