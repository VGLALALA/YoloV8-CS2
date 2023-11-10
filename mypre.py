from ultralytics import YOLO
from PIL import Image
import cv2
'''
model = YOLO("/media/vglalala/File/ultralytics/runs/detect/train/weights/best.pt")
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
#results = model.predict(source="0")
results = model.predict(source="/media/vglalala/File/ultralytics/cg2.v3i.yolov8/valid/images", show=True, save=True) # Display preds. Accepts all YOLO predict arguments
'''
# Initialize the YOLO model
model = YOLO("/media/vglalala/File/ultralytics/runs/detect/train/weights/best.pt")

# Loop to capture the screen continuously
while True:
    # Capture the screen
    screenshot = pyautogui.screenshot()
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Perform prediction using YOLO
    results = model.predict(frame)

    # Display predictions (You can customize how you want to show them)
    for (x, y, w, h, label, conf) in results.xywh[0]:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} {conf:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Screen Capture with YOLO", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cv2.destroyAllWindows()
