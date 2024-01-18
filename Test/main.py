import os

from ultralytics import YOLO
import cv2

VIDEOS_NAME='detty.mov'
VIDEOS_DIR = os.path.join('.', 'my_cat_vids')
OUTPUTS_DIR = os.path.join('.', 'outputs')
OUTPUT_VID_NAME, ext = os.path.splitext(VIDEOS_NAME)

video_path = os.path.join(VIDEOS_DIR, VIDEOS_NAME)
video_path_out = os.path.join(OUTPUTS_DIR, '{}_out.mp4'.format(OUTPUT_VID_NAME))

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape

# Write new video into a directory
out = cv2.VideoWriter(
    video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), 
    int(cap.get(cv2.CAP_PROP_FPS)), 
    (W, H)
)


# Load a model
model_path = "mycatyolosv8.pt"
model = YOLO(model_path)  # load a custom model

threshold = 0.5

while ret:
    results = model(frame)[0]
    # Get predictions
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        # Display prediction and bounding boxes
        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            label = f"{results.names[int(class_id)].upper()} {score:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()