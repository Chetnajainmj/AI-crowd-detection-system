# backend/ai/people_count_debug.py
import sys
import os
import cv2
from ultralytics import YOLO

if len(sys.argv) < 2:
    print("0")
    sys.exit(0)

video_path = sys.argv[1]
model = YOLO("yolov8n.pt")  # change to yolov8m.pt for more accuracy

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("0")
    sys.exit(0)

# Prepare output
dirname, filename = os.path.split(video_path)
out_name = os.path.join(dirname, f"annotated_{filename}")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS) or 25
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(out_name, fourcc, fps, (w, h))

frame_skip = 3        # process every 3rd frame (reduce for more accuracy)
conf_threshold = 0.25 # lower -> detect smaller/weaker boxes
max_people = 0
frame_index = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_index += 1
    if frame_index % frame_skip != 0:
        out.write(frame)  # still write original frame
        continue

    results = model(frame, imgsz=640, conf=conf_threshold, verbose=False)
    detections = results[0].boxes  # Boxes object
    people_in_frame = 0

    # draw boxes for person class only
    if detections is not None and len(detections) > 0:
        for box in detections:
            cls = int(box.cls[0])
            conf = float(box.conf[0]) if hasattr(box, "conf") else float(box[4])
            if cls != 0:
                continue
            people_in_frame += 1
            xyxy = box.xyxy[0].astype(int)  # [x1,y1,x2,y2]
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"person {conf:.2f}", (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    max_people = max(max_people, people_in_frame)
    # overlay count on frame
    cv2.rectangle(frame, (0,0), (220,40), (0,0,0), -1)
    status_text = f"Frame:{frame_index} Count:{people_in_frame} Max:{max_people}"
    cv2.putText(frame, status_text, (8,28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    out.write(frame)

cap.release()
out.release()

# print only the number for Node.js
print(max_people)
# Also print output file path to stderr for debugging if needed
sys.stderr.write(f"ANNOTATED_VIDEO={out_name}\n")