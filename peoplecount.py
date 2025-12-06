import sys
import os
from ultralytics import YOLO
import cv2
import json

video_path = os.path.abspath(sys.argv[1])

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("0")
    sys.exit()

total_people = 0
frame_count = 0
sample_rate = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % sample_rate != 0:
        continue

    results = model(frame, conf=0.25, device="cpu")  # ✅ Force CPU

    for r in results:
        if r.boxes is not None and len(r.boxes.data) > 0:
            for box in r.boxes.data:
                cls = int(box[5])
                if cls == 0:  # ✅ person class
                    total_people += 1

cap.release()

if frame_count == 0:
    print("0")
else:
    processed_frames = frame_count // sample_rate
    avg = total_people // processed_frames if processed_frames > 0 else 0
    print(avg)