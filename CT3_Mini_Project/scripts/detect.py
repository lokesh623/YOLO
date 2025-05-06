import os
import cv2
import numpy as np
from ultralytics import YOLO
from norfair import Detection, Tracker
import argparse
from pathlib import Path

# Load YOLOv8 model
model = YOLO("weights/yolov8n.pt")
tracker = Tracker(distance_function="euclidean", distance_threshold=30)

# Define zones for region-wise counting
zones = {
    "Zone A": (100, 100, 400, 300),
    "Zone B": (500, 100, 800, 300)
}

def process_video(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    cap = cv2.VideoCapture(str(input_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_index = 0
    id_history = {}  # ID -> list of (frame_index, x, y)
    zone_counts = {zone: set() for zone in zones}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            detections.append(Detection(points=np.array([[cx, cy]])))

        tracked_objects = tracker.update(detections)

        for obj in tracked_objects:
            x, y = obj.estimate[0]
            obj_id = obj.id

            if obj_id not in id_history:
                id_history[obj_id] = []
            id_history[obj_id].append((frame_index, x, y))

            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.putText(frame, f"ID:{obj_id}", (int(x), int(y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            for zone_name, (zx1, zy1, zx2, zy2) in zones.items():
                if zx1 <= x <= zx2 and zy1 <= y <= zy2:
                    zone_counts[zone_name].add(obj_id)

            if len(id_history[obj_id]) >= 2:
                t1, x1, y1 = id_history[obj_id][-2]
                t2, x2, y2 = id_history[obj_id][-1]
                dist_px = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                time_sec = (t2 - t1) / fps
                speed = (dist_px / time_sec) if time_sec > 0 else 0
                cv2.putText(frame, f"{speed:.1f}px/s", (int(x), int(y) + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        for zone_name, (zx1, zy1, zx2, zy2) in zones.items():
            cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (255, 0, 0), 2)
            cv2.putText(frame, f"{zone_name}: {len(zone_counts[zone_name])}",
                        (zx1, zy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        out.write(frame)
        frame_index += 1

    cap.release()
    out.release()
    print(f"Processed: {input_path.name} → {output_path}")

# Main CLI entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object tracking and speed estimation using YOLOv8.")
    parser.add_argument("--input", required=True, help="Path to input video file")
    parser.add_argument("--output", required=True, help="Path to save output video")
    args = parser.parse_args()

    process_video(args.input, args.output)
