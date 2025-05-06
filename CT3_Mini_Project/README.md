# Intelligent Object Counting and Speed Monitoring System using YOLOv8

## 📌 Project Objective

To build a real-time system that detects, tracks, counts, and estimates the speed of objects in surveillance videos using YOLOv8. The system includes region-based object analytics and a web-based interface using Flask, containerized with Docker.

## 🎯 Features

- YOLOv8-based Object Detection
- Object Tracking using Norfair
- Global and Region-wise Object Counting
- Speed Estimation based on pixel displacement
- Flask Web App for video upload, processing, and playback
- Dockerized Deployment

## 📁 Folder Structure

CT3\_Mini\_Project/
├── app.py                 # Flask web server
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container configuration
├── weights/
│   └── yolov8n.pt         # YOLOv8 pretrained weights
├── scripts/
│   └── detect.py          # Object detection/tracking logic
├── uploads/               # Uploaded videos (auto-created)
├── outputs/               # Processed videos (auto-created)
├── templates/
│   └── index.html         # Web UI
└── data/                  # (Optional) training/validation videos


## 🚀 How to Run Locally

### 1️⃣ Clone this repo

bash
git clone https://github.com/<your-username>/CT3_Mini_Project.git
cd CT3_Mini_Project

### 2️⃣ Build Docker Image

bash
docker build -t yolov8-object-tracker .


### 3️⃣ Run the Container

```bash
docker run -p 5000:5000 yolov8-object-tracker
```

### 4️⃣ Open in Browser

Navigate to: [http://localhost:5000](http://localhost:5000)

---

## 🌐 Web Interface

* Upload any `.mp4` surveillance video
* Video is processed with detection, tracking, and speed annotations
* Download and view the result directly from the browser

## 🧠 Technical Stack

| Component        | Tool/Library                                                       |
| ---------------- | ------------------------------------------------------------------ |
| Object Detection | [YOLOv8 - Ultralytics](https://github.com/ultralytics/ultralytics) |
| Tracking         | [Norfair](https://github.com/tryolabs/norfair)                     |
| Video Processing | OpenCV                                                             |
| Web App          | Flask                                                              |
| Containerization | Docker                                                             |


## 🎓 Learning Outcomes

* YOLOv8 implementation for real-time object detection
* Multi-object tracking and zone-wise analytics
* Flask app integration with video processing
* Dockerized deployment of a computer vision pipeline


## 🙋‍♂️ Developed By

**Lokesh Kumar Kannan**
M.Tech (AI)