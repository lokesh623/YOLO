# Use official lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install required packages
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download YOLOv8 weights (if not already copied)
RUN mkdir -p weights && \
    python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" || echo "weights will be provided manually"

# Expose port
EXPOSE 5000

# Start the Flask app
CMD ["python", "app.py"]
