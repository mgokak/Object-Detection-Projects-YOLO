# Conveyor Belt Bottle Detection

## Overview

This repository contains a computer vision project that detects **bottles moving on a conveyor belt** using object detection techniques. The system processes video frames in real time, identifies bottles, and draws bounding boxes around detected objects.

The goal of this project is to demonstrate how deep learning models can be integrated into a real-world operational pipeline where data arrives continuously and decisions must be made instantly.

This project demonstrates:
- Video processing using OpenCV  
- Deep learning–based object detection  
- Real-time inference workflow  
- Industrial automation use case  

Applications include manufacturing monitoring, bottle presence verification, and smart factory automation.

---

## Project Flow

The system follows a sequential pipeline:

1. Import required libraries  
2. Load trained YOLO model  
3. Read video stream  
4. Extract frames continuously  
5. Run object detection on each frame  
6. Draw bounding boxes  
7. Display or save output  

```
Video → Frame Extraction → Model Inference → Detection → Visualization
```

This frame-by-frame processing enables the system to work in real-time environments such as production lines or live camera feeds.

---

## Installation

Install required dependencies:

```
pip install opencv-python numpy ultralytics matplotlib
```

---

## Project Structure

```
Conveyor_Belt_Bottle_detection.ipynb
best.pt
input_video.mp4
output_video.mp4
README.md
```

---

## Import Libraries

```python
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
```

### Explanation

- **cv2** – Captures and processes video frames  
- **numpy** – Handles numerical operations and array manipulation  
- **YOLO** – Loads and runs the object detection model  
- **matplotlib** – Optional visualization support  

These libraries form the core components of the video analytics pipeline.

---

## Load the Detection Model

```python
model = YOLO("best.pt")
```

### Explanation

This step loads the trained model weights into memory. The model contains learned features that allow it to recognize bottles in new video frames. Once loaded, the model is ready to perform inference without additional training.

---

## Video Input Processing

```python
cap = cv2.VideoCapture("input_video.mp4")
```

Frame extraction loop:

```python
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
```

### Explanation

The video is processed one frame at a time instead of loading the entire file into memory. This approach:
- Reduces memory usage  
- Enables real-time processing  
- Allows deployment with live camera feeds  

This technique is commonly used in surveillance systems and industrial monitoring.

---

## Object Detection on Frames

```python
results = model(frame, conf=0.5)
```

### Explanation

Each frame is passed through the neural network. The model returns:
- Bounding box coordinates  
- Confidence score  
- Predicted class  

The confidence threshold helps remove weak or incorrect detections, improving output reliability.

---

## Drawing Bounding Boxes

```python
for result in results:
    boxes = result.boxes.xyxy
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
```

### Explanation

Bounding boxes visually highlight detected bottles. This allows quick verification of detection accuracy and helps operators monitor the production line visually.

---

## Display Output

```python
cv2.imshow("Bottle Detection", frame)

if cv2.waitKey(1) & 0xFF == ord('q'):
    break
```

Cleanup:

```python
cap.release()
cv2.destroyAllWindows()
```

### Explanation

Frames are displayed continuously to create a live monitoring interface. Proper resource cleanup ensures the system runs smoothly during long operations without memory or device issues.

---

## How YOLO Works

YOLO (You Only Look Once) is a single-stage object detection algorithm that processes the entire image in one pass. Instead of generating multiple region proposals, the model predicts object locations and classes simultaneously.

This approach makes YOLO:
- Fast  
- Efficient  
- Suitable for real-time applications  

The model divides the image into a grid and predicts bounding boxes along with confidence scores for each region.

---

## Real-Time Processing Concept

Video analytics introduces challenges such as motion blur, lighting changes, and object overlap. Frame-by-frame processing allows the system to handle continuous data streams and respond immediately.

This design is used in:
- Smart manufacturing  
- Traffic monitoring  
- CCTV analytics  
- Retail automation  

---

## Industrial Context

Manual inspection on conveyor belts is slow and prone to errors. This system demonstrates how computer vision can automate:

- Bottle presence detection  
- Production line monitoring  
- Missing object identification  

With additional logic, the system can trigger alerts, stop machinery, or generate operational statistics.

---

## Detection Pipeline (Detailed)

Each frame follows the same lifecycle:

1. Frame Capture  
2. Preprocessing (resize and normalize)  
3. Model Inference  
4. Post-processing (confidence filtering and NMS)  
5. Visualization  

This loop runs continuously to maintain real-time performance.

---

## Limitations

Current implementation focuses only on detection. It does not include:

- Object tracking across frames  
- Bottle counting  
- Defect detection  
- Production analytics  

These features can be added in future versions.

---

## Future Improvements

Possible extensions:

- Object tracking using SORT or DeepSORT  
- Automatic bottle counting  
- Defect detection model  
- Edge deployment (Jetson / Raspberry Pi)  
- Cloud-based monitoring  
- Production analytics dashboard  

---

## Learning Outcomes

This project demonstrates:

- Real-time computer vision pipeline design  
- Integration of deep learning inference into applications  
- Video analytics using OpenCV  
- End-to-end system thinking for industrial AI  

---

## Portfolio Value

This project highlights practical experience in:

- Computer Vision  
- Real-time inference systems  
- Industrial automation use cases  

Relevant for roles such as:
- Computer Vision Engineer  
- Machine Learning Engineer  
- AI Engineer  
- Edge AI Developer  

---

## Author

**Manasa Vijayendra Gokak**  
Graduate Student – Data Science 
