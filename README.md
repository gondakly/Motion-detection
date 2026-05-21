# Smart YOLO Traffic Surveillance System

A real-time intelligent traffic monitoring system built with **YOLOv8**, **OpenCV**, and **Streamlit**. It detects, tracks, and counts vehicles crossing a virtual line while displaying motion masks, velocity trails, and live statistics.

## Features

- **Real-time Vehicle Detection & Tracking** using YOLOv8 + ByteTrack
- **Bidirectional Counting** (IN ↓ / OUT ↑) with virtual line
- **Motion Detection** using Background Subtraction (MOG2)
- **Velocity Estimation** with smoothing
- **Object Trails** (motion history visualization)
- **Live Statistics Dashboard** (FPS, Active Objects, Total Cars, etc.)
- **Customizable Parameters** via Streamlit sidebar
- Supports **uploaded videos** or default local video

## How It Works

1. **Motion Detection**: Uses MOG2 background subtractor to highlight moving objects.
2. **Detection & Tracking**: YOLOv8n model with ByteTrack tracker identifies and follows vehicles/persons/bicycles.
3. **Counting Logic**: A horizontal virtual line is drawn. When an object crosses the line with sufficient vertical velocity, it is counted as IN or OUT.
4. **Visualization**: Bounding boxes, class labels, tracking IDs, and motion trails are drawn on the frame.
5. **Statistics**: Real-time metrics are updated in the dashboard.

## 🛠 Technologies Used

- **Streamlit** - Web Interface
- **Ultralytics YOLOv8** - Object Detection & Tracking
- **OpenCV** - Video processing, drawing, background subtraction
- **Python 3.9+**

---

## Installation & Running (Manual)

### 1. Clone the Repository
git clone https://github.com/gondakly/Motion-detection
-**Run Command**
  -Streamlit run "Path of the file on your device"
