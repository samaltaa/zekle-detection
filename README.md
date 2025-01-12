# Zekle - Face Recognition and Object Detection with YOLOv8

This project integrates YOLOv8 object detection with face recognition for identifying and labeling individuals in real-time video streams. It uses a pre-trained YOLO model for object detection and the `face_recognition` library for facial encoding and matching.

## Features
- **Real-Time Object Detection**: Detects objects in a video stream using YOLOv8.
- **Face Recognition**: Identifies and labels known faces by comparing them to a dataset of pre-encoded facial images.
- **Live Video Processing**: Processes frames from a live webcam feed.
- **Customizable Dataset**: Easily add or update known faces by placing images in the `known_faces` directory.

## Prerequisites
Ensure the following are installed:
- Python 3.8 or later
- OpenCV
- `ultralytics` (for YOLOv8)
- `face_recognition` library
- NumPy

## Installation
1. Clone the repository:
   ```bash
   git clone  https://github.com/samaltaa/zekle-detection.git
   ```
2. Navigate to the project directory:
   ```bash
   cd face_recognition_yolo
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Place images of known individuals in the `known_faces` directory. Ensure the images are in `.jpg`, `.jpeg`, `.png`, or `.webp` format.
2. Run the script:
   ```bash
   python main.py
   ```
3. The live video feed will start. The system will detect objects and recognize faces in the frame. Press `q` to quit the application.

## File Structure
```
project_root/
|— main.py             # Main script
|— known_faces/        # Directory for storing images of known individuals
|— requirements.txt    # List of required Python libraries
```

## How It Works
1. **Face Encoding**: The script pre-encodes all images in the `known_faces` directory and stores their encodings in memory.
2. **YOLOv8 Detection**: The YOLO model detects objects in each frame.
3. **Face Recognition**: If a person is detected, the system extracts the face region, calculates its encoding, and compares it to the known encodings.
4. **Labeling**: Recognized faces are labeled on the video feed.

## Future Updates
### Enhanced Skin Tone Analysis
- Implementing an advanced algorithm to analyze skin color using LAB and HSV color spaces for improved feature encoding.

### Detailed Facial Feature Encoding
- Adding precise extraction and analysis of individual facial features (eyes, nose, lips, jawline) using geometric and texture-based descriptors.

### Facial Structure Mapping
- Developing methods to measure inter-landmark distances for comprehensive representation of facial structure.

### Feature-Based Ethnicity Classification
- Training a robust classifier to predict ethnicity based on encoded facial features and their combinations.

### Improved Landmark Detection
- Integrating Mediapipe Face Mesh or similar tools for enhanced detection of up to 468 facial landmarks.

### Real-Time Feature Processing
- Enabling real-time face detection, feature extraction, and ethnicity prediction using optimized YOLOv8 and FaceNet pipelines.

### Bias Mitigation and Dataset Diversity
- Expanding training datasets to ensure balanced representation across different ethnic groups and reduce prediction bias.

### Integration of Pre-Trained Embeddings
- Utilizing FaceNet embeddings to complement custom features for more accurate and robust ethnicity predictions.

## License
This project is open-source and available under the [MIT License](LICENSE).

## Acknowledgments
- **YOLOv8** by [Ultralytics](https://github.com/ultralytics/ultralytics)
- **Face Recognition** library by [Adam Geitgey](https://github.com/ageitgey/face_recognition)

