# 😷 Face Mask Detection System

A real-time computer vision application that detects whether a person is wearing a face mask using a Convolutional Neural Network (CNN). The system uses a live webcam feed and provides an interactive web interface built with Streamlit.

## 🚀 Live Demo

🔗 https://maskdetectionsystem-8ysvwxcjixzpcfzxuw8let.streamlit.app/

## 🚀 Features

- Face mask detection from browser webcam photos
- Image upload option for mask detection
- CNN-based deep learning model for accurate classification
- Interactive and easy-to-use web interface
- Visual feedback distinguishing **"Mask"** vs **"No Mask"**
- Contrast enhancement, aspect-ratio-preserving model preprocessing, and
  readable in-box labels for uploaded and camera images
- Responsive web application deployed on Streamlit Cloud

## 🛠️ Tech Stack

- **Python** – Core programming language
- **TensorFlow / Keras** – Model building and training
- **OpenCV** – Real-time webcam capture and face detection
- **NumPy** – Numerical operations and data handling
- **Pandas** – Data manipulation and preprocessing
- **Scikit-learn** – Data splitting and evaluation utilities
- **Streamlit** – Interactive web-based user interface
- **Matplotlib** – Data visualization and training metrics
- **Pillow (PIL)** – Image processing and handling

## 📂 Project Structure

```text
Face_Mask_Detection_System/
├── main.py                     # Streamlit application (entry point)
├── face_mask_detector.keras    # Trained MobileNetV2 model (160x160 input)
├── face_detection_yunet_2023mar.onnx # Headless YuNet face detector
├── detect_mask.py              # Optional standalone local script (not used by the Streamlit app)
├── requirements.txt            # Project dependencies
├── runtime.txt                 # Python version pin for Streamlit Cloud
└── README.md                   # Project documentation
```

> The `dataset/` folder used for training is not required to run or deploy
> this app — only `main.py` and `face_mask_detector.keras` need to be present.

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/face-mask-detection.git
cd face-mask-detection
```

### 2. Create a Virtual Environment (Optional)

```bash
python -m venv venv
```

Activate the environment:

**Windows**
```bash
venv\Scripts\activate
```

**Linux/Mac**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Usage

Run the Streamlit application:

```bash
streamlit run main.py
```

This will launch the web interface in your browser, where you can:

- Upload an image for prediction
- Capture a webcam photo for face mask detection
- View prediction results instantly

### Live webcam notes

The webcam tab keeps the camera disabled on initial load. Turn on **Enable
webcam** and use Streamlit's native camera capture to take a frame. The
browser owns camera permission; the captured image is processed in Python and
is not read from a server-side `cv2.VideoCapture(0)` device. Continuous live
browser video requires a JavaScript/WebRTC component and is intentionally not
used here.

### Image pipeline

Both uploaded images and browser-camera captures use the same pipeline:

1. Enhance local contrast with CLAHE and apply gentle unsharp sharpening.
2. Detect faces with the bundled YuNet DNN detector and add a small context margin.
3. Letterbox each face to the model's exact 160x160 input without stretching.
4. Map detections back to the original display image and draw the label inside
   the top edge of each box.

## 🧠 Model Training

The model (`face_mask_detector.keras`) was trained separately in a Google
Colab notebook using transfer learning on MobileNetV2 (160×160 input,
two-stage feature-extraction + fine-tuning). To retrain or reproduce it, see
the training notebook rather than this repository — this repo only contains
the inference-side Streamlit app.

## 📦 Requirements

```text
streamlit==1.56.0
tensorflow-cpu==2.21.0
numpy==2.2.6
pandas==3.0.2
scikit-learn==1.8.0
matplotlib==3.10.9
pillow==12.2.0
opencv-python-headless==4.11.0.86
pyarrow<25.0.0
```

## 📊 Results

The CNN model achieves high accuracy in distinguishing between masked and unmasked faces under varied lighting conditions, face orientations, and backgrounds. The system is suitable for real-world deployment scenarios such as:

- Public safety monitoring
- Educational demonstrations
- Smart surveillance applications
- Health and safety compliance systems
