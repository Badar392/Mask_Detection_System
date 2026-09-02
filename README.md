# 😷 Face Mask Detection System

A real-time computer vision application that detects whether a person is wearing a face mask using a Convolutional Neural Network (CNN). The system uses a live webcam feed and provides an interactive web interface built with Streamlit.

## 🚀 Live Demo

🔗 https://maskdetectionsystem-8ysvwxcjixzpcfzxuw8let.streamlit.app/

## 🚀 Features

- Real-time face mask detection via webcam
- Image upload option for mask detection
- CNN-based deep learning model for accurate classification
- Interactive and easy-to-use web interface
- Visual feedback distinguishing **"Mask"** vs **"No Mask"**
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
- Start the live webcam feed for real-time face mask detection
- View prediction results instantly

### Live webcam notes

The webcam tab uses [`streamlit-webrtc`](https://github.com/whitphx/streamlit-webrtc)
rather than `cv2.VideoCapture(0)`. This matters once deployed: `cv2.VideoCapture(0)`
only ever reads a camera physically attached to the machine running the app —
on a cloud host like Streamlit Community Cloud there is no such camera.
`streamlit-webrtc` instead streams video from the *visitor's own browser
camera* to the server over WebRTC, which is what makes live detection actually
work for anyone visiting the deployed app, not just on your own laptop.

When you click **START** under the video panel, your browser will prompt for
camera permission — this must be allowed for the stream to begin.

## 🧠 Model Training

The model (`face_mask_detector.keras`) was trained separately in a Google
Colab notebook using transfer learning on MobileNetV2 (160×160 input,
two-stage feature-extraction + fine-tuning). To retrain or reproduce it, see
the training notebook rather than this repository — this repo only contains
the inference-side Streamlit app.

## 📦 Requirements

```text
streamlit==1.56.0
tensorflow==2.21.0
numpy==2.4.4
pandas==3.0.2
scikit-learn==1.8.0
matplotlib==3.10.9
pillow==12.2.0
opencv-python-headless==4.13.0.92
streamlit-webrtc==0.47.1
av==12.3.0
```

## 📊 Results

The CNN model achieves high accuracy in distinguishing between masked and unmasked faces under varied lighting conditions, face orientations, and backgrounds. The system is suitable for real-world deployment scenarios such as:

- Public safety monitoring
- Educational demonstrations
- Smart surveillance applications
- Health and safety compliance systems

