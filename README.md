# 😷 Face Mask Detection System

A real-time computer vision application that detects whether a person is wearing a face mask using a Convolutional Neural Network (CNN). The system uses a live webcam feed and provides an interactive web interface built with Streamlit.

## 🚀 Features

- Real-time face mask detection via webcam
- CNN-based deep learning model for accurate classification
- Interactive and easy-to-use web interface
- Visual feedback distinguishing "Mask" vs "No Mask"

## 🛠️ Tech Stack

- **Python** – Core programming language
- **TensorFlow / Keras** – Model building and training
- **OpenCV** – Real-time webcam capture and face detection
- **NumPy** – Numerical operations and data handling
- **Streamlit** – Interactive web-based user interface
- **Matplotlib** – Data visualization and training metrics

## 📂 Project Structure

```
face-mask-detection/
├── dataset/                 # Training and testing images
├── model/                   # Saved trained model files
├── train_model.py           # Script to train the CNN model
├── app.py                   # Streamlit application
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## ⚙️ Installation

1. Clone the repository
   ```bash
   git clone https://github.com/your-username/face-mask-detection.git
   cd face-mask-detection
   ```

2. Create a virtual environment (optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

This will launch the web interface in your browser, where you can start the webcam and view real-time face mask detection results.

## 🧠 Model Training

To train the CNN model on your own dataset:
```bash
python train_model.py
```

The trained model will be saved in the `model/` directory for use in the application.

## 📊 Results

The model achieves high accuracy in distinguishing between masked and unmasked faces under varied lighting and angle conditions, making it suitable for real-world deployment scenarios such as public safety monitoring.



This project is licensed under the MIT License.
