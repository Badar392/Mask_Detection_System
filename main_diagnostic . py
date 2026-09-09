import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import streamlit as st
import numpy as np
import cv2
cv2.setNumThreads(1)

st.write("0: base imports ok")

st.set_page_config(
    page_title="MaskGuard AI",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.write("1: page config ok")

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 20% 10%, rgba(0, 200, 200, 0.08), transparent 30%),
            radial-gradient(circle at 80% 20%, rgba(0, 120, 255, 0.07), transparent 30%),
            #071116;
        color: #e8f4f8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.write("2: css ok")

from tensorflow.keras.models import load_model
st.write("3: tensorflow.keras import ok")

IMG_SIZE = (160, 160)
MODEL_PATH = "face_mask_detector.keras"

@st.cache_resource(show_spinner=False)
def load_mask_model():
    model = load_model(MODEL_PATH, compile=False)
    dummy = np.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
    model.predict(dummy, verbose=0)
    return model

model = load_mask_model()
st.write("4: cached model load ok")

@st.cache_resource(show_spinner=False)
def load_face_cascade():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError("Could not load OpenCV Haar Cascade.")
    return cascade

face_cascade = load_face_cascade()
st.write("5: cached cascade load ok")

tab1, tab2 = st.tabs(["Upload", "Webcam"])
st.write("6: tabs created ok")

with tab1:
    st.write("7: inside tab1 ok")

with tab2:
    st.write("8: inside tab2 ok")

st.write("DONE: everything ran successfully")
