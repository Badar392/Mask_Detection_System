import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import streamlit as st
st.write("1: streamlit ok")

import numpy as np
st.write(f"2: numpy ok {np.__version__}")

import cv2
cv2.setNumThreads(1)
st.write(f"3: cv2 ok {cv2.__version__}")

from tensorflow.keras.models import load_model
st.write("4: tensorflow.keras import ok")

model = load_model("face_mask_detector.keras")
st.write("5: model loaded ok")

model.predict(np.zeros((1, 160, 160, 3), dtype="float32"), verbose=0)
st.write("6: prediction ran ok")
