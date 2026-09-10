import os

import faulthandler
faulthandler.enable()

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Limit TensorFlow thread usage on Streamlit Cloud
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import textwrap

import streamlit as st
import cv2

cv2.setNumThreads(1)

import numpy as np

from PIL import Image
from collections import deque

from tensorflow.keras.models import load_model


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MaskGuard AI",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    textwrap.dedent(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at 20% 10%,
                rgba(0, 200, 200, 0.08),
                transparent 30%
            ),
            radial-gradient(
                circle at 80% 20%,
                rgba(0, 120, 255, 0.07),
                transparent 30%
            ),
            #071116;
        color: #e8f4f8;
    }

    .hero {
        padding: 2rem 0 1rem 0;
        text-align: center;
    }

    .hero-badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 20px;
        background: rgba(0, 200, 200, 0.10);
        border: 1px solid rgba(0, 200, 200, 0.25);
        color: #00cccc;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0.6rem 0;
        color: #e8f4f8;
    }

    .hero-title span {
        color: #00cccc;
    }

    .hero-sub {
        color: #6a8fa8;
        font-size: 1rem;
    }

    .section-label {
        color: #00cccc;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }

    .glass-card {
        padding: 1.5rem;
        border-radius: 18px;
        background: rgba(255,255,255,0.025);
        border: 1px solid rgba(255,255,255,0.08);
    }

    .status-card {
        padding: 1rem;
        border-radius: 14px;
        background: rgba(0, 200, 200, 0.05);
        border: 1px solid rgba(0, 200, 200, 0.15);
        margin-top: 1rem;
    }

    </style>
    """
    ),
    unsafe_allow_html=True,
)


# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = (160, 160)

MODEL_PATH = "face_mask_detector.keras"

CLASS_NAMES = [
    "WithMask",
    "WithoutMask",
]

THRESHOLD = 0.50
CONFIDENCE_THR = 0.70

SMOOTH_FRAMES = 5

# ------------------------------------------------------------
# IMPORTANT:
# Do NOT run neural-network inference on every camera frame.
#
# Example:
# 1 = every frame
# 2 = every second frame
# 3 = every third frame
# ------------------------------------------------------------

INFERENCE_EVERY_N_FRAMES = 4

# Maximum width used for processing webcam frames
MAX_PROCESS_WIDTH = 640

labels_dict = {
    0: "Mask",
    1: "No Mask",
}

color_bgr = {
    0: (0, 210, 90),
    1: (0, 60, 230),
}


# ============================================================
# MODEL LOADER
# ============================================================

@st.cache_resource(show_spinner=False)
def load_mask_model():

    model = load_model(
        MODEL_PATH,
        compile=False
    )

    # Warm-up once
    dummy = np.zeros(
        (1, IMG_SIZE[0], IMG_SIZE[1], 3),
        dtype=np.float32
    )

    model.predict(
        dummy,
        verbose=0
    )

    return model


# ============================================================
# FACE CASCADE
# ============================================================

@st.cache_resource(show_spinner=False)
def load_face_cascade():

    cascade_path = cv2.data.haarcascades + (
        "haarcascade_frontalface_default.xml"
    )

    cascade = cv2.CascadeClassifier(cascade_path)

    if cascade.empty():
        raise RuntimeError(
            "Could not load OpenCV Haar Cascade."
        )

    return cascade


# ============================================================
# MODEL PREPROCESSING
# ============================================================

def preprocess_face(face_bgr):

    face_rgb = cv2.cvtColor(
        face_bgr,
        cv2.COLOR_BGR2RGB
    )

    face_rgb = cv2.resize(
        face_rgb,
        IMG_SIZE,
        interpolation=cv2.INTER_AREA
    )

    face_rgb = face_rgb.astype(
        np.float32
    )

    # --------------------------------------------------------
    # IMPORTANT
    #
    # Your previous frontend used raw 0-255 values.
    #
    # Keep this exactly the same ONLY if the Colab model
    # was trained using raw pixel values.
    #
    # If your new Colab model uses:
    #     MobileNetV2 preprocess_input()
    # or
    #     image / 255.0
    #
    # then this section MUST be changed to match training.
    # --------------------------------------------------------

    return face_rgb


# ============================================================
# IMAGE PREDICTION
# ============================================================

def predict_face(model, face_bgr):

    processed = preprocess_face(
        face_bgr
    )

    batch = np.expand_dims(
        processed,
        axis=0
    )

    prediction = model.predict(
        batch,
        verbose=0
    )

    probability_without_mask = float(
        prediction[0][0]
    )

    return probability_without_mask


# ============================================================
# FACE DETECTION
# ============================================================

def detect_largest_face(
    image_rgb,
    face_cascade
):

    gray = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2GRAY
    )

    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(60, 60)
    )

    if len(faces) == 0:
        return None

    x, y, w, h = max(
        faces,
        key=lambda f: f[2] * f[3]
    )

    pad = int(
        max(w, h) * 0.10
    )

    x1 = max(0, x - pad)
    y1 = max(0, y - pad)

    x2 = min(
        image_rgb.shape[1],
        x + w + pad
    )

    y2 = min(
        image_rgb.shape[0],
        y + h + pad
    )

    return (
        x1,
        y1,
        x2,
        y2
    )


# ============================================================
# WEBCAM FRAME RESIZING
# ============================================================

def resize_for_processing(frame):

    height, width = frame.shape[:2]

    if width <= MAX_PROCESS_WIDTH:
        return frame, 1.0

    scale = (
        MAX_PROCESS_WIDTH / float(width)
    )

    new_width = MAX_PROCESS_WIDTH

    new_height = int(
        height * scale
    )

    resized = cv2.resize(
        frame,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    return resized, scale


# ============================================================
# LIVE FACE DETECTION + MASK PREDICTION
# ============================================================

def detect_and_annotate(
    frame,
    model,
    face_cascade,
    history,
    frame_counter,
    cached_predictions
):

    # --------------------------------------------------------
    # Resize webcam frame
    # --------------------------------------------------------

    processed_frame, scale = (
        resize_for_processing(frame)
    )

    gray = cv2.cvtColor(
        processed_frame,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.equalizeHist(gray)

    # --------------------------------------------------------
    # Detect faces
    # --------------------------------------------------------

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(60, 60)
    )

    if len(faces) == 0:

        history.clear()
        cached_predictions.clear()

        cv2.putText(
            frame,
            "No face detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        return frame

    # --------------------------------------------------------
    # Remove duplicate detections
    # --------------------------------------------------------

    try:

        grouped_faces, _ = cv2.groupRectangles(
            faces.tolist() * 2,
            1,
            0.2
        )

        if len(grouped_faces) == 0:
            grouped_faces = faces

    except Exception:

        grouped_faces = faces

    # --------------------------------------------------------
    # Determine whether this frame should run inference
    # --------------------------------------------------------

    run_inference = (
        frame_counter % INFERENCE_EVERY_N_FRAMES == 0
        or not cached_predictions
    )

    current_predictions = []

    # --------------------------------------------------------
    # Process each detected face
    # --------------------------------------------------------

    for face_index, (x, y, w, h) in enumerate(
        grouped_faces
    ):

        pad = int(
            max(w, h) * 0.10
        )

        x1 = max(
            0,
            x - pad
        )

        y1 = max(
            0,
            y - pad
        )

        x2 = min(
            processed_frame.shape[1],
            x + w + pad
        )

        y2 = min(
            processed_frame.shape[0],
            y + h + pad
        )

        crop = processed_frame[
            y1:y2,
            x1:x2
        ]

        if crop.size == 0:
            continue

        # ----------------------------------------------------
        # Create stable face key
        # ----------------------------------------------------

        key = (
            round(x1 / 40),
            round(y1 / 40),
            round(w / 40),
            round(h / 40)
        )

        # ----------------------------------------------------
        # Run model only when necessary
        # ----------------------------------------------------

        if run_inference:

            try:

                probability = predict_face(
                    model,
                    crop
                )

                cached_predictions[key] = (
                    probability
                )

            except Exception:

                probability = (
                    cached_predictions.get(
                        key,
                        0.5
                    )
                )

        else:

            probability = (
                cached_predictions.get(
                    key,
                    0.5
                )
            )

        current_predictions.append(
            (key, probability)
        )

        # ----------------------------------------------------
        # Temporal smoothing
        # ----------------------------------------------------

        if key not in history:

            history[key] = deque(
                maxlen=SMOOTH_FRAMES
            )

        history[key].append(
            probability
        )

        avg_probability = float(
            np.mean(history[key])
        )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        label = int(
            avg_probability >= THRESHOLD
        )

        if label == 1:

            confidence = (
                avg_probability
            )

        else:

            confidence = (
                1.0 - avg_probability
            )

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        if confidence < CONFIDENCE_THR:

            text = (
                f"Uncertain "
                f"{confidence * 100:.0f}%"
            )

            color = (
                0,
                165,
                255
            )

        else:

            text = (
                f"{labels_dict[label]} "
                f"{confidence * 100:.0f}%"
            )

            color = color_bgr[label]

        # ----------------------------------------------------
        # Convert coordinates back to original frame
        # ----------------------------------------------------

        if scale != 1.0:

            inv_scale = 1.0 / scale

            draw_x1 = int(
                x1 * inv_scale
            )

            draw_y1 = int(
                y1 * inv_scale
            )

            draw_x2 = int(
                x2 * inv_scale
            )

            draw_y2 = int(
                y2 * inv_scale
            )

        else:

            draw_x1 = x1
            draw_y1 = y1
            draw_x2 = x2
            draw_y2 = y2

        # ----------------------------------------------------
        # Draw face box
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (draw_x1, draw_y1),
            (draw_x2, draw_y2),
            color,
            2
        )

        # ----------------------------------------------------
        # Draw label background
        # ----------------------------------------------------

        text_y = max(
            draw_y1 - 10,
            25
        )

        cv2.putText(
            frame,
            text,
            (
                draw_x1,
                text_y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # Keep only current face predictions
    # --------------------------------------------------------

    valid_keys = {
        item[0]
        for item in current_predictions
    }

    cached_predictions = {
        k: v
        for k, v in cached_predictions.items()
        if k in valid_keys
    }

    history_keys = list(
        history.keys()
    )

    for key in history_keys:

        if key not in valid_keys:

            del history[key]

    # --------------------------------------------------------
    # Status text
    # --------------------------------------------------------

    cv2.putText(
        frame,
        "LIVE",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 220, 100),
        2,
        cv2.LINE_AA
    )

    return frame


# ============================================================
# HERO
# ============================================================

st.markdown(
    textwrap.dedent(
    """
    <div class="hero">

        <div class="hero-badge">
            AI-Powered Detection
        </div>

        <h1 class="hero-title">
            Mask<span>Guard</span> AI
        </h1>

        <p class="hero-sub">
            Face mask detection using Deep Learning
        </p>

    </div>
    """
    ),
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

model_loaded = False

with st.spinner("Loading AI model..."):

    try:

        model = load_mask_model()

        face_cascade = (
            load_face_cascade()
        )

        model_loaded = True

    except Exception as e:

        st.error(
            "Unable to load the face-mask model."
        )

        st.code(
            str(e)
        )


# ============================================================
# MAIN APPLICATION
# ============================================================

if model_loaded:

    tab_upload, tab_webcam = st.tabs(
        [
            "📤 Upload Image",
            "🎥 Live Webcam"
        ]
    )

    # ========================================================
    # UPLOAD TAB
    # ========================================================

    with tab_upload:

        st.markdown(
            '<div class="section-label">'
            'Image Detection'
            '</div>',
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader(
            "Upload a face image",
            type=[
                "jpg",
                "jpeg",
                "png",
                "webp"
            ],
            label_visibility="collapsed"
        )

        if uploaded_file:

            image = Image.open(
                uploaded_file
            ).convert("RGB")

            image_rgb = np.array(
                image
            )

            result = detect_largest_face(
                image_rgb,
                face_cascade
            )

            display_image = image_rgb.copy()

            if result is not None:

                x1, y1, x2, y2 = result

                face_rgb = image_rgb[
                    y1:y2,
                    x1:x2
                ]

                face_bgr = cv2.cvtColor(
                    face_rgb,
                    cv2.COLOR_RGB2BGR
                )

                probability = predict_face(
                    model,
                    face_bgr
                )

                label = int(
                    probability >= THRESHOLD
                )

                if label == 1:

                    confidence = probability

                else:

                    confidence = (
                        1.0 - probability
                    )

                if confidence < CONFIDENCE_THR:

                    text = (
                        f"Uncertain "
                        f"{confidence * 100:.1f}%"
                    )

                    color = (
                        255,
                        165,
                        0
                    )

                else:

                    text = (
                        f"{labels_dict[label]} "
                        f"{confidence * 100:.1f}%"
                    )

                    color = color_bgr[label]

                cv2.rectangle(
                    display_image,
                    (x1, y1),
                    (x2, y2),
                    color,
                    3
                )

                cv2.putText(
                    display_image,
                    text,
                    (x1, max(y1 - 12, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                    cv2.LINE_AA
                )

            else:

                st.warning(
                    "No face detected. "
                    "Trying the complete image."
                )

                image_bgr = cv2.cvtColor(
                    image_rgb,
                    cv2.COLOR_RGB2BGR
                )

                probability = predict_face(
                    model,
                    image_bgr
                )

                label = int(
                    probability >= THRESHOLD
                )

                confidence = (
                    probability
                    if label == 1
                    else 1.0 - probability
                )

                text = (
                    f"{labels_dict[label]} "
                    f"{confidence * 100:.1f}%"
                )

                color = color_bgr[label]

                cv2.putText(
                    display_image,
                    text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                    cv2.LINE_AA
                )

            st.image(
                display_image,
                width="stretch"
            )

    # ========================================================
    # WEBCAM TAB
    # ========================================================

    with tab_webcam:

        st.markdown(
            '<div class="section-label">'
            'Live Detection'
            '</div>',
            unsafe_allow_html=True
        )

        st.info(
            "Take a photo with your browser camera. The image is "
            "processed after capture and is not stored by the app."
        )

        camera_image = st.camera_input(
            "Open webcam",
            key="webcam_capture"
        )

        if camera_image is not None:

            image = Image.open(
                camera_image
            ).convert("RGB")

            image_rgb = np.array(
                image
            )

            result = detect_largest_face(
                image_rgb,
                face_cascade
            )

            display_image = image_rgb.copy()

            if result is not None:

                x1, y1, x2, y2 = result

                face_rgb = image_rgb[
                    y1:y2,
                    x1:x2
                ]

                probability = predict_face(
                    model,
                    cv2.cvtColor(
                        face_rgb,
                        cv2.COLOR_RGB2BGR
                    )
                )

                label = int(
                    probability >= THRESHOLD
                )

                confidence = (
                    probability
                    if label == 1
                    else 1.0 - probability
                )

                if confidence < CONFIDENCE_THR:

                    text = (
                        f"Uncertain "
                        f"{confidence * 100:.1f}%"
                    )

                    color = (
                        255,
                        165,
                        0
                    )

                else:

                    text = (
                        f"{labels_dict[label]} "
                        f"{confidence * 100:.1f}%"
                    )

                    color = color_bgr[label]

                cv2.rectangle(
                    display_image,
                    (x1, y1),
                    (x2, y2),
                    color,
                    3
                )

                cv2.putText(
                    display_image,
                    text,
                    (x1, max(y1 - 12, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                    cv2.LINE_AA
                )

            else:

                st.warning(
                    "No face detected in the camera image."
                )

            st.image(
                display_image,
                width="stretch"
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    textwrap.dedent(
    """
    <div style="
        text-align:center;
        padding:2rem 0 1rem 0;
        color:#4a6a7e;
        font-size:0.75rem;
    ">
        MaskGuard AI • Face Mask Detection System
        <br>
        TensorFlow • OpenCV • Streamlit
    </div>
    """
    ),
    unsafe_allow_html=True
)
