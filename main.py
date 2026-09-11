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
import base64
from io import BytesIO
from pathlib import Path

import streamlit as st
import cv2
import streamlit.components.v1 as components
try:
    import mediapipe as mp
except ImportError:
    mp = None

cv2.setNumThreads(1)

import numpy as np

from PIL import Image
from PIL import ImageOps
from collections import deque

from tensorflow.keras.models import load_model

_camera_component = components.declare_component(
    "mask_camera",
    path=str((Path(__file__).parent / "camera_component").absolute()),
)


def camera_input_live(key=None):
    """Return browser camera frames as BytesIO images until stopped."""
    value = _camera_component(
        key=key,
        width=704,
        height=530,
        interval=500,
    )
    if not value:
        return None
    return BytesIO(base64.b64decode(value.split(",", 1)[-1]))


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

st.html(
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

    .stImage img {
        width: 100%;
        max-height: 75vh;
        object-fit: contain;
    }

    </style>
    """
    ),
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

LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX


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


@st.cache_resource(show_spinner=False)
def load_face_detector():
    """Use a trained detector that remains reliable when a mask occludes the face."""
    if mp is None or not hasattr(mp, "solutions"):
        return load_face_cascade()
    return mp.solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.35,
    )


# ============================================================
# SHARED IMAGE PREPROCESSING AND INFERENCE
# ============================================================

def enhance_image(image_bgr):
    """Improve local contrast and detail without changing image geometry."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    lightness, chroma_a, chroma_b = cv2.split(lab)
    lightness = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    ).apply(lightness)
    enhanced = cv2.cvtColor(
        cv2.merge((lightness, chroma_a, chroma_b)),
        cv2.COLOR_LAB2BGR,
    )

    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    return cv2.addWeighted(enhanced, 1.12, blurred, -0.12, 0)


def letterbox_image(image, target_size=IMG_SIZE, pad_value=114):
    """Resize an image into target_size while preserving its aspect ratio."""
    target_width, target_height = target_size
    height, width = image.shape[:2]
    scale = min(target_width / width, target_height / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    pad_x = (target_width - resized_width) // 2
    pad_y = (target_height - resized_height) // 2
    letterboxed = np.full(
        (target_height, target_width, image.shape[2]),
        pad_value,
        dtype=image.dtype,
    )
    letterboxed[
        pad_y:pad_y + resized_height,
        pad_x:pad_x + resized_width,
    ] = resized
    return letterboxed, scale, (pad_x, pad_y)


def map_box_from_letterbox(box, scale, padding, image_shape):
    """Map a box from letterboxed coordinates back to the source image."""
    pad_x, pad_y = padding
    x1, y1, x2, y2 = box
    height, width = image_shape[:2]
    return (
        max(0, min(width, round((x1 - pad_x) / scale))),
        max(0, min(height, round((y1 - pad_y) / scale))),
        max(0, min(width, round((x2 - pad_x) / scale))),
        max(0, min(height, round((y2 - pad_y) / scale))),
    )


def preprocess_face(face_bgr):
    enhanced = enhance_image(face_bgr)
    face_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    letterboxed, _, _ = letterbox_image(face_rgb)
    # The Keras model contains its own Rescaling(1 / 127.5, -1) layer.
    return letterboxed.astype(np.float32)


def predict_face(model, face_bgr):
    batch = np.expand_dims(preprocess_face(face_bgr), axis=0)
    prediction = model.predict(batch, verbose=0)
    return float(prediction[0][0])


def detect_faces(image_bgr, face_detector):
    if hasattr(face_detector, "process"):
        height, width = image_bgr.shape[:2]
        original_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = face_detector.process(original_rgb)
        if not result.detections:
            # Enhancement is for the classifier; run the detector on natural
            # pixels first because its model was trained on natural images.
            result = face_detector.process(
                cv2.cvtColor(enhance_image(image_bgr), cv2.COLOR_BGR2RGB)
            )
        if result.detections:
            detections = []
            for detection in result.detections:
                box = detection.location_data.relative_bounding_box
                x = round(box.xmin * width)
                y = round(box.ymin * height)
                box_width = round(box.width * width)
                box_height = round(box.height * height)
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(width, x + box_width)
                y2 = min(height, y + box_height)
                if x2 > x1 and y2 > y1:
                    pad = round(max(x2 - x1, y2 - y1) * 0.20)
                    detections.append({
                        "display_box": (x1, y1, x2, y2),
                        "crop_box": (
                            max(0, x1 - pad),
                            max(0, y1 - pad),
                            min(width, x2 + pad),
                            min(height, y2 + pad),
                        ),
                    })
            if detections:
                return detections
        return []

    # Compatibility fallback for callers/tests that provide an OpenCV cascade.
    enhanced = enhance_image(image_bgr)
    candidates = []
    gray_images = (
        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY),
    )
    for gray in gray_images:
        for scale in (1.0, 1.5):
            detection_image = gray
            if scale != 1.0:
                detection_image = cv2.resize(
                    gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
                )
            detected = face_detector.detectMultiScale(
                detection_image,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(24, 24),
            )
            for x, y, width, height in detected:
                candidates.append((
                    round(x / scale),
                    round(y / scale),
                    round(width / scale),
                    round(height / scale),
                ))

    if not candidates:
        profile_path = cv2.data.haarcascades + "haarcascade_profileface.xml"
        profile = cv2.CascadeClassifier(profile_path)
        if not profile.empty():
            detected = profile.detectMultiScale(
                cv2.resize(gray_images[1], None, fx=1.5, fy=1.5,
                           interpolation=cv2.INTER_CUBIC),
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(24, 24),
            )
            candidates.extend((
                round(x / 1.5), round(y / 1.5),
                round(width / 1.5), round(height / 1.5),
            ) for x, y, width, height in detected)
    if not candidates:
        return []

    # Keep the strongest spatially distinct detections. groupRectangles can
    # discard a valid single detection when the image has only one face.
    candidates = sorted(candidates, key=lambda item: item[2] * item[3], reverse=True)
    faces = []
    for candidate in candidates:
        x, y, width, height = candidate
        overlaps = False
        for kept_x, kept_y, kept_width, kept_height in faces:
            ix1 = max(x, kept_x)
            iy1 = max(y, kept_y)
            ix2 = min(x + width, kept_x + kept_width)
            iy2 = min(y + height, kept_y + kept_height)
            intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = (width * height) + (kept_width * kept_height) - intersection
            if union and intersection / union > 0.35:
                overlaps = True
                break
        if not overlaps:
            faces.append(candidate)

    detections = []
    image_height, image_width = image_bgr.shape[:2]
    for x, y, width, height in faces:
        pad = int(max(width, height) * 0.20)
        detections.append({
            # Draw the detector's actual face rectangle. Padding is only for
            # context supplied to the mask classifier.
            "display_box": (
                max(0, x), max(0, y),
                min(image_width, x + width), min(image_height, y + height),
            ),
            "crop_box": (
                max(0, x - pad), max(0, y - pad),
                min(image_width, x + width + pad),
                min(image_height, y + height + pad),
            ),
        })
    return detections


def detect_largest_face(image_rgb, face_cascade):
    boxes = detect_faces(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), face_cascade)
    return max(
        (detection["display_box"] for detection in boxes),
        key=lambda box: (box[2] - box[0]) * (box[3] - box[1]),
        default=None,
    )


def predict_image(
    image_rgb,
    model,
    face_cascade,
    history=None,
    cached_predictions=None,
    frame_counter=1,
):
    """Run the same shared processing used by browser video frames."""
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return process_frame(
        image_bgr,
        model,
        face_cascade,
        history,
        cached_predictions,
        frame_counter,
    )


def draw_detections(image_rgb, detections):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        label = detection["label"]
        confidence = detection["confidence"]
        if confidence < CONFIDENCE_THR:
            color = (0, 165, 255)
            text = f"Uncertain {confidence * 100:.0f}%"
        else:
            color = color_bgr[label]
            text = f"{labels_dict[label]} {confidence * 100:.0f}%"
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 3)
        (text_width, text_height), baseline = cv2.getTextSize(
            text, LABEL_FONT, 0.65, 2
        )
        label_x = max(x1, min(x2 - text_width, x1))
        label_y = min(y2, y1 + text_height + baseline + 8)
        cv2.rectangle(
            image_bgr,
            (label_x, y1),
            (label_x + text_width + 8, label_y),
            color,
            -1,
        )
        cv2.putText(
            image_bgr,
            text,
            (label_x + 4, label_y - baseline - 4),
            LABEL_FONT,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


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

def _legacy_detect_and_annotate(
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

    gray = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    ).apply(gray)

    # --------------------------------------------------------
    # Detect faces
    # --------------------------------------------------------

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=5,
        minSize=(40, 40)
    )

    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(32, 32)
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

        pad = int(max(w, h) * 0.20)

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
# STREAMING FRAME PIPELINE
# ============================================================

def detect_and_annotate(
    frame,
    model,
    face_cascade,
    history,
    frame_counter,
    cached_predictions,
):
    """Process one browser-camera frame using the shared image pipeline."""
    return process_frame(
        frame,
        model,
        face_cascade,
        history,
        cached_predictions,
        frame_counter,
    )

    # Kept below only as historical reference; all callers use the shared
    # pipeline above so uploads and browser video cannot drift apart.
    processed_frame, scale = resize_for_processing(frame)
    boxes = detect_faces(processed_frame, face_cascade)
    if not boxes:
        history.clear()
        cached_predictions.clear()
        cv2.putText(
            frame,
            "No face detected",
            (20, 40),
            LABEL_FONT,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return frame

    detections = []
    run_inference = frame_counter % INFERENCE_EVERY_N_FRAMES == 0
    valid_keys = set()
    for detection in boxes:
        x1, y1, x2, y2 = detection["crop_box"]
        display_x1, display_y1, display_x2, display_y2 = detection["display_box"]
        key = (round(display_x1 / 40), round(display_y1 / 40),
               round(display_x2 / 40), round(display_y2 / 40))
        valid_keys.add(key)
        if run_inference or key not in cached_predictions:
            crop = processed_frame[y1:y2, x1:x2]
            if crop.size:
                cached_predictions[key] = predict_face(model, crop)
        probability = cached_predictions.get(key, 0.5)
        history.setdefault(key, deque(maxlen=SMOOTH_FRAMES)).append(probability)
        average = float(np.mean(history[key]))
        label = int(average >= THRESHOLD)
        detections.append({
            "box": (
                round(display_x1 / scale),
                round(display_y1 / scale),
                round(display_x2 / scale),
                round(display_y2 / scale),
            ),
            "label": label,
            "confidence": average if label else 1.0 - average,
        })

    for key in list(cached_predictions):
        if key not in valid_keys:
            del cached_predictions[key]
    for key in list(history):
        if key not in valid_keys:
            del history[key]

    annotated_rgb = draw_detections(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), detections)
    annotated = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
    cv2.putText(
        annotated,
        "LIVE",
        (20, 35),
        LABEL_FONT,
        0.75,
        (0, 220, 100),
        2,
        cv2.LINE_AA,
    )
    return annotated


def process_frame(
    frame_bgr,
    model,
    face_cascade,
    history=None,
    cached_predictions=None,
    frame_counter=1,
):
    """Run the complete shared pipeline on a BGR image."""
    history = {} if history is None else history
    cached_predictions = {} if cached_predictions is None else cached_predictions
    processed_frame, scale = resize_for_processing(frame_bgr)
    boxes = detect_faces(processed_frame, face_cascade)
    if not boxes:
        history.clear()
        cached_predictions.clear()
        output = enhance_image(frame_bgr)
        cv2.putText(
            output, "No face detected", (20, 40), LABEL_FONT, 0.8,
            (255, 255, 255), 2, cv2.LINE_AA,
        )
        return output

    detections = []
    valid_keys = set()
    run_inference = frame_counter % 2 == 0 or not cached_predictions
    for detection in boxes:
        x1, y1, x2, y2 = detection["crop_box"]
        display_x1, display_y1, display_x2, display_y2 = detection["display_box"]
        key = (round(display_x1 / 40), round(display_y1 / 40),
               round(display_x2 / 40), round(display_y2 / 40))
        valid_keys.add(key)
        crop = processed_frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        if run_inference or key not in cached_predictions:
            cached_predictions[key] = predict_face(model, crop)
        probability = cached_predictions[key]
        face_history = history.setdefault(key, deque(maxlen=SMOOTH_FRAMES))
        face_history.append(probability)
        average = float(np.mean(face_history))
        label = int(average >= THRESHOLD)
        detections.append({
            "box": (
                round(display_x1 / scale), round(display_y1 / scale),
                round(display_x2 / scale), round(display_y2 / scale),
            ),
            "label": label,
            "confidence": average if label else 1.0 - average,
        })

    for key in list(cached_predictions):
        if key not in valid_keys:
            del cached_predictions[key]
    for key in list(history):
        if key not in valid_keys:
            del history[key]

    enhanced_frame = enhance_image(frame_bgr)
    annotated_rgb = draw_detections(
        cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB), detections
    )
    output = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
    cv2.putText(
        output, "LIVE", (20, 35), LABEL_FONT, 0.75,
        (0, 220, 100), 2, cv2.LINE_AA,
    )
    return output


@st.fragment
def render_webcam(model, face_cascade):
    """Keep camera reruns isolated from the rest of the Streamlit page."""
    camera_image = camera_input_live(key="webcam_capture")
    output_slot = st.empty()
    if camera_image is None:
        return

    image = ImageOps.exif_transpose(
        Image.open(camera_image)
    ).convert("RGB")
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    history = st.session_state.setdefault("webcam_history", {})
    predictions = st.session_state.setdefault("webcam_predictions", {})
    frame_counter = st.session_state.get("webcam_frame_counter", 0) + 1
    st.session_state["webcam_frame_counter"] = frame_counter

    boxes = detect_faces(image_bgr, face_cascade)
    if boxes:
        annotated = process_frame(
            image_bgr,
            model,
            face_cascade,
            history,
            predictions,
            frame_counter,
        )
        st.session_state["webcam_last_frame"] = annotated
        st.session_state["webcam_missed_frames"] = 0
    else:
        missed = st.session_state.get("webcam_missed_frames", 0) + 1
        st.session_state["webcam_missed_frames"] = missed
        annotated = st.session_state.get("webcam_last_frame")
        if annotated is None or missed > 3:
            annotated = process_frame(
                image_bgr,
                model,
                face_cascade,
                history,
                predictions,
                frame_counter,
            )
            cv2.putText(
                annotated,
                "Face not found - move closer and face the camera",
                (20, 70),
                LABEL_FONT,
                0.55,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )

    output_slot.image(
        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
        width="stretch",
    )


# ============================================================
# HERO
# ============================================================

st.html(
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
)


# ============================================================
# LOAD MODEL
# ============================================================

model_loaded = False

with st.spinner("Loading AI model..."):

    try:

        model = load_mask_model()

        face_cascade = load_face_detector()

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

        st.html(
            '<div class="section-label">'
            'Image Detection'
            '</div>',
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

            image = ImageOps.exif_transpose(
                Image.open(uploaded_file)
            ).convert("RGB")

            image_rgb = np.array(
                image
            )

            display_bgr = predict_image(image_rgb, model, face_cascade)
            display_image = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)

            st.image(
                display_image,
                width="stretch"
            )

    # ========================================================
    # WEBCAM TAB
    # ========================================================

    with tab_webcam:

        st.html(
            '<div class="section-label">'
            'Live Detection'
            '</div>',
        )

        webcam_enabled = st.toggle(
            "Enable webcam",
            value=False,
            key="webcam_enabled",
            help="Allow the browser camera to capture a frame for detection.",
        )

        if not webcam_enabled:
            st.session_state.pop("webcam_capture", None)
            st.session_state.pop("webcam_history", None)
            st.session_state.pop("webcam_predictions", None)
            st.session_state.pop("webcam_last_frame", None)
            st.session_state.pop("webcam_missed_frames", None)
            st.session_state.pop("webcam_frame_counter", None)
            st.info(
                "Webcam is off. Enable it above when you are ready "
                "to take a photo."
            )
        else:
            st.info(
                "Live browser frames are processed through the same "
                "enhancement, face detection, letterbox, inference, and "
                "annotation pipeline as uploaded images."
            )
            render_webcam(model, face_cascade)


# ============================================================
# FOOTER
# ============================================================

st.html(
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
)
