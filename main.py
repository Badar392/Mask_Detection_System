import os
import faulthandler
import textwrap
from collections import deque

faulthandler.enable()

# ============================================================
# ENVIRONMENT SETTINGS
# ============================================================

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Keep resource usage low on Streamlit Cloud
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


# ============================================================
# IMPORTS
# ============================================================

import streamlit as st
import cv2
import numpy as np

from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

# Required for REAL-TIME browser webcam
from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    RTCConfiguration,
    VideoProcessorBase,
)


# ============================================================
# OPENCV SETTINGS
# ============================================================

cv2.setNumThreads(1)


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

        </style>
        """
    )
)


# ============================================================
# CONFIGURATION
# ============================================================

# IMPORTANT:
# This must match the input size expected by your trained model.
IMG_SIZE = (160, 160)

MODEL_PATH = "face_mask_detector.keras"

CLASS_NAMES = [
    "WithMask",
    "WithoutMask",
]

# Model probability threshold
THRESHOLD = 0.50

# Minimum confidence shown as a definite prediction
CONFIDENCE_THR = 0.70

# Temporal smoothing for webcam
SMOOTH_FRAMES = 5

# Run neural-network inference every N frames
# 1 = every frame
# 2 = every second frame
# 3 = every third frame
INFERENCE_EVERY_N_FRAMES = 3

# Maximum webcam processing width
MAX_PROCESS_WIDTH = 640


# ============================================================
# LABELS / COLORS
# ============================================================

labels_dict = {
    0: "Mask",
    1: "No Mask",
}

color_bgr = {
    0: (0, 210, 90),       # Green
    1: (0, 60, 230),       # Red
}

UNCERTAIN_COLOR = (0, 165, 255)

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

    # Warm up model once
    dummy = np.zeros(
        (
            1,
            IMG_SIZE[0],
            IMG_SIZE[1],
            3
        ),
        dtype=np.float32
    )

    model.predict(
        dummy,
        verbose=0
    )

    return model


# ============================================================
# HAAR FACE DETECTOR
# ============================================================

@st.cache_resource(show_spinner=False)
def load_face_cascade():

    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    cascade = cv2.CascadeClassifier(cascade_path)

    if cascade.empty():
        raise RuntimeError(
            "Could not load OpenCV Haar Cascade."
        )

    return cascade


# ============================================================
# IMAGE QUALITY ENHANCEMENT
# ============================================================

def enhance_image(image_bgr):
    """
    Improve image quality without changing geometry.

    Pipeline:
        1. Convert BGR -> LAB
        2. CLAHE on luminance channel
        3. Convert LAB -> BGR
        4. Gentle unsharp masking

    This helps with:
        - dark images
        - underexposed faces
        - low contrast
        - slightly blurry frames
    """

    if image_bgr is None or image_bgr.size == 0:
        return image_bgr

    # --------------------------------------------------------
    # CLAHE
    # --------------------------------------------------------

    lab = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2LAB
    )

    lightness, chroma_a, chroma_b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    lightness = clahe.apply(lightness)

    enhanced = cv2.cvtColor(
        cv2.merge(
            (
                lightness,
                chroma_a,
                chroma_b
            )
        ),
        cv2.COLOR_LAB2BGR
    )

    # --------------------------------------------------------
    # GENTLE SHARPENING
    # --------------------------------------------------------

    blurred = cv2.GaussianBlur(
        enhanced,
        (0, 0),
        1.0
    )

    sharpened = cv2.addWeighted(
        enhanced,
        1.12,
        blurred,
        -0.12,
        0
    )

    return sharpened


# ============================================================
# RESIZE + LETTERBOX
# ============================================================

def letterbox_image(
    image,
    target_size=IMG_SIZE,
    pad_value=114
):
    """
    Resize while preserving aspect ratio.

    The image is NEVER stretched.

    Returns:
        letterboxed
        scale
        padding_x
        padding_y
    """

    target_width, target_height = target_size

    height, width = image.shape[:2]

    if width <= 0 or height <= 0:
        raise ValueError(
            "Invalid image dimensions."
        )

    # --------------------------------------------------------
    # Calculate scale
    # --------------------------------------------------------

    scale = min(
        target_width / float(width),
        target_height / float(height)
    )

    resized_width = max(
        1,
        int(round(width * scale))
    )

    resized_height = max(
        1,
        int(round(height * scale))
    )

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    interpolation = (
        cv2.INTER_AREA
        if scale < 1.0
        else cv2.INTER_LINEAR
    )

    resized = cv2.resize(
        image,
        (
            resized_width,
            resized_height
        ),
        interpolation=interpolation
    )

    # --------------------------------------------------------
    # Calculate padding
    # --------------------------------------------------------

    pad_x = (
        target_width - resized_width
    ) // 2

    pad_y = (
        target_height - resized_height
    ) // 2

    # --------------------------------------------------------
    # Create padded image
    # --------------------------------------------------------

    if image.ndim == 3:

        letterboxed = np.full(
            (
                target_height,
                target_width,
                image.shape[2]
            ),
            pad_value,
            dtype=image.dtype
        )

    else:

        letterboxed = np.full(
            (
                target_height,
                target_width
            ),
            pad_value,
            dtype=image.dtype
        )

    # --------------------------------------------------------
    # Put resized image into padded canvas
    # --------------------------------------------------------

    letterboxed[
        pad_y:
        pad_y + resized_height,
        pad_x:
        pad_x + resized_width
    ] = resized

    return (
        letterboxed,
        scale,
        (pad_x, pad_y)
    )


# ============================================================
# MAP LETTERBOX COORDINATES BACK
# ============================================================

def map_box_from_letterbox(
    box,
    scale,
    padding,
    image_shape
):
    """
    Convert coordinates from letterboxed image
    back to original image coordinates.
    """

    pad_x, pad_y = padding

    x1, y1, x2, y2 = box

    image_height, image_width = (
        image_shape[:2]
    )

    x1 = (x1 - pad_x) / scale
    y1 = (y1 - pad_y) / scale
    x2 = (x2 - pad_x) / scale
    y2 = (y2 - pad_y) / scale

    x1 = max(
        0,
        min(
            image_width - 1,
            int(round(x1))
        )
    )

    y1 = max(
        0,
        min(
            image_height - 1,
            int(round(y1))
        )
    )

    x2 = max(
        0,
        min(
            image_width - 1,
            int(round(x2))
        )
    )

    y2 = max(
        0,
        min(
            image_height - 1,
            int(round(y2))
        )
    )

    return (
        x1,
        y1,
        x2,
        y2
    )


# ============================================================
# FACE PREPROCESSING
# ============================================================

def preprocess_face(face_bgr):
    """
    EXACT preprocessing used before model inference.

    Pipeline:

        Original face
             ↓
        CLAHE + sharpening
             ↓
        BGR -> RGB
             ↓
        Aspect-ratio-preserving resize
             ↓
        160 x 160
             ↓
        float32

    NOTE:
    Your current model contains its own
    Rescaling(1 / 127.5, -1) layer, so we DO NOT
    normalize pixels here.
    """

    if face_bgr is None or face_bgr.size == 0:
        raise ValueError(
            "Empty face crop."
        )

    # Same enhancement used for uploaded
    # images AND webcam frames.
    enhanced = enhance_image(
        face_bgr
    )

    # Convert to RGB because Keras image
    # input is RGB.
    face_rgb = cv2.cvtColor(
        enhanced,
        cv2.COLOR_BGR2RGB
    )

    # Preserve aspect ratio.
    letterboxed, _, _ = letterbox_image(
        face_rgb,
        target_size=IMG_SIZE,
        pad_value=114
    )

    return letterboxed.astype(
        np.float32
    )


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_face(
    model,
    face_bgr
):
    """
    Run mask/no-mask classification
    on one detected face.
    """

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

    # Model output:
    # probability of class 1 = WithoutMask
    probability = float(
        prediction[0][0]
    )

    probability = np.clip(
        probability,
        0.0,
        1.0
    )

    return probability


# ============================================================
# FACE DETECTION
# ============================================================

def detect_faces(
    image_bgr,
    face_cascade
):
    """
    Detect faces using the enhanced image.

    Returns boxes in ORIGINAL image coordinates.
    """

    if image_bgr is None or image_bgr.size == 0:
        return []

    # --------------------------------------------------------
    # Enhance image BEFORE face detection
    # --------------------------------------------------------

    enhanced = enhance_image(
        image_bgr
    )

    gray = cv2.cvtColor(
        enhanced,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # First detection pass
    # --------------------------------------------------------

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=5,
        minSize=(40, 40)
    )

    # --------------------------------------------------------
    # Second, more tolerant pass
    # --------------------------------------------------------

    if len(faces) == 0:

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(32, 32)
        )

    if len(faces) == 0:
        return []

    # --------------------------------------------------------
    # Remove duplicate detections
    # --------------------------------------------------------

    try:

        grouped_faces, _ = cv2.groupRectangles(
            faces.tolist() * 2,
            1,
            0.2
        )

        if len(grouped_faces) > 0:
            faces = grouped_faces

    except cv2.error:
        pass

    # --------------------------------------------------------
    # Convert face rectangles to boxes
    # --------------------------------------------------------

    boxes = []

    image_height, image_width = (
        image_bgr.shape[:2]
    )

    for x, y, width, height in faces:

        # Add context around face.
        # This is useful because mask classification
        # needs to see the mouth/nose region.
        pad = int(
            max(width, height) * 0.20
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
            image_width,
            x + width + pad
        )

        y2 = min(
            image_height,
            y + height + pad
        )

        if x2 > x1 and y2 > y1:

            boxes.append(
                (
                    x1,
                    y1,
                    x2,
                    y2
                )
            )

    return boxes


# ============================================================
# STATIC IMAGE PREDICTION
# ============================================================

def predict_image(
    image_rgb,
    model,
    face_cascade
):
    """
    Complete static-image pipeline.

    Original image
        ↓
    Enhanced face detection
        ↓
    Face crop
        ↓
    Enhancement again
        ↓
    Letterbox
        ↓
    Model
        ↓
    Prediction
    """

    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR
    )

    boxes = detect_faces(
        image_bgr,
        face_cascade
    )

    detections = []

    for (
        x1,
        y1,
        x2,
        y2
    ) in boxes:

        face = image_bgr[
            y1:y2,
            x1:x2
        ]

        if face.size == 0:
            continue

        try:

            probability = predict_face(
                model,
                face
            )

        except Exception:
            continue

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        label = int(
            probability >= THRESHOLD
        )

        if label == 1:

            confidence = probability

        else:

            confidence = (
                1.0 - probability
            )

        detections.append(
            {
                "box": (
                    x1,
                    y1,
                    x2,
                    y2
                ),
                "label": label,
                "confidence": confidence,
            }
        )

    return detections


# ============================================================
# SAFE LABEL DRAWING
# ============================================================

def draw_label(
    image_bgr,
    text,
    x1,
    y1,
    x2,
    y2,
    color
):
    """
    Draw a readable label ABOVE the face box.

    If there isn't enough room above the box,
    automatically move the label INSIDE the box.

    This prevents:
        - text clipping
        - text behind rectangle
        - unreadable labels
        - labels outside the image
    """

    font_scale = 0.60
    thickness = 2

    (
        text_width,
        text_height
    ), baseline = cv2.getTextSize(
        text,
        LABEL_FONT,
        font_scale,
        thickness
    )

    padding_x = 7
    padding_y = 5

    label_width = (
        text_width
        + padding_x * 2
    )

    label_height = (
        text_height
        + baseline
        + padding_y * 2
    )

    image_height, image_width = (
        image_bgr.shape[:2]
    )

    # --------------------------------------------------------
    # Preferred location:
    # ABOVE the bounding box
    # --------------------------------------------------------

    label_x = x1

    label_y = (
        y1 - label_height
    )

    # --------------------------------------------------------
    # Horizontal correction
    # --------------------------------------------------------

    label_x = max(
        0,
        min(
            label_x,
            image_width - label_width
        )
    )

    # --------------------------------------------------------
    # If label doesn't fit ABOVE the box,
    # put it INSIDE the top of the box.
    # --------------------------------------------------------

    if label_y < 0:

        label_y = y1

        # If box is extremely small,
        # make sure label still remains visible.
        if (
            label_y + label_height
            > y2
        ):

            label_y = max(
                0,
                y2 - label_height
            )

    # --------------------------------------------------------
    # Final vertical safety
    # --------------------------------------------------------

    label_y = max(
        0,
        min(
            label_y,
            image_height - label_height
        )
    )

    # --------------------------------------------------------
    # Draw filled background
    # --------------------------------------------------------

    cv2.rectangle(
        image_bgr,
        (
            label_x,
            label_y
        ),
        (
            label_x + label_width,
            label_y + label_height
        ),
        color,
        -1
    )

    # --------------------------------------------------------
    # Draw text
    # --------------------------------------------------------

    text_x = (
        label_x + padding_x
    )

    text_y = (
        label_y
        + padding_y
        + text_height
    )

    cv2.putText(
        image_bgr,
        text,
        (
            text_x,
            text_y
        ),
        LABEL_FONT,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# DRAW DETECTIONS
# ============================================================

def draw_detections(
    image_rgb,
    detections
):
    """
    Draw bounding boxes and labels.
    """

    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR
    )

    for detection in detections:

        x1, y1, x2, y2 = (
            detection["box"]
        )

        label = detection["label"]

        confidence = (
            detection["confidence"]
        )

        # ----------------------------------------------------
        # Low-confidence prediction
        # ----------------------------------------------------

        if confidence < CONFIDENCE_THR:

            text = (
                f"Uncertain "
                f"{confidence * 100:.0f}%"
            )

            color = UNCERTAIN_COLOR

        else:

            text = (
                f"{labels_dict[label]} "
                f"{confidence * 100:.0f}%"
            )

            color = color_bgr[label]

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        cv2.rectangle(
            image_bgr,
            (
                x1,
                y1
            ),
            (
                x2,
                y2
            ),
            color,
            3
        )

        # ----------------------------------------------------
        # SAFE LABEL
        # ----------------------------------------------------

        draw_label(
            image_bgr,
            text,
            x1,
            y1,
            x2,
            y2,
            color
        )

    return cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )


# ============================================================
# WEBCAM FRAME RESIZING
# ============================================================

def resize_for_processing(
    frame
):
    """
    Reduce very large webcam frames while
    preserving aspect ratio.

    IMPORTANT:
    This is NOT model resizing.

    It only reduces the webcam processing
    resolution to improve performance.
    """

    height, width = frame.shape[:2]

    if width <= MAX_PROCESS_WIDTH:

        return frame, 1.0

    scale = (
        MAX_PROCESS_WIDTH
        / float(width)
    )

    new_width = (
        MAX_PROCESS_WIDTH
    )

    new_height = max(
        1,
        int(round(height * scale))
    )

    resized = cv2.resize(
        frame,
        (
            new_width,
            new_height
        ),
        interpolation=cv2.INTER_AREA
    )

    return (
        resized,
        scale
    )


# ============================================================
# WEBCAM VIDEO PROCESSOR
# ============================================================

class MaskDetectionProcessor(
    VideoProcessorBase
):
    """
    Real-time webcam processor.

    Every browser frame:

        Camera frame
             ↓
        Resize
             ↓
        CLAHE
             ↓
        Face detection
             ↓
        Face crop
             ↓
        CLAHE + sharpening
             ↓
        Letterbox
             ↓
        Model
             ↓
        Temporal smoothing
             ↓
        Bounding box + label
    """

    def __init__(
        self,
        model,
        face_cascade
    ):

        self.model = model
        self.face_cascade = face_cascade

        self.frame_counter = 0

        self.history = {}

        self.cached_predictions = {}

    # --------------------------------------------------------
    # Generate stable face key
    # --------------------------------------------------------

    @staticmethod
    def make_face_key(
        x1,
        y1,
        x2,
        y2
    ):

        return (
            round(x1 / 40),
            round(y1 / 40),
            round(x2 / 40),
            round(y2 / 40),
        )

    # --------------------------------------------------------
    # Process one webcam frame
    # --------------------------------------------------------

    def recv(
        self,
        frame
    ):

        self.frame_counter += 1

        # ----------------------------------------------------
        # Convert WebRTC frame -> BGR
        # ----------------------------------------------------

        original_bgr = frame.to_ndarray(
            format="bgr24"
        )

        if (
            original_bgr is None
            or original_bgr.size == 0
        ):

            return frame

        # ----------------------------------------------------
        # Resize for performance
        # ----------------------------------------------------

        processed_frame, scale = (
            resize_for_processing(
                original_bgr
            )
        )

        # ----------------------------------------------------
        # Detect faces
        #
        # detect_faces() itself performs
        # CLAHE + face detection.
        # ----------------------------------------------------

        boxes = detect_faces(
            processed_frame,
            self.face_cascade
        )

        # ----------------------------------------------------
        # No faces
        # ----------------------------------------------------

        if not boxes:

            self.history.clear()

            self.cached_predictions.clear()

            cv2.putText(
                original_bgr,
                "No face detected",
                (
                    20,
                    40
                ),
                LABEL_FONT,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            return self._make_frame(
                original_bgr
            )

        # ----------------------------------------------------
        # Determine inference timing
        # ----------------------------------------------------

        run_inference = (
            self.frame_counter
            % INFERENCE_EVERY_N_FRAMES
            == 0
        )

        detections = []

        valid_keys = set()

        # ----------------------------------------------------
        # Process every detected face
        # ----------------------------------------------------

        for (
            x1,
            y1,
            x2,
            y2
        ) in boxes:

            key = self.make_face_key(
                x1,
                y1,
                x2,
                y2
            )

            valid_keys.add(key)

            # ------------------------------------------------
            # Crop face
            # ------------------------------------------------

            crop = processed_frame[
                y1:y2,
                x1:x2
            ]

            if crop.size == 0:
                continue

            # ------------------------------------------------
            # Run model
            #
            # Always infer if we don't have
            # a cached prediction.
            # ------------------------------------------------

            if (
                run_inference
                or key
                not in self.cached_predictions
            ):

                try:

                    probability = (
                        predict_face(
                            self.model,
                            crop
                        )
                    )

                    self.cached_predictions[
                        key
                    ] = probability

                except Exception:

                    probability = (
                        self.cached_predictions.get(
                            key,
                            0.5
                        )
                    )

            else:

                probability = (
                    self.cached_predictions.get(
                        key,
                        0.5
                    )
                )

            # ------------------------------------------------
            # Temporal smoothing
            # ------------------------------------------------

            if key not in self.history:

                self.history[key] = deque(
                    maxlen=SMOOTH_FRAMES
                )

            self.history[key].append(
                probability
            )

            average_probability = (
                float(
                    np.mean(
                        self.history[key]
                    )
                )
            )

            # ------------------------------------------------
            # Classification
            # ------------------------------------------------

            label = int(
                average_probability
                >= THRESHOLD
            )

            if label == 1:

                confidence = (
                    average_probability
                )

            else:

                confidence = (
                    1.0
                    - average_probability
                )

            # ------------------------------------------------
            # Convert processed coordinates
            # back to ORIGINAL webcam frame.
            # ------------------------------------------------

            if scale != 1.0:

                draw_x1 = int(
                    round(x1 / scale)
                )

                draw_y1 = int(
                    round(y1 / scale)
                )

                draw_x2 = int(
                    round(x2 / scale)
                )

                draw_y2 = int(
                    round(y2 / scale)
                )

            else:

                draw_x1 = x1
                draw_y1 = y1
                draw_x2 = x2
                draw_y2 = y2

            # ------------------------------------------------
            # Clamp coordinates
            # ------------------------------------------------

            frame_height, frame_width = (
                original_bgr.shape[:2]
            )

            draw_x1 = max(
                0,
                min(
                    frame_width - 1,
                    draw_x1
                )
            )

            draw_y1 = max(
                0,
                min(
                    frame_height - 1,
                    draw_y1
                )
            )

            draw_x2 = max(
                0,
                min(
                    frame_width - 1,
                    draw_x2
                )
            )

            draw_y2 = max(
                0,
                min(
                    frame_height - 1,
                    draw_y2
                )
            )

            detections.append(
                {
                    "box": (
                        draw_x1,
                        draw_y1,
                        draw_x2,
                        draw_y2,
                    ),
                    "label": label,
                    "confidence": confidence,
                }
            )

        # ----------------------------------------------------
        # Remove stale predictions
        # ----------------------------------------------------

        for key in list(
            self.cached_predictions.keys()
        ):

            if key not in valid_keys:

                del self.cached_predictions[
                    key
                ]

        for key in list(
            self.history.keys()
        ):

            if key not in valid_keys:

                del self.history[key]

        # ----------------------------------------------------
        # Draw detections directly on BGR frame
        # ----------------------------------------------------

        for detection in detections:

            x1, y1, x2, y2 = (
                detection["box"]
            )

            label = detection["label"]

            confidence = (
                detection["confidence"]
            )

            if confidence < CONFIDENCE_THR:

                text = (
                    f"Uncertain "
                    f"{confidence * 100:.0f}%"
                )

                color = UNCERTAIN_COLOR

            else:

                text = (
                    f"{labels_dict[label]} "
                    f"{confidence * 100:.0f}%"
                )

                color = color_bgr[label]

            # Bounding box
            cv2.rectangle(
                original_bgr,
                (
                    x1,
                    y1
                ),
                (
                    x2,
                    y2
                ),
                color,
                3
            )

            # Safe label
            draw_label(
                original_bgr,
                text,
                x1,
                y1,
                x2,
                y2,
                color
            )

        # ----------------------------------------------------
        # LIVE indicator
        # ----------------------------------------------------

        cv2.putText(
            original_bgr,
            "LIVE",
            (
                20,
                35
            ),
            LABEL_FONT,
            0.75,
            (0, 220, 100),
            2,
            cv2.LINE_AA
        )

        return self._make_frame(
            original_bgr
        )

    # --------------------------------------------------------
    # Convert processed BGR -> WebRTC frame
    # --------------------------------------------------------

    @staticmethod
    def _make_frame(
        image_bgr
    ):

        from av import VideoFrame

        return VideoFrame.from_ndarray(
            image_bgr,
            format="bgr24"
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
    )
)


# ============================================================
# LOAD MODEL
# ============================================================

model_loaded = False

with st.spinner(
    "Loading AI model..."
):

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

    tab_upload, tab_webcam = (
        st.tabs(
            [
                "📤 Upload Image",
                "🎥 Live Webcam"
            ]
        )
    )

    # ========================================================
    # UPLOAD IMAGE
    # ========================================================

    with tab_upload:

        st.html(
            '<div class="section-label">'
            'Image Detection'
            '</div>'
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

            try:

                # ------------------------------------------------
                # Correct EXIF orientation
                # ------------------------------------------------

                image = ImageOps.exif_transpose(
                    Image.open(
                        uploaded_file
                    )
                ).convert("RGB")

                image_rgb = np.array(
                    image
                )

                # ------------------------------------------------
                # Run complete pipeline
                # ------------------------------------------------

                detections = (
                    predict_image(
                        image_rgb,
                        model,
                        face_cascade
                    )
                )

                # ------------------------------------------------
                # Draw results
                # ------------------------------------------------

                display_image = (
                    draw_detections(
                        image_rgb,
                        detections
                    )
                )

                if not detections:

                    st.warning(
                        "No face detected. "
                        "Try a clearer image with "
                        "the face looking toward the camera."
                    )

                else:

                    st.success(
                        f"{len(detections)} "
                        f"face(s) detected."
                    )

                st.image(
                    display_image,
                    width="stretch"
                )

            except Exception as e:

                st.error(
                    "Could not process the image."
                )

                st.exception(e)

    # ========================================================
    # REAL-TIME WEBCAM
    # ========================================================

    with tab_webcam:

        st.html(
            '<div class="section-label">'
            'Live Detection'
            '</div>'
        )

        st.info(
            "Allow camera access in your browser. "
            "Detection runs continuously on the live video."
        )

        # ----------------------------------------------------
        # WebRTC configuration
        # ----------------------------------------------------

        RTC_CONFIGURATION = RTCConfiguration(
            {
                "iceServers": [
                    {
                        "urls": [
                            "stun:stun.l.google.com:19302"
                        ]
                    }
                ]
            }
        )

        # ----------------------------------------------------
        # REAL-TIME STREAM
        # ----------------------------------------------------

        webrtc_ctx = webrtc_streamer(
            key="maskguard-live",

            mode=WebRtcMode.SENDRECV,

            rtc_configuration=RTC_CONFIGURATION,

            media_stream_constraints={
                "video": True,
                "audio": False,
            },

            video_processor_factory=lambda:
                MaskDetectionProcessor(
                    model,
                    face_cascade
                ),

            async_processing=True,
        )

        if webrtc_ctx.state.playing:

            st.success(
                "🟢 Camera is running — "
                "live face detection is active."
            )

        else:

            st.info(
                "Click START and allow your browser "
                "to access the camera."
            )


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
            TensorFlow • OpenCV • Streamlit • WebRTC
        </div>
        """
    )
)
