import os
import threading

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from streamlit_webrtc import WebRtcMode, VideoProcessorBase, RTCConfiguration, webrtc_streamer

try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass

st.set_page_config(
    page_title="MaskGuard AI",
    page_icon="😷",
    layout="wide",
)

IMG_SIZE = (160, 160)
MODEL_PATH = "face_mask_detector.keras"

THRESHOLD = 0.50
MAX_PROCESS_WIDTH = 640
INFERENCE_EVERY_N_FRAMES = 4

CLASS_NAMES = ["WithMask", "WithoutMask"]
DISPLAY_LABELS = {0: "Mask", 1: "No Mask"}


st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #6b7280;
        margin-bottom: 1.2rem;
    }
    </style>

    <div class="main-title">MaskGuard AI</div>
    <div class="sub-title">
        Real-time face mask detection using CNN, OpenCV and Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' was not found. "
            "Place the trained .keras model in the repository root."
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    # Warm up the model once.
    try:
        model(
            np.zeros(
                (1, IMG_SIZE[1], IMG_SIZE[0], 3),
                dtype=np.float32,
            ),
            training=False,
        )
    except Exception:
        pass

    return model


@st.cache_resource
def load_face_cascade():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)

    if cascade.empty():
        raise RuntimeError(
            "OpenCV Haar cascade could not be loaded."
        )

    return cascade


def enhance_image(image_bgr):
    """Improve low-light and low-contrast images without over-processing."""
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )
    l_channel = clahe.apply(l_channel)

    enhanced = cv2.cvtColor(
        cv2.merge((l_channel, a_channel, b_channel)),
        cv2.COLOR_LAB2BGR,
    )

    blurred = cv2.GaussianBlur(
        enhanced,
        (0, 0),
        1.0,
    )

    enhanced = cv2.addWeighted(
        enhanced,
        1.12,
        blurred,
        -0.12,
        0,
    )

    return enhanced


def letterbox_image(
    image,
    target_size=IMG_SIZE,
    pad_value=114,
):
    """Resize while preserving aspect ratio and pad to target size."""
    target_w, target_h = target_size
    h, w = image.shape[:2]

    if h == 0 or w == 0:
        raise ValueError("Invalid image size.")

    scale = min(
        target_w / w,
        target_h / h,
    )

    new_w = max(
        1,
        int(round(w * scale)),
    )
    new_h = max(
        1,
        int(round(h * scale)),
    )

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.full(
        (target_h, target_w, 3),
        pad_value,
        dtype=np.uint8,
    )

    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2

    canvas[
        pad_y:pad_y + new_h,
        pad_x:pad_x + new_w,
    ] = resized

    return canvas


def preprocess_face(face_bgr):
    """Prepare a detected face for the trained CNN."""
    face_bgr = enhance_image(face_bgr)
    face_rgb = cv2.cvtColor(
        face_bgr,
        cv2.COLOR_BGR2RGB,
    )

    face_input = letterbox_image(
        face_rgb,
        IMG_SIZE,
    )

    # Do not divide by 255 here.
    # The trained model already contains its Rescaling layer.
    face_input = face_input.astype(np.float32)

    return np.expand_dims(
        face_input,
        axis=0,
    )


def predict_face(model, face_bgr):
    """Return class id, confidence and raw class-1 probability."""
    input_tensor = preprocess_face(face_bgr)

    prediction = model(
        input_tensor,
        training=False,
    ).numpy()

    probability_no_mask = float(
        np.asarray(prediction).reshape(-1)[0]
    )

    probability_no_mask = float(
        np.clip(
            probability_no_mask,
            0.0,
            1.0,
        )
    )

    label_id = int(
        probability_no_mask >= THRESHOLD
    )

    if label_id == 1:
        confidence = probability_no_mask
    else:
        confidence = 1.0 - probability_no_mask

    return (
        label_id,
        confidence,
        probability_no_mask,
    )


def detect_faces(image_bgr, face_cascade):
    """Detect faces after gentle image enhancement."""
    enhanced = enhance_image(image_bgr)

    gray = cv2.cvtColor(
        enhanced,
        cv2.COLOR_BGR2GRAY,
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=5,
        minSize=(40, 40),
    )

    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(32, 32),
        )

    results = []
    frame_h, frame_w = image_bgr.shape[:2]

    for x, y, fw, fh in faces:
        pad_x = int(fw * 0.20)
        pad_y = int(fh * 0.20)

        x1 = max(
            0,
            x - pad_x,
        )
        y1 = max(
            0,
            y - pad_y,
        )
        x2 = min(
            frame_w - 1,
            x + fw + pad_x,
        )
        y2 = min(
            frame_h - 1,
            y + fh + pad_y,
        )

        if x2 > x1 and y2 > y1:
            results.append(
                (x1, y1, x2, y2)
            )

    return results


def draw_label(
    image,
    text,
    box,
    confidence,
):
    """Draw a readable label above the face or inside it."""
    x1, y1, x2, y2 = box

    text = (
        f"{text} "
        f"{confidence * 100:.1f}%"
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.58
    thickness = 2
    margin = 5

    (text_w, text_h), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness,
    )

    label_w = text_w + margin * 2
    label_h = (
        text_h
        + baseline
        + margin * 2
    )

    frame_h, frame_w = image.shape[:2]

    label_x = max(
        0,
        min(
            x1,
            frame_w - label_w,
        ),
    )

    # Prefer placing the label above the face.
    if y1 - label_h >= 0:
        label_y1 = y1 - label_h
    else:
        # If the face touches the top edge,
        # place the label inside the box.
        label_y1 = y1

        if label_y1 + label_h > frame_h:
            label_y1 = max(
                0,
                y2 - label_h,
            )

    label_y2 = min(
        frame_h - 1,
        label_y1 + label_h,
    )

    text_y = min(
        frame_h - 1,
        label_y1 + margin + text_h,
    )

    if text.startswith("No Mask"):
        color = (0, 0, 255)
    else:
        color = (0, 180, 0)

    cv2.rectangle(
        image,
        (
            label_x,
            label_y1,
        ),
        (
            min(
                frame_w - 1,
                label_x + label_w,
            ),
            label_y2,
        ),
        color,
        -1,
    )

    cv2.putText(
        image,
        text,
        (
            label_x + margin,
            text_y,
        ),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_detections(
    image,
    detections,
):
    """Draw face boxes and labels."""
    output = image.copy()

    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        label = detection["label"]
        confidence = detection["confidence"]

        if label == "No Mask":
            color = (0, 0, 255)
        else:
            color = (0, 180, 0)

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        draw_label(
            output,
            label,
            (x1, y1, x2, y2),
            confidence,
        )

    return output


def resize_for_processing(
    image_bgr,
    max_width=MAX_PROCESS_WIDTH,
):
    """Reduce large webcam frames for faster CPU inference."""
    frame_h, frame_w = image_bgr.shape[:2]

    if frame_w <= max_width:
        return image_bgr, 1.0

    scale = max_width / float(frame_w)
    new_h = max(
        1,
        int(round(frame_h * scale)),
    )

    resized = cv2.resize(
        image_bgr,
        (max_width, new_h),
        interpolation=cv2.INTER_AREA,
    )

    return resized, scale


def predict_image(
    image_bgr,
    model,
    face_cascade,
):
    """Run the complete detection pipeline on one image."""
    faces = detect_faces(
        image_bgr,
        face_cascade,
    )

    detections = []

    for box in faces:
        x1, y1, x2, y2 = box

        face = image_bgr[
            y1:y2,
            x1:x2,
        ]

        if face.size == 0:
            continue

        (
            label_id,
            confidence,
            _,
        ) = predict_face(
            model,
            face,
        )

        detections.append(
            {
                "box": box,
                "label": DISPLAY_LABELS[label_id],
                "confidence": confidence,
            }
        )

    return detections


class MaskDetectionProcessor(VideoProcessorBase):
    """Continuous WebRTC video processor."""

    def __init__(self):
        self.model = load_model()
        self.face_cascade = load_face_cascade()
        self.frame_count = 0
        self.cached_detections = []
        self.lock = threading.Lock()

    def recv(self, frame):
        image = frame.to_ndarray(
            format="bgr24"
        )

        self.frame_count += 1

        processed, scale = resize_for_processing(
            image
        )

        should_infer = (
            self.frame_count
            % INFERENCE_EVERY_N_FRAMES
            == 0
            or len(self.cached_detections) == 0
        )

        if should_infer:
            detections = predict_image(
                processed,
                self.model,
                self.face_cascade,
            )

            with self.lock:
                self.cached_detections = detections
        else:
            with self.lock:
                detections = list(
                    self.cached_detections
                )

        output = draw_detections(
            processed,
            detections,
        )

        if scale != 1.0:
            output = cv2.resize(
                output,
                (
                    image.shape[1],
                    image.shape[0],
                ),
                interpolation=cv2.INTER_LINEAR,
            )

        cv2.putText(
            output,
            "LIVE",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )

        if len(detections) == 0:
            cv2.putText(
                output,
                "No face detected",
                (15, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return frame.from_ndarray(
            output,
            format="bgr24",
        )


def render_webcam():
    st.subheader("Live Webcam Detection")

    st.info(
        "Click START, allow camera access, and keep your face visible. "
        "The detector processes the webcam stream continuously."
    )

    rtc_configuration = RTCConfiguration(
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

    webrtc_streamer(
        key="maskguard-live-camera",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        video_processor_factory=MaskDetectionProcessor,
        async_processing=True,
    )


def render_upload(
    model,
    face_cascade,
):
    st.subheader("Image Detection")

    uploaded_file = st.file_uploader(
        "Upload a face image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
    )

    if uploaded_file is None:
        st.info(
            "Upload an image to start detection."
        )
        return

    pil_image = Image.open(
        uploaded_file
    )

    pil_image = ImageOps.exif_transpose(
        pil_image
    ).convert("RGB")

    image_bgr = cv2.cvtColor(
        np.array(pil_image),
        cv2.COLOR_RGB2BGR,
    )

    detections = predict_image(
        image_bgr,
        model,
        face_cascade,
    )

    result = draw_detections(
        image_bgr,
        detections,
    )

    result_rgb = cv2.cvtColor(
        result,
        cv2.COLOR_BGR2RGB,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            pil_image,
            caption="Original image",
            width="stretch",
        )

    with col2:
        st.image(
            result_rgb,
            caption="Detection result",
            width="stretch",
        )

    if detections:
        for index, item in enumerate(
            detections,
            start=1,
        ):
            st.write(
                f"Face {index}: "
                f"**{item['label']}** "
                f"({item['confidence'] * 100:.1f}%)"
            )
    else:
        st.warning(
            "No face was detected. Try a clearer image "
            "with the face larger and more visible."
        )


def main():
    try:
        model = load_model()
        face_cascade = load_face_cascade()
    except Exception as exc:
        st.error(
            f"Application startup failed: {exc}"
        )
        st.stop()

    tab_upload, tab_webcam = st.tabs(
        [
            "Image Detection",
            "Live Webcam",
        ]
    )

    with tab_upload:
        render_upload(
            model,
            face_cascade,
        )

    with tab_webcam:
        render_webcam()

    st.markdown("---")
    st.caption(
        "MaskGuard AI | CNN + OpenCV + Streamlit"
    )


if __name__ == "__main__":
    main()
