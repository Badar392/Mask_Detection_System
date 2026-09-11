"""Shared face-mask preprocessing, inference, and annotation pipeline."""

from collections import deque

import cv2
import numpy as np


IMG_SIZE = (160, 160)
THRESHOLD = 0.50
CONFIDENCE_THR = 0.70
SMOOTH_FRAMES = 5
MAX_PROCESS_WIDTH = 640

LABELS = {0: "Mask", 1: "No Mask"}
COLORS_BGR = {0: (0, 210, 90), 1: (0, 60, 230)}
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX


def enhance_image(image_bgr):
    """Lift local contrast and add restrained sharpening without resizing."""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Expected a BGR image with three channels")
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    lightness, chroma_a, chroma_b = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
    enhanced = cv2.cvtColor(
        cv2.merge((lightness, chroma_a, chroma_b)), cv2.COLOR_LAB2BGR
    )
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    return cv2.addWeighted(enhanced, 1.12, blurred, -0.12, 0)


def letterbox_image(image, target_size=IMG_SIZE, pad_value=114):
    """Resize to target_size without distortion and return scale/padding."""
    target_width, target_height = target_size
    height, width = image.shape[:2]
    scale = min(target_width / width, target_height / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    pad_x = (target_width - resized_width) // 2
    pad_y = (target_height - resized_height) // 2
    letterboxed = np.full(
        (target_height, target_width, image.shape[2]), pad_value, dtype=image.dtype
    )
    letterboxed[pad_y:pad_y + resized_height, pad_x:pad_x + resized_width] = resized
    return letterboxed, scale, (pad_x, pad_y)


def map_box_from_letterbox(box, scale, padding, image_shape):
    """Map coordinates from a letterboxed image back to the source image."""
    pad_x, pad_y = padding
    height, width = image_shape[:2]
    x1, y1, x2, y2 = box
    return (
        max(0, min(width, round((x1 - pad_x) / scale))),
        max(0, min(height, round((y1 - pad_y) / scale))),
        max(0, min(width, round((x2 - pad_x) / scale))),
        max(0, min(height, round((y2 - pad_y) / scale))),
    )


def preprocess_face(face_bgr):
    """Produce the exact RGB float input expected by the saved Keras model."""
    enhanced = enhance_image(face_bgr)
    face_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    letterboxed, _, _ = letterbox_image(face_rgb)
    # The model contains its own MobileNetV2 Rescaling(1 / 127.5, -1) layer.
    return letterboxed.astype(np.float32)


def predict_face(model, face_bgr):
    prediction = np.asarray(model.predict(np.expand_dims(preprocess_face(face_bgr), 0), verbose=0))
    if prediction.size != 1:
        raise ValueError(f"Expected one sigmoid output, got shape {prediction.shape}")
    return float(prediction.reshape(-1)[0])


def detect_faces(image_bgr, face_cascade):
    """Detect faces on the same enhanced pixels used by the live pipeline."""
    enhanced = enhance_image(image_bgr)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.08, 5, minSize=(40, 40))
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(gray, 1.05, 3, minSize=(32, 32))
    if len(faces) == 0:
        return []
    try:
        grouped, _ = cv2.groupRectangles(faces.tolist() * 2, 1, 0.2)
        faces = grouped if len(grouped) else faces
    except cv2.error:
        pass
    height, width = image_bgr.shape[:2]
    boxes = []
    for x, y, w, h in faces:
        pad = int(max(w, h) * 0.20)
        boxes.append((max(0, x - pad), max(0, y - pad),
                      min(width, x + w + pad), min(height, y + h + pad)))
    return boxes


def resize_for_processing(frame):
    height, width = frame.shape[:2]
    if width <= MAX_PROCESS_WIDTH:
        return frame, 1.0
    scale = MAX_PROCESS_WIDTH / float(width)
    resized = cv2.resize(frame, (MAX_PROCESS_WIDTH, round(height * scale)), cv2.INTER_AREA)
    return resized, scale


def draw_detections(image_bgr, detections):
    output = image_bgr.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        label = detection["label"]
        color = COLORS_BGR[label]
        text = f"{LABELS[label]} {detection['confidence'] * 100:.0f}%"
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
        (tw, th), baseline = cv2.getTextSize(text, LABEL_FONT, 0.65, 2)
        label_x = max(0, min(x1, output.shape[1] - tw - 8))
        label_y = min(output.shape[0], y1 + th + baseline + 8)
        cv2.rectangle(output, (label_x, y1), (label_x + tw + 8, label_y), color, -1)
        cv2.putText(output, text, (label_x + 4, label_y - baseline - 4),
                    LABEL_FONT, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return output


def process_frame(frame_bgr, model, face_cascade, history=None, cached_predictions=None,
                  frame_counter=1):
    """Run enhancement, detection, letterboxed inference, and annotation."""
    history = {} if history is None else history
    cached_predictions = {} if cached_predictions is None else cached_predictions
    processed, scale = resize_for_processing(frame_bgr)
    boxes = detect_faces(processed, face_cascade)
    if not boxes:
        history.clear()
        cached_predictions.clear()
        output = frame_bgr.copy()
        cv2.putText(output, "No face detected", (20, 40), LABEL_FONT, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
        return output

    detections = []
    valid_keys = set()
    run_inference = frame_counter % 2 == 0 or not cached_predictions
    for x1, y1, x2, y2 in boxes:
        key = (round(x1 / 40), round(y1 / 40), round(x2 / 40), round(y2 / 40))
        valid_keys.add(key)
        crop = processed[y1:y2, x1:x2]
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
            "box": (round(x1 / scale), round(y1 / scale), round(x2 / scale), round(y2 / scale)),
            "label": label,
            "confidence": average if label else 1.0 - average,
        })
    for key in list(cached_predictions):
        if key not in valid_keys:
            del cached_predictions[key]
    for key in list(history):
        if key not in valid_keys:
            del history[key]
    output = draw_detections(frame_bgr, detections)
    cv2.putText(output, "LIVE", (20, 35), LABEL_FONT, 0.75, (0, 220, 100), 2, cv2.LINE_AA)
    return output
