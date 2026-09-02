import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

# ── CONFIG ───────────────────────────────────────────────
MODEL_PATH = "mask_detector.h5"

# ⚠️ MUST match your training size
IMG_SIZE = (128, 128)

CONFIDENCE_THR = 0.60
SMOOTH_FRAMES = 5

labels_dict = {0: "Mask", 1: "No Mask"}
color_dict  = {0: (0, 200, 0), 1: (0, 0, 255)}

# ── LOAD MODEL ───────────────────────────────────────────
model = load_model(MODEL_PATH)
model.predict(np.zeros((1, *IMG_SIZE, 3)), verbose=0)  # warm-up

# ── LOAD DNN FACE DETECTOR ───────────────────────────────
face_net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)

# ── TEMPORAL SMOOTHING ───────────────────────────────────
history = deque(maxlen=SMOOTH_FRAMES)

def smooth(pred):
    history.append(pred)
    return np.mean(history, axis=0)

# ── WEBCAM ───────────────────────────────────────────────
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # ── FACE DETECTION USING DNN ─────────────────────────
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence_face = detections[0, 0, i, 2]

        if confidence_face > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # ── FIX BOUNDARIES ───────────────────────────
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            # ── ADD MORE PADDING (IMPORTANT FIX) ─────────
            pad = int(max(x2 - x1, y2 - y1) * 0.25)
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            # ── PREPROCESS ───────────────────────────────
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_rgb = cv2.resize(face_rgb, IMG_SIZE)

            # optional blur (helps stability)
            face_rgb = cv2.GaussianBlur(face_rgb, (3, 3), 0)

            face_rgb = face_rgb.astype("float32") / 255.0
            face_rgb = np.expand_dims(face_rgb, axis=0)

            # ── PREDICTION ───────────────────────────────
            pred = model.predict(face_rgb, verbose=0)[0]

            # smoothing
            avg_pred = smooth(pred)

            label = int(np.argmax(avg_pred))
            confidence = float(np.max(avg_pred))

            # ── DISPLAY ─────────────────────────────────
            if confidence < CONFIDENCE_THR:
                text = f"Uncertain {confidence*100:.0f}%"
                color = (0, 165, 255)
            else:
                text = f"{labels_dict[label]} {confidence*100:.0f}%"
                color = color_dict[label]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA
            )

    cv2.imshow("Mask Detection (Improved)", frame)

    if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
        break

cap.release()
cv2.destroyAllWindows()