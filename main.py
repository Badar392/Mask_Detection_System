import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import time
from collections import deque

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mask Guard AI",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 40%, #0f2336 70%, #091a28 100%);
    min-height: 100vh;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stAppViewContainer"]::before {
    content:''; position:fixed; top:-40%; left:-20%; width:70%; height:70%;
    background:radial-gradient(ellipse,rgba(0,200,200,0.07) 0%,transparent 70%);
    pointer-events:none; z-index:0;
}
[data-testid="stAppViewContainer"]::after {
    content:''; position:fixed; bottom:-30%; right:-10%; width:60%; height:60%;
    background:radial-gradient(ellipse,rgba(56,100,240,0.07) 0%,transparent 70%);
    pointer-events:none; z-index:0;
}
#MainMenu, footer, header { visibility:hidden; }
[data-testid="stToolbar"] { display:none; }
[data-testid="block-container"] {
    padding:2rem 3rem !important; max-width:1100px; margin:auto;
    position:relative; z-index:1;
}

/* Hero */
.hero { text-align:center; padding:2rem 1rem 1.5rem; }
.hero-badge {
    display:inline-block; background:rgba(0,200,200,0.12);
    border:1px solid rgba(0,200,200,0.3); color:#00cccc;
    font-size:0.72rem; font-weight:500; letter-spacing:0.18em;
    text-transform:uppercase; padding:0.3rem 1rem; border-radius:50px; margin-bottom:1rem;
}
.hero-title {
    font-family:'Syne',sans-serif; font-size:clamp(2.2rem,5vw,3.5rem);
    font-weight:800; color:#e8f4f8; line-height:1.1; margin:0 0 0.6rem; letter-spacing:-0.02em;
}
.hero-title span {
    background:linear-gradient(90deg,#00cccc,#3864f0);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}
.hero-sub { font-size:1rem; font-weight:300; color:#6a8fa8; margin:0; }

/* Tabs */
[data-testid="stTabs"] { margin-top:1.5rem; }
[data-testid="stTabs"] [role="tablist"] {
    background:rgba(255,255,255,0.03) !important;
    border:1px solid rgba(255,255,255,0.07) !important;
    border-radius:14px !important; padding:0.3rem !important; gap:0.3rem !important;
}
[data-testid="stTabs"] button[role="tab"] {
    font-family:'Syne',sans-serif !important; font-weight:600 !important;
    font-size:0.88rem !important; color:#5a7a8e !important;
    border-radius:10px !important; padding:0.55rem 1.4rem !important; border:none !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background:linear-gradient(135deg,#00aaaa,#3864f0) !important;
    color:white !important; box-shadow:0 4px 16px rgba(0,170,170,0.3) !important;
}
[data-testid="stTabPanel"] {
    background:rgba(255,255,255,0.02) !important;
    border:1px solid rgba(255,255,255,0.06) !important;
    border-radius:16px !important; padding:1.8rem !important; margin-top:0.5rem !important;
}

/* Glass card */
.glass-card {
    background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
    border-radius:20px; padding:1.6rem; backdrop-filter:blur(12px);
    box-shadow:0 8px 32px rgba(0,0,0,0.3);
}

/* File uploader */
[data-testid="stFileUploader"] {
    background:rgba(255,255,255,0.03) !important;
    border:2px dashed rgba(0,200,200,0.25) !important; border-radius:16px !important;
    padding:1.5rem !important;
}
[data-testid="stFileUploader"]:hover {
    border-color:rgba(0,200,200,0.55) !important; background:rgba(0,200,200,0.04) !important;
}
[data-testid="stFileUploadDropzone"] { background:transparent !important; }

/* Buttons */
.stButton > button {
    background:linear-gradient(135deg,#00aaaa,#3864f0) !important;
    color:white !important; border:none !important; border-radius:12px !important;
    font-family:'Syne',sans-serif !important; font-weight:600 !important;
    font-size:0.92rem !important; padding:0.6rem 1.8rem !important;
    box-shadow:0 4px 20px rgba(0,170,170,0.25) !important; width:100%;
}
.stButton > button:hover { opacity:0.88 !important; transform:translateY(-1px) !important; }

/* Section label */
.section-label {
    font-family:'Syne',sans-serif; font-size:0.68rem; font-weight:600;
    color:#00cccc; text-transform:uppercase; letter-spacing:0.18em; margin-bottom:0.7rem;
}

/* Stat boxes */
.stat-row { display:flex; gap:0.5rem; margin-top:0.7rem; }
.stat-box {
    flex:1; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07);
    border-radius:14px; padding:0.9rem 1rem; text-align:center;
}
.stat-val { font-family:'Syne',sans-serif; font-size:1.3rem; font-weight:700; color:#00cccc; }
.stat-key { font-size:0.7rem; color:#5a7a8e; text-transform:uppercase; letter-spacing:0.1em; margin-top:0.15rem; }

/* Result badges */
.result-safe {
    display:flex; align-items:center; gap:0.8rem;
    background:rgba(0,210,140,0.1); border:1.5px solid rgba(0,210,140,0.35);
    border-radius:14px; padding:1.1rem 1.4rem; margin-bottom:1rem;
}
.result-danger {
    display:flex; align-items:center; gap:0.8rem;
    background:rgba(255,75,90,0.1); border:1.5px solid rgba(255,75,90,0.35);
    border-radius:14px; padding:1.1rem 1.4rem; margin-bottom:1rem;
}
.rlabel { font-family:'Syne',sans-serif; font-size:1.2rem; font-weight:700; }
.rsub   { font-size:0.8rem; margin-top:0.1rem; }
.safe-text  { color:#00d28c; }
.danger-text{ color:#ff4b5a; }
.safe-sub   { color:#4dba8c; }
.danger-sub { color:#cc3d4a; }

/* Confidence track */
.conf-track {
    width:100%; height:8px; background:rgba(255,255,255,0.08);
    border-radius:99px; overflow:hidden;
}

/* Image border */
[data-testid="stImage"] img {
    border-radius:14px !important; border:1px solid rgba(255,255,255,0.08) !important;
}

/* Status pill */
.status-live {
    display:inline-flex; align-items:center; gap:0.4rem;
    background:rgba(0,210,140,0.12); border:1px solid rgba(0,210,140,0.3);
    border-radius:50px; padding:0.25rem 0.8rem;
    font-size:0.72rem; font-weight:600; color:#00d28c;
    text-transform:uppercase; letter-spacing:0.1em;
}
.dot {
    width:7px; height:7px; border-radius:50%; background:#00d28c;
    display:inline-block; animation:blink 1.2s infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }

/* Webcam placeholder */
.cam-placeholder {
    text-align:center; padding:3.5rem 2rem;
    background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06);
    border-radius:16px;
}

/* Tips */
.tips {
    text-align:center; margin-top:1.5rem; padding:0.9rem;
    background:rgba(255,255,255,0.02); border-radius:12px;
    border:1px solid rgba(255,255,255,0.05); color:#4a6a7e; font-size:0.83rem;
}
.tips strong { color:#00cccc; }
.stSpinner > div { border-top-color:#00cccc !important; }
hr { border-color:rgba(255,255,255,0.06) !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE       = (128, 128)
CONFIDENCE_THR = 0.70
SMOOTH_FRAMES  = 5
labels_dict    = {0: "Mask", 1: "No Mask"}
color_bgr      = {0: (0, 210, 90), 1: (0, 60, 230)}

# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_mask_model():
    m = load_model("mask_detector.h5")
    m.predict(np.zeros((1, *IMG_SIZE, 3)), verbose=0)
    return m

@st.cache_resource(show_spinner=False)
def load_face_cascade():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

# ── Helpers ───────────────────────────────────────────────────────────────────
def predict_image(model, image_rgb_np):
    inp = cv2.resize(image_rgb_np, IMG_SIZE).astype("float32") / 255.0
    return model.predict(np.expand_dims(inp, axis=0), verbose=0)[0]

def preprocess_face(face_bgr):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, IMG_SIZE).astype("float32") / 255.0
    return face_rgb

def detect_and_annotate(frame, model, face_cascade, history):
    gray  = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60))

    if len(faces) == 0:
        history.clear()
        return frame

    fl, _ = cv2.groupRectangles(faces.tolist() * 2, 1, 0.2)
    if len(fl) == 0:
        return frame

    crops, regions = [], []
    for (x, y, w, h) in fl:
        pad = int(max(w, h) * 0.10)
        x1, y1 = max(0, x-pad), max(0, y-pad)
        x2, y2 = min(frame.shape[1], x+w+pad), min(frame.shape[0], y+h+pad)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crops.append(preprocess_face(crop))
        regions.append((x1, y1, x2, y2))

    if not crops:
        return frame

    preds = model.predict(np.stack(crops), verbose=0)
    for i, (x1, y1, x2, y2) in enumerate(regions):
        key = (round(x1/50), round(y1/50))
        if key not in history:
            history[key] = deque(maxlen=SMOOTH_FRAMES)
        history[key].append(preds[i])
        avg   = np.mean(history[key], axis=0)
        label = int(np.argmax(avg))
        conf  = float(np.max(avg))

        if conf < CONFIDENCE_THR:
            text, color = f"Uncertain {conf*100:.0f}%", (0, 165, 255)
        else:
            text, color = f"{labels_dict[label]}  {conf*100:.0f}%", color_bgr[label]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, text, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    return frame

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">AI-Powered Detection</div>
    <h1 class="hero-title">Mask<span>Guard</span> AI</h1>
    <p class="hero-sub">Face mask detection — upload a photo or use your webcam live</p>
</div>
""", unsafe_allow_html=True)

# ── Load resources ────────────────────────────────────────────────────────────
model_loaded = False
with st.spinner("Loading model…"):
    try:
        model        = load_mask_model()
        face_cascade = load_face_cascade()
        model_loaded = True
    except Exception as e:
        st.markdown(f"""
        <div class="glass-card" style="border-color:rgba(255,75,90,0.3);">
            <p style="color:#ff4b5a;font-family:'Syne',sans-serif;font-weight:700;margin:0;">⚠️ Model not found</p>
            <p style="color:#6a8fa8;font-size:0.85rem;margin-top:0.5rem;">
                Place <code style="color:#00cccc;">mask_detector.h5</code> in the same folder and restart.<br>
                <small>{e}</small>
            </p>
        </div>""", unsafe_allow_html=True)

if model_loaded:
    tab_upload, tab_webcam = st.tabs(["📤  Upload Image", "📷  Live Webcam"])

    # ════════════════════════════════════════════
    # TAB 1 — UPLOAD
    # ════════════════════════════════════════════
    with tab_upload:
        uploaded_file = st.file_uploader(
            "Drop your image here or click to browse",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="visible",
        )
        st.markdown("<br>", unsafe_allow_html=True)

        if uploaded_file:
            image_pil = Image.open(uploaded_file).convert("RGB")
            image_np  = np.array(image_pil)

            col_img, _, col_res = st.columns([5, 0.3, 5])

            with col_img:
                st.markdown('<div class="section-label">Input Image</div>', unsafe_allow_html=True)
                st.image(image_pil, use_container_width=True)
                h, w = image_np.shape[:2]
                # Stat boxes via native Streamlit columns (no raw HTML tables)
                s1, s2, s3 = st.columns(3)
                with s1:
                    st.markdown(f'<div class="stat-box"><div class="stat-val">{w}px</div><div class="stat-key">Width</div></div>', unsafe_allow_html=True)
                with s2:
                    st.markdown(f'<div class="stat-box"><div class="stat-val">{h}px</div><div class="stat-key">Height</div></div>', unsafe_allow_html=True)
                with s3:
                    st.markdown('<div class="stat-box"><div class="stat-val">RGB</div><div class="stat-key">Mode</div></div>', unsafe_allow_html=True)

            with col_res:
                st.markdown('<div class="section-label">Detection Result</div>', unsafe_allow_html=True)
                with st.spinner("Analysing…"):
                    pred  = predict_image(model, image_np)

                label = int(np.argmax(pred))
                conf  = float(np.max(pred))
                pct   = int(conf * 100)

                # ── Result badge ──
                if label == 0:
                    st.markdown("""
                    <div class="result-safe">
                        <div style="font-size:2rem">😷</div>
                        <div>
                            <div class="rlabel safe-text">Mask Detected</div>
                            <div class="rsub safe-sub">Face covering is present</div>
                        </div>
                    </div>""", unsafe_allow_html=True)
                    bar_color = "linear-gradient(90deg,#00a876,#00d28c)"
                else:
                    st.markdown("""
                    <div class="result-danger">
                        <div style="font-size:2rem">🚫</div>
                        <div>
                            <div class="rlabel danger-text">No Mask Detected</div>
                            <div class="rsub danger-sub">No face covering found</div>
                        </div>
                    </div>""", unsafe_allow_html=True)
                    bar_color = "linear-gradient(90deg,#cc1a28,#ff4b5a)"

                # ── Confidence score ──
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-label">Confidence Score</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="font-family:Syne,sans-serif;font-size:1.9rem;font-weight:700;color:#e8f4f8;margin-bottom:0.4rem;">{pct}<span style="font-size:1rem;color:#6a8fa8;font-weight:400;">%</span></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="conf-track"><div style="height:100%;border-radius:99px;width:{pct}%;background:{bar_color};"></div></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # ── Class probabilities — use native progress bars ──
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-label">Class Probabilities</div>', unsafe_allow_html=True)

                mask_pct    = float(pred[0])
                no_mask_pct = float(pred[1])

                st.markdown(f'<div style="font-size:0.8rem;color:#6a8fa8;margin-bottom:0.2rem;">With Mask &nbsp; <span style="color:#00d28c;font-weight:600;">{mask_pct*100:.1f}%</span></div>', unsafe_allow_html=True)
                st.progress(mask_pct)

                st.markdown(f'<div style="font-size:0.8rem;color:#6a8fa8;margin:0.6rem 0 0.2rem;">Without Mask &nbsp; <span style="color:#ff4b5a;font-weight:600;">{no_mask_pct*100:.1f}%</span></div>', unsafe_allow_html=True)
                st.progress(no_mask_pct)

        else:
            st.markdown("""
            <div class="cam-placeholder">
                <div style="font-size:3.2rem;margin-bottom:0.8rem;">🖼️</div>
                <p style="font-family:'Syne',sans-serif;font-size:1.05rem;font-weight:600;color:#e8f4f8;margin:0 0 0.4rem;">
                    No image uploaded yet
                </p>
                <p style="color:#4a6a7e;font-size:0.85rem;margin:0;">
                    Upload a JPG, PNG, or WebP image above to begin
                </p>
            </div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════
    # TAB 2 — WEBCAM
    # ════════════════════════════════════════════
    with tab_webcam:
        if "webcam_running" not in st.session_state:
            st.session_state.webcam_running = False
        if "cam_history" not in st.session_state:
            st.session_state.cam_history = {}

        col_ctrl, _, col_feed = st.columns([2, 0.3, 7])

        with col_ctrl:
            st.markdown('<div class="section-label">Controls</div>', unsafe_allow_html=True)

            if not st.session_state.webcam_running:
                if st.button("▶  Start Webcam"):
                    st.session_state.webcam_running = True
                    st.session_state.cam_history    = {}
                    st.rerun()
            else:
                if st.button("⏹  Stop Webcam"):
                    st.session_state.webcam_running = False
                    st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)

            if st.session_state.webcam_running:
                st.markdown('<div class="status-live"><span class="dot"></span> LIVE</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="display:inline-flex;align-items:center;gap:0.4rem;background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);border-radius:50px;padding:0.25rem 0.8rem;font-size:0.72rem;font-weight:600;color:#5a7a8e;text-transform:uppercase;letter-spacing:0.1em;">⏸ STOPPED</div>', unsafe_allow_html=True)

            st.markdown("""
            <div style="margin-top:1.4rem;font-size:0.78rem;color:#4a6a7e;line-height:1.8;">
                <div style="color:#6a8fa8;font-weight:500;margin-bottom:0.3rem;">How to use</div>
                Click <strong style="color:#00cccc;">Start</strong> to begin<br>
                Face the camera clearly<br>
                Good lighting helps<br>
                Click <strong style="color:#00cccc;">Stop</strong> when done
            </div>""", unsafe_allow_html=True)

        with col_feed:
            st.markdown('<div class="section-label">Camera Feed</div>', unsafe_allow_html=True)
            frame_slot = st.empty()

            if st.session_state.webcam_running:
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)

                # Run until Stop is pressed — check session flag each frame
                while st.session_state.webcam_running:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame = detect_and_annotate(
                        frame, model, face_cascade, st.session_state.cam_history
                    )

                    frame_slot.image(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        use_container_width=True,
                    )

                cap.release()
                st.session_state.webcam_running = False

            else:
                frame_slot.markdown("""
                <div class="cam-placeholder">
                    <div style="font-size:3rem;margin-bottom:0.8rem;">📷</div>
                    <p style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:600;color:#e8f4f8;margin:0 0 0.4rem;">
                        Webcam is off
                    </p>
                    <p style="color:#4a6a7e;font-size:0.83rem;margin:0;">
                        Click <strong style="color:#00cccc;">Start Webcam</strong> to begin live detection
                    </p>
                </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="tips">
    <strong>💡 Tips:</strong>&nbsp;
    Use well-lit, front-facing photos &nbsp;·&nbsp;
    Avoid heavy filters or blur &nbsp;·&nbsp;
    Single face per image works best
</div>
""", unsafe_allow_html=True)