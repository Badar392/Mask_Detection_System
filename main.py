import streamlit as st

st.set_page_config(
    page_title="Webcam Diagnostic",
    page_icon="🔍"
)

st.title("Webcam Environment Diagnostic")

st.write("Starting diagnostics...")

# --------------------------------------------------
# TEST 1 - NumPy
# --------------------------------------------------

try:
    import numpy as np

    st.success(
        f"NumPy OK: {np.__version__}"
    )

except Exception as e:

    st.error("NumPy FAILED")
    st.exception(e)


# --------------------------------------------------
# TEST 2 - OpenCV
# --------------------------------------------------

try:
    import cv2

    st.success(
        f"OpenCV OK: {cv2.__version__}"
    )

except Exception as e:

    st.error("OpenCV FAILED")
    st.exception(e)


# --------------------------------------------------
# TEST 3 - TensorFlow
# --------------------------------------------------

try:
    import tensorflow as tf

    st.success(
        f"TensorFlow OK: {tf.__version__}"
    )

except Exception as e:

    st.error("TensorFlow FAILED")
    st.exception(e)


# --------------------------------------------------
# TEST 4 - PyAV
# --------------------------------------------------

try:
    import av

    st.success(
        f"PyAV OK: {av.__version__}"
    )

except Exception as e:

    st.error("PyAV FAILED")
    st.exception(e)


# --------------------------------------------------
# TEST 5 - aiortc
# --------------------------------------------------

try:
    import aiortc

    st.success(
        f"aiortc OK: {aiortc.__version__}"
    )

except Exception as e:

    st.error("aiortc FAILED")
    st.exception(e)


# --------------------------------------------------
# TEST 6 - streamlit-webrtc
# --------------------------------------------------

try:

    import streamlit_webrtc

    st.success(
        "streamlit-webrtc import OK"
    )

    st.write(
        "Version:",
        getattr(
            streamlit_webrtc,
            "__version__",
            "unknown"
        )
    )

except Exception as e:

    st.error(
        "streamlit-webrtc FAILED"
    )

    st.exception(e)


# --------------------------------------------------
# TEST 7 - Basic WebRTC import
# --------------------------------------------------

try:

    from streamlit_webrtc import (
        webrtc_streamer,
        VideoProcessorBase,
        RTCConfiguration
    )

    st.success(
        "WebRTC components import OK"
    )

except Exception as e:

    st.error(
        "WebRTC components FAILED"
    )

    st.exception(e)


st.divider()

st.success(
    "If you can see this page, the Python "
    "environment is starting successfully."
)
