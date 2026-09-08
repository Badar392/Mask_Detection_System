import streamlit as st

st.set_page_config(
    page_title="WebRTC Test",
    page_icon="🎥"
)

st.title("WebRTC Camera Test")

import av

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase
)


class VideoProcessor(VideoProcessorBase):

    def recv(self, frame):

        img = frame.to_ndarray(
            format="bgr24"
        )

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


webrtc_streamer(
    key="camera-test",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=False,
)
