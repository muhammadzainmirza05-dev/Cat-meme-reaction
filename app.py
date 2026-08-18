import os
# Ensure legacy protobuf descriptor implementation for MediaPipe compatibility
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

from expression_detector import ExpressionDetector
from gesture_detector import GestureDetector
from cat_mapper import CatMapper

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cat Meme Reactor 🐱",
    page_icon="🐱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=DM+Sans:wght@300;400;500&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    font-family: 'DM Sans', sans-serif;
    background-color: #07071a;
    color: #f0f0ff;
}

#MainMenu, footer, header, .stDeployButton { visibility: hidden; }
[data-testid="collapsedControl"] { display: none; }

.block-container {
    padding: 1.5rem 2rem 2rem 2rem !important;
    max-width: 1400px !important;
}

/* Hero */
.hero {
    position: relative;
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.8rem;
    overflow: hidden;
    background: #0d0d2b;
    border: 1px solid rgba(233,121,249,0.18);
}
.hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background:
        radial-gradient(ellipse 60% 80% at 10% 50%, rgba(147,51,234,0.18) 0%, transparent 70%),
        radial-gradient(ellipse 50% 60% at 90% 20%, rgba(236,72,153,0.14) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    line-height: 1.1;
    background: linear-gradient(135deg, #f0abfc 0%, #e879f9 35%, #a855f7 65%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.4rem;
}
.hero-desc {
    color: rgba(240,240,255,0.5);
    font-size: 0.95rem;
    font-weight: 300;
    line-height: 1.5;
}
.hero-deco {
    position: absolute;
    right: 2.5rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 5rem;
    opacity: 0.1;
    pointer-events: none;
}

/* Section label */
.sec-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #a855f7;
    margin-bottom: 0.7rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.sec-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(168,85,247,0.4), transparent);
}

/* Compact instruction rows */
.instr-wrap { display: flex; flex-direction: column; gap: 0.3rem; }

.instr-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 8px;
    padding: 0.45rem 0.8rem;
    border-left: 3px solid transparent;
    transition: border-color 0.2s, background 0.2s;
}
.instr-row:hover { background: rgba(168,85,247,0.05); }

.instr-row.happy     { border-left-color: #facc15; }
.instr-row.surprised { border-left-color: #38bdf8; }
.instr-row.angry     { border-left-color: #f87171; }
.instr-row.neutral   { border-left-color: #6b7280; }
.instr-row.shush     { border-left-color: #e879f9; }
.instr-row.peace     { border-left-color: #34d399; }
.instr-row.thumbs_up { border-left-color: #fb923c; }

.instr-label {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 0.8rem;
    color: #f0f0ff;
    min-width: 90px;
}
.instr-trigger {
    font-size: 0.72rem;
    color: rgba(240,240,255,0.4);
}

/* Webcam panel */
.cam-panel {
    background: #0d0d2b;
    border: 1px solid rgba(168,85,247,0.18);
    border-radius: 20px;
    padding: 1.2rem 1.2rem 0.8rem 1.2rem;
}
.cam-hint {
    text-align: center;
    margin-top: 0.7rem;
}
.cam-hint span {
    display: inline-block;
    background: rgba(168,85,247,0.08);
    border: 1px solid rgba(168,85,247,0.18);
    border-radius: 100px;
    padding: 0.25rem 1rem;
    color: rgba(240,240,255,0.35);
    font-size: 0.75rem;
}

/* Make the webrtc video fill width */
video { border-radius: 10px !important; width: 100% !important; }

/* Footer */
.footer {
    text-align: center;
    margin-top: 2rem;
    color: rgba(240,240,255,0.12);
    font-size: 0.72rem;
    letter-spacing: 0.05em;
}
.footer-line {
    width: 160px; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(168,85,247,0.3), transparent);
    margin: 0.8rem auto 0.6rem;
}
</style>
""", unsafe_allow_html=True)

# ── RTC Configuration ─────────────────────────────────────────────────────────
# STUN servers allow WebRTC ICE candidate gathering across local and network configurations.
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
        ]
    }
)


# ── Video Processor ───────────────────────────────────────────────────────────
class CatMemeProcessor(VideoProcessorBase):
    def __init__(self):
        self.expr_detector = ExpressionDetector()
        self.gesture_detector = GestureDetector()
        self.cat_mapper = CatMapper()
        self.active_label = "neutral"
        self.candidate_label = "neutral"
        self.stable_count = 0
        self.STABILITY_FRAMES = 2

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        expression, face_lm = self.expr_detector.process(rgb)
        gesture = self.gesture_detector.process(rgb, face_landmarks=face_lm)
        candidate = gesture if gesture else expression

        if candidate == self.candidate_label:
            self.stable_count += 1
            if self.stable_count >= self.STABILITY_FRAMES:
                self.active_label = self.candidate_label
        else:
            self.candidate_label = candidate
            self.stable_count = 1

        active_label = self.active_label
        cat_img = self.cat_mapper.get_cat_image(active_label)
        h, w = img.shape[:2]

        if cat_img is not None:
            cat_panel = cv2.resize(cat_img, (w, h))
        else:
            cat_panel = np.full((h, w, 3), 14, dtype=np.uint8)
            for i, msg in enumerate(["Add cat images", "to assets/cats/"]):
                (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
                tx, ty = (w - tw) // 2, h // 2 + i * 36 - 10
                cv2.putText(cat_panel, msg, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 60, 100), 1)

        combined = np.hstack([img, cat_panel])

        # Label badge on video
        label_text = active_label.upper().replace("_", " ")
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        pad = 10
        cv2.rectangle(combined, (pad, pad), (tw + pad * 3, th + pad * 2 + 4), (13, 13, 35), -1)
        cv2.rectangle(combined, (pad, pad), (tw + pad * 3, th + pad * 2 + 4), (168, 85, 247), 1)
        cv2.putText(combined, label_text, (pad * 2, th + pad + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (232, 121, 249), 2)

        return av.VideoFrame.from_ndarray(combined, format="bgr24")


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">Cat Meme Reactor 🐾</div>
    <div class="hero-desc">Your expressions &amp; hand gestures control the cat in real time — strike a pose and watch it react.</div>
    <div class="hero-deco">🐱</div>
</div>
""", unsafe_allow_html=True)

# ── Two columns: small instructions | big camera ──────────────────────────────
col1, col2 = st.columns([1, 2.8], gap="large")

with col1:
    st.markdown('<div class="sec-label">Gestures &amp; Expressions</div>', unsafe_allow_html=True)

    rows = [
        ("happy",     "😊 Happy",      "Wide smile"),
        ("surprised", "😲 Surprised",  "Mouth open + brows up"),
        ("angry",     "😠 Angry",      "Furrowed brows"),
        ("neutral",   "😐 Neutral",    "Default"),
        ("shush",     "🤫 Shush",      "Index near mouth"),
        ("peace",     "✌️ Peace",      "Index + middle up"),
        ("thumbs_up", "👍 Thumbs Up",  "Thumb up only"),
    ]

    html = '<div class="instr-wrap">'
    for key, label, trigger in rows:
        html += f"""<div class="instr-row {key}">
            <span class="instr-label">{label}</span>
            <span class="instr-trigger">{trigger}</span>
        </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

with col2:
    st.markdown('<div class="sec-label">📸 Live Feed</div>', unsafe_allow_html=True)
    st.markdown('<div class="cam-panel">', unsafe_allow_html=True)

    webrtc_streamer(
        key="cat-meme-reactor",
        video_processor_factory=CatMemeProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": {"width": {"ideal": 1280}, "height": {"ideal": 720}}, "audio": False},
    )

    st.markdown('<div class="cam-hint"><span>⬅ You &nbsp;·&nbsp; Cat ➡</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="footer-line"></div>
    Built by Zain &nbsp;·&nbsp; MediaPipe · OpenCV · Streamlit
    For more projects, e-mail <a href="mailto:muhammmadzainumer@gmail.com">muhammmadzainumer@gmail.com</a>
</div>
""", unsafe_allow_html=True)
