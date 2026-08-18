import os
# Ensure legacy protobuf descriptor implementation for MediaPipe compatibility
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import time
import io
import cv2
import numpy as np
import av
from PIL import Image
import streamlit as st
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
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=DM+Sans:wght@400;500;700&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    font-family: 'DM Sans', sans-serif;
    background-color: #07071a;
    color: #f0f0ff;
}

#MainMenu, footer, header, .stDeployButton { visibility: hidden; }
[data-testid="collapsedControl"] { display: none; }

.block-container {
    padding: 1rem 1.2rem 2rem 1.2rem !important;
    max-width: 1400px !important;
}

/* Hero */
.hero {
    position: relative;
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.2rem;
    overflow: hidden;
    background: #0d0d2b;
    border: 1px solid rgba(233,121,249,0.22);
    box-shadow: 0 8px 32px rgba(168,85,247,0.12);
}
.hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background:
        radial-gradient(ellipse 60% 80% at 10% 50%, rgba(147,51,234,0.22) 0%, transparent 70%),
        radial-gradient(ellipse 50% 60% at 90% 20%, rgba(236,72,153,0.18) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1.1;
    background: linear-gradient(135deg, #f0abfc 0%, #e879f9 35%, #a855f7 65%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
}
.hero-desc {
    color: rgba(240,240,255,0.7);
    font-size: 0.9rem;
    font-weight: 400;
    line-height: 1.4;
}

/* Section label */
.sec-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #a855f7;
    margin-bottom: 0.6rem;
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
.instr-wrap { display: flex; flex-direction: column; gap: 0.35rem; }

.instr-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 0.5rem 0.85rem;
    border-left: 4px solid transparent;
    transition: all 0.2s ease;
}
.instr-row:hover {
    background: rgba(168,85,247,0.08);
    transform: translateX(2px);
}

.instr-row.happy     { border-left-color: #facc15; }
.instr-row.surprised { border-left-color: #38bdf8; }
.instr-row.angry     { border-left-color: #f87171; }
.instr-row.neutral   { border-left-color: #94a3b8; }
.instr-row.shush     { border-left-color: #e879f9; }
.instr-row.peace     { border-left-color: #34d399; }
.instr-row.thumbs_up { border-left-color: #fb923c; }

.instr-label {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    color: #f0f0ff;
}
.instr-trigger {
    font-size: 0.75rem;
    color: rgba(240,240,255,0.55);
}

/* Webcam panel */
.cam-panel {
    background: #0d0d2b;
    border: 1px solid rgba(168,85,247,0.22);
    border-radius: 16px;
    padding: 1rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

/* Make webrtc video responsive & smooth */
video {
    border-radius: 12px !important;
    width: 100% !important;
    max-height: 540px !important;
    object-fit: contain !important;
    background: #050512;
}

/* Control Pills */
.feature-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.25rem 0.65rem;
    border-radius: 20px;
    background: rgba(168,85,247,0.12);
    border: 1px solid rgba(168,85,247,0.25);
    color: #e879f9;
    font-size: 0.75rem;
    font-weight: 600;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 2rem;
    color: rgba(240,240,255,0.3);
    font-size: 0.78rem;
    letter-spacing: 0.05em;
}
.footer a { color: #c084fc; text-decoration: none; }
.footer-line {
    width: 180px; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(168,85,247,0.4), transparent);
    margin: 0.8rem auto 0.6rem;
}
</style>
""", unsafe_allow_html=True)


# ── Robust STUN / TURN Configuration (Mobile & Cloud Compatible) ─────────────
def get_rtc_configuration():
    """
    Returns high-availability STUN + Open Relay TURN servers.
    Works seamlessly across Cloud NAT and Mobile 4G/5G carriers.
    """
    ice_servers = [
        # Google STUN
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        # Cloudflare & Mozilla STUN
        {"urls": ["stun:stun.cloudflare.com:3478"]},
        {"urls": ["stun:stun.services.mozilla.com:3478"]},
        # Twilio STUN
        {"urls": ["stun:global.stun.twilio.com:3478"]},
        # Free OpenRelay TURN (Essential for Symmetric NAT / Mobile Carriers)
        {
            "urls": ["turn:openrelay.metered.ca:80", "turn:openrelay.metered.ca:443"],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
        {
            "urls": ["turn:openrelay.metered.ca:443?transport=tcp"],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
    ]

    # Optional custom secrets support if provided
    if "webrtc" in st.secrets:
        custom_servers = st.secrets["webrtc"].get("iceServers", [])
        if custom_servers:
            ice_servers = custom_servers

    return RTCConfiguration({"iceServers": ice_servers})


RTC_CONFIGURATION = get_rtc_configuration()


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
        self.show_ar_mesh = True
        self.last_frame = None
        self.fps = 0.0
        self._last_time = time.time()

    def recv(self, frame):
        now = time.time()
        dt = now - self._last_time
        self._last_time = now
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)

        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 1. Detect Expressions & Gestures
        expression, face_lm, expr_conf = self.expr_detector.process(rgb)
        gesture, hand_lm, gest_conf = self.gesture_detector.process(rgb, face_landmarks=face_lm)

        candidate = gesture if gesture else expression
        confidence = gest_conf if gesture else expr_conf

        # 2. Debouncing for smooth transitions
        if candidate == self.candidate_label:
            self.stable_count += 1
            if self.stable_count >= self.STABILITY_FRAMES:
                self.active_label = self.candidate_label
        else:
            self.candidate_label = candidate
            self.stable_count = 1

        active_label = self.active_label

        # 3. Optional AR Mesh Overlay
        if self.show_ar_mesh:
            if face_lm is not None:
                self.expr_detector.draw_ar_mesh(img, face_lm)
            if hand_lm is not None:
                self.gesture_detector.draw_ar_hand(img, hand_lm)

        # 4. Cat Meme Panel Rendering
        cat_img = self.cat_mapper.get_cat_image(active_label)
        h, w = img.shape[:2]

        if cat_img is not None:
            cat_panel = cv2.resize(cat_img, (w, h))
        else:
            cat_panel = np.full((h, w, 3), 14, dtype=np.uint8)
            msg = "Add cat images to assets/cats/"
            (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.putText(cat_panel, msg, ((w - tw) // 2, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 100, 180), 1)

        # Combine webcam + cat panel side by side
        combined = np.hstack([img, cat_panel])

        # 5. Cyberpunk HUD Badge Overlay
        label_text = f"{active_label.upper().replace('_', ' ')}"
        pct_text = f"{int(confidence * 100)}%" if confidence > 0 else ""
        hud_text = f"🐱 {label_text}  {pct_text}"

        (tw, th), _ = cv2.getTextSize(hud_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        pad = 12
        cv2.rectangle(combined, (pad, pad), (tw + pad * 2 + 10, th + pad * 2 + 6), (13, 13, 35), -1)
        cv2.rectangle(combined, (pad, pad), (tw + pad * 2 + 10, th + pad * 2 + 6), (168, 85, 247), 1)
        cv2.putText(combined, hud_text, (pad + 8, th + pad + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (232, 121, 249), 2)

        # FPS counter on top right
        fps_text = f"{int(self.fps)} FPS"
        (ftw, fth), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(combined, fps_text, (combined.shape[1] - ftw - 15, fth + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 180), 1)

        self.last_frame = combined
        return av.VideoFrame.from_ndarray(combined, format="bgr24")


# ── Hero Section ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">Cat Meme Reactor 🐾</div>
    <div class="hero-desc">Your facial expressions &amp; hand gestures control matching cat memes in real-time. Strike a pose!</div>
</div>
""", unsafe_allow_html=True)


# ── Main Two-Column Layout ────────────────────────────────────────────────────
col_side, col_main = st.columns([1, 2.5], gap="medium")

with col_side:
    st.markdown('<div class="sec-label">⚡ Trigger Cheatsheet</div>', unsafe_allow_html=True)

    rows = [
        ("happy",     "😊 Happy",      "Wide smile / grin"),
        ("surprised", "😲 Surprised",  "Mouth open + raised brows"),
        ("angry",     "😠 Angry",      "Furrowed brows / scowl"),
        ("neutral",   "😐 Neutral",    "Default resting face"),
        ("shush",     "🤫 Shush",      "Index finger near mouth"),
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

    st.markdown("<br>", unsafe_allow_html=True)

    # Controls Expander
    with st.expander("🎛️ App Settings & AR Controls", expanded=True):
        show_mesh = st.checkbox("Show Cyber AR Mesh & Landmarks", value=True)
        st.caption("✨ Beard-invariant geometric tracking with adaptive calibration.")


with col_main:
    st.markdown('<div class="sec-label">📸 Live Reactor Feed</div>', unsafe_allow_html=True)
    st.markdown('<div class="cam-panel">', unsafe_allow_html=True)

    ctx = webrtc_streamer(
        key="cat-meme-reactor-v2",
        video_processor_factory=CatMemeProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": {
                "facingMode": "user",
                "width": {"ideal": 640},
                "height": {"ideal": 480},
            },
            "audio": False,
        },
        async_processing=True,
    )

    if ctx.video_processor:
        ctx.video_processor.show_ar_mesh = show_mesh

    st.markdown('</div>', unsafe_allow_html=True)

    # Snapshot capture helper
    if ctx.video_processor and ctx.video_processor.last_frame is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📸 Capture Meme Card Snapshot"):
            frame_rgb = cv2.cvtColor(ctx.video_processor.last_frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=95)
            byte_im = buf.getvalue()
            st.image(pil_img, caption="Meme Snapshot Captured!", use_container_width=True)
            st.download_button(
                label="💾 Download Meme Card",
                data=byte_im,
                file_name=f"cat_reaction_{int(time.time())}.jpg",
                mime="image/jpeg",
            )


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="footer-line"></div>
    Built by Zain &nbsp;·&nbsp; MediaPipe · OpenCV · Streamlit-WebRTC
    <br>For feedback or questions: <a href="mailto:muhammmadzainumer@gmail.com">muhammmadzainumer@gmail.com</a>
</div>
""", unsafe_allow_html=True)
