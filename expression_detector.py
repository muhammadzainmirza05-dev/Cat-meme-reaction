"""
Facial expression detection using MediaPipe Face Mesh.
Uses beard-invariant eye-distance normalization, multi-feature geometry
(inner-brow contraction, lip-corner elevation, mouth width, brow height),
and adaptive baseline calibration for robust real-time classification.
"""
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import math
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

# ── Key MediaPipe Face Mesh Landmark Indices ───────────────────────────────────
# Eye outer and inner corners (Beard-invariant scale reference)
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
LEFT_EYE_TOP, LEFT_EYE_BOTTOM = 159, 145
RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM = 386, 374

# Eyebrow landmarks
LEFT_BROW_INNER = 70       # Innermost tip of left eyebrow
RIGHT_BROW_INNER = 300     # Innermost tip of right eyebrow
LEFT_BROW_MID = 105        # Mid arch of left eyebrow
RIGHT_BROW_MID = 334       # Mid arch of right eyebrow

# Mouth landmarks
MOUTH_LEFT = 61            # Left lip corner
MOUTH_RIGHT = 291          # Right lip corner
MOUTH_TOP = 13             # Upper inner lip center
MOUTH_BOTTOM = 14          # Lower inner lip center
UPPER_LIP_OUTER = 0        # Upper outer lip center

# Nose reference
NOSE_BRIDGE = 168          # Glabella / between eyes


class ExpressionDetector:
    def __init__(self, static_image_mode=False, max_num_faces=1,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        # Baseline trackers for adaptive calibration across different facial structures
        self.calibrated = False
        self.frame_count = 0
        self.base_brow_dist = 0.36
        self.base_brow_eye = 0.23
        self.base_mouth_width = 0.65
        self.base_corner_lift = 0.0

    @staticmethod
    def _dist(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    def process(self, rgb_frame):
        """
        Process an RGB video frame and return (expression_label, landmarks_or_None).
        Labels: 'happy', 'angry', 'surprised', 'neutral', 'no_face'
        """
        results = self.face_mesh.process(rgb_frame)
        if not results.multi_face_landmarks:
            return "no_face", None

        lm = results.multi_face_landmarks[0].landmark

        # ── 1. Beard-Invariant Scale Reference ───────────────────────────────
        # Inter-ocular distance (outer eye corners) is completely unaffected by beards or jaw movements.
        eye_dist = self._dist(lm[LEFT_EYE_OUTER], lm[RIGHT_EYE_OUTER]) + 1e-6

        # ── 2. Feature Extraction ─────────────────────────────────────────────
        # A. Inner eyebrow distance (for furrowing/anger - glabella contraction)
        inner_brow_dist = self._dist(lm[LEFT_BROW_INNER], lm[RIGHT_BROW_INNER]) / eye_dist

        # B. Brow to eye vertical distance (eyebrows pulling down in anger or up in surprise)
        left_brow_eye = self._dist(lm[LEFT_BROW_MID], lm[LEFT_EYE_TOP]) / eye_dist
        right_brow_eye = self._dist(lm[RIGHT_BROW_MID], lm[RIGHT_EYE_TOP]) / eye_dist
        brow_eye_dist = (left_brow_eye + right_brow_eye) / 2.0

        # C. Mouth width (horizontal mouth stretch during smiles)
        mouth_width = self._dist(lm[MOUTH_LEFT], lm[MOUTH_RIGHT]) / eye_dist

        # D. Lip corner elevation (corners curl upwards in smiles; y is inverted in image space)
        corner_avg_y = (lm[MOUTH_LEFT].y + lm[MOUTH_RIGHT].y) / 2.0
        lip_center_y = (lm[MOUTH_TOP].y + lm[UPPER_LIP_OUTER].y) / 2.0
        corner_lift = (lip_center_y - corner_avg_y) / eye_dist

        # E. Mouth vertical opening (for surprise / open mouth)
        mouth_open = self._dist(lm[MOUTH_TOP], lm[MOUTH_BOTTOM]) / eye_dist

        # ── 3. Adaptive Baseline Calibration ──────────────────────────────────
        # Learn the user's neutral resting face structure progressively
        if not self.calibrated:
            self.base_brow_dist = inner_brow_dist
            self.base_brow_eye = brow_eye_dist
            self.base_mouth_width = mouth_width
            self.base_corner_lift = corner_lift
            self.calibrated = True
        else:
            # Slow EMA update when near neutral to adapt to head tilt/lighting changes
            alpha = 0.015
            if 0.28 < inner_brow_dist < 0.45 and mouth_open < 0.12 and abs(corner_lift) < 0.04:
                self.base_brow_dist = (1 - alpha) * self.base_brow_dist + alpha * inner_brow_dist
                self.base_brow_eye = (1 - alpha) * self.base_brow_eye + alpha * brow_eye_dist
                self.base_mouth_width = (1 - alpha) * self.base_mouth_width + alpha * mouth_width
                self.base_corner_lift = (1 - alpha) * self.base_corner_lift + alpha * corner_lift

        # ── 4. Expression Classification ──────────────────────────────────────
        # Relative differences from neutral resting baseline
        delta_brow_dist = inner_brow_dist - self.base_brow_dist
        delta_brow_eye = brow_eye_dist - self.base_brow_eye
        delta_mouth_width = mouth_width - self.base_mouth_width
        delta_corner_lift = corner_lift - self.base_corner_lift

        # Smile detection: mouth widens AND/OR lip corners curl upwards
        is_happy = (
            (delta_corner_lift > 0.018 and delta_mouth_width > 0.03)
            or (delta_corner_lift > 0.032)
            or (delta_mouth_width > 0.08 and corner_lift > -0.01)
            or (mouth_width > 0.74 and corner_lift > 0.012)
            or (mouth_width > 0.82)
        )

        # Anger detection: inner brows pinch together (inner_brow_dist drops) AND/OR brows drop towards eyes
        is_angry = (
            (delta_brow_dist < -0.040 and delta_brow_eye < 0.01)
            or (delta_brow_dist < -0.025 and delta_brow_eye < -0.025)
            or (inner_brow_dist < 0.28)
            or (brow_eye_dist < 0.19 and inner_brow_dist < 0.33)
        )

        # Surprise detection: mouth open wide + eyebrows raised
        is_surprised = (
            (mouth_open > 0.14 and (delta_brow_eye > 0.025 or brow_eye_dist > 0.26))
            or (mouth_open > 0.24)
        )

        # Priority resolution
        if is_surprised:
            expression = "surprised"
        elif is_happy and not is_angry:
            expression = "happy"
        elif is_angry:
            expression = "angry"
        else:
            expression = "neutral"

        return expression, lm

    def close(self):
        self.face_mesh.close()