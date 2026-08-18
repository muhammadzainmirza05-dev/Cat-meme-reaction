"""
Facial expression detection using MediaPipe Face Mesh.
Uses beard-invariant eye-distance normalization, multi-feature geometry
(inner-brow contraction, lip-corner elevation, mouth width, brow height),
confidence scoring, and AR cyberpunk face-mesh rendering.
"""
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import math
import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ── Key MediaPipe Face Mesh Landmark Indices ───────────────────────────────────
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
LEFT_EYE_TOP, LEFT_EYE_BOTTOM = 159, 145
RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM = 386, 374

LEFT_BROW_INNER = 70       # Innermost tip of left eyebrow
RIGHT_BROW_INNER = 300     # Innermost tip of right eyebrow
LEFT_BROW_MID = 105        # Mid arch of left eyebrow
RIGHT_BROW_MID = 334       # Mid arch of right eyebrow

MOUTH_LEFT = 61            # Left lip corner
MOUTH_RIGHT = 291          # Right lip corner
MOUTH_TOP = 13             # Upper inner lip center
MOUTH_BOTTOM = 14          # Lower inner lip center
UPPER_LIP_OUTER = 0        # Upper outer lip center
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
        self.base_brow_dist = 0.36
        self.base_brow_eye = 0.23
        self.base_mouth_width = 0.65
        self.base_corner_lift = 0.0

    @staticmethod
    def _dist(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    def draw_ar_mesh(self, img_bgr, landmarks):
        """Draw aesthetic AR neon contours over the user's face."""
        h, w = img_bgr.shape[:2]
        
        # Draw key feature contours
        contours = [
            # Left eyebrow
            [70, 63, 105, 66, 107],
            # Right eyebrow
            [300, 293, 334, 296, 336],
            # Outer lips
            [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61],
            # Left eye
            [33, 160, 158, 133, 153, 144, 33],
            # Right eye
            [362, 385, 387, 263, 373, 380, 362],
        ]

        overlay = img_bgr.copy()
        for contour in contours:
            pts = []
            for idx in contour:
                lm = landmarks[idx]
                pts.append((int(lm.x * w), int(lm.y * h)))
            for i in range(len(pts) - 1):
                cv2.line(overlay, pts[i], pts[i+1], (232, 121, 249), 1, cv2.LINE_AA)

        # Draw glowing landmark dots at key control points
        key_indices = [LEFT_BROW_INNER, RIGHT_BROW_INNER, MOUTH_LEFT, MOUTH_RIGHT, MOUTH_TOP, MOUTH_BOTTOM]
        for idx in key_indices:
            lm = landmarks[idx]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(overlay, (cx, cy), 3, (56, 189, 248), -1, cv2.LINE_AA)

        # Soft blend
        cv2.addWeighted(overlay, 0.7, img_bgr, 0.3, 0, img_bgr)

    def process(self, rgb_frame):
        """
        Process an RGB video frame and return (expression_label, landmarks_or_None, confidence).
        Labels: 'happy', 'angry', 'surprised', 'neutral', 'no_face'
        """
        results = self.face_mesh.process(rgb_frame)
        if not results.multi_face_landmarks:
            return "no_face", None, 0.0

        lm = results.multi_face_landmarks[0].landmark

        # ── 1. Beard-Invariant Scale Reference ───────────────────────────────
        eye_dist = self._dist(lm[LEFT_EYE_OUTER], lm[RIGHT_EYE_OUTER]) + 1e-6

        # ── 2. Feature Extraction ─────────────────────────────────────────────
        # A. Inner eyebrow distance (for furrowing/anger)
        inner_brow_dist = self._dist(lm[LEFT_BROW_INNER], lm[RIGHT_BROW_INNER]) / eye_dist

        # B. Brow to eye vertical distance
        left_brow_eye = self._dist(lm[LEFT_BROW_MID], lm[LEFT_EYE_TOP]) / eye_dist
        right_brow_eye = self._dist(lm[RIGHT_BROW_MID], lm[RIGHT_EYE_TOP]) / eye_dist
        brow_eye_dist = (left_brow_eye + right_brow_eye) / 2.0

        # C. Mouth width (horizontal mouth stretch)
        mouth_width = self._dist(lm[MOUTH_LEFT], lm[MOUTH_RIGHT]) / eye_dist

        # D. Lip corner elevation (corners curl upwards in smiles)
        corner_avg_y = (lm[MOUTH_LEFT].y + lm[MOUTH_RIGHT].y) / 2.0
        lip_center_y = (lm[MOUTH_TOP].y + lm[UPPER_LIP_OUTER].y) / 2.0
        corner_lift = (lip_center_y - corner_avg_y) / eye_dist

        # E. Mouth vertical opening (for surprise)
        mouth_open = self._dist(lm[MOUTH_TOP], lm[MOUTH_BOTTOM]) / eye_dist

        # ── 3. Adaptive Baseline Calibration ──────────────────────────────────
        if not self.calibrated:
            self.base_brow_dist = inner_brow_dist
            self.base_brow_eye = brow_eye_dist
            self.base_mouth_width = mouth_width
            self.base_corner_lift = corner_lift
            self.calibrated = True
        else:
            alpha = 0.015
            if 0.28 < inner_brow_dist < 0.45 and mouth_open < 0.12 and abs(corner_lift) < 0.04:
                self.base_brow_dist = (1 - alpha) * self.base_brow_dist + alpha * inner_brow_dist
                self.base_brow_eye = (1 - alpha) * self.base_brow_eye + alpha * brow_eye_dist
                self.base_mouth_width = (1 - alpha) * self.base_mouth_width + alpha * mouth_width
                self.base_corner_lift = (1 - alpha) * self.base_corner_lift + alpha * corner_lift

        # ── 4. Expression Classification ──────────────────────────────────────
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

        # Anger detection: inner brows pinch together AND/OR brows drop towards eyes
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

        # Calculate confidence score (0.0 to 1.0)
        if is_surprised:
            expression = "surprised"
            conf = min(1.0, (mouth_open / 0.22) * 0.9 + 0.1)
        elif is_happy and not is_angry:
            expression = "happy"
            smile_intensity = max(delta_corner_lift / 0.04, delta_mouth_width / 0.10)
            conf = min(1.0, max(0.65, smile_intensity * 0.95))
        elif is_angry:
            expression = "angry"
            frown_intensity = max(abs(delta_brow_dist) / 0.06, abs(delta_brow_eye) / 0.04)
            conf = min(1.0, max(0.65, frown_intensity * 0.95))
        else:
            expression = "neutral"
            conf = 0.85

        return expression, lm, conf

    def close(self):
        self.face_mesh.close()