"""
Hand gesture detection using MediaPipe Hands.
Rule-based finger-state logic, confidence estimation,
and AR neon hand skeleton rendering.
"""
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Landmark indices per finger: [MCP, PIP, DIP, TIP]
FINGER_TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
FINGER_PIPS = {"thumb": 2, "index": 6, "middle": 10, "ring": 14, "pinky": 18}

HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (5, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (9, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (13, 17), (17, 18), (18, 19), (19, 20),
    # Palm base
    (0, 17)
]


class GestureDetector:
    def __init__(self, max_num_hands=1, min_detection_confidence=0.55,
                 min_tracking_confidence=0.55):
        self.hands = mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    @staticmethod
    def _finger_up(lm, tip_idx, pip_idx):
        return lm[tip_idx].y < lm[pip_idx].y

    def draw_ar_hand(self, img_bgr, landmarks):
        """Draw aesthetic AR neon skeleton over the detected hand."""
        h, w = img_bgr.shape[:2]
        overlay = img_bgr.copy()

        for start_idx, end_idx in HAND_CONNECTIONS:
            p1 = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
            p2 = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
            cv2.line(overlay, p1, p2, (52, 211, 153), 2, cv2.LINE_AA)

        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(overlay, (cx, cy), 4, (251, 146, 60), -1, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.75, img_bgr, 0.25, 0, img_bgr)

    def process(self, rgb_frame, face_landmarks=None):
        """Returns (gesture_label_or_None, hand_landmarks_or_None, confidence)."""
        results = self.hands.process(rgb_frame)
        if not results.multi_hand_landmarks:
            return None, None, 0.0

        lm = results.multi_hand_landmarks[0].landmark

        index_up = self._finger_up(lm, FINGER_TIPS["index"], FINGER_PIPS["index"])
        middle_up = self._finger_up(lm, FINGER_TIPS["middle"], FINGER_PIPS["middle"])
        ring_up = self._finger_up(lm, FINGER_TIPS["ring"], FINGER_PIPS["ring"])
        pinky_up = self._finger_up(lm, FINGER_TIPS["pinky"], FINGER_PIPS["pinky"])
        thumb_up = lm[FINGER_TIPS["thumb"]].y < lm[FINGER_PIPS["thumb"]].y

        # Shush: only index finger up AND index tip near the mouth
        if index_up and not middle_up and not ring_up and not pinky_up:
            if face_landmarks is not None:
                mouth = face_landmarks[13]
                near_mouth = (
                    abs(lm[FINGER_TIPS["index"]].x - mouth.x) < 0.14
                    and abs(lm[FINGER_TIPS["index"]].y - mouth.y) < 0.18
                )
                if near_mouth:
                    return "shush", lm, 0.95
            return "point", lm, 0.85

        # Peace sign: index + middle up, ring + pinky down
        if index_up and middle_up and not ring_up and not pinky_up:
            return "peace", lm, 0.95

        # Thumbs up: thumb up, everything else down
        if thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
            return "thumbs_up", lm, 0.95

        # Fist: nothing up
        if not index_up and not middle_up and not ring_up and not pinky_up and not thumb_up:
            return "fist", lm, 0.90

        return None, lm, 0.0

    def close(self):
        self.hands.close()