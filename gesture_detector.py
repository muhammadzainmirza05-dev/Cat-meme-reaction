"""
Hand gesture detection using MediaPipe Hands.
Rule-based finger-state logic (no trained classifier).
"""
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import mediapipe as mp

mp_hands = mp.solutions.hands

# Landmark indices per finger: [MCP, PIP, DIP, TIP]
FINGER_TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
FINGER_PIPS = {"thumb": 2, "index": 6, "middle": 10, "ring": 14, "pinky": 18}


class GestureDetector:
    def __init__(self, max_num_hands=1, min_detection_confidence=0.6,
                 min_tracking_confidence=0.6):
        self.hands = mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    @staticmethod
    def _finger_up(lm, tip_idx, pip_idx):
        # Finger counted "up" if tip is above (smaller y) than pip joint
        return lm[tip_idx].y < lm[pip_idx].y

    def process(self, rgb_frame, face_landmarks=None):
        """Returns gesture_label or None if no hand / no match."""
        results = self.hands.process(rgb_frame)
        if not results.multi_hand_landmarks:
            return None

        lm = results.multi_hand_landmarks[0].landmark

        index_up = self._finger_up(lm, FINGER_TIPS["index"], FINGER_PIPS["index"])
        middle_up = self._finger_up(lm, FINGER_TIPS["middle"], FINGER_PIPS["middle"])
        ring_up = self._finger_up(lm, FINGER_TIPS["ring"], FINGER_PIPS["ring"])
        pinky_up = self._finger_up(lm, FINGER_TIPS["pinky"], FINGER_PIPS["pinky"])
        thumb_up = lm[FINGER_TIPS["thumb"]].y < lm[FINGER_PIPS["thumb"]].y

        # Shush: only index finger up AND index tip near the mouth
        if index_up and not middle_up and not ring_up and not pinky_up:
            if face_landmarks is not None:
                mouth = face_landmarks[13]  # upper lip landmark
                near_mouth = (
                    abs(lm[FINGER_TIPS["index"]].x - mouth.x) < 0.12
                    and abs(lm[FINGER_TIPS["index"]].y - mouth.y) < 0.15
                )
                if near_mouth:
                    return "shush"
            return "point"

        # Peace sign: index + middle up, ring + pinky down
        if index_up and middle_up and not ring_up and not pinky_up:
            return "peace"

        # Thumbs up: thumb up, everything else down
        if thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
            return "thumbs_up"

        # Fist: nothing up
        if not index_up and not middle_up and not ring_up and not pinky_up and not thumb_up:
            return "fist"

        return None

    def close(self):
        self.hands.close()