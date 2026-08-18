"""
Cat Meme Reactor 🐱
Webcam on the left, matching cat meme on the right — updates live as your
expression or hand gesture changes. Press 'q' to quit.
"""
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import cv2
import numpy as np

from expression_detector import ExpressionDetector
from gesture_detector import GestureDetector
from cat_mapper import CatMapper

PANEL_HEIGHT = 480
PANEL_WIDTH = 360


def resize_to_panel(frame, w=PANEL_WIDTH, h=PANEL_HEIGHT):
    fh, fw = frame.shape[:2]
    scale = min(w / fw, h / fh)
    resized = cv2.resize(frame, (int(fw * scale), int(fh * scale)))
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    y_off = (h - resized.shape[0]) // 2
    x_off = (w - resized.shape[1]) // 2
    canvas[y_off:y_off + resized.shape[0], x_off:x_off + resized.shape[1]] = resized
    return canvas


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam. Check your camera index/permissions.")
        return

    expr_detector = ExpressionDetector()
    gesture_detector = GestureDetector()
    cat_mapper = CatMapper()

    active_label = "neutral"
    candidate_label = "neutral"
    stable_count = 0
    STABILITY_FRAMES = 2

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            expression, face_lm = expr_detector.process(rgb)
            gesture = gesture_detector.process(rgb, face_landmarks=face_lm)

            candidate = gesture if gesture else expression

            # Debounce: only switch cat once label holds steady for 2 frames
            if candidate == candidate_label:
                stable_count += 1
                if stable_count >= STABILITY_FRAMES:
                    active_label = candidate_label
            else:
                candidate_label = candidate
                stable_count = 1

            cat_img = cat_mapper.get_cat_image(active_label)

            left_panel = resize_to_panel(frame)
            if cat_img is not None:
                right_panel = resize_to_panel(cat_img)
            else:
                right_panel = np.zeros((PANEL_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)
                cv2.putText(right_panel, "no cat image", (20, PANEL_HEIGHT // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            combined = np.hstack([left_panel, right_panel])
            cv2.putText(combined, f"label: {active_label}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Cat Meme Reactor", combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        expr_detector.close()
        gesture_detector.close()


if __name__ == "__main__":
    main()