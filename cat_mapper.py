"""
Maps a detected expression/gesture label to a cat image file.
Supports standard asset loading and dynamic user overrides.
"""
import os
import cv2
import numpy as np

ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets", "cats")

LABEL_TO_FILENAME = {
    # gestures
    "shush": "shush",
    "peace": "peace",
    "thumbs_up": "thumbs_up",
    "fist": "angry",
    "point": "neutral",
    # expressions
    "happy": "happy",
    "surprised": "surprised",
    "angry": "angry",
    "neutral": "neutral",
    "no_face": "neutral",
}

_EXTS = (".jpg", ".jpeg", ".png", ".webp")


class CatMapper:
    def __init__(self, custom_overrides=None):
        self._cache = {}
        self._custom_overrides = custom_overrides or {}

    def set_custom_image(self, label, image_bytes_or_array):
        """Allow user to dynamically override any meme image at runtime."""
        if isinstance(image_bytes_or_array, np.ndarray):
            self._custom_overrides[label] = image_bytes_or_array
        elif isinstance(image_bytes_or_array, bytes):
            nparr = np.frombuffer(image_bytes_or_array, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                self._custom_overrides[label] = img

    def _load(self, base_name):
        if base_name in self._custom_overrides:
            return self._custom_overrides[base_name]

        if base_name in self._cache:
            return self._cache[base_name]

        for ext in _EXTS:
            path = os.path.join(ASSET_DIR, base_name + ext)
            if os.path.exists(path):
                img = cv2.imread(path)
                if img is not None:
                    self._cache[base_name] = img
                    return img

        self._cache[base_name] = None
        return None

    def get_cat_image(self, label):
        base_name = LABEL_TO_FILENAME.get(label, "neutral")
        img = self._load(base_name)
        if img is None:
            img = self._load("neutral")
        return img