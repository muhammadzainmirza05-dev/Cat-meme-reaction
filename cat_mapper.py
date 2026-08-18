"""
Maps a detected expression/gesture label to a cat image file.
Drop your own cat images into assets/cats/ using these exact filenames
(jpg or png both work — the loader tries a few extensions).
"""
import os
import cv2

ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets", "cats")

# gesture labels take priority over expression labels (checked first in main.py)
LABEL_TO_FILENAME = {
    # gestures
    "shush": "shush",
    "peace": "peace",
    "thumbs_up": "thumbs_up",
    "fist": "angry",       # reuse angry cat for a fist, or add fist.jpg
    "point": "neutral",
    # expressions
    "happy": "happy",
    "surprised": "surprised",
    "angry": "angry",
    "neutral": "neutral",
    "no_face": "neutral",
}

_EXTS = (".jpg", ".jpeg", ".png")


class CatMapper:
    def __init__(self):
        self._cache = {}

    def _load(self, base_name):
        if base_name in self._cache:
            return self._cache[base_name]
        for ext in _EXTS:
            path = os.path.join(ASSET_DIR, base_name + ext)
            if os.path.exists(path):
                img = cv2.imread(path)
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