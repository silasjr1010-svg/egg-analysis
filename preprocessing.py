"""
Image preprocessing for inference.

Mirrors the exact pipeline used at training time in untitled8.py /
Untitled8.ipynb (`load_and_preprocess`), so predictions from
egg_detector_*.h5 / fertility_classifier_*.h5 match what the models
were trained on:
    resize -> [224, 224] -> float32 -> /255.0 -> batch dim
"""

import numpy as np
from PIL import Image

IMG_SIZE = 224


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Convert a PIL Image into the (1, 224, 224, 3) float32 array
    the detector/classifier models expect.

    Accepts any PIL-loadable image (RGB, RGBA, grayscale, palette, etc.)
    and normalizes it to match training preprocessing.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize((IMG_SIZE, IMG_SIZE))

    arr = np.asarray(image).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # -> (1, 224, 224, 3)

    return arr
