"""
Post-hoc Explainable AI (Grad-CAM) for the fertility classifier.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
Networks via Gradient-based Localization", ICCV 2017.
https://arxiv.org/abs/1610.02391

Why Grad-CAM here: it needs no architecture changes and no retraining
(pure post-hoc), works on any of the frozen-backbone + GAP + Dense
classifiers in this project (MobileNetV2 / EfficientNetB0 /
MobileNetV3Small / SqueezeNet from benchmark_architectures.py), and
produces a heatmap over the *input image* -- i.e. "which pixels made
the model say fertile/infertile/dead" -- which is what a non-technical
user (a farmer, a poultry researcher) actually needs to trust or
challenge a prediction, as opposed to a bare confidence percentage.

Usage:
    from gradcam import explain_prediction

    overlay, heatmap, class_idx, confidence = explain_prediction(
        pil_image=image,
        model=classifier_model,
        preprocess_fn=preprocess_image,   # from utils/preprocessing.py
    )
    overlay.save("explanation.png")
"""

from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image

try:
    import matplotlib.cm as cm
    _HAS_MPL = True
except ImportError:  # matplotlib is already a project dependency, but degrade gracefully
    _HAS_MPL = False


def find_last_conv_layer(model: tf.keras.Model):
    """
    Return the last top-level layer of `model` whose output is a 4D
    feature map (batch, H, W, channels).

    Works for both patterns used in this project:
      - Sequential([backbone, GlobalAveragePooling2D, ..., Dense])
        -> returns the backbone itself (its output IS the last conv map,
           since GAP immediately follows it).
      - A Functional model (e.g. the custom SqueezeNet) where the last
        conv layer sits directly in the top-level graph.
    """
    for layer in reversed(model.layers):
        shape = getattr(layer, "output_shape", None)
        if isinstance(shape, tuple) and len(shape) == 4:
            return layer
    raise ValueError(
        "Could not find a 4D (conv-like) layer in this model. "
        "Grad-CAM requires a spatial feature map to explain."
    )


def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
    class_index: Optional[int] = None,
    last_conv_layer=None,
) -> Tuple[np.ndarray, int, float]:
    """
    img_array: preprocessed (1, H, W, 3) float32 array -- must match the
        exact preprocessing the model was trained with.
    model: trained multi-class Keras classifier (softmax output).
    class_index: which class to explain. Defaults to the model's own
        top prediction (i.e. "explain what the model actually said").

    Returns (heatmap in [0, 1] at conv resolution, class_index used,
    softmax confidence for that class).
    """
    if last_conv_layer is None:
        last_conv_layer = find_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output],
    )

    img_tensor = tf.convert_to_tensor(img_array)

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_tensor, training=False)
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))
        confidence = float(predictions[0][class_index])
        class_channel = predictions[:, class_index]

    grads = tape.gradient(class_channel, conv_output)
    if grads is None:
        raise RuntimeError(
            "Gradient is None -- the chosen conv layer isn't connected "
            "to the prediction in this model's graph."
        )

    # Global-average-pool the gradients -> importance weight per channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU (Grad-CAM only cares about features that positively support
    # the predicted class), then normalize to [0, 1]
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    heatmap = heatmap / (max_val + 1e-8) if max_val > 0 else heatmap

    return heatmap.numpy(), class_index, confidence


def overlay_heatmap(
    pil_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> Image.Image:
    """Resize the low-res Grad-CAM heatmap to the original image size and
    alpha-blend it on top as a color overlay."""
    pil_image = pil_image.convert("RGB")

    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_img = Image.fromarray(heatmap_uint8).resize(
        pil_image.size, resample=Image.BILINEAR
    )
    heatmap_resized = np.asarray(heatmap_img).astype(np.float32) / 255.0

    if _HAS_MPL:
        colored = cm.get_cmap(colormap)(heatmap_resized)[:, :, :3]
    else:
        # Fallback: simple red-channel heat overlay if matplotlib is unavailable
        colored = np.zeros((*heatmap_resized.shape, 3), dtype=np.float32)
        colored[..., 0] = heatmap_resized

    colored_img = Image.fromarray(np.uint8(colored * 255))
    return Image.blend(pil_image, colored_img, alpha)


def explain_prediction(
    pil_image: Image.Image,
    model: tf.keras.Model,
    preprocess_fn,
    class_index: Optional[int] = None,
    alpha: float = 0.45,
):
    """
    High-level convenience wrapper for use in the Streamlit app.

    preprocess_fn: the SAME function used at inference time
        (utils.preprocessing.preprocess_image) -- Grad-CAM is only valid
        if the gradients are computed through the identical input the
        model actually sees.

    Returns: (overlay_image: PIL.Image, heatmap: np.ndarray,
              class_index: int, confidence: float)
    """
    img_array = preprocess_fn(pil_image)
    heatmap, used_class_index, confidence = make_gradcam_heatmap(
        img_array, model, class_index=class_index
    )
    overlay = overlay_heatmap(pil_image, heatmap, alpha=alpha)
    return overlay, heatmap, used_class_index, confidence
