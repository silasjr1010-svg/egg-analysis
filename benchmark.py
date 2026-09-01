"""
Lightweight-architecture benchmark for the Stage-2 fertility classifier
(dead / fertile / infertile).

Compares the project's MobileNetV2 baseline against three alternative
lightweight backbones:

    - EfficientNetB0     (stand-in for "EfficientNet-Lite": Keras does
                           not ship the official *-Lite variants, which
                           differ from EfficientNetB0 mainly by removing
                           squeeze-and-excite blocks and using ReLU6
                           instead of Swish for better edge-device
                           quantization. EfficientNetB0 is the closest
                           available architecture in tf.keras.applications
                           and is reported here under that label.)
    - MobileNetV3Small   (natively lightweight, in tf.keras.applications)
    - SqueezeNet         (custom Fire-module implementation -- not in
                           tf.keras.applications, no ImageNet weights
                           available, so it is trained from scratch.
                           This is flagged explicitly in the results:
                           it is not an apples-to-apples transfer-learning
                           comparison against the other three.)

For each architecture this script reports: validation accuracy, macro
F1, per-class precision/recall, total & trainable parameter count,
saved model size (MB), and mean single-image inference latency (ms) --
the metrics that matter for "which model should actually run on
small-scale-farm hardware", not just raw accuracy.

Expected inputs (produced by untitled8.py / Untitled8.ipynb):
    ./labels_table.csv     (columns: path, label)
    ./label_mapping.json   ({"dead": 0, "fertile": 1, "infertile": 2})

Outputs:
    ./benchmark_results.csv
    ./benchmark_comparison.png   (accuracy / params / latency bar chart)
    ./benchmark_tradeoff.png     (accuracy vs. model size scatter)
    ./<architecture>_classifier.h5  (each trained model, best checkpoint)
"""

import json
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
AUTOTUNE = tf.data.AUTOTUNE
RANDOM_STATE = 42  # matches the split used to train the original MobileNetV2 model


# ============================================================
# Data pipeline (shared across all architectures)
# ============================================================

def load_and_decode(path):
    img = tf.io.read_file(path)
    # decode_image (not decode_jpeg): several source files are PNG-encoded
    # despite a .jpg extension -- see labels_table.csv filenames.
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return img  # float32, range [0, 255] -- NOT rescaled here


def make_dataset(dataframe, preprocess_fn, training, batch_size=BATCH_SIZE):
    """preprocess_fn receives a [0, 255]-range float32 image tensor and
    must return the input format the architecture expects. Each
    architecture applies its own official preprocess_input, so this
    stays fair even though the backbones expect different ranges."""
    paths = dataframe["path"].values
    labels = dataframe["label_idx"].values

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _load(path, label):
        img = load_and_decode(path)
        img = preprocess_fn(img)
        return img, label

    if training:
        ds = ds.shuffle(len(dataframe))
    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds


# ============================================================
# SqueezeNet (custom -- not available in tf.keras.applications)
# ============================================================

def _fire_module(x, squeeze_filters, expand_filters, name):
    x = layers.Conv2D(squeeze_filters, 1, activation="relu", padding="same",
                       name=f"{name}_squeeze")(x)
    left = layers.Conv2D(expand_filters, 1, activation="relu", padding="same",
                          name=f"{name}_expand1x1")(x)
    right = layers.Conv2D(expand_filters, 3, activation="relu", padding="same",
                           name=f"{name}_expand3x3")(x)
    return layers.Concatenate(name=f"{name}_concat")([left, right])


def build_squeezenet(input_shape, num_classes):
    """SqueezeNet 1.1-style architecture (Iandola et al., 2016), adapted
    with a GAP classification head. No pretrained weights exist for this
    architecture in Keras -- trained from scratch."""
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    x = _fire_module(x, 16, 64, "fire2")
    x = _fire_module(x, 16, 64, "fire3")
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    x = _fire_module(x, 32, 128, "fire4")
    x = _fire_module(x, 32, 128, "fire5")
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    x = _fire_module(x, 48, 192, "fire6")
    x = _fire_module(x, 48, 192, "fire7")
    x = _fire_module(x, 64, 256, "fire8")
    x = _fire_module(x, 64, 256, "fire9")

    x = layers.Dropout(0.5)(x)
    x = layers.Conv2D(num_classes, 1, activation="relu", padding="same",
                       name="final_conv")(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Activation("softmax")(x)

    return models.Model(inputs, outputs, name="squeezenet")


# ============================================================
# Model factory
# ============================================================

def build_model(architecture, num_classes, img_size=IMG_SIZE):
    """Returns (model, preprocess_fn, uses_imagenet_pretraining)."""
    input_shape = (img_size, img_size, 3)

    if architecture == "mobilenetv2":
        backbone = tf.keras.applications.MobileNetV2(
            include_top=False, input_shape=input_shape, weights="imagenet"
        )
        backbone.trainable = False
        preprocess_fn = tf.keras.applications.mobilenet_v2.preprocess_input
        pretrained = True

    elif architecture == "efficientnetb0":
        backbone = tf.keras.applications.EfficientNetB0(
            include_top=False, input_shape=input_shape, weights="imagenet"
        )
        backbone.trainable = False
        # EfficientNet's preprocess_input is a documented pass-through:
        # normalization is built into the model as an internal Rescaling
        # layer, so it expects raw [0, 255] input, same as we feed it.
        preprocess_fn = tf.keras.applications.efficientnet.preprocess_input
        pretrained = True

    elif architecture == "mobilenetv3small":
        backbone = tf.keras.applications.MobileNetV3Small(
            include_top=False, input_shape=input_shape, weights="imagenet",
            pooling=None,
        )
        backbone.trainable = False
        # Also a documented pass-through (built-in Rescaling layer).
        preprocess_fn = tf.keras.applications.mobilenet_v3.preprocess_input
        pretrained = True

    elif architecture == "squeezenet":
        model = build_squeezenet(input_shape, num_classes)
        preprocess_fn = lambda img: img / 255.0  # simple rescale; trained from scratch
        return model, preprocess_fn, False

    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    model = models.Sequential([
        backbone,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax"),
    ], name=architecture)

    return model, preprocess_fn, pretrained


# ============================================================
# Train + evaluate a single architecture
# ============================================================

def train_and_evaluate(architecture, train_df, val_df, num_classes, class_names,
                        output_dir=".", epochs=EPOCHS, batch_size=BATCH_SIZE):
    print(f"\n{'=' * 60}\nBenchmarking: {architecture}\n{'=' * 60}")

    model, preprocess_fn, pretrained = build_model(architecture, num_classes)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    train_ds = make_dataset(train_df, preprocess_fn, training=True, batch_size=batch_size)
    val_ds = make_dataset(val_df, preprocess_fn, training=False, batch_size=batch_size)

    ckpt_path = os.path.join(output_dir, f"{architecture}_classifier.h5")
    ckpt = callbacks.ModelCheckpoint(
        ckpt_path, save_best_only=True, monitor="val_accuracy", mode="max", verbose=0
    )
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    start_train = time.perf_counter()
    model.fit(
        train_ds, validation_data=val_ds, epochs=epochs,
        callbacks=[ckpt, early_stop], verbose=1,
    )
    train_seconds = time.perf_counter() - start_train

    # ---- Evaluation ----
    y_true, y_pred = [], []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    # ---- Efficiency metrics ----
    total_params = model.count_params()
    trainable_params = sum(int(tf.size(w)) for w in model.trainable_weights)

    model_size_mb = (
        os.path.getsize(ckpt_path) / (1024 * 1024) if os.path.exists(ckpt_path) else float("nan")
    )

    # Single-image inference latency (mean of 20 runs, after 3 warm-up runs)
    sample_batch = next(iter(val_ds.take(1)))[0][:1]
    for _ in range(3):
        model.predict(sample_batch, verbose=0)
    latencies = []
    for _ in range(20):
        t0 = time.perf_counter()
        model.predict(sample_batch, verbose=0)
        latencies.append((time.perf_counter() - t0) * 1000)
    mean_latency_ms = float(np.mean(latencies))

    return {
        "architecture": architecture,
        "imagenet_pretrained": pretrained,
        "val_accuracy": acc,
        "macro_f1": macro_f1,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": round(model_size_mb, 2),
        "mean_inference_latency_ms": round(mean_latency_ms, 2),
        "train_seconds": round(train_seconds, 1),
        "per_class_report": report,
    }


# ============================================================
# Orchestration
# ============================================================

def run_benchmark(labels_csv="labels_table.csv", label_mapping_json="label_mapping.json",
                   architectures=("mobilenetv2", "efficientnetb0", "mobilenetv3small", "squeezenet"),
                   output_dir="."):
    df = pd.read_csv(labels_csv)
    with open(label_mapping_json) as f:
        label_to_index = json.load(f)

    class_names = sorted(label_to_index, key=label_to_index.get)
    num_classes = len(class_names)
    df["label_idx"] = df["label"].map(label_to_index)
    df = df.dropna(subset=["label_idx"]).reset_index(drop=True)
    df["label_idx"] = df["label_idx"].astype(int)

    train_df, val_df = train_test_split(
        df, test_size=0.15, stratify=df["label_idx"], random_state=RANDOM_STATE
    )

    results = []
    for arch in architectures:
        result = train_and_evaluate(arch, train_df, val_df, num_classes, class_names, output_dir)
        results.append(result)

    results_df = pd.DataFrame(results).drop(columns=["per_class_report"])
    results_df.to_csv(os.path.join(output_dir, "benchmark_results.csv"), index=False)
    print("\n" + results_df.to_string(index=False))

    _plot_comparison(results_df, output_dir)
    _plot_tradeoff(results_df, output_dir)

    return results_df, results


def _plot_comparison(results_df, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].bar(results_df["architecture"], results_df["val_accuracy"], color="#4F46E5")
    axes[0].set_title("Validation Accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(results_df["architecture"], results_df["total_params"] / 1e6, color="#06B6D4")
    axes[1].set_title("Total Parameters (millions)")
    axes[1].tick_params(axis="x", rotation=30)

    axes[2].bar(results_df["architecture"], results_df["mean_inference_latency_ms"], color="#22C55E")
    axes[2].set_title("Mean Inference Latency (ms/image, CPU)")
    axes[2].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "benchmark_comparison.png"), dpi=150)
    plt.close(fig)


def _plot_tradeoff(results_df, output_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        results_df["model_size_mb"], results_df["val_accuracy"],
        s=results_df["mean_inference_latency_ms"] * 8, alpha=0.6, color="#4F46E5",
    )
    for _, row in results_df.iterrows():
        ax.annotate(row["architecture"], (row["model_size_mb"], row["val_accuracy"]),
                    textcoords="offset points", xytext=(6, 6))
    ax.set_xlabel("Model size (MB)")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Accuracy vs. model size (bubble size = inference latency)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "benchmark_tradeoff.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run_benchmark()
