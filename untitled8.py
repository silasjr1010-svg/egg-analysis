import subprocess, sys
subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "pillow", "pandas", "numpy", "matplotlib", "seaborn", "scikit-learn"
])
# ============================================================
# CELL 1: Imports
# ============================================================

import json
import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

print(f"TensorFlow version: {tf.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
print('Num GPUs Available:', len(tf.config.list_physical_devices('GPU')))

# ============================================================
# CELL 2: Load Dataset and Create Negative Samples
# ============================================================
from pathlib import Path
import json
import os
import pandas as pd

possible_paths = [
    Path('.'), # Changed from /kaggle/input/chicken-egg-analysis-dataset
]

base_dir = None
labels_file = None

print('Searching for dataset...')
for path in possible_paths:
    if path.exists():
        # Look for info.labels directly in the path
        if (path / 'info.labels').exists():
            labels_file = path / 'info.labels'
            base_dir = path # Base directory where 'training' and 'testing' are
            print(f'✓ Found dataset at: {base_dir}')
            break

if labels_file is None:
    raise FileNotFoundError('Could not find info.labels in current directory or /content')

with open(labels_file, 'r', errors='ignore') as f:
    labels_json = json.load(f)

records = []
for item in labels_json.get('files', []):
    raw_path = item.get('path', '').strip()
    # normalize Windows-style separators so this resolves on any OS
    raw_path = raw_path.replace('\\', '/')
    # strip a leading "./" prefix (not arbitrary leading '.'/'/' chars)
    if raw_path.startswith('./'):
        raw_path = raw_path[2:]
    rel_path = raw_path

    label_obj = item.get('label', {})

    if isinstance(label_obj, dict):
        label = label_obj.get('label', 'unknown')
    else:
        label = str(label_obj)

    full_path = base_dir / rel_path # Use base_dir for full path

    records.append({
        'rel_path': rel_path,
        'path': str(full_path),
        'label': str(label).lower().strip()
    })

df = pd.DataFrame(records)
df['exists'] = df['path'].apply(os.path.exists)
df = df[df['exists']].reset_index(drop=True)

print(f'Total valid egg images: {len(df)}')
print(df['label'].value_counts())

# ============================================================
# CELL 3: Generate Non-Egg Samples
# ============================================================
negative_samples_dir = './negative_samples' # Changed from /kaggle/working/negative_samples
os.makedirs(negative_samples_dir, exist_ok=True)

negative_paths = []
np.random.seed(42)

num_negatives = int(len(df) * 0.3)
sample_images = np.random.choice(
    df['path'].values,
    size=min(num_negatives, len(df)),
    replace=False
)

for idx, img_path in enumerate(sample_images):
    try:
        img = Image.open(img_path).convert('RGB')
        w, h = img.size

        crop_points = [
            (0, 0),
            (max(w - 112, 0), 0),
            (0, max(h - 112, 0)),
            (max(w - 112, 0), max(h - 112, 0))
        ]

        for corner_idx, (x, y) in enumerate(crop_points):
            if len(negative_paths) >= num_negatives:
                break

            crop = img.crop((x, y, x + 112, y + 112))
            save_path = os.path.join(negative_samples_dir, f'neg_{idx}_{corner_idx}.jpg') # Used os.path.join
            crop.save(save_path)
            negative_paths.append(save_path)

        if len(negative_paths) >= num_negatives:
            break

    except Exception:
        continue

print(f'Generated {len(negative_paths)} negative samples')

# ============================================================
# CELL 4: Create Stage 1 Dataset
# ============================================================
stage1_records = []

for _, row in df.iterrows():
    stage1_records.append({
        'path': row['path'],
        'is_egg': 1,
        'fertility': row['label']
    })

for neg_path in negative_paths:
    stage1_records.append({
        'path': neg_path,
        'is_egg': 0,
        'fertility': 'non-egg'
    })

stage1_df = pd.DataFrame(stage1_records)

print(stage1_df['is_egg'].value_counts())

stage1_df.to_csv('./stage1_detection.csv', index=False) # Changed path
df.to_csv('./labels_table.csv', index=False) # Changed path

# ============================================================
# CELL 5: Visualize Samples
# ============================================================
fig, axs = plt.subplots(3, 4, figsize=(12, 9))
axs = axs.flatten()

egg_samples = stage1_df[stage1_df['is_egg'] == 1].sample(6, random_state=1)
nonegg_samples = stage1_df[stage1_df['is_egg'] == 0].sample(6, random_state=1)
combined = pd.concat([egg_samples, nonegg_samples]).reset_index(drop=True)

for ax, (_, row) in zip(axs, combined.iterrows()):
    img = Image.open(row['path']).convert('RGB')
    ax.imshow(img.resize((224, 224)))

    if row['is_egg'] == 1:
        title = f"EGG: {row['fertility']}"
    else:
        title = 'NON-EGG'

    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.show()

# ============================================================
# CELL 6: TensorFlow Dataset Functions
# ============================================================
IMG_SIZE = 224
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE


def load_and_preprocess(path, label):
    img = tf.io.read_file(path)
    # decode_image (not decode_jpeg) handles both true JPEGs and the
    # PNG-encoded files that were renamed to .jpg during ingestion
    # (visible in labels_table.csv filenames like "...Small.png....jpg")
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


def make_train_ds(dataframe, label_col):
    paths = dataframe['path'].values
    labels = dataframe[label_col].values

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.shuffle(len(dataframe))
    ds = ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_val_ds(dataframe, label_col):
    paths = dataframe['path'].values
    labels = dataframe[label_col].values

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)
    return ds

# ============================================================
# CELL 7: Train Stage 1 Egg Detector
# ============================================================
train_s1, val_s1 = train_test_split(
    stage1_df,
    test_size=0.15,
    stratify=stage1_df['is_egg'],
    random_state=42
)

train_ds_s1 = make_train_ds(train_s1, 'is_egg')
val_ds_s1 = make_val_ds(val_s1, 'is_egg')

base_model_s1 = tf.keras.applications.MobileNetV2(
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    weights='imagenet'
)

base_model_s1.trainable = False

model_s1 = models.Sequential([
    base_model_s1,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

model_s1.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

ckpt_s1 = callbacks.ModelCheckpoint(
    './egg_detector_best.h5', # Changed path
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)

early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history_s1 = model_s1.fit(
    train_ds_s1,
    validation_data=val_ds_s1,
    epochs=5,
    callbacks=[ckpt_s1, early_stop]
)

# ============================================================
# CELL 8: Prepare Stage 2 Labels
# ============================================================
unique_labels = sorted(df['label'].unique())
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
index_to_label = {idx: label for label, idx in label_to_index.items()}

print(label_to_index)

df['label_idx'] = df['label'].map(label_to_index)

train_s2, val_s2 = train_test_split(
    df,
    test_size=0.15,
    stratify=df['label_idx'],
    random_state=42
)

train_ds_s2 = make_train_ds(train_s2, 'label_idx')
val_ds_s2 = make_val_ds(val_s2, 'label_idx')

# ============================================================
# CELL 9: Train Stage 2 Fertility Classifier
# ============================================================
num_classes = len(unique_labels)

base_model_s2 = tf.keras.applications.MobileNetV2(
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    weights='imagenet'
)

base_model_s2.trainable = False

model_s2 = models.Sequential([
    base_model_s2,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation='softmax')
])

model_s2.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

ckpt_s2 = callbacks.ModelCheckpoint(
    './fertility_classifier_best.h5', # Changed path
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)

history_s2 = model_s2.fit(
    train_ds_s2,
    validation_data=val_ds_s2,
    epochs=10,
    callbacks=[ckpt_s2, early_stop] # `early_stop` is defined in Cell 7, assuming it's still in scope.
)

# ============================================================
# CELL 10: Reload Best Models
# ============================================================
detector_model = tf.keras.models.load_model('./egg_detector_best.h5') # Changed path
classifier_model = tf.keras.models.load_model('./fertility_classifier_best.h5') # Changed path

# ============================================================
# CELL 11: Evaluate Stage 2 Classifier
# ============================================================
results_s2 = classifier_model.evaluate(val_ds_s2)
print(results_s2)

y_true = []
y_pred = []

for images, labels in val_ds_s2:
    preds = classifier_model.predict(images, verbose=0)
    pred_labels = np.argmax(preds, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(pred_labels)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print(classification_report(
    y_true,
    y_pred,
    target_names=unique_labels
))

# ============================================================
# CELL 12: Confusion Matrix
# ============================================================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    xticklabels=unique_labels,
    yticklabels=unique_labels
)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ============================================================
# CELL 13: Save Final Models
# ============================================================
detector_model.save('./egg_detector_final.h5') # Changed path
classifier_model.save('./fertility_classifier_final.h5') # Changed path

import json
with open('./label_mapping.json', 'w') as f: # Changed path
    json.dump(label_to_index, f)

print('Saved final models and label mapping')

# ============================================================
# CELL 14: Example Inference Function
# ============================================================
def predict_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.expand_dims(img, axis=0)

    egg_score = detector_model.predict(img, verbose=0)[0][0]

    if egg_score < 0.5:
        return {
            'prediction': 'non-egg',
            'egg_confidence': float(egg_score)
        }

    fertility_pred = classifier_model.predict(img, verbose=0)
    fertility_idx = int(np.argmax(fertility_pred))

    return {
        'prediction': index_to_label[fertility_idx],
        'egg_confidence': float(egg_score),
        'fertility_confidence': float(np.max(fertility_pred))
    }

# ============================================================
# CELL 15: Test Prediction
# ============================================================
sample_path = df.iloc[0]['path']
result = predict_image(sample_path)
print(result)

