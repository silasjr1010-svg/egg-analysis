"""
CBAM: Convolutional Block Attention Module (Woo et al., 2018)
https://arxiv.org/abs/1807.06521

Drop-in attention block for TF/Keras CNNs (tested against TF 2.15,
matching requirements.txt). Apply it to the output feature map of a
backbone (e.g. MobileNetV2) before global pooling.

Usage:
    from cbam import cbam_block

    base_model = tf.keras.applications.MobileNetV2(
        include_top=False, input_shape=(224, 224, 3), weights='imagenet'
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = cbam_block(x, reduction_ratio=8, name='cbam_s1')
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model_s1 = tf.keras.Model(inputs, outputs)
"""

import tensorflow as tf
from tensorflow.keras import layers


def channel_attention(input_feature, reduction_ratio=8, name="channel_attn"):
    """Channel attention: squeeze spatial dims via avg+max pool, shared MLP, sigmoid gate."""
    channels = input_feature.shape[-1]

    shared_dense_one = layers.Dense(
        max(channels // reduction_ratio, 1),
        activation="relu",
        kernel_initializer="he_normal",
        use_bias=True,
        name=f"{name}_dense1",
    )
    shared_dense_two = layers.Dense(
        channels,
        kernel_initializer="he_normal",
        use_bias=True,
        name=f"{name}_dense2",
    )

    avg_pool = layers.GlobalAveragePooling2D(name=f"{name}_avgpool")(input_feature)
    avg_pool = layers.Reshape((1, 1, channels))(avg_pool)
    avg_pool = shared_dense_one(avg_pool)
    avg_pool = shared_dense_two(avg_pool)

    max_pool = layers.GlobalMaxPooling2D(name=f"{name}_maxpool")(input_feature)
    max_pool = layers.Reshape((1, 1, channels))(max_pool)
    max_pool = shared_dense_one(max_pool)
    max_pool = shared_dense_two(max_pool)

    attention = layers.Add(name=f"{name}_add")([avg_pool, max_pool])
    attention = layers.Activation("sigmoid", name=f"{name}_sigmoid")(attention)

    return layers.Multiply(name=f"{name}_scale")([input_feature, attention])


def spatial_attention(input_feature, kernel_size=7, name="spatial_attn"):
    """Spatial attention: pool across channel axis (avg+max), conv, sigmoid gate."""

    avg_pool = layers.Lambda(
        lambda x: tf.reduce_mean(x, axis=-1, keepdims=True),
        name=f"{name}_avgpool",
    )(input_feature)
    max_pool = layers.Lambda(
        lambda x: tf.reduce_max(x, axis=-1, keepdims=True),
        name=f"{name}_maxpool",
    )(input_feature)

    concat = layers.Concatenate(axis=-1, name=f"{name}_concat")([avg_pool, max_pool])

    attention = layers.Conv2D(
        filters=1,
        kernel_size=kernel_size,
        strides=1,
        padding="same",
        activation="sigmoid",
        kernel_initializer="he_normal",
        use_bias=False,
        name=f"{name}_conv",
    )(concat)

    return layers.Multiply(name=f"{name}_scale")([input_feature, attention])


def cbam_block(input_feature, reduction_ratio=8, kernel_size=7, name="cbam"):
    """Full CBAM: channel attention followed by spatial attention."""
    x = channel_attention(input_feature, reduction_ratio=reduction_ratio, name=f"{name}_channel")
    x = spatial_attention(x, kernel_size=kernel_size, name=f"{name}_spatial")
    return x
