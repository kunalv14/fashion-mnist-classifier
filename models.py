"""Model architectures compared in this project.

Every builder takes a `name` and returns an uncompiled keras.Sequential model
that maps a (28, 28, 1) image to 10 class probabilities.
"""
from tensorflow import keras
from tensorflow.keras import layers

INPUT_SHAPE = (28, 28, 1)
NUM_CLASSES = 10


def dense_baseline(name="dense_baseline"):
    """One hidden dense layer of 64 sigmoid neurons."""
    return keras.Sequential([
        keras.Input(shape=INPUT_SHAPE),
        layers.Flatten(),
        layers.Dense(64, activation="sigmoid"),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name=name)


def dense_improved(name="dense_improved"):
    """Two hidden dense layers (256 and 128 neurons) with ReLU."""
    return keras.Sequential([
        keras.Input(shape=INPUT_SHAPE),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name=name)


def cnn_baseline(name="cnn_baseline"):
    """One Conv2D layer with 16 filters, then a 64-neuron dense layer."""
    return keras.Sequential([
        keras.Input(shape=INPUT_SHAPE),
        layers.Conv2D(16, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name=name)


def cnn_two_conv(name="cnn_two_conv"):
    """Two Conv2D layers (32 and 64 filters), no padding, 64-neuron dense layer."""
    return keras.Sequential([
        keras.Input(shape=INPUT_SHAPE),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name=name)


def cnn_final(name="cnn_final"):
    """Two padded Conv2D layers, a 128-neuron dense layer and dropout. Best model."""
    return keras.Sequential([
        keras.Input(shape=INPUT_SHAPE),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name=name)


BUILDERS = {
    "dense_baseline": dense_baseline,
    "dense_improved": dense_improved,
    "cnn_baseline": cnn_baseline,
    "cnn_two_conv": cnn_two_conv,
    "cnn_final": cnn_final,
}
