"""Load and preprocess the Fashion-MNIST dataset."""
from tensorflow import keras

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
NUM_CLASSES = len(CLASS_NAMES)


def load_data():
    """Return ((x_train, y_train), (x_test, y_test)).

    Images are reshaped to (n, 28, 28, 1) and scaled to [0, 1].
    Labels are integers 0-9; one-hot encode them where a model needs it.
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    return (x_train, y_train), (x_test, y_test)
