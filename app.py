"""Gradio demo: classify a clothing image with the trained Fashion-MNIST CNN."""
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, ImageOps
from tensorflow import keras

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
HERE = Path(__file__).parent
model = keras.models.load_model(HERE / "fashion_cnn.keras")


def to_fashion_mnist(img: Image.Image) -> Image.Image:
    """Convert any image to Fashion-MNIST style: 28x28 grayscale,
    light clothing item centred on a black background."""
    img = ImageOps.exif_transpose(img).convert("L")
    if img.size == (28, 28):                 # already a Fashion-MNIST image
        return img

    arr = np.asarray(img, dtype=np.float32)
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    if border.mean() > 127:                  # light background -> invert it
        arr = 255.0 - arr
    img = Image.fromarray(arr.astype(np.uint8))

    bbox = img.point(lambda p: 255 if p > 40 else 0).getbbox()
    if bbox:                                 # crop to the item
        img = img.crop(bbox)

    scale = 28 / max(img.size)               # fit inside 28x28, keep aspect ratio
    new_size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    img = img.resize(new_size, Image.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(img, ((28 - img.width) // 2, (28 - img.height) // 2))
    return canvas


def predict(img):
    if img is None:
        raise gr.Error("Upload an image or pick an example first.")
    small = to_fashion_mnist(img)
    x = np.asarray(small, dtype=np.float32).reshape(1, 28, 28, 1) / 255.0
    probs = model.predict(x, verbose=0)[0]
    scores = {name: float(p) for name, p in zip(CLASS_NAMES, probs)}
    preview = small.resize((224, 224), Image.NEAREST)
    return scores, preview


examples = sorted(str(p) for p in (HERE / "examples").glob("*.png"))

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Clothing image"),
    outputs=[
        gr.Label(num_top_classes=3, label="Prediction"),
        gr.Image(type="pil", label="What the model sees (28 x 28)"),
    ],
    title="Fashion-MNIST clothing classifier",
    description=(
        "A convolutional neural network built with TensorFlow/Keras that sorts clothing "
        "images into 10 classes with 91.5% test accuracy. Try the examples, or upload a "
        "photo of a single item on a plain background (product photos work best). "
        "The image is converted to 28 x 28 grayscale before prediction."
    ),
    examples=examples or None,
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
