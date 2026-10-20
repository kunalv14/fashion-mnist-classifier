"""Train and evaluate the Fashion-MNIST experiments.

Examples
--------
Reproduce all seven experiments and save the best model:
    python src/train.py --experiments all

Train only the final CNN (about 1 minute on a GPU) for the demo app:
    python src/train.py --experiments cnn_final --export-examples app/examples
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tensorflow import keras

from data import CLASS_NAMES, NUM_CLASSES, load_data
from models import BUILDERS

OPTIMIZERS = {
    "adam": keras.optimizers.Adam,
    "sgd": keras.optimizers.SGD,
    "rmsprop": keras.optimizers.RMSprop,
}

# (experiment name, architecture, optimizer, learning rate)
EXPERIMENTS = [
    ("dense_baseline", "dense_baseline", "adam", 1e-3),
    ("dense_improved", "dense_improved", "adam", 1e-3),
    ("cnn_baseline", "cnn_baseline", "adam", 1e-3),
    ("cnn_two_conv", "cnn_two_conv", "adam", 1e-3),
    ("cnn_final", "cnn_final", "adam", 1e-3),
    ("cnn_final_sgd_lr0_01", "cnn_final", "sgd", 1e-2),
    ("cnn_final_adam_lr0_01", "cnn_final", "adam", 1e-2),
]


def run_experiment(name, arch, optimizer, lr, data, epochs, batch_size, seed):
    """Train one model on the training set and evaluate it on the test set."""
    (x_train, y_train), (x_test, y_test) = data
    keras.utils.set_random_seed(seed)
    model = BUILDERS[arch](name=name)
    model.compile(optimizer=OPTIMIZERS[optimizer](learning_rate=lr),
                  loss="categorical_crossentropy", metrics=["accuracy"])

    y_train_oh = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test_oh = keras.utils.to_categorical(y_test, NUM_CLASSES)

    start = time.time()
    history = model.fit(x_train, y_train_oh, validation_data=(x_test, y_test_oh),
                        epochs=epochs, batch_size=batch_size, verbose=2)
    seconds = time.time() - start
    test_loss, test_acc = model.evaluate(x_test, y_test_oh, verbose=0)

    result = {
        "experiment": name,
        "architecture": arch,
        "optimizer": optimizer,
        "learning_rate": lr,
        "parameters": model.count_params(),
        "test_accuracy": round(float(test_acc), 4),
        "test_loss": round(float(test_loss), 4),
        "train_seconds": round(seconds, 1),
    }
    print(f"{name}: test accuracy {test_acc:.4f}")
    return model, result, history.history


def export_examples(x_test, y_test, out_dir):
    """Save the first test image of each class as a PNG (used as demo examples)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for c, class_name in enumerate(CLASS_NAMES):
        i = int(np.where(y_test == c)[0][0])
        img = Image.fromarray((x_test[i].squeeze() * 255).astype("uint8"))
        fname = class_name.lower().replace("/", "_").replace(" ", "_")
        img.save(out_dir / f"{c}_{fname}.png")
    print(f"Saved {NUM_CLASSES} example images to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", nargs="+", default=["all"],
                        help="'all' or experiment names, e.g. cnn_final dense_baseline")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--model-out", default="models/fashion_cnn.keras",
                        help="where to save the most accurate model")
    parser.add_argument("--export-examples", default=None,
                        help="folder to save one test image per class")
    args = parser.parse_args()

    known = {e[0]: e for e in EXPERIMENTS}
    if args.experiments == ["all"]:
        selected = EXPERIMENTS
    else:
        unknown = [n for n in args.experiments if n not in known]
        if unknown:
            parser.error(f"unknown experiments {unknown}; choose from {list(known)}")
        selected = [known[n] for n in args.experiments]

    data = load_data()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    results, histories, best = [], {}, (None, -1.0)
    for name, arch, optimizer, lr in selected:
        model, result, history = run_experiment(name, arch, optimizer, lr, data,
                                                args.epochs, args.batch_size, args.seed)
        results.append(result)
        histories[name] = history
        if result["test_accuracy"] > best[1]:
            best = (model, result["test_accuracy"])

    ranking = pd.DataFrame(results).sort_values("test_accuracy", ascending=False)
    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    ranking.to_csv(results_dir / "results.csv", index=False)
    with open(results_dir / "histories.json", "w") as f:
        json.dump({k: {m: [float(v) for v in vals] for m, vals in h.items()}
                   for k, h in histories.items()}, f, indent=2)
    print(ranking.to_string(index=False))

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    best[0].save(args.model_out)
    print(f"Saved best model ({best[0].name}, {best[1]:.4f}) to {args.model_out}")

    if args.export_examples:
        export_examples(data[1][0], data[1][1], args.export_examples)


if __name__ == "__main__":
    main()
