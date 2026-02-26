"""
deep_learning.py
----------------
TensorFlow/Keras MLP for evaporation rate regression.

Note on architecture: The original notebook used a sigmoid activation on the
output layer, which caps predictions at [0, 1]. Since the target (mer) is
negative and not bounded, a linear activation is more appropriate.
Both options are available here for comparison.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
FIGURES_DIR = Path("reports/figures")


def build_mlp(input_dim: int, use_linear_output: bool = True):
    """
    Build a 3-layer MLP for regression.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    use_linear_output : bool
        If True (recommended for regression), uses linear activation on the
        output layer. If False, uses sigmoid (matches original notebook).

    Returns
    -------
    tf.keras.Model
        Compiled Keras model.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.models import Sequential
    except ImportError:
        raise ImportError("TensorFlow is not installed. Run: pip install tensorflow")

    output_activation = "linear" if use_linear_output else "sigmoid"

    model = Sequential(
        [
            Dense(12, input_dim=input_dim, activation="relu"),
            Dense(8, activation="relu"),
            Dense(1, activation=output_activation),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=["mse", "mae"],
    )
    logger.info("Built MLP: input_dim=%d, output_activation=%s", input_dim, output_activation)
    return model


def train_mlp(
    X: pd.DataFrame,
    y: pd.Series,
    epochs: int = 50,
    batch_size: int = 32,
    use_linear_output: bool = True,
    save_plot: bool = True,
) -> dict:
    """
    Train the MLP and return training history + final metrics.

    Parameters
    ----------
    X : pd.DataFrame
        Full feature matrix (no train/test split — matches original notebook).
    y : pd.Series
        Target vector.
    epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size.
    use_linear_output : bool
        See build_mlp().

    Returns
    -------
    dict with keys: 'model', 'history', 'final_mse', 'final_mae'
    """
    X_arr = X.values.astype(np.float32)
    y_arr = y.values.astype(np.float32)

    model = build_mlp(input_dim=X_arr.shape[1], use_linear_output=use_linear_output)

    history = model.fit(
        X_arr,
        y_arr,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    eval_results = model.evaluate(X_arr, y_arr, verbose=0)
    # eval_results = [loss, mse, mae]
    final_mse = eval_results[1]
    final_mae = eval_results[2]
    logger.info("MLP Final MSE: %.4e | MAE: %.4e", final_mse, final_mae)

    if save_plot:
        _plot_training_history(history, save=True)

    return {
        "model": model,
        "history": history,
        "final_mse": final_mse,
        "final_mae": final_mae,
    }


def _plot_training_history(history, save: bool = True) -> None:
    """Plot MSE and MAE training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["mse"])
    axes[0].set_title("Training MSE over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].set_yscale("log")

    axes[1].plot(history.history["mae"], color="orange")
    axes[1].set_title("Training MAE over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].set_yscale("log")

    plt.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "mlp_training_history.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
