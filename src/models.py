"""
models.py
---------
Train and evaluate classical ML regression models:
  - Linear Regression
  - Decision Tree (with cross-validation)
  - Random Forest (with GridSearchCV hyperparameter tuning)
  - K-Nearest Neighbours
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

logger = logging.getLogger(__name__)
FIGURES_DIR = Path("reports/figures")


@dataclass
class ModelResult:
    """Container for model evaluation metrics."""

    model_name: str
    mse: float
    rmse: float
    mae: float
    r2: float
    explained_variance: float = 0.0
    extra: dict = field(default_factory=dict)

    def log(self) -> None:
        logger.info(
            "[%s] MSE=%.4e | RMSE=%.4e | MAE=%.4e | R²=%.6f",
            self.model_name,
            self.mse,
            self.rmse,
            self.mae,
            self.r2,
        )

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "mse": self.mse,
            "rmse": self.rmse,
            "mae": self.mae,
            "r2": self.r2,
            "explained_variance": self.explained_variance,
        }


def _evaluate(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    **extra: Any,
) -> ModelResult:
    """Compute and return a ModelResult for given predictions."""
    mse = mean_squared_error(y_true, y_pred)
    result = ModelResult(
        model_name=model_name,
        mse=mse,
        rmse=np.sqrt(mse),
        mae=mean_absolute_error(y_true, y_pred),
        r2=r2_score(y_true, y_pred),
        explained_variance=explained_variance_score(y_true, y_pred),
        extra=extra,
    )
    result.log()
    return result


def _plot_actual_vs_predicted(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    result: ModelResult,
    save: bool = True,
) -> None:
    """Plot predicted vs. actual values with metric annotations."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, y_pred, alpha=0.5, s=15, color="#4C72B0")

    lims = [
        min(y_test.min(), y_pred.min()),
        max(y_test.max(), y_pred.max()),
    ]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect fit")
    ax.set_xlabel("Actual values")
    ax.set_ylabel("Predicted values")
    ax.set_title(
        f"{result.model_name}\nR²={result.r2:.4f}  RMSE={result.rmse:.2e}",
        fontsize=12,
    )
    ax.legend()
    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fname = result.model_name.lower().replace(" ", "_") + "_actual_vs_pred.png"
        fig.savefig(FIGURES_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Individual model trainers
# ---------------------------------------------------------------------------


def train_linear_regression(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    save_plot: bool = True,
) -> ModelResult:
    """Fit a Linear Regression model and evaluate on the test set."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    result = _evaluate("Linear Regression", y_test.values, y_pred)
    _plot_actual_vs_predicted(y_test.values, y_pred, result, save=save_plot)
    return result


def train_decision_tree(
    X: pd.DataFrame,
    y: pd.Series,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    cv_folds: int = 5,
    random_state: int = 0,
    save_plot: bool = True,
) -> ModelResult:
    """
    Fit a Decision Tree Regressor with cross-validation MSE reported.

    Parameters
    ----------
    X, y : full feature matrix and target (used for CV)
    X_train, X_test, y_train, y_test : train/test splits
    cv_folds : number of CV folds
    """
    regressor = DecisionTreeRegressor(random_state=random_state)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    # Cross-validation
    cv_scores = cross_val_score(
        DecisionTreeRegressor(random_state=random_state),
        X,
        y,
        cv=cv_folds,
        scoring="neg_mean_squared_error",
    )
    cv_mse = -cv_scores.mean()
    logger.info("Decision Tree CV MSE (k=%d): %.4e", cv_folds, cv_mse)

    result = _evaluate("Decision Tree", y_test.values, y_pred, cv_mse=cv_mse)
    _plot_actual_vs_predicted(y_test.values, y_pred, result, save=save_plot)
    return result


def train_random_forest(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    n_estimators_grid: list[int] | None = None,
    random_state: int = 0,
    save_plot: bool = True,
) -> ModelResult:
    """
    Fit a Random Forest Regressor with GridSearchCV over n_estimators.

    Parameters
    ----------
    n_estimators_grid : list of int
        Values of n_estimators to search. Defaults to [10, 50, 100, 200, 500].
    """
    if n_estimators_grid is None:
        n_estimators_grid = [10, 50, 100, 200, 500]

    rf = RandomForestRegressor(random_state=random_state)
    grid_search = GridSearchCV(
        rf,
        param_grid={"n_estimators": n_estimators_grid},
        cv=KFold(n_splits=5, shuffle=True, random_state=random_state),
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    best_n = grid_search.best_params_["n_estimators"]
    val_mse = -grid_search.best_score_
    logger.info("Random Forest best n_estimators=%d, val MSE=%.4e", best_n, val_mse)

    final_rf = RandomForestRegressor(n_estimators=best_n, random_state=random_state)
    final_rf.fit(X_train, y_train)
    y_pred = final_rf.predict(X_test)

    result = _evaluate(
        "Random Forest", y_test.values, y_pred, best_n_estimators=best_n, val_mse=val_mse
    )
    _plot_actual_vs_predicted(y_test.values, y_pred, result, save=save_plot)
    return result


def train_knn(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    n_neighbors: int = 3,
    save_plot: bool = True,
) -> ModelResult:
    """Fit a K-Nearest Neighbours Regressor."""
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    result = _evaluate("KNN", y_test.values, y_pred, n_neighbors=n_neighbors)
    _plot_actual_vs_predicted(y_test.values, y_pred, result, save=save_plot)
    return result


# ---------------------------------------------------------------------------
# Convenience: train all models
# ---------------------------------------------------------------------------


def train_all_models(
    X: pd.DataFrame,
    y: pd.Series,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    config: dict | None = None,
) -> pd.DataFrame:
    """
    Train all four ML models and return a comparison DataFrame.

    Parameters
    ----------
    config : dict, optional
        Config overrides (e.g. {'knn': {'n_neighbors': 5}}).

    Returns
    -------
    pd.DataFrame
        One row per model with MSE, RMSE, MAE, R² columns.
    """
    cfg = config or {}

    results = [
        train_linear_regression(X_train, X_test, y_train, y_test),
        train_decision_tree(X, y, X_train, X_test, y_train, y_test),
        train_random_forest(
            X_train,
            X_test,
            y_train,
            y_test,
            n_estimators_grid=cfg.get("random_forest", {}).get(
                "n_estimators_grid", [10, 50, 100, 200, 500]
            ),
        ),
        train_knn(
            X_train,
            X_test,
            y_train,
            y_test,
            n_neighbors=cfg.get("knn", {}).get("n_neighbors", 3),
        ),
    ]

    comparison = pd.DataFrame([r.to_dict() for r in results])
    logger.info("Model comparison:\n%s", comparison.to_string(index=False))
    return comparison
