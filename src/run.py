"""
run.py
------
CLI entry point for the Namibia Evaporation ML pipeline.

Usage:
    python -m src.run                        # full pipeline
    python -m src.run --stage eda
    python -m src.run --stage preprocess
    python -m src.run --stage train
    python -m src.run --stage evaluate
    python -m src.run --config configs/config.yaml --stage train
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load YAML config, falling back to defaults if file not found."""
    path = Path(config_path)
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f)
    logger.warning("Config not found at %s; using defaults.", config_path)
    return {}


def run_pipeline(config: dict, stage: str = "all") -> None:
    from src.data_loader import get_features_and_target, load_data
    from src.eda import run_eda
    from src.models import train_all_models
    from src.preprocessing import scale_features, split_data

    data_cfg = config.get("data", {})
    preprocess_cfg = config.get("preprocessing", {})
    model_cfg = config.get("models", {})

    # ---------------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------------
    data_path = data_cfg.get("raw_path", "data/raw/north-namibian-monthly-soil.csv")
    df = load_data(data_path)

    if stage in ("eda", "all"):
        logger.info("=== Stage: EDA ===")
        run_eda(df, save=True)

    if stage in ("preprocess", "train", "evaluate", "all"):
        logger.info("=== Stage: Preprocessing ===")
        X, y = get_features_and_target(df)
        X_train, X_test, y_train, y_test = split_data(
            X,
            y,
            test_size=preprocess_cfg.get("test_size", 0.2),
            random_state=preprocess_cfg.get("random_state", 0),
        )
        scaler_type = preprocess_cfg.get("scaler", "standard")
        X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test, scaler_type)

    if stage in ("train", "evaluate", "all"):
        logger.info("=== Stage: Training ML Models ===")
        comparison = train_all_models(
            X,
            y,
            X_train_scaled,
            X_test_scaled,
            y_train,
            y_test,
            config=model_cfg,
        )
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)
        print(comparison.to_string(index=False))

        if model_cfg.get("run_deep_learning", False):
            logger.info("=== Stage: Deep Learning (TensorFlow) ===")
            from src.deep_learning import train_mlp

            result = train_mlp(
                X,
                y,
                epochs=model_cfg.get("deep_learning", {}).get("epochs", 50),
                use_linear_output=True,
            )
            logger.info("Deep Learning MSE: %.4e", result["final_mse"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Namibia Evaporation ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        choices=["eda", "preprocess", "train", "evaluate", "all"],
        default="all",
        help="Pipeline stage to run (default: all)",
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to YAML config file (default: configs/config.yaml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_pipeline(config, stage=args.stage)


if __name__ == "__main__":
    main()
