"""
models/train.py
Trains XGBoost fraud detection model with MLflow experiment tracking,
SHAP explainability, and model registry.
"""

import json
import joblib
import shap
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score
)
from loguru import logger
import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def evaluate(model, X, y, threshold: float = 0.5) -> dict:
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)
    return {
        "roc_auc": round(roc_auc_score(y, proba), 4),
        "avg_precision": round(average_precision_score(y, proba), 4),
        "f1": round(f1_score(y, preds), 4),
        "precision": round(precision_score(y, preds), 4),
        "recall": round(recall_score(y, preds), 4),
    }


def find_optimal_threshold(model, X_val, y_val) -> float:
    """Find threshold that maximizes F1 on validation set."""
    proba = model.predict_proba(X_val)[:, 1]
    best_thresh, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.9, 0.05):
        preds = (proba >= t).astype(int)
        f = f1_score(y_val, preds)
        if f > best_f1:
            best_f1, best_thresh = f, t
    logger.info(f"Optimal threshold: {best_thresh:.2f} (F1={best_f1:.4f})")
    return round(float(best_thresh), 2)


def compute_shap(model, X_train, X_test, feature_cols, save_path: str = "models/artifacts/"):
    """Compute SHAP values and save summary plot."""
    logger.info("Computing SHAP values...")
    Path(save_path).mkdir(parents=True, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    sample = X_test.sample(min(500, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(sample)

    # Summary plot
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, sample, feature_names=feature_cols,
                      show=False, plot_size=None)
    plt.tight_layout()
    plt.savefig(f"{save_path}/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Feature importance from SHAP
    mean_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "shap_importance": mean_shap
    }).sort_values("shap_importance", ascending=False)
    importance_df.to_csv(f"{save_path}/shap_importance.csv", index=False)

    joblib.dump(explainer, f"{save_path}/shap_explainer.pkl")
    logger.info(f"SHAP artifacts saved to {save_path}")
    return explainer, importance_df


def train(config: dict, data: dict) -> tuple:
    """Full training run with MLflow tracking."""
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    model_params = config["model"]["params"]
    artifacts_path = "models/artifacts"
    Path(artifacts_path).mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=f"xgb_fraud_{config['model']['version']}") as run:
        logger.info(f"MLflow run: {run.info.run_id}")
        mlflow.log_params(model_params)

        # Train
        logger.info("Training XGBoost model...")
        model = XGBClassifier(**model_params, verbosity=0)
        model.fit(
            data["X_train"], data["y_train"],
            eval_set=[(data["X_val"], data["y_val"])],
            verbose=False,
        )

        # Find optimal decision threshold
        threshold = find_optimal_threshold(model, data["X_val"], data["y_val"])

        # Evaluate
        train_metrics = evaluate(model, data["X_train"], data["y_train"], threshold)
        val_metrics = evaluate(model, data["X_val"], data["y_val"], threshold)
        test_metrics = evaluate(model, data["X_test"], data["y_test"], threshold)

        logger.info(f"Test ROC-AUC: {test_metrics['roc_auc']} | F1: {test_metrics['f1']} | "
                    f"Precision: {test_metrics['precision']} | Recall: {test_metrics['recall']}")

        # Log metrics
        for split, metrics in [("train", train_metrics), ("val", val_metrics), ("test", test_metrics)]:
            mlflow.log_metrics({f"{split}_{k}": v for k, v in metrics.items()})
        mlflow.log_metric("optimal_threshold", threshold)

        # Classification report
        proba = model.predict_proba(data["X_test"])[:, 1]
        preds = (proba >= threshold).astype(int)
        report = classification_report(data["y_test"], preds, output_dict=True)
        logger.info(f"\n{classification_report(data['y_test'], preds)}")

        # SHAP
        explainer, importance_df = compute_shap(
            model, data["X_train"], data["X_test"],
            data["feature_cols"], save_path=artifacts_path
        )
        mlflow.log_artifact(f"{artifacts_path}/shap_summary.png")
        mlflow.log_artifact(f"{artifacts_path}/shap_importance.csv")

        # Save model + artifacts
        mlflow.xgboost.log_model(model, "model")
        joblib.dump(model, f"{artifacts_path}/model.pkl")

        # Save metadata
        metadata = {
            "run_id": run.info.run_id,
            "model_version": config["model"]["version"],
            "threshold": threshold,
            "test_metrics": test_metrics,
            "feature_cols": data["feature_cols"],
            "top_features": importance_df.head(10)["feature"].tolist(),
        }
        with open(f"{artifacts_path}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        mlflow.log_artifact(f"{artifacts_path}/metadata.json")

        logger.info(f"Model saved. Run ID: {run.info.run_id}")
        return model, explainer, metadata


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data.pipeline import load_config, prepare_data

    config = load_config()
    data = prepare_data(config)
    model, explainer, metadata = train(config, data)
    logger.info("Training complete.")
    logger.info(f"Test metrics: {metadata['test_metrics']}")
