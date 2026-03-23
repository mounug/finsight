"""
monitoring/drift.py
Data drift detection using Evidently — compares production data against
training reference to flag model degradation before it impacts business.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from datetime import datetime
import yaml

EVIDENTLY_AVAILABLE = False  # Disabled — incompatible with current environment


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def detect_drift_statistical(reference: pd.DataFrame, current: pd.DataFrame,
                               threshold: float = 0.1) -> dict:
    """
    Fallback drift detection using KS test and PSI.
    Works without Evidently.
    """
    from scipy import stats

    drift_results = {}
    numeric_cols = reference.select_dtypes(include=[np.number]).columns.tolist()
    if "isFraud" in numeric_cols:
        numeric_cols.remove("isFraud")

    drifted_features = []
    for col in numeric_cols:
        ref_vals = reference[col].dropna().values
        cur_vals = current[col].dropna().values
        if len(cur_vals) == 0:
            continue
        stat, p_value = stats.ks_2samp(ref_vals, cur_vals)
        is_drifted = p_value < 0.05
        drift_results[col] = {
            "ks_statistic": round(float(stat), 4),
            "p_value": round(float(p_value), 4),
            "drifted": is_drifted
        }
        if is_drifted:
            drifted_features.append(col)

    share_drifted = len(drifted_features) / len(numeric_cols) if numeric_cols else 0
    dataset_drift = share_drifted > threshold

    return {
        "dataset_drift_detected": dataset_drift,
        "share_drifted_features": round(share_drifted, 3),
        "drifted_features": drifted_features,
        "feature_drift": drift_results,
        "threshold": threshold,
        "method": "KS Test",
        "timestamp": datetime.utcnow().isoformat(),
    }


def run_drift_report(reference_path: str = "data/processed/reference.csv",
                     current_data: pd.DataFrame = None,
                     output_path: str = "monitoring/reports/",
                     n_sample: int = 1000) -> dict:
    """
    Run full drift detection report.
    Generates HTML report if Evidently is available, JSON summary always.
    """
    Path(output_path).mkdir(parents=True, exist_ok=True)

    if not Path(reference_path).exists():
        logger.warning("Reference data not found. Run training pipeline first.")
        return {}

    reference = pd.read_csv(reference_path)
    if "isFraud" in reference.columns:
        reference = reference.drop(columns=["isFraud"])

    if current_data is None:
        # Simulate production data with slight drift for demo
        logger.info("Generating simulated production data for drift demo...")
        from data.pipeline import generate_synthetic_transactions, preprocess
        import yaml
        config = load_config()
        df_current = generate_synthetic_transactions(n_samples=n_sample, random_state=999)
        # Inject drift: shift transaction amounts and velocity
        df_current["TransactionAmt"] *= np.random.uniform(1.1, 1.5, len(df_current))
        df_current["velocity_score"] += 0.15
        df_current["velocity_score"] = df_current["velocity_score"].clip(0, 1)
        current, _, _ = preprocess(df_current, fit=False, artifacts_path="data/processed/")
    else:
        current = current_data

    config = load_config()
    threshold = config["monitoring"]["drift_threshold"]

    if EVIDENTLY_AVAILABLE:
        try:
            report = Report(metrics=[DataDriftPreset(), DatasetDriftMetric()])
            report.run(reference_data=reference, current_data=current)
            report_path = f"{output_path}/drift_report_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
            report.save_html(report_path)
            logger.info(f"Evidently drift report saved: {report_path}")
            result = report.as_dict()
            drift_detected = result["metrics"][1]["result"]["dataset_drift"]
            summary = {
                "dataset_drift_detected": drift_detected,
                "method": "Evidently",
                "report_path": report_path,
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.warning(f"Evidently failed: {e}. Falling back to KS test.")
            summary = detect_drift_statistical(reference, current, threshold)
    else:
        summary = detect_drift_statistical(reference, current, threshold)

    # Save JSON summary
    summary_path = f"{output_path}/drift_summary_latest.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if summary.get("dataset_drift_detected"):
        logger.warning(f"⚠️  DRIFT DETECTED — {summary.get('share_drifted_features', '?'):.0%} of features drifted")
        if summary.get("drifted_features"):
            logger.warning(f"Drifted features: {summary['drifted_features']}")
    else:
        logger.info("✅ No significant drift detected")

    return summary


if __name__ == "__main__":
    summary = run_drift_report()
    print(json.dumps(summary, indent=2))
