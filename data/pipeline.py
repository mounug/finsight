"""
data/pipeline.py
Generates realistic synthetic financial transaction data and preprocesses it
for fraud detection modeling.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from loguru import logger
import yaml
import joblib


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_synthetic_transactions(n_samples: int = 50000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate realistic synthetic financial transaction data.
    Mirrors the structure of IEEE-CIS fraud dataset with engineered features.
    """
    np.random.seed(random_state)
    logger.info(f"Generating {n_samples} synthetic transactions...")

    n_fraud = int(n_samples * 0.035)  # 3.5% fraud rate — realistic for financial data
    n_legit = n_samples - n_fraud

    def make_transactions(n, is_fraud):
        if is_fraud:
            amounts = np.random.exponential(scale=300, size=n)
            amounts = np.clip(amounts, 10, 15000)
            hour = np.random.choice(range(0, 6), size=n, p=[0.3, 0.25, 0.2, 0.1, 0.1, 0.05])
        else:
            amounts = np.random.lognormal(mean=4.5, sigma=1.2, size=n)
            amounts = np.clip(amounts, 1, 5000)
            hour = np.random.choice(range(24), size=n)

        product_cd = np.random.choice(["W", "H", "C", "S", "R"], size=n,
                                       p=[0.6, 0.15, 0.12, 0.08, 0.05] if not is_fraud
                                       else [0.2, 0.1, 0.4, 0.2, 0.1])
        card_type = np.random.choice(["credit", "debit"], size=n,
                                      p=[0.45, 0.55] if not is_fraud else [0.75, 0.25])
        card_bank = np.random.choice(["chase", "wells_fargo", "bofa", "citi", "capital_one", "other"],
                                      size=n)
        device = np.random.choice(["desktop", "mobile", "tablet"], size=n,
                                   p=[0.5, 0.4, 0.1] if not is_fraud else [0.2, 0.7, 0.1])
        email_domain = np.random.choice(["gmail", "yahoo", "hotmail", "outlook", "protonmail", "other"],
                                         size=n,
                                         p=[0.45, 0.2, 0.15, 0.1, 0.02, 0.08] if not is_fraud
                                         else [0.2, 0.15, 0.1, 0.08, 0.25, 0.22])
        addr_match = np.random.choice([0, 1], size=n, p=[0.05, 0.95] if not is_fraud else [0.45, 0.55])
        dist = np.random.exponential(scale=50 if not is_fraud else 300, size=n)
        count_1d = np.random.poisson(lam=2 if not is_fraud else 8, size=n)
        count_7d = np.random.poisson(lam=5 if not is_fraud else 20, size=n)
        velocity_score = np.random.uniform(0, 1, size=n)
        if is_fraud:
            velocity_score = np.clip(velocity_score + 0.4, 0, 1)

        return pd.DataFrame({
            "TransactionAmt": amounts.round(2),
            "ProductCD": product_cd,
            "card_type": card_type,
            "card_bank": card_bank,
            "device_type": device,
            "email_domain": email_domain,
            "hour_of_day": hour,
            "addr_match": addr_match,
            "dist_km": dist.round(1),
            "tx_count_1d": count_1d,
            "tx_count_7d": count_7d,
            "velocity_score": velocity_score.round(4),
            "isFraud": int(is_fraud),
        })

    df = pd.concat([make_transactions(n_legit, False), make_transactions(n_fraud, True)], ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    df.insert(0, "TransactionID", range(1, len(df) + 1))

    logger.info(f"Generated dataset: {len(df)} rows | Fraud rate: {df['isFraud'].mean():.2%}")
    return df


def preprocess(df: pd.DataFrame, fit: bool = True, artifacts_path: str = "data/processed/") -> tuple:
    """
    Feature engineering + encoding + scaling.
    fit=True for training, fit=False for inference (loads saved artifacts).
    """
    Path(artifacts_path).mkdir(parents=True, exist_ok=True)
    df = df.copy()

    # Feature engineering
    df["amt_log"] = np.log1p(df["TransactionAmt"])
    df["amt_velocity_ratio"] = df["TransactionAmt"] / (df["velocity_score"] + 1e-6)
    df["tx_count_ratio"] = df["tx_count_1d"] / (df["tx_count_7d"] + 1e-6)
    df["high_risk_hour"] = df["hour_of_day"].apply(lambda h: 1 if h < 6 else 0)
    df["high_dist"] = (df["dist_km"] > 200).astype(int)

    cat_cols = ["ProductCD", "card_type", "card_bank", "device_type", "email_domain"]
    num_cols = ["TransactionAmt", "amt_log", "amt_velocity_ratio", "tx_count_ratio",
                "dist_km", "velocity_score", "tx_count_1d", "tx_count_7d",
                "high_risk_hour", "high_dist", "addr_match", "hour_of_day"]

    if fit:
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        joblib.dump(encoders, f"{artifacts_path}/encoders.pkl")
        joblib.dump(scaler, f"{artifacts_path}/scaler.pkl")
        joblib.dump(num_cols + cat_cols, f"{artifacts_path}/feature_cols.pkl")
        logger.info("Saved preprocessing artifacts.")
    else:
        encoders = joblib.load(f"{artifacts_path}/encoders.pkl")
        scaler = joblib.load(f"{artifacts_path}/scaler.pkl")
        for col in cat_cols:
            le = encoders[col]
            df[col] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
        df[num_cols] = scaler.transform(df[num_cols])

    feature_cols = num_cols + cat_cols
    X = df[feature_cols]
    y = df["isFraud"] if "isFraud" in df.columns else None
    return X, y, feature_cols


def prepare_data(config: dict) -> dict:
    """Full data preparation pipeline."""
    raw_path = Path(config["data"]["raw_path"])
    processed_path = Path(config["data"]["processed_path"])
    processed_path.mkdir(parents=True, exist_ok=True)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        logger.info("No raw data found — generating synthetic dataset...")
        df = generate_synthetic_transactions(
            n_samples=config["data"]["synthetic_samples"],
            random_state=config["training"]["random_state"]
        )
        df.to_csv(raw_path, index=False)
    else:
        logger.info(f"Loading data from {raw_path}")
        df = pd.read_csv(raw_path)

    X, y, feature_cols = preprocess(df, fit=True, artifacts_path=str(processed_path))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"],
        stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=config["training"]["val_size"],
        random_state=config["training"]["random_state"],
        stratify=y_train
    )

    # Save reference data for drift monitoring
    X_train.copy().assign(isFraud=y_train.values).to_csv(
        processed_path / "reference.csv", index=False
    )

    logger.info(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "feature_cols": feature_cols,
    }


if __name__ == "__main__":
    config = load_config()
    data = prepare_data(config)
    logger.info("Data pipeline complete.")
