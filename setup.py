"""
setup.py
One-command setup: installs dependencies, trains model, builds RAG index.
"""

import subprocess
import sys
from pathlib import Path
from loguru import logger


def run(cmd: str, desc: str = ""):
    logger.info(f"▶ {desc or cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if result.returncode != 0:
        logger.error(f"Failed: {cmd}")
        sys.exit(1)


def main():
    logger.info("=" * 60)
    logger.info("  FinSight AI — Setup & Initialization")
    logger.info("=" * 60)

    # 1. Install dependencies
    run("pip install -r requirements.txt -q", "Installing Python dependencies...")

    # 2. Train model
    logger.info("\n📊 Step 1/3 — Training fraud detection model...")
    run("python models/train.py", "Training XGBoost + SHAP + MLflow")

    # 3. Build RAG index
    logger.info("\n📚 Step 2/3 — Building RAG index over financial documents...")
    run("python rag/engine.py", "Embedding documents + building FAISS index")

    # 4. Run drift check
    logger.info("\n📡 Step 3/3 — Running baseline drift check...")
    run("python monitoring/drift.py", "Drift monitoring baseline")

    logger.info("\n" + "=" * 60)
    logger.info("  ✅ Setup complete!")
    logger.info("=" * 60)
    logger.info("\nStart the platform:")
    logger.info("  API:       uvicorn api.main:app --reload --port 8000")
    logger.info("  Dashboard: streamlit run dashboard/app.py")
    logger.info("  MLflow:    mlflow ui --backend-store-uri mlruns")
    logger.info("\nMake sure Ollama is running for RAG:")
    logger.info("  ollama serve")
    logger.info("  ollama pull llama3")


if __name__ == "__main__":
    main()
