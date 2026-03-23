"""
dashboard/app.py
Streamlit dashboard for FinSight AI — real-time fraud detection + financial Q&A.
"""

import json
import joblib
import shap
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, ".")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinSight AI — Risk Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252a3a);
        border-radius: 12px; padding: 20px; margin: 8px 0;
        border-left: 4px solid #4f8ef7;
    }
    .fraud-alert {
        background: linear-gradient(135deg, #3d1010, #4a1515);
        border-radius: 12px; padding: 16px;
        border-left: 4px solid #ff4b4b;
    }
    .safe-alert {
        background: linear-gradient(135deg, #0d2e1a, #102a1e);
        border-radius: 12px; padding: 16px;
        border-left: 4px solid #00c47d;
    }
    .risk-critical { color: #ff4b4b; font-weight: bold; font-size: 1.4em; }
    .risk-high { color: #ff9f43; font-weight: bold; font-size: 1.4em; }
    .risk-medium { color: #ffd32a; font-weight: bold; font-size: 1.4em; }
    .risk-low { color: #00c47d; font-weight: bold; font-size: 1.4em; }
</style>
""", unsafe_allow_html=True)


# ── Load artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    artifacts = "models/artifacts"
    try:
        model = joblib.load(f"{artifacts}/model.pkl")
        explainer = joblib.load(f"{artifacts}/shap_explainer.pkl")
        with open(f"{artifacts}/metadata.json") as f:
            metadata = json.load(f)
        return model, explainer, metadata
    except Exception as e:
        st.error(f"Model not found. Run `python models/train.py` first.\n{e}")
        return None, None, None


@st.cache_resource
def load_rag():
    try:
        import yaml
        with open("configs/config.yaml") as f:
            config = yaml.safe_load(f)
        from rag.engine import get_engine
        return get_engine(config)
    except Exception as e:
        return None


def predict_transaction(tx_dict, model, explainer, metadata):
    from data.pipeline import preprocess
    row = pd.DataFrame([tx_dict])
    row["isFraud"] = 0
    X, _, _ = preprocess(row, fit=False, artifacts_path="data/processed/")
    proba = float(model.predict_proba(X)[:, 1][0])
    threshold = metadata["threshold"]
    is_fraud = proba >= threshold
    risk_level = (
        "CRITICAL" if proba >= 0.85 else
        "HIGH" if proba >= 0.65 else
        "MEDIUM" if proba >= 0.40 else "LOW"
    )
    sv = explainer.shap_values(X)[0]
    feature_cols = metadata["feature_cols"]
    shap_pairs = sorted(zip(feature_cols, sv), key=lambda x: abs(x[1]), reverse=True)
    return proba, is_fraud, risk_level, shap_pairs[:10], X, sv, feature_cols


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank.png", width=60)
    st.title("FinSight AI")
    st.caption("Financial Risk Intelligence Platform")
    st.divider()
    page = st.radio("Navigate", ["🔍 Fraud Detection", "💬 Financial Q&A", "📊 Model Analytics"])
    st.divider()

    model, explainer, metadata = load_artifacts()
    if metadata:
        st.markdown("**Model Status**")
        st.success("✅ Model Loaded")
        col1, col2 = st.columns(2)
        col1.metric("AUC", metadata["test_metrics"]["roc_auc"])
        col2.metric("F1", metadata["test_metrics"]["f1"])
        st.caption(f"v{metadata['model_version']} | Threshold: {metadata['threshold']}")


# ── Page: Fraud Detection ──────────────────────────────────────────────────────
if page == "🔍 Fraud Detection":
    st.title("🔍 Real-Time Fraud Detection")
    st.caption("Submit a transaction to get an instant fraud risk assessment with SHAP explanations")

    tab1, tab2 = st.tabs(["Single Transaction", "Batch Simulation"])

    with tab1:
        with st.form("tx_form"):
            st.subheader("Transaction Details")
            col1, col2, col3 = st.columns(3)

            with col1:
                amt = st.number_input("Transaction Amount ($)", min_value=0.01, value=250.00, step=0.01)
                product = st.selectbox("Product Code", ["W", "H", "C", "S", "R"])
                card_type = st.selectbox("Card Type", ["credit", "debit"])
                card_bank = st.selectbox("Card Bank", ["chase", "wells_fargo", "bofa", "citi", "capital_one", "other"])

            with col2:
                device = st.selectbox("Device Type", ["desktop", "mobile", "tablet"])
                email = st.selectbox("Email Domain", ["gmail", "yahoo", "hotmail", "outlook", "protonmail", "other"])
                hour = st.slider("Hour of Day", 0, 23, 14)
                addr_match = st.selectbox("Address Match", [1, 0], format_func=lambda x: "Yes" if x else "No")

            with col3:
                dist = st.number_input("Distance (km)", min_value=0.0, value=12.0)
                count_1d = st.number_input("Tx Count (1 day)", min_value=0, value=2)
                count_7d = st.number_input("Tx Count (7 days)", min_value=0, value=5)
                velocity = st.slider("Velocity Score", 0.0, 1.0, 0.15)

            submitted = st.form_submit_button("🔎 Analyze Transaction", use_container_width=True)

        if submitted and model:
            tx = {
                "TransactionAmt": amt, "ProductCD": product, "card_type": card_type,
                "card_bank": card_bank, "device_type": device, "email_domain": email,
                "hour_of_day": hour, "addr_match": addr_match, "dist_km": dist,
                "tx_count_1d": count_1d, "tx_count_7d": count_7d, "velocity_score": velocity
            }

            with st.spinner("Analyzing transaction..."):
                proba, is_fraud, risk_level, shap_pairs, X, sv, feature_cols = predict_transaction(
                    tx, model, explainer, metadata
                )

            st.divider()
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Fraud Probability", f"{proba:.1%}")
            col2.metric("Risk Level", risk_level)
            col3.metric("Decision", "⚠️ FRAUD" if is_fraud else "✅ LEGITIMATE")
            col4.metric("Threshold", f"{metadata['threshold']:.2f}")

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                title={"text": "Fraud Risk Score", "font": {"size": 18}},
                number={"suffix": "%", "font": {"size": 28}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#ff4b4b" if is_fraud else "#00c47d"},
                    "steps": [
                        {"range": [0, 40], "color": "#0d2e1a"},
                        {"range": [40, 65], "color": "#2e2a00"},
                        {"range": [65, 85], "color": "#3d1a00"},
                        {"range": [85, 100], "color": "#3d0000"},
                    ],
                    "threshold": {"line": {"color": "white", "width": 3}, "value": metadata["threshold"] * 100}
                }
            ))
            fig.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)", font_color="white")
            st.plotly_chart(fig, use_container_width=True)

            # SHAP waterfall
            st.subheader("🧠 SHAP Explainability — Why This Decision?")
            fig2, ax = plt.subplots(figsize=(10, 5))
            fig2.patch.set_facecolor("#0e1117")
            ax.set_facecolor("#0e1117")
            features = [p[0] for p in shap_pairs]
            values = [p[1] for p in shap_pairs]
            colors = ["#ff4b4b" if v > 0 else "#00c47d" for v in values]
            ax.barh(features[::-1], values[::-1], color=colors[::-1])
            ax.axvline(0, color="white", linewidth=0.8, alpha=0.5)
            ax.set_xlabel("SHAP Value (impact on fraud probability)", color="white")
            ax.tick_params(colors="white")
            ax.spines["bottom"].set_color("#444")
            ax.spines["left"].set_color("#444")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig2)

            st.caption("🔴 Red = increases fraud risk | 🟢 Green = decreases fraud risk")

    with tab2:
        st.subheader("Batch Transaction Simulation")
        n_sim = st.slider("Number of transactions to simulate", 50, 500, 200)
        if st.button("▶ Run Simulation", use_container_width=True) and model:
            from data.pipeline import generate_synthetic_transactions, preprocess
            with st.spinner(f"Simulating {n_sim} transactions..."):
                df_sim = generate_synthetic_transactions(n_samples=n_sim, random_state=99)
                X_sim, y_sim, _ = preprocess(df_sim, fit=False, artifacts_path="data/processed/")
                probas = model.predict_proba(X_sim)[:, 1]
                preds = (probas >= metadata["threshold"]).astype(int)

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Transactions", n_sim)
            col2.metric("Fraud Detected", int(preds.sum()), f"{preds.mean():.1%} rate")
            col3.metric("Avg Risk Score", f"{probas.mean():.3f}")

            fig = px.histogram(
                probas, nbins=50, title="Fraud Probability Distribution",
                labels={"value": "Fraud Probability", "count": "Transactions"},
                color_discrete_sequence=["#4f8ef7"]
            )
            fig.add_vline(x=metadata["threshold"], line_dash="dash", line_color="red",
                          annotation_text=f"Threshold: {metadata['threshold']}")
            fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="white")
            st.plotly_chart(fig, use_container_width=True)


# ── Page: Financial Q&A ────────────────────────────────────────────────────────
elif page == "💬 Financial Q&A":
    st.title("💬 Financial Risk Q&A")
    st.caption("Ask questions about fraud patterns, AML compliance, credit risk, and market risk — powered by RAG over financial documents")

    rag = load_rag()

    example_questions = [
        "What are the key indicators of fraudulent transactions?",
        "What are the SAR filing thresholds under BSA?",
        "What fraud loss projections exist for 2024?",
        "What are the AML transaction monitoring red flags?",
        "How does credit risk relate to DTI ratio?",
    ]

    st.subheader("Example Questions")
    cols = st.columns(len(example_questions))
    selected_q = None
    for i, q in enumerate(example_questions):
        if cols[i].button(q[:35] + "...", key=f"eq_{i}"):
            selected_q = q

    st.divider()
    question = st.text_area(
        "Your Question",
        value=selected_q or "",
        placeholder="Ask about fraud detection, AML compliance, credit risk...",
        height=80
    )

    if st.button("🔍 Ask FinSight AI", use_container_width=True) and question:
        if rag is None:
            st.warning("RAG engine not available. Make sure Ollama is running: `ollama serve`")
        else:
            with st.spinner("Searching financial documents..."):
                result = rag.query(question)

            st.subheader("Answer")
            st.markdown(result["answer"])

            if result["sources"]:
                st.divider()
                st.caption(f"📄 Sources: {', '.join(result['sources'])}")


# ── Page: Model Analytics ──────────────────────────────────────────────────────
elif page == "📊 Model Analytics":
    st.title("📊 Model Analytics")

    if metadata is None:
        st.warning("Train the model first: `python models/train.py`")
    else:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("ROC-AUC", metadata["test_metrics"]["roc_auc"])
        col2.metric("Avg Precision", metadata["test_metrics"]["avg_precision"])
        col3.metric("F1 Score", metadata["test_metrics"]["f1"])
        col4.metric("Precision", metadata["test_metrics"]["precision"])
        col5.metric("Recall", metadata["test_metrics"]["recall"])

        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Top Features (SHAP)")
            shap_path = "models/artifacts/shap_importance.csv"
            if Path(shap_path).exists():
                imp_df = pd.read_csv(shap_path).head(12)
                fig = px.bar(
                    imp_df, x="shap_importance", y="feature", orientation="h",
                    title="Mean |SHAP| Feature Importance",
                    color="shap_importance", color_continuous_scale="Blues"
                )
                fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                                   font_color="white", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("SHAP Summary Plot")
            shap_img = "models/artifacts/shap_summary.png"
            if Path(shap_img).exists():
                st.image(shap_img)

        st.divider()
        st.subheader("Model Metadata")
        st.json({
            "version": metadata["model_version"],
            "run_id": metadata["run_id"],
            "threshold": metadata["threshold"],
            "top_features": metadata["top_features"],
        })
