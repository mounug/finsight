"""
rag/engine.py
RAG pipeline over financial documents using Ollama (local LLM) + FAISS vector store.
Answers financial risk questions grounded in document context.
"""

import os
from pathlib import Path
from loguru import logger
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import yaml


FINANCE_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a senior financial risk analyst. Use the following context from financial documents to answer the question accurately and concisely. If the answer is not in the context, say so clearly — do not fabricate.

Context:
{context}

Question: {question}

Answer (be specific, reference data points where available):"""
)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def seed_documents(docs_path: str):
    """Create sample financial risk documents if none exist."""
    Path(docs_path).mkdir(parents=True, exist_ok=True)
    docs = {
        "fraud_risk_overview.txt": """
FINANCIAL FRAUD RISK OVERVIEW — FY2024

1. FRAUD LANDSCAPE
Payment fraud losses in the US reached $10.9 billion in 2023, a 14% increase YoY.
Card-not-present (CNP) fraud accounts for 73% of all card fraud incidents.
Identity theft-related fraud grew 22% driven by synthetic identity schemes.

2. HIGH-RISK TRANSACTION PATTERNS
- Transactions between 12AM–5AM carry 4.2x higher fraud probability
- Cross-border transactions exceeding $500 show elevated risk profiles
- Velocity anomalies: >5 transactions within 1 hour flagged as suspicious
- Email domains using temporary/disposable providers correlate with 67% higher fraud rates
- Address mismatch between billing and shipping increases fraud likelihood by 3.1x

3. FRAUD DETECTION METRICS — INDUSTRY BENCHMARKS
- False positive rate target: <2% to maintain customer experience
- Recall (fraud catch rate) benchmark: >85% for tier-1 institutions
- Average precision benchmark: >0.72 for production models
- Model refresh cadence: quarterly minimum, monthly recommended

4. REGULATORY REQUIREMENTS
- PCI DSS requires real-time transaction monitoring for all card-present transactions
- BSA/AML mandates SAR filing within 30 days of suspicious activity detection
- CFPB Regulation E requires fraud dispute resolution within 10 business days
- GDPR/CCPA compliance required for customer data used in model training

5. MODEL GOVERNANCE
- All fraud models must include explainability reports (SHAP/LIME) for regulatory review
- Concept drift monitoring mandatory with documented thresholds
- Model validation by independent risk team required before production deployment
        """,
        "credit_risk_report.txt": """
CREDIT RISK INTELLIGENCE REPORT — Q3 2024

EXECUTIVE SUMMARY
Consumer credit risk indicators show mixed signals heading into Q4 2024.
Delinquency rates on credit cards rose to 3.2%, highest since Q2 2019.
Auto loan delinquencies reached 1.8%, driven by subprime segment pressure.

KEY RISK INDICATORS
- Debt-to-income ratios above 43% are primary predictor of default within 12 months
- FICO scores below 620 associated with 8.7x higher default probability vs prime segment
- Revolving utilization above 80% increases default risk by 340 basis points
- Recent hard inquiries (>3 in 90 days) correlate with 2.4x elevated risk

SECTOR EXPOSURE
- Consumer discretionary credit exposure: $2.1T nationally
- Charge-off rate forecast: 3.8% for 2024, up from 2.9% in 2023
- Recovery rates on charged-off debt averaging 18 cents on the dollar

MACROECONOMIC FACTORS
- Federal funds rate at 5.25–5.50% increases refinancing stress for variable-rate borrowers
- Unemployment rate at 3.9% — models should scenario-plan for 5.5% by Q2 2025
- Housing equity cushion declining: HELOCs at risk if home values drop >8%

RISK MITIGATION RECOMMENDATIONS
1. Tighten origination criteria for DTI >40% applicants
2. Implement enhanced monitoring for accounts showing velocity increases
3. Increase loss reserve provisions by 15–20% for subprime portfolio
4. Deploy early warning system triggers at 60-day delinquency vs 90-day current standard
        """,
        "aml_compliance.txt": """
ANTI-MONEY LAUNDERING (AML) COMPLIANCE FRAMEWORK

OVERVIEW
Financial institutions must maintain robust AML programs under the Bank Secrecy Act (BSA),
USA PATRIOT Act, and FinCEN guidance. Non-compliance penalties averaged $1.4B per institution
in 2023, with criminal referrals in 12 major cases.

TRANSACTION MONITORING RULES
Structuring Indicators:
- Multiple cash deposits just below $10,000 CTR threshold (smurfing)
- Round-dollar transactions in rapid succession
- Transactions immediately followed by withdrawals of similar amounts

High-Risk Geographies (FATF Grey/Black List):
- Transactions involving sanctioned countries trigger automatic OFAC screening
- Correspondent banking from high-risk jurisdictions require enhanced due diligence
- PEP (Politically Exposed Person) transactions require additional approval layers

SUSPICIOUS ACTIVITY REPORT (SAR) THRESHOLDS
- Transactions of $5,000+ with known/suspected illicit activity: mandatory SAR
- Insider abuse of any amount: mandatory SAR
- Computer intrusion affecting $5,000+: mandatory SAR
- Tipping off a customer about SAR filing is a federal offense

ML MODEL REQUIREMENTS FOR AML
- Models must achieve <1% false negative rate on known SAR patterns
- Explainability required: rule-based reasoning must accompany ML scores
- Human-in-the-loop review mandatory for all SAR candidates
- Model documentation must include training data provenance and bias assessment
- Annual independent model validation required by OCC/Fed guidelines

KYC INTEGRATION
- CIP (Customer Identification Program) data must feed model features
- Beneficial ownership verification required for legal entity customers
- Ongoing monitoring must update risk scores with new KYC events
        """,
        "market_risk_factors.txt": """
MARKET RISK FACTORS — FINANCIAL SERVICES 2024

SYSTEMIC RISK INDICATORS
VIX Volatility Index: elevated above 20 signals heightened market stress.
Credit spreads on high-yield debt: widening >400bps historically precedes credit events.
Inverted yield curve duration: current inversion at 18 months — longest since 1980.

INTEREST RATE RISK
- Every 100bps rate increase reduces bond portfolio value by ~7% on 7-year duration
- Variable-rate loan books face accelerating prepayment risk in falling rate environment
- Net interest margin (NIM) compression risk for institutions with short asset duration

LIQUIDITY RISK
- LCR (Liquidity Coverage Ratio) must exceed 100% under Basel III
- NSFR (Net Stable Funding Ratio) benchmark: >100% for systemically important banks
- Concentration risk: no single depositor should exceed 10% of funding base
- Uninsured deposit ratio >40% flagged as elevated run risk (per SVB post-mortem)

OPERATIONAL RISK
- Cyber incidents affecting financial data: average cost $5.9M per incident in 2023
- Third-party/vendor concentration risk increasingly scrutinized by OCC
- Model risk: SR 11-7 guidance requires annual validation for all models in production

FRAUD LOSS PROJECTIONS
- Account takeover fraud: projected $7.2B in losses for 2024
- Synthetic identity fraud: $6B+ annually, growing 30% YoY
- Real-time payment fraud: ACH and RTP networks seeing 45% increase in fraud attempts
- Authorized push payment (APP) fraud: $1.7B in 2023, new Reg E protections proposed
        """
    }

    for filename, content in docs.items():
        filepath = Path(docs_path) / filename
        if not filepath.exists():
            filepath.write_text(content.strip())
            logger.info(f"Created document: {filename}")


class RAGEngine:
    def __init__(self, config: dict):
        self.config = config["rag"]
        self.docs_path = self.config["docs_path"]
        self.index_path = self.config["faiss_index_path"]
        self.embedding_model_name = self.config["embedding_model"]
        self.ollama_model = self.config["ollama_model"]
        self.top_k = self.config["top_k"]

        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None

    def _load_embeddings(self):
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

    def build_index(self):
        """Ingest documents, chunk, embed, and build FAISS index."""
        seed_documents(self.docs_path)
        self._load_embeddings()

        logger.info(f"Loading documents from {self.docs_path}")
        loader = DirectoryLoader(self.docs_path, glob="*.txt", loader_cls=TextLoader)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config["chunk_size"],
            chunk_overlap=self.config["chunk_overlap"]
        )
        chunks = splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")

        logger.info("Building FAISS index...")
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        Path(self.index_path).mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(self.index_path)
        logger.info(f"FAISS index saved to {self.index_path}")
        self._build_chain()

    def load_index(self):
        """Load existing FAISS index."""
        if not Path(self.index_path).exists():
            logger.info("No index found — building from scratch...")
            self.build_index()
            return
        self._load_embeddings()
        self.vectorstore = FAISS.load_local(
            self.index_path, self.embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info(f"Loaded FAISS index from {self.index_path}")
        self._build_chain()

    def _build_chain(self):
        """Build the RAG chain with Ollama using modern LCEL syntax."""
        try:
            llm = OllamaLLM(model=self.ollama_model, temperature=0.1)
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.top_k}
            )
            self.qa_chain = (
                {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
                 "question": RunnablePassthrough()}
                | FINANCE_PROMPT
                | llm
                | StrOutputParser()
            )
            logger.info(f"RAG chain ready (Ollama/{self.ollama_model})")
        except Exception as e:
            logger.warning(f"Ollama not available: {e}. RAG will return context only.")
            self.qa_chain = None

    def query(self, question: str) -> dict:
        """Query the RAG engine."""
        if not self.vectorstore:
            return {"answer": "Index not loaded.", "sources": [], "context": ""}

        # Always retrieve context
        docs = self.vectorstore.similarity_search(question, k=self.top_k)
        context = "\n\n---\n\n".join([d.page_content for d in docs])
        sources = list({d.metadata.get("source", "unknown") for d in docs})
        sources = [Path(s).name for s in sources]

        if self.qa_chain:
            try:
                answer = self.qa_chain.invoke(question)
            except Exception as e:
                logger.warning(f"LLM query failed: {e}. Returning context only.")
                answer = f"[LLM unavailable] Relevant context retrieved:\n\n{context[:800]}..."
        else:
            answer = f"Relevant context:\n\n{context[:800]}..."

        return {"answer": answer, "sources": sources, "context": context}


def get_engine(config: dict) -> RAGEngine:
    engine = RAGEngine(config)
    engine.load_index()
    return engine


if __name__ == "__main__":
    config = load_config()
    engine = get_engine(config)
    result = engine.query("What are the key indicators of fraudulent transactions?")
    print("\n=== RAG Response ===")
    print(result["answer"])
    print(f"\nSources: {result['sources']}")
