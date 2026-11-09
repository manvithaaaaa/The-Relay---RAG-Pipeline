# The Relay - 4-Step RAG Pipeline & Load Simulator

A modular 4-step Retrieval-Augmented Generation (RAG) pipeline built in Python with local rate-limit simulation, citation verification, and stress/retrieval benchmarking. Evaluated on arXiv research papers using **Qwen 2.5/3.8** (`qwen/qwen3.8-27b`) via the Groq API.

---

## 📌 Features

- **4-Step RAG Architecture:**
  1. **Intent Guardrail:** Filters unanswerable out-of-scope queries early to preserve token capacity.
  2. **BM25 Paper Retrieval:** Dynamic passage retrieval over local research PDFs.
  3. **Citation Generation:** Forces structured responses containing strict chunk ID citations.
  4. **Verification & Adaptive Branching:** Validates citation integrity; expands retrieval window ($k$) upon citation mismatch or issues controlled refusals.
- **Local Rate-Limit Simulator (`StandinModelService`):** Token bucket and request window simulator calibrated against real Groq API quotas (RPM, TPM, RPD) to model HTTP 429 backpressure.
- **Unified Experiment Logger:** Automatically outputs latency (TTFT / Total MS), retry counts, token usage, and answer accuracy grades across all stress scenarios into `results.csv`.

---

## 📁 Repository Structure

```text
Assignment ML/
├── data/
│   ├── attention.pdf           # Attention Is All You Need (arXiv:1706.03762)
│   ├── dynamic_chunking.pdf    # Dynamic Chunking in Sequence Modeling (arXiv:2507.07955)
│   └── moshi.pdf               # Moshi Speech Language Model (arXiv:2410.00037)
├── src/
│   ├── groq_client.py          # Real Groq API service client
│   ├── paper_retriever.py      # PDF loader & BM25 retrieval engine
│   ├── pipeline.py             # 4-step RAG pipeline core & verification loop
│   ├── runner.py               # Quick single-scenario execution runner
│   ├── scenarios_runner.py     # Comprehensive multi-scenario load & benchmark suite
│   └── standin_service.py      # HTTP 429 token-bucket rate limit simulator
├── .env                        # Environment variables (API keys - git-ignored)
├── .gitignore                  # Git ignore rules
├── results.csv                 # Execution log output across all test scenarios
├── writeup.md                  # 11-section technical report & metric tables
└── README.md                   # Project documentation
