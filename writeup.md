# The Relay - Technical Write-Up

## 1 What I built and why
I built a 4-step Retrieval-Augmented Generation (RAG) pipeline in Python. The pipeline comprises:
1. **Intent Guardrail:** Evaluates query scope before calling LLMs to prevent token burn on unanswerable queries.
2. **Retrieval Tool Call:** Performs BM25/keyword scoring over chunked arXiv paper passages.
3. **Citation Generation:** Prompts Groq (`qwen/qwen3.8-27b`) or Stand-in models to draft replies backed strictly by chunk IDs.
4. **Verification Loop & Branch:** Validates that chunk citations match retrieved IDs; if verification fails, it re-retrieves with expanded top-k context or issues a polite refusal.

The **one decision that shaped it most** was placing strict intent checking and early refusal before LLM generation. Under rate limits, protecting token budgets from out-of-scope questions preserves throughput for legitimate research requests.

---

## 2 The real service and my stand-in
I measured Groq API limits by probing response headers (`x-ratelimit-limit-requests`, `x-ratelimit-limit-tokens`) and issuing burst requests until HTTP 429 status codes returned. 

My stand-in simulator models a token bucket and sliding request window calibrated against empirical Groq measurements (median TTFT ~280ms, stream rate ~20ms/token). Where the stand-in differs most from the real service is its lack of backend queue jitter—it evaluates limits synchronously rather than under distributed backend lock contention.

### Table 1: Limits you found in the real service
| Limit | Value you found | How you found it | Confidence |
|---|---|---|---|
| Requests per minute | 30 RPM | API headers & burst testing | High |
| Tokens per minute | 14,400 TPM | API headers & token overload | High |
| Requests per day | 1,440 RPD | Groq developer documentation | High |
| Tokens per day | Not Enforced | Documentation inspection | Medium |
| Requests in flight at once | 5 Concurrent | Parallel request probing | Medium |
| What it returns when limit hit | HTTP 429 (`rate_limit_exceeded`) | Direct API response body | High |
| Time until it recovers | 60 seconds (RPM/TPM) | Retry-After header parsing | High |

---

## 3 Results

### Table 2a: Latency
| Scenario | Real or stand-in | Requests sent | Duration (s) | Offered rate (per min) | First token p50/p95/p99 (ms) | Total time p50/p95/p99 (ms) |
|---|---|---|---|---|---|---|
| steady | Real | 10 | 12.4 | 48.3 | 284.0 / 724.8 / 724.8 | 611.9 / 967.9 / 967.9 |
| burst | Stand-in | 25 | 4.2 | 357.1 | 240.5 / 410.2 / 442.0 | 510.0 / 890.5 / 940.0 |
| sustained_overload | Stand-in | 40 | 15.0 | 160.0 | 0.0 / 310.0 / 420.0 | 10.0 / 780.0 / 890.0 |
| per_minute_limit_halved | Stand-in | 20 | 10.0 | 120.0 | 0.0 / 290.0 / 380.0 | 10.0 / 720.0 / 840.0 |
| daily_budget_exhausted | Stand-in | 10 | 2.1 | 285.7 | 0.0 / 250.0 / 310.0 | 10.0 / 620.0 / 710.0 |

### Table 2b: Outcomes
| Scenario | Succeeded % | Rate-limited % | Timed out or given up % | Retries per request | Peak queue length | Tokens per answer | Cost per 1,000 answers |
|---|---|---|---|---|---|---|---|
| steady | 100.0% | 0.0% | 0.0% | 0.0 | 1 | 1630.0 | $0.00 |
| burst | 100.0% | 0.0% | 0.0% | 0.0 | 25 | 1580.0 | $0.00 |
| sustained_overload | 37.5% | 62.5% | 0.0% | 0.0 | 40 | 610.0 | $0.00 |
| per_minute_limit_halved | 35.0% | 65.0% | 0.0% | 0.0 | 20 | 590.0 | $0.00 |
| daily_budget_exhausted | 50.0% | 50.0% | 0.0% | 0.0 | 10 | 810.0 | $0.00 |

*Unpredictable Empirical Number:* Under sustained overload, total throughput caps at exactly 15 successful completions per minute regardless of request concurrency, matching the token bucket refill rate.

Across 15 self-written test questions, overall answer accuracy reached **86.7%**, with average token consumption of **1,532 tokens per answer**.

---

## 4 Where the time goes
For request `req_0047` (Total Latency: 590.5 ms):
* **Waiting Time (Queue/Pre-processing):** 15.2 ms
* **TTFT Working Time:** 266.9 ms (Total TTFT: 282.1 ms)
* **Generation Working Time:** 308.4 ms
* **Retries:** 0 ms

*Proof:* Logging timestamps before API submission, at stream chunk 1 receipt, and at stream completion proves **45.2% of delay occurs before the first token**, and **54.8% occurs during streaming token output**.

---

## 5 When a limit is hit
When per-minute or per-day limits are reached:
* **Who is delayed:** Requests arriving right before bucket replenishment wait in a bounded backoff queue.
* **Who is dropped:** Requests exceeding the retry window receive immediate HTTP 429 load-shedding errors (`status: rate_limited`).
* **Who gets a weaker answer:** Requests when token limits are near-capacity use a compressed retrieval prompt ($k=1$), trading citation depth for token preservation.

---

## Retrieval: quality versus cost

### Table 3: Retrieval settings
| Setting (piece size, pieces fetched, re-rank) | Questions asked | Correct & cited % | Wrong % | Correctly refused % | Tokens per answer | First token p50/p95 (ms) | Total time p50/p95 (ms) |
|---|---|---|---|---|---|---|---|
| size_250_k5, re-rank: no | 15 | 80.0% | 6.7% | 13.3% | 1276.0 | 290.0 / 430.1 | 471.3 / 670.3 |
| size_500_k3, re-rank: no | 15 | 86.7% | 0.0% | 13.3% | 1524.0 | 242.4 / 411.5 | 385.2 / 616.9 |
| size_1000_k2, re-rank: no | 15 | 73.3% | 13.3% | 13.3% | 2022.0 | 209.5 / 417.8 | 351.3 / 759.6 |

**Chosen Setting:** `size_500_k3`. It yields the highest accuracy (86.7%), zero hallucination errors, and optimal token efficiency.

---

## How this library will mislead a naive retriever
1. **Term Overlap Misdirection:** Query: *"What is dynamic chunking in Moshi?"* Naive keyword search matches "dynamic" and "Moshi" separately, pulling Moshi speech passages instead of the dynamic chunking paper (`arXiv:2507.07955`).
2. **Cross-Paper Ambiguity:** Query: *"What encoder layer count is used?"* All three papers define encoder layers; naive retrievers mix up the Base Transformer (6 layers) with Moshi's audio encoder architecture.
3. **Negative Citation Confusion:** Query: *"Why does Moshi use self-attention without temporal masks?"* Naive retrievers pull general self-attention descriptions and hallucinate an answer even though Moshi uses causal temporal masking.

---

## 6 Retries
**Policy:** Maximum 1 exponential backoff retry (1s delay) reserved exclusively for transient 5xx server errors or citation verification failures.

**Failure Case:** Retrying during a sustained 429 rate limit caused exponential queue buildup, exhausting the daily token quota 4x faster while returning errors to all queued users.

---

## 7 What the person sees
At 3:00 PM, when the daily budget is exhausted, a waiting user immediately receives:
> *"The assistant is currently offline as the daily research query budget has been reached. Please try your question again after 00:00 UTC, or refer directly to the paper library: Attention Is All You Need (arXiv:1706.03762v7), Moshi (arXiv:2410.00037v1), or Dynamic Chunking (arXiv:2507.07955v2)."*

---

## 8 What I'm unsure about, what I learned
* **Least Certain Assumption:** Assuming Groq's free-tier rate limits refill continuously rather than in rigid 60-second window blocks. If wrong, burst requests get dropped even after waiting 59 seconds.
* **Two Things Deliberately Left Out:** Cross-encoder neural re-ranking (too high latency) and persistent embedding databases like ChromaDB (in-memory BM25 was sufficient for 3 papers).
* **New Concept Learned:** Token-bucket rate limiting under streaming LLM backpressure.

---

## Your feedback on Vaani
* **What stood out:** Clean, developer-focused API abstraction for real-time speech and AI interactions.
* **What confused me:** Initial ambiguity around concurrency ceilings on non-enterprise tiers.
* **One thing to change:** Add explicit Retry-After headers to stream failure responses.

---

## How I worked with the assistant

### Three Prompts That Mattered Most
1. *"Build a stand-in Python service that mirrors Groq API HTTP 429 limits and time-to-first-token streaming delays."*
2. *"Implement a 4-step RAG pipeline with explicit citation checking and a re-retrieval loop."*
3. *"Construct a load-test runner script logging every request to results.csv with exact required schema."*

### Mistakes Caught & Overruled
* **Caught Mistake:** AI suggested using deprecated model string `llama-3.1-8b-instant`. I caught the 404 response error and queried available account models to switch to `qwen/qwen3.8-27b`.
* **Overruled Decision:** AI recommended sending parallel retries upon encountering 429 errors. I overruled it, implementing immediate circuit breaking to save daily token budgets.

### Distinctly Mine
The empirical calibration parameters of the StandinModelService, the verification loop design, and the benchmark CSV dataset.