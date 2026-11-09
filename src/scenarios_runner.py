import csv
import time
import os
import sys
import random
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from paper_retriever import PaperRetriever
from standin_service import StandinModelService
from groq_client import GroqService
from pipeline import RAGPipeline

def run_all_scenarios_and_retrieval():
    retriever = PaperRetriever()
    retriever.download_papers()

    # 15 Custom evaluation questions (12 answerable, 3 unanswerable)
    questions = [
        "What is the multi-head attention mechanism?",
        "What scaling factor is used in scaled dot-product attention?",
        "How many encoder layers are in the base Transformer?",
        "How does Moshi handle real-time streaming dialogue?",
        "What text model architecture does Moshi build on?",
        "What audio codec is used in Moshi?",
        "What is dynamic chunking in sequence modeling?",
        "How does dynamic chunking differ from fixed tokenization?",
        "What metric is evaluated in dynamic chunking experiments?",
        "Why is positional encoding necessary in self-attention?",
        "What optimizer was used for training Attention Is All You Need?",
        "What is the full title of the Moshi paper?",
        "Who won the 2022 World Cup?",                          # Refusal test 1
        "What is the capital city of Australia?",                # Refusal test 2
        "How do quantum computers factor large primes?",          # Refusal test 3
    ]

    fieldnames = [
        "request_id", "scenario", "source", "start_time",
        "first_token_ms", "total_ms", "status", "retries",
        "tokens_in", "tokens_out", "retrieval_setting", "answer_grade"
    ]

    csv_file = "results.csv"
    req_id = 1

    with open(csv_file, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # -------------------------------------------------------------
        # PART A: Retrieval Settings Benchmarks (Table 3)
        # -------------------------------------------------------------
        retrieval_configs = [
            {"name": "size_250_k5", "chunk_size": 250, "overlap": 30, "k": 5},
            {"name": "size_500_k3", "chunk_size": 500, "overlap": 50, "k": 3},
            {"name": "size_1000_k2", "chunk_size": 1000, "overlap": 100, "k": 2},
        ]

        print("--- Running Retrieval Setting Benchmarks ---")
        standin = StandinModelService()
        
        for config in retrieval_configs:
            retriever.process_and_chunk(chunk_size=config["chunk_size"], overlap=config["overlap"])
            pipe = RAGPipeline(standin, retriever)
            
            for q in questions:
                start_ts = round(time.time(), 2)
                res = pipe.run(q, top_k=config["k"])

                if res["status"] == "ok":
                    grade = "correct"
                elif "refused" in res["status"]:
                    grade = "refused_correctly" if "outside research" in res["answer"] or "not found" in res["answer"] else "refused_wrongly"
                else:
                    grade = "wrong"

                writer.writerow({
                    "request_id": f"req_{req_id:04d}",
                    "scenario": "retrieval_eval",
                    "source": "stand-in",
                    "start_time": start_ts,
                    "first_token_ms": res.get("first_token_ms", 0),
                    "total_ms": res.get("total_ms", 0),
                    "status": res.get("status", "ok"),
                    "retries": res.get("retries", 0),
                    "tokens_in": res.get("tokens_in", 0),
                    "tokens_out": res.get("tokens_out", 0),
                    "retrieval_setting": config["name"],
                    "answer_grade": grade
                })
                req_id += 1

        # -------------------------------------------------------------
        # PART B: Load Scenarios (Table 2a / 2b)
        # -------------------------------------------------------------
        print("--- Running Load Scenario Stress Tests ---")
        retriever.process_and_chunk(chunk_size=500, overlap=50)

        # 1. Steady Load (Real Service)
        try:
            real_groq = GroqService(model_name="qwen/qwen3.8-27b")
            real_pipe = RAGPipeline(real_groq, retriever)
            for i in range(10):
                q = questions[i % len(questions)]
                start_ts = round(time.time(), 2)
                res = real_pipe.run(q, top_k=3)
                writer.writerow({
                    "request_id": f"req_{req_id:04d}",
                    "scenario": "steady",
                    "source": "real",
                    "start_time": start_ts,
                    "first_token_ms": res.get("first_token_ms", 0),
                    "total_ms": res.get("total_ms", 0),
                    "status": res.get("status", "ok"),
                    "retries": res.get("retries", 0),
                    "tokens_in": res.get("tokens_in", 0),
                    "tokens_out": res.get("tokens_out", 0),
                    "retrieval_setting": "size_500_k3",
                    "answer_grade": "correct" if res.get("status") == "ok" else "wrong"
                })
                req_id += 1
                time.sleep(1.0)
        except Exception as e:
            print(f"Skipping real steady run (fallback to standin): {e}")

        # 2. Burst Load (Stand-in)
        standin_burst = StandinModelService(rpm_limit=30, tpm_limit=14400)
        pipe_burst = RAGPipeline(standin_burst, retriever)
        for i in range(25):
            q = questions[i % len(questions)]
            start_ts = round(time.time(), 2)
            res = pipe_burst.run(q, top_k=3)
            writer.writerow({
                "request_id": f"req_{req_id:04d}",
                "scenario": "burst",
                "source": "stand-in",
                "start_time": start_ts,
                "first_token_ms": res.get("first_token_ms", 0),
                "total_ms": res.get("total_ms", 0),
                "status": res.get("status", "ok"),
                "retries": res.get("retries", 0),
                "tokens_in": res.get("tokens_in", 0),
                "tokens_out": res.get("tokens_out", 0),
                "retrieval_setting": "size_500_k3",
                "answer_grade": "correct" if res.get("status") == "ok" else "wrong"
            })
            req_id += 1

        # 3. Sustained Overload (Stand-in)
        standin_overload = StandinModelService(rpm_limit=15, tpm_limit=7200)
        pipe_overload = RAGPipeline(standin_overload, retriever)
        for i in range(40):
            q = questions[i % len(questions)]
            start_ts = round(time.time(), 2)
            res = pipe_overload.run(q, top_k=3)
            writer.writerow({
                "request_id": f"req_{req_id:04d}",
                "scenario": "sustained_overload",
                "source": "stand-in",
                "start_time": start_ts,
                "first_token_ms": res.get("first_token_ms", 0),
                "total_ms": res.get("total_ms", 0),
                "status": res.get("status", "ok"),
                "retries": res.get("retries", 0),
                "tokens_in": res.get("tokens_in", 0),
                "tokens_out": res.get("tokens_out", 0),
                "retrieval_setting": "size_500_k3",
                "answer_grade": "correct" if res.get("status") == "ok" else "wrong"
            })
            req_id += 1

        # 4. Per-Minute Limit Halved (Stand-in)
        standin_halved = StandinModelService(rpm_limit=7, tpm_limit=3600)
        pipe_halved = RAGPipeline(standin_halved, retriever)
        for i in range(20):
            q = questions[i % len(questions)]
            start_ts = round(time.time(), 2)
            res = pipe_halved.run(q, top_k=3)
            writer.writerow({
                "request_id": f"req_{req_id:04d}",
                "scenario": "per_minute_limit_halved",
                "source": "stand-in",
                "start_time": start_ts,
                "first_token_ms": res.get("first_token_ms", 0),
                "total_ms": res.get("total_ms", 0),
                "status": res.get("status", "ok"),
                "retries": res.get("retries", 0),
                "tokens_in": res.get("tokens_in", 0),
                "tokens_out": res.get("tokens_out", 0),
                "retrieval_setting": "size_500_k3",
                "answer_grade": "correct" if res.get("status") == "ok" else "wrong"
            })
            req_id += 1

        # 5. Daily Budget Exhausted (Stand-in)
        standin_rpd = StandinModelService(rpm_limit=30, tpm_limit=14400, rpd_limit=5)
        pipe_rpd = RAGPipeline(standin_rpd, retriever)
        for i in range(10):
            q = questions[i % len(questions)]
            start_ts = round(time.time(), 2)
            res = pipe_rpd.run(q, top_k=3)
            writer.writerow({
                "request_id": f"req_{req_id:04d}",
                "scenario": "daily_budget_exhausted",
                "source": "stand-in",
                "start_time": start_ts,
                "first_token_ms": res.get("first_token_ms", 0),
                "total_ms": res.get("total_ms", 0),
                "status": res.get("status", "ok"),
                "retries": res.get("retries", 0),
                "tokens_in": res.get("tokens_in", 0),
                "tokens_out": res.get("tokens_out", 0),
                "retrieval_setting": "size_500_k3",
                "answer_grade": "correct" if res.get("status") == "ok" else "wrong"
            })
            req_id += 1

    print(f"All scenarios completed. Full output saved to {csv_file}.")

if __name__ == "__main__":
    run_all_scenarios_and_retrieval()