import csv
import time
import os
import sys
from pathlib import Path

# Add src folder to python path automatically
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
from paper_retriever import PaperRetriever
from standin_service import StandinModelService
from groq_client import GroqService
from pipeline import RAGPipeline
# Load .env variables at the script entrypoint
load_dotenv()

def run_experiments():
    # 1. Setup Retriever & Data
    retriever = PaperRetriever()
    retriever.download_papers()
    retriever.process_and_chunk(chunk_size=500, overlap=50)

    # 2. Toggle between Real Groq API and Stand-in Service
    USE_REAL_GROQ = True  # Set to False to use the Stand-in Simulator

    if USE_REAL_GROQ:
        print("Using Real Groq API Service...")
        llm_service = GroqService(model_name="qwen/qwen3.8-27b")
    else:
        print("Using Stand-in Service Simulator...")
        llm_service = StandinModelService()

    pipeline = RAGPipeline(llm_service, retriever)

    queries = [
        "What is the multi-head attention mechanism?",
        "How does Moshi achieve real-time latency?",
        "What is dynamic chunking in sequence modeling?",
        "Who won the 2022 World Cup?", # Unanswerable test query (triggers refusal)
    ]

    csv_file = "results.csv"
    fieldnames = [
        "request_id", "scenario", "source", "start_time", 
        "first_token_ms", "total_ms", "status", "retries", 
        "tokens_in", "tokens_out", "retrieval_setting", "answer_grade"
    ]

    print(f"Running test scenario and generating {csv_file}...")

    with open(csv_file, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        req_id = 1
        source_label = "real" if USE_REAL_GROQ else "stand-in"
        
        for q in queries:
            start_ts = round(time.time(), 2)
            res = pipeline.run(q)
            
            grade = "correct" if res["status"] == "ok" else (
                "refused_correctly" if "refused" in res["status"] else "wrong"
            )

            writer.writerow({
                "request_id": f"req_{req_id:03d}",
                "scenario": "steady",
                "source": source_label,
                "start_time": start_ts,
                "first_token_ms": res.get("first_token_ms", 0),
                "total_ms": res.get("total_ms", 0),
                "status": res.get("status", "ok"),
                "retries": res.get("retries", 0),
                "tokens_in": res.get("tokens_in", 0),
                "tokens_out": res.get("tokens_out", 0),
                "retrieval_setting": "size_500_k3",
                "answer_grade": grade
            })
            req_id += 1

    print(f"Finished! Execution log written to {csv_file}.")

if __name__ == "__main__":
    run_experiments()