import os
import urllib.request
import pypdf
from typing import List, Dict, Any

PAPERS = {
    "attention": {
        "url": "https://arxiv.org/pdf/1706.03762v7.pdf",
        "title": "Attention Is All You Need (v7)",
        "file": "data/Attention.pdf"
    },
    "moshi": {
        "url": "https://arxiv.org/pdf/2410.00037v1.pdf",
        "title": "Moshi: a speech-text foundation model for real-time dialogue (v1)",
        "file": "data/Moshi.pdf"
    },
    "dynamic_chunking": {
        "url": "https://arxiv.org/pdf/2507.07955v2.pdf",
        "title": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling (v2)",
        "file": "data/Dynamic_chunking.pdf"
    }
}

class PaperRetriever:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.chunks: List[Dict[str, Any]] = []

    def download_papers(self):
        for key, info in PAPERS.items():
            if not os.path.exists(info["file"]):
                print(f"Downloading {info['title']}...")
                urllib.request.urlretrieve(info["url"], info["file"])

    def process_and_chunk(self, chunk_size: int = 500, overlap: int = 50):
        """Splits papers into sliding-window text chunks."""
        self.chunks = []
        for key, info in PAPERS.items():
            if not os.path.exists(info["file"]):
                continue
            reader = pypdf.PdfReader(info["file"])
            text = ""
            for page_idx, page in enumerate(reader.pages):
                extracted = page.extract_text() or ""
                text += f"\n[Page {page_idx + 1}] " + extracted

            words = text.split()
            step = chunk_size - overlap
            for i in range(0, len(words), step):
                chunk_words = words[i:i + chunk_size]
                chunk_text = " ".join(chunk_words)
                self.chunks.append({
                    "chunk_id": f"{key}_c{i//step}",
                    "paper": info["title"],
                    "text": chunk_text
                })

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Simple BM25/keyword scoring heuristic for passage retrieval."""
        query_words = set(query.lower().split())
        scored_chunks = []
        for chunk in self.chunks:
            score = sum(1 for word in query_words if word in chunk["text"].lower())
            if score > 0:
                scored_chunks.append((score, chunk))
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [c[1] for c in scored_chunks[:top_k]]