import time
import random
from typing import Dict, Any, Generator

class StandinModelService:
    def __init__(self, rpm_limit: int = 30, tpm_limit: int = 14400, rpd_limit: int = 1440):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.rpd_limit = rpd_limit
        
        self.request_timestamps = []
        self.token_timestamps = []
        self.daily_requests = 0

    def _check_rate_limits(self, estimated_tokens: int = 500):
        now = time.time()
        # Clean up timestamps older than 60s
        self.request_timestamps = [t for t in self.request_timestamps if now - t < 60]
        self.token_timestamps = [(t, count) for t, count in self.token_timestamps if now - t < 60]
        
        current_tpm = sum(count for _, count in self.token_timestamps)

        if len(self.request_timestamps) >= self.rpm_limit:
            raise Exception("HTTP 429: Rate limit reached (RPM)")
        if current_tpm + estimated_tokens > self.tpm_limit:
            raise Exception("HTTP 429: Rate limit reached (TPM)")
        if self.daily_requests >= self.rpd_limit:
            raise Exception("HTTP 429: Daily budget exhausted (RPD)")

        self.request_timestamps.append(now)
        self.token_timestamps.append((now, estimated_tokens))
        self.daily_requests += 1

    def generate_response(self, prompt: str, mock_answer: str) -> Dict[str, Any]:
        """Simulates latency, time-to-first-token (TTFT), and token generation."""
        self._check_rate_limits()
        
        start_time = time.time()
        # Simulate initial processing delay (TTFT)
        ttft_delay = random.uniform(0.12, 0.45)  # 120ms - 450ms
        time.sleep(ttft_delay)
        first_token_time = time.time()
        
        # Simulate stream processing time per word
        tokens_out = len(mock_answer.split())
        stream_delay = tokens_out * random.uniform(0.01, 0.03)
        time.sleep(stream_delay)
        
        total_time = time.time() - start_time
        ttft_ms = (first_token_time - start_time) * 1000
        
        return {
            "text": mock_answer,
            "first_token_ms": round(ttft_ms, 1),
            "total_ms": round(total_time * 1000, 1),
            "tokens_in": len(prompt.split()),
            "tokens_out": tokens_out
        }