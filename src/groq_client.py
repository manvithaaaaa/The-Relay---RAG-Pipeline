import os
import time
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv()

class GroqService:
    def __init__(self, model_name: str = "qwen/qwen3.8-27b"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY missing! Ensure it is set in your .env file or environment variables."
            )
        
        self.client = Groq(api_key=api_key)
        self.model_name = model_name

    def generate_response(self, prompt: str) -> dict:
        start_time = time.time()
        first_token_time = None
        full_text = ""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )

            for chunk in response:
                if first_token_time is None and chunk.choices[0].delta.content:
                    first_token_time = time.time()
                if chunk.choices[0].delta.content:
                    full_text += chunk.choices[0].delta.content

            end_time = time.time()
            ttft_ms = ((first_token_time or end_time) - start_time) * 1000
            total_ms = (end_time - start_time) * 1000

            return {
                "text": full_text,
                "first_token_ms": round(ttft_ms, 1),
                "total_ms": round(total_ms, 1),
                "tokens_in": len(prompt.split()),  # Token approximation
                "tokens_out": len(full_text.split()),
                "status": "ok"
            }
        except Exception as e:
            err_msg = str(e)
            print(f"Groq API Error Detail: {err_msg}")
            status = "rate_limited" if "429" in err_msg else "error"
            return {
                "text": "",
                "first_token_ms": 0,
                "total_ms": round((time.time() - start_time) * 1000, 1),
                "tokens_in": 0,
                "tokens_out": 0,
                "status": status,
                "error": err_msg
            }