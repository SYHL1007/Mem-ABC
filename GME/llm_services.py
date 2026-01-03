import numpy as np
from openai import AsyncOpenAI, RateLimitError  # --- Fix 1: Import RateLimitError ---
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from openai import RateLimitError, APITimeoutError
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import json5

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((ValueError, RateLimitError, APITimeoutError, asyncio.TimeoutError))
)
def str_to_json(input_str):
    """Converts a string to a JSON object, supporting both standard JSON and JSON5."""
    if input_str.startswith("json"):
        input_str = input_str[len("json"):]
    if input_str.endswith("json"):
        input_str = input_str[:-len("json")]
    input_str = input_str.strip().replace("\n", "").replace("```json", "").replace("```", "")
    try:
        return json.loads(input_str, strict=False)  # Attempts to parse as standard JSON
    except:
        pass
    try:
        return json5.loads(input_str)  # Attempts to parse as JSON5
    except:
        pass
    print(input_str)
    raise ValueError("Invalid JSON")

async def call_llm_async_with_retry(
    async_client: AsyncOpenAI,
    prompt: str,
    model_name: str,
    json_output: bool = False
):
    """Handles asynchronous LLM requests with retry logic."""
    try:
        response = await async_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=4096,
            timeout=300
        )
        raw_text = response.choices[0].message.content
        if json_output:
            response_obj = str_to_json(raw_text)
            return response_obj["personalized_answer"]
        else:
            return raw_text
    except (RateLimitError, APITimeoutError, asyncio.TimeoutError, ValueError) as e:
        print(f"❗️ Retryable error ({type(e).__name__}), will retry... Error: {e}")
        raise  # Re-raise the exception for retry with tenacity
    except Exception as e:
        print(f"❗️ Non-retryable error: {type(e).__name__}: {e}")
        return None

def parse_llm_update_response(response: str):
    """Parses the LLM's response for update operation instructions."""
    if not response:
        return {"operation": "NOOP", "payload": None}
    parts = response.strip().split('\n', 1)
    operation = parts[0].strip().upper()
    payload = parts[1].strip() if len(parts) > 1 else None
    if operation not in ["ADD", "UPDATE", "DELETE", "NOOP"]:
        return {"operation": "NOOP", "payload": f"Invalid operation: {operation}"}
    return {"operation": operation, "payload": payload}

def retrieve_similar_memories_local_sync(embedding_model: SentenceTransformer, query: str, memories_dict: dict, top_k=30) -> dict:
    """[Synchronous Function] Compute embeddings and similarity locally."""
    if not memories_dict:
        return {}
    memory_ids, memory_texts = list(memories_dict.keys()), list(memories_dict.values())
    all_texts = [query] + memory_texts
    all_embeddings = embedding_model.encode(all_texts)
    query_embedding = all_embeddings[0].reshape(1, -1)
    kb_embeddings = np.array(all_embeddings[1:])
    similarities = cosine_similarity(query_embedding, kb_embeddings)[0]
    actual_top_k = min(top_k, len(memory_ids))
    top_k_indices = np.argsort(similarities)[-actual_top_k:][::-1]
    return {memory_ids[i]: memory_texts[i] for i in top_k_indices}