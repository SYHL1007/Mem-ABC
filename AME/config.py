import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM API Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "local")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8080/v1")
LLM_MODEL_EXTRACT = os.getenv("LLM_MODEL_NAME", "Qwen3-30B-A3B-Instruct-2507")
LLM_MODEL_REASON = os.getenv("LLM_MODEL_NAME", "Qwen2.5-7B-Instruct")
LLM_TIMEOUT = 300.0 # API call timeout

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", 768))

# --- Neo4j Database Configuration ---
NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# --- Deduplication Thresholds ---
# Vector similarity above this value triggers direct merge (e.g., 0.98)
DEDUPE_MERGE_THRESHOLD = float(os.getenv("DEDUPE_MERGE_THRESHOLD", 0.90))
# Vector similarity within this range triggers LLM arbitration (e.g., 0.85 ~ 0.98)
DEDUPE_LLM_THRESHOLD = float(os.getenv("DEDUPE_LLM_THRESHOLD", 0.70))

# --- Knowledge Graph Build Concurrency Control ---
# Concurrency count between different users (inter-group concurrency)
KG_BUILD_USER_CONCURRENCY = int(os.getenv("KG_BUILD_USER_CONCURRENCY", 200))
# Timeout for a single memory processing task (seconds)
KG_BUILD_TASK_TIMEOUT = int(os.getenv("KG_BUILD_TASK_TIMEOUT", 400))
# Maximum retry attempts for a single task
KG_BUILD_MAX_RETRIES = int(os.getenv("KG_BUILD_MAX_RETRIES", 5))

# --- Logging ---
LOG_FILE = "app.log"