import os
from pathlib import Path

# ── PostgreSQL ────────────────────────────────────────────────
PG_HOST     = os.environ.get("PG_HOST",     "svc.sel3.cloudtype.app")
PG_PORT     = os.environ.get("PG_PORT",     "32200")
PG_USER     = os.environ.get("PG_USER",     "root")
PG_PASSWORD = os.environ.get("PG_PASSWORD", "root")
PG_DBNAME   = os.environ.get("PG_DBNAME",   "root")

USER_DB_URL = f"postgresql+asyncpg://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DBNAME}"
JOB_DB_URL  = f"postgresql+asyncpg://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DBNAME}"

# ── Ollama ────────────────────────────────────────────────────
OLLAMA_HOST  = os.environ.get("OLLAMA_HOST",  "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")

# ── OpenSearch ────────────────────────────────────────────────
OPENSEARCH_HOST = os.environ.get("OPENSEARCH_HOST", "http://localhost:9200")
JOBS_INDEX      = "iloon_jobs"

# ── 공고 JSONL 경로 ───────────────────────────────────────────
JOBS_JSONL_DIR = Path(os.environ.get("JOBS_JSONL_DIR", "logs"))
