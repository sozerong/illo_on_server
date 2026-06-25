"""일로온 AI 서버 — FastAPI 엔트리포인트"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .database import init_db
from .services.opensearch_service import ensure_index, ensure_ml_ready
from .routers import (
    survey_router,
    recommendations_router,
    jobs_router,
    importer_router,
    log_router,
    resume_analysis_router,
    chatbot_router,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("ai_server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("일로온 AI 서버 시작")

    await init_db()
    logger.info("PostgreSQL 테이블 초기화 완료")

    import asyncio

    async def _setup_opensearch():
        try:
            model_id = await ensure_ml_ready()
            await ensure_index(model_id=model_id)
            logger.info("OpenSearch Neural Search 준비 완료 (model=%s)", model_id)
        except Exception as e:
            logger.warning("OpenSearch 준비 실패 (SQL fallback으로 동작): %s", e)
            try:
                await ensure_index()
            except Exception:
                pass

    asyncio.create_task(_setup_opensearch())

    yield

    logger.info("AI 서버 종료")


app = FastAPI(
    title       = "일로온 AI 서버",
    description = "공고 추천 · 설문 관리",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start    = time.time()
    response = await call_next(request)
    elapsed  = (time.time() - start) * 1000
    logger.info("%s %s → %d  (%.1fms)", request.method, request.url.path, response.status_code, elapsed)
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s %s — %s", request.method, request.url.path, exc)
    return JSONResponse(status_code=500, content={"detail": "서버 내부 오류가 발생했습니다."})


app.include_router(survey_router,          prefix="/api/v1")
app.include_router(recommendations_router, prefix="/api/v1")
app.include_router(jobs_router,            prefix="/api/v1")
app.include_router(importer_router,        prefix="/api/v1")
app.include_router(log_router,             prefix="/api/v1")
app.include_router(resume_analysis_router, prefix="/api/v1")
app.include_router(chatbot_router,         prefix="/api/v1")


@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok", "service": "일로온 AI 서버"}


@app.get("/", tags=["health"])
async def root():
    return {"service": "일로온 AI 서버", "docs": "/docs", "health": "/health"}
