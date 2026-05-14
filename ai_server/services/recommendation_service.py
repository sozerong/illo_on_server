"""공고 추천 서비스

general_recommend : 설문 데이터 → neural_search → 벡터 기반 추천
find_similar_jobs : 공고 ID → 같은 텍스트 기반 유사 공고
"""

from __future__ import annotations

import logging
import math
import uuid
from typing import List, Optional

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.job import Job
from ..models.user import GeneralRecommendation, Survey
from .opensearch_service import neural_search, build_search_text, search_jobs

logger = logging.getLogger(__name__)


# ── 설문 → 쿼리 텍스트 ───────────────────────────────────────
def _survey_to_query(survey: Optional[Survey]) -> str:
    """설문 필드를 임베딩 검색용 자연어 텍스트로 변환"""
    if not survey:
        return ""
    parts = [
        survey.job_type    or "",
        survey.occupation  or "",
        survey.region      or "",
        survey.career_type or "",
        survey.education   or "",
    ]
    return " ".join(p for p in parts if p).strip()


# ── 코사인 유사도 (PostgreSQL 임베딩 fallback용) ───────────────
def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x ** 2 for x in a))
    nb  = math.sqrt(sum(x ** 2 for x in b))
    return dot / (na * nb + 1e-9)


# ── PostgreSQL에서 공고 조회 ──────────────────────────────────
async def _fetch_jobs_by_ids(job_ids: List[str], job_db: AsyncSession) -> List[Job]:
    """OpenSearch 결과 순서를 유지하며 PostgreSQL에서 공고 가져오기"""
    if not job_ids:
        return []
    result = await job_db.execute(select(Job).where(Job.id.in_(job_ids)))
    jobs_map = {j.id: j for j in result.scalars().all()}
    return [jobs_map[jid] for jid in job_ids if jid in jobs_map]


# ── 일반 추천 (설문 기반 Neural Search) ──────────────────────
async def general_recommend(
    user_id: str,
    user_db: AsyncSession,
    job_db:  AsyncSession,
    top_k:   int = 10,
) -> List[GeneralRecommendation]:
    """
    설문 데이터 → neural_search → 유사 공고 추천
    ML 모델 미준비 시 SQL 필터로 fallback
    """
    # 설문 조회
    result = await user_db.execute(select(Survey).where(Survey.user_id == user_id))
    survey = result.scalar_one_or_none()

    query_text = _survey_to_query(survey)
    jobs: List[Job] = []

    if query_text:
        # Neural Search
        try:
            filters: dict = {}
            if survey and survey.region:
                filters["location"] = survey.region
            if survey and survey.career_type:
                filters["career_type"] = survey.career_type

            os_result = await neural_search(
                query_text=query_text,
                filters=filters or None,
                k=top_k,
            )
            job_ids = [item["id"] for item in os_result.get("items", []) if item.get("id")]
            jobs = await _fetch_jobs_by_ids(job_ids, job_db)
            logger.info("Neural 추천: user=%s query='%s' → %d건", user_id, query_text[:30], len(jobs))
        except Exception as e:
            logger.warning("Neural Search 실패 → SQL fallback: %s", e)

    # Fallback: SQL 필터
    if not jobs:
        from sqlalchemy import or_
        query = select(Job).order_by(Job.created_at.desc())
        if survey:
            conds = []
            if survey.job_type:
                conds.append(Job.job_type.ilike(f"%{survey.job_type}%"))
            if survey.region:
                conds.append(Job.location.ilike(f"%{survey.region}%"))
            if survey.career_type:
                conds.append(Job.career_type.ilike(f"%{survey.career_type}%"))
            if conds:
                query = query.where(or_(*conds))
        result2 = await job_db.execute(query.limit(top_k))
        jobs = list(result2.scalars().all())
        logger.info("SQL fallback 추천: user=%s → %d건", user_id, len(jobs))

    # 기존 일반 추천 초기화 후 저장
    await user_db.execute(delete(GeneralRecommendation).where(GeneralRecommendation.user_id == user_id))
    recs = []
    for job in jobs:
        rec = GeneralRecommendation(
            id      = str(uuid.uuid4()),
            user_id = user_id,
            job_id  = job.id,
        )
        user_db.add(rec)
        recs.append(rec)

    await user_db.commit()
    return recs


# ── 유사 공고 검색 ─────────────────────────────────────────────
async def find_similar_jobs(
    job_id: str,
    job_db: AsyncSession,
    top_k:  int = 6,
) -> List[Job]:
    """
    기준 공고의 search_text → neural_search → 유사 공고 반환
    임베딩 없으면 같은 직무 최신순 fallback
    """
    result   = await job_db.execute(select(Job).where(Job.id == job_id))
    base_job = result.scalar_one_or_none()
    if not base_job:
        return []

    # Neural Search
    try:
        query_text = build_search_text({
            "title":       base_job.title       or "",
            "job_type":    base_job.job_type     or "",
            "occupation":  base_job.occupation   or "",
            "career_type": base_job.career_type  or "",
            "location":    base_job.location     or "",
            "description": base_job.description  or "",
            "requirements":base_job.requirements or "",
        })
        os_result = await neural_search(query_text=query_text, k=top_k + 1)
        job_ids   = [
            item["id"] for item in os_result.get("items", [])
            if item.get("id") and item["id"] != job_id
        ][:top_k]
        jobs = await _fetch_jobs_by_ids(job_ids, job_db)
        if jobs:
            return jobs
    except Exception as e:
        logger.warning("유사 공고 Neural Search 실패: %s", e)

    # Fallback: 같은 직무 최신순
    result2 = await job_db.execute(
        select(Job)
        .where(Job.id != job_id, Job.job_type == base_job.job_type)
        .order_by(Job.created_at.desc())
        .limit(top_k)
    )
    return list(result2.scalars().all())
