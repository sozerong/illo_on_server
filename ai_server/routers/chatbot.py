"""RAG 챗봇 라우터"""

from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from ..chatbot.rag import (
    search_jobs,
    search_trends,
    search_youth_policies,
    generate_answer,
    classify_intent,
    extract_entities,
    category_stats,
    normalize_job_type,
    rewrite_query_with_history,
)
from ..chatbot.loader import get_user_info, load_quick_replies

router = APIRouter(prefix="/chat", tags=["chatbot"])


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None
    user_info: Optional[dict] = None
    is_quick_replies: Optional[bool] = False
    quick_replies_category: Optional[str] = None
    history: Optional[List[ChatMessage]] = []


def extract_sources(jobs: list) -> list:
    return [job.get("job_id", "") for job in jobs if job.get("job_id")]


@router.get("/quick-replies")
def get_quick_replies():
    quick_replies = load_quick_replies()
    result = {}
    for qr in quick_replies:
        category = qr.get("category", "")
        if category not in result:
            result[category] = []
        result[category].append({"id": qr.get("id"), "question": qr.get("question")})
    return result


@router.post("")
def chat(request: ChatRequest):
    query = request.message

    search_query = rewrite_query_with_history(query, request.history)

    intent = classify_intent(search_query)
    if intent == "COMPANY" and any(
        word in search_query
        for word in ["공고", "채용", "연봉", "지원서류", "채용절차", "마감일", "직무", "상세", "자세히"]
    ):
        intent = "GENERAL"
    entities = extract_entities(search_query, intent)

    if len(query.strip()) < 2:
        return {"answer": "질문을 좀 더 구체적으로 입력해주세요 😊"}

    history = [{"role": msg.role, "content": msg.content} for msg in request.history]

    user_info = request.user_info
    if request.user_id and not user_info:
        user_info = get_user_info(request.user_id)

    if request.is_quick_replies:
        print(f"퀵리플라이 처리: {query} (카테고리: {request.quick_replies_category})")
        entities = extract_entities(search_query, "QUICK_REPLIES")
        print(f"추출된 개체명: {entities}")
        if request.quick_replies_category == "나의 검색":
            related_jobs = search_jobs(search_query, user_info=user_info, entities=entities)
            quick_replies_user_info = user_info
        else:
            related_jobs = search_jobs(search_query, user_info=None, entities=entities)
            quick_replies_user_info = None
        related_trends = search_trends(search_query, entities=entities)
        answer = generate_answer(
            query=query, related_jobs=related_jobs, user_info=quick_replies_user_info,
            history=history, related_trends=related_trends, related_policies=[], intent="QUICK_REPLIES",
        )
        return {
            "answer": answer,
            "sources": extract_sources(related_jobs),
            "intent": "QUICK_REPLIES",
            "quick_replies_category": request.quick_replies_category,
            "user_message": {"role": "user", "content": query},
            "assistant_message": {"role": "assistant", "content": answer},
        }

    if entities.get("직무"):
        entities["직무"] = normalize_job_type(entities["직무"])

    print(f"추출된 개체명: {entities}")

    if intent == "INVALID":
        return {
            "answer": "저는 취업 관련 질문만 답변할 수 있어요! 공고 추천, 취업 정보 등을 물어봐주세요 😊",
            "sources": [], "intent": intent,
        }

    if intent == "COMPANY":
        return {
            "answer": "정확한 기업 정보는 공고 상세페이지를 확인해주세요 😊",
            "sources": [], "intent": intent,
        }

    if intent == "PUBLIC_DATA":
        policies = search_youth_policies(search_query, entities=entities, user_info=user_info)
        answer = generate_answer(
            query=query, related_jobs=[], user_info=None, history=history,
            related_trends=[], related_policies=policies, intent=intent,
        )
        return {
            "answer": answer, "sources": [], "intent": intent,
            "user_message": {"role": "user", "content": query},
            "assistant_message": {"role": "assistant", "content": answer},
        }

    if intent == "CAREER_TIP":
        answer = generate_answer(
            query=query, related_jobs=[], user_info=None, history=history,
            related_trends=[], related_policies=[], intent=intent,
        )
        return {
            "answer": answer, "sources": [], "intent": intent,
            "user_message": {"role": "user", "content": query},
            "assistant_message": {"role": "assistant", "content": answer},
        }

    if intent == "TREND":
        related_trends = search_trends(query, entities=entities)
        answer = generate_answer(
            query=query, related_jobs=[], user_info=None, history=history,
            related_trends=related_trends, related_policies=[], intent=intent,
            category_stats=category_stats,
        )
        return {
            "answer": answer, "sources": [], "intent": intent,
            "user_message": {"role": "user", "content": query},
            "assistant_message": {"role": "assistant", "content": answer},
        }

    if intent == "CUSTOM":
        if user_info:
            entities["직무"] = entities.get("직무") or user_info.get("job_type")
            entities["지역"] = entities.get("지역") or user_info.get("region")
            entities["경력"] = entities.get("경력") or user_info.get("career_type")
        if entities.get("직무"):
            entities["직무"] = normalize_job_type(entities["직무"])
        print(f"병합 후 entities: {entities}")
        related_jobs = search_jobs(query, user_info=user_info, entities=entities)
        answer = generate_answer(
            query=query, related_jobs=related_jobs, user_info=user_info, history=history,
            related_trends=[], related_policies=[], intent=intent,
        )
        return {
            "answer": answer, "sources": extract_sources(related_jobs), "intent": intent,
            "user_message": {"role": "user", "content": query},
            "assistant_message": {"role": "assistant", "content": answer},
        }

    # GENERAL
    related_jobs = search_jobs(query, user_info=None, entities=entities)
    answer = generate_answer(
        query=query, related_jobs=related_jobs, user_info=None, history=history,
        related_trends=[], related_policies=[], intent=intent,
    )
    return {
        "answer": answer, "sources": extract_sources(related_jobs), "intent": intent,
        "user_message": {"role": "user", "content": query},
        "assistant_message": {"role": "assistant", "content": answer},
    }
