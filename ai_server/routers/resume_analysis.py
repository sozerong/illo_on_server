from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
from urllib.parse import urlparse


router = APIRouter(
    prefix="/resume",
    tags=["resume-analysis"]
)


class WorkExperience(BaseModel):
    companyName: Optional[str] = None
    department: Optional[str] = None
    role: Optional[str] = None
    employmentType: Optional[str] = None
    position: Optional[str] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    description: Optional[str] = None
    isCurrent: Optional[bool] = None


class Education(BaseModel):
    schoolName: Optional[str] = None
    major: Optional[str] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    status: Optional[str] = None
    description: Optional[str] = None


class Award(BaseModel):
    awardName: Optional[str] = None
    organization: Optional[str] = None
    awardDate: Optional[str] = None
    description: Optional[str] = None


class Language(BaseModel):
    languageName: Optional[str] = None
    testName: Optional[str] = None
    score: Optional[str] = None
    testDate: Optional[str] = None
    description: Optional[str] = None


class Portfolio(BaseModel):
    portfolioName: Optional[str] = None
    url: Optional[str] = None


class CoverLetter(BaseModel):
    content: Optional[str] = None


class Certification(BaseModel):
    certificationName: Optional[str] = None
    issuingOrganization: Optional[str] = None
    issueDate: Optional[str] = None
    expiryDate: Optional[str] = None
    description: Optional[str] = None


class ResumeAnalyzeRequest(BaseModel):
    title: Optional[str] = None
    isDefault: Optional[bool] = None
    workExperiences: Optional[List[WorkExperience]] = []
    educations: Optional[List[Education]] = []
    awards: Optional[List[Award]] = []
    languages: Optional[List[Language]] = []
    portfolios: Optional[List[Portfolio]] = []
    coverLetter: Optional[CoverLetter] = None
    certifications: Optional[List[Certification]] = []


class PortfolioUrl(BaseModel):
    portfolioName: Optional[str] = None
    url: str


class BaseAnalyzeResponse(BaseModel):
    status: str
    user_id: str
    resume_id: str
    portfolio_received: bool
    portfolio_count: int
    portfolio_urls: List[PortfolioUrl]
    invalid_portfolio_urls: List[PortfolioUrl]
    portfolio_message: str


class ChartScore(BaseModel):
    category: str
    score: int


class ScoreResult(BaseModel):
    overall_score: int
    chart_scores: List[ChartScore]


class ScoreResponse(BaseAnalyzeResponse):
    score_result: ScoreResult


class FeedbackItem(BaseModel):
    criterion: str
    meaning: str
    reference_basis: str
    status: str
    level: str
    reason: str


class FeedbackResult(BaseModel):
    job_fit: FeedbackItem
    experience_specificity: FeedbackItem
    technical_competency: FeedbackItem
    document_quality: FeedbackItem
    consistency_uniqueness: FeedbackItem


class FeedbackResponse(BaseAnalyzeResponse):
    feedback_id: str
    feedback_result: FeedbackResult


class ActivityItem(BaseModel):
    title: str
    description: str
    category: str


class ActivityResult(BaseModel):
    recommended_activities: List[ActivityItem]


class ActivitiesResponse(BaseAnalyzeResponse):
    feedback_id: str
    activity_result: ActivityResult


def is_valid_url(url: Optional[str]) -> bool:
    if not url:
        return False
    parsed = urlparse(url)
    return parsed.scheme in ["http", "https"] and bool(parsed.netloc)


def extract_portfolio_urls(portfolios: List[Portfolio]):
    valid_urls = []
    invalid_urls = []
    for portfolio in portfolios:
        url = portfolio.url
        if is_valid_url(url):
            valid_urls.append({"portfolioName": portfolio.portfolioName, "url": url})
        elif url:
            invalid_urls.append({"portfolioName": portfolio.portfolioName, "url": url})
    return valid_urls, invalid_urls


def build_portfolio_status(request: ResumeAnalyzeRequest):
    valid_urls, invalid_urls = extract_portfolio_urls(request.portfolios or [])
    return {
        "portfolio_received": len(valid_urls) > 0,
        "portfolio_count": len(valid_urls),
        "portfolio_urls": valid_urls,
        "invalid_portfolio_urls": invalid_urls,
        "portfolio_message": (
            "포트폴리오 URL을 정상적으로 수신했습니다."
            if valid_urls
            else "수신된 유효한 포트폴리오 URL이 없습니다."
        ),
    }


HARDCODED_SCORE = {
    "overall_score": 73,
    "chart_scores": [
        {"category": "스킬", "score": 81},
        {"category": "경험", "score": 70},
        {"category": "포트폴리오", "score": 70},
        {"category": "직무적합성", "score": 70},
    ],
}

HARDCODED_FEEDBACK = {
    "job_fit": {
        "criterion": "직무 적합도",
        "meaning": "JD/NCS 기반 직무 연관성",
        "reference_basis": "NCS 직무기술서 + 고용24 채용공고",
        "status": "success",
        "level": "보통",
        "reason": "Python, SQL, Apache Spark, Kafka, Airflow, ETL 파이프라인 등 공고 핵심 요구 기술 다수를 보유하고 있으며, Kafka 실시간 스트리밍·Apache Spark 병렬 처리·ETL 파이프라인 구축 경험이 확인됩니다.",
    },
    "experience_specificity": {
        "criterion": "경험·성과 구체성",
        "meaning": "역할·행동·결과·수치",
        "reference_basis": "STAR 경험 평가 방식",
        "status": "success",
        "level": "우수",
        "reason": "프로젝트별 역할, 행동, 결과와 수치가 구체적으로 제시되어 있어 경험의 구체성이 높습니다.",
    },
    "technical_competency": {
        "criterion": "실무·기술 역량",
        "meaning": "기술 활용·구현·문제 해결",
        "reference_basis": "NCS 직무 수행 역량 + 개발자 포트폴리오 평가 관점",
        "status": "success",
        "level": "우수",
        "reason": "Kafka, Spark, TF-IDF, GraphRAG 등 기술을 실제 프로젝트 맥락에서 활용한 경험이 확인됩니다.",
    },
    "document_quality": {
        "criterion": "문서 완성도",
        "meaning": "가독성·구조·핵심 전달",
        "reference_basis": "ATS/이력서 첨삭 관점",
        "status": "success",
        "level": "우수",
        "reason": "역할, 설명, 결과, 수치가 포함되어 있고 핵심 키워드와 기술 스택이 정리되어 있어 문서 완성도가 높습니다.",
    },
    "consistency_uniqueness": {
        "criterion": "경험 일관성·차별성",
        "meaning": "경험 연결성·본인만의 강점",
        "reference_basis": "자기소개서 첨삭 및 직무 방향성 평가 관점",
        "status": "success",
        "level": "우수",
        "reason": "데이터 수집, 전처리, 분석, 추천 시스템 구현이라는 일관된 흐름이 확인되며 GraphRAG 활용 경험이 차별점으로 드러납니다.",
    },
}

HARDCODED_ACTIVITIES = {
    "recommended_activities": [
        {
            "title": "AWS Certified Data Engineer - Associate 취득",
            "description": "클라우드 데이터 파이프라인과 데이터 웨어하우스 활용 역량을 보완하기 위해 AWS 데이터 엔지니어링 자격증 취득을 추천합니다.",
            "category": "자격증",
        },
        {
            "title": "dbt + Airflow 기반 ELT 파이프라인 프로젝트",
            "description": "워크플로우 자동화와 데이터 변환 과정을 포트폴리오에 추가하여 실무형 데이터 엔지니어링 경험을 강화할 수 있습니다.",
            "category": "프로젝트",
        },
        {
            "title": "Kafka 실시간 파이프라인 경험 포트폴리오화",
            "description": "Kafka와 Spark 기반 실시간 처리 경험을 구조도, 처리 흐름, 성능 개선 결과 중심으로 정리하면 직무 적합도 보완에 도움이 됩니다.",
            "category": "포트폴리오 개선",
        },
    ]
}


@router.post("/{user_id}/{resume_id}/analyze/score", response_model=ScoreResponse)
def analyze_score(user_id: str, resume_id: str, request: ResumeAnalyzeRequest):
    portfolio_status = build_portfolio_status(request)
    return {"status": "success", "user_id": user_id, "resume_id": resume_id, **portfolio_status, "score_result": HARDCODED_SCORE}


@router.post("/{user_id}/{resume_id}/analyze/feedback", response_model=FeedbackResponse)
def analyze_feedback(user_id: str, resume_id: str, request: ResumeAnalyzeRequest):
    portfolio_status = build_portfolio_status(request)
    return {"status": "success", "user_id": user_id, "resume_id": resume_id, "feedback_id": "feedback-hardcoded-001", **portfolio_status, "feedback_result": HARDCODED_FEEDBACK}


@router.post("/{user_id}/{resume_id}/analyze/activities", response_model=ActivitiesResponse)
def analyze_activities(user_id: str, resume_id: str, request: ResumeAnalyzeRequest):
    portfolio_status = build_portfolio_status(request)
    return {"status": "success", "user_id": user_id, "resume_id": resume_id, "feedback_id": "feedback-hardcoded-001", **portfolio_status, "activity_result": HARDCODED_ACTIVITIES}


class ChatRequest(BaseModel):
    message: str


@router.post("/chat")
def chat(request: ChatRequest):
    answer = """
부산에서는 미취업 청년을 위한 다양한 지원 정책을 운영하고 있습니다.

🔥 취업 역량 강화 프로그램

* 부산 청년잡(JOB) 성장프로젝트
* 실업 초기 청년을 발굴해 역량강화부터 기업매칭까지 원스톱 지원

🔥 부산 청년일경험 지원사업

* 시청, 공공기관 등에서 실제 행정 경험 제공
* 맞춤형 취업상담 및 기업 탐방 기회 제공

🔥 부산 디지털 인재 양성 프로그램

* AI, 데이터, 클라우드 등 디지털 직무 교육 지원

🔥 부산청년 글로벌 잡(JOB) 챌린지 프로젝트

* 글로벌 기업 취업 대비 프로그램 운영

💡 보다 정확한 정책 안내를 원하시면 말씀해주세요.
"""
    return {
        "answer": answer,
        "sources": [],
        "intent": "PUBLIC_DATA",
        "user_message": {"role": "user", "content": request.message},
        "assistant_message": {"role": "assistant", "content": answer},
    }
