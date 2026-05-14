from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
import logging

from .config import USER_DB_URL, JOB_DB_URL

logger = logging.getLogger(__name__)


# ── 엔진 ─────────────────────────────────────────────────────
user_engine = create_async_engine(USER_DB_URL, echo=False, pool_pre_ping=True, pool_size=5)
UserSession = async_sessionmaker(user_engine, expire_on_commit=False)

job_engine  = create_async_engine(JOB_DB_URL,  echo=False, pool_pre_ping=True, pool_size=5)
JobSession  = async_sessionmaker(job_engine,  expire_on_commit=False)


class UserBase(DeclarativeBase):
    pass

class JobBase(DeclarativeBase):
    pass


# ── 의존성 주입 ───────────────────────────────────────────────
async def get_user_db():
    async with UserSession() as session:
        yield session

async def get_job_db():
    async with JobSession() as session:
        yield session


# ── 테이블 생성 (테이블이 없는 경우에만 생성) ─────────────────
async def init_db():
    from .models import user, job, log  # noqa

    async with user_engine.begin() as conn:
        await conn.run_sync(UserBase.metadata.create_all)

    async with job_engine.begin() as conn:
        await conn.run_sync(JobBase.metadata.create_all)
