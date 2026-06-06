"""
CRUD Operations for all database models
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy import select, func, desc, update
from sqlalchemy.ext.asyncio import AsyncSession
from passlib.context import CryptContext

from backend.database.models import (
    User, ChatSession, ChatMessage, SubjectProgress, TopicTracking
)

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


# ── Auth Helpers ──────────────────────────────────────────────

def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# ── User Operations ───────────────────────────────────────────

async def create_user(
    db: AsyncSession,
    username: str,
    full_name: str,
    password: str,
    email: Optional[str] = None,
    class_level: Optional[int] = None,
) -> User:
    user = User(
        username=username,
        full_name=full_name,
        email=email,
        hashed_password=hash_password(password),
        class_level=class_level,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


async def get_user_by_username(db: AsyncSession, username: str) -> Optional[User]:
    result = await db.execute(select(User).where(User.username == username))
    return result.scalar_one_or_none()


async def get_user_by_id(db: AsyncSession, user_id: int) -> Optional[User]:
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()


async def update_last_login(db: AsyncSession, user_id: int):
    await db.execute(
        update(User).where(User.id == user_id).values(last_login=datetime.utcnow())
    )
    await db.commit()


# ── Chat Session Operations ───────────────────────────────────

async def create_chat_session(
    db: AsyncSession,
    user_id: int,
    title: str = "New Chat",
    subject: Optional[str] = None,
) -> ChatSession:
    session = ChatSession(user_id=user_id, title=title, subject=subject)
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session


async def get_user_sessions(
    db: AsyncSession, user_id: int, limit: int = 20
) -> List[ChatSession]:
    result = await db.execute(
        select(ChatSession)
        .where(ChatSession.user_id == user_id)
        .order_by(desc(ChatSession.updated_at))
        .limit(limit)
    )
    return result.scalars().all()


async def get_session_messages(
    db: AsyncSession, session_id: int
) -> List[ChatMessage]:
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at)
    )
    return result.scalars().all()


async def save_message(
    db: AsyncSession,
    session_id: int,
    role: str,
    content: str,
    sources: Optional[List[Dict]] = None,
    subject: Optional[str] = None,
    response_time_ms: Optional[float] = None,
) -> ChatMessage:
    msg = ChatMessage(
        session_id=session_id,
        role=role,
        content=content,
        sources=sources,
        subject=subject,
        response_time_ms=response_time_ms,
    )
    db.add(msg)
    # Update session timestamp
    await db.execute(
        update(ChatSession)
        .where(ChatSession.id == session_id)
        .values(updated_at=datetime.utcnow())
    )
    await db.commit()
    await db.refresh(msg)
    return msg


async def update_session_title(
    db: AsyncSession, session_id: int, title: str
):
    await db.execute(
        update(ChatSession)
        .where(ChatSession.id == session_id)
        .values(title=title[:200])
    )
    await db.commit()


# ── Progress Operations ───────────────────────────────────────

async def get_or_create_progress(
    db: AsyncSession, user_id: int, subject: str
) -> SubjectProgress:
    result = await db.execute(
        select(SubjectProgress).where(
            SubjectProgress.user_id == user_id,
            SubjectProgress.subject == subject,
        )
    )
    progress = result.scalar_one_or_none()
    if not progress:
        progress = SubjectProgress(user_id=user_id, subject=subject)
        db.add(progress)
        await db.commit()
        await db.refresh(progress)
    return progress


async def increment_subject_questions(
    db: AsyncSession, user_id: int, subject: str, topic: Optional[str] = None
):
    progress = await get_or_create_progress(db, user_id, subject)
    progress.questions_asked += 1
    progress.last_activity = datetime.utcnow()
    if topic and topic not in (progress.topics_explored or []):
        topics = list(progress.topics_explored or [])
        topics.append(topic)
        progress.topics_explored = topics[-20:]  # Keep last 20
    await db.commit()


async def get_user_stats(db: AsyncSession, user_id: int) -> Dict[str, Any]:
    """Get comprehensive user activity statistics."""
    # Total messages
    total_msgs_query = await db.execute(
        select(func.count(ChatMessage.id))
        .join(ChatSession)
        .where(
            ChatSession.user_id == user_id,
            ChatMessage.role == "user"
        )
    )
    total_messages = total_msgs_query.scalar() or 0

    # Subject progress
    progress_result = await db.execute(
        select(SubjectProgress).where(SubjectProgress.user_id == user_id)
    )
    progress_rows = progress_result.scalars().all()

    subject_stats = {row.subject: row.questions_asked for row in progress_rows}

    # Most active subject
    most_active = max(subject_stats, key=subject_stats.get) if subject_stats else None

    # Total sessions
    session_count = await db.execute(
        select(func.count(ChatSession.id)).where(ChatSession.user_id == user_id)
    )
    total_sessions = session_count.scalar() or 0

    return {
        "total_messages": total_messages,
        "total_sessions": total_sessions,
        "subject_stats": subject_stats,
        "most_active_subject": most_active,
        "progress": [
            {
                "subject": row.subject,
                "questions_asked": row.questions_asked,
                "topics_explored": len(row.topics_explored or []),
                "last_activity": row.last_activity.isoformat() if row.last_activity else None,
            }
            for row in progress_rows
        ],
    }


# ── Topic Tracking ─────────────────────────────────────────────

async def track_topic(
    db: AsyncSession, user_id: int, subject: str, topic: str
):
    result = await db.execute(
        select(TopicTracking).where(
            TopicTracking.user_id == user_id,
            TopicTracking.subject == subject,
            TopicTracking.topic == topic,
        )
    )
    existing = result.scalar_one_or_none()
    if existing:
        existing.frequency += 1
        existing.last_asked = datetime.utcnow()
    else:
        db.add(TopicTracking(user_id=user_id, subject=subject, topic=topic))
    await db.commit()


async def get_weak_subjects(db: AsyncSession, user_id: int) -> List[str]:
    """Return subjects with fewer questions (potential weak areas)."""
    result = await db.execute(
        select(SubjectProgress)
        .where(SubjectProgress.user_id == user_id)
        .order_by(SubjectProgress.questions_asked)
    )
    all_progress = result.scalars().all()
    return [p.subject for p in all_progress[:2]]


async def get_frequent_topics(
    db: AsyncSession, user_id: int, limit: int = 5
) -> List[Dict]:
    result = await db.execute(
        select(TopicTracking)
        .where(TopicTracking.user_id == user_id)
        .order_by(desc(TopicTracking.frequency))
        .limit(limit)
    )
    topics = result.scalars().all()
    return [
        {"subject": t.subject, "topic": t.topic, "frequency": t.frequency}
        for t in topics
    ]
