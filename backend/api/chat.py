"""
API Routes: Chat Interface (RAG-powered Q&A with streaming)
"""
import time
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.models import get_db
from backend.database.crud import (
    create_chat_session, get_user_sessions, get_session_messages,
    save_message, update_session_title, increment_subject_questions,
    track_topic,
)
from backend.api.auth import get_current_user
from backend.rag.pipeline import run_rag_query, run_rag_stream, retrieve_chunks

router = APIRouter(prefix="/api/chat", tags=["chat"])


# ── Schemas ───────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: Optional[int] = None
    message: str
    subject: Optional[str] = None   # Filter by subject
    stream: bool = False


class NewSessionRequest(BaseModel):
    title: str = "New Chat"
    subject: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────

@router.post("/sessions")
async def create_session(
    req: NewSessionRequest,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Create a new chat session."""
    session = await create_chat_session(
        db, user_id=current_user.id, title=req.title, subject=req.subject
    )
    return {
        "session_id": session.id,
        "title": session.title,
        "subject": session.subject,
        "created_at": session.created_at,
    }


@router.get("/sessions")
async def list_sessions(
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Get all chat sessions for current user."""
    sessions = await get_user_sessions(db, current_user.id)
    return [
        {
            "session_id": s.id,
            "title": s.title,
            "subject": s.subject,
            "updated_at": s.updated_at,
            "created_at": s.created_at,
        }
        for s in sessions
    ]


@router.get("/sessions/{session_id}/messages")
async def get_messages(
    session_id: int,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Load all messages in a session."""
    messages = await get_session_messages(db, session_id)
    return [
        {
            "id": m.id,
            "role": m.role,
            "content": m.content,
            "sources": m.sources,
            "subject": m.subject,
            "response_time_ms": m.response_time_ms,
            "created_at": m.created_at,
        }
        for m in messages
    ]


@router.post("/ask")
async def ask_question(
    req: ChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Ask a question (non-streaming RAG response)."""
    # Create session if not provided
    if not req.session_id:
        session = await create_chat_session(
            db,
            user_id=current_user.id,
            title=req.message[:60] + ("..." if len(req.message) > 60 else ""),
            subject=req.subject,
        )
        session_id = session.id
    else:
        session_id = req.session_id

    # Load past messages to enrich context
    past_messages = await get_session_messages(db, session_id)
    history_str = ""
    # Only take last 4 messages and format them
    if past_messages:
        recent = past_messages[-4:]
        history_parts = []
        for p in recent:
            history_parts.append(f"{p.role.capitalize()}: {p.content[:200]}") # limit length of each interaction to save context
        history_str = "\n".join(history_parts)

    enriched_query = f"Previous Conversation Context:\n{history_str}\n\nCurrent Question: {req.message}" if history_str else req.message

    # Save user message
    await save_message(db, session_id, "user", req.message, subject=req.subject)

    # Run RAG
    start_time = time.time()
    try:
        result = run_rag_query(
            query=req.message,
            subject_filter=req.subject,
            history=history_str,
        )
        answer = result["answer"]
        sources = result.get("sources", [])
    except Exception as e:
        answer = f"An error occurred: {str(e)}"
        sources = []

    elapsed_ms = (time.time() - start_time) * 1000

    # Detect subject from retrieved chunks
    detected_subject = req.subject
    if not detected_subject and sources:
        detected_subject = sources[0].get("subject")

    # Save assistant message
    await save_message(
        db, session_id, "assistant", answer,
        sources=sources,
        subject=detected_subject,
        response_time_ms=round(elapsed_ms, 2),
    )

    # Update progress & topic tracking
    if detected_subject:
        chapter = sources[0].get("chapter") if sources else None
        await increment_subject_questions(db, current_user.id, detected_subject, chapter)
        if chapter:
            await track_topic(db, current_user.id, detected_subject, chapter)

    # Update session title on first question
    if not req.session_id:
        await update_session_title(db, session_id, req.message[:80])

    return {
        "session_id": session_id,
        "answer": answer,
        "sources": sources,
        "response_time_ms": round(elapsed_ms, 2),
        "subject": detected_subject,
    }


@router.post("/ask/stream")
async def ask_question_stream(
    req: ChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Ask a question with streaming (SSE) response."""
    # Create session if needed
    if not req.session_id:
        session = await create_chat_session(
            db,
            user_id=current_user.id,
            title=req.message[:60],
            subject=req.subject,
        )
        session_id = session.id
    else:
        session_id = req.session_id

    # Load past messages to enrich context
    past_messages = await get_session_messages(db, session_id)
    history_str = ""
    # Only take last 4 messages and format them
    if past_messages:
        recent = past_messages[-4:]
        history_parts = []
        for p in recent:
            history_parts.append(f"{p.role.capitalize()}: {p.content[:200]}")
        history_str = "\n".join(history_parts)

    await save_message(db, session_id, "user", req.message, subject=req.subject)

    full_response = []

    def token_generator():
        for token in run_rag_stream(req.message, subject_filter=req.subject, history=history_str):
            full_response.append(token)
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")


@router.get("/retrieve")
async def retrieve_context(
    query: str,
    subject: Optional[str] = None,
    top_k: int = 4,
    current_user=Depends(get_current_user),
):
    """Retrieve raw FAISS chunks for a query (debug/explore use)."""
    try:
        chunks = retrieve_chunks(query, subject_filter=subject, top_k=top_k)
        return {"query": query, "chunks": chunks}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: int,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Delete a chat session."""
    from sqlalchemy import delete
    from backend.database.models import ChatSession, ChatMessage
    await db.execute(delete(ChatMessage).where(ChatMessage.session_id == session_id))
    await db.execute(
        delete(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.user_id == current_user.id,
        )
    )
    await db.commit()
    return {"message": "Session deleted"}
