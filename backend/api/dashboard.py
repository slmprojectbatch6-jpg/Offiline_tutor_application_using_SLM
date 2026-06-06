"""
API Routes: Dashboard - Stats, Progress, Personalization
"""
from typing import List
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.models import get_db
from backend.database.crud import (
    get_user_stats, get_weak_subjects, get_frequent_topics
)
from backend.api.auth import get_current_user

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

PRACTICE_QUESTIONS = {
    "history": [
        "What were the main causes of the French Revolution?",
        "Explain the significance of the Quit India Movement.",
        "Who was Ashoka and what were his major contributions?",
        "What is the importance of the Harappan Civilization?",
        "Describe the impact of colonialism on India.",
    ],
    "polity": [
        "What are Fundamental Rights in the Indian Constitution?",
        "Explain the role of the Parliament in India.",
        "What is the difference between Lok Sabha and Rajya Sabha?",
        "What are Directive Principles of State Policy?",
        "How does the Supreme Court of India function?",
    ],
    "geography": [
        "What are the major physiographic divisions of India?",
        "Explain the monsoon system in India.",
        "What are natural resources and why are they important?",
        "Describe the major rivers of India.",
        "What is climate change and how does it affect India?",
    ],
    "economics": [
        "What is the difference between developed and developing countries?",
        "Explain the concept of GDP and its importance.",
        "What is poverty and how is it measured in India?",
        "What are the sectors of the Indian economy?",
        "Explain the role of banks in the economy.",
    ],
}

REVISION_TOPICS = {
    "history": ["Ancient India", "Medieval India", "Mughal Empire", "Freedom Struggle", "French Revolution"],
    "polity": ["Constitution", "Fundamental Rights", "Parliament", "Federalism", "Local Government"],
    "geography": ["Physical Features", "Climate", "Natural Vegetation", "Agriculture", "Industries"],
    "economics": ["Development", "Sectors of Economy", "Money and Credit", "Globalisation", "Consumer Rights"],
}


@router.get("/stats")
async def get_dashboard_stats(
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Get full dashboard statistics for the current user."""
    stats = await get_user_stats(db, current_user.id)
    weak = await get_weak_subjects(db, current_user.id)
    frequent = await get_frequent_topics(db, current_user.id)

    return {
        "user": {
            "id": current_user.id,
            "username": current_user.username,
            "full_name": current_user.full_name,
            "class_level": current_user.class_level,
            "avatar_color": current_user.avatar_color,
        },
        **stats,
        "weak_subjects": weak,
        "frequent_topics": frequent,
    }


@router.get("/suggestions")
async def get_personalized_suggestions(
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Return personalized practice questions and revision topics."""
    weak = await get_weak_subjects(db, current_user.id)
    frequent = await get_frequent_topics(db, current_user.id)

    # Build suggestions prioritizing weak subjects
    suggestions = {}
    all_subjects = ["history", "polity", "geography", "economics"]
    priority_subjects = weak + [s for s in all_subjects if s not in weak]

    for subject in priority_subjects[:4]:
        suggestions[subject] = {
            "practice_questions": PRACTICE_QUESTIONS.get(subject, [])[:3],
            "revision_topics": REVISION_TOPICS.get(subject, [])[:4],
            "is_weak": subject in weak,
        }

    return {
        "suggestions": suggestions,
        "frequent_topics": frequent,
        "weak_subjects": weak,
    }


@router.get("/leaderboard")
async def get_subjects_overview():
    """Return subject information (static NCERT overview)."""
    return {
        "subjects": [
            {
                "id": "history",
                "name": "History",
                "icon": "🏛️",
                "description": "Ancient, Medieval & Modern Indian History",
                "chapters": ["Tracing Changes", "New Kings and Kingdoms", "Delhi Sultans",
                             "Mughal Empire", "Rulers and Buildings", "The Making of Regional Cultures"],
                "color": "#f59e0b",
            },
            {
                "id": "polity",
                "name": "Political Science",
                "icon": "⚖️",
                "description": "Indian Constitution, Democracy & Governance",
                "chapters": ["Understanding Diversity", "Diversity and Discrimination",
                             "What is Government", "Key Elements of a Democratic Government",
                             "Panchayati Raj", "Rural Administration"],
                "color": "#6366f1",
            },
            {
                "id": "geography",
                "name": "Geography",
                "icon": "🌏",
                "description": "Physical Features, Climate & Natural Resources",
                "chapters": ["The Earth in the Solar System", "Globe", "Motions of the Earth",
                             "Maps", "Major Domains of the Earth", "Major Landforms"],
                "color": "#10b981",
            },
            {
                "id": "economics",
                "name": "Economics",
                "icon": "📊",
                "description": "Development, Markets & Indian Economy",
                "chapters": ["Development", "Sectors of the Indian Economy",
                             "Money and Credit", "Globalisation and the Indian Economy",
                             "Consumer Rights"],
                "color": "#ef4444",
            },
        ]
    }
