"""
Database service for CasperPrep backend.
Handles all database operations for analysis.
"""

import json
import asyncpg
from typing import Optional, List, Dict, Any
from app.config import get_settings_sync

_pool: Optional[asyncpg.Pool] = None


async def get_pool() -> asyncpg.Pool:
    """Get or create database connection pool."""
    global _pool
    if _pool is None:
        settings = get_settings_sync()
        _pool = await asyncpg.create_pool(
            settings.database_url,
            min_size=2,
            max_size=10,
            statement_cache_size=0  # Required for pgbouncer transaction mode
        )
    return _pool


async def close_pool():
    """Close the database pool."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


# =============================================================================
# READ OPERATIONS
# =============================================================================

async def get_scenario_attempt(attempt_id: int) -> Optional[Dict[str, Any]]:
    """Get a scenario attempt by ID."""
    pool = await get_pool()
    row = await pool.fetchrow(
        """
        SELECT
            sa.id, sa.user_id, sa.scenario_id, sa.session_id, sa.mock_exam_attempt_id,
            sa.status, sa.feedback_status,
            sa.overall_quartile, sa.overall_scores, sa.overall_summary,
            sa.overall_strengths, sa.overall_improvements,
            s.title as scenario_title, s.type as scenario_type,
            s.content as scenario_content, s.description as scenario_description, s.questions
        FROM scenario_attempts sa
        JOIN scenarios s ON s.id = sa.scenario_id
        WHERE sa.id = $1
        """,
        attempt_id
    )
    return dict(row) if row else None


async def get_question_responses(attempt_id: int) -> List[Dict[str, Any]]:
    """Get all question responses for an attempt."""
    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT
            id, attempt_id, question_index, question_text,
            type, text_content, video_url, video_duration,
            response_time_ms, feedback_status
        FROM question_responses
        WHERE attempt_id = $1
        ORDER BY question_index
        """,
        attempt_id
    )
    return [dict(row) for row in rows]


# =============================================================================
# UPDATE OPERATIONS
# =============================================================================

async def update_question_response_analysis(
    response_id: int,
    video_analysis: Optional[Dict[str, Any]],
    score: float,
    strengths: List[str],
    improvements: List[str],
    optimal_response: Optional[str] = None,
    feedback_status: str = "completed"
):
    pool = await get_pool()
    await pool.execute(
        """
        UPDATE question_responses
        SET
            video_analysis = $2::jsonb,
            score = $3,
            strengths = $4,
            improvements = $5,
            optimal_response = $6,
            feedback_status = $7
        WHERE id = $1
        """,
        response_id,
        json.dumps(video_analysis) if video_analysis else None,
        score,
        strengths,
        improvements,
        optimal_response,
        feedback_status
    )


async def update_scenario_attempt_feedback(
    attempt_id: int,
    overall_quartile: int,
    overall_scores: Dict[str, float],
    overall_strengths: List[str],
    overall_improvements: List[str],
    overall_summary: str,
    feedback_status: str = "completed"
):
    """Update scenario attempt with aggregated feedback."""
    pool = await get_pool()
    await pool.execute(
        """
        UPDATE scenario_attempts
        SET
            feedback_status = $2,
            overall_quartile = $3,
            overall_scores = $4::jsonb,
            overall_strengths = $5,
            overall_improvements = $6,
            overall_summary = $7,
            feedback_generated_at = NOW()
        WHERE id = $1
        """,
        attempt_id,
        feedback_status,
        overall_quartile,
        json.dumps(overall_scores),
        overall_strengths,
        overall_improvements,
        overall_summary
    )


async def update_attempt_feedback_status(attempt_id: int, status: str):
    """Update just the feedback status of an attempt."""
    pool = await get_pool()
    await pool.execute(
        """
        UPDATE scenario_attempts
        SET feedback_status = $2
        WHERE id = $1
        """,
        attempt_id,
        status
    )


async def update_response_feedback_status(response_id: int, status: str):
    """Update just the feedback status of a response."""
    pool = await get_pool()
    await pool.execute(
        """
        UPDATE question_responses
        SET feedback_status = $2
        WHERE id = $1
        """,
        response_id,
        status
    )


async def update_question_response_video_url(response_id: int, video_url: str):
    """Update the video URL for a question response (used after segmentation)."""
    pool = await get_pool()
    await pool.execute(
        """
        UPDATE question_responses
        SET video_url = $2
        WHERE id = $1
        """,
        response_id,
        video_url
    )


async def reset_attempt_for_retry(attempt_id: int):
    """Reset attempt and all its responses to pending in a single transaction."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "UPDATE scenario_attempts SET feedback_status = 'pending' WHERE id = $1",
                attempt_id
            )
            await conn.execute(
                "UPDATE question_responses SET feedback_status = 'pending' WHERE attempt_id = $1",
                attempt_id
            )


async def get_mock_exam_attempt(mock_exam_attempt_id: int) -> Optional[Dict[str, Any]]:
    pool = await get_pool()
    row = await pool.fetchrow(
        """
        SELECT
            mea.id, mea.user_id, mea.mock_exam_id, mea.status,
            mea.feedback_status, mea.overall_quartile, mea.overall_scores,
            mea.overall_strengths, mea.overall_improvements, mea.overall_summary,
            me.name as exam_name
        FROM mock_exam_attempts mea
        JOIN mock_exams me ON me.id = mea.mock_exam_id
        WHERE mea.id = $1
        """,
        mock_exam_attempt_id
    )
    return dict(row) if row else None


async def get_scenario_attempts_for_mock(mock_exam_attempt_id: int) -> List[Dict[str, Any]]:
    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT
            sa.id, sa.scenario_id, sa.feedback_status,
            sa.overall_quartile, sa.overall_scores,
            sa.overall_strengths, sa.overall_improvements, sa.overall_summary,
            s.title as scenario_title
        FROM scenario_attempts sa
        JOIN scenarios s ON s.id = sa.scenario_id
        WHERE sa.mock_exam_attempt_id = $1
        ORDER BY sa.id
        """,
        mock_exam_attempt_id
    )
    return [dict(row) for row in rows]


async def update_mock_exam_attempt_feedback(
    mock_exam_attempt_id: int,
    overall_quartile: int,
    overall_scores: Dict[str, float],
    overall_strengths: List[str],
    overall_improvements: List[str],
    overall_summary: str,
    feedback_status: str = "completed"
):
    pool = await get_pool()
    await pool.execute(
        """
        UPDATE mock_exam_attempts
        SET
            feedback_status = $2,
            overall_quartile = $3,
            overall_scores = $4::jsonb,
            overall_strengths = $5,
            overall_improvements = $6,
            overall_summary = $7,
            feedback_generated_at = NOW()
        WHERE id = $1
        """,
        mock_exam_attempt_id,
        feedback_status,
        overall_quartile,
        json.dumps(overall_scores),
        overall_strengths,
        overall_improvements,
        overall_summary
    )


async def update_mock_exam_attempt_status(mock_exam_attempt_id: int, status: str):
    pool = await get_pool()
    await pool.execute(
        "UPDATE mock_exam_attempts SET feedback_status = $2 WHERE id = $1",
        mock_exam_attempt_id,
        status
    )
