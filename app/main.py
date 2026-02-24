import asyncio
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.models import (
    AnalyzeResponse,
    StatusResponse,
    HealthResponse,
    AnalysisStatus
)
from app.services import database as db
from app.services.analyzer import analyze_attempt as run_analysis
from app.services.analyzer import analyze_full_video_attempt as run_full_video_analysis
from app.services.analyzer import analyze_mock_exam as run_mock_exam_analysis


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup: nothing to do (pool created lazily)
    yield
    # Shutdown: close database pool
    await db.close_pool()


app = FastAPI(
    title="Casper Prep Backend",
    description="Backend API for Casper video analysis",
    version="0.1.0",
    lifespan=lifespan
)

# In-process task registry used for cancellation endpoints.
_attempt_tasks: Dict[int, asyncio.Task] = {}
_mock_tasks: Dict[int, asyncio.Task] = {}


def _register_task(task_map: Dict[int, asyncio.Task], key: int, task: asyncio.Task) -> None:
    task_map[key] = task

    def _cleanup(done_task: asyncio.Task) -> None:
        if task_map.get(key) is done_task:
            task_map.pop(key, None)

    task.add_done_callback(_cleanup)

# CORS middleware - configure as needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="0.1.0"
    )


@app.post("/analyze/attempt/{attempt_id}", response_model=AnalyzeResponse)
async def analyze_attempt(attempt_id: int):
    """
    Trigger analysis for a scenario attempt.

    This will:
    1. Fetch all videos for the attempt from R2
    2. Process each video through the analysis pipeline
    3. Store results in the database

    The analysis runs in the background - use /analyze/status/{attempt_id}
    to check progress.

    Args:
        attempt_id: The scenario attempt ID to analyze

    Returns:
        Analysis status response
    """
    try:
        # Verify attempt exists
        attempt = await db.get_scenario_attempt(attempt_id)
        if not attempt:
            raise HTTPException(
                status_code=404,
                detail=f"Attempt {attempt_id} not found"
            )

        active_task = _attempt_tasks.get(attempt_id)
        if active_task and not active_task.done():
            return AnalyzeResponse(
                attempt_id=attempt_id,
                status=AnalysisStatus.PROCESSING,
                message=f"Analysis already in progress for attempt {attempt_id}"
            )

        # Check if already processing
        if attempt["feedback_status"] == "processing":
            return AnalyzeResponse(
                attempt_id=attempt_id,
                status=AnalysisStatus.PROCESSING,
                message=f"Analysis already in progress for attempt {attempt_id}"
            )

        # Run analysis in background
        task = asyncio.create_task(run_analysis_task(attempt_id))
        _register_task(_attempt_tasks, attempt_id, task)

        return AnalyzeResponse(
            attempt_id=attempt_id,
            status=AnalysisStatus.PROCESSING,
            message=f"Analysis started for attempt {attempt_id}"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start analysis: {str(e)}"
        )


async def run_analysis_task(attempt_id: int):
    """Background task wrapper for analysis."""
    try:
        print(f"[Background] Starting analysis for attempt {attempt_id}")
        result = await run_analysis(attempt_id)
        print(f"[Background] Analysis complete: {result}")
    except asyncio.CancelledError:
        print(f"[Background] Analysis cancelled for attempt {attempt_id}")
        raise
    except Exception as e:
        print(f"[Background] Analysis failed for attempt {attempt_id}: {e}")


@app.get("/analyze/status/{attempt_id}", response_model=StatusResponse)
async def get_analysis_status(attempt_id: int):
    """
    Check the status of an analysis job.

    Args:
        attempt_id: The scenario attempt ID

    Returns:
        Current status and progress of the analysis
    """
    try:
        attempt = await db.get_scenario_attempt(attempt_id)
        if not attempt:
            raise HTTPException(
                status_code=404,
                detail=f"Attempt {attempt_id} not found"
            )

        # Map database status to API status
        status_map = {
            "pending": AnalysisStatus.PENDING,
            "processing": AnalysisStatus.PROCESSING,
            "completed": AnalysisStatus.COMPLETED,
            "failed": AnalysisStatus.FAILED
        }
        status = status_map.get(attempt["feedback_status"], AnalysisStatus.PENDING)

        # Get responses to calculate progress
        responses = await db.get_question_responses(attempt_id)
        total = len(responses)
        completed = sum(1 for r in responses if r["feedback_status"] == "completed")
        progress = int((completed / total) * 100) if total > 0 else 0

        # Build result if completed
        result = None
        if status == AnalysisStatus.COMPLETED:
            result = {
                "overall_quartile": attempt.get("overall_quartile"),
                "overall_summary": attempt.get("overall_summary"),
                "questions_analyzed": completed
            }

        return StatusResponse(
            attempt_id=attempt_id,
            status=status,
            progress=progress,
            result=result
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get status: {str(e)}"
        )


@app.post("/analyze/attempt/{attempt_id}/sync")
async def analyze_attempt_sync(attempt_id: int):
    """
    Trigger analysis synchronously (waits for completion).
    Useful for testing. For production, use the async endpoint.
    """
    try:
        result = await run_analysis(attempt_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/retry/{attempt_id}")
async def retry_analysis(attempt_id: int):
    """
    Force retry analysis for an attempt (resets status and re-runs).
    Returns immediately, all work happens in background.
    """
    active_task = _attempt_tasks.get(attempt_id)
    if active_task and not active_task.done():
        active_task.cancel()

    task = asyncio.create_task(run_retry_task(attempt_id))
    _register_task(_attempt_tasks, attempt_id, task)
    return {"message": f"Retry queued for attempt {attempt_id}", "attempt_id": attempt_id}


async def run_retry_task(attempt_id: int):
    """Background task for retry - resets and re-analyzes."""
    try:
        print(f"[Background] Resetting attempt {attempt_id}")
        await db.reset_attempt_for_retry(attempt_id)
        print(f"[Background] Starting analysis for attempt {attempt_id}")
        result = await run_analysis(attempt_id)
        print(f"[Background] Analysis complete: {result}")
    except asyncio.CancelledError:
        print(f"[Background] Retry cancelled for attempt {attempt_id}")
        raise
    except Exception as e:
        print(f"[Background] Retry failed for attempt {attempt_id}: {e}")


@app.post("/analyze/full-video/{attempt_id}", response_model=AnalyzeResponse)
async def analyze_full_video(
    attempt_id: int,
    full_video_key: str
):
    """
    Trigger analysis for a full video (new flow).

    This endpoint handles the new workflow where one video contains all 3 question responses.

    Pipeline:
    1. Download and transcribe the full video
    2. Use LLM to segment the transcript by questions
    3. Chop the video into separate segments using ffmpeg
    4. Upload each segment to R2
    5. Update question responses with segment URLs
    6. Analyze each segment using the existing pipeline

    Args:
        attempt_id: The scenario attempt ID to analyze
        full_video_key: The R2 key of the full video

    Returns:
        Analysis status response
    """
    try:
        # Verify attempt exists
        attempt = await db.get_scenario_attempt(attempt_id)
        if not attempt:
            raise HTTPException(
                status_code=404,
                detail=f"Attempt {attempt_id} not found"
            )

        active_task = _attempt_tasks.get(attempt_id)
        if active_task and not active_task.done():
            return AnalyzeResponse(
                attempt_id=attempt_id,
                status=AnalysisStatus.PROCESSING,
                message=f"Analysis already in progress for attempt {attempt_id}"
            )

        # Check if already processing
        if attempt["feedback_status"] == "processing":
            return AnalyzeResponse(
                attempt_id=attempt_id,
                status=AnalysisStatus.PROCESSING,
                message=f"Analysis already in progress for attempt {attempt_id}"
            )

        # Run full video analysis in background
        task = asyncio.create_task(run_full_video_analysis_task(attempt_id, full_video_key))
        _register_task(_attempt_tasks, attempt_id, task)

        return AnalyzeResponse(
            attempt_id=attempt_id,
            status=AnalysisStatus.PROCESSING,
            message=f"Full video analysis started for attempt {attempt_id}"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start full video analysis: {str(e)}"
        )


async def run_full_video_analysis_task(attempt_id: int, full_video_key: str):
    """Background task wrapper for full video analysis."""
    try:
        print(f"[Background] Starting full video analysis for attempt {attempt_id}")
        result = await run_full_video_analysis(attempt_id, full_video_key)
        print(f"[Background] Full video analysis complete: {result}")
    except asyncio.CancelledError:
        print(f"[Background] Full video analysis cancelled for attempt {attempt_id}")
        raise
    except Exception as e:
        print(f"[Background] Full video analysis failed for attempt {attempt_id}: {e}")


@app.post("/analyze/full-video/{attempt_id}/sync")
async def analyze_full_video_sync(attempt_id: int, full_video_key: str):
    """
    Trigger full video analysis synchronously (waits for completion).
    Useful for testing. For production, use the async endpoint.
    """
    try:
        result = await run_full_video_analysis(attempt_id, full_video_key)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/mock-exam/{mock_exam_attempt_id}", response_model=AnalyzeResponse)
async def analyze_mock_exam(mock_exam_attempt_id: int):
    """
    Aggregate feedback for a completed mock exam.
    Call this after all scenario analyses are complete.
    """
    try:
        mock_attempt = await db.get_mock_exam_attempt(mock_exam_attempt_id)
        if not mock_attempt:
            raise HTTPException(
                status_code=404,
                detail=f"Mock exam attempt {mock_exam_attempt_id} not found"
            )

        active_task = _mock_tasks.get(mock_exam_attempt_id)
        if active_task and not active_task.done():
            return AnalyzeResponse(
                attempt_id=mock_exam_attempt_id,
                status=AnalysisStatus.PROCESSING,
                message=f"Mock exam aggregation already in progress"
            )

        if mock_attempt["feedback_status"] == "processing":
            return AnalyzeResponse(
                attempt_id=mock_exam_attempt_id,
                status=AnalysisStatus.PROCESSING,
                message=f"Mock exam aggregation already in progress"
            )

        if mock_attempt["feedback_status"] == "completed":
            return AnalyzeResponse(
                attempt_id=mock_exam_attempt_id,
                status=AnalysisStatus.COMPLETED,
                message=f"Mock exam feedback already generated"
            )

        task = asyncio.create_task(run_mock_exam_analysis_task(mock_exam_attempt_id))
        _register_task(_mock_tasks, mock_exam_attempt_id, task)

        return AnalyzeResponse(
            attempt_id=mock_exam_attempt_id,
            status=AnalysisStatus.PROCESSING,
            message=f"Mock exam aggregation started"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start mock exam aggregation: {str(e)}"
        )


async def run_mock_exam_analysis_task(mock_exam_attempt_id: int):
    try:
        print(f"[Background] Starting mock exam aggregation for {mock_exam_attempt_id}")
        result = await run_mock_exam_analysis(mock_exam_attempt_id)
        print(f"[Background] Mock exam aggregation complete: {result}")
    except asyncio.CancelledError:
        print(f"[Background] Mock exam aggregation cancelled for {mock_exam_attempt_id}")
        raise
    except Exception as e:
        print(f"[Background] Mock exam aggregation failed for {mock_exam_attempt_id}: {e}")


@app.post("/analyze/cancel/attempt/{attempt_id}")
async def cancel_attempt_analysis(attempt_id: int):
    task = _attempt_tasks.get(attempt_id)
    was_running = bool(task and not task.done())

    if task and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[Background] Attempt cancel cleanup error for {attempt_id}: {e}")

    attempt = await db.get_scenario_attempt(attempt_id)
    if attempt:
        await db.update_attempt_feedback_status(attempt_id, "failed")

    return {
        "attempt_id": attempt_id,
        "cancelled": was_running,
        "message": "Cancellation requested" if was_running else "No active job found"
    }


@app.post("/analyze/cancel/mock-exam/{mock_exam_attempt_id}")
async def cancel_mock_exam_analysis(mock_exam_attempt_id: int):
    task = _mock_tasks.get(mock_exam_attempt_id)
    was_running = bool(task and not task.done())

    if task and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[Background] Mock cancel cleanup error for {mock_exam_attempt_id}: {e}")

    mock_attempt = await db.get_mock_exam_attempt(mock_exam_attempt_id)
    if mock_attempt:
        await db.update_mock_exam_attempt_status(mock_exam_attempt_id, "failed")

    return {
        "mock_exam_attempt_id": mock_exam_attempt_id,
        "cancelled": was_running,
        "message": "Cancellation requested" if was_running else "No active job found"
    }


@app.get("/analyze/mock-exam/{mock_exam_attempt_id}/status")
async def get_mock_exam_status(mock_exam_attempt_id: int):
    """Check status of mock exam analysis and whether aggregation can be triggered."""
    try:
        mock_attempt = await db.get_mock_exam_attempt(mock_exam_attempt_id)
        if not mock_attempt:
            raise HTTPException(
                status_code=404,
                detail=f"Mock exam attempt {mock_exam_attempt_id} not found"
            )

        scenario_attempts = await db.get_scenario_attempts_for_mock(mock_exam_attempt_id)
        
        total = len(scenario_attempts)
        completed = sum(1 for sa in scenario_attempts if sa["feedback_status"] == "completed")
        failed = sum(1 for sa in scenario_attempts if sa["feedback_status"] == "failed")
        
        ready_for_aggregation = completed == total and total > 0

        return {
            "mock_exam_attempt_id": mock_exam_attempt_id,
            "mock_feedback_status": mock_attempt["feedback_status"],
            "scenarios": {
                "total": total,
                "completed": completed,
                "failed": failed,
                "pending": total - completed - failed
            },
            "ready_for_aggregation": ready_for_aggregation,
            "overall_quartile": mock_attempt.get("overall_quartile"),
            "overall_summary": mock_attempt.get("overall_summary")
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get mock exam status: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
