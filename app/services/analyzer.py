"""
Main analysis orchestration service.
Coordinates video download, analysis, and feedback generation.
"""

import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor

from app.services.r2 import r2_service
from app.services.audio import analyze_audio
from app.services.video import analyze_eye_contact
from app.services.feedback import generate_question_feedback, generate_scenario_feedback
from app.services.segmenter import process_full_video
from app.services import database as db

# Thread pool for CPU-bound tasks (Whisper, OpenCV)
_executor = ThreadPoolExecutor(max_workers=2)


def calculate_quartile(avg_score: float) -> int:
    """
    Calculate quartile based on average score (1-9 scale).
    Based on CASPer percentile distribution.
    """
    if avg_score >= 7.0:
        return 4  # Top 25%
    elif avg_score >= 5.5:
        return 3  # 50-75%
    elif avg_score >= 4.0:
        return 2  # 25-50%
    else:
        return 1  # Bottom 25%


def average_competency_scores(question_feedbacks: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Average competency scores across all question feedbacks.
    Returns dict with empathy, communication, ethical_reasoning, professionalism.
    """
    competencies = ["empathy", "communication", "ethical_reasoning", "professionalism"]
    totals = {c: 0.0 for c in competencies}
    counts = {c: 0 for c in competencies}
    
    for feedback in question_feedbacks:
        scores = feedback.get("competency_scores", {})
        for c in competencies:
            if c in scores:
                totals[c] += float(scores[c])
                counts[c] += 1
    
    return {
        c: round(totals[c] / counts[c], 1) if counts[c] > 0 else 5.0
        for c in competencies
    }


async def analyze_attempt(attempt_id: int) -> Dict[str, Any]:
    """
    Full analysis pipeline for a scenario attempt.

    1. Fetch attempt and responses from database
    2. For each video response:
       - Download from R2
       - Run audio analysis (Whisper + filler words)
       - Run eye contact analysis
       - Generate LLM feedback
       - Save to database
    3. Generate scenario-level feedback
    4. Save scenario feedback

    Returns summary of analysis results.
    """
    # Mark attempt as processing
    await db.update_attempt_feedback_status(attempt_id, "processing")

    try:
        # Get attempt info
        attempt = await db.get_scenario_attempt(attempt_id)
        if not attempt:
            raise ValueError(f"Attempt {attempt_id} not found")

        # Get all question responses
        responses = await db.get_question_responses(attempt_id)
        if not responses:
            raise ValueError(f"No responses found for attempt {attempt_id}")

        # Analyze each response
        question_feedbacks = []
        for response in responses:
            if response["type"] == "video" and response["video_url"]:
                feedback = await analyze_video_response(
                    response=response,
                    user_id=str(attempt["user_id"]),
                    attempt_id=attempt_id
                )
                question_feedbacks.append(feedback)
            elif response["type"] == "text" and response["text_content"]:
                feedback = await analyze_text_response(
                    response=response,
                    scenario_content=attempt["scenario_content"]
                )
                question_feedbacks.append(feedback)

        # Calculate competency scores by averaging across questions
        overall_scores = average_competency_scores(question_feedbacks)
        
        # Calculate overall score (average of all competencies)
        avg_score = sum(overall_scores.values()) / len(overall_scores) if overall_scores else 5.0
        
        # Calculate quartile based on average score
        overall_quartile = calculate_quartile(avg_score)

        # Generate scenario-level feedback (key takeaways only)
        scenario_feedback = await generate_scenario_feedback(
            scenario_title=attempt["scenario_title"],
            question_feedbacks=question_feedbacks
        )

        # Save scenario feedback
        await db.update_scenario_attempt_feedback(
            attempt_id=attempt_id,
            overall_quartile=overall_quartile,
            overall_scores=overall_scores,
            overall_strengths=scenario_feedback["strengths"],
            overall_improvements=scenario_feedback["improvements"],
            overall_summary=scenario_feedback["summary"],
            feedback_status="completed"
        )

        return {
            "attempt_id": attempt_id,
            "status": "completed",
            "questions_analyzed": len(question_feedbacks),
            "overall_score": avg_score,
            "overall_quartile": overall_quartile
        }

    except Exception as e:
        # Mark as failed
        await db.update_attempt_feedback_status(attempt_id, "failed")
        raise e


async def analyze_video_response(
    response: Dict[str, Any],
    user_id: str,
    attempt_id: int,
    segment_start_time: Optional[float] = None,
    segment_end_time: Optional[float] = None
) -> Dict[str, Any]:
    """Analyze a single video response."""
    response_id = response["id"]
    video_key = response["video_url"]  # This is the R2 key
    question_text = response["question_text"]

    # Mark as processing
    await db.update_response_feedback_status(response_id, "processing")

    video_path: Optional[Path] = None

    try:
        # Download video from R2
        print(f"[Analyzer] Downloading video: {video_key}")
        loop = asyncio.get_event_loop()
        video_path = await loop.run_in_executor(
            _executor,
            r2_service.download_video,
            video_key
        )
        print(f"[Analyzer] Downloaded to: {video_path}")

        # Run audio analysis (in thread pool - CPU bound)
        print(f"[Analyzer] Running audio analysis...")
        audio_result = await loop.run_in_executor(
            _executor,
            analyze_audio,
            video_path
        )
        print(f"[Analyzer] Audio analysis complete. Transcript: {audio_result['transcript'][:100]}...")

        # Run eye contact analysis (in thread pool - CPU bound)
        print(f"[Analyzer] Running eye contact analysis...")
        eye_result = await loop.run_in_executor(
            _executor,
            analyze_eye_contact,
            video_path,
            5  # sample_rate
        )
        print(f"[Analyzer] Eye contact: {eye_result['eye_contact_percentage']}%")

        # Generate LLM feedback
        print(f"[Analyzer] Generating LLM feedback...")
        llm_feedback = await generate_question_feedback(
            question_text=question_text,
            transcript=audio_result["transcript"],
            filler_word_count=len(audio_result["filler_words"]),
            eye_contact_pct=eye_result["eye_contact_percentage"],
            words_per_minute=audio_result["words_per_minute"]
        )
        print(f"[Analyzer] LLM feedback generated. Score: {llm_feedback['score']}")

        # Build video_analysis JSON
        video_analysis = {
            "transcript": audio_result["transcript"],
            "words": audio_result["words"],  # All words with timestamps for clickable transcript
            "eyeContactPct": eye_result["eye_contact_percentage"],
            "eyeContactTimeline": eye_result["timeline"],  # For graphing
            "eyeContactIssues": eye_result["issues"],
            "fillerWords": audio_result["filler_words"],
            "fillerWordCount": len(audio_result["filler_words"]),
            "wordsPerMinute": audio_result["words_per_minute"],
            # Segment timing from full video (for time distribution display)
            "segmentStartTime": segment_start_time,
            "segmentEndTime": segment_end_time
        }

        # Save to database
        await db.update_question_response_analysis(
            response_id=response_id,
            video_analysis=video_analysis,
            score=llm_feedback["score"],
            strengths=llm_feedback["strengths"],
            improvements=llm_feedback["improvements"],
            feedback_status="completed"
        )

        return {
            "response_id": response_id,
            "score": llm_feedback["score"],
            "competency_scores": llm_feedback.get("competency_scores", {}),
            "strengths": llm_feedback["strengths"],
            "improvements": llm_feedback["improvements"],
            "feedback_text": llm_feedback["feedback_text"]
        }

    except Exception as e:
        await db.update_response_feedback_status(response_id, "failed")
        print(f"[Analyzer] Error analyzing response {response_id}: {e}")
        raise e

    finally:
        if video_path and video_path.exists():
            video_path.unlink()
            print(f"[Analyzer] Cleaned up temp file: {video_path}")


async def analyze_text_response(
    response: Dict[str, Any],
    scenario_content: str
) -> Dict[str, Any]:
    """Analyze a text response (no video analysis, just LLM feedback)."""
    response_id = response["id"]
    question_text = response["question_text"]
    text_content = response["text_content"]

    # Mark as processing
    await db.update_response_feedback_status(response_id, "processing")

    try:
        # Generate LLM feedback for text response
        llm_feedback = await generate_question_feedback(
            question_text=question_text,
            transcript=text_content,  # Use text content as "transcript"
            filler_word_count=0,  # N/A for text
            eye_contact_pct=100.0,  # N/A for text
            words_per_minute=0.0  # N/A for text
        )

        # Save to database (no video_analysis for text)
        await db.update_question_response_analysis(
            response_id=response_id,
            video_analysis=None,
            score=llm_feedback["score"],
            strengths=llm_feedback["strengths"],
            improvements=llm_feedback["improvements"],
            feedback_status="completed"
        )

        return {
            "response_id": response_id,
            "score": llm_feedback["score"],
            "competency_scores": llm_feedback.get("competency_scores", {}),
            "strengths": llm_feedback["strengths"],
            "improvements": llm_feedback["improvements"],
            "feedback_text": llm_feedback["feedback_text"]
        }

    except Exception as e:
        await db.update_response_feedback_status(response_id, "failed")
        raise e


async def analyze_full_video_attempt(
    attempt_id: int,
    full_video_key: str
) -> Dict[str, Any]:
    """
    Full-video analysis pipeline for a scenario attempt.

    This is the new flow where one video contains all 3 question responses.

    Pipeline:
    1. Download and transcribe full video
    2. Segment transcript by questions using LLM
    3. Chop video into segments
    4. Upload segments to R2
    5. Update question responses with segment URLs
    6. Analyze each segment using existing pipeline
    7. Generate scenario-level feedback

    Args:
        attempt_id: Scenario attempt ID
        full_video_key: R2 key of the full video

    Returns:
        Summary of analysis results
    """
    # Mark attempt as processing
    await db.update_attempt_feedback_status(attempt_id, "processing")

    try:
        # Get attempt info
        attempt = await db.get_scenario_attempt(attempt_id)
        if not attempt:
            raise ValueError(f"Attempt {attempt_id} not found")

        # Get all question responses
        responses = await db.get_question_responses(attempt_id)
        if not responses:
            raise ValueError(f"No responses found for attempt {attempt_id}")

        # Extract questions from responses
        questions = [r["question_text"] for r in responses]

        # Process full video: segment, chop, upload
        print(f"[FullVideoAnalyzer] Processing full video for attempt {attempt_id}")
        segmentation_result = await process_full_video(
            video_key=full_video_key,
            user_id=str(attempt["user_id"]),
            attempt_id=attempt_id,
            questions=questions
        )

        # Update question responses with segment URLs (or clear if unanswered)
        print(f"[FullVideoAnalyzer] Updating question responses with segment URLs")
        for segment_info in segmentation_result["segments"]:
            question_idx = segment_info["question_index"]
            response = responses[question_idx]

            # Always update the video_url - either to the segment or to None if unanswered
            await db.update_question_response_video_url(
                response_id=response["id"],
                video_url=segment_info["video_key"]  # None if not answered
            )

        # Analyze each segment using existing pipeline
        print(f"[FullVideoAnalyzer] Analyzing individual segments")
        question_feedbacks = []

        for segment_info in segmentation_result["segments"]:
            question_idx = segment_info["question_index"]
            response = responses[question_idx]

            if not segment_info["answered"]:
                # Mark as failed/skipped
                await db.update_response_feedback_status(response["id"], "failed")
                print(f"[FullVideoAnalyzer] Question {question_idx + 1} was not answered")
                continue

            # Analyze this segment (pass timing info for display on frontend)
            feedback = await analyze_video_response(
                response={
                    **response,
                    "video_url": segment_info["video_key"]
                },
                user_id=str(attempt["user_id"]),
                attempt_id=attempt_id,
                segment_start_time=segment_info.get("start_time"),
                segment_end_time=segment_info.get("end_time")
            )
            question_feedbacks.append(feedback)

        if question_feedbacks:
            # Calculate competency scores by averaging across questions
            overall_scores = average_competency_scores(question_feedbacks)
            avg_score = sum(overall_scores.values()) / len(overall_scores) if overall_scores else 5.0
            overall_quartile = calculate_quartile(avg_score)

            # Generate scenario-level feedback (key takeaways only)
            scenario_feedback = await generate_scenario_feedback(
                scenario_title=attempt["scenario_title"],
                question_feedbacks=question_feedbacks
            )

            # Save scenario feedback
            await db.update_scenario_attempt_feedback(
                attempt_id=attempt_id,
                overall_quartile=overall_quartile,
                overall_scores=overall_scores,
                overall_strengths=scenario_feedback["strengths"],
                overall_improvements=scenario_feedback["improvements"],
                overall_summary=scenario_feedback["summary"],
                feedback_status="completed"
            )

            return {
                "attempt_id": attempt_id,
                "status": "completed",
                "questions_analyzed": len(question_feedbacks),
                "overall_score": avg_score,
                "overall_quartile": overall_quartile
            }
        else:
            # No questions were answered
            await db.update_attempt_feedback_status(attempt_id, "failed")
            return {
                "attempt_id": attempt_id,
                "status": "failed",
                "message": "No questions were answered in the video"
            }

    except Exception as e:
        # Mark as failed
        await db.update_attempt_feedback_status(attempt_id, "failed")
        raise e
