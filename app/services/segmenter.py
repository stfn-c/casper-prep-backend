"""
Video segmentation service for full-video analysis.
Handles transcription, LLM-based segmentation, and ffmpeg video chopping.
"""

import asyncio
import subprocess
import tempfile
import json
import httpx
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor

from app.services.audio import get_whisper, WHISPER_FILLER_PROMPT
from app.services.r2 import r2_service
from app.config import get_settings_sync

# Thread pool for CPU-bound tasks
_executor = ThreadPoolExecutor(max_workers=2)


class VideoSegment:
    """Represents a segmented portion of the video for one question."""

    def __init__(
        self,
        question_index: int,
        start_time: float,
        end_time: float,
        transcript: str
    ):
        self.question_index = question_index
        self.start_time = start_time
        self.end_time = end_time
        self.transcript = transcript


async def transcribe_full_video(video_path: Path) -> Dict[str, Any]:
    """
    Transcribe entire video using Whisper with word-level timestamps.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with:
            - transcript (str): Full transcript text
            - words (list): List of word dicts with {word, start, end}
            - segments (list): List of segment dicts with {start, end, text}
    """
    # Extract audio first
    audio_path = await asyncio.get_event_loop().run_in_executor(
        _executor,
        _extract_audio_for_transcription,
        video_path
    )

    try:
        # Transcribe with Whisper
        result = await asyncio.get_event_loop().run_in_executor(
            _executor,
            _transcribe_audio,
            audio_path
        )
        return result
    finally:
        # Clean up temp audio file
        if audio_path.exists():
            audio_path.unlink()


def _extract_audio_for_transcription(video_path: Path) -> Path:
    """Extract audio from video to temporary WAV file (sync version)."""
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio_path = Path(temp_file.name)
    temp_file.close()

    try:
        subprocess.run([
            "ffmpeg",
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(audio_path),
            "-y"
        ], capture_output=True, text=True, check=True)
        return audio_path
    except subprocess.CalledProcessError as e:
        if audio_path.exists():
            audio_path.unlink()
        raise RuntimeError(f"FFmpeg audio extraction failed: {e.stderr}")
    except FileNotFoundError:
        if audio_path.exists():
            audio_path.unlink()
        raise RuntimeError("FFmpeg not found. Please install ffmpeg: brew install ffmpeg")


def _transcribe_audio(audio_path: Path) -> Dict[str, Any]:
    """Transcribe audio with Whisper (sync version)."""
    model = get_whisper()

    result = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        initial_prompt=WHISPER_FILLER_PROMPT
    )

    transcript = result["text"].strip()

    segments = []
    all_words = []

    for seg in result.get("segments", []):
        segment_words = seg.get("words", [])

        segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip()
        })

        all_words.extend(segment_words)

    return {
        "transcript": transcript,
        "words": all_words,
        "segments": segments
    }


async def segment_transcript_by_questions(
    questions: List[str],
    full_transcript: str,
    words: List[Dict[str, Any]],
    api_key: Optional[str] = None
) -> List[Optional[VideoSegment]]:
    """
    Use LLM to identify which parts of the transcript answer which question.

    Args:
        questions: List of question texts
        full_transcript: Complete transcript
        words: Word-level timestamps from Whisper
        api_key: OpenRouter API key

    Returns:
        List of VideoSegment objects (one per question, or None if not answered)
    """
    if api_key is None:
        api_key = get_settings_sync().openrouter_api_key

    if not api_key:
        raise ValueError("OpenRouter API key is required")

    # Build the prompt
    prompt = f"""You are analyzing a student's video response to a CASPer exam scenario with 3 questions.

The student recorded ONE video answering all 3 questions sequentially. Your task is to identify which portion of the transcript corresponds to which question.

QUESTIONS:
{json.dumps([f"Q{i+1}: {q}" for i, q in enumerate(questions)], indent=2)}

FULL TRANSCRIPT:
{full_transcript}

WORD-LEVEL TIMESTAMPS (for precise segmentation):
{json.dumps(words[:50], indent=2)}
... (truncated, but you have access to all {len(words)} words)

INSTRUCTIONS:
1. Identify where each question's answer begins and ends based on semantic content
2. Use the word timestamps to determine precise start/end times
3. If a question was skipped or not answered, return null for that question
4. Questions are typically answered in order (Q1, Q2, Q3) but may be out of order
5. Be conservative - only include content that clearly answers the question

Return ONLY valid JSON (no markdown, no code blocks):
{{
  "segments": [
    {{
      "question_index": 0,
      "start_time": 0.0,
      "end_time": 95.5,
      "transcript": "My answer to question 1...",
      "confidence": "high"
    }},
    {{
      "question_index": 1,
      "start_time": 96.0,
      "end_time": 180.5,
      "transcript": "My answer to question 2...",
      "confidence": "high"
    }},
    null
  ],
  "reasoning": "Brief explanation of segmentation decisions"
}}

IMPORTANT: The "segments" array MUST have exactly {len(questions)} elements (one per question, in order). Use null for unanswered questions."""

    # Call OpenRouter API
    async with httpx.AsyncClient(timeout=90.0) as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "anthropic/claude-sonnet-4",
                "messages": [{"role": "user", "content": prompt}]
            }
        )

    if response.status_code != 200:
        raise Exception(f"OpenRouter API error: {response.status_code} - {response.text}")

    content = response.json()["choices"][0]["message"]["content"]

    # Parse JSON response
    try:
        import re
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            result = json.loads(json_match.group())
        else:
            raise ValueError("No JSON found in LLM response")
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse JSON from LLM response: {e}\nContent: {content}")

    # Convert to VideoSegment objects
    segments = []
    for i, seg_data in enumerate(result.get("segments", [])):
        if seg_data is None:
            segments.append(None)
        else:
            segments.append(VideoSegment(
                question_index=seg_data["question_index"],
                start_time=float(seg_data["start_time"]),
                end_time=float(seg_data["end_time"]),
                transcript=seg_data["transcript"]
            ))

    return segments


async def chop_video_segment(
    video_path: Path,
    start_time: float,
    end_time: float,
    output_path: Path
) -> Path:
    """
    Extract a segment from video using ffmpeg.

    Args:
        video_path: Path to source video
        start_time: Start time in seconds
        end_time: End time in seconds
        output_path: Where to save the chopped segment

    Returns:
        Path to the output file
    """
    duration = end_time - start_time

    # Use ffmpeg to extract the segment
    # -ss before -i: fast input seeking
    # Re-encode to ensure clean cut (no keyframe issues)
    # Using libvpx-vp9 for webm with good quality
    try:
        result = subprocess.run([
            "ffmpeg",
            "-ss", str(start_time),  # Input seeking (fast)
            "-i", str(video_path),
            "-t", str(duration),
            "-c:v", "libvpx-vp9",    # VP9 codec for webm
            "-crf", "30",             # Quality (lower = better, 30 is good for web)
            "-b:v", "0",              # Variable bitrate mode
            "-c:a", "libopus",        # Opus audio codec
            "-b:a", "128k",           # Audio bitrate
            str(output_path),
            "-y"
        ], capture_output=True, text=True, check=True)

        return output_path

    except subprocess.CalledProcessError as e:
        if output_path.exists():
            output_path.unlink()
        raise RuntimeError(f"FFmpeg video chopping failed: {e.stderr}")


async def upload_video_segment(
    segment_path: Path,
    user_id: str,
    attempt_id: int,
    question_index: int
) -> str:
    """
    Upload a video segment to R2.

    Args:
        segment_path: Path to the segment file
        user_id: User ID
        attempt_id: Scenario attempt ID
        question_index: Question index (0-based)

    Returns:
        The R2 key where the video was uploaded
    """
    # Generate the R2 key
    key = r2_service.generate_video_key(user_id, attempt_id, question_index + 1)

    # Upload to R2
    loop = asyncio.get_event_loop()
    with open(segment_path, 'rb') as f:
        await loop.run_in_executor(
            _executor,
            r2_service.upload_video,
            f,
            key,
            "video/webm"
        )

    return key


async def process_full_video(
    video_key: str,
    user_id: str,
    attempt_id: int,
    questions: List[str]
) -> Dict[str, Any]:
    """
    Full pipeline: download, transcribe, segment, chop, upload.

    Args:
        video_key: R2 key of the full video
        user_id: User ID
        attempt_id: Scenario attempt ID
        questions: List of question texts

    Returns:
        Dictionary with:
            - segments: List of segment info dicts
            - full_transcript: Complete transcript
    """
    video_path: Optional[Path] = None
    segment_paths: List[Path] = []

    try:
        # 1. Download full video from R2
        print(f"[Segmenter] Downloading full video: {video_key}")
        loop = asyncio.get_event_loop()
        video_path = await loop.run_in_executor(
            _executor,
            r2_service.download_video,
            video_key
        )
        print(f"[Segmenter] Downloaded to: {video_path}")

        # 2. Transcribe entire video
        print(f"[Segmenter] Transcribing full video...")
        transcription = await transcribe_full_video(video_path)
        print(f"[Segmenter] Transcription complete. Length: {len(transcription['transcript'])} chars")

        # 3. Segment transcript by questions using LLM
        print(f"[Segmenter] Segmenting transcript by {len(questions)} questions...")
        segments = await segment_transcript_by_questions(
            questions=questions,
            full_transcript=transcription["transcript"],
            words=transcription["words"]
        )
        print(f"[Segmenter] Segmentation complete. Found {sum(1 for s in segments if s is not None)} answered questions")

        # 4. Chop video into segments and upload
        segment_results = []
        for i, segment in enumerate(segments):
            if segment is None:
                # Question was not answered
                segment_results.append({
                    "question_index": i,
                    "answered": False,
                    "video_key": None,
                    "transcript": None,
                    "start_time": None,
                    "end_time": None
                })
                continue

            # Create temp file for segment
            temp_segment = tempfile.NamedTemporaryFile(
                delete=False,
                suffix='.webm',
                prefix=f'segment_q{i+1}_'
            )
            segment_path = Path(temp_segment.name)
            temp_segment.close()
            segment_paths.append(segment_path)

            # Chop video segment
            print(f"[Segmenter] Chopping Q{i+1}: {segment.start_time}s - {segment.end_time}s")
            await chop_video_segment(
                video_path=video_path,
                start_time=segment.start_time,
                end_time=segment.end_time,
                output_path=segment_path
            )

            # Upload segment to R2
            print(f"[Segmenter] Uploading Q{i+1} segment to R2...")
            uploaded_key = await upload_video_segment(
                segment_path=segment_path,
                user_id=user_id,
                attempt_id=attempt_id,
                question_index=i
            )
            print(f"[Segmenter] Uploaded to: {uploaded_key}")

            segment_results.append({
                "question_index": i,
                "answered": True,
                "video_key": uploaded_key,
                "transcript": segment.transcript,
                "start_time": segment.start_time,
                "end_time": segment.end_time
            })

        return {
            "segments": segment_results,
            "full_transcript": transcription["transcript"]
        }

    finally:
        # Clean up temp files
        if video_path and video_path.exists():
            video_path.unlink()
            print(f"[Segmenter] Cleaned up full video: {video_path}")

        for seg_path in segment_paths:
            if seg_path.exists():
                seg_path.unlink()
                print(f"[Segmenter] Cleaned up segment: {seg_path}")
