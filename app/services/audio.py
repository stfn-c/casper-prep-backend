"""
Audio analysis module for video response analysis.
Extracts audio, transcribes with Whisper, and detects filler words.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Optional

# Lazy-loaded Whisper model
_whisper_model = None

# Filler words to detect
FILLER_WORDS = {
    "um", "uh", "like", "you know", "so", "basically",
    "actually", "literally", "right", "i mean"
}

# Prompt to encourage Whisper to preserve filler words
WHISPER_FILLER_PROMPT = "Um, uh, like, you know, so, basically, I mean, right, actually, literally, kind of, sort of, yeah, okay, hmm, ah, oh"


def get_whisper():
    """Lazy load Whisper model (medium for good accuracy vs speed tradeoff)"""
    global _whisper_model
    if _whisper_model is None:
        import whisper
        print("Loading Whisper medium model (first time takes a moment)...")
        _whisper_model = whisper.load_model("medium")
    return _whisper_model


def extract_audio(video_path: Path) -> Path:
    """
    Extract audio from video file to temporary WAV file using ffmpeg.

    Args:
        video_path: Path to video file

    Returns:
        Path to temporary WAV file

    Raises:
        RuntimeError: If ffmpeg extraction fails
    """
    # Create temporary file for audio
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio_path = Path(temp_file.name)
    temp_file.close()

    try:
        # Use ffmpeg to extract audio to WAV format
        # -i: input file
        # -vn: disable video
        # -acodec pcm_s16le: audio codec (16-bit PCM)
        # -ar 16000: sample rate 16kHz (good for speech)
        # -ac 1: mono audio
        result = subprocess.run([
            "ffmpeg",
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(audio_path),
            "-y"  # overwrite output file if exists
        ], capture_output=True, text=True, check=True)

        return audio_path

    except subprocess.CalledProcessError as e:
        # Clean up temp file on error
        if audio_path.exists():
            audio_path.unlink()
        raise RuntimeError(f"FFmpeg audio extraction failed: {e.stderr}")
    except FileNotFoundError:
        if audio_path.exists():
            audio_path.unlink()
        raise RuntimeError("FFmpeg not found. Please install ffmpeg: brew install ffmpeg")


def transcribe(audio_path: Path) -> dict:
    """
    Transcribe audio with Whisper and get word-level timestamps.

    Args:
        audio_path: Path to audio file

    Returns:
        Dictionary with:
            - transcript (str): Full transcript text
            - words (list): List of word dicts with {word, start, end}
            - segments (list): List of segment dicts with {start, end, text}
    """
    model = get_whisper()

    # Transcribe with word timestamps and filler word prompt
    result = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        initial_prompt=WHISPER_FILLER_PROMPT
    )

    # Extract full transcript
    transcript = result["text"].strip()

    # Extract segments and words
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


def detect_filler_words(words: list) -> list:
    """
    Detect filler words from word-level timestamps.

    Args:
        words: List of word dictionaries from Whisper
               Each dict has: {word, start, end}

    Returns:
        List of filler word occurrences with {word, start, end}
    """
    filler_words = []

    for word_info in words:
        # Clean word: lowercase and strip punctuation
        word = word_info.get("word", "").lower().strip(" .,!?")

        # Check if it's a filler word
        if word in FILLER_WORDS:
            filler_words.append({
                "word": word,
                "start": word_info.get("start", 0),
                "end": word_info.get("end", 0)
            })

    return filler_words


def analyze_audio(video_path: Path) -> dict:
    """
    Full audio analysis pipeline.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with:
            - transcript (str): Full transcript
            - words (list): All words with timestamps
            - filler_words (list): Detected filler words
            - words_per_minute (float): Speaking pace
            - total_speaking_time (float): Duration of speech

    Raises:
        RuntimeError: If audio extraction or transcription fails
    """
    audio_path = None

    try:
        # Extract audio from video
        audio_path = extract_audio(video_path)

        # Transcribe audio
        transcription = transcribe(audio_path)

        # Detect filler words
        filler_words = detect_filler_words(transcription["words"])

        # Calculate words per minute
        all_words = transcription["words"]
        if all_words and len(all_words) > 0:
            # Calculate total speaking time from first to last word
            total_time = all_words[-1].get("end", 0) - all_words[0].get("start", 0)
            total_speaking_time = total_time

            # WPM calculation
            words_per_minute = (len(all_words) / total_time * 60) if total_time > 0 else 0
        else:
            total_speaking_time = 0
            words_per_minute = 0

        return {
            "transcript": transcription["transcript"],
            "words": transcription["words"],
            "segments": transcription["segments"],
            "filler_words": filler_words,
            "words_per_minute": round(words_per_minute, 1),
            "total_speaking_time": round(total_speaking_time, 1)
        }

    finally:
        # Clean up temporary audio file
        if audio_path and audio_path.exists():
            audio_path.unlink()
