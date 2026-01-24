# CasperPrep Services

This directory contains the core analysis services for the CasperPrep backend.

## Modules

### `video.py` - Eye Contact Detection

Analyzes video files to detect eye contact using OpenCV Haar cascades.

**Usage:**
```python
from app.services.video import analyze_eye_contact
from pathlib import Path

result = analyze_eye_contact(
    video_path=Path("path/to/video.mp4"),
    sample_rate=5  # analyze every 5th frame
)

# Returns:
# {
#     "eye_contact_percentage": 78.5,
#     "issues": [
#         {"start": 12.3, "end": 15.8},  # lost eye contact for 3.5s
#         {"start": 45.1, "end": 48.2}   # lost eye contact for 3.1s
#     ]
# }
```

**How it works:**
- Uses `haarcascade_frontalface_default.xml` for face detection
- Uses `haarcascade_eye.xml` for eye detection
- Calculates gaze ratio (pupil position / eye width)
- Eye contact = gaze ratio between 0.3-0.7 (centered)
- Tracks periods of lost eye contact > 2 seconds

---

### `feedback.py` - LLM Feedback Generation

Generates AI-powered feedback using Claude Sonnet 4 via OpenRouter.

**Question-level feedback:**
```python
from app.services.feedback import generate_question_feedback

feedback = await generate_question_feedback(
    question_text="What would you do if you saw a colleague cheating?",
    transcript="Um, I would, like, talk to them first...",
    filler_word_count=12,
    eye_contact_pct=75.5,
    words_per_minute=145.2,
    api_key="your-openrouter-api-key"  # or set OPENROUTER_API_KEY env var
)

# Returns:
# {
#     "score": 7.5,
#     "strengths": ["Good empathy", "Clear structure"],
#     "improvements": ["Reduce filler words", "Slow down pace"],
#     "feedback_text": "Overall, strong response with good empathy..."
# }
```

**Scenario-level feedback:**
```python
from app.services.feedback import generate_scenario_feedback

scenario_feedback = await generate_scenario_feedback(
    scenario_title="Medical Ethics Scenario",
    question_feedbacks=[
        {"score": 7.5, "strengths": [...], "improvements": [...]},
        {"score": 8.0, "strengths": [...], "improvements": [...]},
        {"score": 6.5, "strengths": [...], "improvements": [...]}
    ],
    api_key="your-openrouter-api-key"
)

# Returns:
# {
#     "overall_score": 7.3,
#     "overall_quartile": 3,  # 1-4 (bottom 25% to top 25%)
#     "strengths": ["Consistent empathy across responses"],
#     "improvements": ["Work on pacing consistency"],
#     "summary": "Strong performance overall with good ethical reasoning..."
# }
```

**API Key:**
- Set `OPENROUTER_API_KEY` environment variable, or
- Pass `api_key` parameter explicitly

---

## Dependencies

Added to `requirements.txt`:
- `opencv-python` - Computer vision for eye tracking
- `httpx` - Async HTTP client for OpenRouter API
- `numpy` - Array operations for video processing

## Implementation Notes

### Video Analysis
- Based on `VideoAnalyzer` class from `/Users/stefan/Desktop/services/video-analysis-v3/analyzer.py`
- Uses same Haar cascade approach (lines 217-289 in reference)
- Gaze ratio calculation matches reference implementation
- Sample rate parameter allows performance tuning (higher = faster but less accurate)

### LLM Feedback
- Based on `_generate_ai_feedback` method from reference (lines 781-872)
- Uses same prompt structure for JSON responses
- Model: `anthropic/claude-sonnet-4` via OpenRouter
- Handles markdown-wrapped JSON responses
- Async implementation for non-blocking API calls

## Testing

Verify imports work:
```bash
python3 -c "from app.services.video import analyze_eye_contact; print('✓ video.py')"
python3 -c "from app.services.feedback import generate_question_feedback; print('✓ feedback.py')"
```

## Future Enhancements

- [ ] Add emotion detection (DeepFace) from reference implementation
- [ ] Add speech analysis (Whisper) integration
- [ ] Cache LLM responses to reduce API costs
- [ ] Add batch processing for multiple videos
- [ ] GPU acceleration for video processing
