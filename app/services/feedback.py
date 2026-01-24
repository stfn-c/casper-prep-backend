"""
LLM feedback generation service for CasperPrep backend.
Uses Claude via OpenRouter to generate structured feedback.
"""

import httpx
import json
import re
from typing import Optional

from app.config import get_settings_sync


async def generate_question_feedback(
    question_text: str,
    transcript: str,
    filler_word_count: int,
    eye_contact_pct: float,
    words_per_minute: float,
    api_key: Optional[str] = None
) -> dict:
    """
    Generate feedback for a single question response.
    Uses Claude via OpenRouter.

    Args:
        question_text: The question that was answered
        transcript: The user's spoken response
        filler_word_count: Number of filler words detected
        eye_contact_pct: Percentage of time maintaining eye contact
        words_per_minute: Speaking pace (ideal: 120-150 WPM)
        api_key: OpenRouter API key (defaults to config)

    Returns:
        {
            "score": float,  # 1-10
            "strengths": [str, ...],
            "improvements": [str, ...],
            "feedback_text": str
        }
    """
    if api_key is None:
        api_key = get_settings_sync().openrouter_api_key

    if not api_key:
        raise ValueError("OpenRouter API key is required")

    # Build prompt with detailed CASPer rubric
    prompt = f"""You are an experienced CASPer prep coach who has helped hundreds of students get into medical school. You're reviewing a student's practice response. Write feedback like you're speaking directly to them - be warm but honest, like a supportive mentor who genuinely wants them to succeed.

SCORING (1-9 scale, be calibrated):
- 1-2: Missed the mark - harmful, dismissive, or completely off-base
- 3-4: Needs work - shallow reasoning, missed key perspectives
- 5: Decent start - gets the basics but feels generic
- 6-7: Solid response - good insight, considers multiple angles
- 8-9: Excellent - would genuinely impress an admissions committee

WHAT TO LOOK FOR:
- Do they show genuine empathy (not just say "I understand")?
- Do they consider multiple perspectives without being wishy-washy?
- Is their reasoning grounded in ethics, not just rules?
- Do they offer practical solutions, not just platitudes?
- Does it feel authentic, or like they're performing?

RED FLAGS:
- Dismissing feelings or jumping to judgment
- One-sided thinking
- Avoiding the hard parts of the question
- Cookie-cutter responses

QUESTION: {question_text}

THEIR RESPONSE:
{transcript}

DELIVERY NOTES:
- Eye contact: {eye_contact_pct}% {"(good)" if eye_contact_pct >= 60 else "(could use work - try to look at the camera more)"}
- Pace: {words_per_minute} WPM {"(nice steady pace)" if 120 <= words_per_minute <= 150 else "(a bit fast - take a breath)" if words_per_minute > 150 else "(could be more confident - don't be afraid to speak up)"}
- Filler words: {filler_word_count} {"(minimal - nice!)" if filler_word_count <= 3 else "(noticeable - practice pausing instead of 'um')"}

WRITING STYLE:
- Write like you're talking to them, not grading a paper
- Be specific - quote parts of their response when giving feedback
- Give them something concrete to try next time
- It's okay to be encouraging when they do something well
- But don't sugarcoat - they need honest feedback to improve

COMPETENCY SCORING (score each 1-9):
- Empathy: Understanding others' feelings, acknowledging emotions, showing compassion
- Communication: Clarity, structure, appropriate tone, listening/responding appropriately  
- Ethical Reasoning: Identifying ethical issues, weighing values, justifying decisions
- Professionalism: Boundaries, accountability, respect, appropriate conduct

Respond with valid JSON only (no markdown, no code blocks):
{{
  "competency_scores": {{
    "empathy": 5,
    "communication": 5,
    "ethical_reasoning": 5,
    "professionalism": 5
  }},
  "strengths": [
    "I liked how you... (specific thing they did well)"
  ],
  "improvements": [
    "Next time, try... (specific, actionable suggestion)"
  ],
  "red_flags": [],
  "summary": "2-3 sentences of direct, conversational feedback. Talk TO them, not about them. Example: 'You showed good instincts here, especially when you... But I'd push you to dig deeper on...'"
}}"""

    # Call OpenRouter API
    async with httpx.AsyncClient(timeout=60.0) as client:
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

    # Extract content
    content = response.json()["choices"][0]["message"]["content"]

    # Parse JSON from response (handle potential markdown wrapping)
    try:
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            feedback_json = json.loads(json_match.group())
        else:
            raise ValueError("No JSON found in response")
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse JSON from LLM response: {e}")

    competency_scores = feedback_json.get("competency_scores", {})
    score = sum(competency_scores.values()) / len(competency_scores) if competency_scores else 5.0
    
    return {
        "score": round(score, 1),
        "competency_scores": competency_scores,
        "strengths": feedback_json.get("strengths", []),
        "improvements": feedback_json.get("improvements", []),
        "red_flags": feedback_json.get("red_flags", []),
        "feedback_text": feedback_json.get("summary", "")
    }


async def generate_scenario_feedback(
    scenario_title: str,
    question_feedbacks: list[dict],
    api_key: Optional[str] = None
) -> dict:
    """
    Generate key takeaways for a scenario based on question feedbacks.
    Competency scores and quartiles are calculated separately by the analyzer.

    Returns:
        {
            "strengths": [str, ...],
            "improvements": [str, ...],
            "summary": str
        }
    """
    if api_key is None:
        api_key = get_settings_sync().openrouter_api_key

    if not api_key:
        raise ValueError("OpenRouter API key is required")

    prompt = f"""You're wrapping up a review of a student's practice scenario. Look at how they did across all the questions and give them your overall take - what patterns do you see, and what's the one thing they should focus on?

SCENARIO: {scenario_title}

HOW THEY DID ON EACH QUESTION:
{json.dumps(question_feedbacks, indent=2)}

WRITING STYLE:
- Talk to them directly, like you're wrapping up a tutoring session
- Point out patterns you noticed (good and bad)
- Give them ONE main thing to focus on - don't overwhelm
- Be real but encouraging - they're practicing to get better

Respond with valid JSON only (no markdown, no code blocks):
{{
  "strengths": [
    "Across your responses, I noticed you consistently... (pattern)"
  ],
  "improvements": [
    "The biggest thing to work on: ... (one focused area with specific advice)"
  ],
  "summary": "2-3 sentences wrapping up. Example: 'Overall, you're showing good instincts on X. Your main opportunity is Y - I'd spend your next practice session really focusing on that. Keep at it!'"
}}"""

    async with httpx.AsyncClient(timeout=60.0) as client:
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

    try:
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            feedback_json = json.loads(json_match.group())
        else:
            raise ValueError("No JSON found in response")
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse JSON from LLM response: {e}")

    return {
        "strengths": feedback_json.get("strengths", []),
        "improvements": feedback_json.get("improvements", []),
        "summary": feedback_json.get("summary", "")
    }
