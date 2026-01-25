import httpx
import json
import re
from typing import Optional

from app.config import get_settings_sync

BASE_RUBRIC = """SCORING (1-9 scale, be calibrated):
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
- Cookie-cutter responses"""

WRITING_STYLE = """WRITING STYLE:
- Write like you're talking to them, not grading a paper
- Be specific - quote parts of their response when giving feedback
- Give them something concrete to try next time
- It's okay to be encouraging when they do something well
- But don't sugarcoat - they need honest feedback to improve"""

COMPETENCY_SCORING = """COMPETENCY SCORING (score each 1-9):
- Empathy: Understanding others' feelings, acknowledging emotions, showing compassion
- Communication: Clarity, structure, appropriate tone, listening/responding appropriately  
- Ethical Reasoning: Identifying ethical issues, weighing values, justifying decisions
- Professionalism: Boundaries, accountability, respect, appropriate conduct"""

JSON_FORMAT = """{
  "competency_scores": {
    "empathy": 5,
    "communication": 5,
    "ethical_reasoning": 5,
    "professionalism": 5
  },
  "strengths": [
    "I liked how you... (specific thing they did well)"
  ],
  "improvements": [
    "Next time, try... (specific, actionable suggestion)"
  ],
  "red_flags": [],
  "summary": "2-3 sentences of direct, conversational feedback. Talk TO them, not about them.",
  "optimal_response": "A model answer that demonstrates Q4-level competencies. Write it as if YOU were answering this question perfectly - showing genuine empathy, clear ethical reasoning, multiple perspectives, and practical solutions. Keep it concise (2-3 paragraphs) but complete. This should be a realistic high-scoring response, not an impossibly perfect one."
}"""


async def _call_openrouter(prompt: str, api_key: str) -> dict:
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
            return json.loads(json_match.group())
        else:
            raise ValueError("No JSON found in response")
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse JSON from LLM response: {e}")


def _parse_feedback_response(feedback_json: dict) -> dict:
    competency_scores = feedback_json.get("competency_scores", {})
    score = sum(competency_scores.values()) / len(competency_scores) if competency_scores else 5.0
    
    return {
        "score": round(score, 1),
        "competency_scores": competency_scores,
        "strengths": feedback_json.get("strengths", []),
        "improvements": feedback_json.get("improvements", []),
        "red_flags": feedback_json.get("red_flags", []),
        "feedback_text": feedback_json.get("summary", ""),
        "optimal_response": feedback_json.get("optimal_response", "")
    }


async def generate_text_feedback(
    question_text: str,
    response_text: str,
    scenario_context: Optional[str] = None,
    api_key: Optional[str] = None
) -> dict:
    if api_key is None:
        api_key = get_settings_sync().openrouter_api_key

    if not api_key:
        raise ValueError("OpenRouter API key is required")

    context_section = f"\nSCENARIO CONTEXT:\n{scenario_context}\n" if scenario_context else ""

    prompt = f"""You are an experienced CASPer prep coach. You're reviewing a student's WRITTEN practice response. Write feedback like you're speaking directly to them - be warm but honest.

{BASE_RUBRIC}
{context_section}
QUESTION: {question_text}

THEIR WRITTEN RESPONSE:
{response_text}

{WRITING_STYLE}

{COMPETENCY_SCORING}

Respond with valid JSON only (no markdown, no code blocks):
{JSON_FORMAT}"""

    feedback_json = await _call_openrouter(prompt, api_key)
    return _parse_feedback_response(feedback_json)


async def generate_video_feedback(
    question_text: str,
    transcript: str,
    eye_contact_pct: float,
    words_per_minute: float,
    filler_word_count: int,
    scenario_context: Optional[str] = None,
    api_key: Optional[str] = None
) -> dict:
    if api_key is None:
        api_key = get_settings_sync().openrouter_api_key

    if not api_key:
        raise ValueError("OpenRouter API key is required")

    eye_contact_note = "(good)" if eye_contact_pct >= 60 else "(could use work - try to look at the camera more)"
    
    if 120 <= words_per_minute <= 150:
        pace_note = "(nice steady pace)"
    elif words_per_minute > 150:
        pace_note = "(a bit fast - take a breath)"
    else:
        pace_note = "(could be more confident - don't be afraid to speak up)"
    
    filler_note = "(minimal - nice!)" if filler_word_count <= 3 else "(noticeable - practice pausing instead of 'um')"

    context_section = f"\nSCENARIO CONTEXT (what they watched in the video):\n{scenario_context}\n" if scenario_context else ""

    prompt = f"""You are an experienced CASPer prep coach. You're reviewing a student's VIDEO practice response. Write feedback like you're speaking directly to them - be warm but honest.

{BASE_RUBRIC}
{context_section}
QUESTION: {question_text}

TRANSCRIPT OF THEIR VIDEO RESPONSE:
{transcript}

VIDEO DELIVERY METRICS:
- Eye contact: {eye_contact_pct:.0f}% {eye_contact_note}
- Speaking pace: {words_per_minute:.0f} WPM {pace_note}
- Filler words: {filler_word_count} {filler_note}

{WRITING_STYLE}

For video responses, also comment on their delivery (eye contact, pace, filler words) where relevant.

{COMPETENCY_SCORING}

Respond with valid JSON only (no markdown, no code blocks):
{JSON_FORMAT}"""

    feedback_json = await _call_openrouter(prompt, api_key)
    return _parse_feedback_response(feedback_json)


async def generate_scenario_feedback(
    scenario_title: str,
    question_feedbacks: list[dict],
    api_key: Optional[str] = None
) -> dict:
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

    feedback_json = await _call_openrouter(prompt, api_key)
    
    return {
        "strengths": feedback_json.get("strengths", []),
        "improvements": feedback_json.get("improvements", []),
        "summary": feedback_json.get("summary", "")
    }


async def generate_mock_exam_feedback(
    exam_name: str,
    scenario_feedbacks: list[dict],
    api_key: Optional[str] = None
) -> dict:
    if api_key is None:
        api_key = get_settings_sync().openrouter_api_key

    if not api_key:
        raise ValueError("OpenRouter API key is required")

    prompt = f"""You're wrapping up a full mock CASPer exam with a student. They just completed all the scenarios, and you're giving them their overall performance review.

EXAM: {exam_name}

THEIR PERFORMANCE ON EACH SCENARIO:
{json.dumps(scenario_feedbacks, indent=2)}

Your job is to:
1. Identify the 2-3 biggest patterns across ALL scenarios (both strengths and weaknesses)
2. Calculate what quartile they'd likely fall into on the real exam
3. Give them a clear, actionable improvement plan

WRITING STYLE:
- This is the big picture feedback - connect the dots across scenarios
- Be encouraging about what they're doing well
- Be direct about what needs work - they need to know before the real test
- End with specific advice for their next study session

Respond with valid JSON only (no markdown, no code blocks):
{{
  "strengths": [
    "Looking across all your scenarios, a real strength is... (pattern you noticed)",
    "Another thing you consistently did well: ..."
  ],
  "improvements": [
    "The pattern I want you to focus on: ... (your main weakness with specific examples)",
    "Also watch out for: ... (secondary area if relevant)"
  ],
  "summary": "3-4 sentences. Start with overall impression, then the one thing they MUST work on before the real exam, then encouragement. Example: 'You're showing solid [X] instincts across the board. Your main gap is [Y] - in several scenarios you [specific pattern]. For your next practice session, I want you to specifically focus on [concrete action]. Keep pushing!'"
}}"""

    feedback_json = await _call_openrouter(prompt, api_key)
    
    return {
        "strengths": feedback_json.get("strengths", []),
        "improvements": feedback_json.get("improvements", []),
        "summary": feedback_json.get("summary", "")
    }
