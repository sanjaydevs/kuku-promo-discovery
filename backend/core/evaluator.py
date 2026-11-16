
import json

from services.llm_service import generate_text


WEIGHTS = {
    "opening_hook": 0.25,
    "cliffhanger": 0.30,
    "standalone_comprehension": 0.25,
    "narrative_coherence": 0.10,
    "promotional_appeal": 0.10,
}


def build_evaluation_prompt(screenplay_data, candidates):
    """
    Build a prompt to evaluate all promo candidates from one episode.
    """

    episode_context = {
        "episode_number": screenplay_data["episode_number"],
        "logline": screenplay_data.get("logline", ""),
        "continuity_notes": screenplay_data.get("continuity_notes", ""),
    }

    prompt_data = {
        "episode": episode_context,
        "candidates": candidates,
    }

    return f"""
You are an expert short-form video promo editor evaluating promotional clips
for Instagram Reels, TikTok, and YouTube Shorts.

Evaluate every candidate independently based on the screenplay evidence.
Then compare candidates to identify substantial content redundancy.

EPISODE AND CANDIDATE DATA:
{json.dumps(prompt_data, ensure_ascii=False, indent=2)}

SCORING CRITERIA:

1. opening_hook (25%)
Score how effectively the first 3 seconds capture attention.
Consider visual intrigue, conflict, surprise, and immediate curiosity.

2. cliffhanger (30%)
Score how effectively the ending creates suspense, emotional impact,
curiosity, or a desire to watch the full episode.

3. standalone_comprehension (25%)
Score whether a viewer unfamiliar with the show can understand the
situation, characters' immediate motivations, and conflict.

4. narrative_coherence (10%)
Score whether the selected moments form a coherent and understandable
sequence without confusing jumps or missing context.

5. promotional_appeal (10%)
Score the overall potential to make viewers interested in watching
the full episode.

SCORING RULES:
- Give each criterion an integer score from 1 to 10.
- Evaluate only what is supported by the supplied screenplay.
- Do not reward a candidate simply because it contains more dialogue.
- Do not invent visual or narrative details.
- A high score must be justified with specific evidence.
- Consider whether the opening hook actually occurs within the first
  3 seconds of the clip.
- Do not calculate weighted scores. Python will calculate them.

REDUNDANCY:
Compare candidates with one another.
Identify candidates that substantially repeat the same dramatic moment
or reveal. Do not penalize candidates merely because they share a few
seconds of footage.
For each candidate, provide a short redundancy note.

Return ONLY valid JSON in this exact structure:

{{
  "evaluations": [
    {{
      "candidate_id": "P01",
      "scores": {{
        "opening_hook": 8,
        "cliffhanger": 9,
        "standalone_comprehension": 8,
        "narrative_coherence": 9,
        "promotional_appeal": 8
      }},
      "explanations": {{
        "opening_hook": "Reason for the score.",
        "cliffhanger": "Reason for the score.",
        "standalone_comprehension": "Reason for the score.",
        "narrative_coherence": "Reason for the score.",
        "promotional_appeal": "Reason for the score."
      }},
      "redundancy_note": "Comparison with other candidates."
    }}
  ]
}}

Include exactly one evaluation for every supplied candidate.
"""


def parse_evaluation_response(response_text):
    """
    Parse Gemini's JSON response.
    Handles responses wrapped in Markdown code fences.
    """

    cleaned = response_text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned = "\n".join(lines).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Gemini returned invalid evaluation JSON: {error}"
        ) from error


def validate_evaluations(evaluations, candidates):
    """
    Check that every candidate has a complete and valid evaluation.
    """

    candidate_ids = {c["candidate_id"] for c in candidates}

    if not isinstance(evaluations, list):
        raise ValueError("Evaluations must be a list.")

    evaluation_ids = set()

    for evaluation in evaluations:
        candidate_id = evaluation.get("candidate_id")

        if candidate_id not in candidate_ids:
            raise ValueError(
                f"Unknown candidate ID in evaluation: {candidate_id}"
            )

        if candidate_id in evaluation_ids:
            raise ValueError(
                f"Duplicate evaluation for candidate: {candidate_id}"
            )

        evaluation_ids.add(candidate_id)

        scores = evaluation.get("scores", {})
        explanations = evaluation.get("explanations", {})

        for criterion in WEIGHTS:
            score = scores.get(criterion)

            if (
                not isinstance(score, int)
                or isinstance(score, bool)
                or not 1 <= score <= 10
            ):
                raise ValueError(
                    f"Invalid score for {candidate_id}: {criterion}"
                )

            if not isinstance(explanations.get(criterion), str):
                raise ValueError(
                    f"Missing explanation for {candidate_id}: {criterion}"
                )

        if not isinstance(evaluation.get("redundancy_note"), str):
            raise ValueError(
                f"Missing redundancy note for {candidate_id}"
            )

    missing_ids = candidate_ids - evaluation_ids

    if missing_ids:
        raise ValueError(
            f"Missing evaluations for candidates: {sorted(missing_ids)}"
        )

    return evaluations


def calculate_weighted_score(scores):
    """
    Calculate the final score out of 10 using Python.
    """

    return round(
        sum(scores[criterion] * weight
            for criterion, weight in WEIGHTS.items()),
        2
    )


def evaluate_promos(screenplay_data, candidates):
    """
    Evaluate all discovered promo candidates for one episode.
    """

    if not candidates:
        return {
            "episode_number": screenplay_data["episode_number"],
            "evaluations": [],
            "raw_response": None,
        }

    prompt = build_evaluation_prompt(screenplay_data, candidates)

    response_text = generate_text(
        prompt,
        temperature=0.1
    )

    parsed_response = parse_evaluation_response(response_text)

    evaluations = validate_evaluations(
        parsed_response.get("evaluations"),
        candidates
    )

    candidate_lookup = {
        candidate["candidate_id"]: candidate
        for candidate in candidates
    }

    results = []

    for evaluation in evaluations:
        candidate = candidate_lookup[evaluation["candidate_id"]]

        results.append({
            **candidate,
            "scores": evaluation["scores"],
            "explanations": evaluation["explanations"],
            "redundancy_note": evaluation["redundancy_note"],
            "weighted_score": calculate_weighted_score(
                evaluation["scores"]
            ),
        })

    return {
        "episode_number": screenplay_data["episode_number"],
        "evaluations": results,
        "raw_response": response_text,
    }
