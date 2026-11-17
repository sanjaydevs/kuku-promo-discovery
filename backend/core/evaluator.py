
import json
import re

from services.llm_service import generate_text


WEIGHTS = {
    "opening_hook": 0.25,
    "cliffhanger": 0.30,
    "standalone_comprehension": 0.25,
    "narrative_coherence": 0.10,
    "promotional_appeal": 0.10,
}


def build_evaluation_prompt(screenplay_data, candidates):

    episode_context = {
        "episode_number": screenplay_data["episode_number"],
        "logline": screenplay_data.get("logline", ""),
        "continuity_notes": screenplay_data.get("continuity_notes", ""),
    }

    prompt_data = {
        "episode": episode_context,
        "candidates": candidates,
    }

    supplied_ids = [
        candidate["candidate_id"]
        for candidate in candidates
    ]

    return f"""
You are an expert short-form video promo editor evaluating promotional clips
for Instagram Reels, TikTok, and YouTube Shorts.

Evaluate every candidate independently based on screenplay evidence.
Then compare candidates to identify substantial content redundancy.

EPISODE AND CANDIDATE DATA:
{json.dumps(prompt_data, ensure_ascii=False, indent=2)}

IMPORTANT CANDIDATE ID RULES:
- The only valid candidate IDs are: {json.dumps(supplied_ids)}
- Copy every candidate ID exactly as supplied.
- Never create, modify, shorten, or invent candidate IDs.
- Return exactly one evaluation for every supplied candidate.
- Do not omit any candidate.

SCORING CRITERIA:

1. opening_hook (25%)
Score how effectively the first 3 seconds capture attention.

2. cliffhanger (30%)
Score how effectively the ending creates suspense, emotional impact,
curiosity, or a desire to watch the full episode.

3. standalone_comprehension (25%)
Score whether a viewer unfamiliar with the show can understand the
situation, characters' immediate motivations, and conflict.

4. narrative_coherence (10%)
Score whether the selected moments form a coherent sequence.

5. promotional_appeal (10%)
Score the overall potential to make viewers interested in watching
the full episode.

SCORING RULES:
- Give each criterion an integer score from 1 to 10.
- Evaluate only what is supported by the supplied screenplay.
- Do not invent visual or narrative details.
- A high score must be justified with specific evidence.
- Do not calculate weighted scores. Python will calculate them.

REDUNDANCY:
Compare candidates with one another.
Identify candidates that substantially repeat the same dramatic moment
or reveal.
Do not penalize candidates merely because they share a few seconds.
Provide a short redundancy note for each candidate.

OUTPUT RULES:
- Return ONLY valid JSON.
- Use double quotes for every JSON key and string.
- Do not include Markdown code fences.
- Do not include comments or trailing commas.
- Include exactly one evaluation for every supplied candidate.

Return this exact structure:

{{
  "evaluations": [
    {{
      "candidate_id": "EXACT_SUPPLIED_ID",
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
"""


def parse_evaluation_response(response_text):

    cleaned = response_text.strip()

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned
    )

    try:
        return json.loads(cleaned)

    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(
            "The selected model did not return a valid JSON object."
        )

    json_text = cleaned[start:end + 1]

    try:
        return json.loads(json_text)

    except json.JSONDecodeError as error:
        raise ValueError(
            f"Selected model returned invalid evaluation JSON: {error}"
        ) from error


def validate_evaluations(evaluations, candidates):

    candidate_ids = {
        candidate["candidate_id"]
        for candidate in candidates
    }

    if not isinstance(evaluations, list):
        raise ValueError("Evaluations must be a list.")

    evaluation_ids = set()

    for evaluation in evaluations:

        if not isinstance(evaluation, dict):
            raise ValueError("An evaluation entry is not a JSON object.")

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

    return round(
        sum(
            scores[criterion] * weight
            for criterion, weight in WEIGHTS.items()
        ),
        2
    )


def evaluate_promos(
    screenplay_data,
    candidates,
    provider="gemini",
    model=None
):

    if not candidates:
        return {
            "episode_number": screenplay_data["episode_number"],
            "evaluations": [],
            "raw_response": None,
        }

    prompt = build_evaluation_prompt(
        screenplay_data,
        candidates
    )

    response_text = generate_text(
        prompt,
        temperature=0.1,
        provider=provider,
        model=model
    )

    parsed_response = parse_evaluation_response(response_text)

    if not isinstance(parsed_response, dict):
        raise ValueError(
            "Selected model returned an invalid evaluation structure."
        )

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

        candidate = candidate_lookup[
            evaluation["candidate_id"]
        ]

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
