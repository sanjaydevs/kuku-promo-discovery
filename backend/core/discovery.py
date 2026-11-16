
import json
import re

from services.llm_service import generate_text


TIMESTAMP_PATTERN = re.compile(r"^\d{2}:\d{2}\.\d{2}$")


def timestamp_to_seconds(timestamp):
    minutes, seconds = timestamp.split(":")
    return int(minutes) * 60 + float(seconds)


def format_screenplay(screenplay):
    """
    Convert the parsed screenplay into a compact,
    readable representation for Gemini.
    """

    formatted = {
        "episode_number": screenplay["episode_number"],
        "logline": screenplay.get("logline"),
        "continuity_notes": screenplay.get("continuity_notes"),
        "scenes": []
    }

    for scene in screenplay["scenes"]:

        scene_data = {
            "scene_number": scene["scene_number"],
            "heading": scene["heading"],
            "start_timestamp": scene["start_timestamp"],
            "end_timestamp": scene["end_timestamp"],
            "start_seconds": scene["start_seconds"],
            "end_seconds": scene["end_seconds"],
            "events": []
        }

        for event in scene["events"]:
            scene_data["events"].append({
                "event_id": event["event_id"],
                "category": event["category"],
                "speaker": event["speaker"],
                "delivery": event["delivery"],
                "start_timestamp": event["start_timestamp"],
                "end_timestamp": event["end_timestamp"],
                "start_seconds": event["start_seconds"],
                "end_seconds": event["end_seconds"],
                "text": event["text"]
            })

        formatted["scenes"].append(scene_data)

    return formatted


def build_discovery_prompt(screenplay_data):

    screenplay_json = json.dumps(
        screenplay_data,
        ensure_ascii=False,
        indent=2
    )

    return f"""
You are an expert short-form promotional video editor
working on a serialized drama.

Your task is to identify the strongest continuous promotional
clips from the screenplay provided below.

SCREENPLAY:
{screenplay_json}

OBJECTIVE:

Find up to 10 distinct, high-quality promotional clips
from this episode.

Do not force 10 candidates. Return fewer if fewer genuinely
strong clips exist.

CLIP REQUIREMENTS:

1. Every clip must be continuous.
2. Every clip must be between 20 and 60 seconds inclusive.
3. The first 3 seconds should contain a compelling hook.
4. The ending should create suspense, curiosity, emotional
   impact or a strong cliffhanger.
5. The clip should make sense to a viewer who has not watched
   the episode.
6. Prefer clips with a clear emotional or narrative progression.
7. Avoid repetitive clips that use substantially the same
   dramatic moment.
8. Do not invent dialogue, actions, characters or events.
9. Consider dialogue, action, camera descriptions and sound
   events together when understanding the scene.
10. A clip can contain overlapping events. Do not treat
    overlapping dialogue and camera events as sequential.
11. Use the original screenplay timestamps exactly.
12. Choose clip start and end times from the timestamp
    boundaries present in the source events or scene boundaries.
13. The clip must stay within the episode's duration.

IMPORTANT:

- Do not simply select the most dramatic sentence.
- Think like an editor selecting a complete short-form promo.
- The clip must have a beginning, development and compelling end.
- Do not assume the viewer knows the characters or relationships.
- A strong emotional payoff or reveal can be a suitable ending.
- Do not manufacture a cliffhanger if the selected interval does not genuinely have one.

OUTPUT FORMAT:

Return ONLY valid JSON. Do not use Markdown code fences.

Use exactly this structure:

{{
  "episode_number": "{screenplay_data['episode_number']}",
  "candidates": [
    {{
      "candidate_id": "EP1_P01",
      "start_timestamp": "00:00.00",
      "end_timestamp": "00:30.00",
      "hook": "Description of the opening hook.",
      "cliffhanger": "Description of the ending.",
      "why_it_works": "Explanation of why this works as a promo.",
      "source_event_ids": [1, 2, 3]
    }}
  ]
}}

For source_event_ids, include the IDs of the relevant
screenplay events that fall within the selected interval.

Return an empty candidates array if no suitable promo exists.
"""


def parse_json_response(response_text):

    # Remove optional Markdown code fences
    cleaned = response_text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned
        )
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)

    except json.JSONDecodeError:
        # Try extracting the first complete JSON object
        start = cleaned.find("{")

        if start == -1:
            raise ValueError(
                "Gemini did not return a JSON object."
            )

        try:
            result, _ = json.JSONDecoder().raw_decode(
                cleaned[start:]
            )
            return result

        except json.JSONDecodeError as error:
            raise ValueError(
                "Could not parse Gemini's JSON response."
            ) from error


def validate_candidates(candidates, screenplay_data):
    """
    Apply basic deterministic checks to AI-generated candidates.

    Invalid candidates are rejected rather than silently corrected.
    """

    valid_candidates = []
    rejected_candidates = []

    all_events = []

    for scene in screenplay_data["scenes"]:
        all_events.extend(scene["events"])

    event_lookup = {
        event["event_id"]: event
        for event in all_events
    }

    valid_boundaries = set()

    for scene in screenplay_data["scenes"]:

        if scene["start_timestamp"]:
            valid_boundaries.add(scene["start_timestamp"])

        if scene["end_timestamp"]:
            valid_boundaries.add(scene["end_timestamp"])

        for event in scene["events"]:
            if event["start_timestamp"]:
                valid_boundaries.add(event["start_timestamp"])

            if event["end_timestamp"]:
                valid_boundaries.add(event["end_timestamp"])

    for candidate in candidates:

        reasons = []

        start = candidate.get("start_timestamp")
        end = candidate.get("end_timestamp")

        if not isinstance(start, str) or not TIMESTAMP_PATTERN.match(start):
            reasons.append("Invalid start timestamp.")

        if not isinstance(end, str) or not TIMESTAMP_PATTERN.match(end):
            reasons.append("Invalid end timestamp.")

        if reasons:
            rejected_candidates.append({
                "candidate": candidate,
                "reasons": reasons
            })
            continue

        start_seconds = timestamp_to_seconds(start)
        end_seconds = timestamp_to_seconds(end)

        duration = end_seconds - start_seconds

        if not 20 <= duration <= 60:
            reasons.append(
                f"Duration is {duration:.2f}s; expected 20–60s."
            )

        if start not in valid_boundaries:
            reasons.append(
                "Start timestamp does not match a source boundary."
            )

        if end not in valid_boundaries:
            reasons.append(
                "End timestamp does not match a source boundary."
            )

        if end_seconds <= start_seconds:
            reasons.append("End must be after start.")

        # Check that the candidate falls within the episode.
        scene_starts = [
            scene["start_seconds"]
            for scene in screenplay_data["scenes"]
            if scene["start_seconds"] is not None
        ]

        scene_ends = [
            scene["end_seconds"]
            for scene in screenplay_data["scenes"]
            if scene["end_seconds"] is not None
        ]

        if scene_starts and start_seconds < min(scene_starts):
            reasons.append("Clip starts before the episode.")

        if scene_ends and end_seconds > max(scene_ends):
            reasons.append("Clip ends after the episode.")

        # Recalculate source event IDs from the actual interval.
        actual_event_ids = [
            event["event_id"]
            for event in all_events
            if event["start_seconds"] is not None
            and event["end_seconds"] is not None
            and event["start_seconds"] < end_seconds
            and event["end_seconds"] > start_seconds
        ]

        supplied_ids = candidate.get("source_event_ids", [])

        if not isinstance(supplied_ids, list):
            reasons.append("source_event_ids must be a list.")
            supplied_ids = []

        elif any(
            not isinstance(event_id, int)
            or event_id not in event_lookup
            for event_id in supplied_ids
        ):
            reasons.append("Unknown source event ID.")

        if not actual_event_ids:
            reasons.append("Clip contains no timestamped source events.")

        for field in ("hook", "cliffhanger", "why_it_works"):
            if not isinstance(candidate.get(field), str) or not candidate[field].strip():
                reasons.append(f"Missing or empty field: {field}.")

        if reasons:
            rejected_candidates.append({
                "candidate": candidate,
                "reasons": reasons
            })

        else:
            candidate["duration_seconds"] = round(duration, 2)
            candidate["source_event_ids"] = actual_event_ids

            valid_candidates.append(candidate)

    return valid_candidates, rejected_candidates


def discover_promos(screenplay):
    """
    Discover continuous promotional clips from one episode.

    Returns valid candidates, rejected candidates and raw AI output.
    """

    screenplay_data = format_screenplay(screenplay)

    prompt = build_discovery_prompt(screenplay_data)

    raw_response = generate_text(
        prompt=prompt,
        system_instruction=(
            "You are a precise screenplay analysis and "
            "promotional editing assistant. Follow the requested "
            "JSON output format and never invent source details."
        ),
        temperature=0.2
    )

    parsed_response = parse_json_response(raw_response)

    if not isinstance(parsed_response, dict):
        raise ValueError("Gemini response must be a JSON object.")

    candidates = parsed_response.get("candidates")

    if not isinstance(candidates, list):
        raise ValueError(
            "Gemini response does not contain a valid candidates list."
        )

    if len(candidates) > 10:
        candidates = candidates[:10]

    valid, rejected = validate_candidates(
        candidates,
        screenplay_data
    )

    return {
        "episode_number": screenplay["episode_number"],
        "candidates": valid,
        "rejected_candidates": rejected,
        "raw_response": raw_response
    }
