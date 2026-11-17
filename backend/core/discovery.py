
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
    readable representation for the LLM.
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
        start = cleaned.find("{")

        if start == -1:
            raise ValueError(
                "The selected model did not return a JSON object."
            )

        try:
            result, _ = json.JSONDecoder().raw_decode(
                cleaned[start:]
            )
            return result

        except json.JSONDecodeError as error:
            raise ValueError(
                f"Could not parse the selected model's discovery response: {error}"
            ) from error


def validate_candidates(candidates, screenplay_data):
    """
    Apply deterministic checks to AI-generated candidates.
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

        if scene_starts and start_seconds < min(scene_starts):
            reasons.append("Clip starts before the episode.")

        if scene_ends and end_seconds > max(scene_ends):
            reasons.append("Clip ends after the episode.")

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

        elif any(
            not isinstance(event_id, int)
            or event_id not in event_lookup
            for event_id in supplied_ids
        ):
            reasons.append("Unknown source event ID.")

        if not actual_event_ids:
            reasons.append("Clip contains no timestamped source events.")

        for field in ("hook", "cliffhanger", "why_it_works"):
            if (
                not isinstance(candidate.get(field), str)
                or not candidate[field].strip()
            ):
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


def discover_promos(screenplay, provider="gemini", model=None):
    """
    Discover continuous promotional clips from one episode.
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
        temperature=0.2,
        provider=provider,
        model=model
    )

    parsed_response = parse_json_response(raw_response)

    if not isinstance(parsed_response, dict):
        raise ValueError(
            "Selected model response must be a JSON object."
        )

    candidates = parsed_response.get("candidates")

    if not isinstance(candidates, list):
        raise ValueError(
            "Selected model response does not contain a valid candidates list."
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




def build_montage_discovery_prompt(screenplay_data):
    """Build a prompt for discovering edited montages."""

    screenplay_json = json.dumps(
        screenplay_data,
        ensure_ascii=False,
        indent=2
    )

    return f"""
You are an expert short-form promotional video editor
working on a serialized drama.

Your task is to identify strong EDITED MONTAGE promos
from the screenplay provided below.

SCREENPLAY:
{screenplay_json}

OBJECTIVE:

Find up to 10 distinct promotional montages from this episode.

Each montage must combine multiple separate moments from
the SAME episode into one coherent promotional video.

Do not force 10 candidates. Return fewer if fewer genuinely
strong montages exist.

MONTAGE REQUIREMENTS:

1. Each montage must contain between 2 and 5 separate segments.
2. The TOTAL combined duration must be between 20 and 60 seconds.
3. Calculate the duration of every segment using:
   end timestamp minus start timestamp.
4. Add all segment durations together before returning a candidate.
5. Never return a montage whose combined duration is below 20 seconds.
6. Prefer a combined duration of approximately 25–45 seconds.
7. Individual segments should generally be 5–20 seconds long,
   where the source material allows it.
8. The first segment must establish a compelling opening hook.
9. The final segment must create suspense, curiosity,
   emotional impact or a strong cliffhanger.
10. The montage must make sense to a viewer unfamiliar with the show.
11. All segments must contribute to one coherent promotional idea.
12. The editing order may differ from the original screenplay chronology.
13. Avoid selecting overlapping source intervals.
14. Do not repeat the same dramatic moment within a montage.
15. Do not invent dialogue, actions, characters or events.
16. Consider dialogue, action, camera descriptions and sound events
    together when understanding each moment.
17. Use original screenplay timestamps exactly.
18. Choose segment start and end times only from timestamp
    boundaries present in the source events or scene boundaries.
19. Every segment must remain within the episode's duration.

IMPORTANT DURATION CHECK:

Before returning each candidate:
- Calculate the duration of every segment.
- Sum all segment durations.
- Verify that the total is at least 20 seconds and at most 60 seconds.
- If the total is below 20 seconds, extend suitable segments
  using valid source timestamp boundaries.
- Do not artificially extend a segment into unrelated content.
- If no suitable combination reaches 20 seconds, discard
  that montage candidate.

EDITORIAL GUIDANCE:

Think like an editor assembling a short promotional trailer.

For example:
- Segment 1: An intriguing statement or dramatic opening.
- Segment 2: A separate moment that develops the conflict.
- Segment 3: A reveal, confrontation or suspenseful moment.

This is only an illustration. Do not force this exact structure.

The segments must work together as one promo, not as a collection
of unrelated dramatic moments.

The viewer should understand the central situation without
needing to watch the full episode.

Do not manufacture connections between unrelated events.
Do not misrepresent the story by changing the meaning of events.

OUTPUT FORMAT:

Return ONLY valid JSON. Do not use Markdown code fences.

Use exactly this structure:

{{
  "episode_number": "{screenplay_data['episode_number']}",
  "candidates": [
    {{
      "candidate_id": "EP1_M01",
      "segments": [
        {{
          "segment_order": 1,
          "start_timestamp": "00:10.00",
          "end_timestamp": "00:18.00",
          "purpose": "Opening hook",
          "source_event_ids": [1, 2]
        }},
        {{
          "segment_order": 2,
          "start_timestamp": "00:40.00",
          "end_timestamp": "00:50.00",
          "purpose": "Develops the conflict",
          "source_event_ids": [5, 6]
        }},
        {{
          "segment_order": 3,
          "start_timestamp": "01:05.00",
          "end_timestamp": "01:15.00",
          "purpose": "Ending cliffhanger",
          "source_event_ids": [9, 10]
        }}
      ],
      "hook": "Description of the opening hook.",
      "cliffhanger": "Description of the ending.",
      "why_it_works": "Explanation of why these moments work together as a promo."
    }}
  ]
}}

MANDATORY OUTPUT CHECKS:

- Every segment must contain all five fields:
  segment_order, start_timestamp, end_timestamp,
  purpose and source_event_ids.
- Segment order must start at 1 and increase sequentially.
- Every segment must have a meaningful purpose.
- Total duration must be 20–60 seconds.
- source_event_ids must refer to actual screenplay event IDs.
- Do not create one continuous timestamp range covering all segments.
- Return an empty candidates array if no suitable montage exists.
"""



def validate_montage_candidates(candidates, screenplay_data):
    """
    Validate edited montage candidates without modifying
    the existing continuous-clip validator.
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

    for candidate in candidates:

        reasons = []
        segments = candidate.get("segments")

        if not isinstance(segments, list):
            rejected_candidates.append({
                "candidate": candidate,
                "reasons": ["Segments must be a list."]
            })
            continue

        if not 2 <= len(segments) <= 5:
            reasons.append("A montage must contain 2–5 segments.")

        validated_segments = []
        source_intervals = []
        total_duration = 0

        for index, segment in enumerate(segments):

            segment_reasons = []

            if not isinstance(segment, dict):
                reasons.append(
                    f"Segment {index + 1} must be an object."
                )
                continue

            start = segment.get("start_timestamp")
            end = segment.get("end_timestamp")

            if (
                not isinstance(start, str)
                or not TIMESTAMP_PATTERN.match(start)
            ):
                segment_reasons.append("Invalid start timestamp.")

            if (
                not isinstance(end, str)
                or not TIMESTAMP_PATTERN.match(end)
            ):
                segment_reasons.append("Invalid end timestamp.")

            if segment_reasons:
                reasons.extend(
                    [
                        f"Segment {index + 1}: {reason}"
                        for reason in segment_reasons
                    ]
                )
                continue

            start_seconds = timestamp_to_seconds(start)
            end_seconds = timestamp_to_seconds(end)
            duration = end_seconds - start_seconds

            if end_seconds <= start_seconds:
                segment_reasons.append(
                    "End must be after start."
                )

            if start not in valid_boundaries:
                segment_reasons.append(
                    "Start timestamp does not match a source boundary."
                )

            if end not in valid_boundaries:
                segment_reasons.append(
                    "End timestamp does not match a source boundary."
                )

            if scene_starts and start_seconds < min(scene_starts):
                segment_reasons.append(
                    "Segment starts before the episode."
                )

            if scene_ends and end_seconds > max(scene_ends):
                segment_reasons.append(
                    "Segment ends after the episode."
                )

            if (
                isinstance(segment.get("segment_order"), bool)
                or segment.get("segment_order") != index + 1
            ):
                segment_reasons.append(
                    "Segment order must follow the editing sequence."
                )

            if (
                not isinstance(segment.get("purpose"), str)
                or not segment["purpose"].strip()
            ):
                segment_reasons.append(
                    "Missing or empty segment purpose."
                )

            actual_event_ids = [
                event["event_id"]
                for event in all_events
                if event["start_seconds"] is not None
                and event["end_seconds"] is not None
                and event["start_seconds"] < end_seconds
                and event["end_seconds"] > start_seconds
            ]

            supplied_ids = segment.get("source_event_ids", [])

            if not isinstance(supplied_ids, list):
                segment_reasons.append(
                    "source_event_ids must be a list."
                )

            elif any(
                not isinstance(event_id, int)
                or isinstance(event_id, bool)
                or event_id not in event_lookup
                for event_id in supplied_ids
            ):
                segment_reasons.append(
                    "Unknown source event ID."
                )

            if not actual_event_ids:
                segment_reasons.append(
                    "Segment contains no timestamped source events."
                )

            if segment_reasons:
                reasons.extend(
                    [
                        f"Segment {index + 1}: {reason}"
                        for reason in segment_reasons
                    ]
                )
                continue

            total_duration += duration

            source_intervals.append(
                (start_seconds, end_seconds, index + 1)
            )

            validated_segments.append({
                **segment,
                "duration_seconds": round(duration, 2),
                "source_event_ids": actual_event_ids
            })

        # Check that segments do not overlap in source time.
        for i in range(len(source_intervals)):
            start_a, end_a, order_a = source_intervals[i]

            for j in range(i + 1, len(source_intervals)):
                start_b, end_b, order_b = source_intervals[j]

                if start_a < end_b and end_a > start_b:
                    reasons.append(
                        f"Segments {order_a} and {order_b} "
                        "overlap in source time."
                    )

        if not 20 <= total_duration <= 60:
            reasons.append(
                f"Total montage duration is {total_duration:.2f}s; "
                "expected 20–60s."
            )

        for field in ("hook", "cliffhanger", "why_it_works"):
            if (
                not isinstance(candidate.get(field), str)
                or not candidate[field].strip()
            ):
                reasons.append(
                    f"Missing or empty field: {field}."
                )

        if reasons:
            rejected_candidates.append({
                "candidate": candidate,
                "reasons": reasons
            })

        else:
            candidate["segments"] = validated_segments
            candidate["duration_seconds"] = round(
                total_duration,
                2
            )

            valid_candidates.append(candidate)

    return valid_candidates, rejected_candidates


def discover_montage_promos(
    screenplay,
    provider="gemini",
    model=None
):
    """
    Discover edited montage promotional clips from one episode.
    This function is independent of continuous clip discovery.
    """

    screenplay_data = format_screenplay(screenplay)

    prompt = build_montage_discovery_prompt(screenplay_data)

    raw_response = generate_text(
        prompt=prompt,
        system_instruction=(
            "You are a precise screenplay analysis and "
            "promotional editing assistant. Follow the requested "
            "JSON output format and never invent source details."
        ),
        temperature=0.2,
        provider=provider,
        model=model
    )

    parsed_response = parse_json_response(raw_response)

    if not isinstance(parsed_response, dict):
        raise ValueError(
            "Selected model response must be a JSON object."
        )

    candidates = parsed_response.get("candidates")

    if not isinstance(candidates, list):
        raise ValueError(
            "Selected model response does not contain a valid candidates list."
        )

    if len(candidates) > 10:
        candidates = candidates[:10]

    valid, rejected = validate_montage_candidates(
        candidates,
        screenplay_data
    )

    return {
        "episode_number": screenplay["episode_number"],
        "candidates": valid,
        "rejected_candidates": rejected,
        "raw_response": raw_response
    }
