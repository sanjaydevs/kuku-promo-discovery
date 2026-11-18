
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
    """
    Build the discovery prompt for edited montage candidates.

    Montages combine non-contiguous moments from the same episode
    in a deliberate editing sequence.
    """

    screenplay_context = format_screenplay(screenplay_data)

    prompt_data = {
        "episode_number": screenplay_data["episode_number"],
        "logline": screenplay_data.get("logline", ""),
        "continuity_notes": screenplay_data.get("continuity_notes", ""),
        "screenplay": screenplay_context,
    }

    episode_match = re.search(
        r"\d+",
        str(screenplay_data["episode_number"])
    )

    episode_id = (
        episode_match.group()
        if episode_match
        else re.sub(
            r"[^A-Za-z0-9]+",
            "",
            str(screenplay_data["episode_number"])
        )
    )

    return f"""
You are an expert promotional video editor and trailer storyteller.

Your task is to identify compelling EDITED MONTAGE promotional
videos from the supplied episode screenplay.

An edited montage is a sequence of multiple non-contiguous moments
from the SAME episode, assembled in a deliberate editing order.

Unlike a continuous clip, the selected moments do not need to occur
next to each other in the original screenplay.

Your goal is to create engaging promotional videos that make
viewers curious about the episode and encourage them to watch it.

============================================================
1. MONTAGE REQUIREMENTS
============================================================

Generate up to 10 distinct montage candidates.

Each montage must satisfy these requirements:

- Total edited duration MUST be between 20 and 60 seconds.
- Use a minimum of 3 segments.
- Use a maximum of 8 segments.
- Prefer 4 to 7 segments when the source material supports them.
- Prefer a total duration of 35 to 45 seconds when the story
  provides enough meaningful material.
- Do not force additional segments or unnecessary footage merely
  to reach the preferred duration.
- Each segment must be a continuous interval from the original
  screenplay.
- Segments must come from the same episode.
- Segments must not overlap in their original source timestamps.
- Segments may be rearranged into a different editing order.
- Segment order must reflect the intended final video sequence.

There is NO fixed duration requirement for an individual segment.

Choose each segment's duration based on the actual scene,
dialogue, action, emotional reaction and narrative context.

Do not cut a moment prematurely simply to increase the number
of segments.

============================================================
2. BUILD A COMPLETE PROMOTIONAL SEQUENCE
============================================================

Do not construct montages by selecting only three short moments
that represent the beginning, middle and end of a story.

Explore the screenplay for additional meaningful moments.

A strong montage should ideally contain:

- An attention-grabbing opening.
- Sufficient context to establish the situation.
- Escalation of conflict, tension or emotional stakes.
- Meaningful developments, reveals or reactions.
- A powerful final moment that creates curiosity.

These are narrative functions, not mandatory separate segments.
One segment may serve multiple functions.

Use additional cuts when they improve pacing, reveal information,
increase tension or create an effective emotional progression.

Avoid selecting multiple segments that communicate essentially
the same information.

============================================================
3. DURATION AND SEGMENT SELECTION
============================================================

The preferred total duration is 35 to 45 seconds.

When the screenplay contains enough meaningful material,
use the available duration to develop the promotional sequence.

Do not automatically target the minimum 20-second duration.

Do not extend a montage with filler, repetitive dialogue,
unnecessary reactions or irrelevant moments.

Do not shorten a strong scene merely to fit a preferred
number of segments.

The quality of the montage takes priority over hitting
the preferred duration or segment count.

A shorter or longer valid montage is acceptable when
the source material justifies it.

============================================================
3A. SELECT COMPLETE DRAMATIC MOMENTS
============================================================

A segment represents a COMPLETE DRAMATIC MOMENT, not merely
a single dialogue line or isolated screenplay event.

When selecting a moment, consider including:

- The action or dialogue that establishes the situation.
- The central dialogue, reveal or dramatic action.
- The immediate reaction or consequence, when relevant.

A segment may contain multiple consecutive screenplay events.

Do not automatically use the start and end timestamps of a
single event as the segment boundaries.

Prefer a longer continuous interval when it captures the
complete dramatic moment more effectively.

For example, instead of selecting only a 2-second accusation,
consider including the preceding confrontation and the
immediate reaction if they strengthen the moment.

However:

- Do not include unrelated dialogue or filler.
- Do not extend a segment merely to increase its duration.
- Preserve the dramatic meaning of the original scene.
- Both boundaries must still match valid screenplay timestamps.
- Scene boundaries and event boundaries are both valid
  timestamp boundaries.

IMPORTANT:

A segment can span several consecutive events within a scene.
You are selecting footage for an edited video, not extracting
individual dialogue lines.

============================================================
3B. PLAN THE TOTAL EDITED DURATION
============================================================

Before finalizing each montage, estimate the duration
contributed by every selected segment.

Use the following approach:

1. Identify the strongest opening moment.
2. Identify the necessary context and narrative developments.
3. Identify the strongest final cliffhanger.
4. Select complete source intervals for these moments.
5. Calculate the sum of their individual durations.
6. Ensure the total is at least 20 seconds and no more
   than 60 seconds.
7. Prefer 35 to 45 seconds when sufficient meaningful
   source material exists.
8. Add other relevant moments when they improve the
   promotional sequence.

The sum of all segment durations MUST be between
20 and 60 seconds.

Do not submit a candidate that falls below 20 seconds.

Do not use artificial padding, repeated footage or
unnecessary dialogue to meet the duration requirement.

Do not confuse the difference between the earliest start
and latest end timestamps with the actual edited duration.

Only the sum of selected segment durations counts.

============================================================
4. OPENING HOOK
============================================================

The first segment must immediately capture attention.

Look for:

- Unexpected actions or reveals.
- Emotionally intense moments.
- Dramatic confrontations.
- Intriguing dialogue.
- Unusual or visually compelling situations.
- Questions that make viewers want to understand what is happening.

The opening should create curiosity within the first three seconds.

============================================================
5. NARRATIVE DEVELOPMENT
============================================================

Select additional moments that strengthen the story being told.

Consider:

- How the conflict develops.
- What new information is revealed.
- How characters react to events.
- Whether emotional stakes increase.
- Whether the sequence creates anticipation.
- Whether each cut contributes something meaningful.

The moments do not need to follow the original chronological order.

However, the final edited sequence must feel intentional,
coherent and understandable.

Do not create random collections of dramatic moments.

============================================================
6. ENDING CLIFFHANGER
============================================================

The final segment must leave viewers wanting to know what happens next.

Prefer:

- An unresolved confrontation.
- A shocking revelation.
- An unexpected decision.
- A threatening situation.
- An emotional turning point.
- A question or statement that creates suspense.

Avoid endings that completely resolve the central conflict
or make the viewer feel that the entire story has already been told.

============================================================
7. STANDALONE COMPREHENSION
============================================================

The montage must make sense to someone unfamiliar with the show.

The viewer should understand enough about:

- Who is involved.
- What is happening.
- Why the situation matters.
- What is at stake.

Use segment selection and editing order to establish context.

Do not rely on unexplained character relationships or events
that are never established within the selected moments.

============================================================
8. SOURCE TIMESTAMP ACCURACY
============================================================

Timestamp accuracy is critical.

Use only timestamps and event boundaries present in the screenplay.

For every segment:

- start_timestamp must correspond to an actual source boundary.
- end_timestamp must correspond to an actual source boundary.
- start_timestamp must be earlier than end_timestamp.
- Both timestamps must belong to the original episode timeline.
- Do not invent, estimate or round timestamps.
- Do not modify timestamps to achieve a preferred duration.

Use the exact MM:SS.ss timestamp format.

Total montage duration is the SUM of the individual segment
durations, not the difference between the first segment's
start and the final segment's end.

============================================================
9. OUTPUT REQUIREMENTS
============================================================

Return valid JSON only.

Do not include Markdown fences, commentary or explanations
outside the JSON object.

Use this exact structure:

{{
    "episode_number": "{screenplay_data['episode_number']}",
    "candidates": [
        {{
            "candidate_id": "{episode_id}_M01",
            "segments": [
                {{
                    "segment_order": 1,
                    "start_timestamp": "00:00.00",
                    "end_timestamp": "00:08.00",
                    "purpose": "Explain the narrative purpose of this segment.",
                    "source_event_ids": [1, 2, 3]
                }}
            ],
            "hook": "Describe the opening hook.",
            "cliffhanger": "Describe the ending cliffhanger.",
            "why_it_works": "Explain why the complete montage works as a promotional video."
        }}
    ]
}}

IMPORTANT OUTPUT RULES:

- Every candidate MUST have a total duration between 20 and 60 seconds.
- Generate up to 10 candidates, not necessarily 10.
- Every candidate must have between 3 and 8 segments.
- Candidate IDs must follow the format {episode_id}_M01,
  {episode_id}_M02, {episode_id}_M03, and so on.
- Candidate IDs must be unique.
- segment_order must start at 1 and increase sequentially.
- Every segment must include all required fields.
- Every segment must have a meaningful purpose.
- source_event_ids must correspond to actual screenplay events
  occurring within that segment.
- Every candidate must include hook, cliffhanger and why_it_works.
- Do not include calculated duration fields.
- Do not return candidates that fail the duration requirements.
- Do not invent dialogue, events, timestamps or source event IDs.

============================================================
SCREENPLAY DATA
============================================================

{json.dumps(prompt_data, ensure_ascii=False, indent=2)}
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

        if not 3 <= len(segments) <= 8:
            reasons.append(
                f"A montage must contain 3–8 segments; "
                f"received {len(segments)}."
            )

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
