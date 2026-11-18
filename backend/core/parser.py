
import re
from pathlib import Path


# Expected timestamp format: 00:00.00
TIMESTAMP_PATTERN = r"\d{2}:\d{2}\.\d{2}"

EVENT_HEADER_PATTERN = re.compile(r"^\*\*(.+):\*\*$")
SCENE_PATTERN = re.compile(r"^SCENE\s+(\d+)\s*$")

TIME_RANGE_PATTERN = re.compile(
    rf"^Start:\s*\(({TIMESTAMP_PATTERN})\)\s*\|\s*"
    rf"End:\s*\(({TIMESTAMP_PATTERN})\)\s*$"
)


def timestamp_to_seconds(timestamp):
    """Convert MM:SS.ss into seconds."""

    minutes, remaining = timestamp.split(":")
    seconds = float(remaining)

    return int(minutes) * 60 + seconds


def extract_screenplay(file):
    """Read a screenplay from a path or uploaded Streamlit file."""

    if isinstance(file, (str, Path)):
        path = Path(file)
        text = path.read_text(encoding="utf-8-sig")
        filename = path.name

    else:
        text = file.getvalue().decode("utf-8-sig")
        filename = file.name

    return parse_screenplay(text, filename)


def parse_screenplay(text, filename="unknown.txt"):
    """Parse screenplay metadata, scenes and timestamped events."""

    lines = text.splitlines()

    screenplay = {
        "filename": filename,
        "episode_number": None,
        "logline": None,
        "continuity_notes": None,
        "scenes": [],
        "event_count": 0
    }

    current_scene = None
    current_event = None
    source_order = 0

    def save_event():
        nonlocal current_event

        if current_event is not None and current_scene is not None:
            current_event["text"] = "\n".join(
                current_event["text_lines"]
            ).strip()

            del current_event["text_lines"]

            current_scene["events"].append(current_event)
            screenplay["event_count"] += 1

            current_event = None

    for line in lines:
        stripped = line.strip()

        if not stripped:
            continue

        # Episode metadata
        if stripped.startswith("EPISODE NUMBER:"):
            screenplay["episode_number"] = (
                stripped.split(":", 1)[1].strip()
            )
            continue

        if stripped.startswith("EPISODE LOGLINE:"):
            screenplay["logline"] = (
                stripped.split(":", 1)[1].strip()
            )
            continue

        if stripped.startswith("CONTINUITY NOTES:"):
            screenplay["continuity_notes"] = (
                stripped.split(":", 1)[1].strip()
            )
            continue

        # Ignore script boundary markers
        if stripped in ("--- SCRIPT START ---", "--- SCRIPT END ---"):
            continue

        # Detect a new scene
        scene_match = SCENE_PATTERN.match(stripped)

        if scene_match:
            save_event()

            current_scene = {
                "scene_number": int(scene_match.group(1)),
                "heading": None,
                "start_timestamp": None,
                "end_timestamp": None,
                "start_seconds": None,
                "end_seconds": None,
                "events": []
            }

            screenplay["scenes"].append(current_scene)
            continue

        if current_scene is None:
            continue

        # Scene metadata
        if stripped.upper().startswith("SCENE HEADING:"):
            current_scene["heading"] = (
                stripped.split(":", 1)[1].strip()
            )
            continue

        if stripped.upper().startswith("SCENE START:"):
            timestamp = stripped.split(":", 1)[1].strip()

            current_scene["start_timestamp"] = timestamp
            current_scene["start_seconds"] = timestamp_to_seconds(
                timestamp
            )
            continue

        if stripped.upper().startswith("SCENE END:"):
            timestamp = stripped.split(":", 1)[1].strip()

            current_scene["end_timestamp"] = timestamp
            current_scene["end_seconds"] = timestamp_to_seconds(
                timestamp
            )
            continue

        # Ignore section labels
        if stripped == "Actions and dialogue:":
            continue

        # Detect a new event block
        header_match = EVENT_HEADER_PATTERN.match(stripped)

        if header_match:
            save_event()

            header = header_match.group(1).strip()
            header_upper = header.upper()

            if header_upper == "CAMERA":
                category = "camera"
                speaker = None
                delivery = None

            elif header_upper == "ACTION":
                category = "action"
                speaker = None
                delivery = None

            elif header_upper in ("SOUND FX", "SOUND"):
                category = "sound"
                speaker = None
                delivery = None

            else:
                category = "dialogue"

                # Extract character and delivery description
                parts = re.match(
                    r"^(.*?)\s*\((.*)\)$",
                    header
                )

                if parts:
                    speaker = parts.group(1).strip()
                    delivery = parts.group(2).strip()
                else:
                    speaker = header
                    delivery = None

            current_event = {
                "event_id": source_order + 1,
                "source_order": source_order,
                "header": header,
                "category": category,
                "speaker": speaker,
                "delivery": delivery,
                "start_timestamp": None,
                "end_timestamp": None,
                "start_seconds": None,
                "end_seconds": None,
                "text_lines": []
            }

            source_order += 1
            continue

        # Parse event timestamps
        if current_event is not None:
            time_match = TIME_RANGE_PATTERN.match(stripped)

            if time_match:
                start, end = time_match.groups()

                current_event["start_timestamp"] = start
                current_event["end_timestamp"] = end

                current_event["start_seconds"] = (
                    timestamp_to_seconds(start)
                )

                current_event["end_seconds"] = (
                    timestamp_to_seconds(end)
                )

            else:
                current_event["text_lines"].append(stripped)

    # Save the final event
    save_event()

    # Create a chronological view without changing source order
    for scene in screenplay["scenes"]:
        scene["chronological_events"] = sorted(
            scene["events"],
            key=lambda event: (
                event["start_seconds"]
                if event["start_seconds"] is not None
                else float("inf"),
                event["source_order"]
            )
        )

    return screenplay
