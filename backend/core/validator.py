
def validate_screenplay(screenplay):
    """
    Validate parsed screenplay structure and timestamp integrity.

    Returns:
        {
            "valid": bool,
            "errors": list,
            "warnings": list,
            "summary": dict
        }
    """

    errors = []
    warnings = []

    scenes = screenplay.get("scenes", [])

    if not screenplay.get("episode_number"):
        warnings.append("Episode number is missing.")

    if not screenplay.get("logline"):
        warnings.append("Episode logline is missing.")

    if not scenes:
        errors.append("No scenes were detected.")

    total_events = 0
    missing_timestamps = 0
    invalid_intervals = 0
    missing_content = 0

    for scene in scenes:

        scene_number = scene.get("scene_number")

        scene_start = scene.get("start_seconds")
        scene_end = scene.get("end_seconds")

        if scene_start is None or scene_end is None:
            errors.append(
                f"Scene {scene_number}: missing scene boundaries."
            )

        elif scene_end < scene_start:
            errors.append(
                f"Scene {scene_number}: end precedes start."
            )

        events = scene.get("events", [])
        total_events += len(events)

        for event in events:

            event_id = event.get("event_id")

            start = event.get("start_seconds")
            end = event.get("end_seconds")

            if start is None or end is None:
                missing_timestamps += 1
                warnings.append(
                    f"Event {event_id}: missing timestamp."
                )
                continue

            if end < start:
                invalid_intervals += 1
                errors.append(
                    f"Event {event_id}: end precedes start."
                )
                continue

            if scene_start is not None and start < scene_start:
                warnings.append(
                    f"Event {event_id}: starts before scene boundary."
                )

            if scene_end is not None and end > scene_end:
                warnings.append(
                    f"Event {event_id}: ends after scene boundary."
                )

            if not event.get("text"):
                missing_content += 1
                warnings.append(
                    f"Event {event_id}: event content is empty."
                )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "scene_count": len(scenes),
            "event_count": total_events,
            "missing_timestamps": missing_timestamps,
            "invalid_intervals": invalid_intervals,
            "missing_content": missing_content
        }
    }
