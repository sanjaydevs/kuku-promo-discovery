

def timestamp_to_seconds(timestamp):
    """
    Convert MM:SS.ss timestamp into seconds.
    """

    minutes, seconds = timestamp.split(":")
    return int(minutes) * 60 + float(seconds)


def calculate_overlap(candidate_a, candidate_b):
    """
    Calculate the proportion of the shorter clip that overlaps
    with the longer clip.
    """

    start_a = timestamp_to_seconds(candidate_a["start_timestamp"])
    end_a = timestamp_to_seconds(candidate_a["end_timestamp"])

    start_b = timestamp_to_seconds(candidate_b["start_timestamp"])
    end_b = timestamp_to_seconds(candidate_b["end_timestamp"])

    overlap = max(
        0,
        min(end_a, end_b) - max(start_a, start_b)
    )

    shorter_duration = min(
        end_a - start_a,
        end_b - start_b
    )

    if shorter_duration <= 0:
        return 0

    return overlap / shorter_duration


def calculate_selection_score(candidate, selected_candidates):
    """
    Adjust the creative score based on redundancy with
    candidates already selected.
    """

    original_score = candidate["weighted_score"]

    if not selected_candidates:
        return original_score, 0

    max_overlap = max(
        calculate_overlap(candidate, selected)
        for selected in selected_candidates
    )

    # Penalise substantial overlap.
    redundancy_penalty = max_overlap * 2.0

    adjusted_score = max(
        0,
        round(original_score - redundancy_penalty, 2)
    )

    return adjusted_score, round(redundancy_penalty, 2)


def rank_promos(evaluations, max_promos=None):
    """
    Rank promo candidates using creative quality and redundancy.

    Greedily selects the highest-scoring candidate, then adjusts
    the remaining candidates against the selected set.
    """

    candidates = evaluations["evaluations"]

    if not candidates:
        return {
            "episode_number": evaluations["episode_number"],
            "ranked_candidates": [],
        }

    remaining = candidates.copy()
    selected = []

    while remaining:
        scored_candidates = []

        for candidate in remaining:
            adjusted_score, penalty = calculate_selection_score(
                candidate,
                selected
            )

            scored_candidates.append({
                **candidate,
                "adjusted_score": adjusted_score,
                "redundancy_penalty": penalty,
            })

        # Select the candidate with the highest adjusted score.
        # Use original creative score as a tie-breaker.
        best_candidate = max(
            scored_candidates,
            key=lambda item: (
                item["adjusted_score"],
                item["weighted_score"]
            )
        )

        selected.append(best_candidate)

        remaining = [
            candidate
            for candidate in remaining
            if candidate["candidate_id"]
            != best_candidate["candidate_id"]
        ]

        if max_promos is not None and len(selected) >= max_promos:
            break

    for index, candidate in enumerate(selected, start=1):
        candidate["rank"] = index

    return {
        "episode_number": evaluations["episode_number"],
        "ranked_candidates": selected,
    }


def rank_all_episodes(episode_results, max_promos=None):
    """
    Create one global ranking across all processed episodes.

    Redundancy is calculated only between clips from the same
    episode because timestamps from different episodes cannot
    be compared directly.
    """

    candidates = []

    # Combine candidates from successful episodes.
    for episode_result in episode_results:
        if not episode_result.get("success"):
            continue

        episode_number = episode_result.get(
            "episode_number",
            "Unknown"
        )

        for candidate in episode_result.get(
            "ranked_candidates", []
        ):
            candidate_copy = candidate.copy()

            candidate_copy["episode_number"] = episode_number

            # Start global ranking from the original creative score.
            candidate_copy["global_score"] = candidate_copy[
                "weighted_score"
            ]

            candidates.append(candidate_copy)

    if not candidates:
        return []

    remaining = candidates.copy()
    selected = []

    while remaining:
        scored_candidates = []

        for candidate in remaining:
            # Compare overlap only within the same episode.
            same_episode_selected = [
                item for item in selected
                if item["episode_number"]
                == candidate["episode_number"]
            ]

            adjusted_score, penalty = calculate_selection_score(
                candidate,
                same_episode_selected
            )

            scored_candidates.append({
                **candidate,
                "global_score": adjusted_score,
                "global_redundancy_penalty": penalty,
            })

        best_candidate = max(
            scored_candidates,
            key=lambda item: (
                item["global_score"],
                item["weighted_score"]
            )
        )

        selected.append(best_candidate)

        remaining = [
            candidate for candidate in remaining
            if not (
                candidate["candidate_id"]
                == best_candidate["candidate_id"]
                and candidate["episode_number"]
                == best_candidate["episode_number"]
            )
        ]

        if max_promos is not None and len(selected) >= max_promos:
            break

    for index, candidate in enumerate(selected, start=1):
        candidate["global_rank"] = index

    return selected



def rank_top_per_episode(episode_results):
    """
    Select exactly one highest-scoring promo per episode,
    then rank the selected promos globally.
    """

    best_by_episode = {}

    for episode_result in episode_results:
        if not episode_result.get("success"):
            continue

        episode_number = episode_result.get(
            "episode_number", "Unknown"
        )

        ranked_candidates = episode_result.get(
            "ranked_candidates", []
        )

        if not ranked_candidates:
            continue

        # Select the highest adjusted-score candidate.
        best_candidate = max(
            ranked_candidates,
            key=lambda candidate: (
                candidate.get(
                    "adjusted_score",
                    candidate["weighted_score"]
                ),
                candidate["weighted_score"]
            )
        ).copy()

        best_candidate["episode_number"] = episode_number

        # Keep only one candidate for each episode.
        existing = best_by_episode.get(episode_number)

        if existing is None or (
            best_candidate["adjusted_score"],
            best_candidate["weighted_score"]
        ) > (
            existing["adjusted_score"],
            existing["weighted_score"]
        ):
            best_by_episode[episode_number] = best_candidate

    # Combine the selected promos and rank globally.
    top_candidates = sorted(
        best_by_episode.values(),
        key=lambda candidate: (
            candidate["adjusted_score"],
            candidate["weighted_score"]
        ),
        reverse=True
    )

    for index, candidate in enumerate(top_candidates, start=1):
        candidate["global_rank"] = index

    return top_candidates
