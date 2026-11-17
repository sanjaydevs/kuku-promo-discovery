
from core.parser import extract_screenplay
from core.validator import validate_screenplay
from core.discovery import discover_promos
from core.evaluator import evaluate_promos
from core.ranker import rank_promos


def process_episode(uploaded_file, provider="gemini", model=None):
    """
    Run the complete promo discovery pipeline for one episode.
    """

    # Step 1: Parse screenplay
    screenplay = extract_screenplay(uploaded_file)

    # Step 2: Validate parsed screenplay
    validation_result = validate_screenplay(screenplay)

    if not validation_result["valid"]:
        return {
            "success": False,
            "episode_number": screenplay.get("episode_number"),
            "validation": validation_result,
            "error": "Screenplay failed structural validation.",
        }

    # Step 3: Discover promo candidates
    discovery_result = discover_promos(
        screenplay,
        provider=provider,
        model=model
    )

    candidates = discovery_result["candidates"]

    if not candidates:
        return {
            "success": True,
            "episode_number": screenplay["episode_number"],
            "validation": validation_result,
            "candidates": [],
            "ranked_candidates": [],
            "raw_outputs": {
                "discovery": discovery_result.get("raw_response"),
                "evaluation": None,
            },
        }

    # Step 4: Evaluate candidates
    evaluation_result = evaluate_promos(
        screenplay_data=discovery_result,
        candidates=candidates,
        provider=provider,
        model=model
    )

    # Step 5: Rank candidates
    ranking_result = rank_promos(evaluation_result)

    return {
        "success": True,
        "episode_number": screenplay["episode_number"],
        "validation": validation_result,
        "candidates": candidates,
        "evaluations": evaluation_result["evaluations"],
        "ranked_candidates": ranking_result["ranked_candidates"],
        "raw_outputs": {
            "discovery": discovery_result.get("raw_response"),
            "evaluation": evaluation_result.get("raw_response"),
        },
    }


def process_episodes(uploaded_files, provider="gemini", model=None):
    """
    Process multiple uploaded episode files independently.
    A failure in one episode does not stop the remaining episodes.
    """

    results = []

    for uploaded_file in uploaded_files:

        filename = getattr(uploaded_file, "name", str(uploaded_file))

        try:
            result = process_episode(
                uploaded_file,
                provider=provider,
                model=model
            )

            result["filename"] = filename
            result["provider"] = provider
            result["model"] = model

            results.append(result)

        except Exception as error:
            results.append({
                "success": False,
                "filename": filename,
                "provider": provider,
                "model": model,
                "error": str(error),
            })

    return results
