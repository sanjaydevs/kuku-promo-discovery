
import json

from core.parser import extract_screenplay
from core.discovery import discover_promos
from core.evaluator import evaluate_promos


# Load the Episode 1 screenplay
screenplay = extract_screenplay("data/input/episode_1.txt")

# Discover promo candidates
discovery_result = discover_promos(screenplay)

print(f"Discovered {len(discovery_result['candidates'])} candidates.")

# Evaluate the candidates
evaluation_result = evaluate_promos(
    screenplay_data=discovery_result,
    candidates=discovery_result["candidates"]
)

# Display scores
for candidate in evaluation_result["evaluations"]:
    print("\n", candidate["candidate_id"])
    print("Weighted score:", candidate["weighted_score"])
    print("Scores:", candidate["scores"])
    print("Redundancy:", candidate["redundancy_note"])

# Save raw evaluation response for inspection
with open(
    "data/outputs/episode_1_evaluation_raw.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        evaluation_result,
        file,
        indent=2,
        ensure_ascii=False
    )

print("\nEvaluation completed.")
