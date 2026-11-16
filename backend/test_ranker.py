
from core.parser import extract_screenplay
from core.discovery import discover_promos
from core.evaluator import evaluate_promos
from core.ranker import rank_promos


# Load screenplay
screenplay = extract_screenplay("data/input/episode_1.txt")

# Discover candidates
discovery_result = discover_promos(screenplay)

# Evaluate candidates
evaluation_result = evaluate_promos(
    screenplay_data=discovery_result,
    candidates=discovery_result["candidates"]
)

# Rank candidates
ranking_result = rank_promos(evaluation_result)

print("\nFINAL PROMO RANKING")

for candidate in ranking_result["ranked_candidates"]:
    print(f"\nRank: {candidate['rank']}")
    print(f"Candidate: {candidate['candidate_id']}")
    print(f"Original score: {candidate['weighted_score']}")
    print(f"Adjusted score: {candidate['adjusted_score']}")
    print(f"Redundancy penalty: {candidate['redundancy_penalty']}")
    print(f"Time: {candidate['start_timestamp']} - {candidate['end_timestamp']}")
    print(f"Reason: {candidate['redundancy_note']}")
