
from core.ranker import rank_all_episodes


# Simulated results from two episodes.
episode_results = [
    {
        "success": True,
        "episode_number": "Episode 1",
        "ranked_candidates": [
            {
                "candidate_id": "EP1_P01",
                "start_timestamp": "00:00.00",
                "end_timestamp": "00:40.00",
                "weighted_score": 9.0,
            },
            {
                "candidate_id": "EP1_P02",
                "start_timestamp": "00:20.00",
                "end_timestamp": "00:50.00",
                "weighted_score": 8.8,
            },
        ],
    },
    {
        "success": True,
        "episode_number": "Episode 2",
        "ranked_candidates": [
            {
                "candidate_id": "EP2_P01",
                "start_timestamp": "00:10.00",
                "end_timestamp": "00:45.00",
                "weighted_score": 9.2,
            },
            {
                "candidate_id": "EP2_P02",
                "start_timestamp": "01:00.00",
                "end_timestamp": "01:40.00",
                "weighted_score": 8.5,
            },
        ],
    },
]


ranked = rank_all_episodes(episode_results)

print("\nGLOBAL PROMO RANKING")

for candidate in ranked:
    print(
        f"Rank {candidate['global_rank']}: "
        f"{candidate['candidate_id']} | "
        f"{candidate['episode_number']} | "
        f"Original: {candidate['weighted_score']} | "
        f"Global: {candidate['global_score']} | "
        f"Penalty: {candidate['global_redundancy_penalty']}"
    )
