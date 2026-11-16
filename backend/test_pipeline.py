
from core.pipeline import process_episode

result = process_episode("data/input/episode_1.txt")

print("\nPIPELINE RESULT")
print("Success:", result["success"])
print("Episode:", result.get("episode_number"))

if result["success"]:
    print("Candidates discovered:", len(result["candidates"]))
    print("Candidates ranked:", len(result["ranked_candidates"]))

    print("\nFINAL RANKING")

    for candidate in result["ranked_candidates"]:
        print(
            candidate["rank"],
            candidate["candidate_id"],
            candidate["adjusted_score"],
            candidate["start_timestamp"],
            candidate["end_timestamp"],
        )
else:
    print("Error:", result.get("error"))
