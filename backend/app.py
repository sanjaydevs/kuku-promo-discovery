
import json

import pandas as pd
import streamlit as st

from core.pipeline import process_episodes
from core.ranker import rank_all_episodes, rank_top_per_episode
from config import GEMINI_MODEL, OPENROUTER_MODEL

from pathlib import Path

output_file = (
    Path(__file__).resolve().parent
    / "data"
    / "outputs"
    / "latest_results.json"
)

if "pipeline_results" not in st.session_state and output_file.exists():
    with open(output_file, "r", encoding="utf-8") as f:
        st.session_state["pipeline_results"] = json.load(f)
        

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Kuku Promo Discovery",
    page_icon="🎬",
    layout="wide"
)

st.title("Kuku AI — Promo Discovery")

st.write(
    "Discover, evaluate and rank promotional clips "
    "from screenplay episodes."
)


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def render_promo_details(candidate):

    st.divider()

    st.subheader(
        f"{candidate.get('candidate_id', 'Promo')} — Details"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Start",
        candidate.get("start_timestamp", "N/A")
    )

    col2.metric(
        "End",
        candidate.get("end_timestamp", "N/A")
    )

    col3.metric(
        "Creative score",
        f"{candidate.get('weighted_score', 0)}/10"
    )

    if "adjusted_score" in candidate:
        st.metric(
            "Adjusted score",
            f"{candidate['adjusted_score']}/10"
        )

    if "global_score" in candidate:
        st.metric(
            "Global score",
            f"{candidate['global_score']}/10"
        )

    st.markdown("#### Opening hook")
    st.write(candidate.get("hook", "Not available"))

    st.markdown("#### Cliffhanger")
    st.write(candidate.get("cliffhanger", "Not available"))

    st.markdown("#### Why this promo works")
    st.write(candidate.get("why_it_works", "Not available"))

    st.markdown("#### Evaluation scores")
    st.json(candidate.get("scores", {}))

    st.markdown("#### Evaluation explanations")
    st.json(candidate.get("explanations", {}))

    st.markdown("#### Redundancy assessment")
    st.write(
        candidate.get("redundancy_note", "Not available")
    )

    with st.expander("Additional candidate information"):
        st.write("**Source event IDs**")
        st.write(candidate.get("source_event_ids", []))

        st.write("**Redundancy penalty**")
        st.write(
            candidate.get(
                "global_redundancy_penalty",
                candidate.get("redundancy_penalty", 0)
            )
        )


def render_selectable_table(table_data, key):

    dataframe = pd.DataFrame(table_data)

    event = st.dataframe(
        dataframe,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=key
    )

    selected_rows = event.selection.rows

    if selected_rows:
        return selected_rows[0]

    return None


# --------------------------------------------------
# AI MODEL CONFIGURATION
# --------------------------------------------------

st.divider()
st.header("AI Model Configuration")

provider_options = {
    "Gemini": "gemini",
    "OpenRouter": "openrouter"
}

config_col1, config_col2 = st.columns(2)

with config_col1:

    selected_provider_label = st.selectbox(
        "Select AI provider",
        options=list(provider_options.keys()),
        index=0,
        key="provider_selection"
    )

selected_provider = provider_options[selected_provider_label]

with config_col2:

    if selected_provider == "gemini":

        selected_model = st.text_input(
            "Gemini model",
            value=GEMINI_MODEL,
            key="gemini_model_selection"
        )

    else:

        selected_model = st.text_input(
            "OpenRouter model",
            value=OPENROUTER_MODEL,
            key="openrouter_model_selection"
        )


# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------

st.divider()
st.header("Upload Episodes")

uploaded_files = st.file_uploader(
    "Upload screenplay files",
    type=["txt"],
    accept_multiple_files=True
)

if uploaded_files:

    st.caption(
        f"{len(uploaded_files)} episode(s) uploaded."
    )

    if st.button(
        "Discover Promos",
        type="primary",
        use_container_width=True
    ):

        if not selected_model.strip():

            st.error("Please enter a valid model name.")

        else:

            all_results = []

            progress_bar = st.progress(0)
            status_text = st.empty()

            for index, uploaded_file in enumerate(uploaded_files):

                status_text.write(
                    f"Processing {uploaded_file.name}..."
                )

                try:

                    result = process_episodes(
                        [uploaded_file],
                        provider=selected_provider,
                        model=selected_model.strip()
                    )[0]

                    result["provider"] = selected_provider
                    result["model"] = selected_model.strip()

                    all_results.append(result)

                except Exception as error:

                    all_results.append({
                        "success": False,
                        "filename": uploaded_file.name,
                        "provider": selected_provider,
                        "model": selected_model.strip(),
                        "error": str(error)
                    })

                progress_bar.progress(
                    (index + 1) / len(uploaded_files)
                )

            st.session_state["pipeline_results"] = all_results
            
            from pathlib import Path

            output_dir = Path(__file__).resolve().parent / "data" / "outputs"
            output_dir.mkdir(parents=True, exist_ok=True)

            with open(output_dir / "latest_results.json", "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)

            status_text.success("Processing completed.")


# --------------------------------------------------
# DISCOVERY RESULTS
# --------------------------------------------------

if "pipeline_results" in st.session_state:

    results = st.session_state["pipeline_results"]

    successful_results = [
        result for result in results
        if result.get("success")
    ]

    failed_results = [
        result for result in results
        if not result.get("success")
    ]

    st.divider()
    st.header("Discovery Results")


    # --------------------------------------------------
    # PROCESSING SUMMARY
    # --------------------------------------------------

    total_candidates = sum(
        len(result.get("ranked_candidates", []))
        for result in successful_results
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Episodes processed",
        len(successful_results)
    )

    col2.metric(
        "Promos discovered",
        total_candidates
    )

    col3.metric(
        "Failed episodes",
        len(failed_results)
    )


    # --------------------------------------------------
    # OVERALL PROMO RANKING
    # --------------------------------------------------

    st.divider()
    st.header("Overall Promo Ranking")

    ranking_mode = st.radio(
        "Select ranking mode",
        options=[
            "All discovered promos",
            "Top promo from each episode"
        ],
        horizontal=True,
        key="ranking_mode"
    )

    if ranking_mode == "All discovered promos":

        global_ranking = rank_all_episodes(results)

    else:

        global_ranking = rank_top_per_episode(results)

    st.session_state["global_ranking"] = global_ranking

    if global_ranking:

        st.caption(
            f"{len(global_ranking)} promo(s) ranked. "
            "Select a row to view its details."
        )

        global_table = []

        for candidate in global_ranking:

            global_table.append({

                "Global Rank": candidate["global_rank"],

                "Episode": candidate["episode_number"],

                "Candidate": candidate["candidate_id"],

                "Start": candidate["start_timestamp"],

                "End": candidate["end_timestamp"],

                "Creative Score": candidate["weighted_score"],

                "Global Score": candidate.get(
                    "global_score",
                    candidate.get("adjusted_score", 0)
                ),

                "Redundancy Penalty": candidate.get(
                    "global_redundancy_penalty",
                    candidate.get("redundancy_penalty", 0)
                ),

            })

        selected_global_index = render_selectable_table(
            global_table,
            key="global_ranking_table"
        )

        if selected_global_index is not None:

            if selected_global_index < len(global_ranking):

                selected_candidate = global_ranking[
                    selected_global_index
                ]

                render_promo_details(selected_candidate)

    else:

        st.info("No promotional candidates available.")


    # --------------------------------------------------
    # EPISODE-WISE RESULTS
    # --------------------------------------------------

    st.divider()
    st.header("Episode-wise Results")

    if successful_results:

        episode_options = {}

        for index, result in enumerate(successful_results):

            episode_number = result.get(
                "episode_number",
                "Unknown"
            )

            filename = result.get(
                "filename",
                "Unknown file"
            )

            label = f"Episode {episode_number} — {filename}"

            episode_options[label] = index

        selected_episode_label = st.selectbox(
            "Select an episode to inspect",
            options=list(episode_options.keys()),
            key="episode_selection"
        )

        selected_episode = successful_results[
            episode_options[selected_episode_label]
        ]

        ranked_candidates = selected_episode.get(
            "ranked_candidates",
            []
        )

        st.subheader(
            f"Ranked Promos — Episode "
            f"{selected_episode.get('episode_number', 'Unknown')}"
        )

        if ranked_candidates:

            st.caption(
                "Select a promo row to view its complete details."
            )

            episode_table = []

            for candidate in ranked_candidates:

                episode_table.append({

                    "Rank": candidate["rank"],

                    "Candidate": candidate["candidate_id"],

                    "Start": candidate["start_timestamp"],

                    "End": candidate["end_timestamp"],

                    "Creative Score": candidate["weighted_score"],

                    "Adjusted Score": candidate["adjusted_score"],

                    "Redundancy Penalty": candidate["redundancy_penalty"],

                })

            selected_episode_index = render_selectable_table(
                episode_table,
                key=f"episode_table_{selected_episode_label}"
            )

            if selected_episode_index is not None:

                if selected_episode_index < len(ranked_candidates):

                    selected_candidate = ranked_candidates[
                        selected_episode_index
                    ]

                    render_promo_details(selected_candidate)

        else:

            st.info(
                "No promotional candidates were discovered "
                "for this episode."
            )


        # --------------------------------------------------
        # RAW LLM OUTPUTS AND TECHNICAL DATA
        # --------------------------------------------------

        st.divider()

        with st.expander("Raw LLM outputs and technical data"):

            st.subheader("Discovery response")

            raw_outputs = selected_episode.get(
                "raw_outputs",
                {}
            )

            st.text_area(
                "Raw discovery output",
                value=raw_outputs.get("discovery") or "Not available",
                height=200,
                disabled=True,
                key=f"raw_discovery_{selected_episode_label}"
            )

            st.subheader("Evaluation response")

            st.text_area(
                "Raw evaluation output",
                value=raw_outputs.get("evaluation") or "Not available",
                height=200,
                disabled=True,
                key=f"raw_evaluation_{selected_episode_label}"
            )

            st.subheader("Structural validation")

            st.json(
                selected_episode.get("validation", {})
            )

            st.download_button(
                label="Download episode results (JSON)",
                data=json.dumps(
                    selected_episode,
                    indent=2,
                    ensure_ascii=False
                ),
                file_name=(
                    f"{selected_episode.get('filename', 'episode')}"
                    .rsplit(".", 1)[0] + "_results.json"
                ),
                mime="application/json",
                key=f"download_{selected_episode_label}"
            )


    # --------------------------------------------------
    # FAILED EPISODES
    # --------------------------------------------------

    if failed_results:

        st.divider()

        with st.expander(
            f"Failed Episodes ({len(failed_results)})"
        ):

            for result in failed_results:

                st.error(
                    f"{result.get('filename', 'Unknown file')}: "
                    f"{result.get('error', 'Processing failed.')}"
                )
