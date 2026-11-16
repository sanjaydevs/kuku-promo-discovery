
import json
import streamlit as st
import pandas as pd

from core.pipeline import process_episodes


st.set_page_config(
    page_title="Kuku Promo Discovery",
    page_icon="🎬",
    layout="wide"
)

st.title("Kuku AI — Promo Discovery")
st.write(
    "Upload screenplay files to discover, evaluate and rank "
    "promotional video clips."
)

st.divider()

uploaded_files = st.file_uploader(
    "Upload screenplay files",
    type=["txt"],
    accept_multiple_files=True
)

if uploaded_files:
    st.write(f"**{len(uploaded_files)} episode(s) uploaded.**")

    if st.button("Discover Promos", type="primary"):
        all_results = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for index, uploaded_file in enumerate(uploaded_files):
            status_text.write(
                f"Processing {uploaded_file.name}..."
            )

            try:
                result = process_episodes([uploaded_file])[0]
                all_results.append(result)

            except Exception as error:
                all_results.append({
                    "success": False,
                    "filename": uploaded_file.name,
                    "error": str(error)
                })

            progress_bar.progress(
                (index + 1) / len(uploaded_files)
            )

        st.session_state["pipeline_results"] = all_results

        status_text.success("Processing completed.")

if "pipeline_results" in st.session_state:
    results = st.session_state["pipeline_results"]

    st.divider()
    st.header("Discovery Results")

    successful_results = [
        result for result in results if result["success"]
    ]

    failed_results = [
        result for result in results if not result["success"]
    ]

    col1, col2, col3 = st.columns(3)

    total_candidates = sum(
        len(result.get("ranked_candidates", []))
        for result in successful_results
    )

    col1.metric("Episodes processed", len(successful_results))
    col2.metric("Promos discovered", total_candidates)
    col3.metric("Failed episodes", len(failed_results))

    for result in results:
        filename = result.get("filename", "Unknown file")

        with st.expander(
            f"{filename} — "
            f"{'Successful' if result['success'] else 'Failed'}",
            expanded=result["success"]
        ):

            if not result["success"]:
                st.error(result.get("error", "Processing failed."))

                if result.get("validation"):
                    st.json(result["validation"])

                continue

            st.subheader(
                f"Episode: {result.get('episode_number', 'Unknown')}"
            )

            st.write("**Structural validation**")
            st.json(result.get("validation", {}))

            ranked_candidates = result.get(
                "ranked_candidates", []
            )

            if not ranked_candidates:
                st.info("No promotional candidates were discovered.")
                continue

            st.subheader("Ranked promotional clips")

            ranking_table = []

            for candidate in ranked_candidates:
                ranking_table.append({
                    "Rank": candidate["rank"],
                    "Candidate": candidate["candidate_id"],
                    "Start": candidate["start_timestamp"],
                    "End": candidate["end_timestamp"],
                    "Original score": candidate["weighted_score"],
                    "Adjusted score": candidate["adjusted_score"],
                    "Redundancy penalty": candidate["redundancy_penalty"],
                })

            st.dataframe(
                pd.DataFrame(ranking_table),
                use_container_width=True,
                hide_index=True
            )

            for candidate in ranked_candidates:
                with st.container(border=True):
                    st.markdown(
                        f"### Rank {candidate['rank']} — "
                        f"{candidate['candidate_id']}"
                    )

                    st.write(
                        f"**Duration:** "
                        f"{candidate['start_timestamp']} – "
                        f"{candidate['end_timestamp']}"
                    )

                    col1, col2 = st.columns(2)

                    col1.metric(
                        "Creative score",
                        f"{candidate['weighted_score']}/10"
                    )

                    col2.metric(
                        "Adjusted score",
                        f"{candidate['adjusted_score']}/10"
                    )

                    st.write("**Opening hook**")
                    st.write(candidate.get("hook", "Not available"))

                    st.write("**Cliffhanger**")
                    st.write(candidate.get("cliffhanger", "Not available"))

                    st.write("**Why this promo works**")
                    st.write(
                        candidate.get("why_it_works", "Not available")
                    )

                    st.write("**Evaluation scores**")
                    st.json(candidate.get("scores", {}))

                    st.write("**Evaluation explanations**")
                    st.json(candidate.get("explanations", {}))

                    st.write("**Redundancy assessment**")
                    st.write(
                        candidate.get("redundancy_note", "Not available")
                    )

                    st.write("**Source event IDs**")
                    st.write(candidate.get("source_event_ids", []))

            st.subheader("Raw AI outputs")

            raw_outputs = result.get("raw_outputs", {})

            with st.expander("Discovery response"):
                st.text(raw_outputs.get("discovery") or "Not available")

            with st.expander("Evaluation response"):
                st.text(raw_outputs.get("evaluation") or "Not available")

            st.download_button(
                label="Download episode results (JSON)",
                data=json.dumps(
                    result,
                    indent=2,
                    ensure_ascii=False
                ),
                file_name=f"{filename.rsplit('.', 1)[0]}_results.json",
                mime="application/json",
                key=f"download_{filename}"
            )

    if failed_results:
        st.warning(
            f"{len(failed_results)} episode(s) could not be processed. "
            "Check the errors shown above."
        )
