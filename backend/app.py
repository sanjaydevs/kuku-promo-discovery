
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from config import GEMINI_MODEL, OPENROUTER_MODEL
from core.pipeline import process_episodes
from core.ranker import (
    rank_all_episodes,
    rank_top_per_episode,
    rank_all_montages,
    rank_top_montage_per_episode,
)


# ============================================================
# PATHS
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BACKEND_DIR / "data" / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "latest_results.json"
CSS_FILE = BACKEND_DIR / "styles.css"
FAVICON_PATH = Path(__file__).parent / "favicon.png"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Promo Finder - By Sanjay",
    page_icon=str(FAVICON_PATH),
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# DATA AND STYLE HELPERS
# ============================================================

def load_saved_results():
    """Load the last saved run without making any API requests."""

    if "pipeline_results" in st.session_state:
        return

    if OUTPUT_FILE.exists():
        try:
            with OUTPUT_FILE.open("r", encoding="utf-8") as file:
                loaded = json.load(file)

            if isinstance(loaded, list):
                st.session_state["pipeline_results"] = loaded
            else:
                st.session_state["pipeline_results"] = []

        except (OSError, json.JSONDecodeError) as error:
            st.session_state["pipeline_results"] = []
            st.warning(f"Could not load saved results: {error}")


def save_results(results):
    """Persist the latest run safely to the local outputs folder."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temporary_file = OUTPUT_FILE.with_suffix(".tmp")

    with temporary_file.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    temporary_file.replace(OUTPUT_FILE)


def load_css():
    """Load editable visual styles from backend/styles.css."""

    if CSS_FILE.exists():
        css = CSS_FILE.read_text(encoding="utf-8")

        st.markdown(
            f"<style>{css}</style>",
            unsafe_allow_html=True,
        )


def get_successful_and_failed(results):

    successful = [
        result for result in results
        if result.get("success")
    ]

    failed = [
        result for result in results
        if not result.get("success")
    ]

    return successful, failed


# ============================================================
# DISPLAY HELPERS
# ============================================================

def render_promo_details(candidate):
    """
    Shared renderer for Continuous Clips and Edited Montages.

    Montage candidates display their editing sequence.
    Continuous candidates display their source timestamps.
    """

    is_montage = bool(candidate.get("segments"))

    if is_montage:

        detail_meta = (
            f"Episode {candidate.get('episode_number', '—')}"
            f" &nbsp; · &nbsp; "
            f"{len(candidate.get('segments', []))} segments"
            f" &nbsp; · &nbsp; "
            f"{candidate.get('duration_seconds', 0):.2f}s"
        )

    else:

        detail_meta = (
            f"Episode {candidate.get('episode_number', '—')}"
            f" &nbsp; · &nbsp; "
            f"{candidate.get('start_timestamp', '—')}"
            f" – {candidate.get('end_timestamp', '—')}"
        )

    st.markdown(
        f"""
        <div class="detail-heading">
            <div class="eyebrow">SELECTED PROMO</div>
            <h3>{candidate.get("candidate_id", "Promo")}</h3>
            <div class="detail-meta">
                {detail_meta}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    creative_score = candidate.get("weighted_score", 0)

    adjusted_score = candidate.get(
        "adjusted_score",
        candidate.get("global_score", 0),
    )

    redundancy_penalty = candidate.get(
        "global_redundancy_penalty",
        candidate.get("redundancy_penalty", 0),
    )

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Creative score",
        f"{creative_score}/10",
    )

    metric2.metric(
        "Adjusted / global score",
        f"{adjusted_score}/10",
    )

    metric3.metric(
        "Redundancy penalty",
        f"{redundancy_penalty}",
    )

    # --------------------------------------------------------
    # MONTAGE EDITING SEQUENCE
    # --------------------------------------------------------

    if is_montage:

        st.markdown("#### Editing sequence")

        for segment in candidate.get("segments", []):

            with st.container(border=True):

                st.markdown(
                    f"**SEGMENT {segment.get('segment_order', '—')}**"
                )

                st.markdown(
                    f"**{segment.get('start_timestamp', '—')} "
                    f"– {segment.get('end_timestamp', '—')}**"
                )

                st.caption(
                    f"Duration: "
                    f"{segment.get('duration_seconds', 0):.2f}s"
                )

                st.write(
                    segment.get(
                        "purpose",
                        "No purpose provided.",
                    )
                )

                with st.expander("Source event IDs"):
                    st.write(
                        segment.get("source_event_ids", [])
                    )

    # --------------------------------------------------------
    # PROMOTIONAL EXPLANATION
    # --------------------------------------------------------

    left_col, right_col = st.columns(2)

    with left_col:

        st.markdown("#### Opening hook")
        st.write(
            candidate.get("hook") or "Not available"
        )

        if is_montage:
            st.markdown("#### Why this montage works")
        else:
            st.markdown("#### Why this promo works")

        st.write(
            candidate.get("why_it_works") or "Not available"
        )

    with right_col:

        st.markdown("#### Cliffhanger")
        st.write(
            candidate.get("cliffhanger") or "Not available"
        )

        st.markdown("#### Redundancy assessment")
        st.write(
            candidate.get("redundancy_note") or "Not available"
        )

    # --------------------------------------------------------
    # EVALUATION BREAKDOWN
    # --------------------------------------------------------

    with st.expander("Evaluation scores and explanations"):

        scores = candidate.get("scores", {})
        explanations = candidate.get("explanations", {})

        score_rows = []

        for criterion, score in scores.items():

            score_rows.append({
                "Criterion": criterion.replace("_", " ").title(),
                "Score": score,
                "Explanation": explanations.get(criterion, "—"),
            })

        if score_rows:

            st.dataframe(
                pd.DataFrame(score_rows),
                use_container_width=True,
                hide_index=True,
            )

        else:
            st.info("No evaluation scores are available.")

    # Continuous clips retain their source event information.
    if not is_montage:

        with st.expander("Additional source information"):

            st.write("**Source event IDs**")
            st.write(
                candidate.get("source_event_ids", [])
            )


def render_ranking_table(candidates, table_key):
    """Render a selectable global ranking table."""

    if not candidates:
        st.info("No promotional candidates are available.")
        return None

    table_rows = []

    for candidate in candidates:

        is_montage = bool(candidate.get("segments"))

        if is_montage:

            table_rows.append({
                "Global Rank": candidate.get("global_rank", "—"),
                "Episode": candidate.get("episode_number", "—"),
                "Candidate": candidate.get("candidate_id", "—"),
                "Segments": len(candidate.get("segments", [])),
                "Duration": (
                    f"{candidate.get('duration_seconds', 0):.2f}s"
                ),
                "Creative Score": candidate.get("weighted_score", 0),
                "Global Score": candidate.get(
                    "global_score",
                    candidate.get("adjusted_score", 0),
                ),
                "Redundancy Penalty": candidate.get(
                    "global_redundancy_penalty",
                    candidate.get("redundancy_penalty", 0),
                ),
            })

        else:

            table_rows.append({
                "Global Rank": candidate.get("global_rank", "—"),
                "Episode": candidate.get("episode_number", "—"),
                "Candidate": candidate.get("candidate_id", "—"),
                "Start": candidate.get("start_timestamp", "—"),
                "End": candidate.get("end_timestamp", "—"),
                "Creative Score": candidate.get("weighted_score", 0),
                "Global Score": candidate.get(
                    "global_score",
                    candidate.get("adjusted_score", 0),
                ),
                "Redundancy Penalty": candidate.get(
                    "global_redundancy_penalty",
                    candidate.get("redundancy_penalty", 0),
                ),
            })

    st.caption(
        "Select a row in the table to view that promo's details below."
    )

    selection = st.dataframe(
        pd.DataFrame(table_rows),
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=table_key,
    )

    selected_rows = selection.selection.rows

    if selected_rows:

        selected_index = selected_rows[0]

        if 0 <= selected_index < len(candidates):
            return candidates[selected_index]

    return None


# ============================================================
# INITIALIZE SESSION AND CSS
# ============================================================

load_css()
load_saved_results()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="brand-header">
        <div class="brand-lockup">
            <div class="brand-name">PROMO FINDER</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MAIN TOP AREA: CONFIGURATION / UPLOAD + RANKING BOARD
# ============================================================

left_panel, right_panel = st.columns(
    [0.36, 0.64],
    gap="large",
)


# ============================================================
# LEFT: MODEL CONFIGURATION AND UPLOAD
# ============================================================

with left_panel:

    st.markdown(
        '<div class="panel-title">Upload your episodes</div>',
        unsafe_allow_html=True,
    )

    discovery_mode = st.radio(
        "DISCOVERY MODE",
        options=[
            "Continuous Clip",
            "Edited Montage",
        ],
        horizontal=True,
        key="discovery_mode",
        help=(
            "**Continuous Clip:** Finds one uninterrupted 20–60 second "
            "portion of an episode that works as a standalone promo.\n\n"
            "**Edited Montage:** Combines non-contiguous moments "
            "from the same episode, arranged in editing order, into "
            "a 20–60 second promotional video."
        ),
    )

    uploaded_files = st.file_uploader(
        "UPLOAD SCREENPLAY FILES",
        type=["txt"],
        accept_multiple_files=True,
        key="screenplay_uploader",
    )

    if uploaded_files:

        st.markdown(
            f"""
            <div class="upload-count">
                {len(uploaded_files)} episode(s) ready to process
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            """
            <div class="upload-hint">
                Add your episode screenplay files (.txt).
            </div>
            """,
            unsafe_allow_html=True,
        )

    provider_options = {
        "Gemini": "gemini",
        "OpenRouter": "openrouter",
    }

    provider_col, model_col = st.columns(
        [0.9, 1.1],
        gap="small",
    )

    with provider_col:

        selected_provider_label = st.selectbox(
            "AI ROUTE",
            options=list(provider_options.keys()),
            key="provider_selection",
        )

    selected_provider = provider_options[selected_provider_label]

    with model_col:

        if selected_provider == "gemini":

            selected_model = st.text_input(
                "MODEL",
                value=GEMINI_MODEL,
                key="gemini_model_selection",
            )

        else:

            selected_model = st.text_input(
                "MODEL",
                value=OPENROUTER_MODEL,
                key="openrouter_model_selection",
            )

    st.markdown(
        """
        <div class="helper-copy">
            Choose the model route for the next processing run.
        </div>
        """,
        unsafe_allow_html=True,
    )

    discover_clicked = st.button(
        "DISCOVER PROMOS",
        type="primary",
        use_container_width=True,
        disabled=not bool(uploaded_files),
    )

    if discover_clicked:

        if not selected_model.strip():

            st.error("Enter a model name before processing.")

        else:

            all_results = []

            selected_discovery_mode = st.session_state["discovery_mode"]

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
                        model=selected_model.strip(),
                        discovery_mode=st.session_state["discovery_mode"],
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
                        "error": str(error),
                    })

                progress_bar.progress(
                    (index + 1) / len(uploaded_files)
                )

            st.session_state["pipeline_results"] = all_results

            st.session_state["run_provider"] = selected_provider
            st.session_state["run_model"] = selected_model.strip()

            st.session_state["run_discovery_mode"] = (
                selected_discovery_mode
            )

            for result in all_results:
                result["discovery_mode"] = selected_discovery_mode

            save_results(all_results)

            status_text.success(
                "Processing complete. Results saved locally."
            )

            st.rerun()


# ============================================================
# LOAD RESULTS FOR DISPLAY
# ============================================================

results = st.session_state.get("pipeline_results", [])

successful, failed = get_successful_and_failed(results)


# ============================================================
# RANKING METHODOLOGY DIALOG
# ============================================================

@st.dialog("HOW PROMO FINDER RANKS", width="large")
def show_ranking_methodology():

    st.markdown("### 1 — Creative evaluation")

    st.write(
        "Every promo is independently evaluated against five "
        "criteria. Each criterion receives a score from 1 to 10."
    )

    criteria = [
        ("Opening hook", "25%", "Captures attention in the first 3 seconds."),
        ("Cliffhanger", "30%", "Creates suspense, curiosity or emotional impact."),
        ("Standalone comprehension", "25%", "Makes sense to viewers unfamiliar with the show."),
        ("Narrative coherence", "10%", "Forms a coherent sequence of events."),
        ("Promotional appeal", "10%", "Encourages viewers to watch the full episode."),
    ]

    for name, weight, description in criteria:

        col1, col2 = st.columns([0.75, 0.25])

        with col1:
            st.markdown(f"**{name}**")
            st.caption(description)

        with col2:
            st.markdown(f"**{weight}**")

    st.divider()

    st.markdown("### 2 — Weighted creative score")

    st.write(
        "The five criterion scores are combined using their "
        "respective weights to produce a creative score out of 10."
    )

    st.latex(
        r"S = 0.25H + 0.30C + 0.25U + 0.10N + 0.10P"
    )

    st.caption(
        "H: Hook | C: Cliffhanger | U: Standalone comprehension "
        "| N: Narrative coherence | P: Promotional appeal"
    )

    st.divider()

    st.markdown("### 3 — Redundancy adjustment")

    st.write(
        "Candidates are compared based on temporal overlap. "
        "The overlap is measured as a proportion of the shorter clip."
    )

    st.latex(
        r"\text{Penalty} = \text{Maximum Overlap} \times 2"
    )

    st.latex(
        r"\text{Adjusted Score} = "
        r"\max(0,\text{Creative Score}-\text{Penalty})"
    )

    st.write(
        "A candidate receives a penalty based on its maximum overlap "
        "with a previously selected candidate from the same episode. "
        "This encourages variety in the final selection."
    )

    st.divider()

    st.markdown("### 4 — Ranking process")

    st.write(
        "Promo Finder uses a greedy selection approach:"
    )

    st.markdown(
        """
        1. Select the candidate with the highest creative score.
        2. Compare remaining candidates against selected candidates.
        3. Recalculate their adjusted scores.
        4. Select the highest adjusted-scoring candidate.
        5. Repeat until all candidates are ranked.
        """
    )

    st.write(
        "For global rankings, redundancy is calculated only between "
        "candidates belonging to the same episode."
    )

    st.divider()

    st.markdown("### 5 — Ranking views")

    st.markdown(
        """
        **All discovered promos**

        All candidates from successful episodes are ranked together,
        with redundancy adjustments applied during global selection.

        **Top promo from each episode**

        The highest adjusted-scoring promo from each episode is
        selected first. These selected promos are then ranked globally.
        """
    )

    st.divider()

    st.markdown("### OUR PUBLISHING RECOMMENDATION")

    st.write(
        "The highest-ranked promo is our recommended first post. "
        "This recommendation is based on its creative score and "
        "redundancy-adjusted ranking, rather than a prediction "
        "of actual audience engagement."
    )


# ============================================================
# RIGHT: RANKING BOARD
# ============================================================

with right_panel:

    title_col, methodology_col, clear_col = st.columns(
        [0.60, 0.19, 0.21],
        gap="small",
    )

    with title_col:

        st.markdown(
            '<div class="panel-title">Your promo shortlist</div>',
            unsafe_allow_html=True,
        )

    with methodology_col:

        if st.button(
            "HOW WE RANK",
            key="show_methodology",
            use_container_width=True,
        ):
            show_ranking_methodology()

    with clear_col:

        if st.button(
            "CLEAR RESULTS",
            key="clear_results",
            use_container_width=True,
        ):

            st.session_state["pipeline_results"] = []

            keys_to_clear = [
                "global_ranking",
                "selected_global_promo",
                "ranking_mode",
                "global_ranking_all",
                "global_ranking_top",
                "episode_explorer_selection",
            ]

            for key in keys_to_clear:
                st.session_state.pop(key, None)

            for key in list(st.session_state.keys()):

                if key.startswith("episode_promos_"):
                    st.session_state.pop(key, None)

                if key.startswith("montage_promos_"):
                    st.session_state.pop(key, None)

            if OUTPUT_FILE.exists():
                OUTPUT_FILE.unlink()

            st.rerun()

    if results:

        total_promos = sum(
            len(result.get("ranked_candidates", []))
            for result in successful
        )

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "EPISODES",
            len(successful),
        )

        metric2.metric(
            "PROMOS FOUND",
            total_promos,
        )

        metric3.metric(
            "FAILED",
            len(failed),
        )

        ranking_mode = st.radio(
            "RANKING VIEW",
            options=[
                "All discovered promos",
                "Top promo from each episode",
            ],
            horizontal=True,
            key="ranking_mode",
        )

        current_mode = st.session_state.get(
            "run_discovery_mode",
            results[0].get("discovery_mode", "Continuous Clip"),
        )

        if current_mode == "Edited Montage":

            if ranking_mode == "All discovered promos":

                global_ranking = rank_all_montages(results)
                table_key = "global_ranking_all"

            else:

                global_ranking = rank_top_montage_per_episode(results)
                table_key = "global_ranking_top"

        else:

            if ranking_mode == "All discovered promos":

                global_ranking = rank_all_episodes(results)
                table_key = "global_ranking_all"

            else:

                global_ranking = rank_top_per_episode(results)
                table_key = "global_ranking_top"

        st.session_state["global_ranking"] = global_ranking

        selected_global_candidate = render_ranking_table(
            global_ranking,
            table_key,
        )

        st.session_state["selected_global_promo"] = (
            selected_global_candidate
        )

    else:

        st.markdown(
            """
            <div class="empty-board">
                <div class="empty-mark">✳</div>
                <h3>Your promo board is waiting.</h3>
                <p>
                    Upload episode screenplays and run discovery.
                    Your ranked promo table will appear here.
                </p>
                <div class="empty-steps">
                    <span>1 &nbsp; Upload episodes</span>
                    <span>2 &nbsp; Choose Gemini or OpenRouter</span>
                    <span>3 &nbsp; Discover and review promos</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# SELECTED GLOBAL PROMO DETAILS
# ============================================================

selected_global_promo = st.session_state.get(
    "selected_global_promo"
)

if selected_global_promo:

    st.markdown(
        '<div class="panel-title">Selected promo details</div>',
        unsafe_allow_html=True,
    )

    # Both continuous clips and montages use the same renderer.
    render_promo_details(selected_global_promo)


# ============================================================
# EPISODE EXPLORER
# ============================================================

if successful:

    st.markdown(
        '<div class="panel-title">Explore individual episodes</div>',
        unsafe_allow_html=True,
    )

    episode_labels = []
    episode_lookup = {}

    for index, result in enumerate(successful):

        episode_number = result.get(
            "episode_number",
            "Unknown",
        )

        filename = result.get(
            "filename",
            f"Episode {episode_number}",
        )

        label = f"EPISODE {episode_number} / {filename}"

        episode_labels.append(label)
        episode_lookup[label] = index

    selected_episode_label = st.selectbox(
        "CHOOSE AN EPISODE",
        options=episode_labels,
        key="episode_explorer_selection",
    )

    selected_episode = successful[
        episode_lookup[selected_episode_label]
    ]

    episode_mode = selected_episode.get(
        "discovery_mode",
        "Continuous Clip",
    )

    episode_candidates = selected_episode.get(
        "ranked_candidates",
        [],
    )

    # ========================================================
    # EDITED MONTAGE DISPLAY
    # ========================================================

    if episode_mode == "Edited Montage":

        if episode_candidates:

            st.caption(
                f"{len(episode_candidates)} ranked montage(s) discovered."
            )

            montage_table_rows = []

            for candidate in episode_candidates:

                montage_table_rows.append({
                    "Rank": candidate.get("rank", "—"),
                    "Candidate": candidate.get("candidate_id", "—"),
                    "Segments": len(candidate.get("segments", [])),
                    "Duration": (
                        f"{candidate.get('duration_seconds', 0):.2f}s"
                    ),
                    "Creative Score": candidate.get("weighted_score", 0),
                    "Adjusted Score": candidate.get("adjusted_score", 0),
                    "Redundancy Penalty": candidate.get(
                        "redundancy_penalty",
                        0,
                    ),
                })

            st.caption(
                "Select a montage to inspect its segments and evaluation."
            )

            montage_selection = st.dataframe(
                pd.DataFrame(montage_table_rows),
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                key=f"montage_promos_{selected_episode_label}",
            )

            selected_montage_rows = montage_selection.selection.rows

            if selected_montage_rows:

                selected_montage = episode_candidates[
                    selected_montage_rows[0]
                ]

                st.divider()

                # Same detail renderer used by the global shortlist.
                render_promo_details(selected_montage)

        else:

            st.info(
                "No ranked montages are available for this episode."
            )

    # ========================================================
    # CONTINUOUS CLIP DISPLAY
    # ========================================================

    else:

        if episode_candidates:

            episode_table_rows = []

            for candidate in episode_candidates:

                episode_table_rows.append({
                    "Rank": candidate.get("rank", "—"),
                    "Candidate": candidate.get("candidate_id", "—"),
                    "Start": candidate.get("start_timestamp", "—"),
                    "End": candidate.get("end_timestamp", "—"),
                    "Creative Score": candidate.get("weighted_score", 0),
                    "Adjusted Score": candidate.get("adjusted_score", 0),
                    "Redundancy Penalty": candidate.get(
                        "redundancy_penalty",
                        0,
                    ),
                })

            st.caption(
                "Select a promo row to view its details below."
            )

            episode_selection = st.dataframe(
                pd.DataFrame(episode_table_rows),
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                key=f"episode_promos_{selected_episode_label}",
            )

            selected_episode_rows = episode_selection.selection.rows

            if selected_episode_rows:

                selected_episode_candidate = episode_candidates[
                    selected_episode_rows[0]
                ]

                render_promo_details(selected_episode_candidate)

        else:

            st.info(
                "No valid promos were discovered for this episode."
            )

    # ========================================================
    # TECHNICAL OUTPUTS
    # ========================================================

    with st.expander("Raw LLM data and technical details"):

        raw_outputs = selected_episode.get(
            "raw_outputs",
            {},
        )

        st.markdown("**Discovery response**")

        st.text_area(
            "Raw discovery output",
            value=raw_outputs.get("discovery") or "Not available",
            height=180,
            disabled=True,
            key=f"raw_discovery_{selected_episode_label}",
        )

        st.markdown("**Evaluation response**")

        st.text_area(
            "Raw evaluation output",
            value=raw_outputs.get("evaluation") or "Not available",
            height=180,
            disabled=True,
            key=f"raw_evaluation_{selected_episode_label}",
        )

        st.markdown("**Structural validation**")

        st.json(
            selected_episode.get("validation", {})
        )

        st.download_button(
            "Download this episode's JSON",
            data=json.dumps(
                selected_episode,
                indent=2,
                ensure_ascii=False,
            ),
            file_name=(
                f"{selected_episode.get('filename', 'episode')}"
                .rsplit(".", 1)[0]
                + "_results.json"
            ),
            mime="application/json",
            key=f"download_{selected_episode_label}",
        )


# ============================================================
# FAILED EPISODES
# ============================================================

if failed:

    with st.expander(
        f"Processing issues ({len(failed)})"
    ):

        for result in failed:

            st.error(
                f"{result.get('filename', 'Unknown episode')}: "
                f"{result.get('error', 'Processing failed.')}"
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="site-footer">
        <div class="footer-brand">
            <div class="footer-name">PROMO FINDER</div>
            <div class="footer-credit">BUILT BY Sanjay</div>
        </div>
        <div class="footer-right">
            <div class="footer-links">
                <a href="https://github.com/sanjaydevs/kuku-promo-discovery"
                   target="_blank"
                   rel="noreferrer">GITHUB ↗</a>
                <a href="https://www.linkedin.com/in/sanjay-s-74551a2ba/"
                   target="_blank"
                   rel="noreferrer">LINKEDIN ↗</a>
            </div>
            <div class="footer-caption">
                © 2026 Promo finder
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
