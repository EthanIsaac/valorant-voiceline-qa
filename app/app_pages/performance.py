"""Performance test: accuracy of the app on held-out test clips (report, section 9.4)."""

import streamlit as st

from ui.components import loading_bar
from ui.resources import TEST_CLIPS_DIR, get_voiceline_qa, load_test_clips
from voiceline_qa.agents import AGENTS
from voiceline_qa.evaluation import (EvaluationClip, PerformanceRun, agent_accuracy_by_agent, run_performance_test,
                                     script_accuracy_by_subset, summarize_accuracy, summary_table)

RUN_KEY = "performance_run"


def run_with_progress(clips: list[EvaluationClip]) -> PerformanceRun:
    """Runs the test behind a game-style bar with one segment per agent."""
    progress, agents = st.empty(), len({clip.agent for clip in clips})
    loading_bar(progress, "Loading models (first run only, about a minute)", f"0/{len(clips)}", 0, agents)
    qa = get_voiceline_qa()

    def show_progress(index: int, clip: EvaluationClip) -> None:
        loading_bar(progress, f"Testing {AGENTS[clip.agent].name} · {clip.file}", f"{index + 1}/{len(clips)}",
                    done_segments=index * agents // len(clips), total_segments=agents)

    run = run_performance_test(qa, clips, TEST_CLIPS_DIR, on_clip_start=show_progress)
    progress.empty()
    return run


def show_accuracy(run: PerformanceRun) -> None:
    accuracy = summarize_accuracy(run.results)
    st.subheader("Accuracy")
    with st.container(horizontal=True):
        st.metric("Agent accuracy", f"{accuracy.agent_accuracy:.1%}", border=True,
                  help="Clips whose predicted agent is the agent who recorded them")
        st.metric("Script exact match", f"{accuracy.script_exact_match:.1%}", border=True,
                  help="Clips whose transcript matches the script line word for word (after normalisation)")
        st.metric("Word error rate", f"{accuracy.word_error_rate:.3f}", border=True,
                  help="Word errors divided by script words, over all clips (lower is better)")
        st.metric("Names and callouts kept", f"{accuracy.protected_terms_kept:.1%}", border=True,
                  help=f"Spanish drafts keeping every agent name and site letter, over the "
                       f"{accuracy.clips_with_protected_terms} clips whose line contains any")

    st.markdown("**Script check on lines never heard in training vs. lines whose text was**")
    st.dataframe(script_accuracy_by_subset(run.results),
                 column_config={"clips": "Clips",
                                "exact_match": st.column_config.NumberColumn("Exact match", format="percent"),
                                "word_error_rate": st.column_config.NumberColumn("Word error rate", format="%.3f")})
    st.markdown("**Agent accuracy by agent**")
    st.dataframe(agent_accuracy_by_agent(run.results), height=260,
                 column_config={"clips": "Clips", "correct": "Correct",
                                "accuracy": st.column_config.ProgressColumn("Accuracy", format="percent",
                                                                            min_value=0, max_value=1)})


def show_downloads(run: PerformanceRun) -> None:
    with st.expander("All results"):
        st.dataframe(run.results, hide_index=True,
                     column_config={"agent_probability": st.column_config.NumberColumn(format="percent")})
    stamp = run.started_utc.replace(":", "").replace("-", "")[:13]
    with st.container(horizontal=True):
        st.download_button("Results per clip (CSV)", run.results.to_csv(index=False).encode(),
                           file_name=f"app_performance_clips_{stamp}.csv", mime="text/csv", icon=":material/download:")
        st.download_button("Summary (CSV)", summary_table(run).to_csv(index=False).encode(),
                           file_name=f"app_performance_summary_{stamp}.csv", mime="text/csv",
                           icon=":material/download:")


clips = load_test_clips()
agent_count = len({clip.agent for clip in clips})
st.header("Performance test")
st.markdown(
    f"Measures the **accuracy** of this app on **{len(clips)} held-out test clips**: "
    f"{len(clips) // agent_count} per agent for all {agent_count} agents, drawn with a fixed seed from the test "
    "split, so no clip was used to train or select a model. Each clip runs through the same code as a QA check "
    "(decode the audio, identify the agent, transcribe and compare with its script line, translate). "
    "Accuracy does not depend on the machine, so the test can run locally or on Streamlit Cloud.")
st.caption(f"{sum(clip.duration_s for clip in clips):.0f} s of audio. Keep this tab open until the test finishes.")

if st.button("Run performance test", type="primary", icon=":material/speed:"):
    st.session_state[RUN_KEY] = run_with_progress(clips)

if RUN_KEY in st.session_state:
    performance_run = st.session_state[RUN_KEY]
    show_accuracy(performance_run)
    show_downloads(performance_run)
