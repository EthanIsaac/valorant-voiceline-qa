"""Performance of the deployed app on a fixed sample of held-out test clips (report §9.4): accuracy and runtime.

The sample (test_clips/manifest.csv) holds 4 clips per agent drawn with a fixed seed from the English test
split of notebook 01; none was used to train or select a model. Each clip goes through the same code path as a
tester's check (decode the audio, then VoiceLineQA.run with the clip's script line), and every stage is timed.
"""

import os
import platform
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from voiceline_qa.agents import AGENTS
from voiceline_qa.audio import load_recording
from voiceline_qa.service import VoiceLineQA

STAGE_COLUMNS = {"seconds_audio": "Decode audio", "seconds_agent": "Identify agent",
                 "seconds_script": "Transcribe and compare", "seconds_subtitle": "Translate"}


@dataclass(frozen=True)
class EvaluationClip:
    file: str
    agent: str              # label of the agent who recorded the clip
    script_line: str
    duration_s: float
    seen_in_train: bool     # the line's text also occurs in the speech-to-text training data


@dataclass(frozen=True)
class AccuracySummary:
    clips: int
    agent_accuracy: float
    script_exact_match: float        # share of clips whose transcript matches the script line
    word_error_rate: float           # word errors / script words, over all clips
    clips_with_protected_terms: int
    protected_terms_kept: float      # share of those clips whose subtitle keeps every name and callout


@dataclass(frozen=True)
class RuntimeSummary:
    mean_seconds_per_clip: float
    median_seconds_per_clip: float
    p95_seconds_per_clip: float
    max_seconds_per_clip: float
    real_time_factor: float          # processing time / audio duration (below 1: faster than real time)


@dataclass(frozen=True)
class PerformanceRun:
    results: pd.DataFrame       # one row per clip (see evaluate_clip)
    started_utc: str
    total_seconds: float        # wall-clock duration of the whole test
    model_load_seconds: float   # time the server took to load the models
    environment: dict[str, str]


def load_evaluation_set(directory: Path) -> list[EvaluationClip]:
    manifest = pd.read_csv(directory / "manifest.csv")
    return [EvaluationClip(row.file, row.agent, row.script_line, row.duration_s, bool(row.seen_in_train))
            for row in manifest.itertuples()]


def evaluate_clip(qa: VoiceLineQA, clip: EvaluationClip, directory: Path) -> dict:
    """Runs one clip through the app's checks and returns one result row, with the time of every stage."""
    start = time.perf_counter()
    recording = load_recording((directory / "clips" / clip.file).read_bytes())
    seconds_audio = time.perf_counter() - start
    report = qa.run(recording, clip.script_line)
    return {
        **asdict(clip),
        "predicted_agent": report.agent.predicted.agent.label,
        "agent_probability": round(report.agent.predicted.probability, 4),
        "agent_correct": report.agent.matches(clip.agent),
        "transcript": report.script.transcript,
        "word_errors": report.script.word_errors,
        "script_words": report.script.script_word_count,
        "script_exact_match": report.script.matches_script,
        "spanish_line": report.subtitle.spanish_line,
        "protected_terms": ", ".join(report.subtitle.protected_terms),
        "lost_terms": ", ".join(report.subtitle.lost_terms),
        "seconds_audio": round(seconds_audio, 4),
        "seconds_agent": round(report.agent.seconds, 4),
        "seconds_script": round(report.script.seconds, 4),
        "seconds_subtitle": round(report.subtitle.seconds, 4),
    }


def run_performance_test(qa: VoiceLineQA, clips: list[EvaluationClip], directory: Path,
                         on_clip_start: Callable[[int, EvaluationClip], None] | None = None) -> PerformanceRun:
    """Evaluates every clip in order; `on_clip_start(index, clip)` lets the caller report progress."""
    started_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    start = time.perf_counter()
    rows = []
    for index, clip in enumerate(clips):
        if on_clip_start:
            on_clip_start(index, clip)
        rows.append(evaluate_clip(qa, clip, directory))
    return PerformanceRun(pd.DataFrame(rows), started_utc, time.perf_counter() - start, qa.load_seconds,
                          environment())


def summary_table(run: PerformanceRun) -> pd.DataFrame:
    """Every headline number of a run as metric/value rows, for the report and the experiments Excel."""
    accuracy, runtime = summarize_accuracy(run.results), summarize_runtime(run.results)
    rows = {**{f"accuracy.{key}": value for key, value in asdict(accuracy).items()},
            **{f"runtime.{key}": value for key, value in asdict(runtime).items()},
            "runtime.model_load_seconds": run.model_load_seconds,
            "runtime.total_test_seconds": run.total_seconds,
            "run.started_utc": run.started_utc,
            **{f"environment.{key}": value for key, value in run.environment.items()}}
    return pd.DataFrame({"metric": rows.keys(), "value": rows.values()})


# ---------------------------------------------------------------- accuracy

def summarize_accuracy(results: pd.DataFrame) -> AccuracySummary:
    with_terms = results[results["protected_terms"] != ""]
    return AccuracySummary(
        clips=len(results),
        agent_accuracy=float(results["agent_correct"].mean()),
        script_exact_match=float(results["script_exact_match"].mean()),
        word_error_rate=float(results["word_errors"].sum() / results["script_words"].sum()),
        clips_with_protected_terms=len(with_terms),
        protected_terms_kept=float((with_terms["lost_terms"] == "").mean()) if len(with_terms) else float("nan"),
    )


def script_accuracy_by_subset(results: pd.DataFrame) -> pd.DataFrame:
    """Script check on lines never heard in training vs. lines whose text was (report §9.2)."""
    subsets = results.assign(subset=results["seen_in_train"].map({False: "Unseen lines", True: "Seen lines"}))
    table = subsets.groupby("subset").agg(clips=("file", "size"), exact_match=("script_exact_match", "mean"),
                                          word_errors=("word_errors", "sum"), script_words=("script_words", "sum"))
    table["word_error_rate"] = table["word_errors"] / table["script_words"]
    table = table[["clips", "exact_match", "word_error_rate"]].reindex(["Unseen lines", "Seen lines"]).dropna(how="all")
    return table.rename_axis("Lines")


def agent_accuracy_by_agent(results: pd.DataFrame) -> pd.DataFrame:
    table = results.groupby("agent").agg(clips=("file", "size"), correct=("agent_correct", "sum"))
    table["accuracy"] = table["correct"] / table["clips"]
    table = table.sort_values(["accuracy", "agent"])
    return table.set_axis([AGENTS[label].name for label in table.index]).rename_axis("Agent")


# ---------------------------------------------------------------- runtime

def seconds_per_clip(results: pd.DataFrame) -> pd.Series:
    return results[list(STAGE_COLUMNS)].sum(axis=1)


def summarize_runtime(results: pd.DataFrame) -> RuntimeSummary:
    per_clip = seconds_per_clip(results)
    return RuntimeSummary(
        mean_seconds_per_clip=float(per_clip.mean()),
        median_seconds_per_clip=float(per_clip.median()),
        p95_seconds_per_clip=float(per_clip.quantile(0.95)),
        max_seconds_per_clip=float(per_clip.max()),
        real_time_factor=float(per_clip.sum() / results["duration_s"].sum()),
    )


def runtime_by_stage(results: pd.DataFrame) -> pd.DataFrame:
    stages = results[list(STAGE_COLUMNS)].rename(columns=STAGE_COLUMNS).assign(**{"Whole check": seconds_per_clip(results)})
    return pd.DataFrame({"mean": stages.mean(), "median": stages.median(), "p95": stages.quantile(0.95),
                         "share": stages.mean() / stages["Whole check"].mean()}).rename_axis("Stage")


def environment() -> dict[str, str]:
    """Where the test ran, for the report."""
    return {"platform": platform.platform(), "python": platform.python_version(), "cpu_count": str(os.cpu_count()),
            "torch": version("torch"), "transformers": version("transformers"), "streamlit": version("streamlit")}
