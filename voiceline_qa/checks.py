"""Results of the three QA checks. Plain data: the app renders them, the performance test scores them."""

from dataclasses import dataclass
from typing import Literal

from voiceline_qa.agents import AGENTS, Agent


@dataclass(frozen=True)
class AgentCandidate:
    agent: Agent
    probability: float


@dataclass(frozen=True)
class AgentCheck:
    """Check 1: which agent is speaking."""
    candidates: list[AgentCandidate]   # most probable first
    seconds: float

    @property
    def predicted(self) -> AgentCandidate:
        return self.candidates[0]

    def matches(self, expected_label: str) -> bool:
        return self.predicted.agent.label == expected_label


@dataclass(frozen=True)
class WordDifference:
    """One aligned span of the script and the transcript, after normalisation."""
    kind: Literal["equal", "substitute", "delete", "insert"]
    script_words: str   # empty for an insertion
    heard_words: str    # empty for a deletion


@dataclass(frozen=True)
class ScriptCheck:
    """Check 2: does the recording say the script line."""
    transcript: str
    script_line: str                       # empty when the tester gave none
    alignment: list[WordDifference]        # empty when there is no script line
    word_errors: int                       # substitutions + deletions + insertions
    script_word_count: int
    seconds: float

    @property
    def has_script(self) -> bool:
        return bool(self.script_line)

    @property
    def matches_script(self) -> bool:
        return self.has_script and self.word_errors == 0


@dataclass(frozen=True)
class SubtitleDraft:
    """Check 3: Spanish subtitle draft and the names or callouts it lost."""
    english_line: str
    translated_from: Literal["script line", "transcript"]
    spanish_line: str
    protected_terms: list[str]
    lost_terms: list[str]
    seconds: float

    @property
    def keeps_protected_terms(self) -> bool:
        return not self.lost_terms


@dataclass(frozen=True)
class QAReport:
    agent: AgentCheck
    script: ScriptCheck
    subtitle: SubtitleDraft

    @property
    def seconds(self) -> float:
        return self.agent.seconds + self.script.seconds + self.subtitle.seconds


def agent_candidates(classifier_output: list[dict]) -> list[AgentCandidate]:
    return [AgentCandidate(AGENTS[item["label"]], item["score"]) for item in classifier_output]
