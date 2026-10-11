"""Resources shared by all pages and visitors."""

from pathlib import Path

import streamlit as st

from voiceline_qa.evaluation import EvaluationClip, load_evaluation_set
from voiceline_qa.service import VoiceLineQA

# 104 held-out test clips (4 per agent): offered in the QA check and used by the performance test
TEST_CLIPS_DIR = Path(__file__).parents[1] / "test_clips"


@st.cache_resource(show_spinner=False)
def get_voiceline_qa() -> VoiceLineQA:
    """The three models, loaded once per server (about a minute on the first call)."""
    return VoiceLineQA.from_pretrained()


@st.cache_data
def load_test_clips() -> list[EvaluationClip]:
    return load_evaluation_set(TEST_CLIPS_DIR)
