"""Resources shared by all pages and visitors."""

import streamlit as st

from voiceline_qa.service import VoiceLineQA


@st.cache_resource(show_spinner=False)
def get_voiceline_qa() -> VoiceLineQA:
    """The three models, loaded once per server (about a minute on the first call)."""
    return VoiceLineQA.from_pretrained()
