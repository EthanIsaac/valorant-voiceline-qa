"""QA check: the tester's workflow on one recording."""

from pathlib import Path

import pandas as pd
import streamlit as st

from ui.components import (StepProgress, agent_card, agent_strip, author_button, script_card,
                           show_author_once, subtitle_card)
from ui.resources import get_voiceline_qa
from voiceline_qa.agents import AGENTS
from voiceline_qa.audio import InvalidRecordingError, load_recording

SAMPLES_DIR = Path(__file__).parents[1] / "samples"
SECTION_TITLES = ["1 · Who is speaking?", "2 · Does it match the script?", "3 · Spanish subtitle draft"]
STEP_NAMES = ["Identifying agent", "Transcribing voice line", "Translating to Spanish"]


@st.cache_data
def load_samples() -> pd.DataFrame:
    return pd.read_csv(SAMPLES_DIR / "samples.csv")


st.markdown(
    "<p>Checks a VALORANT voice-line recording before release. Select a sample or upload a recording "
    "to run three checks:</p>"
    '<ol class="vq-steps">'
    "<li><b>Right agent:</b> identifies which of the 26 agents is speaking, catching recordings filed under "
    "the wrong agent.</li>"
    "<li><b>Matches the script:</b> transcribes the recording and compares it with the script line word by word, "
    "flagging missing, extra or misheard words.</li>"
    "<li><b>Spanish subtitle draft:</b> translates the line for localization and flags agent names or callouts "
    "(A, B, C site) lost in translation.</li></ol>", unsafe_allow_html=True)
agent_strip()
author_button()
show_author_once()

# ---------------------------------------------------------------- input

source = st.radio("Voice line", ["Sample from the test set", "Upload a recording"], horizontal=True)
if source == "Sample from the test set":
    samples = load_samples()
    labels = [f'{AGENTS[sample.agent].name}: "{sample.script_line}"' for sample in samples.itertuples()]
    sample = samples.iloc[labels.index(st.selectbox("Sample voice line (held-out test set, not used in training)",
                                                    labels))]
    audio_bytes, audio_format = (SAMPLES_DIR / sample.file).read_bytes(), "audio/flac"
    expected_agent, default_script_line = sample.agent, sample.script_line
else:
    upload = st.file_uploader("Recording (WAV, FLAC, OGG or MP3, up to 30 s)", type=["wav", "flac", "ogg", "mp3"])
    audio_bytes, audio_format = (upload.getvalue(), upload.type) if upload else (None, None)
    expected_agent, default_script_line = None, ""

script_line = st.text_input("Script line (what the agent should say)", value=default_script_line,
                            help="Optional. Leave empty to only transcribe and translate.")
if audio_bytes:
    st.audio(audio_bytes, format=audio_format)

# ---------------------------------------------------------------- checks

if st.button("Run QA check", type="primary", disabled=not audio_bytes):
    try:
        recording = load_recording(audio_bytes)
    except InvalidRecordingError as error:
        st.error(str(error))
    else:
        progress = StepProgress(SECTION_TITLES, STEP_NAMES)

        progress.start(0, "Loading models (first run only, about a minute)")
        qa = get_voiceline_qa()
        progress.start(0)
        agent = qa.identify_agent(recording)
        agent_card(progress.slots[0], agent, expected_agent)

        progress.start(1)
        script = qa.check_script(recording, script_line)
        script_card(progress.slots[1], script)

        progress.start(2)
        subtitle = qa.draft_subtitle(script)
        subtitle_card(progress.slots[2], subtitle)

        st.caption(f"Processing time on CPU: agent {agent.seconds:.1f} s · transcript {script.seconds:.1f} s · "
                   f"Spanish {subtitle.seconds:.1f} s · clip length {recording.duration_s:.1f} s")
