"""QA check: the tester's workflow on one recording."""

import streamlit as st

from ui.components import StepProgress, agent_card, agent_strip, author_button, script_card, subtitle_card
from ui.resources import TEST_CLIPS_DIR, get_voiceline_qa, load_test_clips
from voiceline_qa.agents import AGENTS
from voiceline_qa.audio import InvalidRecordingError, load_recording

SECTION_TITLES = ["1 · Who is speaking?", "2 · Does it match the script?", "3 · Spanish subtitle draft"]
STEP_NAMES = ["Identifying agent", "Transcribing voice line", "Translating to Spanish"]


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

# ---------------------------------------------------------------- input

source = st.radio("Voice line", ["Test clip", "Upload a recording"], horizontal=True)
if source == "Test clip":
    clips = load_test_clips()
    clip = st.selectbox("Test clip (held-out, never used in training; type to search)", clips,
                        format_func=lambda c: f'{AGENTS[c.agent].name}: "{c.script_line}"')
    audio_bytes, audio_format = (TEST_CLIPS_DIR / "clips" / clip.file).read_bytes(), "audio/flac"
    expected_agent, default_script_line = clip.agent, clip.script_line
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
