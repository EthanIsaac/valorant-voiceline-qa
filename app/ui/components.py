"""Visual components: author dialog, game-style step progress and the three result cards."""

import html

import streamlit as st

from voiceline_qa.agents import AGENTS
from voiceline_qa.checks import AgentCheck, ScriptCheck, SubtitleDraft

PORTRAIT_URL = "app/static/student_portrait.webp"
AUTHOR_BUTTON_KEY = "author_button"


# ---------------------------------------------------------------- author

@st.dialog("Author", width="medium")
def author_dialog() -> None:
    st.markdown(
        f'<div class="vq-dossier"><img src="{PORTRAIT_URL}" alt="Ethan Isaac Bautista Trevizo">'
        '<div class="info"><div class="vq-kicker">Student dossier</div>'
        '<div class="name">Ethan Isaac<span>Bautista Trevizo</span></div>'
        '<div class="role">Role · Sentinel <small>(guards voice-line quality)</small></div></div></div>'
        '<dl class="vq-facts">'
        "<dt>Course</dt><dd>ISOM5240 · The Hong Kong University of Science and Technology (HKUST)</dd>"
        "<dt>Project</dt><dd>VALORANT Voice-Line QA &amp; Localization Assistant: checks that a voice-line "
        "recording comes from the right agent and matches its script word for word, and drafts its Spanish "
        "subtitle.</dd>"
        "<dt>Company case</dt><dd>Riot Games (case study; no affiliation with Riot Games)</dd>"
        '<dt>Links</dt><dd><a href="https://github.com/EthanIsaac">GitHub</a> · '
        '<a href="https://huggingface.co/eibautistat">Hugging Face models</a></dd></dl>', unsafe_allow_html=True)
    if st.button("Start QA", type="primary", icon=":material/play_arrow:", width="stretch"):
        st.rerun()


def author_button() -> None:
    """Opens the author dialog; glows until it is clicked for the first time."""
    if st.session_state.get(AUTHOR_BUTTON_KEY):
        st.session_state.author_button_clicked = True
    if not st.session_state.get("author_button_clicked"):
        st.markdown(f"<style>.st-key-{AUTHOR_BUTTON_KEY} button {{ animation: vq-glow 1.6s ease-in-out infinite; }}"
                    f"@media (prefers-reduced-motion: reduce) {{ .st-key-{AUTHOR_BUTTON_KEY} button {{ animation: none;"
                    " box-shadow: 0 0 14px 3px #FF465599; } }</style>", unsafe_allow_html=True)
    if st.button("Author · Ethan Bautista", icon=":material/badge:", type="primary", key=AUTHOR_BUTTON_KEY):
        author_dialog()


def agent_strip() -> None:
    portraits = "".join(f'<img src="{agent.image_url()}" alt="{agent.name}" title="{agent.name}" loading="lazy">'
                        for agent in AGENTS.values())
    st.markdown(f'<div class="vq-strip">{portraits}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- progress

def loading_bar(slot, label: str, counter: str, done_segments: int, total_segments: int) -> None:
    """Game-style loading bar: slanted segments, finished ones full, the next one animated."""
    segments = "".join(
        f'<div class="seg {"done" if i < done_segments else "active" if i == done_segments else ""}"></div>'
        for i in range(total_segments))
    slot.markdown(
        f'<div class="vq-load{" many" if total_segments > 6 else ""}"><div class="top"><span>{html.escape(label)}'
        f'<span class="dots"></span></span>'
        f'<span>{counter}</span></div><div class="segs">{segments}</div></div>', unsafe_allow_html=True)


class StepProgress:
    """One section per step; the running step shows the loading bar, later steps wait in the queue."""

    def __init__(self, section_titles: list[str], step_names: list[str]):
        self._step_names = step_names
        self.slots = []
        for title in section_titles:
            st.markdown(f"### {title}")
            self.slots.append(st.empty())
        for step in range(len(step_names)):
            self._show_queued(step)

    def start(self, step: int, label: str | None = None) -> None:
        loading_bar(self.slots[step], label or self._step_names[step], f"{step + 1}/{len(self._step_names)}",
                    done_segments=step, total_segments=len(self._step_names))

    def _show_queued(self, step: int) -> None:
        self.slots[step].markdown(
            f'<div class="vq-load queued"><div class="top"><span>Queued: {self._step_names[step].lower()}</span>'
            f'<span>{step + 1}/{len(self._step_names)}</span></div></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- result cards

def _badge(passed: bool, text: str) -> str:
    return f'<span class="vq-badge {"vq-ok" if passed else "vq-bad"}">{html.escape(text)}</span>'


def agent_card(slot, check: AgentCheck, expected_label: str | None) -> None:
    predicted = check.predicted
    verdict = ""
    if expected_label:
        verdict = (_badge(True, "Matches the file's agent") if check.matches(expected_label)
                   else _badge(False, f"File says {AGENTS[expected_label].name}"))
    rows = "".join(
        f'<div class="vq-row"><img src="{c.agent.image_url()}" alt=""><span class="label">{c.agent.name}</span>'
        f'<div class="bar"><div style="width:{c.probability:.0%}"></div></div>'
        f'<span class="pct">{c.probability:.0%}</span></div>' for c in check.candidates)
    slot.markdown(
        f'<div class="vq-card" style="background: linear-gradient(90deg, #{predicted.agent.colour}cc, #1F2731 70%);">'
        f'<div class="vq-agent"><img class="portrait" src="{predicted.agent.image_url("fullportrait")}" '
        f'alt="{predicted.agent.name}"><div><div class="name">{predicted.agent.name}</div>'
        f'<div>{predicted.probability:.0%} confidence</div>{verdict}</div></div></div>'
        f'<div class="vq-card"><b>Top {len(check.candidates)}</b>{rows}</div>', unsafe_allow_html=True)


def script_card(slot, check: ScriptCheck) -> None:
    body = f"<b>Heard:</b> {html.escape(check.transcript)}"
    if check.has_script:
        plural = "s" if check.word_errors > 1 else ""
        badge = (_badge(True, "Matches the script") if check.matches_script
                 else _badge(False, f"{check.word_errors} word difference{plural}: listen again"))
        body = f"{badge}<p style='margin-top:.6rem'>{body}</p>"
        if not check.matches_script:
            body += (f"<div class='vq-diff'>{_alignment_html(check)}</div><small>Compared word by word, ignoring "
                     "case, punctuation and spelling variants: <del>in the script</del> → <ins>heard instead</ins>."
                     "</small>")
    slot.markdown(f'<div class="vq-card">{body}</div>', unsafe_allow_html=True)


def _alignment_html(check: ScriptCheck) -> str:
    spans = []
    for difference in check.alignment:
        script, heard = html.escape(difference.script_words), html.escape(difference.heard_words)
        spans.append({"equal": script, "substitute": f"<del>{script}</del> <ins>{heard}</ins>",
                      "delete": f"<del>{script}</del>", "insert": f"<ins>{heard}</ins>"}[difference.kind])
    return " ".join(spans)


def subtitle_card(slot, draft: SubtitleDraft) -> None:
    check = (_badge(True, "Names and callouts kept") if draft.keeps_protected_terms
             else _badge(False, f"Check: {', '.join(draft.lost_terms)} changed or missing"))
    slot.markdown(
        f'<div class="vq-card">{check}<p class="vq-es" style="margin-top:.6rem">{html.escape(draft.spanish_line)}</p>'
        f"<small>Translated from the {draft.translated_from}, one sentence at a time. "
        "A machine draft for a human translator, not a final subtitle.</small></div>", unsafe_allow_html=True)
