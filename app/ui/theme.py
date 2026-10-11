"""Layout shared by every page: styles, banner and footer."""

from pathlib import Path

import streamlit as st

from voiceline_qa.config import ASR_ID, CLASSIFIER_ID, TRANSLATOR_ID

STYLESHEET = Path(__file__).with_name("styles.css")
# Images in static/ are served by Streamlit (server.enableStaticServing) and cached by the browser
BANNER_URL = "app/static/banner.webp"   # agent line-up composed from official agent portraits
GITHUB_URL = "https://github.com/EthanIsaac/valorant-voiceline-qa"


def apply_styles() -> None:
    st.markdown(f"<style>{STYLESHEET.read_text()}</style>", unsafe_allow_html=True)


def banner() -> None:
    st.markdown(
        f'<div class="vq-hero" style="background-image: linear-gradient(90deg, #0F1923e6 0%, #0F192366 40%, '
        f'#0F192300 70%), linear-gradient(0deg, #0F1923 0%, #0F192300 60%), url({BANNER_URL});">'
        '<span class="vq-tag">Unofficial · educational project</span>'
        '<div class="vq-kicker">VALORANT · audio &amp; localization QA</div>'
        '<div class="vq-title">Voice-line <span>QA</span></div></div>', unsafe_allow_html=True)


def footer(other_page: st.Page, other_page_label: str) -> None:
    """Link to the other workflow, and Riot's required notice."""
    st.divider()
    st.page_link(other_page, label=other_page_label, icon=other_page.icon)
    st.markdown(
        '<div class="vq-footer">Voice-Line QA was created under Riot Games\' "Legal Jibber Jabber" policy using '
        "assets owned by Riot Games. Riot Games does not endorse or sponsor this project.<br>"
        "Non-commercial course project for HKUST ISOM5240; not an official Riot Games website or tool. "
        "Voice lines from the NgThVinh/ValorantAgentVoiceLines dataset; agent art via valorant-api.com. "
        f'Models: <a href="https://huggingface.co/{CLASSIFIER_ID}">agent classifier</a>, '
        f'<a href="https://huggingface.co/{ASR_ID}">speech-to-text</a>, '
        f'<a href="https://huggingface.co/{TRANSLATOR_ID}">translation</a>. '
        f'<a href="{GITHUB_URL}">Source code</a>.</div>', unsafe_allow_html=True)
