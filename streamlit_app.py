# VALORANT Voice-Line QA & Localization Assistant (HKUST ISOM5240 project, educational only)
#
# One recorded voice line in, three checks out:
#   1. Which agent is speaking?          eibautistat/valorant-agent-voice-classifier (fine-tuned wav2vec2-base)
#   2. Does the recording match the script?  eibautistat/valorant-whisper-base-en (fine-tuned whisper-base.en)
#   3. A draft Spanish subtitle           Helsinki-NLP/opus-mt-en-es
# Models and settings follow notebooks 01-04; see the report for how each was selected.

import html
import io
import re
import time
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd
import soundfile as sf
import soxr
import streamlit as st
from transformers import pipeline
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

CLASSIFIER_ID = "eibautistat/valorant-agent-voice-classifier"
ASR_ID = "eibautistat/valorant-whisper-base-en"
TRANSLATOR_ID = "Helsinki-NLP/opus-mt-en-es"

TARGET_SR = 16_000            # all three speech models were trained on 16 kHz audio
CLASSIFIER_MAX_SECONDS = 4.0  # the classifier was trained on the first 4 s of each clip (notebook 01)
MAX_UPLOAD_SECONDS = 30       # voice lines are a few seconds; Whisper reads at most 30 s at a time
TRANSLATION_SETTINGS = {"num_beams": 4, "max_new_tokens": 128}   # as evaluated in notebook 04

SAMPLES_DIR = Path(__file__).parent / "samples"
GITHUB_URL = "https://github.com/EthanIsaac/valorant-voiceline-qa"

# Official in-game agent art, served by the community asset CDN (valorant-api.com): label -> (name, asset id, colour)
AGENTS = {
    "astra": ("Astra", "41fb69c1-4189-7b37-f117-bcaf1e96f1bf", "26146c"),
    "breach": ("Breach", "5f8d3a7f-467b-97f3-062c-13acf203c006", "81331a"),
    "brimstone": ("Brimstone", "9f0d8ba9-4140-b941-57d3-a7ad57c6b417", "363c4f"),
    "chamber": ("Chamber", "22697a3d-45bf-8dd7-4fec-84a9e28c69d7", "20435b"),
    "clove": ("Clove", "1dbf2edd-4729-0984-3115-daa5eed44993", "4b1d80"),
    "cypher": ("Cypher", "117ed9e3-49f3-6512-3ccf-0cada7e3823b", "2f5078"),
    "deadlock": ("Deadlock", "cc8b64c8-4b25-4ff9-6e7f-37b4da43d235", "425495"),
    "fade": ("Fade", "dade69b4-4f5a-8528-247b-219e5a1facd6", "1d2846"),
    "gekko": ("Gekko", "e370fa57-4757-3604-3648-499e1f642d3f", "371c5c"),
    "harbor": ("Harbor", "95b78ed7-4637-86d9-7e41-71ba8c293152", "275146"),
    "iso": ("Iso", "0e38b510-41a8-5780-5e8f-568b2a4f2d6c", "30336e"),
    "jett": ("Jett", "add6443a-41bd-e414-f6ad-e58d267f4e95", "25607a"),
    "kayo": ("KAY/O", "601dbbe7-43ce-be57-2a40-4abd24953621", "1c2a69"),
    "killjoy": ("Killjoy", "1e58de9c-4950-5125-93e9-a0aee9f98746", "522162"),
    "neon": ("Neon", "bb2a4828-46eb-8cd1-e765-15848195d751", "413476"),
    "omen": ("Omen", "8e253930-4c05-31dd-1b6c-968525494517", "433178"),
    "phoenix": ("Phoenix", "eb93336a-449b-9c1b-0a54-a891f7921d69", "74321c"),
    "raze": ("Raze", "f94c3b30-42be-e959-889c-5aa313dba261", "742e1e"),
    "reyna": ("Reyna", "a3bfb853-43b2-7238-a4f1-ad90e9e46bcc", "662d62"),
    "sage": ("Sage", "569fdd95-4d10-43ab-ca70-79becc718b46", "1f5148"),
    "skye": ("Skye", "6f2a04ca-43e0-be17-7f36-b3908627744d", "436a51"),
    "sova": ("Sova", "320b2a48-4d9b-a075-30f1-1f93a9b638fa", "355285"),
    "tejo": ("Tejo", "b444168c-4e35-8076-db47-ef9bf368f384", "80451b"),
    "viper": ("Viper", "707eab51-4836-f488-046a-cda6bf494859", "1a5f46"),
    "vyse": ("Vyse", "efba5359-4016-a1e5-7626-b1ae76895940", "492280"),
    "yoru": ("Yoru", "7f94d92c-4234-0a36-9646-3a87eb8b5c89", "222b67"),
}


def agent_image(label, kind="killfeedportrait"):
    return f"https://media.valorant-api.com/agents/{AGENTS[label][1]}/{kind}.png"


# Names and callouts that must survive translation (same check as notebook 04)
NAME_PATTERNS = {name: re.compile(rf"(?<!\w){re.escape(name)}(?!\w)", re.IGNORECASE)
                 for name in [a[0] for a in AGENTS.values()] + ["Mid"]}
SITE_PATTERNS = {site: re.compile(rf"(?<!\w){site}(?!\w)") for site in ["A", "B", "C"]}   # case-sensitive
# In English "A" is also an article ("A little help."): it only counts as a site when not followed by an ordinary word
SOURCE_SITE_PATTERNS = {**SITE_PATTERNS, "A": re.compile(
    r"(?<!\w)A(?![\w-])(?! (?!(?:site|long|short|main|link|lobby|heaven)\b)[a-z])")}


def lost_terms(source, translation):
    terms = [n for n, p in NAME_PATTERNS.items() if p.search(source)]
    terms += [s for s, p in SOURCE_SITE_PATTERNS.items() if p.search(source)]
    return [t for t in terms if not {**NAME_PATTERNS, **SITE_PATTERNS}[t].search(translation)]


# ---------------------------------------------------------------- models

@st.cache_resource(show_spinner=False)
def load_models():
    classifier = pipeline("audio-classification", model=CLASSIFIER_ID, device=-1)
    asr = pipeline("automatic-speech-recognition", model=ASR_ID, device=-1)
    translator = pipeline("translation", model=TRANSLATOR_ID, device=-1)
    normalizer = EnglishTextNormalizer(asr.tokenizer.english_spelling_normalizer)
    return classifier, asr, translator, normalizer


def load_audio(audio_bytes):
    """Audio file bytes -> (mono float32 at 16 kHz, same for the classifier's first 4 s, duration in s)."""
    waveform, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    def resample(w):   # identical to librosa.resample (soxr_hq, fixed length) used in the notebooks
        out = soxr.resample(w, sr, TARGET_SR, quality="HQ")
        n = int(np.ceil(len(w) * TARGET_SR / sr))
        return np.pad(out, (0, max(0, n - len(out))))[:n]

    return resample(waveform), resample(waveform[: int(CLASSIFIER_MAX_SECONDS * sr)]), len(waveform) / sr


def split_sentences(text):
    """Translate sentence by sentence: whole multi-sentence lines made the model silently drop a sentence (report 9.3)."""
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def word_diff_html(reference, hypothesis):
    """Word-level alignment of normalised script vs transcript, as HTML."""
    out = jiwer.process_words(reference, hypothesis)
    ref, hyp = out.references[0], out.hypotheses[0]
    parts = []
    for chunk in out.alignments[0]:
        r = " ".join(ref[chunk.ref_start_idx:chunk.ref_end_idx])
        h = " ".join(hyp[chunk.hyp_start_idx:chunk.hyp_end_idx])
        if chunk.type == "equal":
            parts.append(html.escape(r))
        elif chunk.type == "substitute":
            parts.append(f'<del>{html.escape(r)}</del> <ins>{html.escape(h)}</ins>')
        elif chunk.type == "delete":
            parts.append(f'<del>{html.escape(r)}</del>')
        else:
            parts.append(f'<ins>{html.escape(h)}</ins>')
    errors = out.substitutions + out.deletions + out.insertions
    return " ".join(parts), errors, out.wer


STEPS = ["Identifying agent", "Transcribing voice line", "Translating to Spanish"]


def loading_bar(slot, step, label=None):
    """Game-style loading bar: one segment per QA step; earlier steps full, current one animated."""
    segments = "".join(f'<div class="seg {"done" if i < step else "active" if i == step else ""}"></div>'
                       for i in range(len(STEPS)))
    slot.markdown(
        f'<div class="vq-load"><div class="top"><span>{label or STEPS[step]}<span class="dots"></span></span>'
        f'<span>{step + 1}/{len(STEPS)}</span></div><div class="segs">{segments}</div></div>', unsafe_allow_html=True)


def queued(slot, step):
    slot.markdown(f'<div class="vq-load queued"><div class="top"><span>Queued: {STEPS[step].lower()}</span>'
                  f'<span>{step + 1}/{len(STEPS)}</span></div></div>', unsafe_allow_html=True)


def footer():
    """Riot's required notice, shown on every page state."""
    st.markdown(
        f'<div class="vq-footer">Voice-Line QA was created under Riot Games\' "Legal Jibber Jabber" policy using assets '
        "owned by Riot Games. Riot Games does not endorse or sponsor this project.<br>"
        "This is a non-commercial student project for HKUST ISOM5240 and is not an official Riot Games website or tool. "
        "Voice lines from the NgThVinh/ValorantAgentVoiceLines dataset; agent art via valorant-api.com. "
        f'Models: <a href="https://huggingface.co/{CLASSIFIER_ID}">agent classifier</a>, '
        f'<a href="https://huggingface.co/{ASR_ID}">speech-to-text</a>, '
        f'<a href="https://huggingface.co/{TRANSLATOR_ID}">translation</a>. '
        f'<a href="{GITHUB_URL}">Source code</a>.</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- page

st.set_page_config(page_title="Voice-Line QA | Educational project", page_icon="🎧", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Anton&display=swap');
h1, h2, h3, .vq-display { font-family: 'Anton', sans-serif !important; text-transform: uppercase;
                          letter-spacing: .02em; font-weight: 400 !important; }
.vq-notice { border: 1px solid #FF465566; background: #FF465514; padding: .5rem .8rem;
             font-size: .85rem; margin-bottom: 1rem; }
.vq-kicker { color: #FF4655; font-weight: 700; letter-spacing: .2em; font-size: .8rem; text-transform: uppercase; }
.vq-title { font-family: 'Anton', sans-serif; font-size: clamp(2.4rem, 8vw, 3.6rem); line-height: 1;
            text-transform: uppercase; margin: .2rem 0 .4rem; }
.vq-title span { color: #FF4655; }
.vq-strip { display: flex; flex-wrap: wrap; gap: 4px; margin: .8rem 0 1.2rem; }
.vq-strip img { width: 34px; height: 34px; object-fit: cover; background: #1F2731; opacity: .85; }
.vq-card { background: #1F2731; border-left: 4px solid #FF4655; padding: 1rem 1.2rem; margin: .6rem 0 1rem;
           clip-path: polygon(0 0, 100% 0, 100% calc(100% - 16px), calc(100% - 16px) 100%, 0 100%); }
.vq-agent { display: flex; align-items: flex-end; gap: 1rem; overflow: hidden; }
.vq-agent img.portrait { height: 190px; margin: -1rem 0 -1rem -.6rem; }
.vq-agent > div { flex: 1; min-width: 0; }
@media (max-width: 640px) { .vq-agent img.portrait { height: 130px; } .vq-agent .name { font-size: 2rem; } }
.vq-agent .name { font-family: 'Anton', sans-serif; font-size: 2.6rem; line-height: 1; text-transform: uppercase; }
.vq-row { display: flex; align-items: center; gap: .6rem; margin: .35rem 0; }
.vq-row img { width: 30px; height: 30px; background: #0F1923; }
.vq-row .bar { flex: 1; height: 8px; background: #0F1923; }
.vq-row .bar div { height: 100%; background: #FF4655; }
.vq-row .label { width: 6.5rem; } .vq-row .pct { width: 3.5rem; text-align: right; font-variant-numeric: tabular-nums; }
.vq-badge { display: inline-block; padding: .15rem .6rem; font-weight: 700; letter-spacing: .08em;
            font-size: .8rem; text-transform: uppercase; }
.vq-ok { background: #2BD9A1; color: #0F1923; } .vq-bad { background: #FF4655; color: #fff; }
.vq-diff { font-size: 1.05rem; line-height: 1.9; }
.vq-card del { color: #FF8A94; text-decoration: line-through; } .vq-card ins { color: #2BD9A1; text-decoration: none; }
.vq-es { font-size: 1.25rem; }
.vq-load { background: #1F2731; border-left: 4px solid #FF4655; padding: .9rem 1.2rem 1.1rem; margin: .6rem 0 1rem;
           clip-path: polygon(0 0, 100% 0, 100% calc(100% - 16px), calc(100% - 16px) 100%, 0 100%); }
.vq-load .top { display: flex; justify-content: space-between; font-family: 'Anton', sans-serif; font-size: 1.1rem;
                text-transform: uppercase; letter-spacing: .08em; margin-bottom: .6rem; }
.vq-load .top span:last-child { color: #FF4655; font-variant-numeric: tabular-nums; }
.vq-load .dots::after { content: ""; animation: vq-dots 1.2s steps(4) infinite; }
.vq-load .segs { display: flex; gap: 8px; padding: 0 6px; }
.vq-load .seg { flex: 1; height: 12px; background: #0F1923; transform: skewX(-25deg); position: relative; overflow: hidden; }
.vq-load .seg.done { background: #FF4655; }
.vq-load .seg.active { background: #FF465533; }
.vq-load .seg.active::after { content: ""; position: absolute; top: 0; bottom: 0; width: 45%;
                              background: linear-gradient(90deg, transparent, #FF4655, transparent);
                              animation: vq-sweep 1.1s ease-in-out infinite; }
.vq-load.queued { opacity: .45; border-left-color: #ECE8E166; padding-bottom: .5rem; }
.vq-load.queued .top { margin: 0; font-size: .95rem; } .vq-load.queued .top span:last-child { color: inherit; }
@keyframes vq-sweep { from { left: -45%; } to { left: 100%; } }
@keyframes vq-dots { 0% { content: ""; } 25% { content: "."; } 50% { content: ".."; } 75% { content: "..."; } }
@media (prefers-reduced-motion: reduce) { .vq-load .seg.active::after, .vq-load .dots::after { animation: none; } }
.vq-footer { font-size: .8rem; opacity: .75; border-top: 1px solid #ECE8E133; margin-top: 2rem; padding-top: .8rem; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="vq-notice"><b>Educational project, not a Riot Games site.</b> '
    "Built by a student for the HKUST ISOM5240 course. It is not affiliated with, endorsed by "
    "or sponsored by Riot Games.</div>", unsafe_allow_html=True)

strip = "".join(f'<img src="{agent_image(k)}" alt="{v[0]}" title="{v[0]}" loading="lazy">' for k, v in AGENTS.items())
st.markdown(
    '<div class="vq-kicker">VALORANT · audio &amp; localization QA</div>'
    '<div class="vq-title">Voice-line <span>QA</span></div>'
    "Check a recorded voice line in one step: <b>who</b> is speaking, whether the recording <b>matches the "
    "script</b>, and a <b>draft Spanish subtitle</b> for the localization team."
    f'<div class="vq-strip">{strip}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- input

samples = pd.read_csv(SAMPLES_DIR / "samples.csv")
source = st.radio("Voice line", ["Sample from the test set", "Upload a recording"], horizontal=True)

if source == "Sample from the test set":
    options = {f'{AGENTS[r.agent][0]}: "{r.script_line}"': r for r in samples.itertuples()}
    choice = options[st.selectbox("Sample (a test-set clip, not used in training)", list(options))]
    audio_bytes = (SAMPLES_DIR / choice.file).read_bytes()
    audio_format = "audio/flac"
    expected_agent, default_script = choice.agent, choice.script_line
else:
    upload = st.file_uploader("Recording (WAV, FLAC, OGG or MP3, up to 30 s)", type=["wav", "flac", "ogg", "mp3"])
    audio_bytes = upload.getvalue() if upload else None
    audio_format = upload.type if upload else None
    expected_agent, default_script = None, ""

script_line = st.text_input("Script line (what the agent should say)", value=default_script,
                            help="Optional. Leave empty to only transcribe and translate.")

if audio_bytes:
    st.audio(audio_bytes, format=audio_format)

if not st.button("Run QA check", type="primary", disabled=not audio_bytes):
    footer()
    st.stop()

try:
    waveform, waveform_4s, duration = load_audio(audio_bytes)
except Exception:
    st.error("Could not read this audio file. Please upload a WAV, FLAC, OGG or MP3 recording.")
    footer()
    st.stop()
if duration > MAX_UPLOAD_SECONDS:
    st.error(f"This recording is {duration:.0f} s long; voice lines are a few seconds. Please upload up to {MAX_UPLOAD_SECONDS} s.")
    footer()
    st.stop()

timings = {}

# All three sections appear at once: the current step shows the loading bar, later steps wait in the queue
slots = []
for title in ["1 · Who is speaking?", "2 · Does it match the script?", "3 · Spanish subtitle draft"]:
    st.markdown(f"### {title}")
    slots.append(st.empty())
for step, slot in enumerate(slots):
    queued(slot, step)

loading_bar(slots[0], 0, "Loading models (first run only, about a minute)")
classifier, asr, translator, normalizer = load_models()
loading_bar(slots[0], 0)

# ---------------------------------------------------------------- 1. agent

start = time.perf_counter()
top3 = classifier({"raw": waveform_4s, "sampling_rate": TARGET_SR}, top_k=3)
timings["Agent"] = time.perf_counter() - start

best = top3[0]["label"]
name, _, colour = AGENTS[best]
verdict = ""
if expected_agent:
    verdict = ('<span class="vq-badge vq-ok">Matches the file\'s agent</span>' if best == expected_agent else
               f'<span class="vq-badge vq-bad">File says {AGENTS[expected_agent][0]}</span>')
rows = "".join(
    f'<div class="vq-row"><img src="{agent_image(p["label"])}" alt=""><span class="label">{AGENTS[p["label"]][0]}</span>'
    f'<div class="bar"><div style="width:{p["score"]:.0%}"></div></div><span class="pct">{p["score"]:.0%}</span></div>'
    for p in top3)
slots[0].markdown(
    f'<div class="vq-card" style="background: linear-gradient(90deg, #{colour}cc, #1F2731 70%);">'
    f'<div class="vq-agent"><img class="portrait" src="{agent_image(best, "fullportrait")}" alt="{name}">'
    f'<div><div class="name">{name}</div><div>{top3[0]["score"]:.0%} confidence</div>{verdict}</div></div></div>'
    f'<div class="vq-card"><b>Top 3</b>{rows}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- 2. transcript vs script

loading_bar(slots[1], 1)
start = time.perf_counter()
transcript = asr({"raw": waveform, "sampling_rate": TARGET_SR})["text"].strip()
timings["Transcript"] = time.perf_counter() - start

body = f"<b>Heard:</b> {html.escape(transcript)}"
if script_line.strip():
    diff, errors, wer = word_diff_html(normalizer(script_line), normalizer(transcript))
    badge = ('<span class="vq-badge vq-ok">Matches the script</span>' if errors == 0 else
             f'<span class="vq-badge vq-bad">{errors} word difference{"s" if errors > 1 else ""}: listen again</span>')
    body = f"{badge}<p style='margin-top:.6rem'>{body}</p>"
    if errors:
        body += (f"<div class='vq-diff'>{diff}</div><small>Compared word by word, ignoring case, punctuation "
                 "and spelling variants: <del>in the script</del> → <ins>heard instead</ins>.</small>")
slots[1].markdown(f'<div class="vq-card">{body}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- 3. Spanish subtitle

text_to_translate = script_line.strip() or transcript
loading_bar(slots[2], 2)
start = time.perf_counter()
sentences = split_sentences(text_to_translate)
spanish = " ".join(o["translation_text"] for o in translator(sentences, **TRANSLATION_SETTINGS)) if sentences else ""
timings["Spanish"] = time.perf_counter() - start

lost = lost_terms(text_to_translate, spanish)
check = ('<span class="vq-badge vq-ok">Names and callouts kept</span>' if not lost else
         f'<span class="vq-badge vq-bad">Check: {html.escape(", ".join(lost))} changed or missing</span>')
slots[2].markdown(
    f'<div class="vq-card">{check}<p class="vq-es" style="margin-top:.6rem">{html.escape(spanish)}</p>'
    f'<small>Translated from the {"script line" if script_line.strip() else "transcript"}, one sentence at a time. '
    "A machine draft for a human translator, not a final subtitle.</small></div>", unsafe_allow_html=True)

st.caption("Processing time on CPU: " + " · ".join(f"{k} {v:.1f} s" for k, v in timings.items())
           + f" · clip length {duration:.1f} s")

footer()
