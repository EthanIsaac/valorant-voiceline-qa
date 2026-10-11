# VALORANT Voice-Line QA & Localization Assistant

ISOM5240 (HKUST) project, with **Riot Games** as the case study. Every VALORANT agent ships with hundreds of voice lines, recorded and localized for many languages, and audio QA testers must check each recording against its script. This app automates that check:

1. **Who is speaking?** A fine-tuned audio classifier identifies which of the 26 agents is speaking.
2. **What is said?** Speech-to-text transcribes the line so it can be checked against the script.
3. **Subtitles.** The transcript is translated into Spanish as a subtitle draft.

## Repository layout

```
notebooks/            Google Colab notebooks (fine-tuning and model comparisons)
results/              metrics produced by the notebooks (CSV/JSON) and the experiment results Excel

streamlit_app.py      app entry point: page setup, shared layout, navigation
app_pages/
  qa_check.py         QA check: a tester checks one recording
  performance.py      performance test: the app's accuracy on held-out test clips (linked from the footer)
voiceline_qa/         business layer (no Streamlit): audio loading, the three checks, evaluation
ui/                   presentation layer: theme, stylesheet, visual components
tests/                unit tests of the business layer (python -m pytest tests)

samples/              five held-out test clips offered as examples in the QA check
evaluation/           performance-test set: 104 held-out test clips (4 per agent) and their script lines
static/               app images: banner (composed from official agent portraits) and author portrait
requirements.txt      app dependencies (CPU-only PyTorch)
```

Run the app locally: `pip install -r requirements.txt && streamlit run streamlit_app.py`

## Dataset

[`NgThVinh/ValorantAgentVoiceLines`](https://huggingface.co/datasets/NgThVinh/ValorantAgentVoiceLines), pinned to revision `c2e0d34`: 12,164 voice lines from 26 VALORANT agents, with transcripts.

## Disclaimer

Non-commercial course project for HKUST ISOM5240. Not an official Riot Games website or tool, and not affiliated with Riot Games.

VALORANT Voice-Line QA was created under Riot Games' "Legal Jibber Jabber" policy using assets owned by Riot Games. Riot Games does not endorse or sponsor this project.
