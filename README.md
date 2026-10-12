# VALORANT Voice-Line QA & Localization Assistant

ISOM5240 (HKUST) project, with **Riot Games** as the case study. Every VALORANT agent ships with hundreds of voice lines, recorded and localized for many languages, and audio QA testers must check each recording against its script. This app automates that check:

1. **Who is speaking?** A fine-tuned audio classifier identifies which of the 26 agents is speaking.
2. **What is said?** Speech-to-text transcribes the line so it can be checked against the script.
3. **Subtitles.** The transcript is translated into Spanish as a subtitle draft.

## Repository layout

```
model_development/        how the models were built and chosen (Google Colab)
  notebooks/              01 agent classifier fine-tuning · 02 speech-to-text comparison
                          03 Whisper fine-tuning · 04 Spanish subtitle comparison
  results/                metrics produced by the notebooks (CSV/JSON, SHA-256 fingerprints)

app/                      the Streamlit app deployed on Streamlit Community Cloud
  app.py                  entry point: page setup, shared layout, navigation
  app_pages/              qa_check.py (a tester checks one recording) · performance.py (accuracy and runtime test)
  voiceline_qa/           business layer (no Streamlit): audio loading, the three checks, evaluation
  ui/                     presentation layer: theme, stylesheet, visual components
  test_clips/             104 held-out test clips (4 per agent) with their script lines
  static/                 images: banner (composed from official agent portraits) and author portrait
  tests/                  unit tests of the business layer
  requirements.txt        app dependencies (CPU-only PyTorch)

.streamlit/config.toml    app theme and server settings (read from the repository root)
```

Run the app from the repository root:

```bash
pip install -r app/requirements.txt
streamlit run app/app.py
```

Run the unit tests: `cd app && python -m pytest tests`

## Dataset

[`NgThVinh/ValorantAgentVoiceLines`](https://huggingface.co/datasets/NgThVinh/ValorantAgentVoiceLines), pinned to revision `c2e0d34`: 12,164 voice lines from 26 VALORANT agents, with transcripts.

## Disclaimer

Non-commercial course project for HKUST ISOM5240. Not an official Riot Games website or tool, and not affiliated with Riot Games.

VALORANT Voice-Line QA was created under Riot Games' "Legal Jibber Jabber" policy using assets owned by Riot Games. Riot Games does not endorse or sponsor this project.
