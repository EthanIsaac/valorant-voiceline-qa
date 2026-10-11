# VALORANT Voice-Line QA & Localization Assistant

ISOM5240 (HKUST) project for **Riot Games**. Every VALORANT agent ships with hundreds of voice lines, recorded and localized for many languages. This app checks a voice-line recording automatically:

1. **Who is speaking?** A fine-tuned audio classifier identifies which of the 26 agents is speaking.
2. **What is said?** Speech-to-text transcribes the line so it can be checked against the script.
3. **Subtitles.** The transcript is translated into Spanish as a subtitle draft.

## Repository layout

```
notebooks/          Google Colab notebooks (fine-tuning and model comparisons)
results/            metrics produced by the notebooks (CSV/JSON) and the experiment results Excel
streamlit_app.py    Streamlit app (deployed on Streamlit Community Cloud)
requirements.txt    app dependencies (CPU-only PyTorch)
samples/            five test-set voice lines used as app examples
```

Run the app locally: `pip install -r requirements.txt && streamlit run streamlit_app.py`

## Dataset

[`NgThVinh/ValorantAgentVoiceLines`](https://huggingface.co/datasets/NgThVinh/ValorantAgentVoiceLines), pinned to revision `c2e0d34`: 12,164 voice lines from 26 VALORANT agents, with transcripts.

## Disclaimer

This is a non-commercial student project for HKUST ISOM5240. It is not an official Riot Games website or tool.

VALORANT Voice-Line QA was created under Riot Games' "Legal Jibber Jabber" policy using assets owned by Riot Games. Riot Games does not endorse or sponsor this project.
