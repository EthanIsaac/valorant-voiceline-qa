# VALORANT Voice-Line QA & Localization Assistant

ISOM5240 (HKUST) project for **Riot Games**. Every VALORANT agent ships with hundreds of voice lines, recorded and localized for many languages. This app checks a voice-line recording automatically:

1. **Who is speaking?** A fine-tuned audio classifier identifies which of the 26 agents is speaking.
2. **What is said?** Speech-to-text transcribes the line so it can be checked against the script.
3. **Subtitles.** The transcript is translated into Spanish as a subtitle draft.

## Repository layout

```
notebooks/  Google Colab notebooks (fine-tuning and model comparisons)
results/    metrics produced by the notebooks (CSV/JSON) and the experiment results Excel
app/        Streamlit app
```

## Dataset

[`NgThVinh/ValorantAgentVoiceLines`](https://huggingface.co/datasets/NgThVinh/ValorantAgentVoiceLines), pinned to revision `c2e0d34`: 12,164 voice lines from 26 VALORANT agents, with transcripts.
