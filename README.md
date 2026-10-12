# VALORANT Voice-Line QA & Localization Assistant

ISOM5240 (HKUST) project, with **Riot Games** as the case study. Every VALORANT agent ships with hundreds of voice lines, recorded and localized for many languages, and audio QA testers must check each recording against its script. This app automates that check:

1. **Who is speaking?** A fine-tuned audio classifier identifies which of the 26 agents is speaking.
2. **What is said?** Speech-to-text transcribes the line so it can be checked against the script.
3. **Subtitles.** The transcript is translated into Spanish as a subtitle draft.

**Live app:** https://valorant-voiceline-app.streamlit.app/

## Models

| Task | Model | Base model | How it was chosen |
|---|---|---|---|
| Agent voice classifier | [`eibautistat/valorant-agent-voice-classifier`](https://huggingface.co/eibautistat/valorant-agent-voice-classifier) | `facebook/wav2vec2-base`, fine-tuned | Best of 3 fine-tuned encoders (notebook 01) |
| Speech-to-text | [`eibautistat/valorant-whisper-base-en`](https://huggingface.co/eibautistat/valorant-whisper-base-en) | `openai/whisper-base.en`, fine-tuned | Best of 3 pre-trained models (notebook 02), then fine-tuned (notebook 03) |
| Spanish subtitles | [`Helsinki-NLP/opus-mt-en-es`](https://huggingface.co/Helsinki-NLP/opus-mt-en-es) | pre-trained | Best of 3 in a blind human rating (notebook 04) |

## Results

| Measure (held-out test data) | Result |
|---|---|
| Agent classifier: test accuracy (26 agents) | 90.5% (macro F1 0.905) |
| Speech-to-text: test word error rate | 4.2% (11.8% before fine-tuning) |
| Subtitles: mean human rating (1–5) | 4.03; 70% rated usable (4 or 5) |
| App, 104 test clips: agent identified | 91.3% |
| App, 104 test clips: transcript matches the script exactly | 83.7% (word error rate 4.4%) |
| App, 104 test clips: subtitles keep every agent name and callout | 100% |

Every number comes from the notebooks and the app's performance page. The full tables are in [`model_development/results/`](model_development/results/), and [`Experimental_results.xlsx`](model_development/results/Experimental_results.xlsx) collects them in one workbook.

## Repository layout

```
model_development/        how the models were built and chosen (Google Colab)
  notebooks/              01 agent classifier fine-tuning · 02 speech-to-text comparison
                          03 Whisper fine-tuning · 04 Spanish subtitle comparison
  results/                metrics produced by the notebooks (CSV/JSON, SHA-256 fingerprints),
                          the app's performance test (app_performance/) and Experimental_results.xlsx

app/                      the Streamlit app deployed on Streamlit Community Cloud
  app.py                  entry point: page setup, shared layout, navigation
  app_pages/              qa_check.py (a tester checks one recording) · performance.py (accuracy test on the test clips)
  voiceline_qa/           business layer (no Streamlit): audio loading, the three checks, evaluation
  ui/                     presentation layer: theme, stylesheet, visual components
  test_clips/             104 held-out test clips (4 per agent) with their script lines
  static/                 images: banner (composed from official agent portraits) and author portrait
  tests/                  unit tests of the business layer
  requirements.txt        app dependencies (CPU-only PyTorch)

.streamlit/config.toml    app theme and server settings (read from the repository root)
```

## Running the notebooks

Open a notebook from its **Open in Colab** badge, choose a GPU runtime (Runtime → Change runtime type → T4 GPU) and use **Run all**. No token or secret is needed: the notebooks download the public dataset and models, and notebooks 02–04 read the data split and ratings from this repository.

- Fine-tuning on a GPU is not bit-for-bit reproducible, so a rerun (especially on another GPU type) can differ slightly from the saved results. The data split and cleaning are exactly reproducible and checked against SHA-256 fingerprints.
- Notebooks 01 and 03 publish their model to Hugging Face only when the environment variable `UPLOAD_HF_MODELS` is `true` (run `%env UPLOAD_HF_MODELS=true` before **Run all**; the token is then read from the Colab secret `HF_TOKEN`). Otherwise the upload is skipped and the published model is used.
- Each notebook ends by downloading a zip of its results.

## Running the app

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
