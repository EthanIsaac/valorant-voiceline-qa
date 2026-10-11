"""Models and inference settings, identical to the ones evaluated in notebooks 01-04."""

CLASSIFIER_ID = "eibautistat/valorant-agent-voice-classifier"   # fine-tuned wav2vec2-base (notebook 01)
ASR_ID = "eibautistat/valorant-whisper-base-en"                  # fine-tuned whisper-base.en (notebook 03)
TRANSLATOR_ID = "Helsinki-NLP/opus-mt-en-es"                     # selected by blind human rating (notebook 04)

SAMPLING_RATE = 16_000            # all three speech models were trained on 16 kHz audio
CLASSIFIER_WINDOW_SECONDS = 4.0   # the classifier was trained on the first 4 s of each clip
MAX_RECORDING_SECONDS = 30        # voice lines last a few seconds; Whisper reads at most 30 s at a time
TOP_AGENTS = 3                    # candidates shown to the tester
TRANSLATION_SETTINGS = {"num_beams": 4, "max_new_tokens": 128}
