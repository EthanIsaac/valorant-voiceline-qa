"""Unit tests of the business layer. They use small stand-ins for the three models, so they run in seconds.

Run from the repository root: python -m pytest tests
"""

import io
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from voiceline_qa.audio import InvalidRecordingError, load_recording
from voiceline_qa.config import SAMPLING_RATE
from voiceline_qa.evaluation import (EvaluationClip, run_performance_test, runtime_by_stage, script_accuracy_by_subset,
                                     summarize_accuracy, summarize_runtime, summary_table)
from voiceline_qa.service import VoiceLineQA, split_sentences
from voiceline_qa.terminology import lost_terms, protected_terms


def wav_bytes(seconds: float, rate: int = 48_000, channels: int = 1) -> bytes:
    buffer = io.BytesIO()
    samples = np.zeros((int(seconds * rate), channels), dtype="float32")
    sf.write(buffer, samples, rate, format="WAV")
    return buffer.getvalue()


def fake_qa(transcript: str, spanish: dict[str, str] | None = None) -> VoiceLineQA:
    """VoiceLineQA with stand-ins: the classifier always answers Jett, Whisper always hears `transcript`."""
    def classifier(inputs, top_k):
        return [{"label": "jett", "score": 0.9}, {"label": "sage", "score": 0.06}, {"label": "sova", "score": 0.04}]

    def speech_recognizer(inputs):
        return {"text": f" {transcript}"}
    speech_recognizer.tokenizer = SimpleNamespace(english_spelling_normalizer={})

    def translator(sentences, **settings):
        return [{"translation_text": (spanish or {}).get(sentence, sentence)} for sentence in sentences]

    return VoiceLineQA(classifier, speech_recognizer, translator)


# ---------------------------------------------------------------- audio

def test_recording_is_resampled_to_16_khz_with_a_4_second_classifier_window():
    recording = load_recording(wav_bytes(6.0, rate=48_000, channels=2))
    assert recording.duration_s == pytest.approx(6.0)
    assert len(recording.waveform) == 6 * SAMPLING_RATE
    assert len(recording.classifier_window) == 4 * SAMPLING_RATE


def test_unreadable_audio_is_rejected():
    with pytest.raises(InvalidRecordingError, match="Could not read"):
        load_recording(b"not audio")


def test_recording_longer_than_30_seconds_is_rejected():
    with pytest.raises(InvalidRecordingError, match="31 s long"):
        load_recording(wav_bytes(31.0, rate=16_000))


# ---------------------------------------------------------------- checks

def test_matching_transcript_passes_the_script_check():
    report = fake_qa("Let's rush C.").run(load_recording(wav_bytes(1.0)), "Let's rush C.")
    assert report.agent.predicted.agent.name == "Jett"
    assert report.agent.matches("jett") and not report.agent.matches("sage")
    assert report.script.matches_script and report.script.word_errors == 0


def test_misheard_word_is_reported_as_a_substitution():
    check = fake_qa("There's something about you, Kale.").check_script(
        load_recording(wav_bytes(1.0)), "There's something about you, Sage.")
    assert not check.matches_script
    assert check.word_errors == 1
    assert [(d.script_words, d.heard_words) for d in check.alignment if d.kind == "substitute"] == [("sage", "kale")]


def test_without_a_script_line_the_transcript_is_translated():
    qa = fake_qa("Watching here.")
    script = qa.check_script(load_recording(wav_bytes(1.0)))
    subtitle = qa.draft_subtitle(script)
    assert not script.has_script and not script.matches_script
    assert subtitle.translated_from == "transcript" and subtitle.english_line == "Watching here."


def test_subtitle_is_translated_sentence_by_sentence_and_flags_lost_names():
    spanish = {"You did well, Breach.": "Lo hiciste bien, Violación.", "Don't expect a hug.": "No esperes un abrazo."}
    qa = fake_qa("", spanish)
    subtitle = qa.draft_subtitle(qa.check_script(load_recording(wav_bytes(1.0)),
                                                 "You did well, Breach. Don't expect a hug."))
    assert subtitle.spanish_line == "Lo hiciste bien, Violación. No esperes un abrazo."
    assert subtitle.lost_terms == ["Breach"] and not subtitle.keeps_protected_terms


def test_split_sentences():
    assert split_sentences("Stop them! This is our turf!") == ["Stop them!", "This is our turf!"]
    assert split_sentences("  ") == []


# ---------------------------------------------------------------- terminology

@pytest.mark.parametrize("english, spanish, lost", [
    ("Let's rush A.", "Vamos a apresurar a A.", []),
    ("Let's rush A.", "Vamos a acelerar a.", ["A"]),          # lower-case "a" is a preposition, not the site
    ("Be careful, KAY/O.", "Cuidado, Kay.", ["KAY/O"]),
    ("Sova, scan Mid.", "Sova, escanea Mid.", []),
])
def test_lost_terms(english, spanish, lost):
    assert lost_terms(english, spanish) == lost


def test_article_a_is_not_a_site():
    assert protected_terms("A little help here.") == []
    assert protected_terms("Enemy on A long.") == ["A"]


# ---------------------------------------------------------------- evaluation

def test_performance_test_measures_accuracy_and_runtime(tmp_path):
    (tmp_path / "clips").mkdir()
    (tmp_path / "clips" / "clip.wav").write_bytes(wav_bytes(1.0))
    clips = [EvaluationClip("clip.wav", "jett", "Let's rush C.", 1.0, True),
             EvaluationClip("clip.wav", "sage", "Let's rush B.", 1.0, False)]
    started = []
    run = run_performance_test(fake_qa("Let's rush C."), clips, tmp_path,
                               on_clip_start=lambda index, clip: started.append(index))

    assert started == [0, 1]
    accuracy = summarize_accuracy(run.results)
    assert accuracy.clips == 2
    assert accuracy.agent_accuracy == 0.5                   # the stand-in always answers Jett
    assert accuracy.script_exact_match == 0.5               # "B" heard as "C" on the second clip
    assert accuracy.word_error_rate == pytest.approx(1 / 8)  # 1 error over 2 x 4 normalised words ("let us rush c")
    assert accuracy.clips_with_protected_terms == 2 and accuracy.protected_terms_kept == 1.0
    assert list(script_accuracy_by_subset(run.results).index) == ["Unseen lines", "Seen lines"]

    runtime = summarize_runtime(run.results)
    assert 0 < runtime.median_seconds_per_clip <= runtime.max_seconds_per_clip
    assert runtime.real_time_factor == pytest.approx(runtime.mean_seconds_per_clip / 1.0)
    assert runtime_by_stage(run.results).loc["Whole check", "share"] == pytest.approx(1.0)
    assert run.total_seconds >= run.results[["seconds_agent", "seconds_script", "seconds_subtitle"]].sum().sum()

    summary = summary_table(run).set_index("metric")["value"]
    assert summary["accuracy.agent_accuracy"] == 0.5 and "environment.cpu_count" in summary
