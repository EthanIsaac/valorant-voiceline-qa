"""The three QA checks on one recording, with the models and settings evaluated in notebooks 01-04."""

import re
import time

import jiwer
from transformers import pipeline
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

from voiceline_qa.audio import Recording
from voiceline_qa.checks import AgentCheck, QAReport, ScriptCheck, SubtitleDraft, WordDifference, agent_candidates
from voiceline_qa.config import (ASR_ID, CLASSIFIER_ID, SAMPLING_RATE, TOP_AGENTS, TRANSLATION_SETTINGS,
                                 TRANSLATOR_ID)
from voiceline_qa.terminology import lost_terms, protected_terms


class VoiceLineQA:
    """Identifies the agent, checks the recording against its script and drafts the Spanish subtitle."""

    def __init__(self, classifier, speech_recognizer, translator):
        self._classifier = classifier
        self._speech_recognizer = speech_recognizer
        self._translator = translator
        # Whisper's English normaliser: ignores casing, punctuation and spelling variants when comparing
        self._normalize = EnglishTextNormalizer(speech_recognizer.tokenizer.english_spelling_normalizer)

    @classmethod
    def from_pretrained(cls) -> "VoiceLineQA":
        """Downloads (first time only) and loads the three models on CPU."""
        return cls(pipeline("audio-classification", model=CLASSIFIER_ID, device=-1),
                   pipeline("automatic-speech-recognition", model=ASR_ID, device=-1),
                   pipeline("translation", model=TRANSLATOR_ID, device=-1))

    def run(self, recording: Recording, script_line: str = "") -> QAReport:
        agent = self.identify_agent(recording)
        script = self.check_script(recording, script_line)
        return QAReport(agent, script, self.draft_subtitle(script))

    def identify_agent(self, recording: Recording) -> AgentCheck:
        start = time.perf_counter()
        output = self._classifier({"raw": recording.classifier_window, "sampling_rate": SAMPLING_RATE},
                                  top_k=TOP_AGENTS)
        return AgentCheck(agent_candidates(output), time.perf_counter() - start)

    def check_script(self, recording: Recording, script_line: str = "") -> ScriptCheck:
        start = time.perf_counter()
        transcript = self._speech_recognizer({"raw": recording.waveform, "sampling_rate": SAMPLING_RATE})["text"]
        seconds = time.perf_counter() - start

        transcript, script_line = transcript.strip(), script_line.strip()
        if not script_line:
            return ScriptCheck(transcript, "", [], 0, 0, seconds)
        alignment, word_errors, script_word_count = self._align(script_line, transcript)
        return ScriptCheck(transcript, script_line, alignment, word_errors, script_word_count, seconds)

    def draft_subtitle(self, script: ScriptCheck) -> SubtitleDraft:
        """Translates the script line, or the transcript when there is none, one sentence at a time."""
        english_line, source = (script.script_line, "script line") if script.has_script else (script.transcript,
                                                                                               "transcript")
        start = time.perf_counter()
        sentences = split_sentences(english_line)
        translations = self._translator(sentences, **TRANSLATION_SETTINGS) if sentences else []
        spanish_line = " ".join(item["translation_text"] for item in translations)
        return SubtitleDraft(english_line, source, spanish_line, protected_terms(english_line),
                             lost_terms(english_line, spanish_line), time.perf_counter() - start)

    def _align(self, script_line: str, transcript: str) -> tuple[list[WordDifference], int, int]:
        script_words, heard_words = self._normalize(script_line), self._normalize(transcript)
        if not script_words.strip():   # e.g. a script line made only of punctuation
            return [], len(heard_words.split()), 0
        result = jiwer.process_words(script_words, heard_words)
        script_tokens, heard_tokens = result.references[0], result.hypotheses[0]
        alignment = [WordDifference(chunk.type,
                                    " ".join(script_tokens[chunk.ref_start_idx:chunk.ref_end_idx]),
                                    " ".join(heard_tokens[chunk.hyp_start_idx:chunk.hyp_end_idx]))
                     for chunk in result.alignments[0]]
        word_errors = result.substitutions + result.deletions + result.insertions
        return alignment, word_errors, len(script_tokens)


def split_sentences(text: str) -> list[str]:
    """Whole multi-sentence lines made the translation model silently drop a sentence (report, section 9.3)."""
    return [sentence for sentence in re.split(r"(?<=[.!?])\s+", text.strip()) if sentence]
