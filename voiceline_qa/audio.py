"""Decoding a recording into the inputs the models expect."""

import io
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import soxr

from voiceline_qa.config import CLASSIFIER_WINDOW_SECONDS, MAX_RECORDING_SECONDS, SAMPLING_RATE


class InvalidRecordingError(ValueError):
    """The file is not readable audio, or is too long to be a voice line."""


@dataclass(frozen=True)
class Recording:
    waveform: np.ndarray            # mono float32 at SAMPLING_RATE: input of speech recognition
    classifier_window: np.ndarray   # its first CLASSIFIER_WINDOW_SECONDS: input of agent identification
    duration_s: float


def load_recording(audio_bytes: bytes) -> Recording:
    """WAV, FLAC, OGG or MP3 bytes -> Recording, decoded and resampled exactly as in the notebooks."""
    try:
        waveform, rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    except (RuntimeError, TypeError, ValueError) as error:   # soundfile's LibsndfileError is a RuntimeError
        raise InvalidRecordingError("Could not read this audio file. Use a WAV, FLAC, OGG or MP3 recording.") from error
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    duration_s = len(waveform) / rate
    if duration_s > MAX_RECORDING_SECONDS:
        raise InvalidRecordingError(
            f"This recording is {duration_s:.0f} s long; voice lines last a few seconds. "
            f"Use a recording of up to {MAX_RECORDING_SECONDS} s.")

    window = waveform[: int(CLASSIFIER_WINDOW_SECONDS * rate)]
    return Recording(_resample(waveform, rate), _resample(window, rate), duration_s)


def _resample(waveform: np.ndarray, rate: int) -> np.ndarray:
    """Same result as librosa.resample(res_type="soxr_hq", fix=True), which the notebooks used."""
    resampled = soxr.resample(waveform, rate, SAMPLING_RATE, quality="HQ")
    length = int(np.ceil(len(waveform) * SAMPLING_RATE / rate))
    return np.pad(resampled, (0, max(0, length - len(resampled))))[:length]
