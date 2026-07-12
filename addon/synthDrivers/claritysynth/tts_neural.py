# -*- coding: utf-8 -*-
"""Neural TTS voice tier — lightweight single-package build.

Uses the bundled MIT/BSD `tts_arabic` package with the FAST models:
MixerTTS-80 (Text->Mel) + Vocos (vocoder), at 22.05 kHz. Our own
diacritizer supplies vowelized text, so the neural voice receives ready
input and never self-downloads. Formant engine remains the fallback.
"""
import os
import sys
import threading

_here = os.path.dirname(os.path.abspath(__file__))
_model = None
_lock = threading.Lock()
_tried = False
_sr = 22050
_model_id = "mixer80"
_vocoder_id = "vocos"

NEURAL_TTS_AVAILABLE = False
SPEAKERS = [0]


def _data_dir():
    return os.path.join(_here, "lib", "tts_arabic", "data")


def _detect_models():
    data = _data_dir()
    if not os.path.isdir(data):
        return None, None
    have = set(os.listdir(data))
    # Prefer FastPitch (if a pack is installed) for top quality, then
    # mixer128 (multi-speaker, only 13.5 MB — 4 voices), then mixer80.
    mel = "fastpitch" if "fp_ms.onnx" in have else (
        "mixer128" if "mixer128.onnx" in have else (
            "mixer80" if "mixer80.onnx" in have else None))
    voc = "vocos" if "vocos22.onnx" in have else (
        "hifigan" if "hifigan.onnx" in have else (
            "vocos44" if "vocos44.onnx" in have else None))
    return mel, voc


def _try_init():
    global _model, _tried, NEURAL_TTS_AVAILABLE, _sr, SPEAKERS
    global _model_id, _vocoder_id
    if _tried:
        return NEURAL_TTS_AVAILABLE
    with _lock:
        if _tried:
            return NEURAL_TTS_AVAILABLE
        _tried = True
        try:
            import warnings
            warnings.filterwarnings("ignore")
            mel, voc = _detect_models()
            if mel is None or voc is None:
                return False
            _model_id, _vocoder_id = mel, voc
            _sr = 44100 if voc == "vocos44" else 22050
            # Force-load OUR bundled numpy/onnxruntime before any broken
            # copy from another add-on can bind (see _libboot).
            from . import _libboot
            if not _libboot.boot():
                raise ImportError("bundled numpy/onnxruntime unavailable")
            from tts_arabic import get_model
            _model = get_model(model_id=mel, vocoder_id=voc, cuda=False)
            SPEAKERS = [0, 1, 2, 3] if mel in ("fastpitch", "mixer128") \
                else [0]
            _ = _model.infer("نَصٌّ.", speaker=SPEAKERS[0])
            NEURAL_TTS_AVAILABLE = True
        except Exception as e:
            NEURAL_TTS_AVAILABLE = False
            try:
                from logHandler import log
                log.warning("ClaritySynth neural TTS init failed: %r" % e,
                            exc_info=True)
            except Exception:
                pass
        return NEURAL_TTS_AVAILABLE


def sample_rate():
    return _sr


def synth_wave(diacritized_text, speaker=0, pace=1.0, pitch_mul=1.0,
               pitch_add=0.0, volume=0.9, denoise=0.025):
    """Return int16 PCM bytes for diacritized Arabic, or None on failure."""
    if not _try_init():
        return None
    try:
        import numpy as np
        if speaker not in SPEAKERS:
            speaker = SPEAKERS[0]
        out = _model.infer(diacritized_text, speaker=speaker, pace=pace,
                           denoise=denoise,
                           volume=volume, pitch_mul=pitch_mul,
                           pitch_add=pitch_add, vowelizer=None)
        wave = out[0] if isinstance(out, tuple) else out
        wave = np.asarray(wave, dtype=np.float32)
        if wave.size == 0:
            return None
        m = float(np.max(np.abs(wave))) or 1.0
        if m > 1.0:
            wave = wave / m
        return (wave * 32000.0).astype(np.int16).tobytes()
    except Exception:
        return None
