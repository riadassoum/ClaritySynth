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

def _import_data_paths():
    """Import the data_paths module whether we are loaded as part of the
    claritysynth package (normal NVDA) or as a loose module (tests)."""
    try:
        from . import data_paths as _dp
        return _dp
    except Exception:
        import os as _os, sys as _sys
        _d = _os.path.dirname(_os.path.abspath(__file__))
        if _d not in _sys.path:
            _sys.path.insert(0, _d)
        import data_paths as _dp
        return _dp



_here = os.path.dirname(os.path.abspath(__file__))
_model = None
_lock = threading.Lock()
_tried = False
_sr = 22050
_model_id = "mixer80"
_vocoder_id = "vocos"
_forced_model = None      # set by the Primary variant combo; None = auto
_forced_vocoder = None    # set by the Vocoder combo; None = auto

NEURAL_TTS_AVAILABLE = False
SPEAKERS = [0]


def _data_dir():
    return os.path.join(_here, "lib", "tts_arabic", "data")


def _model_dirs():
    """Directories that may hold Arabic acoustic models / vocoders: the
    persistent user data dir (downloads, survive updates) first, then the
    bundled lib/tts_arabic/data."""
    dirs = []
    try:
        data_paths = _import_data_paths()
        dirs.extend(data_paths.search_dirs(os.path.join("tts_arabic", "data")))
    except Exception:
        pass
    b = _data_dir()
    if b not in dirs and os.path.isdir(b):
        dirs.append(b)
    return dirs


def _find_file(fname):
    """Return the directory containing `fname`, searching the persistent dir
    before the bundled one, or None."""
    for d in _model_dirs():
        if os.path.exists(os.path.join(d, fname)):
            return d
    return None


def _detect_models():
    # union of files available across all model dirs
    have = set()
    for d in _model_dirs():
        try:
            have |= set(os.listdir(d))
        except OSError:
            pass
    if not have:
        return None, None
    _model_file = {"fastpitch": "fp_ms.onnx", "mixer128": "mixer128.onnx",
                   "mixer80": "mixer80.onnx"}
    if _forced_model and _model_file.get(_forced_model) in have:
        mel = _forced_model
    else:
        mel = "fastpitch" if "fp_ms.onnx" in have else (
            "mixer128" if "mixer128.onnx" in have else (
                "mixer80" if "mixer80.onnx" in have else None))
    _voc_file = {"vocos": "vocos22.onnx", "hifigan": "hifigan.onnx",
                 "vocos44": "vocos44.onnx"}
    if _forced_vocoder and _voc_file.get(_forced_vocoder) in have:
        voc = _forced_vocoder
    else:
        voc = "vocos" if "vocos22.onnx" in have else (
            "hifigan" if "hifigan.onnx" in have else (
                "vocos44" if "vocos44.onnx" in have else None))
    return mel, voc


def select_model(name):
    """Pin the Arabic acoustic model (mixer128 / mixer80 / fastpitch), or
    None for automatic. Forces a lazy reload on the next synth."""
    global _forced_model, _model, _tried, NEURAL_TTS_AVAILABLE
    if name == _forced_model:
        return
    _forced_model = name
    _model = None
    _tried = False
    NEURAL_TTS_AVAILABLE = False


def select_vocoder(name):
    """Pin the vocoder (vocos / vocos44 / hifigan), or None for automatic.
    Forces a lazy reload on the next synth."""
    global _forced_vocoder, _model, _tried, NEURAL_TTS_AVAILABLE
    if name == _forced_vocoder:
        return
    _forced_vocoder = name
    _model = None
    _tried = False
    NEURAL_TTS_AVAILABLE = False


def _speaker_count_from_onnx(path):
    """Read the number of speakers from a model\'s ONNX speaker embedding
    (rows of speaker_emb.weight / emb_g.weight). Returns 1 if not found or on
    any error, so a single-speaker model still works."""
    try:
        import onnx
        m = onnx.load(path, load_external_data=False)
        for init in m.graph.initializer:
            nm = init.name.lower()
            if "speaker_emb" in nm or "emb_g" in nm or nm.endswith("speaker.weight"):
                if len(init.dims) >= 1 and init.dims[0] > 1:
                    return int(init.dims[0])
        # also check the 'speaker' input's embedding via a value_info scan
    except Exception:
        pass
    return 1


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
            _mf = {"fastpitch": "fp_ms.onnx", "mixer128": "mixer128.onnx",
                   "mixer80": "mixer80.onnx"}
            _vf = {"vocos": "vocos22.onnx", "hifigan": "hifigan.onnx",
                   "vocos44": "vocos44.onnx"}
            _mdir = _find_file(_mf.get(mel, "")) if mel else None
            _vdir = _find_file(_vf.get(voc, "")) if voc else None
            _model = get_model(model_id=mel, vocoder_id=voc, cuda=False,
                               model_dir=_mdir, vocoder_dir=_vdir)
            # detect the real speaker count from the model file (mixer80 and
            # other multi-speaker models are no longer assumed single-speaker)
            _mpath = None
            if _mdir and mel:
                _mpath = os.path.join(_mdir, _mf.get(mel, ""))
            _ncount = _speaker_count_from_onnx(_mpath) if _mpath else 1
            if _ncount <= 1:
                # fall back to known multi-speaker families
                _ncount = 4 if mel in ("fastpitch", "mixer128", "mixer80") else 1
            # expose at most 4 distinct speakers (models carry more embedding
            # slots than trained-distinct voices)
            _ncount = max(1, min(4, _ncount))
            SPEAKERS = list(range(_ncount))
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
        text = diacritized_text
        # If the utterance ends on a bare long vowel (alef / alef maqsura,
        # optionally with a diacritic before it) and has no final
        # punctuation, the model tends to clip that last vowel. Appending a
        # period gives it a word-boundary token so it articulates the vowel;
        # we trim the brief trailing silence afterwards.
        _added_boundary = False
        _stripped = text.rstrip()
        if _stripped and _stripped[-1] in "\u0627\u0649":  # ا  ى
            text = _stripped + " ."
            _added_boundary = True
        out = _model.infer(text, speaker=speaker, pace=pace,
                           denoise=denoise,
                           volume=volume, pitch_mul=pitch_mul,
                           pitch_add=pitch_add, vowelizer=None)
        wave = out[0] if isinstance(out, tuple) else out
        wave = np.asarray(wave, dtype=np.float32)
        if wave.size == 0:
            return None
        if _added_boundary and wave.size > 0:
            # remove the short trailing silence the period produced (~120ms)
            # by trimming trailing near-zero samples, then a small fixed tail
            sr = 44100 if _vocoder_id == "vocos44" else 22050
            thr = 0.02 * (float(np.max(np.abs(wave))) or 1.0)
            end = wave.size
            while end > 0 and abs(wave[end - 1]) < thr:
                end -= 1
            # keep a tiny natural release after the vowel
            end = min(wave.size, end + int(sr * 0.02))
            if end > sr // 20:      # never trim to less than ~50ms
                wave = wave[:end]
        m = float(np.max(np.abs(wave))) or 1.0
        if m > 1.0:
            wave = wave / m
        return (wave * 32000.0).astype(np.int16).tobytes()
    except Exception:
        return None
