# -*- coding: utf-8 -*-
"""Neural English voice tier (Piper / VITS via ONNX).

Self-contained: phonemization is done by calling espeak-ng directly via
ctypes (no phonemizer/joblib/numpy dependency chain), and phonemes are
fed to the Piper ONNX model with is_phonemes=True. This avoids the heavy,
fragile phonemizer import tree entirely.

When the bundled espeak-ng library and a Piper voice load, English text is
spoken with a natural neural voice; otherwise the formant engine is used.

Credits: Piper (rhasspy / OHF-Voice, MIT); piper-onnx (thewh1teagle, MIT);
espeak-ng (GPL-3, phonemizer library); voice per its MODEL_CARD license.
"""
import os
import sys
import ctypes
import threading

_here = os.path.dirname(os.path.abspath(__file__))
_piper = None
_espeak = None
_sr = 16000
_lock = threading.Lock()
_tried = False

NEURAL_EN_AVAILABLE = False


def _voice_dir():
    return os.path.join(_here, "lib", "piper_voices")


def _find_voice():
    d = _voice_dir()
    if not os.path.isdir(d):
        return None, None
    onnx = None
    for f in sorted(os.listdir(d)):
        if f.endswith(".onnx"):
            onnx = os.path.join(d, f)
            break
    if not onnx:
        return None, None
    cfg = onnx + ".json"
    return (onnx, cfg) if os.path.exists(cfg) else (None, None)


def _init_espeak():
    """Load espeak-ng via ctypes for direct text->IPA phonemization."""
    global _espeak
    lib_dir = os.path.join(_here, "lib", "espeakng_loader")
    # locate the platform library
    names = ["espeak-ng.dll", "libespeak-ng.so", "libespeak-ng.dylib",
             "libespeak-ng.so.1"]
    lib_path = None
    for n in names:
        p = os.path.join(lib_dir, n)
        if os.path.exists(p):
            lib_path = p
            break
    if not lib_path:
        return False
    data_path = os.path.join(lib_dir, "espeak-ng-data")
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(lib_dir)
        except Exception:
            pass
    lib = ctypes.CDLL(lib_path)
    # AUDIO_OUTPUT_SYNCHRONOUS = 0x02; returns sample rate or -1
    if lib.espeak_Initialize(0x02, 0, data_path.encode("utf-8"), 0) == -1:
        return False
    lib.espeak_SetVoiceByName(b"en-us")
    lib.espeak_TextToPhonemes.restype = ctypes.c_char_p
    _espeak = lib
    return True


def _phonemize(text):
    """Text -> eSpeak IPA string (space-separated per clause), matching
    what Piper's phonemizer would produce."""
    out = []
    tptr = ctypes.c_char_p(text.encode("utf-8"))
    vptr = ctypes.cast(ctypes.pointer(tptr), ctypes.c_void_p)
    guard = 0
    while guard < 5000:
        guard += 1
        ph = _espeak.espeak_TextToPhonemes(vptr, 1, 0x02)  # UTF8 in, IPA
        if ph:
            out.append(ph.decode("utf-8"))
        remaining = ctypes.cast(
            vptr, ctypes.POINTER(ctypes.c_char_p)).contents.value
        if not remaining:
            break
    return " ".join(out)


def _try_init():
    global _piper, _tried, NEURAL_EN_AVAILABLE, _sr
    if _tried:
        return NEURAL_EN_AVAILABLE
    with _lock:
        if _tried:
            return NEURAL_EN_AVAILABLE
        _tried = True
        try:
            import warnings
            warnings.filterwarnings("ignore")
            from . import _libboot
            _libboot.boot()   # our numpy/onnxruntime first
            onnx, cfg = _find_voice()
            if not onnx:
                return False
            if not _init_espeak():
                return False
            import json
            with open(cfg, encoding="utf-8") as f:
                conf = json.load(f)
            _sr = conf.get("audio", {}).get("sample_rate", 16000)
            _pmap = conf.get("phoneme_id_map", {})
            import onnxruntime as ort
            import os as _os
            # Tune the session for LOW LATENCY: enable all graph
            # optimizations and use all CPU cores. This roughly halves the
            # per-call inference time versus default options.
            _so = ort.SessionOptions()
            try:
                _so.graph_optimization_level = \
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                _so.intra_op_num_threads = max(1, _os.cpu_count() or 2)
                _so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                _so.enable_mem_pattern = True
            except Exception:
                pass
            sess = ort.InferenceSession(
                onnx, sess_options=_so,
                providers=["CPUExecutionProvider"])
            _piper = _PiperRunner(sess, conf, _pmap)
            _piper.synth("ok")   # warm-up
            NEURAL_EN_AVAILABLE = True
        except Exception as e:
            NEURAL_EN_AVAILABLE = False
            try:
                from logHandler import log
                log.warning("ClaritySynth Piper English init failed: %r"
                            % e, exc_info=True)
            except Exception:
                pass
        return NEURAL_EN_AVAILABLE


class _PiperRunner(object):
    """Minimal Piper inference: IPA phonemes -> ids -> ONNX -> audio."""
    _BOS = "^"
    _EOS = "$"
    _PAD = "_"

    def __init__(self, sess, conf, pmap):
        self.sess = sess
        self.conf = conf
        self.pmap = pmap
        inf = conf.get("inference", {})
        self.length_scale = inf.get("length_scale", 1.0)
        self.noise_scale = inf.get("noise_scale", 0.667)
        self.noise_w = inf.get("noise_w", 0.8)
        self.n_speakers = conf.get("num_speakers", 1)
        self.in_names = [i.name for i in sess.get_inputs()]

    def _ids(self, phonemes):
        ids = []
        pad = self.pmap.get(self._PAD, [0])
        for sym in [self._BOS] + list(phonemes):
            if sym in self.pmap:
                ids.extend(self.pmap[sym])
                ids.extend(pad)
        ids.extend(self.pmap.get(self._EOS, []))
        return ids

    def synth(self, text, length_scale=None, phonemes=None,
              noise_scale=None, noise_w=None):
        import numpy as np
        # `phonemes` lets the caller supply exact IPA (character mode), which
        # avoids eSpeak mis-guessing short pseudo-words like "ay"/"eff".
        phon = phonemes if phonemes else _phonemize(text)
        ids = self._ids(phon)
        if not ids:
            return None, _sr
        ls = length_scale or self.length_scale
        x = np.array([ids], dtype=np.int64)
        x_len = np.array([x.shape[1]], dtype=np.int64)
        ns = self.noise_scale if noise_scale is None else noise_scale
        nw = self.noise_w if noise_w is None else noise_w
        scales = np.array([ns, ls, nw],
                          dtype=np.float32)
        feed = {"input": x, "input_lengths": x_len, "scales": scales}
        if "sid" in self.in_names and self.n_speakers > 1:
            feed["sid"] = np.array([0], dtype=np.int64)
        feed = {k: v for k, v in feed.items() if k in self.in_names}
        out = self.sess.run(None, feed)[0]
        return out.squeeze(), _sr


from collections import OrderedDict as _OD

_CACHE = _OD()
_CACHE_MAX = 64          # entries; short strings only, so memory is small
_CACHE_MAXLEN = 48       # only cache strings up to this many characters
_cache_lock = threading.Lock()


def _cache_get(key):
    with _cache_lock:
        v = _CACHE.get(key)
        if v is not None:
            _CACHE.move_to_end(key)     # LRU
        return v


def _cache_put(key, val):
    with _cache_lock:
        _CACHE[key] = val
        _CACHE.move_to_end(key)
        while len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)


def clear_cache():
    with _cache_lock:
        _CACHE.clear()


def sample_rate():
    return _sr


def synth_wave(text, length_scale=None, volume=1.0, phonemes=None,
               clarity=None):
    """Synthesize `text`, or — if `phonemes` is given — render that exact IPA
    string directly (used for single characters, where eSpeak's G2P is
    unreliable)."""
    if not _try_init():
        return None
    key = None
    if text and len(text) <= _CACHE_MAXLEN and phonemes is None:
        key = (text, length_scale, volume, clarity)
        hit = _cache_get(key)
        if hit is not None:
            return hit
    try:
        import numpy as np
        # `clarity` (0..100) reduces the model's stochastic noise, which
        # audibly cleans the voice up. None/0 = the model's own defaults.
        ns = nw = None
        if clarity:
            k = max(0.0, min(1.0, clarity / 100.0))
            ns = _piper.noise_scale * (1.0 - 0.5 * k)
            nw = _piper.noise_w * (1.0 - 0.5 * k)
        audio, sr = _piper.synth(text, length_scale=length_scale,
                                 phonemes=phonemes,
                                 noise_scale=ns, noise_w=nw)
        if audio is None:
            return None
        a = np.asarray(audio, dtype=np.float32)
        if a.size == 0:
            return None
        m = float(np.max(np.abs(a))) or 1.0
        if m > 1.0:
            a = a / m
        a = a * volume
        pcm = (a * 32000.0).astype(np.int16).tobytes()
        if key is not None and pcm:
            _cache_put(key, pcm)
        return pcm
    except Exception:
        return None
