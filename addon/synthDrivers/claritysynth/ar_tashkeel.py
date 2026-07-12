# -*- coding: utf-8 -*-
"""Multi-backend Arabic diacritization (tashkeel) for ClaritySynth.

Exposes several interchangeable tashkeel engines so the user can pick the
one that reads their material best:

  * libtashkeel  - the libtashkeel engine (compiled), fast and robust
  * rawi         - the Rawi ensemble ONNX diacritizer
  * shakkelha    - Shakkelha RNN (ONNX)
  * shakkala     - Shakkala (ONNX)
  * off          - no automatic diacritization

Each backend is loaded lazily on first use and cached, so switching in the
settings dialog is cheap and a backend that fails to load never blocks the
others (we simply fall back to the next available one).

The Rawi backend is adapted from the NabraTTS add-on by "pbt", shared by
Ilyas Dragonoid. libtashkeel is bundled from the same add-on.
"""
import json
import os
import re
import threading
import unicodedata

_here = os.path.dirname(os.path.abspath(__file__))
_LIB = os.path.join(_here, "lib")
_VOWELIZERS = os.path.join(_LIB, "vowelizers")

# backend id -> human label (label is filled in by the driver for i18n)
BACKENDS = ("libtashkeel", "rawi", "shakkelha", "shakkala", "off")
DEFAULT_BACKEND = "libtashkeel"

_lock = threading.Lock()
_cache = {}          # backend id -> callable(text) -> str, or False if broken
_current = DEFAULT_BACKEND

_DIAC_RE = re.compile(r"[\u064B-\u065F\u0670\u0640]|[\u0610-\u061A]"
                      r"|[\u06D6-\u06ED]")


def set_backend(name):
    """Select the active tashkeel backend."""
    global _current
    if name in BACKENDS:
        _current = name


def get_backend():
    return _current


def _strip(text):
    return _DIAC_RE.sub("", text)


class _Rawi(object):
    """Rawi ensemble ONNX diacritizer (adapted from NabraTTS)."""

    def __init__(self, onnx_path):
        vocab_path = os.path.join(os.path.dirname(onnx_path), "vocab.json")
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        self.char_to_idx = vocab["char_to_idx"]
        self.diac_to_idx = vocab["diac_to_idx"]
        self.idx_to_diac = {v: k for k, v in self.diac_to_idx.items()}
        self._pad_id = self.char_to_idx.get("<PAD>", 0)
        self._unk_id = self.char_to_idx.get("<UNK>", 1)
        import onnxruntime as ort
        so = ort.SessionOptions()
        try:
            so.graph_optimization_level = \
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.intra_op_num_threads = max(1, os.cpu_count() or 2)
        except Exception:
            pass
        self.sess = ort.InferenceSession(
            onnx_path, sess_options=so, providers=["CPUExecutionProvider"])

    def _encode(self, text):
        import numpy as np
        text = unicodedata.normalize("NFD", text)
        text = _strip(text)
        ids = []
        for ch in text:
            idx = self.char_to_idx.get(ch)
            if idx is None:
                nfc = unicodedata.normalize("NFC", ch)
                idx = self.char_to_idx.get(nfc, self._unk_id)
            ids.append(idx)
        return np.array(ids, dtype=np.int64)

    def _decode(self, base, cls):
        if cls.ndim == 2:
            cls = cls[0]
        out = []
        for i, ch in enumerate(base):
            out.append(ch)
            if i < len(cls):
                d = int(cls[i])
                if d:
                    s = self.idx_to_diac.get(d, "")
                    if s:
                        out.append(s)
        return "".join(out)

    def __call__(self, text):
        import numpy as np
        if not text or not text.strip():
            return text
        stripped = _strip(unicodedata.normalize("NFD", text))
        if not stripped:
            return text
        ids = self._encode(stripped)
        if ids.size == 0:
            return text
        outs = self.sess.run(None, {"input": ids[np.newaxis, :]})
        out = self._decode(stripped, outs[0])
        return _restore_hamza(text, out)




_HAMZA_FORMS = "\u0621\u0622\u0623\u0624\u0625\u0626"   # ء آ أ إ ؤ ئ
_HAMZA_BASE = {
    "\u0622": "\u0627",   # آ -> ا
    "\u0623": "\u0627",   # أ -> ا
    "\u0625": "\u0627",   # إ -> ا
    "\u0624": "\u0648",   # ؤ -> و
    "\u0626": "\u064A",   # ئ -> ي
}


def _restore_hamza(original, diacritized):
    """Safety net for the Rawi model, whose vocabulary has no composed
    hamza letters. Given enough context it reproduces them correctly, but
    a short/isolated word can come back with the hamza flattened
    (سأل -> سَالْ, مؤمن -> مومن). Walk both strings and, wherever the
    original had a hamza form and the output has its bare base letter,
    put the original letter back. Diacritics are left untouched."""
    if not original or not diacritized:
        return diacritized
    if not any(c in _HAMZA_FORMS for c in original):
        return diacritized
    orig_letters = [c for c in original
                    if not ("\u064B" <= c <= "\u0652" or c == "\u0670")]
    out = []
    oi = 0
    for ch in diacritized:
        if "\u064B" <= ch <= "\u0652" or ch == "\u0670":
            out.append(ch)          # a diacritic: keep as-is
            continue
        if ch in ("\u0653", "\u0654", "\u0655"):
            # a stray combining hamza/maddah mark: it belongs to the letter
            # we already restored in composed form, so drop it (otherwise we
            # end up with a doubled hamza such as مُؤَٔمِن).
            continue
        if oi < len(orig_letters):
            o = orig_letters[oi]
            if o in _HAMZA_FORMS and (ch == _HAMZA_BASE.get(o) or ch == o):
                out.append(o)       # restore the hamza form
            else:
                out.append(ch)
            oi += 1
        else:
            out.append(ch)
    # compose (ا + combining hamza -> أ) and drop any leftover marks
    return unicodedata.normalize("NFC", "".join(out))

def _load(name):
    """Load a backend, returning a callable or False."""
    try:
        from . import _libboot
        _libboot.boot()
    except Exception:
        pass

    if name == "libtashkeel":
        # pylibtashkeel is a compiled extension (Windows .pyd). Make sure our
        # lib/ is importable AND registered as a DLL search path, otherwise
        # the extension can fail to resolve its dependencies.
        import sys as _sys
        if _LIB not in _sys.path:
            _sys.path.append(_LIB)
        if hasattr(os, "add_dll_directory") and os.path.isdir(_LIB):
            try:
                os.add_dll_directory(_LIB)
            except Exception:
                pass
        import pylibtashkeel
        # The model may be embedded in the extension; load an external one
        # only if it ships alongside.
        model = os.path.join(_LIB, "libtashkeel_model.bin")
        if os.path.exists(model):
            for loader in ("tashkeel_load", "load", "load_model"):
                fn = getattr(pylibtashkeel, loader, None)
                if callable(fn):
                    try:
                        fn(model)
                        break
                    except Exception:
                        pass

        _fn = None
        for cand in ("tashkeel", "diacritize", "vocalize"):
            f = getattr(pylibtashkeel, cand, None)
            if callable(f):
                _fn = f
                break
        if _fn is None:
            return False

        def _run(text):
            return _fn(text)
        out = _run("نص")     # surface failures now, not mid-speech
        if not isinstance(out, str):
            return False
        return _run

    if name == "rawi":
        path = os.path.join(_VOWELIZERS, "rawi_ensemble.onnx")
        if not os.path.exists(path):
            return False
        r = _Rawi(path)
        r("نص")
        return r

    if name in ("shakkelha", "shakkala"):
        from arabic_vocalizer import vocalize as _v

        def _run(text, _m=name):
            return _v(text, model=_m)
        _run("نص")
        return _run

    return False


def _get(name):
    if name in _cache:
        return _cache[name]
    with _lock:
        if name in _cache:
            return _cache[name]
        try:
            _cache[name] = _load(name)
        except Exception:
            _cache[name] = False
        return _cache[name]


def available():
    """Backends that actually load on this machine (cheap after first call)."""
    return [b for b in BACKENDS
            if b == "off" or _get(b)]


def diacritize_text(text):
    """Diacritize with the selected backend, falling back to any other
    working backend if the chosen one fails. Returns None if none work."""
    if _current == "off" or not text:
        return None
    order = [_current] + [b for b in BACKENDS
                          if b not in (_current, "off")]
    for name in order:
        fn = _get(name)
        if not fn:
            continue
        try:
            out = fn(text)
            if out and any("\u064B" <= c <= "\u0652" for c in out):
                return out
        except Exception:
            _cache[name] = False      # don't retry a broken backend
    return None
