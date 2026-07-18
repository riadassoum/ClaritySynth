# -*- coding: utf-8 -*-
"""Arabic diacritization (tashkeel) backends for ClaritySynth.

Two engines are offered:

  * libtashkeel - the libtashkeel engine (compiled), fast and robust
  * rawi        - the Rawi ensemble ONNX diacritizer
  * off         - no automatic diacritization

CRITICAL: listing the backends must NEVER load them. NVDA builds the
synthesizer settings (and the Tashkeel combo box) on its main GUI thread
when the user presses Ctrl+NVDA+S. Loading a 25 MB compiled extension and
ONNX sessions there froze NVDA for many seconds and could take it down
entirely. So `available()` is a pure, static list, and a backend is only
ever loaded lazily, off the GUI thread, the first time it is actually used
to speak.

libtashkeel and the Rawi model are bundled from the NabraTTS add-on by
"pbt", shared by Ilyas Dragonoid.
"""
import json
import os
import re
import threading
import unicodedata

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
_LIB = os.path.join(_here, "lib")
_VOWELIZERS = os.path.join(_LIB, "vowelizers")


def _find_tashkeel_file(fname):
    """Locate a tashkeel model file, checking the persistent user data dir
    (downloads that survive updates) before the bundled locations."""
    cands = []
    try:
        data_paths = _import_data_paths()
        for d in data_paths.search_dirs("vowelizers"):
            cands.append(os.path.join(d, fname))
        for d in data_paths.search_dirs(os.path.join("tts_arabic", "data")):
            cands.append(os.path.join(d, fname))
    except Exception:
        pass
    cands.append(os.path.join(_VOWELIZERS, fname))
    cands.append(os.path.join(_LIB, "tts_arabic", "data", fname))
    for c in cands:
        if os.path.exists(c):
            return c
    return None

BACKENDS = ("libtashkeel", "rawi", "catt", "shakkelha", "shakkala", "off")
DEFAULT_BACKEND = "libtashkeel"

_lock = threading.Lock()
_cache = {}          # backend id -> callable(text) -> str, or False if broken
_current = DEFAULT_BACKEND

_DIAC_RE = re.compile(r"[\u064B-\u065F\u0670\u0640]|[\u0610-\u061A]"
                      r"|[\u06D6-\u06ED]")


def available():
    """The selectable backends. libtashkeel + rawi are always offered
    (bundled); CATT is offered only if its model file is present (it is an
    optional download). This does at most a cheap os.path.exists — it never
    imports an extension or starts an ONNX session, so it is safe to call
    while NVDA builds its settings dialog on the GUI thread."""
    out = ["libtashkeel", "rawi"]
    try:
        if _find_tashkeel_file("catt_eo.onnx"):
            out.append("catt")
    except Exception:
        pass
    # Shakkelha / Shakkala are optional neural diacritizer downloads
    try:
        if _find_tashkeel_file("shakkelha.onnx"):
            out.append("shakkelha")
        if _find_tashkeel_file("shakkala.onnx"):
            out.append("shakkala")
    except Exception:
        pass
    out.append("off")
    return out


def set_backend(name):
    global _current
    if name in BACKENDS:
        _current = name


def get_backend():
    return _current


def _strip(text):
    return _DIAC_RE.sub("", text)


_HAMZA_FORMS = "\u0621\u0622\u0623\u0624\u0625\u0626"   # ء آ أ إ ؤ ئ
_HAMZA_BASE = {
    "\u0622": "\u0627",   # آ -> ا
    "\u0623": "\u0627",   # أ -> ا
    "\u0625": "\u0627",   # إ -> ا
    "\u0624": "\u0648",   # ؤ -> و
    "\u0626": "\u064A",   # ئ -> ي
}


def _restore_hamza(original, diacritized):
    """Safety net for models whose vocabulary has no composed hamza letters
    (Rawi). Given full context they reproduce hamza correctly, but a short
    or isolated word can come back flattened (سأل -> سَالْ, مؤمن -> مومن).
    Walk both strings and, wherever the original had a hamza form and the
    output has its bare base letter, restore the original letter."""
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
            out.append(ch)          # a diacritic: keep
            continue
        if ch in ("\u0653", "\u0654", "\u0655"):
            # stray combining hamza/maddah: the composed letter is already
            # restored, so dropping this avoids a doubled hamza (مُؤَٔمِن).
            continue
        if oi < len(orig_letters):
            o = orig_letters[oi]
            if o in _HAMZA_FORMS and (ch == _HAMZA_BASE.get(o) or ch == o):
                out.append(o)
            else:
                out.append(ch)
            oi += 1
        else:
            out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


class _Rawi(object):
    """Rawi ensemble ONNX diacritizer (adapted from NabraTTS)."""

    def __init__(self, onnx_path):
        vocab_path = os.path.join(os.path.dirname(onnx_path), "vocab.json")
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        self.char_to_idx = vocab["char_to_idx"]
        self.diac_to_idx = vocab["diac_to_idx"]
        self.idx_to_diac = {v: k for k, v in self.diac_to_idx.items()}
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


def _load(name):
    """Load a backend lazily. Returns a callable, or False if unusable.
    Never called from the GUI thread."""
    try:
        from . import _libboot
        _libboot.boot()
    except Exception:
        pass

    if name == "libtashkeel":
        import sys as _sys
        if _LIB not in _sys.path:
            _sys.path.append(_LIB)
        if hasattr(os, "add_dll_directory") and os.path.isdir(_LIB):
            try:
                os.add_dll_directory(_LIB)
            except Exception:
                pass
        import pylibtashkeel
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
        out = _run("نص")            # surface a failure now, not mid-speech
        if not isinstance(out, str):
            return False
        return _run

    if name == "rawi":
        path = _find_tashkeel_file("rawi_ensemble.onnx")
        if not path:
            return False
        r = _Rawi(path)
        r("نص")
        return r

    if name == "catt":
        # CATT is a transformer with its OWN tokenizer — NabraTTS broke it by
        # loading it as a Rawi model. We use the proper CATT class bundled in
        # tts_arabic, so it actually works.
        path = _find_tashkeel_file("catt_eo.onnx")
        if not path:
            return False
        try:
            import sys as _sys
            if _LIB not in _sys.path:
                _sys.path.append(_LIB)
            from tts_arabic.vocalizer.models.catt.network import CATTModel
        except Exception:
            return False
        model = CATTModel(sd_path=path)

        def _run(text):
            return model.predict(text)
        out = _run("نص")
        if not isinstance(out, str):
            return False
        return _run

    if name in ("shakkelha", "shakkala"):
        # neural diacritizers in the arabic_vocalizer package; the model file
        # must be present (bundled or downloaded)
        fname = name + ".onnx"
        if not _find_tashkeel_file(fname):
            return False
        try:
            import sys as _sys
            if _LIB not in _sys.path:
                _sys.path.append(_LIB)
            from arabic_vocalizer import vocalize as _v
        except Exception:
            return False

        def _run(text):
            return _v(text, model=name)
        try:
            probe = _run("نص")
        except Exception:
            return False
        if not isinstance(probe, str):
            return False
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


def preload():
    """Warm the selected backend off the GUI thread, so the first spoken
    word is not delayed by a cold model."""
    if _current != "off":
        _get(_current)


def diacritize_strict(text, backend=None):
    """Fully diacritize `text` using ONLY `backend` (no fallback to a
    different library), exactly the way the speech engine would read it.

    The text is split into contiguous ARABIC runs and non-Arabic runs. Each
    Arabic run is diacritized AS A WHOLE (full sentence context — critical
    for CATT and the neural diacritizers, and the reason word-by-word gave
    incomplete/garbled output). Non-Arabic runs (English, digits, symbols)
    pass through untouched. The result matches what the chosen voice would
    pronounce, in text form. This is what the Tools-menu window uses."""
    if not text:
        return text
    name = backend or _current
    if name == "off":
        return text
    fn = _get(name)
    if not fn:
        return text

    import re
    # split into runs that are either mostly-Arabic or non-Arabic, keeping
    # everything so the output reassembles exactly. A run boundary is where
    # we cross between Arabic-script characters and non-Arabic ones (spaces
    # and Arabic punctuation stay attached to the Arabic side so context is
    # preserved).
    def _is_ar(ch):
        return ("\u0600" <= ch <= "\u06FF") or ("\u0750" <= ch <= "\u077F") \
            or ("\uFB50" <= ch <= "\uFDFF") or ("\uFE70" <= ch <= "\uFEFF")

    runs = []
    cur = []
    cur_ar = None
    for ch in text:
        # whitespace joins whichever run it is in (keeps sentences whole)
        if ch.isspace():
            cur.append(ch)
            continue
        a = _is_ar(ch)
        if cur_ar is None:
            cur_ar = a
            cur.append(ch)
        elif a == cur_ar:
            cur.append(ch)
        else:
            runs.append((cur_ar, "".join(cur)))
            cur = [ch]
            cur_ar = a
    if cur:
        runs.append((bool(cur_ar), "".join(cur)))

    out = []
    for is_ar_run, chunk in runs:
        if not is_ar_run or not chunk.strip():
            out.append(chunk)             # keep non-Arabic exactly
            continue
        try:
            d = fn(chunk)                 # WHOLE run -> full context
        except Exception:
            d = None
        out.append(d if (d and isinstance(d, str) and d.strip()) else chunk)
    return "".join(out)


def diacritize_text(text):
    """Diacritize with the selected backend, falling back to the other
    working one if it fails. Returns None if nothing works."""
    if _current == "off" or not text:
        return None
    order = [_current] + [b for b in BACKENDS if b not in (_current, "off")]
    for name in order:
        fn = _get(name)
        if not fn:
            continue
        try:
            out = fn(text)
            if out and any("\u064B" <= c <= "\u0652" for c in out):
                return out
        except Exception:
            _cache[name] = False      # never retry a broken backend
    return None
