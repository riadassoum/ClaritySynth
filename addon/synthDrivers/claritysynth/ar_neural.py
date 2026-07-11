# -*- coding: utf-8 -*-
"""Neural Arabic diacritization tier (Shakkelha / Shakkala via ONNX).

Bundles the MIT-licensed `arabic_vocalizer` package (nipponjo) plus its
ONNX models and the onnxruntime + numpy win_amd64 wheels for NVDA's
CPython 3.13. When these load, Arabic is diacritized by the neural model
(~97-99% word accuracy, correct context/case endings); otherwise this
tier stays dormant and the statistical + Mishkal pipeline handles Arabic.

Credits: Shakkelha (Ali Fadel et al., EMNLP-IJCNLP 2019, MIT); Shakkala
(Barqawiz, MIT); ONNX packaging (nipponjo/arabic_vocalizer, MIT).
"""
import os
import sys
import threading

_here = os.path.dirname(os.path.abspath(__file__))
_vocalize = None
_lock = threading.Lock()
_tried = False
_model = "shakkelha"

NEURAL_AVAILABLE = False


def _try_init():
    global _vocalize, _tried, NEURAL_AVAILABLE
    if _tried:
        return NEURAL_AVAILABLE
    with _lock:
        if _tried:
            return NEURAL_AVAILABLE
        _tried = True
        try:
            import warnings
            warnings.filterwarnings("ignore")
            from . import _libboot
            if not _libboot.boot():
                return False
            # onnxruntime's native .pyd sits in capi/; ensure the folder
            # is importable and DLLs resolve next to it.
            from arabic_vocalizer import vocalize as _v
            # warm one tiny inference so failures surface now, not later
            _ = _v("نص", model=_model)
            _vocalize = _v
            NEURAL_AVAILABLE = True
        except Exception:
            NEURAL_AVAILABLE = False
        return NEURAL_AVAILABLE


def set_model(name):
    global _model
    if name in ("shakkelha", "shakkala"):
        _model = name


def diacritize_text(text):
    """Diacritize a whole sentence/segment with the neural model.
    Returns vocalized text, or None if the tier is dormant/fails."""
    if not _try_init():
        return None
    try:
        out = _vocalize(text, model=_model)
        if out and any("\u064B" <= c <= "\u0652" for c in out):
            return out
    except Exception:
        pass
    return None


def diacritize(word, prev=None, nxt=None):
    """Word-level entry matching the diacritize_hook contract. The neural
    model works best on full context, so per-word calls fall back to
    passing the word plus its neighbours as a mini-context."""
    if not _try_init():
        return None
    seg = word
    if prev or nxt:
        seg = " ".join(x for x in (prev, word, nxt) if x)
    out = diacritize_text(seg)
    if not out:
        return None
    # pull back the middle token corresponding to `word`
    toks = out.split()
    if prev and len(toks) >= 2:
        return toks[1] if len(toks) > (1 if not nxt else 0) else toks[-1]
    return toks[0] if toks else None
