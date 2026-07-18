# -*- coding: utf-8 -*-
"""Discovery of installed voice / vocoder / tashkeel packs for ClaritySynth.

The add-on ships with a bundled Arabic neural voice (mixer128 + vocos) and a
bundled English Piper voice, but it is built to ACCEPT extra packs that the
user downloads into the add-on's data folders. This module scans those
folders so newly-downloaded voices appear in the settings without any code
change.

Layout (all under the add-on's lib/ or a writable data dir):

    piper_voices/<name>.onnx  + <name>.onnx.json   -> secondary (non-Arabic)
    tts_arabic/data/*.onnx                          -> primary (Arabic) models
    vocoders/<name>.onnx                            -> selectable vocoders
    vowelizers/<name>.onnx  + vocab.json            -> tashkeel models

Everything here is pure filesystem scanning — NO model is loaded — so it is
safe to call while NVDA builds its settings dialog on the GUI thread.
"""
import json
import os

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

# A writable location the user (or a future downloader) can drop packs into,
# checked in addition to the bundled lib/. Kept next to the add-on so it
# survives as long as the add-on does.
def _iter_dirs(sub):
    """Yield every folder that may hold this kind of content: the persistent
    user data dir (downloads, outside the add-on so they survive updates)
    first, then the bundled lib/. `sub` is a path relative to each root."""
    try:
        data_paths = _import_data_paths()
        for d in data_paths.search_dirs(sub if sub else ""):
            yield d
        return
    except Exception:
        pass
    # fallback if data_paths is unavailable: bundled lib + legacy drop-in
    for base in (_LIB, os.path.join(_here, "voices")):
        d = os.path.join(base, sub) if sub else base
        if os.path.isdir(d):
            yield d


def is_arabic_voice(stem, lang):
    """True if a Piper voice is Arabic, judged from its language code and,
    as a fallback, its filename stem. Used to keep Arabic Piper voices (e.g.
    Kareem, ar_JO-kareem-*) out of the non-Arabic Secondary list and route
    them to the Primary side instead."""
    if lang and str(lang).lower().startswith("ar"):
        return True
    s = (stem or "").lower()
    # stems look like "ar_JO-kareem-low", "ar-kareem-medium", etc.
    return s.startswith("ar_") or s.startswith("ar-") or "-ar_" in s


def piper_voices():
    """Return a list of (id, label, lang) for every installed Piper voice.

    id is the model filename stem; the .json sidecar (if present) supplies a
    friendly language/name. Bundled voices and downloaded ones are merged."""
    out = []
    seen = set()
    for d in _iter_dirs("piper_voices"):
        try:
            files = os.listdir(d)
        except OSError:
            continue
        for f in files:
            if not f.endswith(".onnx"):
                continue
            stem = f[:-5]
            if stem in seen:
                continue
            seen.add(stem)
            label, lang = stem, None
            j = os.path.join(d, f + ".json")
            if os.path.exists(j):
                try:
                    with open(j, "r", encoding="utf-8") as fh:
                        meta = json.load(fh)
                    lang = (meta.get("language", {}) or {}).get("code") \
                        or meta.get("language_code")
                    dataset = meta.get("dataset") or stem
                    if lang:
                        label = "%s (%s)" % (dataset, lang)
                    else:
                        label = dataset
                except Exception:
                    pass
            out.append((stem, label, lang))
    out.sort(key=lambda t: t[1].lower())
    return out


def arabic_models():
    """Installed Arabic acoustic models (filename stems)."""
    out = []
    seen = set()
    for d in _iter_dirs(os.path.join("tts_arabic", "data")):
        try:
            files = os.listdir(d)
        except OSError:
            continue
        for f in files:
            if f.endswith(".onnx") and "voc" not in f.lower() \
                    and "denois" not in f.lower():
                stem = f[:-5]
                if stem not in seen:
                    seen.add(stem)
                    out.append(stem)
    return sorted(out)


def vocoders():
    """Installed vocoder models (filename stems)."""
    out = []
    seen = set()
    for sub in (os.path.join("tts_arabic", "data"), "vocoders"):
        for d in _iter_dirs(sub):
            try:
                files = os.listdir(d)
            except OSError:
                continue
            for f in files:
                if f.endswith(".onnx") and ("voc" in f.lower()):
                    stem = f[:-5]
                    if stem not in seen:
                        seen.add(stem)
                        out.append(stem)
    return sorted(out)


def tashkeel_models():
    """Installed tashkeel/vowelizer model stems (in addition to the built-in
    libtashkeel + rawi that ar_tashkeel exposes)."""
    out = []
    seen = set()
    for d in _iter_dirs("vowelizers"):
        try:
            files = os.listdir(d)
        except OSError:
            continue
        for f in files:
            if f.endswith(".onnx"):
                stem = f[:-5]
                if stem not in seen:
                    seen.add(stem)
                    out.append(stem)
    return sorted(out)


def has_english():
    """True if any Piper (secondary) voice is installed."""
    return bool(piper_voices())
