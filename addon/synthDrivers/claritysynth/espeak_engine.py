# -*- coding: UTF-8 -*-
"""A lightweight multilingual formant voice for the ClaritySynth Formant
driver, powered by the bundled eSpeak NG library.

eSpeak NG is a compact formant synthesizer that speaks a hundred-plus
languages — including Arabic — in a very small footprint. The neural driver
already bundles ``espeak-ng.dll`` (used there only as a phonemizer); here we
drive its *audio* synthesis path so the Formant driver gains a real
multilingual voice with its own set of language "makharij" (articulation
points), selectable alongside the NV Speech Player and the built-in engine.

This module is self-contained: it loads the library via ctypes, initialises
it for synchronous synthesis, and renders text to 16-bit PCM through a
sample callback. It never imports the heavy neural stack, so it stays usable
even when onnxruntime is unavailable.
"""

import os
import sys
import ctypes
import threading

try:
    from logHandler import log
except Exception:                       # pragma: no cover - non-NVDA test
    class _L(object):
        def __getattr__(self, n):
            return lambda *a, **k: None
    log = _L()

_here = os.path.dirname(os.path.abspath(__file__))

# eSpeak event / audio constants
_AUDIO_OUTPUT_SYNCHRONOUS = 0x02
_espeakCHARS_UTF8 = 1
_EVENT_LIST_TERMINATED = 0
_EVENT_SAMPLERATE = 1

_lib = None
_sample_rate = 22050
_lock = threading.RLock()
_current_voice = None

# a place for the active render to accumulate samples
_render_buf = None


# The synth callback signature: int (*)(short *wav, int numsamples,
#                                       espeak_EVENT *events)
_SYNTH_CB = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.POINTER(ctypes.c_short), ctypes.c_int,
    ctypes.c_void_p)


def _find_library():
    lib_dir = os.path.join(_here, "lib", "espeakng_loader")
    names = ["espeak-ng.dll", "libespeak-ng.so", "libespeak-ng.dylib",
             "libespeak-ng.so.1"]
    for n in names:
        p = os.path.join(lib_dir, n)
        if os.path.exists(p):
            return lib_dir, p
    return lib_dir, None


@_SYNTH_CB
def _synth_callback(wav, numsamples, events):
    # Copy the delivered samples into the active render buffer. Returning 0
    # means "keep going". eSpeak calls this repeatedly during synthesis.
    global _render_buf
    try:
        if _render_buf is not None and wav and numsamples > 0:
            _render_buf.extend(
                ctypes.cast(
                    wav, ctypes.POINTER(ctypes.c_short * numsamples)
                ).contents)
    except Exception:
        pass
    return 0


def is_available():
    """True if the eSpeak library is present (does not force a load)."""
    _dir, path = _find_library()
    return path is not None


def _ensure_loaded():
    global _lib, _sample_rate
    if _lib is not None:
        return True
    lib_dir, lib_path = _find_library()
    if not lib_path:
        return False
    data_path = os.path.join(lib_dir, "espeak-ng-data")
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(lib_dir)
        except Exception:
            pass
    try:
        lib = ctypes.CDLL(lib_path)
        sr = lib.espeak_Initialize(
            _AUDIO_OUTPUT_SYNCHRONOUS, 0, data_path.encode("utf-8"), 0)
        if sr == -1:
            log.debugWarning("espeak_engine: Initialize failed")
            return False
        _sample_rate = int(sr) if sr and sr > 0 else 22050
        lib.espeak_SetSynthCallback(_synth_callback)
        _lib = lib
        log.info("espeak_engine: eSpeak NG audio engine loaded (%d Hz)"
                 % _sample_rate)
        return True
    except Exception:
        log.debugWarning("espeak_engine: load failed", exc_info=True)
        return False


def sample_rate():
    _ensure_loaded()
    return _sample_rate


def set_voice(voice, variant=None):
    """Set the eSpeak voice/language by name or code (e.g. 'ar', 'en-us',
    'fr'). An optional variant (e.g. 'klatt2', 'Michael') applies a timbre
    preset via eSpeak's '<lang>+<variant>' syntax. Returns True on success."""
    global _current_voice
    if not _ensure_loaded():
        return False
    target = voice
    if variant and variant != "none":
        target = "%s+%s" % (voice, variant)
    if target == _current_voice:
        return True
    try:
        rc = _lib.espeak_SetVoiceByName(target.encode("utf-8"))
        if rc == 0:
            _current_voice = target
            return True
        # if the variant combo failed, fall back to the plain language
        if variant and variant != "none":
            rc = _lib.espeak_SetVoiceByName(voice.encode("utf-8"))
            if rc == 0:
                _current_voice = voice
                return True
    except Exception:
        pass
    return False


def available_languages():
    """ALL languages the bundled eSpeak NG provides, enumerated dynamically
    via espeak_ListVoices so nothing is ever left out. Arabic is floated to
    the top; the rest follow alphabetically by label. Falls back to a small
    built-in list only if enumeration fails."""
    langs = _list_voices_languages()
    if langs:
        # Arabic first, then alphabetical by label
        ar = [t for t in langs if t[0] == "ar" or t[0].startswith("ar-")]
        rest = [t for t in langs if t not in ar]
        rest.sort(key=lambda t: t[1].lower())
        return ar + rest
    return [
        ("ar", "Arabic"), ("en-us", "English (US)"), ("en-gb", "English (UK)"),
        ("fr", "French"), ("es", "Spanish"), ("de", "German"),
        ("it", "Italian"), ("ru", "Russian"), ("tr", "Turkish"),
        ("fa", "Persian"), ("ur", "Urdu"),
    ]


class _ESPEAK_VOICE(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("languages", ctypes.c_char_p),
        ("identifier", ctypes.c_char_p),
        ("gender", ctypes.c_ubyte),
        ("age", ctypes.c_ubyte),
        ("variant", ctypes.c_ubyte),
        ("xx1", ctypes.c_ubyte),
        ("score", ctypes.c_int),
        ("spare", ctypes.c_void_p),
    ]


def _list_voices_languages():
    """Return [(code, label)] for every language voice eSpeak exposes."""
    if not _ensure_loaded():
        return []
    out = []
    seen = set()
    try:
        _lib.espeak_ListVoices.restype = ctypes.POINTER(
            ctypes.POINTER(_ESPEAK_VOICE))
        arr = _lib.espeak_ListVoices(None)
        if not arr:
            return []
        i = 0
        while arr[i]:
            v = arr[i].contents
            i += 1
            try:
                name = v.name.decode("utf-8", "replace") if v.name else ""
                langs = v.languages
                # languages is a packed list: <priority-byte><lang>\0...
                code = ""
                if langs:
                    raw = ctypes.string_at(langs)
                    # first entry: skip the leading priority byte
                    if len(raw) > 1:
                        code = raw[1:].split(b"\x00")[0].decode(
                            "utf-8", "replace")
            except Exception:
                continue
            if not code or code in seen:
                continue
            seen.add(code)
            label = "%s (%s)" % (name, code) if name else code
            out.append((code, label))
    except Exception:
        log.debugWarning("espeak_engine: ListVoices failed", exc_info=True)
        return []
    return out


# eSpeak voice VARIANTS (timbre presets) shipped in the data folder, including
# the Klatt-style formant variants. Selecting one applies via "<lang>+<variant>".
def available_variants():
    """[(id, label)] of voice variants. 'none' = the plain language voice.
    Includes the Klatt variants (klatt..klatt6) and the named ones."""
    out = [("none", "None (default timbre)")]
    try:
        vdir = os.path.join(
            _find_library()[0], "espeak-ng-data", "voices", "!v")
        if os.path.isdir(vdir):
            names = []
            for f in os.listdir(vdir):
                # skip hidden/backup files
                if f.startswith(".") or f.endswith("~"):
                    continue
                names.append(f)
            # Klatt variants first (formant character), then the rest sorted
            klatt = sorted([n for n in names if n.lower().startswith("klatt")])
            other = sorted([n for n in names if not n.lower()
                            .startswith("klatt")], key=str.lower)
            for n in klatt + other:
                out.append((n, n))
    except Exception:
        pass
    return out


def set_rate_wpm(wpm):
    if not _ensure_loaded():
        return
    try:
        # espeakRATE = 1
        _lib.espeak_SetParameter(1, int(max(80, min(450, wpm))), 0)
    except Exception:
        pass


def set_pitch(pitch0to100):
    if not _ensure_loaded():
        return
    try:
        # espeakPITCH = 3
        _lib.espeak_SetParameter(3, int(max(0, min(100, pitch0to100))), 0)
    except Exception:
        pass


def set_volume(vol0to200):
    if not _ensure_loaded():
        return
    try:
        # espeakVOLUME = 2 (0..200)
        _lib.espeak_SetParameter(2, int(max(0, min(200, vol0to200))), 0)
    except Exception:
        pass


def _is_arabic_char(ch):
    return ("\u0600" <= ch <= "\u06FF") or ("\u0750" <= ch <= "\u077F") \
        or ("\uFB50" <= ch <= "\uFDFF") or ("\uFE70" <= ch <= "\uFEFF")


def _is_latin_letter(ch):
    return ("a" <= ch <= "z") or ("A" <= ch <= "Z") \
        or ("\u00C0" <= ch <= "\u024F")   # Latin-1 + extended


def _split_scripts(text):
    """Split text into (run, is_arabic) pieces so each is spoken by the
    right language. Arabic-script runs are Arabic; runs containing Latin
    letters are non-Arabic. Whitespace/neutral characters attach to the
    current run. Returns a list of (text, is_arabic)."""
    runs = []
    cur = []
    cur_ar = None            # None until we see the first letter
    for ch in text:
        if _is_arabic_char(ch):
            k = True
        elif _is_latin_letter(ch):
            k = False
        else:
            # neutral (space, digit, punctuation, symbol): stay in the
            # current run
            cur.append(ch)
            continue
        if cur_ar is None:
            cur_ar = k
            cur.append(ch)
        elif k == cur_ar:
            cur.append(ch)
        else:
            runs.append(("".join(cur), bool(cur_ar)))
            cur = [ch]
            cur_ar = k
    if cur:
        runs.append(("".join(cur), bool(cur_ar) if cur_ar is not None
                     else True))
    return runs


def synth_pcm(text, voice=None, variant=None, rate_wpm=None, pitch=None,
              volume=None, secondary_voice="en-us", split_scripts=True):
    """Render `text` to 16-bit mono PCM bytes at sample_rate(). Returns
    b'' on any failure so the caller can fall back to another engine.

    split_scripts controls how mixed Arabic/Latin text is handled:

    * split_scripts=False (the Formant driver's choice): the WHOLE text —
      Arabic and Latin together — is handed to eSpeak in one call with the
      chosen `voice`. eSpeak NG is genuinely multilingual and reads both
      scripts itself, so nothing is routed to a separate voice. This is what
      a formant voice should do: it is proficient in every language.

    * split_scripts=True: Arabic-script runs are spoken with `voice` and
      Latin runs with `secondary_voice`, concatenated in order. (Kept for any
      caller that specifically wants per-script routing.)"""
    global _render_buf
    if not text or not text.strip():
        return b""
    if not _ensure_loaded():
        return b""

    base_voice = voice or "ar"

    with _lock:
        if rate_wpm is not None:
            set_rate_wpm(rate_wpm)
        if pitch is not None:
            set_pitch(pitch)
        if volume is not None:
            set_volume(volume)

        def _render_one(run_text, run_voice, run_variant):
            global _render_buf
            if not run_text or not run_text.strip():
                return b""
            set_voice(run_voice, run_variant)
            _render_buf = bytearray()
            try:
                data = run_text.encode("utf-8")
                _lib.espeak_Synth(
                    data, len(data) + 1, 0, 0, 0,
                    _espeakCHARS_UTF8, None, None)
                _lib.espeak_Synchronize()
            except Exception:
                log.debugWarning("espeak_engine: synth failed",
                                 exc_info=True)
            out = bytes(_render_buf) if _render_buf else b""
            _render_buf = None
            return out

        has_ar = any(_is_arabic_char(c) for c in text)
        has_latin = any(_is_latin_letter(c) for c in text)

        # Whole-text mode, or single-script text: render in one shot with the
        # chosen voice. eSpeak reads the mixed text itself.
        if not split_scripts or not (has_ar and has_latin):
            return _render_one(text, base_voice, variant)

        # Per-script routing (only when split_scripts=True and both present).
        chunks = []
        for run_text, is_ar in _split_scripts(text):
            if is_ar:
                chunks.append(_render_one(run_text, base_voice, variant))
            else:
                chunks.append(_render_one(run_text, secondary_voice, None))
        return b"".join(chunks)

