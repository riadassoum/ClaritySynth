# -*- coding: utf-8 -*-
# ClaritySynth: NVDA synthesizer driver.
# A completely self-contained English formant synthesizer — no external
# engines, DLLs, voices, or network access. Text is phonemized by rule
# (g2p.py) and rendered by a pure-Python Klatt-style engine (engine.py).

import os
import threading
import queue
from collections import OrderedDict

import config
import nvwave
import synthDriverHandler

# Setting classes. Newer NVDA exposes these from autoSettingsUtils, older
# builds from driverHandler. DriverSetting (the combo-box setting) and
# StringParameterInfo must be imported on BOTH paths, or the class body of
# SynthDriver raises NameError at import time and the driver never registers.
try:
    from autoSettingsUtils.driverSetting import (NumericDriverSetting,
                                                 BooleanDriverSetting,
                                                 DriverSetting)
except ImportError:
    from driverHandler import (NumericDriverSetting, BooleanDriverSetting,
                               DriverSetting)
try:
    from autoSettingsUtils.utils import StringParameterInfo
except ImportError:
    from driverHandler import StringParameterInfo
from synthDriverHandler import VoiceInfo, synthIndexReached, synthDoneSpeaking
from logHandler import log

# Bind _() to this add-on's translation catalogue. Without this, _() is
# Python's identity builtin and the UI stays English even when a translation
# is installed. Must run at import time, before any _() call.
try:
    import addonHandler
    addonHandler.initTranslation()
except Exception:
    # running outside NVDA (tests) — provide a passthrough so _() exists
    import builtins
    if not hasattr(builtins, "_"):
        builtins._ = lambda s: s


try:
    from speech.commands import (
        IndexCommand,
        CharacterModeCommand,
        PitchCommand,
        BreakCommand,
    )
except ImportError:  # very old NVDA
    from speech import (
        IndexCommand,
        CharacterModeCommand,
        PitchCommand,
        BreakCommand,
    )

from . import engine, g2p

import os
import json


def _loadClonedProfile():
    """If the user drops cloned_voice.wav next to the driver, analyze it
    once and cache the speaker profile; a 'Cloned' voice then appears."""
    here = os.path.dirname(os.path.abspath(__file__))
    wav = os.path.join(here, "cloned_voice.wav")
    cache = os.path.join(here, "cloned_voice.json")
    try:
        if os.path.exists(cache) and (not os.path.exists(wav)
                or os.path.getmtime(cache) >= os.path.getmtime(wav)):
            return json.load(open(cache))
        if os.path.exists(wav):
            from . import voice_profile
            prof = voice_profile.analyze_wav(wav)
            json.dump(prof, open(cache, "w"))
            return prof
    except Exception:
        pass
    return None


_CLONED = _loadClonedProfile()

_neuralTTS = None
_neuralSpeakers = []
_piperEN = None

# ---------------------------------------------------------------------------
# IMPORTANT — do NOT load the neural models here.
#
# NVDA imports EVERY synth driver when the user presses Ctrl+NVDA+S to pick a
# synthesizer, and it does so on the main GUI thread. Initialising the ONNX
# sessions (Piper + the Arabic mixer/vocoder, hundreds of MB) at import time
# therefore froze NVDA for several seconds whenever that dialog was opened,
# and any fault while loading a native library took NVDA down with it — which
# is exactly the "NVDA crashes on Ctrl+NVDA+S until the add-on is removed"
# report.
#
# Instead we only check CHEAPLY (does the model file exist?) so the voice list
# is still correct, and the models themselves are loaded lazily on a worker
# thread the first time the driver is actually used to speak.
# ---------------------------------------------------------------------------

def _neural_models_present():
    """Cheap, file-existence-only probe. No imports, no ONNX, no DLLs."""
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")
    ar = os.path.join(base, "tts_arabic", "data")
    en = os.path.join(base, "piper_voices")
    have_ar = (os.path.isdir(ar)
               and any(f.endswith(".onnx") for f in os.listdir(ar)))
    have_en = (os.path.isdir(en)
               and any(f.endswith(".onnx") for f in os.listdir(en)))
    return have_ar, have_en


_HAVE_AR, _HAVE_EN = (False, False)
try:
    _HAVE_AR, _HAVE_EN = _neural_models_present()
except Exception:
    pass

# Speaker count for the voice list. Known for the bundled multi-speaker
# model; probing it would mean loading the model, which is what we are
# avoiding. Corrected once the model is really loaded.
_neuralSpeakers = [0, 1, 2, 3] if _HAVE_AR else []

_engines_lock = threading.Lock()
_engines_ready = False


def _ensure_engines():
    """Load the neural engines. Called from the speech worker thread (and
    the background warm-up), NEVER from NVDA's GUI thread."""
    global _neuralTTS, _neuralSpeakers, _piperEN, _engines_ready
    if _engines_ready:
        return
    with _engines_lock:
        if _engines_ready:
            return
        _engines_ready = True
        if _HAVE_EN:
            try:
                from . import piper_neural
                if piper_neural._try_init():
                    _piperEN = piper_neural
                    log.info("ClaritySynth: neural English (Piper) active")
            except Exception:
                log.debugWarning("ClaritySynth: Piper English unavailable",
                                 exc_info=True)
        if _HAVE_AR:
            try:
                from . import tts_neural
                if tts_neural._try_init():
                    _neuralTTS = tts_neural
                    _neuralSpeakers = tts_neural.SPEAKERS
                    log.info("ClaritySynth: neural Arabic voice ACTIVE - "
                             "model=%s vocoder=%s speakers=%d"
                             % (tts_neural._model_id,
                                tts_neural._vocoder_id,
                                len(_neuralSpeakers)))
            except Exception:
                log.debugWarning("ClaritySynth: neural Arabic unavailable",
                                 exc_info=True)


_bridge = None
_bridge_tried = False


def _get_bridge():
    """Lazily load the NV Speech Player DLL bridge the first time it is
    actually needed (a formant-fallback synthesis). Loading it at import
    time is unsafe on some setups — notably PORTABLE NVDA — where a native
    DLL load can fault and take NVDA down the instant the synth is selected,
    with no chance to speak a warning. Loading lazily on the worker thread
    keeps synth SELECTION safe; if the DLL is unavailable we simply fall
    back to the pure-Python formant engine."""
    global _bridge, _bridge_tried
    if _bridge_tried:
        return _bridge
    _bridge_tried = True
    try:
        from . import dll_engine
        _bridge = dll_engine.Bridge()
        log.info("ClaritySynth: speechPlayer.dll bridge active")
    except Exception:
        _bridge = None
        log.debugWarning("ClaritySynth: DLL bridge unavailable; pure engine",
                         exc_info=True)
    return _bridge


def _outputDevice():
    try:
        return config.conf["audio"]["outputDevice"]
    except Exception:
        try:
            return config.conf["speech"]["outputDevice"]
        except Exception:
            return None


def _classifyChar(ch):
    """Return 'ar', 'en', or 'neutral' for a character."""
    if "\u0600" <= ch <= "\u06FF" or "\u0750" <= ch <= "\u077F" \
            or "\uFB50" <= ch <= "\uFDFF" or "\uFE70" <= ch <= "\uFEFF":
        return "ar"
    if ("a" <= ch.lower() <= "z"):
        return "en"
    # Arabic-Indic digits count as Arabic; ASCII digits are neutral
    if "\u0660" <= ch <= "\u0669":
        return "ar"
    return "neutral"


def _punctPause(text, sr, is_arabic=False, scale=1.0):
    """Return trailing silence for the punctuation ENDING this chunk.

    The neural ARABIC model flattens ALL punctuation, so Arabic needs
    explicit pauses at commas/colons AND sentence enders. The English
    (Piper) voice renders its own clause phrasing, so English only gets a
    small extra pause at sentence enders to avoid doubling up.

    `scale` comes from the user's "Pause length" setting: 0 removes the
    added pauses entirely, 1.0 is the default, and higher values lengthen
    them."""
    t = text.rstrip()
    if not t:
        return b""
    last = t[-1]
    if is_arabic:
        if last in ".!?\u061f\u06d4":       # sentence enders
            ms = 0.20
        elif last in ":\u061b;":            # colon / Arabic semicolon
            ms = 0.15
        elif last in ",\u060c":             # comma / Arabic comma
            ms = 0.11
        else:
            return b""
    else:
        if last in ".!?":                    # English: sentence enders
            ms = 0.13
        elif last in ",;:":                  # English: clause punctuation
            ms = 0.05
        else:
            return b""
    ms *= max(0.0, scale)
    n = int(sr * ms)
    if n <= 0:
        return b""
    return b"\x00\x00" * n


def _englishClauses(text):
    """Split English into natural clause/sentence units at punctuation
    ONLY, keeping each unit whole so the neural voice does not add
    sentence-boundary intonation mid-phrase and so nothing is chopped
    into meaningless fragments. A unit with no punctuation is yielded
    intact — a short phrase like "This is a test" is spoken as one whole
    thing, never split. Only a very long punctuation-free run is broken,
    and then on a large word boundary so any seam is minimally audible."""
    import re
    parts = re.split(r'(?<=[.!?;:,])\s+|\s*[\u2014]\s*|\n+', text)
    for p in parts:
        if p is None:
            continue
        p = p.strip()
        if not p:
            continue
        if len(p) <= 220:
            yield p
        else:
            words = p.split()
            cur = []
            for w in words:
                cur.append(w)
                if len(" ".join(cur)) >= 200:
                    yield " ".join(cur)
                    cur = []
            if cur:
                yield " ".join(cur)


def _wordGroups(text, n=8):
    """Yield groups of up to n whitespace-separated tokens, so long runs
    (e.g. a pasted URL or address) are synthesized in small pieces and the
    first audio is produced almost immediately instead of after the whole
    run. Very long single tokens are yielded alone."""
    toks = text.split()
    if not toks:
        return
    i = 0
    while i < len(toks):
        yield " ".join(toks[i:i + n])
        i += n


# --- English character names, as exact IPA -----------------------------------
# We give Piper the PHONEMES directly instead of a spelled-out word, because
# eSpeak's letter-to-sound rules mangle short pseudo-words: "ay" came out as
# /ˈaɪ/ ("eye") instead of /ˈeɪ/, and "eff" as /ˌiːˌɛfˈɛf/ ("ee-eff-eff").
# Every symbol below is present in the Piper voice's phoneme map.
_EN_LETTER_IPA = {
    "a": "ˈeɪ",   "b": "bˈiː",  "c": "sˈiː",  "d": "dˈiː",  "e": "ˈiː",
    "f": "ˈɛf",   "g": "dʒˈiː", "h": "ˈeɪtʃ", "i": "ˈaɪ",   "j": "dʒˈeɪ",
    "k": "kˈeɪ",  "l": "ˈɛl",   "m": "ˈɛm",   "n": "ˈɛn",   "o": "ˈoʊ",
    "p": "pˈiː",  "q": "kjˈuː", "r": "ˈɑːɹ",  "s": "ˈɛs",   "t": "tˈiː",
    "u": "jˈuː",  "v": "vˈiː",  "w": "dˈʌbəljˌuː", "x": "ˈɛks",
    "y": "wˈaɪ",  "z": "zˈiː",
}

_EN_DIGIT_IPA = {
    "0": "zˈiəɹoʊ", "1": "wˈʌn",   "2": "tˈuː",   "3": "θɹˈiː",
    "4": "fˈɔːɹ",   "5": "fˈaɪv",  "6": "sˈɪks",  "7": "sˈɛvən",
    "8": "ˈeɪt",    "9": "nˈaɪn",
}

# Punctuation / symbols spoken by name (plain words; eSpeak handles these
# reliably, so we phonemize them at runtime rather than hard-coding IPA).
_EN_SYMBOL_WORD = {
    " ": "space",        ".": "dot",          ",": "comma",
    ";": "semicolon",    ":": "colon",        "?": "question",
    "!": "exclamation",  "'": "apostrophe",   '"': "quote",
    "-": "dash",         "_": "underscore",   "/": "slash",
    "\\": "backslash",   "|": "bar",          "@": "at",
    "#": "number",       "$": "dollar",       "%": "percent",
    "^": "caret",        "&": "and",          "*": "star",
    "(": "left paren",   ")": "right paren",  "[": "left bracket",
    "]": "right bracket", "{": "left brace",  "}": "right brace",
    "<": "less than",    ">": "greater than", "=": "equals",
    "+": "plus",         "~": "tilde",        "`": "backtick",
}


def _englishCharIPA(ch):
    """Exact IPA for a single English letter or digit, or None."""
    if not ch:
        return None
    low = ch.lower()
    if low in _EN_LETTER_IPA:
        return _EN_LETTER_IPA[low]
    if ch in _EN_DIGIT_IPA:
        return _EN_DIGIT_IPA[ch]
    return None


def _englishCharWord(ch):
    """Spoken word for a punctuation/symbol character, or None."""
    return _EN_SYMBOL_WORD.get(ch)


def _isEnglishChar(ch):
    """True if this single character should be spoken by the ENGLISH neural
    voice in character mode (letter, digit, punctuation or symbol)."""
    if not ch:
        return False
    return (_englishCharIPA(ch) is not None
            or _englishCharWord(ch) is not None)


def _englishLetterName(ch):
    """Back-compat: a spoken name for a single English character."""
    low = ch.lower()
    if low in _EN_LETTER_IPA:
        return low
    return _EN_SYMBOL_WORD.get(ch, ch)


def _arabicCharName(ch):
    """Arabic letter -> its spoken name (for character navigation), so the
    neural voice announces e.g. 'أَلِف', 'بَاء'. Falls back to the char
    itself for anything not in the table (diacritics, digits, etc.)."""
    names = {
        "\u0627": "أَلِف", "\u0628": "بَاء", "\u062A": "تَاء",
        "\u062B": "ثَاء", "\u062C": "جِيم", "\u062D": "حَاء",
        "\u062E": "خَاء", "\u062F": "دَال", "\u0630": "ذَال",
        "\u0631": "رَاء", "\u0632": "زَاي", "\u0633": "سِين",
        "\u0634": "شِين", "\u0635": "صَاد", "\u0636": "ضَاد",
        "\u0637": "طَاء", "\u0638": "ظَاء", "\u0639": "عَين",
        "\u063A": "غَين", "\u0641": "فَاء", "\u0642": "قَاف",
        "\u0643": "كَاف", "\u0644": "لَام", "\u0645": "مِيم",
        "\u0646": "نُون", "\u0647": "هَاء", "\u0648": "وَاو",
        "\u064A": "يَاء",
        # each hamza form gets its own descriptive name instead of all
        # saying just "همزة", the way a teacher names them
        "\u0621": "هَمْزَة",                 # ء  bare hamza
        "\u0623": "أَلِف عَلَيْهَا هَمْزَة",   # أ  alef with hamza above
        "\u0625": "أَلِف تَحْتَهَا هَمْزَة",   # إ  alef with hamza below
        "\u0624": "وَاو عَلَيْهَا هَمْزَة",    # ؤ  waw with hamza
        "\u0626": "يَاء عَلَيْهَا هَمْزَة",    # ئ  ya with hamza
        "\u0622": "أَلِف مَدّ", "\u0629": "تَاء مَربُوطَة",
        "\u0649": "أَلِف مَقْصُورَة", "\u0640": "تَطْوِيل",
        # Arabic diacritic (tashkeel) marks — named when read on their own.
        # Merged from a user-supplied fix dictionary; applies to ClaritySynth
        # only (not a global NVDA symbol dictionary).
        "\u064E": "فَتْحَة",              # َ  fatha
        "\u064B": "تَنْوِينُ الْفَتْحِ",    # ً  tanween fath
        "\u064F": "ضَمَّة",               # ُ  damma
        "\u064C": "تَنْوِينُ الضَّمِّ",     # ٌ  tanween damm
        "\u0650": "كَسْرَة",              # ِ  kasra
        "\u064D": "تَنْوِينُ الْكَسْرِ",    # ٍ  tanween kasr
        "\u0652": "سُكُون",               # ْ  sukoon
        "\u0651": "شَدَّة",               # ّ  shadda
        "\u0670": "أَلِف خَنْجَرِيَّة",      # ٰ  superscript (dagger) alef
    }
    return names.get(ch, ch)


def _splitByScript(text):
    """Yield (segment, is_arabic) runs. Arabic letters/marks/Arabic-digits
    group as Arabic; Latin letters group as English. Neutral characters
    (spaces, ASCII digits, punctuation, symbols like + / = *) attach to an
    adjacent run — but a neutral chunk containing ASCII letters/symbols
    that is NOT purely whitespace is routed to the ENGLISH/formant side so
    the neural Arabic voice never pronounces things like '++' or '/'.
    Pure-whitespace neutrals attach to the preceding run."""
    # first, tokenize into (text, kind) atoms
    atoms = []
    cur = ""
    cur_k = None
    for ch in text:
        k = _classifyChar(ch)
        if cur_k is None:
            cur_k = k
            cur = ch
        elif k == cur_k:
            cur += ch
        else:
            atoms.append((cur, cur_k))
            cur = ch
            cur_k = k
    if cur:
        atoms.append((cur, cur_k))

    # resolve neutrals: whitespace-only merges into the previous run;
    # symbol/digit neutrals go to English (formant) unless surrounded by
    # Arabic on both sides with no symbols (then they stay Arabic).
    runs = []  # list of [text, is_ar]

    def _push(txt, is_ar):
        if runs and runs[-1][1] == is_ar:
            runs[-1][0] += txt
        else:
            runs.append([txt, is_ar])

    def _neighbour_ar(idx):
        """True if the nearest lettered atom (either direction) is Arabic.
        Looks right first (numbers usually modify what follows: '13 كتاب'),
        then left."""
        for j in range(idx + 1, len(atoms)):
            if atoms[j][1] in ("ar", "en"):
                return atoms[j][1] == "ar"
        for j in range(idx - 1, -1, -1):
            if atoms[j][1] in ("ar", "en"):
                return atoms[j][1] == "ar"
        return False

    import re
    for i, (txt, k) in enumerate(atoms):
        if k == "ar":
            _push(txt, True)
        elif k == "en":
            _push(txt, False)
        else:  # neutral: whitespace, ASCII digits, and/or symbols
            if txt.strip() == "":
                if runs:
                    runs[-1][0] += txt
                else:
                    _push(txt, False)
                continue
            has_digit = any(c.isdigit() for c in txt)
            has_symbol = any(c in "+*/=<>&%#@^~|\\" for c in txt)
            if has_digit and not has_symbol:
                # a number (possibly with . , : - and spaces). Route to the
                # side of its lettered neighbour so Arabic numbers are read
                # in Arabic and English numbers in English.
                _push(txt, _neighbour_ar(i))
            elif has_symbol:
                # real symbols (++, /, =) -> formant engine says them right
                _push(txt, False)
            else:
                # only punctuation/spaces (، . - etc). Attach to the
                # neighbour's side so it doesn't split a phrase oddly.
                _push(txt, _neighbour_ar(i))
    return [(t, bool(a)) for t, a in runs]


# punctuation that should ALWAYS end a chunk (so a pause + intonation reset
# happens there): sentence enders and clause separators, Arabic and Latin.
_BREAK_PUNCT = ".!?\u061f\u06d4:\u061b;,\u060c"


def _neuralChunks(text, limit=180):
    """Yield clause-sized pieces of (already diacritized) text.

    A chunk boundary is placed after every sentence/clause punctuation mark
    (. ! ? \u061f \u06d4 : \u061b ; , \u060c) so the caller can insert a
    real pause and the model restarts intonation there — this is what makes
    ':' and '?' actually pause. Punctuation-free text is kept whole (never
    chopped mid-phrase); only a genuinely over-long run with no punctuation
    is split on a word boundary as a last resort."""
    import re
    # split AFTER any break punctuation, keeping the mark attached to the
    # left piece; also split on newlines and quote guillemets
    parts = re.split(
        r'(?<=[' + _BREAK_PUNCT + r'\u00bb\u00ab])\s+|\n+', text)
    for p in parts:
        if p is None:
            continue
        p = p.strip()
        if not p:
            continue
        if len(p) <= limit:
            yield p
        else:
            # over-long punctuation-free run: wrap on word boundaries
            while len(p) > limit:
                cut = p.rfind(" ", 0, limit)
                if cut <= 0:
                    cut = limit
                yield p[:cut].strip()
                p = p[cut:].strip()
            if p:
                yield p


class SynthDriver(synthDriverHandler.SynthDriver):
    name = "claritysynth"
    # Translators: description of the ClaritySynth synthesizer.
    description = _("ClaritySynth Neural")

    supportedSettings = (
        DriverSetting("primaryEngine",
                      _("Primary voice &engine"),
                      availableInSettingsRing=True,
                      defaultVal="mixer",
                      displayName=_("Primary engine")),
        synthDriverHandler.SynthDriver.VoiceSetting(),
        DriverSetting("primaryVariant",
                      _("Primary voice &variant (quality/speed)"),
                      availableInSettingsRing=True,
                      defaultVal="auto",
                      displayName=_("Primary variant")),
        DriverSetting("vocoder",
                      _("Primary voice v&ocoder (Mixer only)"),
                      availableInSettingsRing=True,
                      defaultVal="auto",
                      displayName=_("Vocoder")),
        DriverSetting("secondaryVoice",
                      _("&Secondary voice (non-Arabic: English, etc.)"),
                      availableInSettingsRing=True,
                      defaultVal="auto",
                      displayName=_("Secondary voice")),
        DriverSetting("secondaryVariant",
                      _("Secondary voice v&ariant (quality/speed)"),
                      availableInSettingsRing=True,
                      defaultVal="auto",
                      displayName=_("Secondary variant")),
        DriverSetting("tashkeel",
                      _("&Tashkeel library (Arabic diacritization)"),
                      availableInSettingsRing=True,
                      defaultVal="libtashkeel",
                      displayName=_("Tashkeel library")),
        synthDriverHandler.SynthDriver.RateSetting(),
        BooleanDriverSetting("rateBoost",
                             _("Rate boo&st (extra-fast speech)"),
                             defaultVal=False,
                             availableInSettingsRing=True),
        synthDriverHandler.SynthDriver.PitchSetting(),
        synthDriverHandler.SynthDriver.VolumeSetting(),
        # Note: inflection, breathiness, roughness, head size and stress
        # emphasis are formant-synthesis parameters with no effect on the
        # neural voices, so they are intentionally NOT exposed here (they
        # remain on the ClaritySynth Formant driver, where they do work).
        NumericDriverSetting("clarity",
                             _("&Clarity (noise reduction)"),
                             defaultVal=5, minStep=1),
        NumericDriverSetting("pauseLength", _("Pause &length"),
                             defaultVal=40, availableInSettingsRing=True),
        BooleanDriverSetting("tanweenPause",
                             _("Pronounce &tanween on isolated words"),
                             defaultVal=False),
    )
    supportedCommands = {
        IndexCommand,
        CharacterModeCommand,
        PitchCommand,
        BreakCommand,
    }
    supportedNotifications = {synthIndexReached, synthDoneSpeaking}

    @classmethod
    def check(cls):
        return True

    def __init__(self):
        super().__init__()
        self._rate = 50
        self._pitch = 50
        self._inflection = 60
        self._volume = 90
        # default to the neural Arabic voice whenever the models are
        # installed (decided by the cheap probe, not by loading them)
        # always default to a neural voice id; the neural engine loads
        # lazily and there is no "adam" voice on the neural synth anymore
        self._voice = "neural0"
        self._primaryEngine = "mixer"
        self._vocoder = "auto"
        self._secondaryVoice = "auto"
        self._primaryVariant = "auto"
        self._secondaryVariant = "auto"
        self._breathiness = 6
        self._roughness = 18
        self._headSize = 50
        self._stressEmphasis = 50
        self._pauseLength = 40
        self._clarity = 5          # -> denoise strength 0.025 (see _denoise)
        self._tanweenPause = False
        self._neuralArabic = True
        self._rateBoost = False
        # default tashkeel backend (libtashkeel, with automatic fallback)
        try:
            from . import ar_tashkeel
            ar_tashkeel.set_backend(ar_tashkeel.DEFAULT_BACKEND)
        except Exception:
            pass
        self._cancelFlag = threading.Event()
        self._gen = 0            # bumped on each cancel; guards stale audio
        self._queue = queue.Queue()
        self._player = None
        self._makePlayer()
        self._thread = threading.Thread(
            target=self._worker, name="ClaritySynthWorker", daemon=True
        )
        self._thread.start()
        # Preload/warm both neural voices off-thread so the first real
        # utterance (even a single character) is instant and not "sloppy"
        # from a cold model. Keeps strong refs alive too.
        self._warm = (_neuralTTS, _piperEN)
        threading.Thread(target=self._warmup, name="ClaritySynthWarm",
                         daemon=True).start()

    def _warmup(self):
        # Load the neural engines and the tashkeel backend HERE — on a
        # background thread — so NVDA's GUI thread never does it.
        try:
            _ensure_engines()
        except Exception:
            pass
        try:
            from . import ar_tashkeel
            ar_tashkeel.preload()
        except Exception:
            pass
        """Synthesize tiny throwaway utterances to make both neural models
        hot. Silent: results are discarded, nothing is fed to a player."""
        try:
            if _neuralTTS:
                for _ in range(2):
                    _neuralTTS.synth_wave("نَعَم", speaker=0, pace=1.0,
                                          volume=1.0)
        except Exception:
            pass
        try:
            if _piperEN:
                for _ in range(2):
                    _piperEN.synth_wave("ok", length_scale=1.0, volume=1.0)
        except Exception:
            pass

    def _makePlayer(self):
        kwargs = dict(
            channels=1,
            samplesPerSec=engine.SR,
            bitsPerSample=16,
        )
        device = _outputDevice()
        try:
            if device is not None:
                self._player = nvwave.WavePlayer(outputDevice=device, **kwargs)
            else:
                self._player = nvwave.WavePlayer(**kwargs)
        except TypeError:
            # Older/newer signature mismatch: fall back to positional basics
            self._player = nvwave.WavePlayer(1, engine.SR, 16)

    def terminate(self):
        self.cancel()
        self._queue.put(None)
        self._thread.join(timeout=2.0)
        if self._player:
            try:
                self._player.close()
            except Exception:
                pass
            self._player = None

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------
    def _get_voice(self):
        return self._voice

    def _set_voice(self, value):
        if value in self.availableVoices:
            self._voice = value

    def _get_availableSecondaryVoices(self):
        """The installed non-Arabic (Piper) voices, plus an Auto entry.
        Pure filesystem scan — safe on the GUI thread, and new downloaded
        voices appear here automatically."""
        out = OrderedDict()
        out["auto"] = StringParameterInfo("auto", _("Auto (default English)"))
        try:
            from . import voice_packs
            for vid, label, lang in voice_packs.piper_voices():
                # Arabic Piper voices (e.g. Kareem) belong on the PRIMARY
                # side, never here — skip them.
                if voice_packs.is_arabic_voice(vid, lang):
                    continue
                out[vid] = StringParameterInfo(vid, label)
        except Exception:
            pass
        return out

    # ---- casing aliases so NVDA's capitalize()-based name derivation also
    # ---- resolves (prevents 'no attribute availablePrimaryvariants' crash)
    def _get_availablePrimaryvariants(self):
        return self._get_availablePrimaryVariants()

    def _get_availableSecondaryvariants(self):
        return self._get_availableSecondaryVariants()

    def _get_availableSecondaryvoices(self):
        return self._get_availableSecondaryVoices()

    # ---- Primary voice engine: choose between the Mixer (multi-speaker
    # ---- Arabic acoustic models) and Piper (Arabic Piper voices like
    # ---- Kareem). Keeping them as separate engines — rather than mixing
    # ---- Piper voices into the Mixer variant list — means a Piper voice
    # ---- always goes through the same (correct-pitch) Piper path.
    def _get_availablePrimaryEngines(self):
        out = OrderedDict()
        out["mixer"] = StringParameterInfo(
            "mixer", _("Mixer (multi-speaker Arabic)"))
        # only offer Piper if at least one Arabic Piper voice is installed
        try:
            from . import voice_packs
            if any(voice_packs.is_arabic_voice(vid, lang)
                   for vid, label, lang in voice_packs.piper_voices()):
                out["piper"] = StringParameterInfo(
                    "piper", _("Piper (Arabic Piper voices)"))
        except Exception:
            pass
        return out

    def _get_availablePrimaryengines(self):
        return self._get_availablePrimaryEngines()

    def _get_primaryEngine(self):
        return getattr(self, "_primaryEngine", "mixer")

    def _set_primaryEngine(self, value):
        if value not in ("mixer", "piper"):
            value = "mixer"
        changed = (value != getattr(self, "_primaryEngine", "mixer"))
        self._primaryEngine = value
        # The Voice list depends on the engine, but NVDA caches it in
        # self._availableVoices. Invalidate that cache so the next read
        # rebuilds the list for the new engine (Mixer speakers vs Arabic
        # Piper voices). Also pick a valid default voice for the new engine.
        if changed:
            try:
                if hasattr(self, "_availableVoices"):
                    del self._availableVoices
            except Exception:
                pass
        try:
            voices = self.availableVoices
            if self._voice not in voices and voices:
                self._voice = next(iter(voices))
        except Exception:
            pass

    # ---- Vocoder (Mixer engine only). Lets the user make sure the vocoder
    # ---- matches the model so audio is rendered correctly.
    def _get_availableVocoders(self):
        out = OrderedDict()
        out["auto"] = StringParameterInfo("auto", _("Auto (recommended)"))
        # map an installed vocoder file stem to the model id + a clean label
        stem_map = {"vocos22": ("vocos", _("Vocos 22 kHz (recommended)")),
                    "vocos44": ("vocos44", _("Vocos 44 kHz (higher fidelity)")),
                    "hifigan": ("hifigan", _("HiFi-GAN"))}
        try:
            from . import voice_packs
            seen = set()
            for stem in voice_packs.vocoders():
                vid, label = stem_map.get(
                    stem, (stem, stem))
                if vid not in seen:
                    seen.add(vid)
                    out[vid] = StringParameterInfo(vid, label)
        except Exception:
            pass
        return out

    def _get_availableVocoders_alias(self):
        return self._get_availableVocoders()

    def _get_vocoder(self):
        return getattr(self, "_vocoder", "auto")

    def _set_vocoder(self, value):
        self._vocoder = value
        try:
            from . import tts_neural
            if hasattr(tts_neural, "select_vocoder"):
                tts_neural.select_vocoder(None if value == "auto" else value)
        except Exception:
            pass

    def _get_availablePrimaryVariants(self):
        """Quality/speed variants for the primary voice, depending on the
        selected primary engine:
          * Mixer  -> the installed Arabic acoustic models
                      (mixer128 / mixer80 / fastpitch).
          * Piper  -> the quality tiers of the chosen Arabic Piper voice.
        Pure filesystem scan (safe on the GUI thread)."""
        out = OrderedDict()
        engine = getattr(self, "_primaryEngine", "mixer")

        if engine == "piper":
            # Under the Piper engine each voice/tier is already a distinct
            # entry in the Voice list (e.g. Kareem low vs Kareem medium), so
            # a separate variant is not applicable here.
            out["auto"] = StringParameterInfo(
                "auto", _("Not applicable (choose the voice above)"))
            return out

        # Mixer engine: the installed Arabic acoustic models
        out["auto"] = StringParameterInfo("auto", _("Auto (best installed)"))
        labels = {"mixer128": _("Mixer 128 (standard, 4 speakers)"),
                  "mixer80": _("Mixer 80 (faster)"),
                  "fastpitch": _("FastPitch (high quality)")}
        try:
            from . import voice_packs
            for mid in voice_packs.arabic_models():
                out[mid] = StringParameterInfo(mid, labels.get(mid, mid))
        except Exception:
            pass
        return out

    def _get_primaryVariant(self):
        return getattr(self, "_primaryVariant", "auto")

    def _set_primaryVariant(self, value):
        self._primaryVariant = value
        try:
            from . import tts_neural
            if value == "auto":
                if hasattr(tts_neural, "select_model"):
                    tts_neural.select_model(None)
            elif value.startswith("piper:"):
                # Arabic Piper voice as primary — handled at speak time
                pass
            elif hasattr(tts_neural, "select_model"):
                tts_neural.select_model(value)
        except Exception:
            pass

    def _get_availableSecondaryVariants(self):
        """Quality variants for the non-Arabic Piper voice. Piper voices come
        in quality tiers (x_low/low = fastest, medium/high = slower but
        clearer). Lists the tiers available for the currently installed
        secondary voices."""
        out = OrderedDict()
        out["auto"] = StringParameterInfo(
            "auto", _("Auto (prefer fastest)"))
        out["fast"] = StringParameterInfo(
            "fast", _("Fast (low quality, lowest latency)"))
        out["standard"] = StringParameterInfo(
            "standard", _("Standard (higher quality, slower)"))
        return out

    def _get_secondaryVariant(self):
        return getattr(self, "_secondaryVariant", "auto")

    def _set_secondaryVariant(self, value):
        self._secondaryVariant = value
        try:
            from . import piper_neural
            if hasattr(piper_neural, "select_variant"):
                piper_neural.select_variant(value)
        except Exception:
            pass

    def _get_secondaryVoice(self):
        return getattr(self, "_secondaryVoice", "auto")

    def _set_secondaryVoice(self, value):
        # the secondary voice is the NON-Arabic voice; never accept an Arabic
        # Piper voice here (it belongs on the primary side)
        try:
            from . import voice_packs
            if value and value != "auto" and \
                    voice_packs.is_arabic_voice(value, None):
                value = "auto"
        except Exception:
            pass
        self._secondaryVoice = value
        # tell the Piper layer which voice model to use ("auto" = bundled)
        try:
            from . import piper_neural
            if hasattr(piper_neural, "select_voice"):
                piper_neural.select_voice(None if value == "auto" else value)
        except Exception:
            pass

    def _getAvailableVoices(self):
        """The neural Arabic voices for the CURRENTLY selected primary engine.

        Called while NVDA builds the settings dialog on its GUI thread, so it
        must NOT load the models — the voices come from cheap file-existence
        probes; the models are loaded lazily by the speech worker.

        * Mixer engine -> the multi-speaker Arabic acoustic model's speakers.
        * Piper engine -> the installed Arabic Piper voices (e.g. Kareem).

        The formant voices live in the separate "ClaritySynth Formant"
        driver. English is spoken automatically by the secondary voice.
        """
        voices = OrderedDict()
        engine = getattr(self, "_primaryEngine", "mixer")

        if engine == "piper":
            # Arabic Piper voices become the primary voice list
            try:
                from . import voice_packs
                for vid, label, lang in voice_packs.piper_voices():
                    if voice_packs.is_arabic_voice(vid, lang):
                        voices["piper:" + vid] = VoiceInfo(
                            "piper:" + vid, label, "ar")
            except Exception:
                pass
            if not voices:
                # engine set to piper but no Arabic Piper voice installed:
                # show a clear placeholder rather than nothing
                voices["piper:none"] = VoiceInfo(
                    "piper:none", _("(no Arabic Piper voice installed)"), "ar")
            return voices

        # Mixer engine: speakers of the installed Arabic acoustic model
        speakers = _neuralSpeakers or ([0, 1, 2, 3] if _HAVE_AR else [])
        if speakers:
            label = "Std"
            if _neuralTTS is not None:
                q = getattr(_neuralTTS, "_model_id", None) or "neural"
                label = {"fastpitch": "HQ", "mixer128": "Std",
                         "mixer80": "Fast"}.get(q, q)
            for s in speakers:
                if len(speakers) > 1:
                    nm = _("Arabic Neural %s - Speaker %d") % (label, s + 1)
                else:
                    nm = _("Arabic Neural %s") % label
                voices["neural%d" % s] = VoiceInfo("neural%d" % s, nm, "ar")
        if not voices:
            if _HAVE_AR:
                for s in (0, 1, 2, 3):
                    nm = _("Arabic Neural - Speaker %d") % (s + 1)
                    voices["neural%d" % s] = VoiceInfo(
                        "neural%d" % s, nm, "ar")
            else:
                voices["neural0"] = VoiceInfo(
                    "neural0", _("Arabic Neural"), "ar")
        return voices

    def _get_rate(self):
        return self._rate

    def _set_rate(self, value):
        self._rate = max(0, min(100, value))

    def _get_pitch(self):
        return self._pitch

    def _set_pitch(self, value):
        self._pitch = max(0, min(100, value))

    def _get_inflection(self):
        return self._inflection

    def _set_inflection(self, value):
        self._inflection = max(0, min(100, value))

    def _get_volume(self):
        return self._volume

    def _set_volume(self, value):
        self._volume = max(0, min(100, value))

    def _get_breathiness(self):
        return self._breathiness

    def _set_breathiness(self, value):
        self._breathiness = max(0, min(100, value))

    def _get_roughness(self):
        return self._roughness

    def _set_roughness(self, value):
        self._roughness = max(0, min(100, value))

    def _get_headSize(self):
        return self._headSize

    def _set_headSize(self, value):
        self._headSize = max(0, min(100, value))

    def _get_stressEmphasis(self):
        return self._stressEmphasis

    def _set_stressEmphasis(self, value):
        self._stressEmphasis = max(0, min(100, value))

    def _get_clarity(self):
        return self._clarity

    def _set_clarity(self, value):
        self._clarity = max(0, min(100, int(value)))

    def _denoise(self):
        """Map the Clarity setting (0..100) to the vocoder's denoise
        strength. 5 -> 0.025, which is the value the NabraTTS add-on uses
        and which audibly lowers the noise floor of the Arabic voice."""
        return self._clarity / 200.0

    def _volumeGain(self):
        """Volume 0..100 -> output gain. The baseline (normalize_rms) is
        already a healthy level, so this scales around it and lets 100 push
        genuinely louder into the soft limiter (which prevents clipping)."""
        v = max(0, min(100, self._volume)) / 100.0
        # 100 -> 1.15x. Kept modest on purpose: the baseline (normalize_rms)
        # is already loud, and pushing harder crushes speech into the soft
        # limiter's knee, which sounds saturated/distorted. 1.15x stays out
        # of the knee while still being clearly louder at the top.
        return v * 1.15

    def _pauseScale(self):
        """Map the Pause length setting (0..100, default 40) to a multiplier
        for the pauses inserted between neural clauses. 0 -> no added
        pauses, 40 -> 1.0 (default), 100 -> 2.5x."""
        return (self._pauseLength / 40.0) if self._pauseLength <= 40 else \
            (1.0 + (self._pauseLength - 40) / 60.0 * 1.5)

    def _get_pauseLength(self):
        return self._pauseLength

    def _set_pauseLength(self, value):
        self._pauseLength = max(0, min(100, value))

    def _get_availableTashkeels(self):
        """Populate the Tashkeel combo box with the libraries that actually
        load on this machine (a backend whose binary/model is missing is
        simply not offered)."""
        labels = {
            "libtashkeel": _("Libtashkeel (recommended)"),
            "rawi": _("Rawi ensemble"),
            "catt": _("CATT"),
            "shakkelha": _("Shakkelha (neural)"),
            "shakkala": _("Shakkala (neural)"),
            "off": _("Off (read text as written)"),
        }
        out = OrderedDict()
        try:
            from . import ar_tashkeel
            names = ar_tashkeel.available()
        except Exception:
            names = ["off"]
        for n in names:
            out[n] = StringParameterInfo(n, labels.get(n, n))
        if not out:
            out["off"] = StringParameterInfo("off", labels["off"])
        return out

    def _get_tashkeel(self):
        try:
            from . import ar_tashkeel
            return ar_tashkeel.get_backend()
        except Exception:
            return "off"

    def _set_tashkeel(self, value):
        try:
            from . import ar_tashkeel, ar_g2p
            ar_tashkeel.set_backend(value)
            # keep the legacy neural flag in step with the selection
            ar_g2p._NEURAL_ON = (value != "off")
        except Exception:
            pass

    def _get_rateBoost(self):
        return self._rateBoost

    def _set_rateBoost(self, value):
        self._rateBoost = bool(value)

    def _get_neuralArabic(self):
        return self._neuralArabic

    def _set_neuralArabic(self, value):
        self._neuralArabic = value
        try:
            from . import ar_g2p, ar_neural
            ar_g2p._NEURAL_ON = bool(value)
        except Exception:
            pass

    def _get_tanweenPause(self):
        return self._tanweenPause

    def _set_tanweenPause(self, value):
        self._tanweenPause = value
        try:
            from . import ar_g2p
            ar_g2p.pronounce_tanween_pause = bool(value)
        except Exception:
            pass

    def _durationScale(self):
        # Formant engine duration multiplier.
        # rate 0 -> 2.2x durations (slow), 50 -> 1.0x, 100 -> 0.42x
        return 2.2 * ((0.42 / 2.2) ** (self._rate / 100.0))

    def _neuralLengthScale(self):
        """length_scale for the neural voices, kept ALWAYS in the model's
        high-quality zone (1.45 slow .. 0.90 mildly fast). We never push
        the model hard enough to drop phonemes; extra speed is delivered
        by OLA time-compression instead (see _speedFactor)."""
        r = self._rate / 100.0
        # gentle: 1.45 (slow, clear) -> 1.0 (natural ~rate45) -> 0.90 fast
        ls = 1.45 * ((0.90 / 1.45) ** r)
        return max(0.85, min(1.6, ls))

    def _pitchSemitones(self, offset=0):
        """Map NVDA pitch (0..100, 50=neutral) plus any command offset to a
        semitone shift applied identically to BOTH neural voices, so the
        pitch slider behaves the same everywhere. Range about -7..+7."""
        p = max(0, min(100, self._pitch + offset))
        return (p - 50) / 50.0 * 7.0

    def _speedFactor(self):
        """Post-synthesis OLA compression factor (>1 = faster) that gives
        reliable fast speech with NO phoneme loss. The model already
        provides up to ~1.1x via length_scale; this adds the rest.

        Total target speed by rate (at boost 0):
          rate 50 -> ~1.0x, rate 75 -> ~1.35x, rate 100 -> ~1.8x
        Rate boost raises the ceiling substantially (up to ~3.3x)."""
        r = self._rate / 100.0
        # model already gives ~ (1/length_scale) of speed; compute the
        # residual needed to reach the target, then apply as compression.
        model_speed = 1.0 / self._neuralLengthScale()
        top = 3.3 if self._rateBoost else 1.8   # checkbox: on=fast
        # target total speed grows with rate; below mid rate no boost
        target = 1.0 * ((top / 1.0) ** max(0.0, (r - 0.45) / 0.55))
        if r <= 0.45:
            target = model_speed   # let the (slightly slow) model handle it
        factor = target / model_speed
        return max(1.0, min(3.5, factor))

    def _baseF0(self, pitchOffset=0):
        p = max(0, min(100, self._pitch + pitchOffset))
        f0 = 62.0 * (2.0 ** (p / 100.0 * 1.5))  # ~62..175 Hz, mid ~104
        if self._voice == "clara":
            f0 *= 1.75
        elif self._voice == "cloned" and _CLONED:
            f0 = _CLONED["base_f0"] * (2.0 ** ((p - 50) / 100.0))
        return f0

    def _inflectionValue(self):
        if self._voice == "robby":
            return 0.0
        return self._inflection / 100.0

    def _formantScale(self):
        base = 1.0
        if self._voice == "cloned" and _CLONED:
            base = _CLONED["fscale"]
        elif self._voice == "clara":
            base = 1.15
        elif self._voice == "robby":
            base = 0.97
        # head size 0 -> big head (low formants), 100 -> small head
        return base * (1.18 - 0.33 * self._headSize / 100.0)

    # ------------------------------------------------------------------
    # Speech
    # ------------------------------------------------------------------
    def speak(self, speechSequence):
        items = []
        charMode = False
        pitchOffset = 0
        for item in speechSequence:
            if isinstance(item, str):
                items.append(("text", item, charMode, pitchOffset))
            elif isinstance(item, IndexCommand):
                items.append(("index", item.index))
            elif isinstance(item, CharacterModeCommand):
                charMode = item.state
            elif isinstance(item, PitchCommand):
                pitchOffset = getattr(item, "offset", 0)
            elif isinstance(item, BreakCommand):
                items.append(("break", getattr(item, "time", 50)))
        self._queue.put(items)

    def cancel(self):
        self._cancelFlag.set()
        self._gen += 1
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        if self._player:
            try:
                self._player.stop()
            except Exception:
                pass
        # also stop the neural voice player (Ctrl / interrupt must work
        # while Arabic neural speech is playing)
        np = getattr(self, "_neuralPlayer", None)
        if np:
            try:
                np.stop()
            except Exception:
                pass
        pp = getattr(self, "_piperPlayer", None)
        if pp:
            try:
                pp.stop()
            except Exception:
                pass

    def pause(self, switch):
        if self._player:
            try:
                self._player.pause(switch)
            except Exception:
                pass
        np = getattr(self, "_neuralPlayer", None)
        if np:
            try:
                np.pause(switch)
            except Exception:
                pass
        pp = getattr(self, "_piperPlayer", None)
        if pp:
            try:
                pp.pause(switch)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------
    def _notifyIndex(self, index):
        try:
            synthIndexReached.notify(synth=self, index=index)
        except Exception:
            log.error("ClaritySynth: index notify failed", exc_info=True)

    def _feed(self, data):
        try:
            self._player.feed(data)
        except Exception:
            log.debugWarning("ClaritySynth: feed failed", exc_info=True)

    def _speakMixedNeural(self, text, pitchOffset=0, pre_diacritized=False):
        """Speak mixed Arabic/English with NO inter-segment latency:
        pre-synthesize every run to PCM (unified sample rate), then play
        the concatenated audio through a single player. Arabic uses the
        neural Arabic voice; English uses Piper if available, else formant
        rendered to PCM. Returns True if handled."""
        my_gen = self._gen
        cancelled = lambda: (self._cancelFlag.is_set()
                             or self._gen != my_gen)
        try:
            from . import timescale
            import numpy as np
        except Exception:
            return False
        # unified output sample rate = the Arabic neural rate (22.05k)
        try:
            out_sr = _neuralTTS.sample_rate()
        except Exception:
            out_sr = 22050
        # ensure the neural player is at the unified rate
        if getattr(self, "_neuralSR", None) != out_sr:
            self._neuralPlayer = nvwave.WavePlayer(
                channels=1, samplesPerSec=out_sr, bitsPerSample=16)
            self._neuralSR = out_sr

        def _resample(arr, seg_sr):
            if seg_sr == out_sr or not arr.size:
                return arr
            ratio = out_sr / float(seg_sr)
            idx = np.linspace(0, arr.size - 1,
                              int(arr.size * ratio)).astype(np.float32)
            i0 = np.floor(idx).astype(np.int32)
            i1 = np.minimum(i0 + 1, arr.size - 1)
            fr = idx - i0
            return (arr[i0] * (1 - fr) + arr[i1] * fr).astype(np.int16)

        # Build the ordered list of work units (each an Arabic chunk or an
        # English clause). A background PRODUCER thread synthesizes them in
        # order into a queue while the main CONSUMER feeds the player. So
        # while unit N plays, unit N+1 is already being synthesized — the
        # gap at an Arabic/English boundary (or any unit boundary) is
        # eliminated. Absolute synchronization: one player, one ordered
        # stream, no overlap, no inter-unit silence gap.
        units = []
        for seg, is_ar in _splitByScript(text):
            if not seg.strip():
                continue
            if is_ar:
                for sub in _neuralChunks(seg):
                    units.append((True, sub))
            else:
                for sub in _englishClauses(seg):
                    units.append((False, sub))
        if not units:
            return True

        import queue as _q
        pcmq = _q.Queue(maxsize=4)   # small buffer of ready audio

        def _produce():
            for is_ar, sub in units:
                if cancelled():
                    break
                try:
                    if is_ar:
                        raw = self._neuralArabicPCM(
                            sub, pre_diacritized=pre_diacritized)
                        seg_sr = out_sr
                        pause = _punctPause(sub, out_sr, is_arabic=True,
                                            scale=self._pauseScale())
                    else:
                        raw, seg_sr = self._englishPCM(sub, pitchOffset)
                        pause = _punctPause(sub, out_sr, is_arabic=False,
                                            scale=self._pauseScale())
                    if raw:
                        audio = _resample(
                            np.frombuffer(raw, np.int16), seg_sr).tobytes()
                        if pause:
                            audio += pause
                        pcmq.put(audio)
                except Exception:
                    pass
            pcmq.put(None)   # sentinel: production done

        prod = threading.Thread(target=_produce, name="ClaritySynthProd",
                                daemon=True)
        prod.start()

        fed_any = False
        while True:
            if cancelled():
                break
            try:
                audio = pcmq.get(timeout=5.0)
            except Exception:
                break
            if audio is None:
                break
            if cancelled():
                break
            self._neuralPlayer.feed(audio)
            fed_any = True
        if fed_any and not cancelled():
            # small trailing silence so the final consonant is never
            # clipped by the audio device buffer
            self._neuralPlayer.feed(b"\x00\x00" * int(out_sr * 0.06))
            self._neuralPlayer.idle()
        return True

    def _neuralArabicPCM(self, seg, pre_diacritized=False):
        """Arabic run -> normalized/pitched/sped PCM bytes (or None).

        pre_diacritized=True means `seg` is already correctly vocalized (a
        letter name) and must not be re-diacritized."""
        try:
            from . import timescale
        except Exception:
            timescale = None
        spk = 0
        if self._voice.startswith("neural"):
            try:
                spk = int(self._voice[6:])
            except ValueError:
                spk = 0
        try:
            if pre_diacritized:
                diac = seg
            else:
                if hasattr(ar_g2p, "clean_arabic_text"):
                    seg = ar_g2p.clean_arabic_text(seg) or seg
                diac = ar_g2p._neural_pre(seg) if hasattr(ar_g2p,
                                                          "_neural_pre") else seg
        except Exception:
            diac = seg
        # Arabic PIPER voice as primary: when the Primary engine is "piper",
        # the Voice combo holds the Arabic Piper voice (e.g. "piper:kareem").
        # Diacritization is already done above (same tashkeel libraries);
        # Piper renders the diacritized Arabic directly. This is the SAME code
        # path a Piper voice takes as a secondary voice, so its pitch is
        # identical either way.
        engine = getattr(self, "_primaryEngine", "mixer")
        pv = getattr(self, "_primaryVariant", "auto")
        piper_vid = None
        if engine == "piper" and self._voice.startswith("piper:"):
            piper_vid = self._voice.split(":", 1)[1]
        elif engine == "piper":
            # engine is Piper but the Voice combo hasn't been refreshed to a
            # piper: voice yet (NVDA builds the combo once per dialog). Fall
            # back to the first installed Arabic Piper voice so selecting the
            # Piper engine works even before reopening settings.
            try:
                from . import voice_packs
                for vid, label, lang in voice_packs.piper_voices():
                    if voice_packs.is_arabic_voice(vid, lang):
                        piper_vid = vid
                        break
            except Exception:
                piper_vid = None
        elif pv.startswith("piper:"):        # backward compat with old config
            piper_vid = pv.split(":", 1)[1]
        if piper_vid and piper_vid != "none" and _piperEN is not None:
            try:
                from . import piper_neural, timescale as _ts
                vid = piper_vid
                if hasattr(piper_neural, "synth_wave_with"):
                    raw, sr = piper_neural.synth_wave_with(
                        diac, vid, length_scale=self._neuralLengthScale(),
                        clarity=self._clarity)
                    if raw:
                        raw = _ts.finalize_audio(
                            raw, sr, self._volumeGain(),
                            semitones=self._pitchSemitones(0),
                            speed=self._speedFactor(), even_droop=False)
                        # the caller (_speakMixedNeural) treats all Arabic
                        # output as the Mixer's rate (out_sr). A Piper voice
                        # may use a different native rate, so resample to the
                        # expected rate here, otherwise it plays too fast/slow
                        # (which also shifts the perceived pitch).
                        try:
                            want_sr = (_neuralTTS.sample_rate()
                                       if _neuralTTS is not None else 22050)
                        except Exception:
                            want_sr = 22050
                        if sr and sr != want_sr:
                            try:
                                import numpy as _np
                                a = _np.frombuffer(raw, dtype=_np.int16)
                                if a.size:
                                    ratio = want_sr / float(sr)
                                    idx = _np.round(
                                        _np.arange(0, a.size * ratio) / ratio
                                    ).astype(_np.int64)
                                    idx = idx[idx < a.size]
                                    raw = a[idx].astype(_np.int16).tobytes()
                            except Exception:
                                pass
                        return raw
            except Exception:
                pass
        out = []
        try:
            chunks = _neuralChunks(diac)
        except Exception:
            chunks = [diac]
        out_sr = _neuralTTS.sample_rate()
        pscale = self._pauseScale()
        for chunk in chunks:
            if self._cancelFlag.is_set():
                break
            pcm = _neuralTTS.synth_wave(
                chunk, speaker=spk, pace=1.0, pitch_mul=1.0, volume=1.0,
                denoise=self._denoise())
            if pcm and timescale:
                sr = _neuralTTS.sample_rate()
                pcm = timescale.finalize_audio(
                    pcm, sr, self._volumeGain(),
                    semitones=self._pitchSemitones(0),
                    speed=self._speedFactor(), even_droop=True)
            if pcm:
                out.append(pcm)
                # insert a real silence after this clause/sentence so ':' and
                # '?' actually pause (the model itself flattens punctuation)
                pause = _punctPause(chunk, out_sr, is_arabic=True,
                                    scale=pscale)
                if pause:
                    out.append(pause)
        return b"".join(out) if out else None

    def _englishPCM(self, seg, pitchOffset=0):
        """English run -> (PCM bytes, sample_rate).

        Rendered by the neural Piper voice. The formant engine is only used
        when Piper is genuinely unavailable (no model installed); it must
        never be mixed in alongside a working neural voice."""
        try:
            from . import timescale
        except Exception:
            timescale = None
        if _piperEN and any(c.isalnum() for c in seg):
            try:
                ls = self._neuralLengthScale()
                pcm = _piperEN.synth_wave(seg, length_scale=ls, volume=1.0,
                                          clarity=self._clarity)
                if pcm:
                    sr = _piperEN.sample_rate()
                    if timescale:
                        pcm = timescale.finalize_audio(
                            pcm, sr, self._volumeGain(),
                            semitones=self._pitchSemitones(pitchOffset),
                            speed=self._speedFactor(), even_droop=False)
                    return pcm, sr
            except Exception:
                pass
        # formant fallback -> render to PCM bytes
        try:
            tokens = g2p.text_to_tokens(seg)
            if not tokens:
                return None, engine.SR
            buf = []
            _b = _get_bridge()
            _eng = _b.synthesize if _b else engine.synthesize
            for block in _eng(tokens, dscale=self._durationScale(),
                              base_f0=self._baseF0(pitchOffset),
                              inflection=self._inflectionValue(),
                              volume=self._volume / 100.0,
                              is_cancelled=self._cancelFlag.is_set,
                              fscale=self._formantScale(),
                              breath_amt=self._breathiness / 100.0,
                              jitter=self._roughness / 100.0 * 0.6,
                              shimmer=self._roughness / 100.0 * 0.5):
                buf.append(block)
            return (b"".join(buf), engine.SR) if buf else (None, engine.SR)
        except Exception:
            return None, engine.SR

    def _speakNeural(self, text, pre_diacritized=False):
        """Diacritize (via existing pipeline) then synthesize with the
        neural voice. Returns True if it produced audio.

        If pre_diacritized is True, `text` is already correctly vocalized
        (e.g. a letter name like 'سِين') and MUST NOT be re-diacritized —
        re-running the diacritizer adds spurious case endings (إعراب) and
        corrupts the vowels (سِين -> سِينَ, جِيم -> جَيمَ)."""
        try:
            from . import ar_g2p
            # Fully diacritize via our pipeline (neural Shakkelha if
            # present, else statistical+Mishkal) so the neural VOICE
            # always receives vocalized text and never self-downloads.
            diac = text
            if not pre_diacritized:
                try:
                    diac = ar_g2p._neural_pre(text)
                except Exception:
                    pass
                if not any("\u064B" <= c <= "\u0652" for c in diac):
                    # fall back to per-word diacritization
                    try:
                        from . import ar_diacritizer
                        ar_diacritizer._load()
                        words = diac.split()
                        out = []
                        prev = None
                        for wd in words:
                            dd = ar_diacritizer.diacritize(wd, prev) or wd
                            out.append(dd); prev = wd
                        diac = " ".join(out)
                    except Exception:
                        pass
            spk = 0
            if self._voice.startswith("neural"):
                try:
                    spk = int(self._voice[6:])
                except ValueError:
                    spk = 0
            pace = 1.0 / self._neuralLengthScale()  # safe-zone model speed
            sr = _neuralTTS.sample_rate()
            if getattr(self, "_neuralSR", None) != sr:
                self._neuralPlayer = nvwave.WavePlayer(
                    channels=1, samplesPerSec=sr, bitsPerSample=16)
                self._neuralSR = sr
            # Split long text into clause-sized chunks so (a) the neural
            # diacritizer/model is not confused by very long strings with
            # : " punctuation, and (b) we can honour cancel between chunks
            # for responsive Ctrl/interrupt.
            for chunk in _neuralChunks(diac):
                if self._cancelFlag.is_set():
                    return True
                pcm = _neuralTTS.synth_wave(
                    chunk, speaker=spk, pace=max(0.85, min(1.18, pace)),
                    pitch_mul=1.0, volume=1.0)  # normalize below
                if self._cancelFlag.is_set():
                    return True
                if pcm:
                    try:
                        from . import timescale
                        sr = _neuralTTS.sample_rate()
                        # identical pipeline to the English voice so both
                        # behave the same at a given rate/pitch/volume
                        pcm = timescale.finalize_audio(
                            pcm, sr, self._volumeGain(),
                            semitones=self._pitchSemitones(0),
                            speed=self._speedFactor(), even_droop=True)
                    except Exception:
                        pass
                    self._neuralPlayer.feed(pcm)
            sr = _neuralTTS.sample_rate()
            self._neuralPlayer.feed(b"\x00\x00" * int(sr * 0.06))
            self._neuralPlayer.idle()
            return True
        except Exception:
            log.debugWarning("ClaritySynth neural speak failed",
                             exc_info=True)
            return False

    def _worker(self):
        while True:
            items = self._queue.get()
            if items is None:
                break
            self._cancelFlag.clear()
            try:
                self._speakItems(items)
            except Exception:
                log.error("ClaritySynth: synthesis error", exc_info=True)
            if not self._cancelFlag.is_set():
                try:
                    self._player.idle()
                except Exception:
                    pass
                synthDoneSpeaking.notify(synth=self)

    def _coalesce(self, items):
        """Merge runs of adjacent text items that share the same charMode
        into a single text item, so NVDA's separate UI-field strings are
        spoken as one continuous utterance (no phantom breaks or delays
        between them). Index/break commands are preserved in order; any
        index commands that fell between merged text are attached to the
        merged item so they still fire after it is queued."""
        out = []
        buf = None          # [texts, charMode, pitchOffset, pending_idx]
        for item in items:
            if item[0] == "text":
                _, text, charMode, pitchOffset = item
                if buf is not None and buf[1] == charMode:
                    # join with a space only if needed (avoid gluing words)
                    if buf[0] and not buf[0][-1].endswith((" ",)) \
                            and not text.startswith(" "):
                        buf[0].append(" ")
                    buf[0].append(text)
                else:
                    if buf is not None:
                        out.append(("mtext", "".join(buf[0]), buf[1],
                                    buf[2], buf[3]))
                    buf = [[text], charMode, pitchOffset, []]
            elif item[0] == "index":
                # keep the index with the current merged text so ordering
                # is preserved; if no text yet, emit standalone
                if buf is not None:
                    buf[3].append(item[1])
                else:
                    out.append(item)
            else:
                # break or other: flush current text first
                if buf is not None:
                    out.append(("mtext", "".join(buf[0]), buf[1],
                                buf[2], buf[3]))
                    buf = None
                out.append(item)
        if buf is not None:
            out.append(("mtext", "".join(buf[0]), buf[1], buf[2], buf[3]))
        return out

    def _speakItems(self, items):
        cancelled = self._cancelFlag.is_set
        # guarantee the engines are up (no-op after the first call). This
        # runs on the speech worker thread, never on NVDA's GUI thread.
        if not _engines_ready:
            _ensure_engines()
        items = self._coalesce(items)
        for item in items:
            if cancelled():
                return
            kind = item[0]
            if kind == "index":
                # Notify once the audio produced so far has been queued.
                self._notifyIndex(item[1])
            elif kind == "break":
                ms = max(10, min(2000, int(item[1])))
                self._feed(b"\x00\x00" * int(engine.SR * ms / 1000.0))
            elif kind == "mtext":
                _, text, charMode, pitchOffset, pending_idx = item
                # process as a text item, then fire any pending indices
                self._speakOneText(text, charMode, pitchOffset)
                for idx in pending_idx:
                    self._notifyIndex(idx)
            elif kind == "text":
                _, text, charMode, pitchOffset = item
                self._speakOneText(text, charMode, pitchOffset)

    def _feedNeural(self, pcm, sr):
        """Feed PCM to the neural player, re-creating it if the sample rate
        changed, and pad a little trailing silence so the audio device never
        clips the final consonant."""
        if not pcm:
            return
        if getattr(self, "_neuralSR", None) != sr:
            self._neuralPlayer = nvwave.WavePlayer(
                channels=1, samplesPerSec=sr, bitsPerSample=16)
            self._neuralSR = sr
        self._neuralPlayer.feed(pcm)
        self._neuralPlayer.feed(b"\x00\x00" * int(sr * 0.05))
        self._neuralPlayer.idle()

    def _speakEnglishChar(self, ch, pitchOffset=0):
        """Speak ONE non-Arabic character with the secondary neural voice.

        For an English voice, letters and digits are rendered from an exact
        IPA table (eSpeak mispronounces single English letter names). For a
        non-English voice (French, Spanish, etc.), the character is handed to
        that voice so it is phonemized and spoken in ITS OWN language, rather
        than with English letter names. Returns True if handled."""
        if not _piperEN:
            return False
        try:
            from . import timescale
        except Exception:
            timescale = None

        # is the current secondary voice English?
        is_en = True
        try:
            if hasattr(_piperEN, "current_voice_is_english"):
                is_en = _piperEN.current_voice_is_english()
        except Exception:
            is_en = True

        ipa = None
        word = None
        if is_en:
            ipa = _englishCharIPA(ch)
            word = None if ipa else _englishCharWord(ch)
            if not ipa and not word:
                return False
        else:
            # non-English voice: let it phonemize the character itself so it
            # is spoken in its own alphabet/language. Pass the character (or,
            # for a lone digit, the digit) straight through as text.
            word = ch
            if not word or not word.strip():
                return False
        try:
            pcm = _piperEN.synth_wave(
                word or "", length_scale=self._neuralLengthScale(),
                volume=1.0, phonemes=ipa)
            if not pcm:
                return False
            sr = _piperEN.sample_rate()
            if timescale:
                pcm = timescale.finalize_audio(
                    pcm, sr, self._volumeGain(),
                    semitones=self._pitchSemitones(pitchOffset),
                    speed=self._speedFactor(), even_droop=False)
            self._feedNeural(pcm, sr)
            return True
        except Exception:
            return False

    def _speakRuntimeWarning(self):
        """Speak a short, actionable message with the formant engine (which
        needs no onnxruntime) when the neural engine could not start."""
        msg = ("ClaritySynth neural voices could not start on this computer. "
               "Please install the Microsoft Visual C plus plus Redistributable, "
               "then restart NVDA.")
        try:
            tokens = g2p.text_to_tokens(msg)
            if not tokens:
                return
            _b = _get_bridge()
            _eng = _b.synthesize if _b else engine.synthesize
            for chunk in _eng(tokens, dscale=self._durationScale(),
                              base_f0=self._baseF0(0),
                              volume=self._volume / 100.0,
                              is_cancelled=self._cancelFlag.is_set):
                if self._cancelFlag.is_set():
                    return
                self._feed(chunk)
        except Exception:
            pass

    def _speakOneText(self, text, charMode, pitchOffset):
        cancelled = self._cancelFlag.is_set
        engine = getattr(self, "_primaryEngine", "mixer")
        stripped = text.strip()
        is_single_ar = (charMode and len(stripped) == 1
                        and any("\u0600" <= c <= "\u06FF" for c in stripped))
        # Single Arabic character in character mode -> speak its Arabic letter
        # name with the Arabic voice. Works for BOTH engines: Mixer (via
        # _speakNeural) and Piper (via the Arabic-Piper path in
        # _speakMixedNeural / _neuralArabicPCM).
        if is_single_ar and _neuralTTS and engine != "piper" \
                and self._voice.startswith("neural"):
            nm = _arabicCharName(stripped)
            if nm and self._speakNeural(nm, pre_diacritized=True):
                return
        if is_single_ar and engine == "piper":
            nm = _arabicCharName(stripped)
            if nm and self._speakMixedNeural(nm, pitchOffset,
                                             pre_diacritized=True):
                return
        if (self._voice.startswith("neural") and _neuralTTS
                and not charMode
                and any("\u0600" <= c <= "\u06FF" for c in text)):
            if self._speakMixedNeural(text, pitchOffset):
                return
        # Pure/mixed English via the gapless streamed neural path.
        if (_piperEN and not charMode
                and any(c.isalpha() and c < "\u0600" for c in text)):
            if self._speakMixedNeural(text, pitchOffset):
                return
        # Single English character in char mode -> ALWAYS the neural English
        # voice (letters, digits, punctuation and symbols alike). Nothing
        # here may fall through to the formant engine.
        if charMode and _piperEN:
            ch = text.strip()
            if len(ch) == 1:
                if self._speakEnglishChar(ch, pitchOffset):
                    return
                # Any other single character (currency, bullets, arrows...):
                # let the neural voice name it rather than dropping to the
                # formant engine.
                if self._speakMixedNeural(ch, pitchOffset):
                    return
                return          # nothing sensible to say; stay silent
            # a bare space still needs announcing
            if text and not ch and self._speakEnglishChar(" ", pitchOffset):
                return
        # NEURAL DRIVER: never fall through to the formant engine while a
        # neural voice is selected — that is what made formant timbre "bleed"
        # into the neural synth. Speak it neurally, or say nothing.
        if self._voice.startswith("neural") or _HAVE_AR or _HAVE_EN:
            if text and text.strip():
                produced = self._speakMixedNeural(text, pitchOffset)
                # neural genuinely unavailable at runtime (models present but
                # engine did not start) -> warn once via the formant engine
                if not produced and not _engines_ready:
                    _ensure_engines()
                    produced = self._speakMixedNeural(text, pitchOffset)
                if not produced and _neuralTTS is None and _piperEN is None \
                        and not getattr(self, "_warnedNoRuntime", False):
                    self._warnedNoRuntime = True
                    self._speakRuntimeWarning()
            return
        # Only reached when NO neural models are installed at all.
        if charMode and len(text.strip()) == 1:
            tokens = g2p.char_to_tokens(text.strip())
        else:
            tokens = g2p.text_to_tokens(text)
        if not tokens:
            return
        _b = _get_bridge()
        _eng = _b.synthesize if _b else engine.synthesize
        gen = _eng(
            tokens,
            dscale=self._durationScale(),
            base_f0=self._baseF0(pitchOffset),
            inflection=self._inflectionValue(),
            volume=self._volume / 100.0,
            is_cancelled=cancelled,
            fscale=self._formantScale(),
            breath_amt=(_CLONED["breath_amt"]
                        if self._voice == "cloned" and _CLONED
                        else self._breathiness / 100.0),
            jitter=self._roughness / 100.0 * 0.6,
            shimmer=self._roughness / 100.0 * 0.5,
            accent=self._stressEmphasis / 50.0,
            pause_scale=0.5 + self._pauseLength / 100.0 * 1.4,
        )
        for chunk in gen:
            if cancelled():
                return
            self._feed(chunk)
