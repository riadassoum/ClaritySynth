# -*- coding: utf-8 -*-
# ClaritySynth: NVDA synthesizer driver.
# A completely self-contained English formant synthesizer — no external
# engines, DLLs, voices, or network access. Text is phonemized by rule
# (g2p.py) and rendered by a pure-Python Klatt-style engine (engine.py).

import threading
import queue
from collections import OrderedDict

import config
import nvwave
import synthDriverHandler

try:
    from autoSettingsUtils.driverSetting import (NumericDriverSetting,
                                                  BooleanDriverSetting)
except ImportError:
    from driverHandler import NumericDriverSetting, BooleanDriverSetting
from synthDriverHandler import VoiceInfo, synthIndexReached, synthDoneSpeaking
from logHandler import log

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
try:
    from . import piper_neural
    if piper_neural._try_init():
        _piperEN = piper_neural
        log.info("ClaritySynth: neural English (Piper) voice active")
except Exception:
    log.debugWarning("ClaritySynth: Piper English unavailable",
                     exc_info=True)
try:
    from . import tts_neural
    if tts_neural._try_init():
        _neuralTTS = tts_neural
        _neuralSpeakers = tts_neural.SPEAKERS
        log.info("ClaritySynth: neural Arabic voice ACTIVE - model=%s "
                 "vocoder=%s speakers=%d. Select 'Arabic Neural ...' in "
                 "the Voice list." % (tts_neural._model_id,
                 tts_neural._vocoder_id, len(_neuralSpeakers)))
    else:
        log.info("ClaritySynth: neural voice models present but runtime "
                 "did not initialise; using formant voices.")
except Exception:
    log.info("ClaritySynth: neural TTS unavailable (no onnxruntime or "
             "models); formant voices only.", exc_info=True)

_bridge = None
try:
    from . import dll_engine
    _bridge = dll_engine.Bridge()
    log.info("ClaritySynth: speechPlayer.dll bridge active")
except Exception:
    log.debugWarning("ClaritySynth: DLL bridge unavailable; pure engine",
                     exc_info=True)


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


def _punctPause(text, sr, is_arabic=False):
    """Return trailing silence for the punctuation ENDING this chunk.

    The neural ARABIC model flattens ALL punctuation, so Arabic needs
    explicit pauses at commas/colons AND sentence enders. The English
    (Piper) voice renders its own clause phrasing, so English only gets a
    small extra pause at sentence enders to avoid doubling up."""
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
        if last in ".!?":                    # English: sentence enders only
            ms = 0.13
        else:
            return b""
    return b"\x00\x00" * int(sr * ms)


def _englishClauses(text):
    """Split English into natural clause/sentence units at punctuation
    ONLY, keeping each unit whole so the neural voice does not add
    sentence-boundary intonation mid-phrase. A unit with no punctuation is
    yielded intact (no arbitrary word chopping); only a very long
    punctuation-free run is broken, and then on a large word boundary so
    any seam is minimally audible."""
    import re
    # split AFTER sentence/clause punctuation, keeping the mark with its
    # clause. Covers . ! ? ; : , and — (em dash) and newlines.
    parts = re.split(r'(?<=[.!?;:,])\s+|\s*[\u2014]\s*|\n+', text)
    for p in parts:
        if p is None:
            continue
        p = p.strip()
        if not p:
            continue
        # only break if a single punctuation-free run is very long
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


def _englishLetterName(ch):
    """Spoken name of a single English letter for the neural voice."""
    names = {
        "a": "ay", "b": "bee", "c": "see", "d": "dee", "e": "ee",
        "f": "eff", "g": "jee", "h": "aitch", "i": "eye", "j": "jay",
        "k": "kay", "l": "ell", "m": "em", "n": "en", "o": "oh",
        "p": "pee", "q": "cue", "r": "arr", "s": "ess", "t": "tee",
        "u": "you", "v": "vee", "w": "double you", "x": "eks",
        "y": "why", "z": "zee",
    }
    return names.get(ch.lower(), ch)


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
        "\u064A": "يَاء", "\u0621": "هَمْزَة", "\u0623": "هَمْزَة",
        "\u0625": "هَمْزَة", "\u0624": "هَمْزَة", "\u0626": "هَمْزَة",
        "\u0622": "أَلِف مَدّ", "\u0629": "تَاء مَربُوطَة",
        "\u0649": "أَلِف مَقْصُورَة", "\u0640": "تَطْوِيل",
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


def _neuralChunks(text, limit=180):
    """Yield clause-sized pieces of (already diacritized) text. The neural
    model handles short clauses far better than very long strings, and it
    lets speech be interrupted between clauses. Splits on sentence and
    clause punctuation (Arabic and Latin), keeping the delimiter."""
    import re
    # split after . ! ? ; : , and their Arabic forms, and on quotes/newlines
    parts = re.split(r'(?<=[\.!\?;:،؛؟\n\"\u00bb\u00ab])\s+', text)
    buf = ""
    for p in parts:
        if not p:
            continue
        if len(buf) + len(p) + 1 <= limit:
            buf = (buf + " " + p).strip()
        else:
            if buf:
                yield buf
            # a single over-long clause: hard-wrap on whitespace
            while len(p) > limit:
                cut = p.rfind(" ", 0, limit)
                if cut <= 0:
                    cut = limit
                yield p[:cut].strip()
                p = p[cut:].strip()
            buf = p
    if buf:
        yield buf


class SynthDriver(synthDriverHandler.SynthDriver):
    name = "claritysynth"
    # Translators: description of the ClaritySynth synthesizer.
    description = _("ClaritySynth (Neural Arabic & English)")

    supportedSettings = (
        synthDriverHandler.SynthDriver.VoiceSetting(),
        synthDriverHandler.SynthDriver.RateSetting(),
        BooleanDriverSetting("rateBoost",
                             _("Rate boo&st (extra-fast speech)"),
                             defaultVal=False,
                             availableInSettingsRing=True),
        synthDriverHandler.SynthDriver.PitchSetting(),
        synthDriverHandler.SynthDriver.InflectionSetting(),
        synthDriverHandler.SynthDriver.VolumeSetting(),
        # Translators: labels for ClaritySynth voice parameters.
        NumericDriverSetting("breathiness", _("&Breathiness"),
                             defaultVal=6, availableInSettingsRing=True),
        NumericDriverSetting("roughness", _("Rou&ghness"),
                             defaultVal=18, availableInSettingsRing=True),
        NumericDriverSetting("headSize", _("&Head size"),
                             defaultVal=50, availableInSettingsRing=True),
        NumericDriverSetting("stressEmphasis", _("Stress &emphasis"),
                             defaultVal=50, availableInSettingsRing=True),
        NumericDriverSetting("pauseLength", _("Pause &length"),
                             defaultVal=40, availableInSettingsRing=True),
        BooleanDriverSetting("tanweenPause",
                             _("Pronounce &tanween on isolated words"),
                             defaultVal=False),
        BooleanDriverSetting("neuralArabic",
                             _("Use &neural Arabic diacritizer if available"),
                             defaultVal=True),
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
        # default to the neural Arabic voice when available, else formant
        self._voice = ("neural0" if (_neuralTTS and _neuralSpeakers)
                       else "adam")
        self._breathiness = 6
        self._roughness = 18
        self._headSize = 50
        self._stressEmphasis = 50
        self._pauseLength = 40
        self._tanweenPause = False
        self._neuralArabic = True
        self._rateBoost = False
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

    def _getAvailableVoices(self):
        voices = OrderedDict()
        # Neural Arabic voices listed FIRST when available (headline
        # feature). The label shows which model is active so users know
        # whether a quality pack is installed.
        if _neuralTTS and _neuralSpeakers:
            q = _neuralTTS._model_id or "neural"
            label = {"fastpitch": "HQ", "mixer128": "Std",
                     "mixer80": "Fast"}.get(q, q)
            for s in _neuralSpeakers:
                if len(_neuralSpeakers) > 1:
                    nm = _("Arabic Neural %s - Speaker %d") % (label, s + 1)
                else:
                    nm = _("Arabic Neural %s") % label
                voices["neural%d" % s] = VoiceInfo("neural%d" % s, nm, "ar")
        # NOTE: the formant voices (Adam/Clara/Robby) are intentionally NOT
        # listed here — they live in the separate "ClaritySynth Formant"
        # driver. This driver is the neural synth; English is spoken by the
        # neural Piper voice automatically. If no neural voice is available
        # at all, fall back to a single formant entry so the synth still
        # works.
        if not voices:
            voices["adam"] = VoiceInfo("adam", _("Adam (fallback)"), None)
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

    def _get_pauseLength(self):
        return self._pauseLength

    def _set_pauseLength(self, value):
        self._pauseLength = max(0, min(100, value))

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

    def _speakMixedNeural(self, text, pitchOffset=0):
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
                        raw = self._neuralArabicPCM(sub)
                        seg_sr = out_sr
                        pause = _punctPause(sub, out_sr, is_arabic=True)
                    else:
                        raw, seg_sr = self._englishPCM(sub, pitchOffset)
                        pause = _punctPause(sub, out_sr, is_arabic=False)
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

    def _neuralArabicPCM(self, seg):
        """Arabic run -> normalized/pitched/sped PCM bytes (or None)."""
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
            diac = ar_g2p._neural_pre(seg) if hasattr(ar_g2p,
                                                      "_neural_pre") else seg
        except Exception:
            diac = seg
        out = []
        try:
            chunks = _neuralChunks(diac)
        except Exception:
            chunks = [diac]
        for chunk in chunks:
            if self._cancelFlag.is_set():
                break
            pcm = _neuralTTS.synth_wave(
                chunk, speaker=spk, pace=1.0, pitch_mul=1.0, volume=1.0)
            if pcm and timescale:
                sr = _neuralTTS.sample_rate()
                pcm = timescale.normalize_rms(pcm)
                pcm = timescale.pitch_shift_pcm(pcm,
                                                self._pitchSemitones(0), sr)
                pcm = timescale.compress_pcm(pcm, self._speedFactor(), sr)
                pcm = timescale.apply_gain(pcm, self._volume / 100.0)
            if pcm:
                out.append(pcm)
        return b"".join(out) if out else None

    def _englishPCM(self, seg, pitchOffset=0):
        """English run -> (PCM bytes, sample_rate). Uses Piper if present,
        else renders the formant engine to PCM."""
        try:
            from . import timescale
        except Exception:
            timescale = None
        if _piperEN and any(c.isalpha() for c in seg):
            try:
                ls = self._neuralLengthScale()
                pcm = _piperEN.synth_wave(seg, length_scale=ls, volume=1.0)
                if pcm:
                    sr = _piperEN.sample_rate()
                    if timescale:
                        pcm = timescale.normalize_rms(pcm)
                        pcm = timescale.pitch_shift_pcm(
                            pcm, self._pitchSemitones(pitchOffset), sr)
                        pcm = timescale.compress_pcm(
                            pcm, self._speedFactor(), sr)
                        pcm = timescale.apply_gain(pcm,
                                                   self._volume / 100.0)
                    return pcm, sr
            except Exception:
                pass
        # formant fallback -> render to PCM bytes
        try:
            tokens = g2p.text_to_tokens(seg)
            if not tokens:
                return None, engine.SR
            buf = []
            _eng = _bridge.synthesize if _bridge else engine.synthesize
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

    def _speakNeural(self, text):
        """Diacritize (via existing pipeline) then synthesize with the
        neural voice. Returns True if it produced audio."""
        try:
            from . import ar_g2p
            # Fully diacritize via our pipeline (neural Shakkelha if
            # present, else statistical+Mishkal) so the neural VOICE
            # always receives vocalized text and never self-downloads.
            diac = text
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
                        pcm = timescale.normalize_rms(pcm)
                        pcm = timescale.pitch_shift_pcm(
                            pcm, self._pitchSemitones(0), sr)
                        pcm = timescale.compress_pcm(
                            pcm, self._speedFactor(), sr)
                        pcm = timescale.apply_gain(pcm,
                                                   self._volume / 100.0)
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

    def _speakOneText(self, text, charMode, pitchOffset):
        cancelled = self._cancelFlag.is_set
        # Single Arabic character in character mode -> neural letter name.
        if (self._voice.startswith("neural") and _neuralTTS
                and charMode and len(text.strip()) == 1
                and any("\u0600" <= c <= "\u06FF" for c in text.strip())):
            nm = _arabicCharName(text.strip())
            if nm and self._speakNeural(nm):
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
        # Single English letter in char mode -> neural voice, letter name.
        if (charMode and len(text.strip()) == 1 and _piperEN
                and "a" <= text.strip().lower() <= "z"):
            if self._speakMixedNeural(_englishLetterName(text.strip()),
                                      pitchOffset):
                return
        if charMode and len(text.strip()) == 1:
            tokens = g2p.char_to_tokens(text.strip())
        else:
            tokens = g2p.text_to_tokens(text)
        if not tokens:
            return
        _eng = _bridge.synthesize if _bridge else engine.synthesize
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
