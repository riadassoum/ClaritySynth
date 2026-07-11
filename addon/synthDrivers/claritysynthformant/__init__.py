# -*- coding: utf-8 -*-
"""ClaritySynth Formant — the pure-Python formant voices as an independent
NVDA synth driver.

This exposes ONLY the lightweight Klatt-style formant voices (Adam, Clara,
Robby), with no neural models loaded. It shares the engine and G2P code
with the main ClaritySynth add-on. Use this driver when you want a tiny,
fast, always-available synth without the neural voices — or as a reliable
fallback. The main "ClaritySynth" driver provides the neural Arabic and
English voices (with these formant voices still available inside it too).
"""
import os
import sys
import threading
import queue

import synthDriverHandler
from synthDriverHandler import SynthDriver as _BaseSynthDriver
from synthDriverHandler import VoiceInfo, synthIndexReached, synthDoneSpeaking
from logHandler import log
import nvwave

from autoSettingsUtils.driverSetting import (NumericDriverSetting,
                                             BooleanDriverSetting)
try:
    from autoSettingsUtils.utils import StringParameterInfo
except Exception:
    pass
from collections import OrderedDict

# Access the shared engine and g2p from the main ClaritySynth package.
# We append (never insert) the package dir to sys.path so we cannot shadow
# any other add-on's modules, and we import the leaf modules by name. The
# shared modules use package-relative imports among themselves (g2p ->
# ar_g2p -> phonemes); those resolve because they all sit in this dir.
# Importing engine/g2p does NOT run the main driver's __init__, so no neural
# models load here — this driver stays lightweight and independent.
_MAIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))), "claritysynth")
if _MAIN not in sys.path:
    sys.path.append(_MAIN)

# Import as a package so g2p's "from . import ar_g2p" works, but reach the
# leaf modules directly to avoid the heavy package __init__.
import importlib
engine = importlib.import_module("engine")
try:
    g2p = importlib.import_module("g2p")
except ImportError:
    import ar_g2p  # noqa: F401
    import phonemes  # noqa: F401
    g2p = importlib.import_module("g2p")

# Optional: the NV Speech Player DLL engine (higher-quality formant DSP),
# same as the main driver uses. Falls back to the pure-Python engine.
_dllBridge = None
try:
    _dll_engine = importlib.import_module("dll_engine")
    _dllBridge = _dll_engine.Bridge()
except Exception:
    _dllBridge = None

# Optional: the neural Shakkelha diacritizer, so the formant driver can
# also auto-vocalize undiacritized Arabic when enabled.
_ar_g2p_mod = None
try:
    _ar_g2p_mod = importlib.import_module("ar_g2p")
except Exception:
    _ar_g2p_mod = None

try:
    import speech
    from speech.commands import (IndexCommand, BreakCommand, PitchCommand,
                                 CharacterModeCommand)
except Exception:
    IndexCommand = BreakCommand = PitchCommand = None
    CharacterModeCommand = None


_EN_LETTER_NAMES = {
    "a": "ay", "b": "bee", "c": "see", "d": "dee", "e": "ee",
    "f": "eff", "g": "jee", "h": "aitch", "i": "eye", "j": "jay",
    "k": "kay", "l": "ell", "m": "em", "n": "en", "o": "oh",
    "p": "pee", "q": "cue", "r": "arr", "s": "ess", "t": "tee",
    "u": "you", "v": "vee", "w": "double you", "x": "eks",
    "y": "why", "z": "zee",
}


def _english_letter(ch):
    """Spoken name of a single English letter, so character navigation
    announces 'ay', 'bee', ... instead of trying to pronounce the bare
    letter as a word."""
    return _EN_LETTER_NAMES.get(ch.lower(), ch)


class SynthDriver(_BaseSynthDriver):
    name = "claritysynthformant"
    description = _("ClaritySynth Formant (lightweight, English & Arabic)")

    supportedSettings = (
        _BaseSynthDriver.VoiceSetting(),
        _BaseSynthDriver.RateSetting(),
        _BaseSynthDriver.PitchSetting(),
        _BaseSynthDriver.InflectionSetting(),
        _BaseSynthDriver.VolumeSetting(),
        NumericDriverSetting("breathiness", _("&Breathiness"),
                             defaultVal=6),
        NumericDriverSetting("roughness", _("&Roughness"), defaultVal=6),
        BooleanDriverSetting("neuralArabic",
                             _("Use &neural Arabic diacritizer if "
                               "available"), defaultVal=True),
    )

    supportedCommands = frozenset(
        c for c in (IndexCommand, BreakCommand, PitchCommand,
                    CharacterModeCommand) if c)
    supportedNotifications = frozenset(
        [synthIndexReached, synthDoneSpeaking])

    @classmethod
    def check(cls):
        return True

    def __init__(self):
        super().__init__()
        self._rate = 50
        self._pitch = 50
        self._inflection = 50
        self._volume = 90
        self._voice = "adam"
        self._breathiness = 6
        self._roughness = 6
        self._neuralArabic = True
        self._queue = queue.Queue()
        self._cancelFlag = threading.Event()
        self._player = nvwave.WavePlayer(channels=1, samplesPerSec=engine.SR,
                                         bitsPerSample=16)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def terminate(self):
        self.cancel()
        try:
            self._queue.put(None)
        except Exception:
            pass
        try:
            self._player.close()
        except Exception:
            pass

    def _getAvailableVoices(self):
        return OrderedDict((
            ("adam", VoiceInfo("adam", _("Adam (formant, low)"), None)),
            ("clara", VoiceInfo("clara", _("Clara (formant, high)"),
                                None)),
            ("robby", VoiceInfo("robby", _("Robby (formant, robot)"),
                                None)),
        ))

    def _get_voice(self):
        return self._voice

    def _set_voice(self, v):
        if v in self.availableVoices:
            self._voice = v

    def _get_rate(self):
        return self._rate

    def _set_rate(self, v):
        self._rate = max(0, min(100, int(v)))

    def _get_pitch(self):
        return self._pitch

    def _set_pitch(self, v):
        self._pitch = max(0, min(100, int(v)))

    def _get_inflection(self):
        return self._inflection

    def _set_inflection(self, v):
        self._inflection = max(0, min(100, int(v)))

    def _get_volume(self):
        return self._volume

    def _set_volume(self, v):
        self._volume = max(0, min(100, int(v)))

    def _get_breathiness(self):
        return self._breathiness

    def _set_breathiness(self, v):
        self._breathiness = max(0, min(100, int(v)))

    def _get_roughness(self):
        return self._roughness

    def _set_roughness(self, v):
        self._roughness = max(0, min(100, int(v)))

    def _get_neuralArabic(self):
        return self._neuralArabic

    def _set_neuralArabic(self, value):
        self._neuralArabic = bool(value)
        try:
            if _ar_g2p_mod is not None:
                _ar_g2p_mod._NEURAL_ON = bool(value)
        except Exception:
            pass

    def _durationScale(self):
        return 2.2 * ((0.42 / 2.2) ** (self._rate / 100.0))

    def _baseF0(self, off=0):
        p = max(0, min(100, self._pitch + off))
        f0 = 62.0 * (2.0 ** (p / 100.0 * 1.5))
        if self._voice == "clara":
            f0 *= 1.75
        return f0

    def _formantScale(self):
        return 1.18 if self._voice == "clara" else 1.0

    def speak(self, speechSequence):
        self._cancelFlag.clear()
        pitchOffset = 0
        charMode = False
        for item in speechSequence:
            if isinstance(item, str):
                self._queue.put(("text", item, charMode, pitchOffset))
            elif IndexCommand and isinstance(item, IndexCommand):
                self._queue.put(("index", item.index))
            elif BreakCommand and isinstance(item, BreakCommand):
                self._queue.put(("break", item.time))
            elif CharacterModeCommand and isinstance(
                    item, CharacterModeCommand):
                charMode = item.state
            elif PitchCommand and isinstance(item, PitchCommand):
                try:
                    pitchOffset = int(item.offset)
                except Exception:
                    pitchOffset = 0

    def cancel(self):
        self._cancelFlag.set()
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._player.stop()
        except Exception:
            pass

    def pause(self, switch):
        try:
            self._player.pause(switch)
        except Exception:
            pass

    def _feed(self, block):
        try:
            self._player.feed(block)
        except Exception:
            pass

    def _worker(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            try:
                kind = item[0]
                if kind == "index":
                    synthIndexReached.notify(synth=self, index=item[1])
                elif kind == "break":
                    ms = item[1]
                    self._feed(b"\x00\x00" * int(engine.SR * ms / 1000.0))
                elif kind == "text":
                    _, text, charMode, pitchOffset = item
                    # Character mode: a single letter should be spoken as
                    # its NAME. For Arabic, that is handled by g2p's char
                    # path; for a single English/Latin letter we speak the
                    # letter name so it is never mis-phonemized.
                    if charMode and len(text.strip()) == 1:
                        ch = text.strip()
                        if "a" <= ch.lower() <= "z":
                            tokens = g2p.text_to_tokens(_english_letter(ch))
                        else:
                            tokens = g2p.char_to_tokens(ch) \
                                if hasattr(g2p, "char_to_tokens") \
                                else g2p.text_to_tokens(ch)
                    else:
                        spk_text = text
                        # optional neural diacritization of Arabic
                        if (self._neuralArabic and _ar_g2p_mod is not None
                                and any("\u0600" <= c <= "\u06FF"
                                        for c in text)):
                            try:
                                spk_text = _ar_g2p_mod._neural_pre(text)
                            except Exception:
                                spk_text = text
                        tokens = g2p.text_to_tokens(spk_text)
                    if not tokens:
                        continue
                    # prefer the DLL engine (richer formant DSP) if present
                    _synth = (_dllBridge.synthesize if _dllBridge
                              else engine.synthesize)
                    gen = _synth(
                        tokens, dscale=self._durationScale(),
                        base_f0=self._baseF0(pitchOffset),
                        inflection=self._inflection / 100.0,
                        volume=self._volume / 100.0,
                        is_cancelled=self._cancelFlag.is_set,
                        fscale=self._formantScale(),
                        breath_amt=self._breathiness / 100.0,
                        jitter=self._roughness / 100.0 * 0.6,
                        shimmer=self._roughness / 100.0 * 0.5)
                    for block in gen:
                        if self._cancelFlag.is_set():
                            break
                        self._feed(block)
                if self._queue.empty():
                    self._player.idle()
                    synthDoneSpeaking.notify(synth=self)
            except Exception:
                log.error("claritysynthformant worker error",
                          exc_info=True)
