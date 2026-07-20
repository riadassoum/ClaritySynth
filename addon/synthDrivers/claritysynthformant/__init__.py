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
                                             BooleanDriverSetting,
                                             DriverSetting)
try:
    from autoSettingsUtils.utils import StringParameterInfo
except Exception:
    pass
from collections import OrderedDict

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
#
# IMPORTANT: the DLL is loaded LAZILY (first time it is actually used to
# synthesize), NOT at import. Loading a native DLL at import runs the instant
# the synth is selected, and on PORTABLE NVDA a fault there closes NVDA with
# no chance to warn. Lazy loading keeps synth selection safe.
_dllBridge = None
_dllBridge_tried = False


def _dll_present():
    """Cheap check (no DLL load) for whether the speechPlayer DLL ships, so
    the engine list can offer it without loading it on the GUI thread."""
    try:
        arch = "x64" if sys.maxsize > 2 ** 32 else "x86"
        return os.path.exists(os.path.join(
            _MAIN, "sp", arch, "speechPlayer.dll"))
    except Exception:
        return False


def _get_dll_bridge():
    global _dllBridge, _dllBridge_tried
    if _dllBridge_tried:
        return _dllBridge
    _dllBridge_tried = True
    try:
        _dll_engine = importlib.import_module("dll_engine")
        _dllBridge = _dll_engine.Bridge()
        log.info("ClaritySynth Formant: NV Speech Player DLL engine loaded")
    except Exception:
        _dllBridge = None
        log.debugWarning(
            "ClaritySynth Formant: NV Speech Player DLL unavailable, "
            "using built-in engine", exc_info=True)
    return _dllBridge

# Optional: the eSpeak NG multilingual formant voice (loose module in the
# claritysynth dir, on sys.path as _MAIN).
_espeak_engine = None
try:
    _espeak_engine = importlib.import_module("espeak_engine")
except Exception:
    _espeak_engine = None

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
    description = _("ClaritySynth Formant")

    supportedSettings = (
        _BaseSynthDriver.VoiceSetting(),
        _BaseSynthDriver.RateSetting(),
        BooleanDriverSetting("rateBoost", _("Rate boos&t"),
                             availableInSettingsRing=True,
                             defaultVal=False),
        _BaseSynthDriver.PitchSetting(),
        _BaseSynthDriver.InflectionSetting(),
        _BaseSynthDriver.VolumeSetting(),
        NumericDriverSetting("breathiness", _("&Breathiness"),
                             defaultVal=6),
        NumericDriverSetting("roughness", _("&Roughness"), defaultVal=6),
        DriverSetting("engine",
                      _("Synthesis &engine"),
                      availableInSettingsRing=True,
                      defaultVal="auto",
                      displayName=_("Engine")),
        DriverSetting("espeakLanguage",
                      _("eSpeak &language (eSpeak engine only)"),
                      availableInSettingsRing=True,
                      defaultVal="ar",
                      displayName=_("eSpeak language")),
        DriverSetting("espeakSecondaryLanguage",
                      _("eSpeak &second language (for non-Arabic words)"),
                      availableInSettingsRing=True,
                      defaultVal="en-us",
                      displayName=_("eSpeak second language")),
        DriverSetting("espeakVariant",
                      _("eSpeak &voice variant (eSpeak engine only)"),
                      availableInSettingsRing=True,
                      defaultVal="none",
                      displayName=_("eSpeak variant")),
        DriverSetting("tashkeel",
                      _("&Tashkeel library (Arabic diacritization)"),
                      availableInSettingsRing=True,
                      defaultVal="libtashkeel",
                      displayName=_("Tashkeel library")),
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
        self._rateBoost = False
        self._pitch = 50
        self._inflection = 50
        self._volume = 90
        self._voice = "adam"
        self._breathiness = 6
        self._roughness = 6
        self._engine = "auto"
        self._espeakLanguage = "ar"
        self._espeakSecondaryLanguage = "en-us"
        self._espeakVariant = "none"
        self._tashkeel = "libtashkeel"
        try:
            import ar_tashkeel
            ar_tashkeel.set_backend("libtashkeel")
        except Exception:
            pass
        self._queue = queue.Queue()
        self._cancelFlag = threading.Event()
        self._gen = 0          # bumped on cancel; in-flight synth checks it
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

    def _get_rateBoost(self):
        return getattr(self, "_rateBoost", False)

    def _set_rateBoost(self, value):
        self._rateBoost = bool(value)

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

    def _get_availableEngines(self):
        out = OrderedDict()
        out["auto"] = StringParameterInfo("auto", _("Auto (best available)"))
        if _dll_present():
            out["dll"] = StringParameterInfo(
                "dll", _("NV Speech Player (richer DSP)"))
        out["builtin"] = StringParameterInfo(
            "builtin", _("Built-in (pure Python)"))
        # eSpeak NG: a compact multilingual formant voice (100+ languages
        # including Arabic), each with its own articulation. Offered when the
        # bundled eSpeak library is present.
        try:
            if _espeak_engine is not None and _espeak_engine.is_available():
                out["espeak"] = StringParameterInfo(
                    "espeak", _("eSpeak NG (multilingual formant)"))
        except Exception:
            pass
        return out

    def _get_engine(self):
        return getattr(self, "_engine", "auto")

    def _set_engine(self, value):
        self._engine = value

    # ---- eSpeak language selection (only meaningful when engine == espeak).
    # ---- Lets the user pick which language's formant articulation to use,
    # ---- Arabic first.
    def _get_availableEspeakLanguages(self):
        out = OrderedDict()
        try:
            if _espeak_engine is not None:
                for code, label in _espeak_engine.available_languages():
                    out[code] = StringParameterInfo(code, label)
        except Exception:
            pass
        if not out:
            out["ar"] = StringParameterInfo("ar", _("Arabic"))
        return out

    # NVDA derives the availableX property from the setting id via
    # id.capitalize(), which LOWERCASES the rest — so "espeakLanguage" ->
    # "Espeaklanguage" and it looks up _get_availableEspeaklanguages (lower L).
    # Provide that spelling so the lookup resolves.
    def _get_availableEspeaklanguages(self):
        return self._get_availableEspeakLanguages()

    def _get_espeakLanguage(self):
        return getattr(self, "_espeakLanguage", "ar")

    def _set_espeakLanguage(self, value):
        self._espeakLanguage = value

    def _get_availableEspeakSecondaryLanguages(self):
        return self._get_availableEspeakLanguages()

    # NVDA capitalize() lookup -> availableEspeaksecondarylanguages (lower s/l)
    def _get_availableEspeaksecondarylanguages(self):
        return self._get_availableEspeakLanguages()

    def _get_espeakSecondaryLanguage(self):
        return getattr(self, "_espeakSecondaryLanguage", "en-us")

    def _set_espeakSecondaryLanguage(self, value):
        self._espeakSecondaryLanguage = value

    def _get_availableEspeakVariants(self):
        out = OrderedDict()
        try:
            if _espeak_engine is not None:
                for vid, label in _espeak_engine.available_variants():
                    out[vid] = StringParameterInfo(vid, label)
        except Exception:
            pass
        if not out:
            out["none"] = StringParameterInfo("none", _("None"))
        return out

    # NVDA capitalize() lookup -> availableEspeakvariants (lower v)
    def _get_availableEspeakvariants(self):
        return self._get_availableEspeakVariants()

    def _get_espeakVariant(self):
        return getattr(self, "_espeakVariant", "none")

    def _set_espeakVariant(self, value):
        self._espeakVariant = value

    def _get_availableTashkeels(self):
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
            import ar_tashkeel
            names = ar_tashkeel.available()
        except Exception:
            names = ["off"]
        for n in names:
            out[n] = StringParameterInfo(n, labels.get(n, n))
        if not out:
            out["off"] = StringParameterInfo("off", labels["off"])
        return out

    # casing alias so NVDA's capitalize()-derived lookup also resolves
    def _get_availableTashkeels_alias(self):
        return self._get_availableTashkeels()

    def _get_tashkeel(self):
        return getattr(self, "_tashkeel", "libtashkeel")

    def _set_tashkeel(self, value):
        self._tashkeel = value
        try:
            import ar_tashkeel
            ar_tashkeel.set_backend(value)
        except Exception:
            pass
        try:
            if _ar_g2p_mod is not None:
                _ar_g2p_mod._NEURAL_ON = (value != "off")
        except Exception:
            pass

    def _durationScale(self):
        # lower duration scale = faster speech. Rate boost pushes the fast end
        # further so the top of the rate slider is much quicker.
        fast_end = 0.28 if self._get_rateBoost() else 0.42
        return 2.2 * ((fast_end / 2.2) ** (self._rate / 100.0))

    def _baseF0(self, off=0):
        p = max(0, min(100, self._pitch + off))
        f0 = 62.0 * (2.0 ** (p / 100.0 * 1.5))
        if self._voice == "clara":
            f0 *= 1.75
        return f0

    def _formantScale(self):
        return 1.18 if self._voice == "clara" else 1.0

    def speak(self, speechSequence):
        # A new utterance clears cancellation, but we key "am I cancelled?"
        # on the generation counter, NOT a shared flag — otherwise clearing
        # the flag here would revive a still-running previous synthesis and
        # the two would overlap. Each queued item carries the generation it
        # belongs to; the worker abandons any item whose generation is stale.
        self._cancelFlag.clear()
        gen = self._gen
        pitchOffset = 0
        charMode = False
        for item in speechSequence:
            if isinstance(item, str):
                self._queue.put(("text", item, charMode, pitchOffset, gen))
            elif IndexCommand and isinstance(item, IndexCommand):
                self._queue.put(("index", item.index, gen))
            elif BreakCommand and isinstance(item, BreakCommand):
                self._queue.put(("break", item.time, gen))
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
        self._gen += 1          # stale-mark everything queued/in-flight
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

    def _resamplePcm(self, pcm, src_sr, dst_sr):
        """Nearest-neighbour resample of 16-bit mono PCM. Cheap and good
        enough for a formant fallback; avoids a numpy dependency here."""
        try:
            import array
            src = array.array("h")
            src.frombytes(pcm)
            n_src = len(src)
            if n_src == 0 or src_sr == dst_sr:
                return pcm
            n_dst = int(n_src * dst_sr / float(src_sr))
            out = array.array("h", bytes(2 * n_dst))
            ratio = src_sr / float(dst_sr)
            for i in range(n_dst):
                out[i] = src[int(i * ratio)]
            return out.tobytes()
        except Exception:
            return pcm

    def _speakEspeak(self, text, charMode, cancelled):
        """Render `text` with the eSpeak NG multilingual engine and feed it
        to the player. Returns True if audio was produced.

        eSpeak NG is proficient in every language, so — unlike the Neural
        driver, which sends only Arabic to the diacritizer and routes other
        languages to their own voices — the Formant driver sends the WHOLE
        segment (Arabic and English/French/etc. together) to the diacritizer
        and then to eSpeak, which reads all of it itself. Diacritization only
        adds marks to the Arabic words and leaves the rest untouched, so the
        non-Arabic parts are preserved and spoken (they used to be dropped
        when only the Arabic parts were sent)."""
        if not text or not text.strip():
            return False
        spk_text = text
        try:
            if (self._get_tashkeel() != "off"
                    and _ar_g2p_mod is not None
                    and any("\u0600" <= c <= "\u06FF" for c in text)):
                try:
                    # diacritize the WHOLE segment; _neural_pre marks the
                    # Arabic words and passes English/other through unchanged
                    spk_text = _ar_g2p_mod._neural_pre(text)
                except Exception:
                    spk_text = text
        except Exception:
            spk_text = text
        # map ClaritySynth rate/pitch/volume (0..100-ish) to eSpeak ranges.
        # rate boost extends the top of the WPM range for much faster speech.
        top_wpm = 700 if self._get_rateBoost() else 350
        wpm = int(90 + (self._rate / 100.0) * (top_wpm - 90))
        pitch = int(max(0, min(100, self._pitch)))
        vol = int(max(0, min(200, self._volume * 2)))       # 0..200
        lang = self._get_espeakLanguage()
        variant = self._get_espeakVariant()
        secondary = self._get_espeakSecondaryLanguage()
        try:
            # split_scripts=True: Arabic-script runs are read with the chosen
            # Arabic voice, and non-Arabic (Latin) runs with the chosen SECOND
            # language (French, English, etc.) — so "يا guys" reads BOTH parts,
            # each in its own language, and a French run (text AND numbers) is
            # read in French, not accented English.
            pcm = _espeak_engine.synth_pcm(
                spk_text, voice=lang, variant=variant,
                rate_wpm=wpm, pitch=pitch, volume=vol,
                secondary_voice=secondary, split_scripts=True)
        except Exception:
            pcm = b""
        if not pcm:
            return False
        # eSpeak renders at its own sample rate; if that differs from the
        # player's rate, resample (nearest-neighbour) so pitch/tempo are right
        try:
            esr = _espeak_engine.sample_rate()
        except Exception:
            esr = engine.SR
        if esr and esr != engine.SR:
            pcm = self._resamplePcm(pcm, esr, engine.SR)
        if cancelled():
            return True
        # feed in blocks so cancellation stays responsive
        step = 4096
        for i in range(0, len(pcm), step):
            if cancelled():
                break
            self._feed(pcm[i:i + step])
        return True

    def _worker(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            try:
                kind = item[0]
                item_gen = item[-1] if isinstance(item[-1], int) else None
                # abandon anything from a superseded utterance
                stale = (item_gen is not None and item_gen != self._gen)
                cancelled = (lambda: self._cancelFlag.is_set()
                             or (item_gen is not None
                                 and item_gen != self._gen))
                if kind == "index":
                    if not stale:
                        synthIndexReached.notify(synth=self, index=item[1])
                elif kind == "break":
                    if not stale:
                        ms = item[1]
                        self._feed(
                            b"\x00\x00" * int(engine.SR * ms / 1000.0))
                elif kind == "text" and not stale:
                    _, text, charMode, pitchOffset = item[0], item[1], \
                        item[2], item[3]
                    # eSpeak NG engine: it takes text directly and renders
                    # audio itself (its own multilingual formant voice), so
                    # it bypasses the token/frame path entirely.
                    if self._get_engine() == "espeak" \
                            and _espeak_engine is not None \
                            and _espeak_engine.is_available():
                        if self._speakEspeak(text, charMode, cancelled):
                            continue
                        # if eSpeak produced nothing, fall through to the
                        # formant engines below
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
                        # diacritize Arabic using the selected tashkeel
                        # library (Libtashkeel / Rawi / CATT); "off" leaves
                        # the text exactly as written
                        if (self._get_tashkeel() != "off"
                                and _ar_g2p_mod is not None
                                and any("\u0600" <= c <= "\u06FF"
                                        for c in text)):
                            try:
                                spk_text = _ar_g2p_mod._neural_pre(text)
                            except Exception:
                                spk_text = text
                        tokens = g2p.text_to_tokens(spk_text)
                    if not tokens:
                        continue
                    # choose engine per the setting: auto -> DLL if loaded
                    # else built-in; dll -> force DLL if present; builtin ->
                    # always the pure-Python engine
                    _eng = self._get_engine()
                    _b = _get_dll_bridge() if _eng in ("auto", "dll") \
                        else None
                    if _eng == "builtin" or _b is None:
                        _synth = engine.synthesize
                    else:
                        _synth = _b.synthesize
                    gen = _synth(
                        tokens, dscale=self._durationScale(),
                        base_f0=self._baseF0(pitchOffset),
                        inflection=self._inflection / 100.0,
                        volume=self._volume / 100.0,
                        is_cancelled=cancelled,
                        fscale=self._formantScale(),
                        breath_amt=self._breathiness / 100.0,
                        jitter=self._roughness / 100.0 * 0.6,
                        shimmer=self._roughness / 100.0 * 0.5)
                    for block in gen:
                        if cancelled():
                            break
                        self._feed(block)
                if self._queue.empty() and not stale \
                        and not self._cancelFlag.is_set():
                    self._player.idle()
                    synthDoneSpeaking.notify(synth=self)
            except Exception:
                log.error("claritysynthformant worker error",
                          exc_info=True)
