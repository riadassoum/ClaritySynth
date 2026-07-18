# -*- coding: utf-8 -*-
"""Bridge: ClaritySynth phoneme track -> TGSpeechBox speechPlayer.dll
(GPL2, Tamas Geczy / NV Access).

v5.2. Uses the DLL's FULL surface discovered by reading its source:
  * queueFrameEx()  -> breathiness / jitter / shimmer / within-frame
                       formant ramps / equal-power crossfades
  * setVoicingTone() -> global source shaping: high-shelf presence,
                       chorus (fold asymmetry), tremor, aspiration tilt
  * Fujisaki fields  -> smooth phrase+accent pitch contour in the DSP
Pharyngeal haa is rendered as its own turbulent fricative band so it can
never collapse to glottal /h/. The pure-Python engine remains fallback.
"""
import os
import sys
import ctypes
from ctypes import c_short, POINTER, cast

try:
    from . import engine as pyengine   # imported as claritysynth.dll_engine
except ImportError:
    import engine as pyengine           # imported as a loose module (Formant)

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "sp"))
try:
    import speechPlayer as _sp
except Exception:
    _sp = None

CHUNK = 4096
_NAN = float("nan")


class Bridge(object):
    def __init__(self):
        if _sp is None:
            raise RuntimeError("speechPlayer module unavailable")
        old = os.getcwd()
        try:
            os.chdir(os.path.join(_here, "sp"))
            self.player = _sp.SpeechPlayer(pyengine.SR)
        finally:
            os.chdir(old)
        self.hasEx = self.player.hasFrameExSupport()
        self._applyTone(0.0)

    def _applyTone(self, rough):
        """Global voice source shaping. rough in [0,1]."""
        try:
            tone = _sp.VoicingTone.defaults()
            tone.highShelfGainDb = 5.0        # presence / less muffled
            tone.highShelfFcHz = 2200.0
            tone.voicedTiltDbPerOct = -1.0    # a little warmth
            tone.chorusDepth = 0.12 + 0.25 * rough
            tone.chorusDetuneHz = 2.5
            tone.tremorDepth = 0.05 * rough
            tone.aspirationTiltDbPerOct = -6.0  # aspiration darker than
            #                                     frication -> h vs haa
            self.player.setVoicingTone(tone)
        except Exception:
            pass

    def _frame(self, fr, nxt, fscale, volume):
        f = _sp.Frame()
        f.voicePitch = fr[8]
        f.endVoicePitch = nxt[8] if nxt else fr[8]
        f.voiceAmplitude = min(1.0, fr[3])
        f.glottalOpenQuotient = 0.62
        # explicit: no turbulence unless the profile asks (kills the
        # "breathy at 0" bug - the DLL default was non-zero)
        f.voiceTurbulenceAmplitude = 0.0
        f.vibratoPitchOffset = 0.0
        f.vibratoSpeed = 0.0
        f.cf1, f.cf2, f.cf3 = fr[0] * fscale, fr[1] * fscale, fr[2] * fscale
        f.cb1, f.cb2, f.cb3 = fr[9], fr[10], fr[11]
        f.cf4, f.cf5, f.cf6 = 3300.0, 3750.0, 4900.0
        f.cb4, f.cb5, f.cb6 = 250.0, 220.0, 1000.0
        f.cfN0, f.cfNP = 270.0, 250.0
        f.cbN0, f.cbNP = 100.0, 100.0
        f.caNP = 0.0
        f.aspirationAmplitude = fr[4]         # /h/ and stop release only
        # fixed parallel centres (no glide-through-vowel chirp)
        f.pf1, f.pf2, f.pf3 = 300.0, 1300.0, 2500.0
        f.pf4, f.pf5, f.pf6 = 3300.0, 5000.0, 6500.0
        f.pb1, f.pb2, f.pb3 = 120.0, 400.0, 500.0
        f.pb4, f.pb5, f.pb6 = 600.0, 900.0, 1200.0
        af = fr[5]
        f.fricationAmplitude = af
        if af > 0.0:
            ff, fbw = fr[6], fr[7]
            if ff <= 1600.0:          # haa (760/1150 loci, low noise)
                f.pf2, f.pb2, f.pa2 = ff, fbw, 1.0
                f.pf1, f.pa1 = 700.0, 0.5   # pharyngeal F1 colour
            elif ff <= 3200.0:        # shin, khaa
                f.pf3, f.pb3, f.pa3 = ff, fbw, 1.0
            else:                     # seen, saad, thaa, faa
                f.pf5, f.pb5, f.pa5 = ff, fbw, 1.0
            f.parallelBypass = 0.10
        f.preFormantGain = 2.3 * volume
        f.outputGain = 1.6
        return f

    def _frameEx(self, fr, nxt, fscale, breath_amt, jitter, shimmer):
        ex = _sp.FrameEx()
        for name, _ in ex._fields_:
            setattr(ex, name, 0.0)
        ex.breathiness = min(1.0, breath_amt)
        ex.jitter = min(1.0, jitter)
        ex.shimmer = min(1.0, shimmer)
        ex.sharpness = 0.0
        if nxt is not None:
            ex.endCf1 = nxt[0] * fscale
            ex.endCf2 = nxt[1] * fscale
            ex.endCf3 = nxt[2] * fscale
        else:
            ex.endCf1 = ex.endCf2 = ex.endCf3 = _NAN
        ex.endPf1 = ex.endPf2 = ex.endPf3 = _NAN
        ex.transAmplitudeMode = 1.0
        ex.cf7, ex.cb7 = 6500.0, 720.0
        ex.cf8, ex.cb8 = 7500.0, 1250.0
        ex.fricationTiltDb = 0.0
        return ex

    def synthesize(self, tokens, dscale=1.0, base_f0=110.0,
                   inflection=0.5, volume=1.0, is_cancelled=None,
                   fscale=1.0, accent=1.0, pause_scale=1.0,
                   breath_amt=0.05, jitter=0.016, shimmer=0.10):
        if is_cancelled is None:
            is_cancelled = lambda: False
        self._applyTone(min(1.0, shimmer * 2.0))
        frames = pyengine.build_track(
            tokens, dscale=dscale, base_f0=base_f0,
            inflection=inflection, accent=accent,
            pause_scale=pause_scale)
        n = len(frames)
        first = True
        for i, fr in enumerate(frames):
            if is_cancelled():
                return
            nxt = frames[i + 1] if i + 1 < n else None
            f = self._frame(fr, nxt, fscale, volume)
            if self.hasEx:
                ex = self._frameEx(fr, nxt, fscale, breath_amt, jitter,
                                   shimmer)
                self.player.queueFrameEx(f, ex, pyengine.FRAME_MS, 3,
                                         -1, first)
            else:
                self.player.queueFrame(f, pyengine.FRAME_MS, 3, -1, first)
            first = False
        if frames:
            tail = _sp.Frame()
            lf = frames[-1]
            tail.voicePitch = tail.endVoicePitch = lf[8]
            tail.voiceTurbulenceAmplitude = 0.0
            tail.cf1, tail.cf2, tail.cf3 = 500.0, 1400.0, 2400.0
            tail.cb1, tail.cb2, tail.cb3 = 60.0, 90.0, 130.0
            tail.cf4, tail.cf5, tail.cf6 = 3300.0, 3750.0, 4900.0
            tail.cb4, tail.cb5, tail.cb6 = 250.0, 220.0, 1000.0
            tail.pf2, tail.pf3, tail.pf5 = 1300.0, 2500.0, 5000.0
            tail.pb2, tail.pb3, tail.pb5 = 400.0, 500.0, 900.0
            tail.preFormantGain = 2.3 * volume
            tail.outputGain = 1.6
            self.player.queueFrame(tail, 90, 45, -1, False)
        buf = (c_short * CHUNK)()
        while not is_cancelled():
            got = self.player._dll.speechPlayer_synthesize(
                self.player._speechHandle, CHUNK,
                cast(buf, POINTER(c_short)))
            if not got or got <= 0:
                break
            yield bytes(memoryview(buf)[:got])
