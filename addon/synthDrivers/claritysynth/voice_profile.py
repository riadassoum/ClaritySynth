# -*- coding: utf-8 -*-
"""Voice adaptation backbone. Estimates a speaker profile (median pitch,
vocal-tract length ratio, breathiness) from a 16-bit mono WAV sample and
maps it onto the engine parameters (base_f0, fscale, breath_amt). This is
honest parametric adaptation, not neural cloning; it gives any recorded
voice, English or Arabic, a matching register while keeping the add-on's
own text processing and tashkeel."""
import wave, math
from array import array


def analyze_wav(path):
    w = wave.open(path, "rb")
    sr = w.getframerate()
    a = array("h")
    a.frombytes(w.readframes(min(w.getnframes(), sr * 6)))
    w.close()
    if w.getnchannels() == 2:
        a = a[::2]
    # median f0 by autocorrelation over voiced 40ms windows
    f0s = []
    step = int(sr * 0.04)
    for off in range(0, len(a) - step, step):
        seg = a[off:off + step]
        e = sum(x * x for x in seg) / step
        if e < 2e5:
            continue
        best, bl = 0.0, 0
        for lag in range(int(sr / 320), int(sr / 60)):
            s = 0
            for i in range(0, step - lag, 4):
                s += seg[i] * seg[i + lag]
            if s > best:
                best, bl = s, lag
        if bl:
            f0s.append(sr / bl)
    f0s.sort()
    f0 = f0s[len(f0s) // 2] if f0s else 110.0
    # crude spectral tilt -> breathiness proxy
    hi = sum(abs(a[i] - a[i - 1]) for i in range(1, len(a), 8))
    lo = sum(abs(x) for x in a[::8]) + 1
    tilt = hi / lo
    # pitch movement -> inflection; f0 spread as IQR
    if len(f0s) >= 8:
        iqr = f0s[3 * len(f0s) // 4] - f0s[len(f0s) // 4]
        infl = max(0.2, min(1.0, iqr / max(20.0, f0 * 0.12) * 0.5))
    else:
        infl = 0.6
    return {
        "base_f0": max(60.0, min(300.0, f0)),
        "fscale": max(0.85, min(1.25, (f0 / 110.0) ** 0.35)),
        "breath_amt": max(0.01, min(0.25, (tilt - 0.6) * 0.3)),
        "inflection": infl,
    }
