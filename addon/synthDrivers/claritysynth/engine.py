# -*- coding: utf-8 -*-
# ClaritySynth: cascade formant synthesis engine (v2).
#
# Signal chain:
#   Rosenberg glottal pulse train (jitter+shimmer) --+
#   aspiration / breath noise                       --+-> R1..R3 (moving)
#                                                        -> R4, R5 (fixed)
#   frication noise -> parallel resonator ------------------------> (+) -> out
#
# v2 changes: 22.05 kHz output; formant transitions confined to the first
# ~30 ms of each segment (fixes the "bw" gliding artifact); Rosenberg
# glottal source with jitter, shimmer and breathiness instead of a bare
# impulse train; extra fixed formant for presence; per-voice formant
# scaling; geminate-aware rendering including a true multi-contact trill
# for the doubled Arabic raa.

import math
import random
from array import array

try:
    from . import phonemes
except ImportError:  # standalone testing outside NVDA
    import phonemes

SR = 22050
FRAME_MS = 5
FRAME = SR * FRAME_MS // 1000
CHUNK_FRAMES = 40

_TWO_PI = 2.0 * math.pi
_BW1, _BW2, _BW3 = 60.0, 90.0, 130.0   # after NV Speech Player
_NEUTRAL = (500.0, 1400.0, 2400.0)


def _coeffs(f, bw):
    r = math.exp(-math.pi * bw / SR)
    c = -(r * r)
    b = 2.0 * r * math.cos(_TWO_PI * f / SR)
    return (1.0 - b - c, b, c)


def _fric_envelope(frames, n):
    for j in range(n):
        t = (j + 1.0) / n
        env = min(1.0, t / 0.45)
        if t > 0.78:
            env *= 1.0 - 0.35 * (t - 0.78) / 0.22
        frames[-n + j][5] *= env


def build_track(tokens, dscale=1.0, base_f0=110.0, inflection=0.5,
                accent=1.0, pause_scale=1.0):
    """tokens -> acoustic frames [f1,f2,f3,av,ah,af,ff,fbw,f0]."""
    PH = phonemes.PHONEMES
    PAUSES = phonemes.PAUSES

    frames = []
    clause_marks = []
    cur = list(_NEUTRAL)
    word_start = True

    def n_frames(ms):
        return max(1, int(round(ms * dscale / FRAME_MS)))

    def add(n, target, av, ah, af, ff, fbw, stress, glide_to=None,
            trans=None, bw=None):
        b1, b2, b3 = bw if bw else (_BW1, _BW2, _BW3)
        """Append n frames. Formants move cur->target over the first
        `trans` frames only, then hold (or glide to glide_to)."""
        start = tuple(cur)
        target = (target[0] * (1.0 + (random.random() - 0.5) * 0.028),
                  target[1] * (1.0 + (random.random() - 0.5) * 0.028),
                  target[2] * (1.0 + (random.random() - 0.5) * 0.020))
        if trans is None:
            trans = max(2, min(6, n // 2))          # <= 30 ms transition
        for i in range(n):
            if glide_to is None:
                if i < trans:
                    t = (i + 1.0) / trans
                    f1 = start[0] + (target[0] - start[0]) * t
                    f2 = start[1] + (target[1] - start[1]) * t
                    f3 = start[2] + (target[2] - start[2]) * t
                else:
                    f1, f2, f3 = target
            else:
                t = (i + 1.0) / n
                if t < 0.35:
                    u = t / 0.35
                    f1 = start[0] + (target[0] - start[0]) * u
                    f2 = start[1] + (target[1] - start[1]) * u
                    f3 = start[2] + (target[2] - start[2]) * u
                else:
                    u = (t - 0.35) / 0.65
                    f1 = target[0] + (glide_to[0] - target[0]) * u
                    f2 = target[1] + (glide_to[1] - target[1]) * u
                    f3 = target[2] + (glide_to[2] - target[2]) * u
            frames.append([f1, f2, f3, av, ah, af, ff, fbw, stress,
                           b1, b2, b3])
        cur[0], cur[1], cur[2] = frames[-1][0], frames[-1][1], frames[-1][2]

    def next_sonorant_target(idx):
        for tok in tokens[idx + 1:idx + 5]:
            if tok and tok[-1] in "012" and tok not in PH:
                tok = tok[:-1]
            d = PH.get(tok)
            if d and d["kind"] in ("vowel", "diph", "liquid", "glide"):
                return d["f"]
        return tuple(cur)

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "_w":
            word_start = True
            i += 1
            continue
        if tok in PAUSES:
            if frames and frames[-1][3] > 0.05:
                lastf = frames[-1]
                for fade in (0.55, 0.28, 0.12):
                    frames.append([lastf[0], lastf[1], lastf[2],
                                   lastf[3] * fade, 0.0, 0.0,
                                   1000.0, 500.0, lastf[8],
                                   lastf[9], lastf[10], lastf[11]])
            n = n_frames(PAUSES[tok] * pause_scale)
            for _ in range(n):
                frames.append([cur[0], cur[1], cur[2],
                               0.0, 0.0, 0.0, 1000.0, 500.0, 1.0,
                               _BW1, _BW2, _BW3])
            if tok in ("_.", "_?", "_!", "_;", "_:", "_,"):
                clause_marks.append((len(frames), tok))
            i += 1
            continue

        stress_mark = False
        if tok and tok[-1] in "012" and tok not in PH:
            stress_mark = tok.endswith("1")
            tok = tok[:-1]
        d = PH.get(tok)
        if d is None:
            i += 1
            continue
        rep = 1
        while i + rep < len(tokens):
            t2 = tokens[i + rep]
            if t2 and t2[-1] in "012" and t2 not in PH:
                if t2.endswith("1"):
                    stress_mark = True
                t2 = t2[:-1]
            if t2 == tok:
                rep += 1
            else:
                break
        mult = 1.0 + 0.85 * (rep - 1)
        kind = d["kind"]
        stress = 1.0

        if kind in ("vowel", "diph"):
            if stress_mark:
                stress = 1.07
                word_start = False
            elif word_start:
                stress = 1.04
                word_start = False
            prepause = any(t in PAUSES for t in tokens[i + rep:i + rep + 3])
            n = n_frames(d["dur"] * mult * (1.12 if stress_mark else 1.0)
                         * (1.25 if prepause else 1.0))
            if kind == "vowel":
                add(n, d["f"], 1.0 * (1.05 if stress > 1 else 1.0),
                    0.0, 0.0, 1000.0, 500.0, stress, bw=d.get("bw"))
            else:
                add(n, d["f"], 1.0 * (1.05 if stress > 1 else 1.0),
                    0.0, 0.0, 1000.0, 500.0, stress, glide_to=d["f_end"])

        elif kind in ("nasal", "liquid", "glide"):
            n = n_frames(d["dur"] * mult)
            add(n, d["f"], d["amp"], 0.0, 0.0, 1000.0, 500.0, 1.0,
                bw=d.get("bw"))

        elif kind == "aspf":
            n = n_frames(d["dur"] * mult)
            add(n, d["f"], 0.0, d["amp"], 0.0, 1000.0, 500.0, 1.0)

        elif kind == "asp":
            tgt = next_sonorant_target(i)
            n = n_frames(d["dur"] * mult)
            add(n, tgt, 0.0, 0.34, 0.0, 1000.0, 500.0, 1.0)

        elif kind == "fric":
            n = n_frames(d["dur"] * mult)
            av = 0.30 if d["voiced"] else 0.0
            add(n, d["f"], av, 0.0, d["amp"], d["ff"], d["fbw"], 1.0)
            _fric_envelope(frames, n)

        elif kind == "stop":
            if d.get("tap"):
                # Arabic raa. F3 stays HIGH (~2600): dropping it is what
                # makes an English r. Fully voiced contacts at a natural
                # trill rate (~28 Hz). Intervocalic single raa is one
                # tap; initial/final/preconsonantal raa gets two
                # contacts (MSA usage); shadda gives a full trill.
                tgt = next_sonorant_target(i)
                nxt = tokens[i + rep] if i + rep < len(tokens) else ""
                if nxt and nxt[-1] in "012" and nxt not in PH:
                    nxt = nxt[:-1]
                nd = PH.get(nxt)
                next_vowel = bool(nd) and nd["kind"] in ("vowel", "diph")
                prev_vowel = bool(frames) and frames[-1][3] > 0.6
                # relaxed: one light contact for single raa, two under
                # shadda. Shallow dip, smooth entry and exit.
                contacts = 2 if rep >= 2 else 1
                del prev_vowel, next_vowel
                dip = d["f"]
                for c in range(contacts):
                    add(2, dip, 0.52, 0.0, 0.0, 1000.0, 500.0, 1.0,
                        trans=2)
                    if c < contacts - 1:
                        add(4, tgt, 1.0, 0.0, 0.0, 1000.0, 500.0, 1.0,
                            trans=2)
                    else:
                        add(2, tgt, 0.95, 0.0, 0.0, 1000.0, 500.0, 1.0,
                            trans=2)
                i += rep
                continue
            total = n_frames(d["dur"])
            closure = max(1, int(total * 0.55 * (1.0 + 0.9 * (rep - 1))))
            burst = max(1, int(total * 0.15))
            release = max(1, total - max(1, int(total * 0.55)) - burst)
            voicebar = 0.14 if d["voiced"] else 0.0
            add(closure, d["f"], voicebar, 0.0, 0.0, 1000.0, 500.0, 1.0)
            if d.get("glottal"):
                add(burst + release, d["f"], 0.0, 0.0, 0.0,
                    1000.0, 500.0, 1.0)
                i += rep
                continue
            add(burst, d["f"], voicebar, 0.0, 0.55,
                d["burstf"], d["burstbw"], 1.0)
            if d["voiced"]:
                # release starts already well on the way to the next sound
                tgt = next_sonorant_target(i)
                mid = tuple(d["f"][k] + 0.65 * (tgt[k] - d["f"][k])
                            for k in range(3))
                add(release, mid, 0.55, 0.0, 0.0, 1000.0, 500.0, 1.0,
                    trans=1)
            else:
                tgt = next_sonorant_target(i)
                mid = tuple(d["f"][k] + 0.6 * (tgt[k] - d["f"][k])
                            for k in range(3))
                add(release, mid, 0.0, 0.20, 0.0, 1000.0, 500.0, 1.0,
                    trans=1)

        elif kind == "affric":
            sd = PH[d["stop"]]
            fd = PH[d["fric"]]
            total = n_frames(d["dur"] * mult)
            closure = max(1, int(total * 0.4))
            friclen = max(1, total - closure)
            voicebar = 0.14 if d["voiced"] else 0.0
            add(closure, sd["f"], voicebar, 0.0, 0.0, 1000.0, 500.0, 1.0)
            av = 0.30 if d["voiced"] else 0.0
            add(friclen, fd["f"], av, 0.0, fd["amp"], fd["ff"], fd["fbw"],
                1.0)
            _fric_envelope(frames, friclen)

        i += rep

    if not frames:
        return []
    if not clause_marks or clause_marks[-1][0] < len(frames):
        clause_marks.append((len(frames), "_."))

    # ------------------------------------------------------------------
    # Prosody: superposition intonation model.
    #   f0 = base * phrase(pos) * (1 + accent(t)) * micro * drift
    # Accent gestures are smoothed pulses on stressed syllables; the
    # phrase component is an exponential declination with boundary
    # tones; microprosody follows the segmental structure; a slow
    # random-walk drift keeps long stretches from sounding mechanical.
    # ------------------------------------------------------------------
    infl = max(0.0, min(1.0, inflection))
    N = len(frames)

    # accent pulses at the onsets of stressed vowels
    acc = [0.0] * N
    prev_s = 1.0
    for idx in range(N):
        s = frames[idx][8]
        if s > 1.0 and prev_s <= 1.0:
            acc[idx] = (s - 1.0) * 10.0     # 1.04 -> 0.4, 1.07 -> 0.7
        prev_s = s
    # zero-phase smoothing -> bell-shaped rise-fall gestures
    a = 0.16
    s = 0.0
    for idx in range(N):
        s += a * (acc[idx] - s)
        acc[idx] = s
    s = 0.0
    for idx in range(N - 1, -1, -1):
        s += a * (acc[idx] - s)
        acc[idx] = s
    acc_gain = 0.95 * (0.3 + 1.4 * infl) * accent

    out = []
    start = 0
    drift = 0.0
    post_onset = 0
    prev_av = 0.0
    for mark_idx, punct in clause_marks:
        span = frames[start:mark_idx]
        n = len(span)
        for j, fr in enumerate(span):
            gidx = start + j
            pos = j / max(1, n - 1)
            k = 0.35 + 1.3 * infl
            # phrase component: high reset, exponential fall
            ph = 0.955 + 0.21 * math.exp(-2.4 * pos) * k
            if punct == "_,":
                ph = 0.985 + 0.075 * math.exp(-2.6 * pos) * k
                if pos > 0.72:                    # continuation rise
                    u = (pos - 0.72) / 0.28
                    ph += 0.09 * u * u * (3 - 2 * u) * k
            elif punct == "_?":
                if pos > 0.62:
                    u = (pos - 0.62) / 0.38
                    ph += 0.34 * u * u * (3 - 2 * u) * k
            else:
                if pos > 0.82:                    # terminal fall
                    u = (pos - 0.82) / 0.18
                    ph -= 0.17 * u * u * (3 - 2 * u) * k
            # microprosody
            av = fr[3]
            micro = 1.0
            if 0.05 < av < 0.45:                  # voiced obstruent dip
                micro *= 0.965
            if av > 0.55 and prev_av < 0.10:
                post_onset = 3                    # rise after voiceless
            if post_onset > 0 and av > 0.4:
                micro *= 1.0 + 0.010 * post_onset
                post_onset -= 1
            prev_av = av
            # slow random drift, scaled by inflection
            drift += (random.random() - 0.5) * 0.006 * (0.2 + infl)
            drift *= 0.988
            f0 = (base_f0 * ph * (1.0 + acc_gain * acc[gidx])
                  * micro * (1.0 + drift))
            # amplitude declination + slight accent loudness
            av_out = fr[3] * (1.0 - 0.10 * pos) * (1.0 + 0.25 * acc[gidx])
            out.append((fr[0], fr[1], fr[2], av_out, fr[4], fr[5],
                        fr[6], fr[7], f0, fr[9], fr[10], fr[11]))
        start = mark_idx

    # final zero-step smoothing of the pitch contour
    if out:
        sm = 0.0
        first = True
        smoothed = []
        for fr in out:
            if first:
                sm = fr[8]
                first = False
            sm += 0.22 * (fr[8] - sm)
            smoothed.append(fr[:8] + (sm,) + fr[9:])
        out = smoothed
    return out


def render(frames, volume=1.0, is_cancelled=None, fscale=1.0,
           breath_amt=0.05, jitter=0.016, shimmer=0.10):
    """Frames -> 16-bit mono PCM chunks. Rosenberg glottal source with
    jitter, shimmer and breath; five-resonator cascade + frication."""
    if is_cancelled is None:
        is_cancelled = lambda: False

    a4, b4, c4 = _coeffs(3300.0 * fscale, 250.0)
    a5, b5, c5 = _coeffs(3750.0 * fscale, 220.0)
    y11 = y12 = y21 = y22 = y31 = y32 = 0.0
    y41 = y42 = y51 = y52 = 0.0
    fy1 = fy2 = 0.0
    lp = 0.0
    hp = 0.0
    prev = 0.0
    seed = 22222
    gain = 8200.0 * max(0.0, min(1.0, volume))

    # glottal state
    tpp = 0.0            # samples into current pitch period
    period = 200.0
    Lg = 120.0
    prevflow = 0.0
    ampj = 1.0

    buf = array("h")
    frames_in_buf = 0

    for fr in frames:
        if is_cancelled():
            break
        f1, f2, f3, av, ah, af, ff, fbw, f0, b1c, b2c, b3c = fr
        a1, b1, c1 = _coeffs(f1 * fscale, b1c)
        a2, b2, c2 = _coeffs(f2 * fscale, b2c)
        a3, b3, c3 = _coeffs(f3 * fscale, b3c)
        if af > 0.0:
            fa, fb, fc = _coeffs(ff * fscale, fbw)
        else:
            fa = fb = fc = 0.0
        base_period = SR / max(30.0, f0)
        av9 = av * 5.0
        breath = av * breath_amt
        ah35 = ah * 0.42
        af2 = af * 0.75

        n = 0
        while n < FRAME:
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            noise = (seed >> 16) / 16384.0 - 1.0
            # glottal source: Rosenberg flow derivative
            tpp += 1.0
            if tpp >= period:
                tpp -= period
                seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
                j1 = (seed >> 16) / 32768.0 - 0.5
                period = base_period * (1.0 + jitter * j1)
                Lg = period * 0.62
                seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
                ampj = 1.0 + shimmer * ((seed >> 16) / 32768.0 - 0.5)
                prevflow = 0.0
            if tpp < Lg:
                kk = tpp / Lg
                flow = kk * kk * (3.0 - 2.0 * kk)
            else:
                flow = 0.0
            dv = flow - prevflow
            prevflow = flow
            # light spectral shaping of the source
            lp += 0.45 * (dv - lp)
            x = lp * av9 * ampj + noise * (ah35 + breath * av9 * 0.012)
            y = a1 * x + b1 * y11 + c1 * y12
            y12 = y11; y11 = y
            y = a2 * y + b2 * y21 + c2 * y22
            y22 = y21; y21 = y
            y = a3 * y + b3 * y31 + c3 * y32
            y32 = y31; y31 = y
            y = a4 * y + b4 * y41 + c4 * y42
            y42 = y41; y41 = y
            y = a5 * y + b5 * y51 + c5 * y52
            y52 = y51; y51 = y
            if af2:
                fy = fa * noise + fb * fy1 + fc * fy2
                fy2 = fy1; fy1 = fy
                y += fy * af2
            hp = y - prev + 0.985 * hp
            prev = y
            s = int(hp * gain)
            if s > 32000:
                s = 32000
            elif s < -32000:
                s = -32000
            buf.append(s)
            n += 1

        frames_in_buf += 1
        if frames_in_buf >= CHUNK_FRAMES:
            yield buf.tobytes()
            del buf[:]
            frames_in_buf = 0

    if len(buf):
        yield buf.tobytes()


def synthesize(tokens, dscale=1.0, base_f0=110.0, inflection=0.5,
               volume=1.0, is_cancelled=None, fscale=1.0,
               accent=1.0, pause_scale=1.0, breath_amt=0.05,
               jitter=0.016, shimmer=0.10):
    frames = build_track(tokens, dscale=dscale, base_f0=base_f0,
                         inflection=inflection, accent=accent,
                         pause_scale=pause_scale)
    return render(frames, volume=volume, is_cancelled=is_cancelled,
                  fscale=fscale, breath_amt=breath_amt, jitter=jitter,
                  shimmer=shimmer)
