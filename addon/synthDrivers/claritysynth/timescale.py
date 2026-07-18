# -*- coding: utf-8 -*-
"""WSOLA time-scale compression (glitch-free, low-latency).

Speeds up already-synthesized audio WITHOUT changing pitch and WITHOUT
touching phonemes. Uses Waveform Similarity Overlap-Add: each output frame
is taken from the input position (within a small search window) whose
waveform best matches the previous frame's continuation. This pitch-
synchronous alignment removes the metallic/buzzy artifacts that plain OLA
produces, giving natural fast speech.

Operates on int16 PCM bytes. Designed to be light: a coarse correlation
search (step 4) keeps CPU per call low so there is no added latency.
"""
import numpy as np


def compress_pcm(pcm_bytes, factor, sr, frame_ms=30, seek_ms=6):
    """factor > 1 makes audio faster/shorter. <= 1.02 is a no-op.
    Pitch and timbre preserved; only duration changes."""
    if factor is None or factor <= 1.02 or not pcm_bytes:
        return pcm_bytes
    x = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    n = x.size
    fl = max(96, int(sr * frame_ms / 1000.0))
    if n < fl * 3:
        return pcm_bytes                      # too short to bother
    hop_s = fl // 2                           # output hop (analysis frame /2)
    hop_a_f = hop_s * float(factor)           # ideal input hop (fractional)
    seek = int(sr * seek_ms / 1000.0)
    win = np.hanning(fl).astype(np.float32)

    out_len = int(n / factor) + fl * 4
    out = np.zeros(out_len, dtype=np.float32)
    ow = np.zeros(out_len, dtype=np.float32)

    ia = 0
    io = 0
    consumed = 0
    ideal = 0.0        # exact (fractional) input position; prevents drift
    prev_tail = np.zeros(hop_s, dtype=np.float32)
    while ia + fl < n and io + fl < out_len:
        # WSOLA: within +/- seek of ia, find the frame head whose samples
        # best continue prev_tail (max cross-correlation).
        if seek > 0 and prev_tail.any():
            lo = max(0, ia - seek)
            hi = min(n - fl, ia + seek)
            best = -1e18
            best_off = ia
            # coarse step for speed; fine enough for smoothness
            for off in range(lo, hi + 1, 4):
                head = x[off:off + hop_s]
                if head.shape[0] == hop_s:
                    c = float(np.dot(head, prev_tail))
                    if c > best:
                        best = c
                        best_off = off
            ia = best_off
        seg = x[ia:ia + fl]
        if seg.shape[0] < fl:
            break
        out[io:io + fl] += seg * win
        ow[io:io + fl] += win
        if ia + fl > consumed:
            consumed = ia + fl      # high-water mark of input used
        te = ia + hop_s
        if te + hop_s <= n:
            prev_tail = x[te:te + hop_s].copy()
        else:
            prev_tail = np.zeros(hop_s, dtype=np.float32)
        ideal += hop_a_f
        ia = int(ideal)          # exact rate; no accumulated rounding error
        io += hop_s

    # Normalize the overlap-added region FIRST, then append the leftover
    # tail as clean, unweighted audio. Doing it in this order avoids the
    # bug where the tail was divided by the window weights of the last
    # frame (which halved the final consonant's amplitude).
    ow[ow < 1e-6] = 1.0
    y = (out / ow)[:io + fl]

    # `consumed` = furthest input sample already represented in the output.
    # WSOLA's similarity search can move `ia` backwards, so we must use the
    # high-water mark, not the final `ia`, or we would repeat audio.
    tail = x[consumed:] if consumed < n else None
    if tail is not None and tail.size:
        # The last frame's window tapers to zero at io+fl. Splice the tail
        # in starting where the reconstruction is still at full amplitude
        # (io + hop_s), and cross-fade over the overlap so there is no click.
        splice = io + hop_s
        if splice > y.size:
            splice = y.size
        head = y[:splice]
        xf = min(hop_s // 2, tail.size, head.size)
        if xf > 0:
            ramp = np.linspace(0.0, 1.0, xf, dtype=np.float32)
            head = head.copy()
            head[-xf:] = head[-xf:] * (1.0 - ramp) + tail[:xf] * ramp
            y = np.concatenate([head, tail[xf:]])
        else:
            y = np.concatenate([head, tail])

    m = float(np.max(np.abs(y))) or 1.0
    if m > 32767.0:
        y = y * (32767.0 / m)
    return y.astype(np.int16).tobytes()


def _resample_linear(x, ratio):
    """Linear resample by ratio (out_len = in_len/ratio). Used for pitch
    shifting: time-stretch by ratio then resample back changes pitch."""
    import numpy as np
    n = x.size
    out_n = max(1, int(n / ratio))
    idx = np.linspace(0, n - 1, out_n).astype(np.float32)
    i0 = np.floor(idx).astype(np.int32)
    i1 = np.minimum(i0 + 1, n - 1)
    frac = idx - i0
    return (x[i0] * (1 - frac) + x[i1] * frac).astype(np.float32)


def pitch_shift_pcm(pcm_bytes, semitones, sr):
    """Shift pitch by `semitones` WITHOUT changing duration. Method:
    WSOLA time-stretch by `ratio`, then linear-resample by the same ratio.
    The stretch and the resample lengths cancel, so duration is preserved
    while the pitch changes. Small shifts (a few semitones) stay clean."""
    import numpy as np
    if pcm_bytes is None or abs(semitones) < 0.1 or not pcm_bytes:
        return pcm_bytes
    ratio = 2.0 ** (semitones / 12.0)          # >1 = higher pitch
    # 1) time-stretch to ratio x LONGER (compress factor = 1/ratio < 1
    #    slows it down when ratio>1). compress_pcm only speeds up, so for
    #    stretching we implement it here directly is overkill — instead we
    #    resample first (changes pitch+length) then WSOLA back to original
    #    length.
    x = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    orig_len = x.size
    # resample by ratio: shortens (ratio>1) and raises pitch
    y = _resample_linear(x, ratio)
    # now time-scale y back to orig_len using WSOLA (preserves the new
    # pitch, restores duration)
    cur_len = y.size
    if cur_len < 8:
        return pcm_bytes
    factor = cur_len / orig_len   # >1 if we need to speed up? no: we need
    # to LENGTHEN y from cur_len back to orig_len. compress_pcm speeds up
    # (shortens). If ratio>1, cur_len<orig_len so we must stretch (slow).
    yb = (np.clip(y, -32767, 32767)).astype(np.int16).tobytes()
    if orig_len > cur_len:
        out = _time_stretch(yb, orig_len / cur_len, sr)
    else:
        out = compress_pcm(yb, cur_len / orig_len, sr)
    # Keep duration stable, but NEVER hard-trim: cutting to orig_len would
    # chop the final consonant if the stretch overshot slightly. Pad when
    # short; when long, only trim TRAILING NEAR-SILENCE, never real audio.
    o = np.frombuffer(out, dtype=np.int16)
    if o.size < orig_len:
        o = np.concatenate([o, np.zeros(orig_len - o.size, dtype=np.int16)])
    elif o.size > orig_len:
        excess = o[orig_len:]
        # if everything past orig_len is essentially silent, drop it;
        # otherwise keep the audio (a few ms longer is harmless, a cut is not)
        if excess.size and int(np.max(np.abs(excess.astype(np.int32)))) < 300:
            o = o[:orig_len]
    return o.tobytes()


def _time_stretch(pcm_bytes, factor, sr, frame_ms=30, seek_ms=6):
    """WSOLA time-stretch (factor > 1 = LONGER / slower). Mirror of
    compress_pcm. Preserves pitch; only duration changes.

    Two correctness details that matter for speech:
      * the input position is accumulated as a FLOAT, so integer rounding
        never accumulates and the output length is exact;
      * the leftover tail is spliced in AFTER overlap-add normalization,
        so it is never divided by the last frame's window weights (which
        previously halved or erased the final consonant).
    """
    import numpy as np
    if factor is None or abs(factor - 1.0) < 0.02 or not pcm_bytes:
        return pcm_bytes
    x = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    n = x.size
    fl = max(96, int(sr * frame_ms / 1000.0))
    if n < fl * 3:
        return pcm_bytes

    hop_s = fl // 2                      # output hop
    hop_a_f = hop_s / float(factor)      # ideal (fractional) input hop
    seek = int(sr * seek_ms / 1000.0)
    win = np.hanning(fl).astype(np.float32)

    out_len = int(n * max(1.0, factor)) + fl * 8
    out = np.zeros(out_len, dtype=np.float32)
    ow = np.zeros(out_len, dtype=np.float32)

    ia = 0
    io = 0
    consumed = 0
    ideal = 0.0
    prev_tail = np.zeros(hop_s, dtype=np.float32)

    while ia + fl < n and io + fl < out_len:
        if seek > 0 and prev_tail.any():
            lo = max(0, ia - seek)
            hi = min(n - fl, ia + seek)
            best = -1e18
            best_off = ia
            for off in range(lo, hi + 1, 4):
                head = x[off:off + hop_s]
                if head.shape[0] == hop_s:
                    c = float(np.dot(head, prev_tail))
                    if c > best:
                        best = c
                        best_off = off
            ia = best_off
        seg = x[ia:ia + fl]
        if seg.shape[0] < fl:
            break
        out[io:io + fl] += seg * win
        ow[io:io + fl] += win
        if ia + fl > consumed:
            consumed = ia + fl
        te = ia + hop_s
        prev_tail = (x[te:te + hop_s].copy() if te + hop_s <= n
                     else np.zeros(hop_s, dtype=np.float32))
        ideal += hop_a_f
        ia = int(ideal)
        io += hop_s

    ow[ow < 1e-6] = 1.0
    y = (out / ow)[:io + fl]

    tail = x[consumed:] if consumed < n else None
    if tail is not None and tail.size:
        splice = min(io + hop_s, y.size)
        head = y[:splice]
        xf = min(hop_s // 2, tail.size, head.size)
        if xf > 0:
            ramp = np.linspace(0.0, 1.0, xf, dtype=np.float32)
            head = head.copy()
            head[-xf:] = head[-xf:] * (1.0 - ramp) + tail[:xf] * ramp
            y = np.concatenate([head, tail[xf:]])
        else:
            y = np.concatenate([head, tail])

    m = float(np.max(np.abs(y))) or 1.0
    if m > 32767.0:
        y = y * (32767.0 / m)
    return y.astype(np.int16).tobytes()


def level_envelope(pcm_bytes, sr, win_ms=160, target=6500.0,
                   max_boost=1.8, floor_ratio=0.18, peak_ceiling=20000.0):
    """Even out slow volume drift within an utterance (e.g. a speaker whose
    voice fades toward the end) WITHOUT pushing peaks into the limiter.

    A smoothed short-term RMS envelope is computed and each region is scaled
    toward `target`. Crucially, the boost applied to any region is also
    capped so the region's own PEAK stays under `peak_ceiling` — so a sharp
    consonant sitting inside an otherwise-quiet stretch is never amplified
    into distortion. This is what previously made the Arabic (Mixer) voice
    sound saturated even at low volume."""
    import numpy as np
    if not pcm_bytes:
        return pcm_bytes
    x = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    n = x.size
    if n < sr // 4:
        return pcm_bytes
    win = max(256, int(sr * win_ms / 1000.0))
    power = x * x
    kernel = np.ones(win, dtype=np.float32) / win
    st_rms = np.sqrt(np.convolve(power, kernel, mode="same")) + 1.0
    # short-term PEAK envelope (max |x| in the same window) so we know how
    # much headroom each region has before it would clip
    absx = np.abs(x)
    st_peak = np.sqrt(np.convolve(absx * absx, kernel, mode="same"))
    st_peak = np.maximum(st_peak, 1.0)
    peak_rms = float(np.percentile(st_rms, 95)) or 1.0
    gate = peak_rms * floor_ratio
    # desired gain to reach the RMS target
    gain = target / st_rms
    # peak-safety cap: never boost a region past peak_ceiling on its peak
    peak_cap = peak_ceiling / st_peak
    gain = np.minimum(gain, peak_cap)
    gain = np.clip(gain, 1.0 / max_boost, max_boost)
    quiet = st_rms < gate
    gain[quiet] = 1.0 + (gain[quiet] - 1.0) * (st_rms[quiet] / gate)
    # noise-floor guard: fade the boost back toward 1.0 as the level drops
    # toward the bottom of the utterance, so hiss in the drooped tail is not
    # amplified (this was the grainy "old radio" artifact on speakers 2 & 4)
    nf = peak_rms * 0.30
    low = st_rms < nf
    gain[low] = 1.0 + (gain[low] - 1.0) * (st_rms[low] / nf) ** 2
    # smooth the gain with a LONGER kernel so it can't pump/breathe
    sk = max(512, win // 2)
    smoother = np.ones(sk, dtype=np.float32) / sk
    gain = np.convolve(gain, smoother, mode="same")
    y = x * gain
    # gentle safety limit well below full scale — should rarely engage now
    return soft_limit(y, ceiling=30000.0).astype(np.int16).tobytes()


def finalize_audio(pcm_bytes, sr, volume_gain, semitones=0.0, speed=1.0,
                   even_droop=False):
    """The ONE audio finishing chain used by every voice (Arabic Mixer,
    English Piper, Arabic Piper) so they all sound equally loud and clean.

    Order: optional droop-evening (peak-safe) -> RMS normalize to a shared
    target -> pitch shift -> time-compress -> apply volume. A single limiter
    pass inside normalize keeps peaks clean; nothing here pushes a voice into
    saturation, and both languages land at the same perceived loudness."""
    if not pcm_bytes:
        return pcm_bytes
    out = pcm_bytes
    if even_droop:
        out = level_envelope(out, sr)
    out = normalize_rms(out)                 # shared target -> equal loudness
    out = pitch_shift_pcm(out, semitones, sr)
    out = compress_pcm(out, speed, sr)
    out = apply_gain(out, volume_gain)
    return out


def soft_limit(x, ceiling=32200.0, knee=0.72):
    """Smoothly compress peaks above `knee * ceiling` instead of clipping
    them. Below the knee the signal is untouched, so quiet speech is not
    coloured; above it, peaks are curved down to the ceiling. This lets the
    voice be genuinely louder while staying clean."""
    import numpy as np
    k = knee * ceiling
    a = np.abs(x)
    over = a > k
    if not over.any():
        return x
    head = ceiling - k
    if head <= 0:
        return np.clip(x, -ceiling, ceiling)
    excess = (a[over] - k) / head
    # tanh knee: asymptotically approaches the ceiling, never exceeds it
    shaped = k + head * np.tanh(excess)
    y = x.copy()
    y[over] = np.sign(x[over]) * shaped
    return y


def normalize_rms(pcm_bytes, target_rms=8000.0, limit=32200.0):
    """Scale audio to a consistent RMS loudness so different neural voices
    match in perceived volume. Peak-limited to avoid clipping."""
    import numpy as np
    if not pcm_bytes:
        return pcm_bytes
    x = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    if x.size == 0:
        return pcm_bytes
    rms = float(np.sqrt(np.mean(x * x))) or 1.0
    gain = target_rms / rms
    y = x * gain
    y = soft_limit(y, ceiling=limit)
    return y.astype(np.int16).tobytes()


def apply_gain(pcm_bytes, gain, limit=32200.0):
    """Apply the user's volume as a gain, with a soft limiter rather than a
    hard cap so the top of the range is genuinely louder and still clean."""
    import numpy as np
    if not pcm_bytes or abs(gain - 1.0) < 0.005:
        return pcm_bytes
    x = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) * gain
    x = soft_limit(x, ceiling=limit)
    return x.astype(np.int16).tobytes()
