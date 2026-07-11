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
    hop_a = int(hop_s * factor)               # input hop
    seek = int(sr * seek_ms / 1000.0)
    win = np.hanning(fl).astype(np.float32)

    out_len = int(n / factor) + fl * 4
    out = np.zeros(out_len, dtype=np.float32)
    ow = np.zeros(out_len, dtype=np.float32)

    ia = 0
    io = 0
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
        te = ia + hop_s
        if te + hop_s <= n:
            prev_tail = x[te:te + hop_s].copy()
        else:
            prev_tail = np.zeros(hop_s, dtype=np.float32)
        ia += hop_a
        io += hop_s

    # Append any remaining tail (the loop stops when < one frame is left,
    # which would otherwise drop the final consonant). Add the leftover
    # input samples from the last analysis point onward.
    if ia < n:
        tail = x[ia:]
        if tail.size:
            end = min(io + tail.size, out.size)
            k = end - io
            out[io:end] += tail[:k]
            ow[io:end] += 1.0
            io = end

    ow[ow < 1e-6] = 1.0
    y = (out / ow)[:io]
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
    # enforce EXACT original length (pad with silence or trim) so pitch
    # shifting never changes duration — keeps voices in sync, no drift.
    o = np.frombuffer(out, dtype=np.int16)
    if o.size < orig_len:
        o = np.concatenate([o, np.zeros(orig_len - o.size, dtype=np.int16)])
    elif o.size > orig_len:
        o = o[:orig_len]
    return o.tobytes()


def _time_stretch(pcm_bytes, factor, sr, frame_ms=30, seek_ms=6):
    """WSOLA stretch (factor>1 = LONGER/slower). Mirror of compress."""
    import numpy as np
    if factor is None or abs(factor - 1.0) < 0.02 or not pcm_bytes:
        return pcm_bytes
    x = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    n = x.size
    fl = max(96, int(sr * frame_ms / 1000.0))
    if n < fl * 3:
        return pcm_bytes
    hop_s = fl // 2
    hop_a = max(1, int(hop_s / factor))     # smaller input hop -> longer out
    seek = int(sr * seek_ms / 1000.0)
    win = np.hanning(fl).astype(np.float32)
    out_len = int(n * factor) + fl * 4
    out = np.zeros(out_len, dtype=np.float32)
    ow = np.zeros(out_len, dtype=np.float32)
    ia = 0
    io = 0
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
        te = ia + hop_s
        prev_tail = x[te:te + hop_s].copy() if te + hop_s <= n \
            else np.zeros(hop_s, dtype=np.float32)
        ia += hop_a
        io += hop_s
    if ia < n:
        tail = x[ia:]
        if tail.size:
            end = min(io + tail.size, out.size)
            k = end - io
            out[io:end] += tail[:k]
            ow[io:end] += 1.0
            io = end
    ow[ow < 1e-6] = 1.0
    y = (out / ow)[:io]
    m = float(np.max(np.abs(y))) or 1.0
    if m > 32767.0:
        y = y * (32767.0 / m)
    return y.astype(np.int16).tobytes()


def normalize_rms(pcm_bytes, target_rms=6000.0, limit=30000.0):
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
    m = float(np.max(np.abs(y))) or 1.0
    if m > limit:
        y = y * (limit / m)
    return y.astype(np.int16).tobytes()


def apply_gain(pcm_bytes, gain, limit=32000.0):
    """Apply a linear volume gain (0..1+) with peak limiting."""
    import numpy as np
    if not pcm_bytes or abs(gain - 1.0) < 0.01:
        return pcm_bytes
    x = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) * gain
    m = float(np.max(np.abs(x))) or 1.0
    if m > limit:
        x = x * (limit / m)
    return x.astype(np.int16).tobytes()
