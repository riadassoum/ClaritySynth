# -*- coding: utf-8 -*-
"""Neural English voice tier (Piper / VITS via ONNX).

Self-contained: phonemization is done by calling espeak-ng directly via
ctypes (no phonemizer/joblib/numpy dependency chain), and phonemes are
fed to the Piper ONNX model with is_phonemes=True. This avoids the heavy,
fragile phonemizer import tree entirely.

When the bundled espeak-ng library and a Piper voice load, English text is
spoken with a natural neural voice; otherwise the formant engine is used.

Credits: Piper (rhasspy / OHF-Voice, MIT); piper-onnx (thewh1teagle, MIT);
espeak-ng (GPL-3, phonemizer library); voice per its MODEL_CARD license.
"""
import os
import sys
import ctypes
import threading
import re

_here = os.path.dirname(os.path.abspath(__file__))
_piper = None
_selected_voice = None      # stem chosen via the Secondary voice combo
_variant_pref = "auto"      # "auto"/"fast"/"standard" from the variant combo
_espeak = None
_sr = 16000
_lock = threading.Lock()
_tried = False

NEURAL_EN_AVAILABLE = False


def _voice_dir():
    return os.path.join(_here, "lib", "piper_voices")


def _voice_dirs():
    """All folders that may hold Piper voices. The persistent user data dir
    (downloads, outside the add-on so they survive updates) comes FIRST, then
    the bundled lib/piper_voices."""
    dirs = []
    try:
        data_paths = _import_data_paths()
        dirs.extend(data_paths.search_dirs("piper_voices"))
    except Exception:
        pass
    bundled = _voice_dir()
    if bundled not in dirs:
        dirs.append(bundled)
    alt = os.path.join(_here, "voices", "piper_voices")
    if os.path.isdir(alt) and alt not in dirs:
        dirs.append(alt)
    return dirs


# Piper quality tiers, FASTEST (lowest latency) first. The "fast" variant
# prefers the earliest available tier; "standard" prefers the latest.
_QUALITY_ORDER = ["x_low", "low", "medium", "high"]


def _quality_of(stem):
    s = stem.lower()
    for i, q in enumerate(_QUALITY_ORDER):
        if s.endswith("-" + q) or ("-" + q + "-") in s or s.endswith(q):
            return i
    return 1        # default: treat as "low"


def _find_voice():
    """Locate the Piper model to load.

    Selection priority:
      1. the exact voice chosen in the Secondary voice combo, if set;
      2. otherwise, honour the variant preference (fast = fastest tier,
         standard = highest tier, auto = fastest) across installed voices.
    Searches the bundled folder and the user drop-in folder, so downloaded
    voices work with no code change."""
    want = _selected_voice
    candidates = []
    for d in _voice_dirs():
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".onnx"):
                continue
            onnx = os.path.join(d, f)
            cfg = onnx + ".json"
            if not os.path.exists(cfg):
                continue
            stem = f[:-5]
            if want and stem == want:
                return onnx, cfg
            candidates.append((stem, onnx, cfg))
    if not candidates:
        return None, None
    # pick by variant preference. "standard" -> highest quality tier;
    # "fast"/"auto" -> lowest tier (fastest, lowest latency = preferred).
    if _variant_pref == "standard":
        candidates.sort(key=lambda c: -_quality_of(c[0]))
    else:
        candidates.sort(key=lambda c: _quality_of(c[0]))
    return candidates[0][1], candidates[0][2]


def select_variant(pref):
    """Choose the Piper quality variant: "fast" (fastest, lowest latency),
    "standard" (higher quality, slower), or "auto" (prefer fastest). Forces
    a lazy reload on the next synth."""
    global _variant_pref, _piper, _tried, NEURAL_EN_AVAILABLE
    if pref == _variant_pref:
        return
    _variant_pref = pref or "auto"
    _piper = None
    _tried = False
    NEURAL_EN_AVAILABLE = False
    try:
        clear_cache()
    except Exception:
        pass


def select_voice(name):
    """Choose which installed Piper voice speaks non-Arabic text. None = the
    default (first installed). Forces a reload on the next synth."""
    global _selected_voice, _piper, _tried, NEURAL_EN_AVAILABLE
    if name == _selected_voice:
        return
    _selected_voice = name
    # drop the current runner so the new voice is loaded lazily (off the
    # GUI thread, from the speech worker)
    _piper = None
    _tried = False
    NEURAL_EN_AVAILABLE = False
    try:
        clear_cache()
    except Exception:
        pass


def _init_espeak():
    """Load espeak-ng via ctypes for direct text->IPA phonemization."""
    global _espeak
    lib_dir = os.path.join(_here, "lib", "espeakng_loader")
    # locate the platform library
    names = ["espeak-ng.dll", "libespeak-ng.so", "libespeak-ng.dylib",
             "libespeak-ng.so.1"]
    lib_path = None
    for n in names:
        p = os.path.join(lib_dir, n)
        if os.path.exists(p):
            lib_path = p
            break
    if not lib_path:
        return False
    data_path = os.path.join(lib_dir, "espeak-ng-data")
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(lib_dir)
        except Exception:
            pass
    lib = ctypes.CDLL(lib_path)
    # AUDIO_OUTPUT_SYNCHRONOUS = 0x02; returns sample rate or -1
    if lib.espeak_Initialize(0x02, 0, data_path.encode("utf-8"), 0) == -1:
        return False
    lib.espeak_SetVoiceByName(b"en-us")
    lib.espeak_TextToPhonemes.restype = ctypes.c_char_p
    _espeak = lib
    return True


_current_espeak_voice = "en-us"


# a few languages where eSpeak's canonical name differs from the region code
_ESPEAK_ALIASES = {
    "en": "en-us", "en-gb": "en-gb", "en-us": "en-us",
    "pt-br": "pt-br", "pt": "pt-pt", "zh": "cmn", "zh-cn": "cmn",
    "nb": "nb", "no": "nb",
    # Some community Piper voices (e.g. the salvaged Arabic ones) put a
    # language NAME in their config's espeak.voice instead of a code. Map the
    # common ones to the eSpeak code so SetVoiceByName succeeds and the voice
    # is phonemized in the right language instead of silently going English.
    "arabic": "ar", "ar": "ar", "ar-sa": "ar", "ar-ae": "ar",
    "ar-qa": "ar", "ar-jo": "ar", "ar-eg": "ar",
    "english": "en-us", "french": "fr", "spanish": "es",
    "german": "de", "italian": "it",
}


def _espeak_candidates(voice):
    """Ordered list of eSpeak voice names to try for a requested voice,
    since Piper configs use region forms (fr-fr, es-es) that eSpeak's
    SetVoiceByName rejects — the base language (fr, es) is what works."""
    v = (voice or "").strip().lower()
    cands = []
    def _add(x):
        if x and x not in cands:
            cands.append(x)
    _add(v)
    if v in _ESPEAK_ALIASES:
        _add(_ESPEAK_ALIASES[v])
    # base language before the first hyphen/underscore (fr-fr -> fr)
    for sep in ("-", "_"):
        if sep in v:
            base = v.split(sep)[0]
            _add(_ESPEAK_ALIASES.get(base, base))
    return cands


def current_espeak_voice():
    """The espeak voice/language of the currently selected secondary voice
    (e.g. 'en-us', 'fr', 'es'). Used to decide how to speak single
    characters — English uses the exact-IPA table, other languages let the
    voice phonemize the character itself."""
    try:
        if _try_init() and _piper is not None:
            return getattr(_piper, "espeak_voice", None) \
                or _current_espeak_voice
    except Exception:
        pass
    return _current_espeak_voice


def current_voice_is_english():
    v = (current_espeak_voice() or "en").lower()
    return v.startswith("en")


def _set_espeak_voice(voice):
    """Set the eSpeak NG voice/language, robust to region forms. Each Piper
    voice supplies its own espeak voice in its config; we try the exact name
    then fall back to the base language so a French/Spanish/etc. voice is
    phonemized in the RIGHT language instead of silently staying English."""
    global _current_espeak_voice
    if not voice or voice == _current_espeak_voice:
        return
    if _espeak is None:
        return
    for cand in _espeak_candidates(voice):
        try:
            rc = _espeak.espeak_SetVoiceByName(cand.encode("utf-8"))
            # espeak returns 0 (EE_OK) on success
            if rc == 0:
                _current_espeak_voice = voice
                return
        except Exception:
            continue


def _phonemize(text, voice=None):
    """Text -> eSpeak IPA string (space-separated per clause), matching
    what Piper's phonemizer would produce. `voice` selects the eSpeak
    language so non-English voices phonemize correctly."""
    if voice:
        _set_espeak_voice(voice)
    out = []
    tptr = ctypes.c_char_p(text.encode("utf-8"))
    vptr = ctypes.cast(ctypes.pointer(tptr), ctypes.c_void_p)
    guard = 0
    while guard < 5000:
        guard += 1
        ph = _espeak.espeak_TextToPhonemes(vptr, 1, 0x02)  # UTF8 in, IPA
        if ph:
            out.append(ph.decode("utf-8"))
        remaining = ctypes.cast(
            vptr, ctypes.POINTER(ctypes.c_char_p)).contents.value
        if not remaining:
            break
    result = " ".join(out)
    # eSpeak inserts a language-switch marker like "(en)" or "(fr)" into the
    # phoneme stream whenever it decides a word belongs to another language
    # (e.g. an English word inside French text). Piper's phoneme_id_map has
    # no id for the literal characters "(", "e", "n", ")", so the voice ends
    # up trying to PRONOUNCE the marker — the stray "en" sound users hear in
    # front of foreign words. Strip these markers so the word is simply
    # phonemized with the current voice's own letters.
    result = _LANG_SWITCH_RE.sub(" ", result)
    return result


_LANG_SWITCH_RE = re.compile(r"\([a-z]{2,3}(?:-[a-z]+)?\)")


def _try_init():
    global _piper, _tried, NEURAL_EN_AVAILABLE, _sr
    if _tried:
        return NEURAL_EN_AVAILABLE
    with _lock:
        if _tried:
            return NEURAL_EN_AVAILABLE
        _tried = True
        try:
            import warnings
            warnings.filterwarnings("ignore")
            from . import _libboot
            _libboot.boot()   # our numpy/onnxruntime first
            onnx, cfg = _find_voice()
            if not onnx:
                return False
            if not _init_espeak():
                return False
            import json
            with open(cfg, encoding="utf-8") as f:
                conf = json.load(f)
            _sr = conf.get("audio", {}).get("sample_rate", 16000)
            _pmap = conf.get("phoneme_id_map", {})
            import onnxruntime as ort
            import os as _os
            # Tune the session for LOW LATENCY: enable all graph
            # optimizations and use all CPU cores. This roughly halves the
            # per-call inference time versus default options.
            _so = ort.SessionOptions()
            try:
                _so.graph_optimization_level = \
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                # cap threads: more threads HURT these small models on CPU
                _cores = _os.cpu_count() or 2
                _so.intra_op_num_threads = 1 if _cores <= 2 else 2
                _so.inter_op_num_threads = 1
                _so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                _so.enable_mem_pattern = True
            except Exception:
                pass
            sess = ort.InferenceSession(
                onnx, sess_options=_so,
                providers=["CPUExecutionProvider"])
            _piper = _PiperRunner(sess, conf, _pmap)
            _piper.synth("ok")   # warm-up
            NEURAL_EN_AVAILABLE = True
        except Exception as e:
            NEURAL_EN_AVAILABLE = False
            try:
                from logHandler import log
                log.warning("ClaritySynth Piper English init failed: %r"
                            % e, exc_info=True)
            except Exception:
                pass
        return NEURAL_EN_AVAILABLE


class _PiperRunner(object):
    """Minimal Piper inference: IPA phonemes -> ids -> ONNX -> audio."""
    _BOS = "^"
    _EOS = "$"
    _PAD = "_"

    def __init__(self, sess, conf, pmap):
        self.sess = sess
        self.conf = conf
        self.pmap = pmap
        inf = conf.get("inference", {})
        self.length_scale = inf.get("length_scale", 1.0)
        self.noise_scale = inf.get("noise_scale", 0.667)
        self.noise_w = inf.get("noise_w", 0.8)
        self.n_speakers = conf.get("num_speakers", 1)
        self.in_names = [i.name for i in sess.get_inputs()]
        # each Piper voice carries its eSpeak voice/language in its config;
        # use it so non-English voices phonemize in the right language
        esp = conf.get("espeak", {}) or {}
        self.espeak_voice = esp.get("voice") or "en-us"

    def _ids(self, phonemes):
        ids = []
        pad = self.pmap.get(self._PAD, [0])

        def _emit(sym):
            if sym in self.pmap:
                ids.extend(self.pmap[sym])
                ids.extend(pad)
                return True
            return False

        # group each base character with any following combining marks
        # (U+0300..U+036F) and IPA tie bars, so a nasal/affricate stays whole
        chars = list(phonemes)
        i = 0
        # start-of-sentence symbol
        _emit(self._BOS)
        while i < len(chars):
            base = chars[i]
            j = i + 1
            # attach trailing combining marks / tie bars to this base
            while j < len(chars) and (
                    "\u0300" <= chars[j] <= "\u036F"
                    or chars[j] in "\u0361\u035C\u02de"):
                j += 1
            cluster = "".join(chars[i:j])
            if len(cluster) > 1 and _emit(cluster):
                i = j
                continue
            # cluster not in map: emit the base, then each mark on its own
            if _emit(base):
                for k in range(i + 1, j):
                    _emit(chars[k])
                i = j
                continue
            # base alone unknown: try each piece; skip only truly unknown ones
            emitted_any = False
            for k in range(i, j):
                if _emit(chars[k]):
                    emitted_any = True
            i = j
        ids.extend(self.pmap.get(self._EOS, []))
        return ids

    def synth(self, text, length_scale=None, phonemes=None,
              noise_scale=None, noise_w=None):
        import numpy as np
        # `phonemes` lets the caller supply exact IPA (character mode), which
        # avoids eSpeak mis-guessing short pseudo-words like "ay"/"eff".
        phon = phonemes if phonemes else _phonemize(
            text, voice=getattr(self, "espeak_voice", None))
        ids = self._ids(phon)
        if not ids:
            return None, _sr
        ls = length_scale or self.length_scale
        x = np.array([ids], dtype=np.int64)
        x_len = np.array([x.shape[1]], dtype=np.int64)
        ns = self.noise_scale if noise_scale is None else noise_scale
        nw = self.noise_w if noise_w is None else noise_w
        scales = np.array([ns, ls, nw],
                          dtype=np.float32)
        feed = {"input": x, "input_lengths": x_len, "scales": scales}
        if "sid" in self.in_names and self.n_speakers > 1:
            feed["sid"] = np.array([0], dtype=np.int64)
        feed = {k: v for k, v in feed.items() if k in self.in_names}
        out = self.sess.run(None, feed)[0]
        return out.squeeze(), _sr


from collections import OrderedDict as _OD

def _import_data_paths():
    """Import the data_paths module whether we are loaded as part of the
    claritysynth package (normal NVDA) or as a loose module (tests)."""
    try:
        from . import data_paths as _dp
        return _dp
    except Exception:
        import os as _os, sys as _sys
        _d = _os.path.dirname(_os.path.abspath(__file__))
        if _d not in _sys.path:
            _sys.path.insert(0, _d)
        import data_paths as _dp
        return _dp



_CACHE = _OD()
_CACHE_MAX = 64          # entries; short strings only, so memory is small
_CACHE_MAXLEN = 48       # only cache strings up to this many characters
_cache_lock = threading.Lock()


def _cache_get(key):
    with _cache_lock:
        v = _CACHE.get(key)
        if v is not None:
            _CACHE.move_to_end(key)     # LRU
        return v


def _cache_put(key, val):
    with _cache_lock:
        _CACHE[key] = val
        _CACHE.move_to_end(key)
        while len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)


def clear_cache():
    with _cache_lock:
        _CACHE.clear()


_extra_runners = {}      # voice stem -> (_PiperRunner, sample_rate)


def _load_runner_for(stem):
    """Load (and cache) a Piper runner for a specific installed voice stem.
    Used for e.g. an Arabic Piper voice selected as the primary voice, so it
    does not disturb the main English runner."""
    if stem in _extra_runners:
        return _extra_runners[stem]
    onnx = cfg = None
    for d in _voice_dirs():
        cand = os.path.join(d, stem + ".onnx")
        if os.path.exists(cand) and os.path.exists(cand + ".json"):
            onnx, cfg = cand, cand + ".json"
            break
    if not onnx:
        _extra_runners[stem] = (None, None)
        return None, None
    try:
        from . import _libboot
        _libboot.boot()
        if not _init_espeak():
            _extra_runners[stem] = (None, None)
            return None, None
        import json as _json
        with open(cfg, encoding="utf-8") as f:
            conf = _json.load(f)
        pmap = conf.get("phoneme_id_map", {})
        sr = conf.get("audio", {}).get("sample_rate", 22050)
        import onnxruntime as ort
        import os as _os
        so = ort.SessionOptions()
        try:
            so.graph_optimization_level = \
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            _cores2 = _os.cpu_count() or 2
            so.intra_op_num_threads = 1 if _cores2 <= 2 else 2
            so.inter_op_num_threads = 1
        except Exception:
            pass
        sess = ort.InferenceSession(
            onnx, sess_options=so, providers=["CPUExecutionProvider"])
        runner = _PiperRunner(sess, conf, pmap)
        _extra_runners[stem] = (runner, sr)
        return runner, sr
    except Exception:
        _extra_runners[stem] = (None, None)
        return None, None


def synth_wave_with(text, stem, volume=1.0, length_scale=None, clarity=None):
    """Synthesize `text` with the specific installed voice `stem`. Returns
    (pcm_bytes, sample_rate) or (None, None). The language of the voice is
    whatever that model was trained for (e.g. an Arabic Piper voice).

    `length_scale` and `clarity` are honoured exactly as in synth_wave so a
    voice (e.g. Kareem) sounds identical whether used as the primary Arabic
    voice or as a secondary voice."""
    if not text or not text.strip():
        return None, None
    runner, sr = _load_runner_for(stem)
    if runner is None:
        return None, None
    try:
        import numpy as np
        # apply the same clarity->noise mapping as synth_wave
        ns = nw = None
        if clarity:
            k = max(0.0, min(1.0, clarity / 100.0))
            ns = runner.noise_scale * (1.0 - 0.5 * k)
            nw = runner.noise_w * (1.0 - 0.5 * k)
        # length_scale is passed EXACTLY as in synth_wave (the secondary-voice
        # path) so a voice like Kareem has identical tempo AND pitch whether
        # it is the primary or the secondary voice. (A previous version scaled
        # it relative to the voice's own length_scale, which shifted Kareem's
        # perceived pitch when used as primary.)
        audio, _sr2 = runner.synth(text, length_scale=length_scale,
                                   noise_scale=ns, noise_w=nw)
        if audio is None:
            return None, None
        a = np.asarray(audio, dtype=np.float32)
        m = float(np.max(np.abs(a))) or 1.0
        if m > 1.0:
            a = a / m
        a = a * volume
        return (a * 32000.0).astype(np.int16).tobytes(), sr
    except Exception:
        return None, None


def sample_rate():
    return _sr


def synth_wave(text, length_scale=None, volume=1.0, phonemes=None,
               clarity=None):
    """Synthesize `text`, or — if `phonemes` is given — render that exact IPA
    string directly (used for single characters, where eSpeak's G2P is
    unreliable)."""
    if not _try_init():
        return None
    key = None
    if text and len(text) <= _CACHE_MAXLEN and phonemes is None:
        key = (_selected_voice, text, length_scale, volume, clarity)
        hit = _cache_get(key)
        if hit is not None:
            return hit
    try:
        import numpy as np
        # `clarity` (0..100) reduces the model's stochastic noise, which
        # audibly cleans the voice up. None/0 = the model's own defaults.
        ns = nw = None
        if clarity:
            k = max(0.0, min(1.0, clarity / 100.0))
            ns = _piper.noise_scale * (1.0 - 0.5 * k)
            nw = _piper.noise_w * (1.0 - 0.5 * k)
        audio, sr = _piper.synth(text, length_scale=length_scale,
                                 phonemes=phonemes,
                                 noise_scale=ns, noise_w=nw)
        if audio is None:
            return None
        a = np.asarray(audio, dtype=np.float32)
        if a.size == 0:
            return None
        m = float(np.max(np.abs(a))) or 1.0
        if m > 1.0:
            a = a / m
        a = a * volume
        pcm = (a * 32000.0).astype(np.int16).tobytes()
        if key is not None and pcm:
            _cache_put(key, pcm)
        return pcm
    except Exception:
        return None
