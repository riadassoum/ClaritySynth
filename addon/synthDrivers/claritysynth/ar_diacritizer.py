# -*- coding: utf-8 -*-
# ClaritySynth: automatic Arabic diacritization.
#
# Three layers, tried in order per word (after clitic-prefix stripping):
#
#   1. Word lexicon learned from the Tashkeela corpus (~2.1M words):
#      the most frequent vocalization of each bare form, case ending
#      stripped (pausal-neutral).
#   2. Morphological templates (awzan): a nonsense word like
#      استحماش still matches the istif'aal pattern and comes out
#      اِسْتِحْمَاش, exactly as a native reader would vocalize it.
#   3. A character-level n-gram model with backoff, also learned from
#      Tashkeela, predicting the most likely diacritic for each letter
#      from its neighbours.
#
# All models are plain pickled dicts, pure-Python inference, microseconds
# per word after the one-time lazy load.

import os
import re
import gzip
import pickle
import threading

FATHA = "\u064E"
DAMMA = "\u064F"
KASRA = "\u0650"
SUKUN = "\u0652"
SHADDA = "\u0651"
DIACS = set("\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0670")

SUN = set("\u062A\u062B\u062F\u0630\u0631\u0632\u0633\u0634"
          "\u0635\u0636\u0637\u0638\u0644\u0646")

_lex = None
_ngram = None
_bigram = {}
_fullform = {}
_revbigram = {}
_mishkal = None
_mishkal_tried = False


def _get_mishkal():
    """Lazy-load the bundled mishkal vocalizer (deep OOV tier)."""
    global _mishkal, _mishkal_tried
    if _mishkal_tried:
        return _mishkal
    _mishkal_tried = True
    try:
        import sys
        import warnings
        warnings.filterwarnings("ignore", category=SyntaxWarning)
        lib = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "lib")
        if os.path.isdir(lib) and lib not in sys.path:
            sys.path.insert(0, lib)
        import mishkal.tashkeel
        _mishkal = mishkal.tashkeel.TashkeelClass()
    except Exception:
        _mishkal = None
    return _mishkal
_lock = threading.Lock()


def _load():
    global _lex, _ngram
    if _lex is not None:
        return
    with _lock:
        if _lex is not None:
            return
        here = os.path.dirname(os.path.abspath(__file__))
        try:
            with gzip.open(os.path.join(here, "ar_lex.pkl.gz"), "rb") as f:
                lex = pickle.load(f)
        except Exception:
            lex = {}
        try:
            with gzip.open(os.path.join(here, "ar_ngram.pkl.gz"),
                           "rb") as f:
                ngram = pickle.load(f)
        except Exception:
            ngram = ({}, {}, {}, {})
        try:
            with gzip.open(os.path.join(here, "ar_bigram.pkl.gz"),
                           "rb") as f:
                bigram = pickle.load(f)
        except Exception:
            bigram = {}
        try:
            with gzip.open(os.path.join(here, "ar_fullform.pkl.gz"),
                           "rb") as f:
                globals()["_fullform"] = pickle.load(f)
        except Exception:
            globals()["_fullform"] = {}
        try:
            with gzip.open(os.path.join(here, "ar_revbigram.pkl.gz"),
                           "rb") as f:
                globals()["_revbigram"] = pickle.load(f)
        except Exception:
            globals()["_revbigram"] = {}
        global _bigram
        _bigram = bigram
        _ngram = ngram
        _lex = lex


# ---------------------------------------------------------------------------
# Morphological templates (awzan). C slots are root consonants; digits in
# the output refer to captured roots. Ordered most-specific first.
# ---------------------------------------------------------------------------
_CLS = "[\u0628\u062A-\u063A\u0641-\u0646\u0647]"   # root consonant class

_TEMPLATES_SRC = [
    ("استCCاC", "اِسْتِ1ْ2َا3"),      # istif'aal  (e.g. استحماش)
    ("استCCC",  "اِسْتَ1ْ2َ3"),       # istaf'al
    ("مستCCC",  "مُسْتَ1ْ2ِ3"),       # mustaf'il
    ("انCCاC",  "اِنْ1ِ2َا3"),        # infi'aal
    ("انCCC",   "اِنْ1َ2َ3"),         # infa'al
    ("اCتCاC",  "اِ1ْتِ2َا3"),        # ifti'aal
    ("اCتCC",   "اِ1ْتَ2َ3"),         # ifta'al
    ("تCاCC",   "تَ1َا2ُ3"),          # tafaa'ul
    ("تCCيC",   "تَ1ْ2ِي3"),          # taf'iil
    ("مCاCيC",  "مَ1َا2ِي3"),         # mafaa'iil
    ("مCاCC",   "مَ1َا2ِ3"),          # mafaa'il
    ("مCCوC",   "مَ1ْ2ُو3"),          # maf'uul
    ("CCاCة",   "1ِ2َا3َة"),          # fi'aala
    ("CCيCة",   "1َ2ِي3َة"),          # fa'iila
    ("CاCوC",   "1َا2ُو3"),           # faa'uul
    ("CCCان",   "1َ2ْ3َان"),          # fa'laan
    ("CاCC",    "1َا2ِ3"),            # faa'il
    ("CCيC",    "1َ2ِي3"),            # fa'iil
    ("CCوC",    "1ُ2ُو3"),            # fu'uul
    ("CCاC",    "1ِ2َا3"),            # fi'aal
    ("مCCCة",   "مَ1ْ2َ3َة"),         # maf'ala
    ("مCCC",    "مَ1ْ2َ3"),           # maf'al
    ("CCCة",    "1َ2ْ3َة"),           # fa'la
    ("يCCC",    "يَ1ْ2ُ3"),           # yaf'ul
    ("تCCC",    "تَ1ْ2ُ3"),           # taf'ul
    ("CCC",     "1َ2َ3"),             # fa'al
]


def _compile_templates():
    out = []
    for pat, voc in _TEMPLATES_SRC:
        rx = ""
        for ch in pat:
            rx += "(" + _CLS + ")" if ch == "C" else re.escape(ch)
        out.append((re.compile("^" + rx + "$"), voc))
    return out


_TEMPLATES = _compile_templates()


def _apply_templates(word):
    for rx, voc in _TEMPLATES:
        m = rx.match(word)
        if not m:
            continue
        groups = m.groups()
        out = []
        for ch in voc:
            if ch.isdigit():
                out.append(groups[int(ch) - 1])
            else:
                out.append(ch)
        return "".join(out)
    return None


def _apply_ngram(word):
    """Per-letter most-likely diacritic with 4->3->2->1 backoff."""
    m4, m3, m2, m1 = _ngram
    n = len(word)
    out = []
    for i, cur in enumerate(word):
        out.append(cur)
        if i == n - 1:
            break                       # pausal: bare final letter
        p1 = word[i - 1] if i > 0 else "^"
        n1 = word[i + 1] if i + 1 < n else "$"
        n2 = word[i + 2] if i + 2 < n else "$"
        d = m4.get((p1, cur, n1, n2))
        if d is None:
            d = m3.get((p1, cur, n1))
        if d is None:
            d = m2.get((cur, n1))
        if d is None:
            d = m1.get(cur, "")
        out.append(d)
    return "".join(out)


_SUFFIXES = [
    ("\u0647\u0645\u0627", "\u0647\u064F\u0645\u064E\u0627"),  # -humaa
    ("\u0647\u0646", "\u0647\u064F\u0646\u0651\u064E"),          # -hunna
    ("\u0647\u0627", "\u0647\u064E\u0627"),                        # -haa
    ("\u0647\u0645", "\u0647\u064F\u0645"),                        # -hum
    ("\u0643\u0645", "\u0643\u064F\u0645"),                        # -kum
    ("\u0646\u0627", "\u0646\u064E\u0627"),                        # -naa
    ("\u0648\u0646", "\u064F\u0648\u0646"),                        # -uun
    ("\u064A\u0646", "\u0650\u064A\u0646"),                        # -iin
    ("\u0627\u062A", "\u064E\u0627\u062A"),                        # -aat
    ("\u064A\u0629", "\u0650\u064A\u0651\u064E\u0629"),          # -iyya
    ("\u0647", "\u0647\u064F"),                                      # -hu
    ("\u0643", "\u0643\u064E"),                                      # -ka
]


def _strip_case(w):
    while w and w[-1] in DIACS and w[-1] != SHADDA:
        w = w[:-1]
    return w


def _solve(word):
    """Diacritize a bare word (no clitics)."""
    if word in _lex:
        return _lex[word]
    t = _apply_templates(word)
    if t:
        return t
    # inflected OOV: strip a pronoun/plural suffix, solve the stem
    for suf, voc in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            stem = word[:-len(suf)]
            ss = _lex.get(stem) or _apply_templates(stem)
            if ss:
                return _strip_case(ss) + voc
    # deep tier: bundled mishkal morphological vocalizer
    v = _get_mishkal()
    if v is not None and len(word) >= 3:
        try:
            m = v.tashkeel(word).strip()
            if m and m != word and any(c in DIACS for c in m):
                _lex[word] = m          # cache for instant reuse
                return m
        except Exception:
            pass
    if len(word) >= 2:
        return _apply_ngram(word)
    return None


def _article(rest_solved):
    first = rest_solved[0]
    if first in SUN:
        i = 1
        while i < len(rest_solved) and rest_solved[i] in DIACS:
            i += 1
        return "\u0627\u0644" + first + SHADDA + rest_solved[1:]
    return "\u0627\u0644" + SUKUN + rest_solved


_PREFIXES = [
    ("\u0648\u0627\u0644", "\u0648" + FATHA, True),    # wa + al
    ("\u0641\u0627\u0644", "\u0641" + FATHA, True),    # fa + al
    ("\u0628\u0627\u0644", "\u0628" + KASRA, True),    # bi + al
    ("\u0643\u0627\u0644", "\u0643" + FATHA, True),    # ka + al
    ("\u0644\u0644", "\u0644" + KASRA + "\u0644", None),  # li + l
    ("\u0627\u0644", "", True),                        # al
    ("\u0648", "\u0648" + FATHA, False),               # wa
    ("\u0641", "\u0641" + FATHA, False),               # fa
    ("\u0628", "\u0628" + KASRA, False),               # bi
    ("\u0643", "\u0643" + FATHA, False),               # ka
    ("\u0644", "\u0644" + KASRA, False),               # li
]


_neural = None


def _get_neural():
    global _neural
    if _neural is None:
        try:
            try:
                from . import ar_neural
            except ImportError:
                import ar_neural
            _neural = ar_neural
        except Exception:
            _neural = False
    return _neural


def diacritize(word, prev=None, nxt=None):
    """Return a diacritized form of an undiacritized word, or None.
    `prev` is the previous bare word, enabling the context bigram model
    (which can also restore mid-sentence case endings)."""
    _load()
    word = word.replace("\u0640", "")
    if not word or any(c in DIACS for c in word):
        return None
    # Neural tier (Shakkelha ONNX) front-runs everything when installed.
    _nn = _get_neural()
    if _nn and getattr(_nn, "NEURAL_AVAILABLE", False):
        nres = _nn.diacritize(word, prev, nxt)
        if nres and nres != word:
            return nres
    if prev:
        b = _bigram.get((prev, word))
        if b:
            return b
    b = _bigram.get(("^", word))
    if b and prev is None:
        return b
    if nxt is not None:                  # ending depends on next word
        rf = _revbigram.get((word, nxt))
        if rf:
            return rf
    if prev is not None:                 # mid-sentence: keep the ending
        f = _fullform.get(word)
        if f:
            return f
    s = _solve(word)
    if s and s != word:
        return s
    # clitic prefixes
    for pref, voc, is_article in _PREFIXES:
        if word.startswith(pref) and len(word) - len(pref) >= 2:
            rest = word[len(pref):]
            rs = _solve(rest)
            if rs and rs != rest:
                if is_article:
                    return voc + _article(rs)
                if is_article is None:      # li+l
                    return voc + rs
                return voc + rs
    return s
