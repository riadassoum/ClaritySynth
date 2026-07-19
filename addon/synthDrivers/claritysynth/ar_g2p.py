# -*- coding: utf-8 -*-
# ClaritySynth: Arabic text-to-phoneme backbone.
#
# Design note. Arabic splits into two very different problems:
#
#   1. Diacritized text -> phonemes: almost fully rule-governed. Each
#      consonant + haraka maps deterministically to sound; the residue is a
#      short list of orthographic rules (shadda gemination, tanween, madd,
#      sun-letter assimilation, hamzat al-wasl, ta marbuta, dagger alif).
#      This module implements that completely: give it vocalized text and
#      it speaks correctly.
#
#   2. Plain (undiacritized) text -> diacritized text: NOT rule-solvable in
#      general, because internal vowels and case endings depend on
#      morphology, syntax and meaning. This module ships a lexicon of
#      high-frequency vocalized words plus a consonant-skeleton fallback so
#      plain text is still intelligible, with a clear hook
#      (diacritize_hook) where a statistical diacritizer can be plugged in.

import re

# Diacritics
FATHA = "\u064E"
DAMMA = "\u064F"
KASRA = "\u0650"
SUKUN = "\u0652"
SHADDA = "\u0651"
TAN_FATH = "\u064B"
TAN_DAMM = "\u064C"
TAN_KASR = "\u064D"
DAGGER = "\u0670"   # superscript alif
TATWEEL = "\u0640"
DIACS = {FATHA, DAMMA, KASRA, SUKUN, SHADDA, TAN_FATH, TAN_DAMM, TAN_KASR,
         DAGGER}

# Letters
ALIF = "\u0627"
ALIF_MADDA = "\u0622"
ALIF_HAMZA = "\u0623"
ALIF_HAMZA_BELOW = "\u0625"
ALIF_WASLA = "\u0671"
HAMZA = "\u0621"
WAW_HAMZA = "\u0624"
YA_HAMZA = "\u0626"
TA_MARBUTA = "\u0629"
ALIF_MAQSURA = "\u0649"
WAW = "\u0648"
YA = "\u064A"
LAM = "\u0644"

CONS = {
    "\u0628": "B",    # ba
    "\u062A": "T",    # ta
    "\u062B": "TH",   # tha
    "\u062C": "JH",   # jim
    "\u062D": "HH2",  # haa (pharyngeal)
    "\u062E": "KH",   # khaa
    "\u062F": "D",    # dal
    "\u0630": "DH",   # dhal
    "\u0631": "DX",   # raa (tap)
    "\u0632": "Z",    # zay
    "\u0633": "S",    # sin
    "\u0634": "SH",   # shin
    "\u0635": "SS",   # sad (emphatic)
    "\u0636": "DD",   # dad (emphatic)
    "\u0637": "TT",   # taa (emphatic)
    "\u0638": "ZZ",   # zaa (emphatic)
    "\u0639": "AYN",  # ayn
    "\u063A": "GH",   # ghayn
    "\u0641": "F",    # fa
    "\u0642": "Q",    # qaf
    "\u0643": "K",    # kaf
    LAM: "L",         # lam
    "\u0645": "M",    # mim
    "\u0646": "N",    # nun
    "\u0647": "HH",   # ha
    WAW: "W",
    YA: "Y",
    HAMZA: "GS",
    ALIF_HAMZA: "GS",
    ALIF_HAMZA_BELOW: "GS",
    WAW_HAMZA: "GS",
    YA_HAMZA: "GS",
}

# Emphatic / backing environment: /a/ realized as back [ɑ] (AA) near these
EMPHATIC = {"SS", "DD", "TT", "ZZ", "Q", "KH", "GH", "DXQ"}

SUN_LETTERS = set("\u062A\u062B\u062F\u0630\u0631\u0632\u0633\u0634"
                  "\u0635\u0636\u0637\u0638\u0644\u0646")

# High-frequency vocalized words, stored directly as phonemes.
LEX_AR = {
    "\u0641\u064a": "F IY IY",                       # fi
    "\u0645\u0646": "M IH N",                        # min
    "\u0625\u0644\u0649": "GS IH L AE AE",           # ila
    "\u0639\u0644\u0649": "AYN AA L AA AA",          # ala
    "\u0639\u0646": "AYN AA N",                      # an
    "\u0645\u0639": "M AA AYN AE",                   # ma'a
    "\u0647\u0630\u0627": "HH AE AE DH AE AE",       # hadha
    "\u0647\u0630\u0647": "HH AE AE DH IH HH IH",    # hadhihi
    "\u0630\u0644\u0643": "DH AE AE L IH K AE",      # dhalika
    "\u0627\u0644\u0630\u064a": "GS AE L L AE DH IY IY",   # alladhi
    "\u0627\u0644\u062a\u064a": "GS AE L L AE T IY IY",    # allati
    "\u0643\u0627\u0646": "K AE AE N AE",            # kana
    "\u0642\u0627\u0644": "Q AA AA L AE",            # qala
    "\u0644\u0627": "L AE AE",                       # la
    "\u0645\u0627": "M AE AE",                       # ma
    "\u0644\u0645": "L AE M",                        # lam
    "\u0644\u0646": "L AE N",                        # lan
    "\u0647\u0648": "HH UH W AE",                    # huwa
    "\u0647\u064a": "HH IH Y AE",                    # hiya
    "\u0647\u0645": "HH UH M",                       # hum
    "\u0623\u0646\u0627": "GS AE N AE AE",           # ana
    "\u0623\u0646\u062a": "GS AE N T AE",            # anta
    "\u0646\u062d\u0646": "N AE HH2 N UH",           # nahnu
    "\u0623\u0646": "GS AE N",                       # an
    "\u0625\u0646": "GS IH N N AE",                  # inna
    "\u0623\u0648": "GS AE W",                       # aw
    "\u062b\u0645": "TH UH M M AE",                  # thumma
    "\u0643\u0644": "K UH L L",                      # kull
    "\u0628\u0639\u062f": "B AA AYN D AE",           # ba'da
    "\u0642\u0628\u0644": "Q AA B L AE",             # qabla
    "\u0639\u0646\u062f": "AYN IH N D AE",           # inda
    "\u0628\u064a\u0646": "B AY N AE",               # bayna
    "\u062d\u062a\u0649": "HH2 AE T T AE AE",        # hatta
    "\u0625\u0630\u0627": "GS IH DH AE AE",          # idha
    "\u0644\u0643\u0646": "L AE AE K IH N",          # lakin
    "\u0647\u0646\u0627\u0643": "HH UH N AE AE K AE",  # hunaka
    "\u0647\u0646\u0627": "HH UH N AE AE",           # huna
    "\u063a\u064a\u0631": "GH AA Y DX",              # ghayr
    "\u0627\u0644\u0644\u0647": "GS AA L L AA AA HH",  # Allah
    "\u064a\u0648\u0645": "Y AW M",                  # yawm
    "\u0634\u064a\u0621": "SH AY GS",                # shay'
    "\u0627\u0644\u064a\u0648\u0645": "GS AE L Y AW M",  # al-yawm
    "\u0623\u064a\u0636\u0627": "GS AA Y DD AA N",   # aydan
    "\u062c\u062f\u0627": "JH IH D D AE N",          # jiddan
    "\u0648": "W AE",                                # wa (conjunction alone)
}

# Arabic-Indic digits, read digit by digit for now
DIGITS_AR = {
    "\u0660": "SS IH F R",
    "\u0661": "W AE AE HH2 IH D",
    "\u0662": "GS IH TH N AE AE N",
    "\u0663": "TH AE L AE AE TH AE",
    "\u0664": "GS AA DX B AA AYN AA",
    "\u0665": "KH AA M S AE",
    "\u0666": "S IH T T AE",
    "\u0667": "S AE B AYN AE",
    "\u0668": "TH AE M AE AE N IH Y AE",
    "\u0669": "T IH S AYN AE",
}

# Hook: a callable taking an undiacritized word and returning a
# diacritized word or None. Auto-wired to ar_diacritizer if bundled.
diacritize_hook = None

# When True, tanween is pronounced (nun / -an) even phrase-finally, as
# when reciting or teaching individual words. Default False = classical
# waqf (drops -un/-in, -an -> long aa).
pronounce_tanween_pause = False


def _init_diacritizer():
    global diacritize_hook
    try:
        try:
            from . import ar_diacritizer
        except ImportError:
            import ar_diacritizer
        diacritize_hook = ar_diacritizer.diacritize
    except Exception:
        pass


def _short_vowel(q, emphatic):
    if q == "a":
        return "AA" if emphatic else "AE"
    if q == "i":
        return "IH"
    return "UH"


def _long_vowel(q, emphatic):
    if q == "a":
        v = "AA" if emphatic else "AE"
        return [v, v]
    if q == "i":
        return ["IY", "IY"]
    return ["UW", "UW"]


_HARAKA_Q = {FATHA: "a", DAMMA: "u", KASRA: "i"}
_TANWEEN_Q = {TAN_FATH: "a", TAN_DAMM: "u", TAN_KASR: "i"}


def _has_diacritics(w):
    return any(c in DIACS for c in w)



def _is_wasl_alif(w, i):
    """True if the alif at index i is a silent hamzat-al-wasl seat:
    followed by article lam (sukun, or unmarked before a shadda sun
    letter) or by a consonant bearing sukun (form VII-X verbs etc)."""
    n = len(w)
    j = i + 1
    if j >= n:
        return False
    if w[j] == LAM:
        j2 = j + 1
        lam_marks = []
        while j2 < n and w[j2] in DIACS:
            lam_marks.append(w[j2])
            j2 += 1
        if SUKUN in lam_marks:
            return True
        if not lam_marks and j2 < n:
            j3 = j2 + 1
            while j3 < n and w[j3] in DIACS:
                if w[j3] == SHADDA:
                    return True
                j3 += 1
        return False
    if w[j] in CONS:
        j2 = j + 1
        while j2 < n and w[j2] in DIACS:
            if w[j2] == SUKUN:
                return True
            j2 += 1
    return False


def _diacritized_word(w, meta=None):
    """Fully rule-based phonemization of a vocalized Arabic word."""
    out = []
    n = len(w)
    i = 0
    emph = False
    if meta is None:
        meta = {}

    def collect_marks(j):
        gem = False
        marks = []
        while j < n and w[j] in DIACS:
            if w[j] == SHADDA:
                gem = True
            else:
                marks.append(w[j])
            j += 1
        return j, gem, marks

    # Definite article / hamzat al-wasl at word start
    if w.startswith(ALIF) or w.startswith(ALIF_WASLA):
        j = 1
        # skip a haraka written on the alif
        while j < n and w[j] in DIACS:
            j += 1
        if j < n and w[j] == LAM:
            # al- : check the lam's own marks first
            lam_marks = []
            jm = j + 1
            while jm < n and w[jm] in DIACS:
                lam_marks.append(w[jm])
                jm += 1
            out.append("GS")
            out.append("AE")
            if SHADDA in lam_marks:
                # The lam itself is geminated (e.g. الَّذِي, الَّتِي,
                # الَّذِينَ). This is NOT sun-letter assimilation — keep a
                # DOUBLED lam and continue into the next letter normally.
                out.append("L")
                out.append("L")
                i = jm            # resume at the letter after the lam+marks
            else:
                # al- : check sun-letter assimilation
                k = j + 1
                while k < n and w[k] in DIACS:
                    k += 1
                if k < n and w[k] in SUN_LETTERS:
                    # drop the lam; the shadda written on the sun letter
                    # produces the gemination when it is processed normally
                    i = k
                else:
                    out.append("L")
                    i = j + 1
        else:
            # bare initial alif: hamzat wasl; honour a written haraka,
            # default to /i/
            j0 = 1
            q = "i"
            while j0 < n and w[j0] in DIACS:
                if w[j0] == FATHA:
                    q = "a"
                elif w[j0] == DAMMA:
                    q = "u"
                j0 += 1
            out.append("GS")
            out.append(_short_vowel(q, False))
            i = j

    # A hamzat-al-wasl alif right after the article (al + iftiʿāl,
    # e.g. الانتخاب = al-intikhaab) is ELIDED so the lam joins the
    # next consonant directly instead of inserting a spurious aa.
    if out and out[-1] == "L":
        # Skip the lam's own diacritics (sukun) to find the next letter.
        p2 = i
        while p2 < n and w[p2] in DIACS:
            p2 += 1
        if p2 < n and w[p2] in (ALIF, ALIF_WASLA):
            # Elide the stem's wasl-alif (al-intikhaab, al-istiqlaal):
            # skip the alif and its haraka so the lam joins the next
            # consonant, which keeps its own vowel (no spurious aa).
            k2 = p2 + 1
            while k2 < n and w[k2] in DIACS:
                k2 += 1
            if k2 < n and w[k2] in CONS:
                i = k2

    while i < n:
        c = w[i]
        if c == TATWEEL or c in DIACS:
            i += 1
            continue

        if c == ALIF_MADDA:                      # 'aa
            out.extend(["GS"] + _long_vowel("a", emph))
            i += 1
            continue

        if (c == ALIF and out and out[-1] in ("AE", "AA", "IH", "UH")
                and _is_wasl_alif(w, i)):
            # Alif right after a one-letter prefix (wa/fa/bi/ka/la +
            # vowel): hamzat al-wasl if it introduces the article (lam
            # with sukun, or unmarked lam before a shadda sun letter) or a
            # verb whose first radical bears sukun. Silent: skip it.
            j = i + 1
            if j < n and w[j] == LAM:
                j2 = j + 1
                lam_marks = []
                while j2 < n and w[j2] in DIACS:
                    lam_marks.append(w[j2])
                    j2 += 1
                nxt_shadda = False
                j3 = j2 + 1
                while j3 < n and w[j3] in DIACS:
                    if w[j3] == SHADDA:
                        nxt_shadda = True
                    j3 += 1
                if SUKUN in lam_marks or (not lam_marks and nxt_shadda):
                    if j2 < n and w[j2] in SUN_LETTERS and nxt_shadda:
                        i = j2        # assimilated: sun letter geminates
                    else:
                        out.append("L")
                        i = j2
                        # elide a stem wasl-alif after the article
                        # (bi-al-intikhaab): skip alif + its haraka so
                        # the lam joins the next consonant directly.
                        if i < n and w[i] in (ALIF, ALIF_WASLA):
                            e2 = i + 1
                            while e2 < n and w[e2] in DIACS:
                                e2 += 1
                            if e2 < n and w[e2] in CONS:
                                i = e2
                    continue
            elif j < n and w[j] in CONS:
                j2 = j + 1
                marks2 = []
                while j2 < n and w[j2] in DIACS:
                    marks2.append(w[j2])
                    j2 += 1
                if SUKUN in marks2:   # e.g. wa-staqbala, wa-ntalaqa
                    i = j
                    continue

        if c == ALIF or c == ALIF_MAQSURA:
            # long /a:/ (bare alif mid-word) unless it is a silent tanween
            # seat (handled when the tanween itself was read)
            out.extend(_long_vowel("a", emph))
            i += 1
            continue

        if c == TA_MARBUTA:
            i2, gem, marks = collect_marks(i + 1)
            if marks and marks[0] in _HARAKA_Q:      # connected: /t/ + vowel
                out.append("T")
                out.append(_short_vowel(_HARAKA_Q[marks[0]], False))
                meta["tm"] = True
            elif marks and marks[0] in _TANWEEN_Q:
                out.append("T")
                out.append(_short_vowel(_TANWEEN_Q[marks[0]], False))
                out.append("N")
                meta["tm"] = True
            elif meta.get("_ctx"):                   # followed by a word
                out.append("T")                      # samakat al-qirsh
                out.append("AE")
                meta["tm"] = True
            else:                                    # pausal: -ah
                out.append("AE")
                out.append("HH")
            i = i2
            continue

        ph = CONS.get(c)
        if ph is None:
            i += 1
            continue

        i2, gem, marks = collect_marks(i + 1)
        # hamza-carrying alifs supply an inherent vowel when unmarked:
        # ا"إ"=kasra, "أ"=fatha. (آ handled as ALIF_MADDA above.)
        if c == ALIF_HAMZA_BELOW and not any(m in _HARAKA_Q for m in marks) \
                and SUKUN not in marks and i == 0:
            marks = [KASRA] + marks
        elif c == ALIF_HAMZA and not any(m in _HARAKA_Q for m in marks) \
                and SUKUN not in marks and not any(m in _TANWEEN_Q for m in marks):
            marks = [FATHA] + marks
        out.append(ph)
        if gem:
            out.append(ph)
        emph = ph in EMPHATIC
        if DAGGER in marks:
            marks = [m for m in marks if m != FATHA]

        vowel_done = False
        for m in marks:
            if m in _HARAKA_Q:
                q = _HARAKA_Q[m]
                nxt = w[i2] if i2 < n else ""
                nxt2 = w[i2 + 1] if i2 + 1 < n else ""
                if q == "a" and nxt in (ALIF, ALIF_MAQSURA) \
                        and nxt2 not in _TANWEEN_Q \
                        and not (nxt == ALIF and _is_wasl_alif(w, i2)):
                    out.extend(_long_vowel("a", emph))
                    i2 += 1
                elif q == "a" and nxt == WAW and (nxt2 == SUKUN or
                                                  nxt2 in CONS or not nxt2):
                    out.append("AW")
                    i2 += 1 + (1 if nxt2 == SUKUN else 0)
                elif q == "a" and nxt == YA and (nxt2 == SUKUN or
                                                 nxt2 in CONS or not nxt2):
                    out.append("AY")
                    i2 += 1 + (1 if nxt2 == SUKUN else 0)
                elif q == "u" and nxt == WAW and nxt2 not in _HARAKA_Q \
                        and nxt2 != SHADDA:
                    out.extend(_long_vowel("u", emph))
                    i2 += 1 + (1 if nxt2 == SUKUN else 0)
                elif q == "i" and nxt == YA and nxt2 not in _HARAKA_Q \
                        and nxt2 != SHADDA:
                    out.extend(_long_vowel("i", emph))
                    i2 += 1 + (1 if nxt2 == SUKUN else 0)
                else:
                    out.append(_short_vowel(q, emph))
                vowel_done = True
            elif m in _TANWEEN_Q:
                out.append(_short_vowel(_TANWEEN_Q[m], emph))
                out.append("N")
                if i2 < n and w[i2] == ALIF:      # silent tanween seat
                    i2 += 1
                vowel_done = True
            elif m == DAGGER:
                out.extend(_long_vowel("a", emph))
                vowel_done = True
            # SUKUN: no vowel
        del vowel_done
        i = i2

    return out


def _skeleton_word(w):
    """Fallback for undiacritized words: consonant skeleton with default
    short /a/ vowels; long-vowel letters read long. Intelligible, not
    grammatical."""
    # strip diacritics that may be sprinkled inconsistently
    w = "".join(c for c in w if c not in DIACS and c != TATWEEL)
    out = []
    n = len(w)
    i = 0
    emph = False

    # definite article
    if w.startswith(ALIF + LAM) and n > 3:
        out.extend(["GS", "AE"])
        if w[2] in SUN_LETTERS:
            out.append(CONS.get(w[2], ""))   # gemination: ash-shams
            i = 2
        else:
            out.append("L")
            i = 2
    # single-letter conjunction prefix waw
    elif w.startswith(WAW) and n > 3:
        out.extend(["W", "AE"])
        i = 1

    while i < n:
        c = w[i]
        last = i == n - 1
        if c == ALIF_MADDA:
            out.extend(["GS", "AE", "AE"])
        elif c == ALIF and i == 0:
            out.extend(["GS", "AE"])          # word-initial hamza(t wasl)
        elif c in (ALIF, ALIF_MAQSURA):
            v = "AA" if emph else "AE"
            out.extend([v, v])
        elif c == TA_MARBUTA:
            out.append("AE")
        elif c == WAW and 0 < i and not last and w[i - 1] not in (ALIF,):
            out.extend(["UW", "UW"])      # mid-word waw as /u:/
        elif c == YA and 0 < i:
            out.extend(["IY", "IY"])      # mid/final ya as /i:/
        else:
            ph = CONS.get(c)
            if ph:
                out.append(ph)
                emph = ph in EMPHATIC
                nxt = w[i + 1] if not last else ""
                if not last and nxt not in (ALIF, ALIF_MAQSURA, WAW, YA,
                                            TA_MARBUTA, ALIF_MADDA):
                    out.append("AA" if emph else "AE")
        i += 1
    return out


SHORT_V = {"AE", "AA", "IH", "UH"}
_DIPH = {"AW", "AY"}
_V_ALL = SHORT_V | {"IY", "UW"} | _DIPH


_V_FOR_RAA = {"AE", "AHA", "AA", "IH", "UH", "IY", "UW", "AW", "AY"}


def _raa_tafkhim(p):
    """Tajweed quality of raa. Heavy (DXQ): raa with fatha or damma,
    or sakinah preceded by a/u, or sakinah before an emphatic. Light
    (DX): raa with kasra, sakinah after kasra, or after yaa sakinah."""
    n = len(p)
    i = 0
    while i < n:
        if p[i] != "DX":
            i += 1
            continue
        j = i
        while j + 1 < n and p[j + 1] == "DX":     # geminate run
            j += 1
        nxt = p[j + 1] if j + 1 < n else ""
        nxt2 = p[j + 2] if j + 2 < n else ""
        prev = p[i - 1] if i > 0 else ""
        heavy = False
        if nxt in ("AE", "AHA", "AA", "UH", "UW", "AW"):
            heavy = True                      # raa with fatha/damma
        elif nxt not in _V_FOR_RAA:           # raa sakinah / final
            prev2 = p[i - 2] if i > 1 else ""
            if prev in ("AE", "AHA", "AA", "UH", "UW", "AW") and prev2 != prev:
                heavy = True                  # short a/u only, not aa/uu
            elif prev in ("IH", "IY"):
                if nxt in EMPHATIC or nxt2 in EMPHATIC:
                    heavy = True
        if heavy:
            for k in range(i, j + 1):
                p[k] = "DXQ"
        i = j + 1
    return p


def _spread_emphasis(p):
    """Bidirectional tafkhim: /a/ near emphatics (and non-front raa)
    backs to [AA]. Whole long-vowel runs shift together."""
    n = len(p)
    mark = set()
    for i, ph in enumerate(p):
        if ph in EMPHATIC:
            cand = (i - 3, i - 2, i - 1, i + 1, i + 2, i + 3)

        else:
            continue
        for j in cand:
            if 0 <= j < n and p[j] == "AE":
                mark.add(j)
    for j in list(mark):
        k = j - 1
        while k >= 0 and p[k] == "AE":
            mark.add(k)
            k -= 1
        k = j + 1
        while k < n and p[k] == "AE":
            mark.add(k)
            k += 1
    for j in mark:
        p[j] = "AA"
    return p


def _assign_stress(p):
    """Classical Arabic stress from syllable weight: final superheavy,
    else heavy penult, else antepenult. Marks the nucleus with '1'."""
    sylls = []          # (nucleus index, weight in morae+coda)
    i = 0
    n = len(p)
    while i < n:
        if p[i] in _V_ALL:
            start = i
            if p[i] in _DIPH:
                morae = 2
                j = i + 1
            else:
                morae = 1
                j = i + 1
                while j < n and p[j] == p[i]:
                    morae = 2
                    j += 1
            cons = 0
            k = j
            while k < n and p[k] not in _V_ALL:
                cons += 1
                k += 1
            coda = cons if k >= n else max(0, cons - 1)
            sylls.append((start, morae + coda))
            i = k
        else:
            i += 1
    if not sylls:
        return p
    if len(sylls) == 1:
        if sylls[0][1] < 2:
            return p          # light monosyllable: unstressed clitic
        s = 0
    elif sylls[-1][1] >= 3:
        s = len(sylls) - 1
    elif sylls[-2][1] >= 2:
        s = len(sylls) - 2
    elif len(sylls) >= 3:
        s = len(sylls) - 3
    else:
        s = 0
    idx = sylls[s][0]
    p[idx] = p[idx] + "1"
    return p


def _pausal(p, meta):
    """Waqf: pausal form of a phrase-final word."""
    p = list(p)
    if meta.get("tm"):
        if p and p[-1] == "N":
            p.pop()
        if p and p[-1] in SHORT_V:
            p.pop()
        if p and p[-1] == "T":
            p.pop()
            p.append("HH")
        return p
    if (len(p) >= 2 and p[-1] == "N" and p[-2] in SHORT_V
            and (len(p) < 3 or p[-3] != p[-2])):
        if pronounce_tanween_pause:
            return p                     # keep the nun (teaching mode)
        # classical waqf: -an becomes long -aa; -un/-in drop entirely
        if p[-2] in ("AE", "AA"):
            v = p[-2]
            return p[:-2] + [v, v]
        return p[:-2]
    if (len(p) >= 2 and p[-1] in SHORT_V and p[-2] not in _V_ALL):
        p.pop()          # final short case vowel
    return p


def _fix_allah(p):
    """The word Allah: long backed aa on a dark, emphatic geminate lam."""
    for i in range(len(p) - 3):
        if (p[i] == "L" and p[i + 1] == "L" and p[i + 2] in ("AE", "AA")
                and p[i + 3] == "HH"):
            p[i] = "LD"
            p[i + 1] = "LD"
            p[i + 2:i + 3] = ["AA", "AA"]
            if i >= 1 and p[i - 1] in ("AE",):
                p[i - 1] = "AA"
            break
    return p


_NATIVE = {"AE": "AHA", "L": "LT"}


def _nativize(p):
    """Swap in the Arabic-specific segments (clear lam, central fatha),
    preserving stress digits."""
    out = []
    for ph in p:
        digit = ""
        if ph and ph[-1] in "012":
            ph, digit = ph[:-1], ph[-1]
        out.append(_NATIVE.get(ph, ph) + digit)
    return out


def _split_marks(w):
    out = []
    for c in w:
        if c in DIACS and out:
            out[-1][1] += c
        else:
            out.append([c, ""])
    return out


def _merge_partial(word, prev=None):
    """Partial harakat are constraints, not a full transcription: fill
    the unmarked letters from the known/predicted vocalization, letting
    the user's own marks override it letter by letter."""
    pairs = _split_marks(word)
    unmarked = sum(1 for c, m in pairs[:-1]
                   if c in CONS and not m)
    if unmarked == 0:
        return word
    bare = "".join(c for c, m in pairs)
    cand = None
    if diacritize_hook:
        try:
            cand = diacritize_hook(bare, prev)
        except TypeError:
            cand = diacritize_hook(bare)
    if not cand:
        return word
    cpairs = _split_marks(cand)
    if [c for c, m in cpairs] != [c for c, m in pairs]:
        return word
    out = []
    for (c, um), (_, cm) in zip(pairs, cpairs):
        m = cm
        if um:
            if SHADDA in um and SHADDA not in cm and \
                    not any(x in um for x in _HARAKA_Q) and \
                    not any(x in um for x in _TANWEEN_Q):
                m = um + "".join(x for x in cm if x != SHADDA)
            else:
                m = um
        out.append(c + m)
    return "".join(out)


def _word(word, prev=None, nxt=None):
    """Phonemize one Arabic word; returns (phones, meta)."""
    word = word.strip(TATWEEL)
    meta = {"wasl": bool(word) and word[0] in (ALIF, ALIF_WASLA)}
    if not word:
        return [], meta
    if nxt is not None:
        meta["_ctx"] = True
    if _has_diacritics(word):
        word = _merge_partial(word, prev)
        return _fix_allah(_spread_emphasis(_raa_tafkhim(
            _diacritized_word(word, meta)))), meta
    if word in LEX_AR:
        return _spread_emphasis(_raa_tafkhim(LEX_AR[word].split())), meta
    if diacritize_hook:
        try:
            d = diacritize_hook(word, prev, nxt)
        except TypeError:
            try:
                d = diacritize_hook(word, prev)
            except TypeError:
                d = diacritize_hook(word)
        if d:
            return _fix_allah(_spread_emphasis(_raa_tafkhim(
                _diacritized_word(d, meta)))), meta
    return _spread_emphasis(_raa_tafkhim(_skeleton_word(word))), meta


def word_to_phonemes(word):
    return _word(word)[0]


# ---------------------------------------------------------------------------
# Arabic numbers (masculine counting forms, simplified pausal endings)
# ---------------------------------------------------------------------------
_N_UNIT = {
    0: "SS IH F R", 1: "W AA AA HH2 IH D", 2: "GS IH TH N AE AE N",
    3: "TH AE L AE AE TH AE", 4: "GS AA DX B AA AYN AA",
    5: "KH AA M S AE", 6: "S IH T T AE", 7: "S AE B AYN AE",
    8: "TH AE M AE AE N IH Y AE", 9: "T IH S AYN AE",
    10: "AYN AA SH AA DX AX",
}
_N_TEEN_10 = "AYN AA SH AA DX"          # 'ashar in 11-19
_N_TENS = {
    2: "AYN IH SH DX UW UW N", 3: "TH AE L AE AE TH UW UW N",
    4: "GS AA DX B AA AYN UW UW N", 5: "KH AA M S UW UW N",
    6: "S IH T T UW UW N", 7: "S AE B AYN UW UW N",
    8: "TH AE M AE AE N UW UW N", 9: "T IH S AYN UW UW N",
}
_N_WA = "W AE"
_N_MIA = "M IH GS AE"
_N_MIATAN = "M IH GS AE T AE AE N"
_N_ALF = "GS AE L F"
_N_ALFAN = "GS AE L F AE AE N"
_N_AALAF = "GS AE AE L AE AE F"
_N_MILYUN = "M IH L Y UW UW N"
_N_MALAYIN = "M AE L AE Y IY Y IY N"
_N_FASILA = "F AA SS IH L AE"
_N_TEEN_1 = "GS AE HH2 AA D"            # ahada (11)
_N_TEEN_2 = "GS IH TH N AE AE"          # ithna (12)


def _num_lt100(n):
    words = []
    if n < 3 and n >= 0:
        return [_N_UNIT[n]]
    if n <= 10:
        return [_N_UNIT[n]]
    if n < 20:
        u = n - 10
        first = {1: _N_TEEN_1, 2: _N_TEEN_2}.get(u, _N_UNIT[u])
        return [first, _N_TEEN_10]
    t, u = divmod(n, 10)
    if u:
        words.append(_N_UNIT[u])
        words.append(_N_WA)
    words.append(_N_TENS[t])
    return words


def _num_lt1000(n):
    words = []
    h, r = divmod(n, 100)
    if h == 1:
        words.append(_N_MIA)
    elif h == 2:
        words.append(_N_MIATAN)
    elif h:
        words.append(_N_UNIT[h])
        words.append(_N_MIA)
    if r:
        if words:
            words.append(_N_WA)
        words.extend(_num_lt100(r))
    return words


def _int_words(n):
    if n == 0:
        return [_N_UNIT[0]]
    words = []
    m, rest = divmod(n, 10 ** 6)
    if m:
        if m == 1:
            words.append(_N_MILYUN)
        elif 3 <= m <= 10:
            words.extend(_num_lt1000(m) + [_N_MALAYIN])
        else:
            words.extend(_int_words(m) + [_N_MILYUN])
    k, rest = divmod(rest, 1000)
    if k:
        if words:
            words.append(_N_WA)
        if k == 1:
            words.append(_N_ALF)
        elif k == 2:
            words.append(_N_ALFAN)
        elif 3 <= k <= 10:
            words.extend(_num_lt1000(k) + [_N_AALAF])
        else:
            words.extend(_num_lt1000(k) + [_N_ALF])
    if rest:
        if words:
            words.append(_N_WA)
        words.extend(_num_lt1000(rest))
    return words


_AR_DIGIT_TRANS = {0x0660 + i: str(i) for i in range(10)}


def _number_words(tok):
    tok = tok.translate(_AR_DIGIT_TRANS).replace("\u066B", ".")
    try:
        if "." in tok:
            whole, frac = tok.split(".", 1)
            words = _int_words(int(whole)) if whole else [_N_UNIT[0]]
            words.append(_N_FASILA)
            for d in frac:
                if d.isdigit():
                    words.append(_N_UNIT[int(d)])
            return words
        n = int(tok)
        if n >= 10 ** 9:
            return [_N_UNIT[int(d)] for d in tok]
        return _int_words(n)
    except (ValueError, OverflowError):
        return [_N_UNIT[int(d)] for d in tok if d.isdigit()]


# ---------------------------------------------------------------------------
# Sentence level
# ---------------------------------------------------------------------------
_AR_TOK_RE = re.compile(
    "[\u0621-\u064A\u0671\u0640\u064B-\u0652\u0670]+"
    "|[0-9]+(?:[.][0-9]+)?"
    "|[\u0660-\u0669]+(?:\u066B[\u0660-\u0669]+)?"
    "|.", re.S)

_PUNCT = {
    ".": "_.", "!": "_!", "?": "_?", ",": "_,", ";": "_;", ":": "_:",
    "\u060C": "_,",   # Arabic comma
    "\u061F": "_?",   # Arabic question mark
    "\u061B": "_;",   # Arabic semicolon
    "\n": "_.",
}

_PAUSE_SET = {"_.", "_!", "_?", "_,", "_;", "_:"}


_NEURAL_ON = True



# Arabic number -> spelled-out Arabic TEXT (for the neural voice, whose
# vocabulary has no digit symbols). Uses masculine base forms; good enough
# for TTS. Handles 0..billions and simple decimals.
_AR_ONES = ["صفر", "واحد", "اثنان", "ثلاثة", "أربعة", "خمسة", "ستة",
            "سبعة", "ثمانية", "تسعة", "عشرة", "أحد عشر", "اثنا عشر",
            "ثلاثة عشر", "أربعة عشر", "خمسة عشر", "ستة عشر",
            "سبعة عشر", "ثمانية عشر", "تسعة عشر"]
_AR_TENS = {2: "عشرون", 3: "ثلاثون", 4: "أربعون", 5: "خمسون",
            6: "ستون", 7: "سبعون", 8: "ثمانون", 9: "تسعون"}
_AR_HUND = {1: "مئة", 2: "مئتان", 3: "ثلاثمئة", 4: "أربعمئة",
            5: "خمسمئة", 6: "ستمئة", 7: "سبعمئة", 8: "ثمانمئة",
            9: "تسعمئة"}


def _ar_int_text(n):
    if n < 20:
        return _AR_ONES[n]
    if n < 100:
        t, o = divmod(n, 10)
        return (_AR_ONES[o] + " و" + _AR_TENS[t]) if o else _AR_TENS[t]
    if n < 1000:
        h, r = divmod(n, 100)
        s = _AR_HUND[h]
        return (s + " و" + _ar_int_text(r)) if r else s
    if n < 1000000:
        th, r = divmod(n, 1000)
        if th == 1:
            s = "ألف"
        elif th == 2:
            s = "ألفان"
        elif th < 11:
            s = _ar_int_text(th) + " آلاف"
        else:
            s = _ar_int_text(th) + " ألف"
        return (s + " و" + _ar_int_text(r)) if r else s
    if n < 10**9:
        m, r = divmod(n, 1000000)
        if m == 1:
            s = "مليون"
        elif m == 2:
            s = "مليونان"
        else:
            s = _ar_int_text(m) + " مليون"
        return (s + " و" + _ar_int_text(r)) if r else s
    # very large: read digit by digit
    return " ".join(_AR_ONES[int(d)] for d in str(n))


def _spell_numbers_ar(text):
    """Replace ASCII and Arabic-Indic digit runs with spelled Arabic words
    so the neural voice pronounces them. Decimals read 'فاصلة' + digits."""
    import re
    trans = str.maketrans("٠١٢٣٤٥"
                          "٦٧٨٩", "0123456789")

    def repl(m):
        tok = m.group(0).translate(trans).replace("٫", ".")
        try:
            if "." in tok:
                whole, frac = tok.split(".", 1)
                w = _ar_int_text(int(whole)) if whole else "صفر"
                f = " ".join(_AR_ONES[int(d)] for d in frac if d.isdigit())
                return w + " فاصلة " + f
            return _ar_int_text(int(tok))
        except (ValueError, OverflowError):
            return tok

    return re.sub(r"[0-9٠-٩]+(?:[.٫][0-9٠-٩]+)?",
                  repl, text)


# --- Arabic text cleanup (symbol names, emoji, noise) -----------------------
# Symbol names are pre-diacritized so the neural voice reads them correctly.
# Adapted from the NabraTTS add-on (author "pbt"), shared by Ilyas Dragonoid.
_AR_SYMBOL_MAP = {
    "+": " زَائِد ",
    "*": " ضَرْب ",
    "=": " يُسَاوِي ",
    ">": " أَكْبَرُ مِنْ ",
    "<": " أَصْغَرُ مِنْ ",
    "%": " بِالْمِئَةِ ",
    "$": " دُولَار ",
    "\u20ac": " يُورُو ",
    "\u00a3": " جُنَيْه ",
    "&": " وَ ",
    "|": " خَطٌّ عَمُودِيّ ",
    "~": " تَقْرِيبًا ",
}

_AR_EMOJI_RE = re.compile(
    "[\U00002600-\U000027BF"
    "\U0001F300-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\u200d\u200c\ufeff"
    "\u2066-\u2069]+",
    flags=re.UNICODE,
)
# runs of repeated decoration characters ("=====", "-----", "***")
_AR_NOISE_RE = re.compile(r"[-_=*~`|\\<>{}]{3,}")
_AR_MULTISPACE_RE = re.compile(r"[ \t]{2,}")


# Quranic ornate brackets ﴿﴾ and waqf/pause signs (U+06D6..U+06ED, U+08xx)
# are visual ornaments that some diacritizers mishandle. Remove them before
# processing so text after an ayah still gets diacritized. NOTE: the
# superscript alef U+0670 is NOT removed — it affects pronunciation.
_QURANIC_ORNAMENTS_RE = re.compile(
    "[\uFD3E\uFD3F]"            # ornate parentheses ﴿ ﴾
    "|[\u06D6-\u06DC]"          # small high waqf marks ۖ ۗ ۘ ۙ ۚ ۛ ۜ
    "|[\u06DD-\u06ED]"          # other Quranic annotation signs
    "|[\u0615-\u061A]"          # small high signs
)


def clean_arabic_text(text):
    """Tidy an Arabic run before diacritization/synthesis: drop emoji and
    decorative runs, strip Quranic ornaments that confuse diacritizers, and
    speak stray symbols as Arabic words. URLs and English words are
    intentionally left alone — ClaritySynth speaks them with its English
    voice instead of hiding or transliterating them."""
    if not text:
        return text
    text = _QURANIC_ORNAMENTS_RE.sub("", text)
    text = _AR_NOISE_RE.sub(" ", text)
    text = _AR_EMOJI_RE.sub(" ", text)
    for sym, word in _AR_SYMBOL_MAP.items():
        if sym in text:
            text = text.replace(sym, word)
    text = _AR_MULTISPACE_RE.sub(" ", text)
    return text.strip()




def _strip_marks(s):
    """Remove Arabic diacritics (keeping the letters) for comparison."""
    return "".join(c for c in s
                   if not ("\u064B" <= c <= "\u0652" or c == "\u0670"))

def _diacritize(text):
    """Diacritize using the user's chosen tashkeel library (libtashkeel,
    Rawi, Shakkelha, Shakkala) with automatic fallback."""
    try:
        from . import ar_tashkeel
    except ImportError:
        try:
            import ar_tashkeel
        except Exception:
            ar_tashkeel = None
    except Exception:
        ar_tashkeel = None
    if ar_tashkeel is not None:
        try:
            out = ar_tashkeel.diacritize_text(text)
            if out:
                return out
        except Exception:
            pass
    # legacy fallback
    try:
        return ar_neural.diacritize_text(text)
    except Exception:
        return None

def _neural_pre(text):
    """If the neural diacritizer is available, vocalize the whole text
    first (full-context inference). Returns diacritized text or original."""
    if not _NEURAL_ON:
        return text
    try:
        try:
            from . import ar_neural
        except ImportError:
            import ar_neural
        # tatweel (kashida) confuses the diacritizer — remove it first
        clean = text.replace("\u0640", "")
        # the neural voice has no digit tokens: spell numbers as Arabic
        # words so they are actually pronounced (e.g. 13 -> ثلاثة عشر)
        clean = _spell_numbers_ar(clean)
        # Count Arabic letters vs diacritics. Fully-diacritized text has a
        # diacritic after most letters; PARTIALLY diacritized text (e.g.
        # only tanween on a few words) must still be fully vocalized, so we
        # do NOT skip just because a few marks exist.
        letters = sum(1 for c in clean if "\u0621" <= c <= "\u064A")
        marks = sum(1 for c in clean if "\u064B" <= c <= "\u0652")
        has_ar = letters > 0
        # Text that already carries a meaningful amount of diacritics is
        # human/source-vocalized (e.g. a hadith with full tashkeel). Keep
        # it verbatim — re-diacritizing would discard correct scholarly
        # marks and risk worse output. Only vocalize genuinely bare text
        # (under ~25% marked). We still spell numbers either way.
        already_vocalized = letters > 0 and marks >= letters * 0.25
        if has_ar and already_vocalized:
            # Keep the human's diacritics, but fill in words that are still
            # bare or badly under-marked.
            #
            # IMPORTANT: diacritize the WHOLE CLAUSE in one pass and take
            # only the words we need from it. Diacritizing an isolated word
            # starves the model of context — Rawi in particular then drops
            # hamza (سأل -> سَالْ, وقرأ -> وَقَرَا, وائل -> وَأَيْلَ). With the
            # full clause it places every hamza correctly.
            try:
                import re as _re
                toks = _re.split(r"(\s+)", clean)
                # find CONTIGUOUS runs of bare words (ignoring whitespace) so
                # each run is diacritized with its own local context, then
                # spliced back by position. This is robust to brackets,
                # braces and digits inside the text (e.g. a Quran citation),
                # which previously broke a single whole-clause alignment and
                # left everything after the ayah un-diacritized.
                def _bare(tok):
                    wl = sum(1 for c in tok if "\u0621" <= c <= "\u064A")
                    wm = sum(1 for c in tok if "\u064B" <= c <= "\u0652")
                    return wl >= 2 and wm < wl * 0.4

                out_words = list(toks)
                i = 0
                n = len(toks)
                changed = False
                while i < n:
                    tok = toks[i]
                    if not tok or tok.isspace() or not _bare(tok):
                        i += 1
                        continue
                    # gather a run: bare words + the whitespace between them
                    j = i
                    run_idx = []
                    while j < n:
                        tj = toks[j]
                        if tj.isspace():
                            j += 1
                            continue
                        if _bare(tj):
                            run_idx.append(j)
                            j += 1
                        else:
                            break
                    # the raw substring spanning this run (with its spaces)
                    run_text = "".join(toks[i:j])
                    diac = _diacritize(run_text)
                    if diac:
                        # map diacritized words back onto the bare tokens by
                        # matching stripped skeletons in order
                        d_words = [w for w in _re.split(r"\s+", diac) if w]
                        di = 0
                        for k in run_idx:
                            base_o = _strip_marks(toks[k])
                            # advance di to the matching skeleton
                            while (di < len(d_words)
                                   and _strip_marks(d_words[di]) != base_o):
                                di += 1
                            if di < len(d_words):
                                cand = d_words[di]
                                if (sum(1 for c in cand
                                        if "\u064B" <= c <= "\u0652")
                                        > sum(1 for c in toks[k]
                                              if "\u064B" <= c <= "\u0652")):
                                    out_words[k] = cand
                                    changed = True
                                di += 1
                    i = j
                merged = "".join(out_words)
                if merged and changed:
                    return merged
            except Exception:
                pass
            return clean
        if has_ar and not already_vocalized:
            # Diacritize clause by clause: long strings with : " ; etc.
            # confuse the neural model, so split, vocalize each piece, and
            # rejoin with the original delimiters preserved.
            import re
            pieces = re.split(r'([\.!\?;:،؛؟\n\"\u00ab\u00bb]+\s*)', clean)
            outp = []
            for seg in pieces:
                if seg and any("\u0621" <= c <= "\u064A" for c in seg):
                    dd = _diacritize(seg)
                    outp.append(dd if dd else seg)
                else:
                    outp.append(seg)
            joined = "".join(outp)
            if any("\u064B" <= c <= "\u0652" for c in joined):
                return joined
        # Even if diacritization did not run (already-diacritized text, or
        # neural tier dormant), still return the number-spelled version so
        # digits are spoken as words rather than skipped/hummed.
        if clean != text:
            return clean
    except Exception:
        pass
    return text


def text_to_tokens(text):
    text = _neural_pre(text)
    # First pass: words / numbers / pauses
    entries = []
    _prev_bare = [None]
    _matches = [m.group(0) for m in _AR_TOK_RE.finditer(text)]
    for mi, tok in enumerate(_matches):
        if tok.isspace():
            continue
        c0 = tok[0]
        if c0.isdigit() or "\u0660" <= c0 <= "\u0669":
            for wphones in _number_words(tok):
                entries.append(("w", _spread_emphasis(
                    _raa_tafkhim(wphones.split())), {}))
            continue
        if "\u0621" <= c0 <= "\u0671":
            b = "".join(c for c in tok if c not in DIACS)
            nb = None
            for jj in range(mi + 1, len(_matches)):
                nt = _matches[jj]
                if nt and "\u0621" <= nt[0] <= "\u0671":
                    nb = "".join(c for c in nt if c not in DIACS)
                    break
            phones, meta = _word(tok, _prev_bare[0], nb)
            _prev_bare[0] = b
            if phones:
                entries.append(("w", phones, meta))
            continue
        p = _PUNCT.get(tok)
        if p:
            entries.append(("p", p, None))

    # Second pass: pausal forms at phrase ends, liaison across words
    tokens = []
    prev_vowel = False
    last_was_pause = True
    for idx, (kind, val, meta) in enumerate(entries):
        if kind == "p":
            if not last_was_pause:
                tokens.append(val)
                last_was_pause = True
            prev_vowel = False
            continue
        phones = val
        nxt_pause = (idx + 1 >= len(entries)
                     or entries[idx + 1][0] == "p")
        if nxt_pause:
            phones = _pausal(phones, meta)
            # raa quality depends on the pausal form, not the full form
            phones = _raa_tafkhim(
                ["DX" if x == "DXQ" else x for x in phones])
        if (meta.get("wasl") and prev_vowel and len(phones) > 2
                and phones[0] == "GS"):
            phones = phones[2:]          # bismi-llaahi, not bismi 'allaahi
        if not phones:
            continue
        phones = _nativize(_assign_stress(list(phones)))
        tokens.append("_w")
        tokens.extend(phones)
        last = phones[-1]
        if last and last[-1] in "012":
            last = last[:-1]
        prev_vowel = last in _V_ALL
        last_was_pause = False
    return tokens


_init_diacritizer()


# ---------------------------------------------------------------------------
# Character echo: proper Arabic names for letters, harakat and punctuation
# ---------------------------------------------------------------------------
_CHAR_NAMES_AR = {
    "\u0627": "\u0623\u064e\u0644\u0650\u0641",                # alif
    "\u0628": "\u0628\u064e\u0627\u0621",                       # baa
    "\u062a": "\u062a\u064e\u0627\u0621",
    "\u062b": "\u062b\u064e\u0627\u0621",
    "\u062c": "\u062c\u0650\u064a\u0645",
    "\u062d": "\u062d\u064e\u0627\u0621",
    "\u062e": "\u062e\u064e\u0627\u0621",
    "\u062f": "\u062f\u064e\u0627\u0644",
    "\u0630": "\u0630\u064e\u0627\u0644",
    "\u0631": "\u0631\u064e\u0627\u0621",
    "\u0632": "\u0632\u064e\u0627\u064a",
    "\u0633": "\u0633\u0650\u064a\u0646",
    "\u0634": "\u0634\u0650\u064a\u0646",
    "\u0635": "\u0635\u064e\u0627\u062f",
    "\u0636": "\u0636\u064e\u0627\u062f",
    "\u0637": "\u0637\u064e\u0627\u0621",
    "\u0638": "\u0638\u064e\u0627\u0621",
    "\u0639": "\u0639\u064e\u064a\u0652\u0646",
    "\u063a": "\u063a\u064e\u064a\u0652\u0646",
    "\u0641": "\u0641\u064e\u0627\u0621",
    "\u0642": "\u0642\u064e\u0627\u0641",
    "\u0643": "\u0643\u064e\u0627\u0641",
    "\u0644": "\u0644\u064e\u0627\u0645",
    "\u0645": "\u0645\u0650\u064a\u0645",
    "\u0646": "\u0646\u064f\u0648\u0646",
    "\u0647": "\u0647\u064e\u0627\u0621",
    "\u0648": "\u0648\u064e\u0627\u0648",
    "\u064a": "\u064a\u064e\u0627\u0621",
    "\u0621": "\u0647\u064e\u0645\u0652\u0632\u064e\u0629",   # hamza
    "\u0623": "\u0623\u064e\u0644\u0650\u0641 \u0647\u064e\u0645\u0652\u0632\u064e\u0629",
    "\u0625": "\u0623\u064e\u0644\u0650\u0641 \u0647\u064e\u0645\u0652\u0632\u064e\u0629",
    "\u0622": "\u0623\u064e\u0644\u0650\u0641 \u0645\u064e\u062f\u0651\u064e\u0629",
    "\u0624": "\u0648\u064e\u0627\u0648 \u0647\u064e\u0645\u0652\u0632\u064e\u0629",
    "\u0626": "\u064a\u064e\u0627\u0621 \u0647\u064e\u0645\u0652\u0632\u064e\u0629",
    "\u0629": "\u062a\u064e\u0627\u0621 \u0645\u064e\u0631\u0652\u0628\u064f\u0648\u0637\u064e\u0629",
    "\u0649": "\u0623\u064e\u0644\u0650\u0641 \u0645\u064e\u0642\u0652\u0635\u064f\u0648\u0631\u064e\u0629",
    FATHA: "\u0641\u064e\u062a\u0652\u062d\u064e\u0629",
    DAMMA: "\u0636\u064e\u0645\u0651\u064e\u0629",
    KASRA: "\u0643\u064e\u0633\u0652\u0631\u064e\u0629",
    SUKUN: "\u0633\u064f\u0643\u064f\u0648\u0646",
    SHADDA: "\u0634\u064e\u062f\u0651\u064e\u0629",
    TAN_FATH: "\u062a\u064e\u0646\u0652\u0648\u0650\u064a\u0646 \u0641\u064e\u062a\u0652\u062d",
    TAN_DAMM: "\u062a\u064e\u0646\u0652\u0648\u0650\u064a\u0646 \u0636\u064e\u0645\u0651",
    TAN_KASR: "\u062a\u064e\u0646\u0652\u0648\u0650\u064a\u0646 \u0643\u064e\u0633\u0652\u0631",
    "\u060c": "\u0641\u064e\u0627\u0635\u0650\u0644\u064e\u0629",
    "\u061f": "\u0639\u064e\u0644\u064e\u0627\u0645\u064e\u0629 \u0627\u0633\u0652\u062a\u0650\u0641\u0652\u0647\u064e\u0627\u0645",
    "\u061b": "\u0641\u064e\u0627\u0635\u0650\u0644\u064e\u0629 \u0645\u064e\u0646\u0652\u0642\u064f\u0648\u0637\u064e\u0629",
}


def char_to_tokens(ch):
    """Speak an Arabic character by its proper name."""
    name = _CHAR_NAMES_AR.get(ch)
    if name is None and "\u0660" <= ch <= "\u0669":
        return ["_w"] + _nativize(DIGITS_AR[ch].split())
    if name is None:
        return text_to_tokens(ch)
    toks = []
    for wd in name.split():
        phones, meta = _word(wd)
        phones = _pausal(phones, meta)
        toks.append("_w")
        toks.extend(_nativize(_assign_stress(list(phones))))
    return toks
