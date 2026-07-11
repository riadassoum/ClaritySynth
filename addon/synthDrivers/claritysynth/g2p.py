# -*- coding: utf-8 -*-
# ClaritySynth: English text-to-phoneme conversion.
#
# Pipeline: normalize text -> tokenize -> per word: exceptions dictionary,
# then NRL-style letter-to-sound rules (after the classic 1976 Naval Research
# Laboratory rule set, condensed and adapted). Output is a flat token list of
# phoneme names plus control tokens: "_w" (word boundary) and pause tokens
# ("_.", "_,", "_?", ...).

import re
import os
import gzip
import threading

# ---------------------------------------------------------------------------
# CMU Pronouncing Dictionary (converted to ClaritySynth phonemes, gzipped).
# 126k words. Loaded lazily (or in the background by the driver) so NVDA
# startup is never blocked; until it is ready, the letter-to-sound rules
# carry the load alone.
# ---------------------------------------------------------------------------
LEXICON = {}
_lex_lock = threading.Lock()
_lex_loaded = False


def ensure_lexicon():
    """Load the bundled pronouncing dictionary. Safe to call repeatedly."""
    global _lex_loaded
    if _lex_loaded:
        return
    with _lex_lock:
        if _lex_loaded:
            return
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "cmulex.txt.gz")
        try:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                for line in f:
                    sp = line.find(" ")
                    if sp > 0:
                        LEXICON[line[:sp]] = line[sp + 1:-1]
        except Exception:
            pass  # rules-only fallback
        _lex_loaded = True

VOWELS = "aeiouy"
VOICED_CONS = "bdvgjlmnrwz"
FRONT = "eiy"
SIB1 = "scgzxj"            # single-letter sibilant-ish set for '&'
AT1 = "tsrdlzn j".replace(" ", "")  # single letters for '@' (t s r d l z n j)
SUFFIXES = ("ing", "ely", "er", "es", "ed", "e")

# Phonemes considered voiced at word end (for plural/possessive s)
_VOICED_PH = {
    "IY", "IH", "EH", "AE", "AA", "AO", "UH", "UW", "AH", "AX", "ER",
    "EY", "AY", "OY", "AW", "OW", "M", "N", "NG", "L", "R", "W", "Y",
    "B", "D", "G", "V", "DH", "Z", "ZH", "JH",
}
_SIB_PH = {"S", "Z", "SH", "ZH", "CH", "JH"}

# ---------------------------------------------------------------------------
# Exceptions dictionary: words the rules get wrong or that deserve care
# ---------------------------------------------------------------------------
EXCEPTIONS = {
    "a": "AX", "an": "AE N", "the": "DH AX", "of": "AX V", "to": "T UW",
    "and": "AX N D", "was": "W AH Z", "is": "IH Z", "his": "HH IH Z",
    "has": "HH AE Z", "as": "AE Z", "said": "S EH D", "says": "S EH Z",
    "are": "AA R", "were": "W ER", "been": "B IH N", "have": "HH AE V",
    "does": "D AH Z", "done": "D AH N", "gone": "G AO N", "one": "W AH N",
    "once": "W AH N S", "two": "T UW", "who": "HH UW", "whom": "HH UW M",
    "whose": "HH UW Z", "do": "D UW", "you": "Y UW", "your": "Y AO R",
    "yours": "Y AO R Z", "women": "W IH M IH N", "woman": "W UH M AX N",
    "people": "P IY P AX L", "because": "B IH K AO Z", "would": "W UH D",
    "could": "K UH D", "should": "SH UH D", "some": "S AH M",
    "come": "K AH M", "comes": "K AH M Z", "other": "AH DH ER",
    "others": "AH DH ER Z", "mother": "M AH DH ER",
    "brother": "B R AH DH ER", "another": "AX N AH DH ER",
    "water": "W AO T ER", "again": "AX G EH N",
    "against": "AX G EH N S T", "eye": "AY", "eyes": "AY Z",
    "heart": "HH AA R T", "great": "G R EY T", "pretty": "P R IH T IY",
    "busy": "B IH Z IY", "business": "B IH Z N IH S",
    "minute": "M IH N IH T", "very": "V EH R IY", "many": "M EH N IY",
    "any": "EH N IY", "only": "OW N L IY", "our": "AW ER",
    "hour": "AW ER", "hours": "AW ER Z", "half": "HH AE F",
    "laugh": "L AE F", "enough": "IH N AH F", "through": "TH R UW",
    "thought": "TH AO T", "though": "DH OW", "tough": "T AH F",
    "rough": "R AH F", "cough": "K AO F", "iron": "AY ER N",
    "island": "AY L AX N D", "answer": "AE N S ER", "often": "AO F AX N",
    "listen": "L IH S AX N", "friend": "F R EH N D",
    "friends": "F R EH N D Z", "beautiful": "B Y UW T IH F UH L",
    "door": "D AO R", "floor": "F L AO R", "blood": "B L AH D",
    "flood": "F L AH D", "foot": "F UH T", "good": "G UH D",
    "book": "B UH K", "look": "L UH K", "took": "T UH K",
    "stood": "S T UH D", "wolf": "W UH L F", "push": "P UH SH",
    "pull": "P UH L", "full": "F UH L", "sugar": "SH UH G ER",
    "sure": "SH UH R", "machine": "M AX SH IY N", "ocean": "OW SH AX N",
    "question": "K W EH S CH AX N", "nature": "N EY CH ER",
    "picture": "P IH K CH ER", "want": "W AA N T", "watch": "W AA CH",
    "wash": "W AA SH", "heard": "HH ER D", "earth": "ER TH",
    "early": "ER L IY", "year": "Y IH R", "years": "Y IH R Z",
    "dear": "D IH R", "hear": "HH IY R", "live": "L IH V",
    "lives": "L IH V Z", "love": "L AH V", "loves": "L AH V Z",
    "move": "M UW V", "moved": "M UW V D", "lose": "L UW Z",
    "loose": "L UW S", "wednesday": "W EH N Z D EY",
    "february": "F EH B R UW EH R IY", "colonel": "K ER N AX L",
    "sword": "S AO R D", "castle": "K AE S AX L",
    "world": "W ER L D", "word": "W ER D", "words": "W ER D Z",
    "work": "W ER K", "works": "W ER K S",
    # contractions
    "don't": "D OW N T", "won't": "W OW N T", "can't": "K AE N T",
    "isn't": "IH Z AX N T", "wasn't": "W AH Z AX N T",
    "aren't": "AA R AX N T", "weren't": "W ER AX N T",
    "doesn't": "D AH Z AX N T", "didn't": "D IH D AX N T",
    "couldn't": "K UH D AX N T", "wouldn't": "W UH D AX N T",
    "shouldn't": "SH UH D AX N T", "ain't": "EY N T",
    "it's": "IH T S", "that's": "DH AE T S", "let's": "L EH T S",
    "i'm": "AY M", "i'll": "AY L", "i've": "AY V", "i'd": "AY D",
    "you're": "Y UH R", "you'll": "Y UW L", "you've": "Y UW V",
    "we're": "W IY R", "we'll": "W IY L", "we've": "W IY V",
    "they're": "DH EH R", "they'll": "DH EY L", "they've": "DH EY V",
    "he's": "HH IY Z", "she's": "SH IY Z", "there's": "DH EH R Z",
    "what's": "W AA T S", "here's": "HH IY R Z",
    # number words the rules mangle
    "zero": "Z IY R OW", "hundred": "HH AH N D R IH D",
    "thousand": "TH AW Z AX N D", "million": "M IH L Y AX N",
    "billion": "B IH L Y AX N", "trillion": "T R IH L Y AX N",
    "eleven": "IH L EH V AX N", "seven": "S EH V AX N",
    "seventy": "S EH V AX N T IY", "seventeen": "S EH V AX N T IY N",
    "twelve": "T W EH L V", "twenty": "T W EH N T IY",
    "thirty": "TH ER T IY", "forty": "F AO R T IY",
    "fifty": "F IH F T IY", "sixty": "S IH K S T IY",
    "eighty": "EY T IY", "ninety": "N AY N T IY", "eight": "EY T",
    "eighteen": "EY T IY N", "eighth": "EY T TH",
    "thirteen": "TH ER T IY N", "fifteen": "F IH F T IY N",
    "fourteen": "F AO R T IY N", "sixteen": "S IH K S T IY N",
    "nineteen": "N AY N T IY N", "second": "S EH K AX N D",
    "third": "TH ER D", "fourth": "F AO R TH", "fifth": "F IH F TH",
    "ninth": "N AY N TH", "twelfth": "T W EH L F TH",
    # tech / interface words a screen reader hits constantly
    "computer": "K AX M P Y UW T ER", "computers": "K AX M P Y UW T ER Z",
    "read": "R IY D", "reads": "R IY D Z", "reader": "R IY D ER",
    "readers": "R IY D ER Z", "reading": "R IY D IH NG",
    "window": "W IH N D OW", "knowledge": "N AA L IH JH",
    "education": "EH JH UW K EY SH AX N", "today": "T AX D EY",
    "tomorrow": "T AX M AA R OW", "tonight": "T AX N AY T",
    "together": "T AX G EH DH ER", "different": "D IH F R AX N T",
    "difference": "D IH F R AX N S", "immediately": "IH M IY D IY AX T L IY",
    "immediate": "IH M IY D IY AX T", "important": "IH M P AO R T AX N T",
    "hermione": "HH ER M AY1 AX N IY",
    "nvda": "EH N V IY D IY EY", "menu": "M EH N Y UW",
    "dialog": "D AY AX L AO G", "dialogue": "D AY AX L AO G",
    "button": "B AH T AX N", "icon": "AY K AA N",
    "email": "IY M EY L", "www": "D AH B AX L Y UW D AH B AX L Y UW D AH B AX L Y UW",
    "http": "EY CH T IY T IY P IY", "https": "EY CH T IY T IY P IY EH S",
    "com": "K AA M", "org": "AO R G", "gov": "G AH V",
    "claude": "K L AO D", "linux": "L IH N AX K S",
    "windows": "W IH N D OW Z", "python": "P AY TH AA N",
    "github": "G IH T HH AH B", "wifi": "W AY F AY",
    "ok": "OW K EY", "okay": "OW K EY", "etc": "EH T S EH T ER AX",
    "mr": "M IH S T ER", "mrs": "M IH S IH Z", "ms": "M IH Z",
    "dr": "D AA K T ER", "st": "S T R IY T", "vs": "V ER S AX S",
}

LETTER_NAMES = {
    "a": "EY", "b": "B IY", "c": "S IY", "d": "D IY", "e": "IY",
    "f": "EH F", "g": "JH IY", "h": "EY CH", "i": "AY", "j": "JH EY",
    "k": "K EY", "l": "EH L", "m": "EH M", "n": "EH N", "o": "OW",
    "p": "P IY", "q": "K Y UW", "r": "AA R", "s": "EH S", "t": "T IY",
    "u": "Y UW", "v": "V IY", "w": "D AH B AX L Y UW", "x": "EH K S",
    "y": "W AY", "z": "Z IY",
    "0": "Z IY R OW", "1": "W AH N", "2": "T UW", "3": "TH R IY",
    "4": "F AO R", "5": "F AY V", "6": "S IH K S", "7": "S EH V AX N",
    "8": "EY T", "9": "N AY N",
}

CHAR_NAMES = {
    ".": "D AA T", ",": "K AA M AX", "?": "K W EH S CH AX N M AA R K",
    "!": "B AE NG", "@": "AE T", "#": "HH AE SH", "$": "D AA L ER",
    "%": "P ER S EH N T", "&": "AE M P ER S AE N D", "*": "S T AA R",
    "-": "D AE SH", "_": "AH N D ER S K AO R", "+": "P L AH S",
    "=": "IY K W AX L Z", "/": "S L AE SH", "\\": "B AE K S L AE SH",
    "(": "L EH F T P ER EH N", ")": "R AY T P ER EH N",
    "'": "AX P AA S T R AX F IY", '"': "K W OW T",
    ":": "K OW L AX N", ";": "S EH M IY K OW L AX N",
    " ": "S P EY S",
}

SYMBOL_WORDS = {
    "%": " percent ", "&": " and ", "+": " plus ", "=": " equals ",
    "@": " at ", "$": " dollar ", "€": " euro ", "£": " pound ",
    "~": " tilde ", "°": " degrees ",
}

# ---------------------------------------------------------------------------
# NRL-style letter-to-sound rules.
# Format: (left context, match, right context, phonemes)
# Context symbols:
#   '#' one or more vowels        ':' zero or more consonants
#   '^' one consonant             '.' one voiced consonant
#   '+' front vowel (e i y)       '&' sibilant (s c g z x j, ch, sh)
#   '@' t s r d l z n j th ch sh  '%' suffix (e er es ed ing ely)
#   ' ' word boundary             letters match literally
# First matching rule wins; rules are tried in order.
# ---------------------------------------------------------------------------
_R = {
    "a": [
        (" ", "a", " ", "AX"),
        (" ", "are", " ", "AA R"),
        (" ", "ar", "o", "AX R"),
        ("", "ar", "#", "EH R"),
        ("^", "as", "#", "EY S"),
        ("", "a", "wa", "AX"),
        ("", "aw", "", "AO"),
        (" :", "any", "", "EH N IY"),
        ("", "a", "^+#", "EY"),
        ("#:", "ally", "", "AX L IY"),
        (" ", "al", "#", "AX L"),
        ("", "again", "", "AX G EH N"),
        ("#:", "ag", "e", "IH JH"),
        ("", "a", "^+:#", "AE"),
        (" :", "a", "^+ ", "EY"),
        ("", "a", "^%", "EY"),
        (" ", "arr", "", "AX R"),
        ("", "arr", "", "AE R"),
        (" :", "ar", " ", "AA R"),
        ("", "ar", " ", "ER"),
        ("", "ar", "", "AA R"),
        ("", "air", "", "EH R"),
        ("", "ai", "", "EY"),
        ("", "ay", "", "EY"),
        ("", "au", "", "AO"),
        ("#:", "al", " ", "AX L"),
        ("#:", "als", " ", "AX L Z"),
        ("", "alk", "", "AO K"),
        ("", "al", "^", "AO L"),
        (" :", "able", "", "EY B AX L"),
        ("", "able", "", "AX B AX L"),
        ("", "ang", "+", "EY N JH"),
        ("", "a", "", "AE"),
    ],
    "b": [
        (" ", "be", "^#", "B IH"),
        ("", "being", "", "B IY IH NG"),
        (" ", "both", " ", "B OW TH"),
        (" ", "bus", "#", "B IH Z"),
        ("", "buil", "", "B IH L"),
        ("", "b", "", "B"),
    ],
    "c": [
        (" ", "ch", "^", "K"),
        ("^e", "ch", "", "K"),
        ("", "ch", "", "CH"),
        (" s", "ci", "#", "S AY"),
        ("", "ci", "a", "SH"),
        ("", "ci", "o", "SH"),
        ("", "ci", "en", "SH"),
        ("", "c", "+", "S"),
        ("", "ck", "", "K"),
        ("", "com", "%", "K AH M"),
        ("", "c", "", "K"),
    ],
    "d": [
        ("#:", "ded", " ", "D IH D"),
        (".e", "d", " ", "D"),
        ("#:^e", "d", " ", "T"),
        (" ", "de", "^#", "D IH"),
        (" ", "do", " ", "D UW"),
        (" ", "does", "", "D AH Z"),
        (" ", "doing", "", "D UW IH NG"),
        (" ", "dow", "", "D AW"),
        ("", "du", "a", "JH UW"),
        ("", "d", "", "D"),
    ],
    "e": [
        ("#:", "e", " ", ""),
        ("'^", "e", " ", ""),
        (" :", "e", " ", "IY"),
        ("#", "ed", " ", "D"),
        ("#:", "e", "d ", ""),
        ("", "ev", "er", "EH V"),
        ("", "e", "^%", "IY"),
        ("", "eri", "#", "IY R IY"),
        ("", "eri", "", "EH R IH"),
        ("#:", "er", "#", "ER"),
        ("", "er", "#", "EH R"),
        ("", "er", "", "ER"),
        (" ", "even", "", "IY V EH N"),
        ("#:", "e", "w", ""),
        ("@", "ew", "", "UW"),
        ("", "ew", "", "Y UW"),
        ("", "e", "o", "IY"),
        ("#:&", "es", " ", "IH Z"),
        ("#:", "e", "s ", ""),
        ("#:", "ely", " ", "L IY"),
        ("#:", "ement", "", "M EH N T"),
        ("", "eful", "", "F UH L"),
        ("", "ee", "", "IY"),
        ("", "earn", "", "ER N"),
        (" ", "ear", "^", "ER"),
        ("", "ead", "", "EH D"),
        ("#:", "ea", " ", "IY AX"),
        ("", "ea", "su", "EH"),
        ("", "ea", "", "IY"),
        ("", "eigh", "", "EY"),
        ("", "ei", "", "IY"),
        (" ", "eye", "", "AY"),
        ("", "ey", "", "IY"),
        ("", "eu", "", "Y UW"),
        ("", "e", "", "EH"),
    ],
    "f": [
        ("", "ful", "", "F UH L"),
        ("", "f", "", "F"),
    ],
    "g": [
        ("", "gh", "", ""),
        (" ", "gn", "", "N"),
        ("", "giv", "", "G IH V"),
        (" ", "g", "i^", "G"),
        ("", "ge", "t", "G EH"),
        ("su", "gges", "", "G JH EH S"),
        ("", "gg", "", "G"),
        (" b#", "g", "", "G"),
        ("", "g", "+", "JH"),
        ("", "great", "", "G R EY T"),
        ("#", "gh", "", ""),
        ("", "g", "", "G"),
    ],
    "h": [
        (" ", "hav", "", "HH AE V"),
        (" ", "here", "", "HH IY R"),
        (" ", "hour", "", "AW ER"),
        ("", "how", "", "HH AW"),
        ("", "h", "#", "HH"),
        ("", "h", "", ""),
    ],
    "i": [
        (" ", "in", "", "IH N"),
        (" ", "i", " ", "AY"),
        ("", "in", "d", "AY N"),
        ("", "ier", "", "IY ER"),
        ("#:r", "ied", "", "IY D"),
        ("", "ied", " ", "AY D"),
        ("", "ien", "", "IY EH N"),
        ("", "ie", "t", "AY EH"),
        (" :", "i", "%", "AY"),
        ("", "i", "%", "IY"),
        ("", "ie", "", "IY"),
        ("", "i", "^+:#", "IH"),
        ("", "ir", "#", "AY R"),
        ("", "iz", "%", "AY Z"),
        ("", "is", "%", "AY Z"),
        ("", "i", "d%", "AY"),
        ("+^", "i", "^+", "IH"),
        ("", "i", "t%", "AY"),
        ("#:^", "i", "^+", "IH"),
        ("", "i", "^+", "AY"),
        ("", "ir", "", "ER"),
        ("", "igh", "", "AY"),
        ("", "ild", "", "AY L D"),
        ("", "ign", " ", "AY N"),
        ("", "ign", "^", "AY N"),
        ("", "ign", "%", "AY N"),
        ("", "ique", "", "IY K"),
        ("", "i", "", "IH"),
    ],
    "j": [
        ("", "j", "", "JH"),
    ],
    "k": [
        (" ", "k", "n", ""),
        ("", "k", "", "K"),
    ],
    "l": [
        ("", "lo", "c#", "L OW"),
        ("l", "l", "", ""),
        ("#:^", "l", "%", "AX L"),
        ("", "lead", "", "L IY D"),
        ("", "l", "", "L"),
    ],
    "m": [
        ("", "mov", "", "M UW V"),
        ("", "m", "", "M"),
    ],
    "n": [
        ("e", "ng", "+", "N JH"),
        ("", "ng", "r", "NG G"),
        ("", "ng", "#", "NG G"),
        ("", "ngl", "%", "NG G AX L"),
        ("", "ng", "", "NG"),
        ("", "nk", "", "NG K"),
        (" ", "now", " ", "N AW"),
        ("", "n", "", "N"),
    ],
    "o": [
        (" ", "of", " ", "AX V"),
        ("", "orough", "", "ER OW"),
        ("#:", "or", " ", "ER"),
        ("#:", "ors", " ", "ER Z"),
        ("", "or", "", "AO R"),
        (" ", "one", "", "W AH N"),
        ("", "ow", "", "OW"),
        (" ", "over", "", "OW V ER"),
        ("", "ov", "", "AH V"),
        ("", "o", "^%", "OW"),
        ("", "o", "^en", "OW"),
        ("", "o", "^i#", "OW"),
        ("", "ol", "d", "OW L"),
        ("", "ought", "", "AO T"),
        ("", "ough", "", "AH F"),
        (" ", "ou", "", "AW"),
        ("h", "ou", "s#", "AW"),
        ("", "ous", "", "AX S"),
        ("", "our", "", "AO R"),
        ("", "ould", "", "UH D"),
        ("^", "ou", "^l", "AH"),
        ("", "oup", "", "UW P"),
        ("", "ou", "", "AW"),
        ("", "oy", "", "OY"),
        ("", "oing", "", "OW IH NG"),
        (" :", "o", " ", "OW"),
        ("", "o", "e", "OW"),
        ("", "o", " ", "OW"),
        ("", "oar", "", "AO R"),
        ("", "oa", "", "OW"),
        (" ", "only", "", "OW N L IY"),
        (" ", "once", "", "W AH N S"),
        ("", "on't", "", "OW N T"),
        ("c", "o", "n", "AA"),
        ("", "o", "ng", "AO"),
        (" :^", "o", "n", "AH"),
        ("i", "on", "", "AX N"),
        ("#:", "on", " ", "AX N"),
        ("#^", "on", "", "AX N"),
        ("", "o", "st ", "OW"),
        ("", "of", "^", "AO F"),
        ("", "other", "", "AH DH ER"),
        ("", "oss", " ", "AO S"),
        ("#:^", "om", "", "AH M"),
        ("", "o", "", "AA"),
    ],
    "p": [
        ("", "ph", "", "F"),
        ("", "peop", "", "P IY P"),
        ("", "pow", "", "P AW"),
        ("", "put", " ", "P UH T"),
        ("", "p", "", "P"),
    ],
    "q": [
        ("", "quar", "", "K W AO R"),
        ("", "qu", "", "K W"),
        ("", "q", "", "K"),
    ],
    "r": [
        (" ", "re", "^#", "R IY"),
        ("", "r", "", "R"),
    ],
    "s": [
        ("", "sh", "", "SH"),
        ("#", "sion", "", "ZH AX N"),
        ("", "some", "", "S AH M"),
        ("#", "sur", "#", "ZH ER"),
        ("", "sur", "#", "SH ER"),
        ("#", "su", "#", "ZH UW"),
        ("#", "ssu", "#", "SH UW"),
        ("#", "sed", " ", "Z D"),
        ("#", "s", "#", "Z"),
        ("", "said", "", "S EH D"),
        ("^", "sion", "", "SH AX N"),
        ("", "s", "s", ""),
        (".", "s", " ", "Z"),
        ("#:.e", "s", " ", "Z"),
        ("#:^#", "s", " ", "S"),
        ("u", "s", " ", "S"),
        (" :#", "s", " ", "Z"),
        (" ", "sch", "", "S K"),
        ("", "s", "c+", ""),
        ("#", "sm", "", "Z M"),
        ("#", "sn", "'", "Z AX N"),
        ("", "s", "", "S"),
    ],
    "t": [
        (" ", "the", " ", "DH AX"),
        ("", "to", " ", "T UW"),
        ("", "that", " ", "DH AE T"),
        (" ", "this", " ", "DH IH S"),
        (" ", "they", "", "DH EY"),
        (" ", "there", "", "DH EH R"),
        ("", "ther", "", "DH ER"),
        ("", "their", "", "DH EH R"),
        (" ", "than", " ", "DH AE N"),
        (" ", "them", " ", "DH EH M"),
        ("", "these", " ", "DH IY Z"),
        (" ", "then", "", "DH EH N"),
        ("", "through", "", "TH R UW"),
        ("", "those", "", "DH OW Z"),
        ("", "though", " ", "DH OW"),
        (" ", "thus", "", "DH AH S"),
        ("", "th", "", "TH"),
        ("#:", "ted", " ", "T IH D"),
        ("s", "ti", "#n", "CH"),
        ("", "ti", "o", "SH"),
        ("", "ti", "a", "SH"),
        ("", "tien", "", "SH AX N"),
        ("", "tur", "#", "CH ER"),
        ("", "tu", "a", "CH UW"),
        (" ", "two", "", "T UW"),
        ("", "t", "", "T"),
    ],
    "u": [
        (" ", "u", " ", "Y UW"),
        (" ", "un", "i", "Y UW N"),
        (" ", "un", "", "AH N"),
        (" ", "upon", "", "AX P AO N"),
        ("@", "ur", "#", "UH R"),
        ("", "ur", "#", "Y UH R"),
        ("", "ur", "", "ER"),
        ("", "u", "^ ", "AH"),
        ("", "u", "^^", "AH"),
        ("", "uy", "", "AY"),
        (" g", "u", "#", ""),
        ("g", "u", "%", ""),
        ("g", "u", "#", "W"),
        ("#n", "u", "", "Y UW"),
        ("@", "u", "", "UW"),
        ("", "u", "", "Y UW"),
    ],
    "v": [
        ("", "view", "", "V Y UW"),
        ("", "v", "", "V"),
    ],
    "w": [
        (" ", "were", "", "W ER"),
        ("", "wa", "s", "W AA"),
        ("", "wa", "t", "W AA"),
        ("", "where", "", "W EH R"),
        ("", "what", "", "W AA T"),
        ("", "whol", "", "HH OW L"),
        ("", "who", "", "HH UW"),
        ("", "wh", "", "W"),
        ("", "war", "", "W AO R"),
        ("", "wor", "^", "W ER"),
        ("", "wr", "", "R"),
        ("", "w", "", "W"),
    ],
    "x": [
        (" ", "x", "", "Z"),
        ("", "x", "", "K S"),
    ],
    "y": [
        ("", "young", "", "Y AH NG"),
        (" ", "you", "", "Y UW"),
        (" ", "yes", "", "Y EH S"),
        (" ", "y", "", "Y"),
        ("#:^", "y", " ", "IY"),
        ("#:^", "y", "i", "IY"),
        (" :", "y", " ", "AY"),
        (" :", "y", "#", "AY"),
        (" :", "y", "^+:#", "IH"),
        (" :", "y", "^#", "AY"),
        ("", "y", "", "IH"),
    ],
    "z": [
        ("", "z", "", "Z"),
    ],
}


def _is_vowel(c):
    return c in VOWELS


def _is_cons(c):
    return c.isalpha() and c not in VOWELS


def _match_left(word, i, pattern):
    """Match `pattern` against word ending at index i (inclusive), scanning
    the pattern right-to-left. Returns True on success."""
    for sym in reversed(pattern):
        if sym == " ":
            if i >= 0 and word[i].isalpha():
                return False
            i -= 1
        elif sym == "#":
            if i < 0 or not _is_vowel(word[i]):
                return False
            while i >= 0 and _is_vowel(word[i]):
                i -= 1
        elif sym == ":":
            while i >= 0 and _is_cons(word[i]):
                i -= 1
        elif sym == "^":
            if i < 0 or not _is_cons(word[i]):
                return False
            i -= 1
        elif sym == ".":
            if i < 0 or word[i] not in VOICED_CONS:
                return False
            i -= 1
        elif sym == "+":
            if i < 0 or word[i] not in FRONT:
                return False
            i -= 1
        elif sym == "&":
            if i < 0:
                return False
            if word[i] in SIB1:
                i -= 1
            elif i >= 1 and word[i - 1:i + 1] in ("ch", "sh"):
                i -= 2
            else:
                return False
        elif sym == "@":
            if i < 0:
                return False
            if i >= 1 and word[i - 1:i + 1] in ("th", "ch", "sh"):
                i -= 2
            elif word[i] in AT1:
                i -= 1
            else:
                return False
        else:  # literal
            if i < 0 or word[i] != sym:
                return False
            i -= 1
    return True


def _match_right(word, i, pattern):
    """Match `pattern` against word starting at index i, left-to-right."""
    n = len(word)
    for sym in pattern:
        if sym == " ":
            if i < n and word[i].isalpha():
                return False
            i += 1
        elif sym == "#":
            if i >= n or not _is_vowel(word[i]):
                return False
            while i < n and _is_vowel(word[i]):
                i += 1
        elif sym == ":":
            while i < n and _is_cons(word[i]):
                i += 1
        elif sym == "^":
            if i >= n or not _is_cons(word[i]):
                return False
            i += 1
        elif sym == ".":
            if i >= n or word[i] not in VOICED_CONS:
                return False
            i += 1
        elif sym == "+":
            if i >= n or word[i] not in FRONT:
                return False
            i += 1
        elif sym == "&":
            if i >= n:
                return False
            if word[i:i + 2] in ("ch", "sh"):
                i += 2
            elif word[i] in SIB1:
                i += 1
            else:
                return False
        elif sym == "@":
            if i >= n:
                return False
            if word[i:i + 2] in ("th", "ch", "sh"):
                i += 2
            elif word[i] in AT1:
                i += 1
            else:
                return False
        elif sym == "%":
            ok = False
            for suf in SUFFIXES:
                if word[i:i + len(suf)] == suf:
                    ok = True
                    i += len(suf)
                    break
            if not ok:
                return False
        else:
            if i >= n or word[i] != sym:
                return False
            i += 1
    return True


def _rules_word(word):
    """Apply the letter-to-sound rules to a lowercase word."""
    w = " " + word + " "
    out = []
    pos = 1
    end = len(w) - 1
    while pos < end:
        ch = w[pos]
        rules = _R.get(ch)
        if rules is None:
            pos += 1
            continue
        for left, match, right, phones in rules:
            m = len(match)
            if w[pos:pos + m] != match:
                continue
            if not _match_left(w, pos - 1, left):
                continue
            if not _match_right(w, pos + m, right):
                continue
            if phones:
                out.extend(phones.split())
            pos += m
            break
        else:
            pos += 1
    return out


def _spell(word):
    out = []
    for ch in word.lower():
        name = LETTER_NAMES.get(ch) or CHAR_NAMES.get(ch)
        if name:
            out.extend(name.split())
            out.append("_br")
    return out


def _plural_suffix(base_phones):
    if not base_phones:
        return ["Z"]
    last = base_phones[-1]
    if last and last[-1] in "012":
        last = last[:-1]
    if last in _SIB_PH:
        return ["IH", "Z"]
    if last in _VOICED_PH:
        return ["Z"]
    return ["S"]


def _strip_digits(phones):
    return [p[:-1] if p and p[-1] in "012" else p for p in phones]


def _morph(lw):
    """Try to pronounce an out-of-dictionary word via its stem + suffix."""
    ensure_lexicon()
    L = LEXICON
    if lw.endswith("s") and not lw.endswith("ss") and len(lw) > 3:
        stem = lw[:-2] if lw.endswith("es") and lw[:-2] in L else lw[:-1]
        if stem in L:
            base = L[stem].split()
            return base + _plural_suffix(_strip_digits(base))
    if lw.endswith("ed") and len(lw) > 4:
        stems = [lw[:-2], lw[:-1]]
        if len(lw) > 5 and lw[-3] == lw[-4]:
            stems.append(lw[:-3])
        for stem in stems:
            if stem in L:
                base = L[stem].split()
                last = _strip_digits(base)[-1]
                if last in ("T", "D"):
                    return base + ["IH", "D"]
                return base + (["D"] if last in _VOICED_PH else ["T"])
    if lw.endswith("ing") and len(lw) > 5:
        stems = [lw[:-3], lw[:-3] + "e"]
        if len(lw) > 6 and lw[-4] == lw[-5]:
            stems.append(lw[:-4])
        for stem in stems:
            if stem in L:
                return L[stem].split() + ["IH", "NG"]
    if lw.endswith("ly") and len(lw) > 4 and lw[:-2] in L:
        return L[lw[:-2]].split() + ["L", "IY"]
    if lw.endswith("er") and len(lw) > 4:
        for stem in (lw[:-2], lw[:-1], lw[:-3]):
            if stem in L:
                return L[stem].split() + ["ER"]
    if lw.endswith("est") and len(lw) > 5:
        for stem in (lw[:-3], lw[:-2]):
            if stem in L:
                return L[stem].split() + ["AX", "S", "T"]
    if lw.endswith("ness") and len(lw) > 6 and lw[:-4] in L:
        return L[lw[:-4]].split() + ["N", "IH", "S"]
    # productive prefixes: un-, re-, pre-, dis-, mis-, non-, over-, out-
    for pre, ph in (("un", ["AH", "N"]), ("non", ["N", "AA", "N"]),
                    ("re", ["R", "IY"]), ("pre", ["P", "R", "IY"]),
                    ("dis", ["D", "IH", "S"]), ("mis", ["M", "IH", "S"]),
                    ("over", ["OW", "V", "ER"]), ("out", ["AW", "T"])):
        if lw.startswith(pre) and len(lw) - len(pre) >= 3:
            stem = lw[len(pre):]
            if stem in L:
                return ph + L[stem].split()
            m = _morph(stem)
            if m:
                return ph + m
    return None


def word_to_phonemes(word):
    """Phonemize one word token (may contain apostrophes)."""
    lw = word.lower()
    if lw in EXCEPTIONS:
        return EXCEPTIONS[lw].split()
    ensure_lexicon()
    # short all-caps tokens read as acronyms (4-letter ones only when they
    # are not ordinary dictionary words)
    if word.isupper() and (2 <= len(word) <= 3
                           or (len(word) == 4 and lw not in LEXICON)):
        return _spell(word)
    if lw in LEXICON:
        return LEXICON[lw].split()
    # contraction suffixes
    for suf, tail in (("'s", None), ("'ll", ["L"]), ("'re", ["ER"]),
                      ("'ve", ["AX", "V"]), ("'d", ["D"]),
                      ("n't", ["AX", "N", "T"])):
        if lw.endswith(suf) and len(lw) > len(suf):
            base = word_to_phonemes(lw[:-len(suf)])
            if suf == "'s":
                return base + _plural_suffix(base)
            return base + tail
    lw = lw.replace("'", "")
    if not lw:
        return []
    if lw in LEXICON:
        return LEXICON[lw].split()
    m = _morph(lw)
    if m:
        return m
    phones = _rules_word(lw)
    if not phones:
        return _spell(lw)
    out = [phones[0]]
    for p in phones[1:]:
        if p != out[-1]:
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# Number expansion
# ---------------------------------------------------------------------------
_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven",
         "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
         "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]
_SCALE = [(10 ** 12, "trillion"), (10 ** 9, "billion"),
          (10 ** 6, "million"), (10 ** 3, "thousand"), (100, "hundred")]


def _int_to_words(n):
    if n < 20:
        return _ONES[n]
    if n < 100:
        t, r = divmod(n, 10)
        return _TENS[t] + ((" " + _ONES[r]) if r else "")
    for val, name in _SCALE:
        if n >= val:
            head, rest = divmod(n, val)
            s = _int_to_words(head) + " " + name
            if rest:
                s += " " + _int_to_words(rest)
            return s
    return _ONES[0]


def number_to_words(tok):
    try:
        if "." in tok:
            whole, frac = tok.split(".", 1)
            words = _int_to_words(int(whole)) if whole else "zero"
            words += " point"
            for d in frac:
                if d.isdigit():
                    words += " " + _ONES[int(d)]
            return words
        n = int(tok)
        if n > 10 ** 15:
            return " ".join(_ONES[int(d)] for d in tok)
        return _int_to_words(n)
    except (ValueError, OverflowError):
        return " ".join(_ONES[int(d)] for d in tok if d.isdigit())


# ---------------------------------------------------------------------------
# Top level: text -> token stream
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*|\d+(?:\.\d+)?|.", re.S)
_PUNCT_TOKENS = {
    ".": "_.", "!": "_!", "?": "_?", ",": "_,", ";": "_;", ":": "_:",
    "(": "_(", ")": "_)", "—": "_-", "–": "_-",
    "\n": "_.", "…": "_.",
}


def _is_ar(ch):
    return ("\u0600" <= ch <= "\u06FF") or ("\u0750" <= ch <= "\u077F")


def _split_runs(text):
    """Split text into (script, chunk) runs: 'en' or 'ar'."""
    runs = []
    cur = []
    cur_s = None
    pending = []
    for ch in text:
        if _is_ar(ch):
            s = "ar"
        elif ord(ch) < 128 and ch.isalpha():
            s = "en"
        else:
            s = None   # digits, spaces, punctuation follow the current run
        if s is None:
            (cur if cur_s else pending).append(ch)
        elif s == cur_s:
            if pending:
                cur.extend(pending)
                pending = []
            cur.append(ch)
        else:
            if cur_s:
                runs.append((cur_s, "".join(cur)))
            cur = pending + [ch]
            pending = []
            cur_s = s
    if cur_s:
        runs.append((cur_s, "".join(cur + pending)))
    return runs or [("en", text)]


def text_to_tokens(text):
    """Convert text into phoneme/control tokens, routing by script."""
    tokens = []
    for script, chunk in _split_runs(text):
        if script == "ar":
            try:
                from . import ar_g2p
            except ImportError:
                import ar_g2p
            tokens.extend(ar_g2p.text_to_tokens(chunk))
        else:
            tokens.extend(_en_text_to_tokens(chunk))
    return tokens


def _en_text_to_tokens(text):
    """Convert an English text string into a flat token list."""
    tokens = []
    for sym, repl in SYMBOL_WORDS.items():
        if sym in text:
            text = text.replace(sym, repl)
    last_was_pause = True
    for m in _TOKEN_RE.finditer(text):
        tok = m.group(0)
        if tok.isspace():
            continue
        first = tok[0]
        if first.isalpha():
            phones = word_to_phonemes(tok)
            if phones:
                tokens.append("_w")
                tokens.extend(phones)
                last_was_pause = False
        elif first.isdigit():
            for w in number_to_words(tok).split():
                tokens.append("_w")
                tokens.extend(word_to_phonemes(w))
            last_was_pause = False
        else:
            p = _PUNCT_TOKENS.get(tok)
            if p and not last_was_pause:
                tokens.append(p)
                last_was_pause = True
    return tokens


def char_to_tokens(ch):
    """Tokens for speaking a single character by name (character mode)."""
    if _is_ar(ch):
        try:
            from . import ar_g2p
        except ImportError:
            import ar_g2p
        return ar_g2p.char_to_tokens(ch)
    lw = ch.lower()
    name = LETTER_NAMES.get(lw) or CHAR_NAMES.get(lw)
    if name:
        return ["_w"] + name.split()
    if ch.isalpha():
        return ["_w"] + word_to_phonemes(ch)
    return []
