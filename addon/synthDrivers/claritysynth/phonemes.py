# -*- coding: utf-8 -*-
# ClaritySynth: phoneme inventory and acoustic targets.
# A Klatt-style cascade formant model (Klatt 1980) drives synthesis, so every
# phoneme is described by formant targets (F1, F2, F3 in Hz), a typical
# duration in milliseconds, and source information (voicing, frication).
# Values are classic adult-male averages (Peterson & Barney 1952 and friends),
# hand-tuned by ear where needed.

def _vowel(f1, f2, f3, dur):
    return {"kind": "vowel", "f": (float(f1), float(f2), float(f3)), "dur": dur}


def _diph(start, end, dur):
    return {
        "kind": "diph",
        "f": tuple(float(x) for x in start),
        "f_end": tuple(float(x) for x in end),
        "dur": dur,
    }


def _son(kind, f1, f2, f3, dur, amp=0.8):
    return {
        "kind": kind,
        "f": (float(f1), float(f2), float(f3)),
        "dur": dur,
        "amp": amp,
    }


def _fric(ff, fbw, amp, dur, voiced, locus):
    return {
        "kind": "fric",
        "ff": float(ff),
        "fbw": float(fbw),
        "amp": amp,
        "dur": dur,
        "voiced": voiced,
        "f": tuple(float(x) for x in locus),
    }


def _stop(burstf, burstbw, locus, voiced, dur):
    return {
        "kind": "stop",
        "burstf": float(burstf),
        "burstbw": float(burstbw),
        "f": tuple(float(x) for x in locus),
        "voiced": voiced,
        "dur": dur,
    }


PHONEMES = {
    # Monophthong vowels
    "IY": _vowel(270, 2290, 3010, 125),
    "IH": _vowel(400, 1990, 2550, 90),
    "EH": _vowel(530, 1840, 2480, 100),
    "AE": _vowel(660, 1720, 2410, 135),
    "AA": _vowel(730, 1090, 2440, 135),
    "AO": _vowel(570, 840, 2410, 135),
    "UH": _vowel(440, 1020, 2240, 90),
    "UW": _vowel(300, 870, 2240, 125),
    "AH": _vowel(640, 1190, 2390, 100),
    "AX": _vowel(500, 1400, 2400, 60),   # schwa
    "ER": _vowel(490, 1350, 1690, 115),
    # Diphthongs (start target glides to end target)
    "EY": _diph((480, 1900, 2500), (330, 2200, 2800), 150),
    "AY": _diph((700, 1200, 2500), (400, 2000, 2600), 175),
    "OY": _diph((560, 880, 2400), (400, 1900, 2600), 185),
    "AW": _diph((700, 1200, 2500), (430, 940, 2300), 175),
    "OW": _diph((540, 900, 2400), (360, 870, 2250), 150),
    # Nasals (reduced amplitude, heavy low F1)
    "M": _son("nasal", 280, 900, 2200, 70, amp=0.62),
    "N": _son("nasal", 280, 1700, 2600, 80, amp=0.62),
    "NG": _son("nasal", 280, 2300, 2750, 90, amp=0.5),
    # Liquids and glides
    "L": _son("liquid", 360, 1300, 2700, 65, amp=0.75),
    "R": _son("liquid", 310, 1060, 1380, 70, amp=0.75),
    "W": _son("glide", 290, 610, 2150, 60, amp=0.75),
    "Y": _son("glide", 260, 2070, 3020, 60, amp=0.75),
    # /h/: aspiration noise shaped by the following vowel's formants
    "HH": {"kind": "asp", "dur": 70},
    # Fricatives: (noise centre freq, bandwidth, amplitude, duration, voicing,
    # formant locus used for transitions in/out)
    "S": _fric(5100, 900, 0.28, 95, False, (320, 1600, 2600)),
    "Z": _fric(5100, 900, 0.24, 90, True, (320, 1600, 2600)),
    "SH": _fric(2500, 900, 0.38, 110, False, (300, 1900, 2500)),
    "ZH": _fric(2500, 900, 0.38, 90, True, (300, 1900, 2500)),
    "F": _fric(4300, 3200, 0.12, 95, False, (320, 1100, 2500)),
    "V": _fric(4300, 3200, 0.09, 60, True, (320, 1100, 2500)),
    "TH": _fric(5300, 2600, 0.08, 90, False, (320, 1400, 2600)),
    "DH": _fric(5300, 2600, 0.13, 50, True, (320, 1400, 2600)),
    # Stops: burst locus + formant locus for CV transitions
    "P": _stop(900, 1800, (250, 800, 2200), False, 95),
    "B": _stop(900, 1800, (250, 800, 2200), True, 80),
    "T": _stop(4500, 2200, (250, 1800, 2700), False, 95),
    "D": _stop(4500, 2200, (250, 1800, 2700), True, 80),
    "K": _stop(2200, 1400, (280, 1900, 2300), False, 100),
    "G": _stop(2200, 1400, (280, 1900, 2300), True, 85),
    # Affricates: rendered as stop closure + fricative release
    "CH": {"kind": "affric", "voiced": False, "dur": 130,
           "fric": "SH", "stop": "T"},
    "JH": {"kind": "affric", "voiced": True, "dur": 110,
           "fric": "ZH", "stop": "D"},
}

# Pause lengths (milliseconds) for punctuation tokens produced by the G2P.
PAUSES = {
    "_,": 160,
    "_.": 340,
    "_?": 340,
    "_!": 340,
    "_;": 220,
    "_:": 220,
    "_(": 180,
    "_)": 180,
    "_-": 120,
    "_br": 60,   # small structural break
}

# ---------------------------------------------------------------------------
# Arabic phonemes (used by ar_g2p). English set is reused where the sounds
# coincide (B T TH JH D DH Z S SH F K L M N HH W Y ...).
# ---------------------------------------------------------------------------
PHONEMES.update({
    # uvular stop qaf
    "Q": _stop(1950, 600, (350, 1100, 2300), False, 100),
    # glottal stop (hamza): closure only, no burst or aspiration
    "GS": {"kind": "stop", "burstf": 500.0, "burstbw": 3000.0,
           "f": (400.0, 1400.0, 2400.0), "voiced": False, "dur": 60,
           "glottal": True},
    # voiced pharyngeal approximant (ayn): high F1, constricted pharynx
    "AYN": _son("liquid", 700, 1100, 2500, 95, amp=0.50),
    # voiceless pharyngeal fricative (haa): turbulence shaped by the
    # whole vocal tract with a pharyngeal posture, like a deep /h/
    "HH2": _fric(900, 500, 0.95, 125, False, (600, 1050, 2450)),
    # uvular fricatives (khaa / ghayn)
    "KH": _fric(1900, 450, 0.40, 105, False, (400, 1300, 2300)),
    "GH": _fric(1550, 700, 0.24, 80, True, (400, 1300, 2300)),
    # emphatic (pharyngealized) consonants: lowered noise centre / F2 locus
    "SS": _fric(3900, 700, 0.44, 115, False, (450, 1050, 2350)),
    "ZZ": _fric(4400, 2400, 0.14, 90, True, (450, 1050, 2400)),
    "TT": _stop(2900, 1600, (350, 1050, 2350), False, 95),
    "DD": _stop(2900, 1600, (350, 1050, 2350), True, 85),
    # alveolar tap for Arabic raa (also usable as a flap)
    "DX": {"kind": "stop", "burstf": 4200.0, "burstbw": 2200.0,
           "f": (400.0, 1500.0, 2600.0), "voiced": True, "dur": 34,
           "tap": True},
    # heavy (mufakhkham) raa: pharyngealized posture, low F2 dip
    "DXQ": {"kind": "stop", "burstf": 4200.0, "burstbw": 2200.0,
            "f": (450.0, 1150.0, 2350.0), "voiced": True, "dur": 34,
            "tap": True},
    # Arabic clear (light) lam: fronter, brighter than English dark /l/
    "LT": _son("liquid", 340, 1650, 2900, 60, amp=0.78),
    # Dark, emphatic lam of the word Allah
    "LD": _son("liquid", 340, 950, 2500, 70, amp=0.75),
    # Arabic fatha: a true central open [a], not English ash [ae]
    "AHA": _vowel(690, 1450, 2500, 85),
})

# tgSpeechBox / NV Speech Player parameter adoption (GPL2)
PHONEMES["IY"]["f"] = (310.0, 2020.0, 2960.0)
PHONEMES["IY"]["bw"] = (49.5, 150.0, 300.0)
PHONEMES["IH"]["f"] = (360.0, 1900.0, 2570.0)
PHONEMES["IH"]["bw"] = (55.0, 75.0, 105.0)
PHONEMES["EH"]["f"] = (530.0, 1750.0, 2500.0)
PHONEMES["EH"]["bw"] = (90.0, 110.0, 170.0)
PHONEMES["AE"]["f"] = (620.0, 1780.0, 2430.0)
PHONEMES["AE"]["bw"] = (77.0, 112.5, 240.0)
PHONEMES["AA"]["f"] = (700.0, 1330.0, 2600.0)
PHONEMES["AA"]["bw"] = (170.0, 120.0, 150.0)
PHONEMES["AO"]["f"] = (420.0, 840.0, 2400.0)
PHONEMES["AO"]["bw"] = (99.0, 90.0, 180.0)
PHONEMES["UH"]["f"] = (405.0, 900.0, 2420.0)
PHONEMES["UH"]["bw"] = (88.0, 75.0, 60.0)
PHONEMES["UW"]["f"] = (290.0, 950.0, 2280.0)
PHONEMES["UW"]["bw"] = (71.5, 82.5, 105.0)
PHONEMES["AH"]["f"] = (580.0, 1200.0, 2550.0)
PHONEMES["AH"]["bw"] = (95.0, 110.0, 170.0)
PHONEMES["AX"]["f"] = (500.0, 1400.0, 2300.0)
PHONEMES["AX"]["bw"] = (130.0, 100.0, 160.0)
PHONEMES["ER"]["f"] = (500.0, 1400.0, 2300.0)
PHONEMES["ER"]["bw"] = (110.0, 45.0, 82.5)
PHONEMES["L"]["f"] = (310.0, 1050.0, 2880.0)
PHONEMES["L"]["bw"] = (55.0, 75.0, 210.0)
PHONEMES["R"]["bw"] = (66.0, 75.0, 127.5)
PHONEMES["W"]["f"] = (290.0, 610.0, 2150.0)
PHONEMES["W"]["bw"] = (55.0, 60.0, 45.0)
PHONEMES["Y"]["f"] = (290.0, 2000.0, 2920.0)
PHONEMES["Y"]["bw"] = (65.0, 200.0, 400.0)
PHONEMES["M"]["f"] = (280.0, 1100.0, 2500.0)
PHONEMES["M"]["bw"] = (50.0, 200.0, 120.0)
PHONEMES["N"]["f"] = (280.0, 1550.0, 2740.0)
PHONEMES["N"]["bw"] = (90.0, 260.0, 225.0)
PHONEMES["NG"]["f"] = (410.0, 1800.0, 2400.0)
PHONEMES["NG"]["bw"] = (90.0, 260.0, 200.0)
PHONEMES["AHA"]["bw"] = (60.0, 80.0, 140.0)
PHONEMES["LT"]["bw"] = (55.0, 75.0, 210.0)
