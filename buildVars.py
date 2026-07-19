# -*- coding: UTF-8 -*-
def _(arg):
    return arg
addon_info = {
    "addon_name": "claritySynth",
    "addon_summary": _("ClaritySynth — Neural Multilingual Speech Synthesizer (Arabic-first)"),
    "addon_description": _("A neural, Arabic-first, multilingual speech synthesizer for NVDA. Mixer and Piper primary voice engines with automatic diacritization (tashkeel), downloadable Arabic and multilingual Piper voices, and a lightweight Formant driver offering NV Speech Player, a pure-Python engine, and eSpeak NG (multilingual formant, including Arabic and Klatt voices)."),
    "addon_version": "2.0.1",
    "addon_author": "Riad Assoum",
    "addon_url": "https://github.com/riadassoum/claritysynth",
    "addon_docFileName": "readme.html",
    "addon_minimumNVDAVersion": "2026.1",
    "addon_lastTestedNVDAVersion": "2026.2",
    "addon_updateChannel": None,
}
i18nSources = ["addon", "buildVars.py"]
excludedFiles = []
