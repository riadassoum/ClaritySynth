# ClaritySynth

A neural, Arabic-first, multilingual speech synthesizer add-on for the NVDA screen reader.

Mixer and Piper primary voice engines with automatic diacritization (tashkeel), downloadable Arabic and multilingual Piper voices, and a lightweight Formant driver offering NV Speech Player, a pure-Python engine, and eSpeak NG (multilingual formant, including Arabic and the classic Klatt voices).

All ClaritySynth tools live under a **ClaritySynth** submenu in NVDA's Tools menu.

## Install
Download the latest `claritySynth-*.nvda-addon` from Releases (or `dist/`) and open it with NVDA. Then press **NVDA+Ctrl+S** and choose **ClaritySynth Neural** or **ClaritySynth Formant**.

## Build from source
`python build.py` compiles translations and writes `dist/claritySynth-<version>.nvda-addon`. Options: `--clean`, `--out DIR`.
