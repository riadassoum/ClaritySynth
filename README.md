# ClaritySynth

A bilingual (Arabic & English) **neural speech synthesizer** add-on for the [NVDA](https://www.nvaccess.org/) screen reader.

ClaritySynth combines natural neural voices for both Arabic and English, automatic Arabic diacritization (tashkeel), tajweed-aware Arabic phonology, and automatic language switching within a single sentence — plus a lightweight formant synthesizer included as a separate, always-available driver.

## Features

- **Four neural Arabic voices** (MixerTTS-128 + Vocos, 22.05 kHz).
- **Neural English voice** (Piper / VITS), phonemized directly through bundled eSpeak NG — no heavy dependencies.
- **Automatic diacritization** — undiacritized Arabic is vocalized by a neural model (Shakkelha) with a large statistical fallback; already-diacritized text is preserved.
- **Automatic language switching** — Arabic and English in one sentence are each spoken by the right voice; numbers are read in the surrounding language's words.
- **Reliable fast speech** — speed comes from pitch-preserving, glitch-free time-compression (WSOLA), so voices never drop or slur phonemes even when sped up. Optional **Rate boost** checkbox.
- **Normalized controls** — Rate, Pitch, and Volume behave identically across both languages.
- **Character navigation** — Arabic and English letters announced by name.
- **Separate formant synthesizer** (`ClaritySynth Formant`) — three pure-Python voices, tiny and always available, with optional DLL DSP and the diacritizer toggle.

## Installation

1. Download the latest `claritySynth-<version>.nvda-addon` from the [Releases](../../releases) page.
2. Open it with NVDA (or NVDA menu → Tools → Add-on store → Install from external source).
3. Restart NVDA when prompted.
4. In NVDA → Preferences → Settings → Speech (`NVDA+Ctrl+V`), set the synthesizer to **ClaritySynth** and pick an **Arabic Neural** voice.

## Building from source

The add-on is the contents of the `addon/` folder packaged as a ZIP renamed to `.nvda-addon`:

```bash
python build.py
```

This produces `claritySynth-<version>.nvda-addon` in the project root.

## Credits

ClaritySynth builds on many open-source projects — see the in-app documentation (`addon/doc/en/readme.html`) for the full credits list, including tts_arabic, arabic_vocalizer, Shakkelha, Piper, eSpeak NG, Mishkal, Tashkeela, NV Speech Player, and ONNX Runtime.

## Author

**Riad Assoum** — [github.com/riadassoum](https://github.com/riadassoum)

## License

Released under the GNU General Public License (GPL). Bundled components retain their own licenses as described in the in-app documentation.
