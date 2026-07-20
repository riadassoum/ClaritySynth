# ClaritySynth — Changelog

## 2.0.3

Fixes after 2.0.2:

- **Saving the configuration now works every time.** Changing the voice (or other settings) and saving with NVDA+Ctrl+C now persists on every save, not only the first time in a session. The voice setting no longer rejects a valid choice because of a stale internal list.

## 2.0.2

Fixes after 2.0.1:

- **Formant + eSpeak now reads mixed Arabic/English sentences fully.** When the Formant driver uses eSpeak, the whole sentence — Arabic and English together — is now spoken. Previously only the Arabic parts were read. (eSpeak is multilingual and reads both scripts itself; the Neural driver's separate Arabic/other-language voice handling is unchanged.)
- **No more stray "en" sound in French and other voices.** Non-Arabic voices no longer insert an odd sound before a foreign word (such as an English word inside French text).

## 2.0.1

Fixes after the 2.0 release:

- **Portable NVDA no longer closes when the neural voice is selected.** A native audio library was being loaded the instant the synthesizer was chosen; on portable copies this could close NVDA silently. It is now loaded only when actually needed, so selecting the voice is safe.
- **Letter names read correctly again.** Reading Arabic letters by name (e.g. س, ج) no longer adds spurious case-endings or wrong vowels — the names are spoken as a teacher would say them.
- **الذي / التي and similar words.** Words with a doubled lām (الَّذِي, الَّتِي, الَّذِينَ) are now pronounced with the correct geminated lām instead of dropping it.
- **The ـنا ending (بيتنا, صديقنا).** The first-person-plural ending نا is no longer mistaken for tanwīn — بيتنا is read *baytunā*, not *baytan*. This affects every tashkeel library except Libtashkeel, which was already correct.

## 2.0

ClaritySynth grows from a bilingual (Arabic/English) synthesizer into a neural, Arabic-first, **multilingual** one, with downloadable voices, two-voice selection, an in-app diacritization tool, and major reliability and audio-quality fixes.

### New features
- **A ClaritySynth menu.** The three ClaritySynth items (downloads, diacritization tool, and update check) are now grouped in a single <b>ClaritySynth</b> submenu under NVDA's Tools menu, instead of sitting loose in it.
- **More Arabic voices to download.** The downloads window now offers several Arabic Piper voices for the Piper engine — Kareem (Jordan, low and medium), an Emirati female voice, and three Saudi voices (Miro v1, Miro v2, and Dii, all diacritized).
- **eSpeak NG formant voice.** The Formant driver can now use eSpeak NG, a compact multilingual formant synthesizer, selectable alongside NV Speech Player and the built-in engine. It supports **all of eSpeak's languages** (over a hundred, including Arabic) and **all of its voice variants**, including the classic **Klatt** formant voices — chosen with a language selector and a variant selector, and cycleable from the settings ring. **Rate boost** applies here too for very fast speech.
- **Reliable Formant engine switching.** The NV Speech Player engine now loads correctly in the Formant driver and appears as an option, and switching Formant settings from the settings ring (Ctrl+NVDA+V) no longer freezes.
- **More reliable Arabic voice downloads.** Corrected the Saudi (Miro, Dii) download links and verified the Emirati one; each appears under the Piper engine once installed. (Miro and Dii are diacritized Arabic voices.)
- **More downloadable Arabic voices.** Added community Arabic Piper voices to the downloads window — an Emirati voice and two Saudi voices (Miro and Dii) — which appear under the Piper engine once installed.
- **Tabbed, tidier downloads window.** Downloads are now grouped into tabs (Arabic voices, other-language voices, vocoders, tashkeel libraries), and Escape closes the window.
- **Primary voice engine selector.** Choose your primary Arabic voice engine — Mixer (the multi-speaker neural models, with speaker, model and vocoder choices) or Piper (Arabic Piper voices such as Kareem, with a voice and quality-tier choice). Picking Piper here plays it at its correct natural pitch, and any Arabic Piper voices you download later appear here automatically.
- **Diacritization window.** A Tools-menu item, "Diacritize Arabic text (tashkeel)", diacritizes Arabic text of any length with a tashkeel library of your choice and gives you the result to copy — regardless of which synthesizer is active.
- **Formant engine choice.** The formant voice can use the NV Speech Player engine or the built-in one, selectable in its settings.
- **Multilingual voices.** Beyond the built-in Arabic and English voices, download Piper voices for many languages (French, Spanish, German, Italian, Arabic, and more) — including fast real-time streaming variants — from a downloads window under the NVDA **Tools** menu. Each voice is spoken in its own language.
- **Two voices, each with a variant.** Voice settings offer a **Primary voice** (Arabic) and a **Secondary voice** (non-Arabic), each with a quality/speed variant selector. A downloaded Arabic Piper voice (e.g. Kareem) can serve as the Primary voice, using the same tashkeel pipeline.
- **Batch diacritization tool.** A new **Diacritize Arabic text** window under the Tools menu adds tashkeel to Arabic text of any length using the library you choose, and works no matter which synthesizer is active. The result is shown ready to copy.
- **Downloads window and auto-update.** Install extra Arabic acoustic models, vocoders, tashkeel libraries, and Piper voices, with a progress bar showing size and time remaining. The add-on also checks GitHub for new versions, shows the changelog, and can download and install an update directly.
- **Arabic interface.** The add-on is fully translatable and now correctly shows its Arabic translation when NVDA is set to Arabic.
- **Downloaded content survives updates.** Voices, vocoders and tashkeel libraries you download are stored outside the add-on folder, so updating or reinstalling ClaritySynth no longer deletes them.

### Audio quality
- **Clean, consistent volume, no distortion.** Reworked the entire audio chain so the voice is never saturated or "clipped", even at moderate volume and even on the Arabic Mixer voices. Volume reaches a genuinely louder maximum without distorting.
- **No more grainy "old radio" sound on some speakers.** The mid-sentence volume evening used on the quieter Mixer speakers (2 and 4) no longer amplifies background noise, so those voices stay clean.
- **Arabic and non-Arabic voices are always equally loud.** Both languages are matched to the same loudness and no longer depend on the order in which you switch voices.
- **Reliable fast speech.** Raising the rate speeds speech up through pitch-preserving time-compression, so voices stay clear and never drop or slur phonemes. An optional **Rate boost** pushes the top speed much further.

### Voices and pronunciation
- **Non-English voices read correctly.** Downloaded voices for other languages are phonemized in their own language (previously some were read with English pronunciation), and no longer drop letters.
- **All Mixer speakers work.** Every speaker of a multi-speaker Arabic model — including the faster Mixer model — now works, instead of every choice sounding like speaker 1.
- **Arabic Piper voice keeps its natural sound as primary.** An Arabic Piper voice used as the Primary voice keeps its own natural pitch and tempo.
- **Numbers use the neural voice.** A number alone on a line is read by the neural voice rather than the formant fallback.
- **Punctuation is spoken properly.** Colons, question marks and exclamation marks inside a sentence produce a real pause and fresh intonation instead of being flattened.
- **Text after a Quran quotation is diacritized again.** Everything following a fully-diacritized verse is now diacritized as expected.
- **بالله / والله / تالله** are no longer mispronounced with the leading letter dropped (fixed in both Libtashkeel and Rawi).
- **Diacritized Quranic text** no longer stops the diacritization of the text that follows it.
- **Hamza letters and diacritic marks are named** individually when read one character at a time.
- **Whole sentences are never chopped into fragments** — a short phrase like "This is a test" is spoken as one unit; text is only divided at real punctuation.
- Adopted an improved tanween (tanwīn + alif) phonemization rule.

### Tashkeel (diacritization)
- Tashkeel libraries streamlined to **Libtashkeel** (default) and **Rawi**, plus **CATT** and the **Shakkala / Shakkelha** diacritizers when downloaded, or **Off**. CATT works correctly here.
- **The formant voice can now use the same tashkeel libraries** as the neural voice.

### Reliability
- **Fixed the "Adam fallback" bug.** On some machines the neural voices silently fell back to a formant voice ("Adam"). The cause was the ONNX runtime failing to load when the system was missing the Microsoft Visual C++ Redistributable. The runtime now loads far more robustly, the fallback voice was removed, and if the neural engine genuinely cannot start, ClaritySynth says so and explains how to fix it.
- **Fixed the synthesizer-selection freeze/crash** when choosing a voice with **Ctrl+NVDA+S**, including with other add-ons (such as Google TTS) installed. Listing voices is now instant; models load in the background.
- **Formant driver fixes.** The lightweight formant voice no longer starts the next sentence before finishing the current one when you move quickly through text, and its timing is steadier.

### Voice naming
- The two drivers are now simply **ClaritySynth Neural** and **ClaritySynth Formant**.

### Credits
Thanks to the users who reported issues and contributed pronunciation fixes. ClaritySynth builds on eSpeak NG, Piper/VITS voices (rhasspy and the Sonata project by Musharraf Omer for the real-time voices), the Vocos vocoder, and Arabic diacritization and phonology models. Some downloadable Arabic models, vocoders and tashkeel libraries, and salvaged components, come by way of the NabraTTS add-on (by "pbt", shared by Ilyas Dragonoid).
