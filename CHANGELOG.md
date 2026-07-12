# ClaritySynth v1.2

This release adds a **choice of Tashkeel (Arabic diacritization) libraries**, improves how Arabic text is cleaned before it is spoken, and fixes a bug that could cut off the end of a sentence.

## Choose your Tashkeel library

A new **Tashkeel library** combo box now sits directly beneath the **Voice** selector in NVDA's speech settings. It lets you choose which engine adds the diacritics (tashkeel) to Arabic text before it is spoken:

- **Libtashkeel** (default) — fast and robust.
- **Rawi ensemble** — a neural diacritizer with very good handling of case endings.
- **Shakkelha** — the previous default.
- **Shakkala** — an alternative neural model.
- **Off** — no automatic diacritization; text is read exactly as written.

Only the libraries that load successfully on your machine are offered, and if the one you pick cannot start for any reason, ClaritySynth quietly falls back to another working library rather than failing. Text that already carries diacritics is still respected: existing marks are preserved, and only bare or partially-marked words are completed.

## Cleaner Arabic reading

Arabic text is now tidied before it is spoken:

- Emoji and decorative runs of symbols (for example `=====`) are no longer read out.
- Stray symbols are spoken as proper Arabic words — `%` as "بِالْمِئَةِ", `+` as "زَائِد", `=` as "يُسَاوِي", and so on.

Links and English words are deliberately **left alone**: ClaritySynth still announces URLs and speaks English with its own neural English voice, rather than hiding them or reading them with Arabic letters.

## Fixes

### No more clipped endings
The end of a sentence could be cut off — a final consonant swallowed, or the last moments of a paragraph lost. This happened most noticeably with **Rate boost switched off**. The cause was in the time-scaling that produces fast speech: the final fragment of audio was being blended with the wrong weighting, which halved or erased it, and small rounding errors made the speaking rate drift from what you had set. Both are fixed. Endings are now preserved intact, and the speaking rate is accurate, at every combination of rate, pitch and Rate boost.

## Notes
Your existing settings are preserved. The new Tashkeel library selector defaults to Libtashkeel; if you preferred the previous behaviour, choose **Shakkelha** in the combo box.

## Thanks

Special thanks to **Ilyas Dragonoid** for sharing the **NabraTTS** add-on (by **pbt**), from which the **libtashkeel** engine and the **Rawi ensemble** diacritizer are bundled, and on which ClaritySynth's Rawi support and parts of its Arabic text cleanup are based.
