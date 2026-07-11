# ClaritySynth v1.1

This release focuses on the fluidity and responsiveness of speech, especially when navigating quickly and when Arabic and English appear together.

## Fixes and improvements

### Continuous, gap-free speech across UI fields
NVDA often sends a single line as several separate pieces of text (for example, an item name, its state, and its position: “Blind Temple Run”, “not selected”, “1 of 85”). Previously each piece was synthesized on its own, which produced phantom pauses that sounded like commas that were not there, and small delays between the pieces. These adjacent pieces are now **merged into one continuous utterance**, so the line is spoken smoothly as written — while the corresponding cursor/index positions are still reported correctly.

### Absolute Arabic/English synchronization
When Arabic and English alternate within one line (for example, “…GPT-5.6 بعد…”), there was a short delay at each switch between the two voices while the next part was being synthesized. Synthesis now runs **ahead of playback on a background thread**: while one part is being spoken, the next part is already being prepared. The result is seamless switching between Arabic and English with no gap, and the two voices never overlap.

### Reliable interruption when moving quickly
When moving rapidly through items — particularly stopping on a comma or period — the voice could briefly start the next item and then cut off. Interruption is now **immediate and clean**: audio prepared for a cancelled item is discarded at once, so moving quickly no longer produces a false start.

### Natural English phrasing
The English neural voice no longer inserts an artificial break every few words. English is split only at **real punctuation** (clauses and sentences), so each phrase is spoken whole with correct intonation. Pauses are language-aware: the Arabic neural voice, which does not itself vary timing for punctuation, is given appropriate pauses at commas and clause boundaries, while the English voice relies on its own natural phrasing.

## Notes
No settings changes are required. Existing configurations continue to work.
