# Enabling the neural diacritizer (optional, highest accuracy)

ClaritySynth's built-in Arabic diacritizer (statistical + Mishkal, ~92%
word accuracy) works with no extra setup. For state-of-the-art accuracy
(~97-99%, including correct case endings / i'rab), you can optionally add
the neural Shakkelha model. This runs entirely offline once installed.

## Why it's optional
The neural model needs `onnxruntime`, which ships a compiled binary that
must exactly match NVDA's bundled Python version and CPU architecture
(win_amd64). That binary can't be shipped blind, so it's a user opt-in.

## Steps (on the Windows machine running NVDA)
1. Find NVDA's Python version (About NVDA, or the log). NVDA 2026.x uses
   CPython 3.11 (64-bit).
2. Download the matching onnxruntime wheel from pypi.org
   (e.g. `onnxruntime-1.17-cp311-cp311-win_amd64.whl`), unzip it, and copy
   its `onnxruntime/` folder into:
   `synthDrivers/claritysynth/lib/onnxruntime/`
   Also copy `numpy` (matching cp311 win_amd64) into `lib/` if not already
   bundled.
3. Get the ONNX model + maps from the MIT-licensed project
   `github.com/nipponjo/arabic_vocalizer` (Shakkelha). Place:
   `synthDrivers/claritysynth/shakkelha.onnx`
   `synthDrivers/claritysynth/shakkelha_maps.json`
   (`shakkelha_maps.json` must contain `char_to_id` and `id_to_diacritic`.)
4. Restart NVDA. ClaritySynth auto-detects the model and routes Arabic
   through it; if anything is missing it silently falls back to the
   built-in diacritizer.

## Credits
- Shakkelha: Ali Fadel et al., "Neural Arabic Text Diacritization",
  EMNLP-IJCNLP 2019 (MIT).
- Shakkala: Barqawiz (MIT).
- ONNX packaging: nipponjo/arabic_vocalizer (MIT).
