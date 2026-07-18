# -*- coding: utf-8 -*-
"""Where ClaritySynth keeps downloaded voices, vocoders and tashkeel libs.

IMPORTANT: downloaded content must live OUTSIDE the add-on folder, because
NVDA deletes and recreates the add-on directory on every update/reinstall —
anything stored inside it (e.g. lib/piper_voices) is lost. So downloads go to
a persistent folder under NVDA's configuration directory
(``%APPDATA%\\nvda\\claritysynth`` on Windows), which survives updates.

The add-on's OWN bundled defaults stay where they ship, inside the add-on's
``lib/`` — those are reinstalled with each update and never need to persist.

Every place that discovers or loads a model checks BOTH locations:
  * the persistent user data dir (downloads), and
  * the bundled ``lib/`` (defaults).
"""
import os

_here = os.path.dirname(os.path.abspath(__file__))

# bundled defaults (shipped in the add-on; replaced on update — fine)
BUNDLED_LIB = os.path.join(_here, "lib")


def _config_root():
    """NVDA's config directory, or a sensible fallback off-NVDA (tests)."""
    try:
        import globalVars
        p = globalVars.appArgs.configPath
        if p:
            return p
    except Exception:
        pass
    # fallbacks: APPDATA/nvda, else a temp dir (never inside the add-on)
    appdata = os.environ.get("APPDATA")
    if appdata:
        return os.path.join(appdata, "nvda")
    return os.path.join(os.path.expanduser("~"), ".config", "nvda")


def user_data_dir():
    """Persistent, update-surviving folder for downloaded content."""
    d = os.path.join(_config_root(), "claritysynth")
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return d


# Sub-folders for each kind of downloadable content, in the persistent dir.
def arabic_models_dir():
    return _sub(os.path.join("tts_arabic", "data"))


def vocoders_dir():
    # vocoders live beside the Arabic models (same loader dir)
    return _sub(os.path.join("tts_arabic", "data"))


def vowelizers_dir():
    return _sub("vowelizers")


def piper_voices_dir():
    return _sub("piper_voices")


def _sub(rel):
    d = os.path.join(user_data_dir(), rel)
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return d


def search_dirs(rel):
    """Both places to look for a given kind of content: the persistent
    user data dir FIRST (so a downloaded model overrides a bundled one),
    then the bundled lib/. `rel` is a path relative to each root, e.g.
    "piper_voices" or os.path.join("tts_arabic", "data")."""
    out = []
    u = os.path.join(user_data_dir(), rel)
    if os.path.isdir(u):
        out.append(u)
    b = os.path.join(BUNDLED_LIB, rel)
    if os.path.isdir(b):
        out.append(b)
    return out
