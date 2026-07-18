# -*- coding: utf-8 -*-
"""ClaritySynth downloads window (under NVDA's Tools menu).

Lets the user download extra Arabic voice models, vocoders, and tashkeel
(diacritization) libraries into ClaritySynth's data folders, so the add-on
can support more voices and languages without a new release. The catalogue
and the Google-Drive download logic are adapted from the NabraTTS add-on by
"pbt", shared by Ilyas Dragonoid.

All UI text is in English.
"""
import os
import re
import threading
import time
import http.cookiejar
import urllib.request
import urllib.error

import globalPluginHandler
import gui
import wx
from logHandler import log

# Bind _() to this add-on's translation catalogue. Without this, _() is
# Python's identity builtin and the UI stays English even when a translation
# is installed. Must run at import time, before any _() call.
try:
    import addonHandler
    addonHandler.initTranslation()
except Exception:
    # running outside NVDA (tests) — provide a passthrough so _() exists
    import builtins
    if not hasattr(builtins, "_"):
        builtins._ = lambda s: s


# ClaritySynth's data folders. Downloads go to a PERSISTENT directory outside
# the add-on (NVDA config dir) so an add-on update/reinstall does not wipe
# them. The driver's data_paths module owns these locations and the engines
# search them, so downloaded content is found automatically after an update.
_SYNTH_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "synthDrivers", "claritysynth")


def _dirs():
    try:
        import sys
        if _SYNTH_DIR not in sys.path:
            sys.path.append(_SYNTH_DIR)
        import data_paths
        return (data_paths.arabic_models_dir(), data_paths.vocoders_dir(),
                data_paths.vowelizers_dir(), data_paths.piper_voices_dir())
    except Exception:
        # last-resort fallback (should not normally happen): add-on lib/
        lib = os.path.join(_SYNTH_DIR, "lib")
        a = os.path.join(lib, "tts_arabic", "data")
        return (a, a, os.path.join(lib, "vowelizers"),
                os.path.join(lib, "piper_voices"))


_ARABIC_DIR, _VOCODER_DIR, _VOWELIZER_DIR, _PIPER_DIR = _dirs()

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
]

# name -> (download URL, destination folder, human description)
_CATALOGUE = [
    # Arabic acoustic (TTS) models
    ("mixer128.onnx",
     "https://drive.google.com/file/d/1Aki7_E5KRzWTWG8721xl7-SZQ2bE5U19/view",
     _ARABIC_DIR, "Arabic voice - MixerTTS-128 (standard, 4 speakers)"),
    ("mixer80.onnx",
     "https://drive.google.com/file/d/1C95RVIjhVttC8pdFAp1TOiEee9Cq50c8/view",
     _ARABIC_DIR, "Arabic voice - MixerTTS-80 (faster, smaller)"),
    ("fp_ms.onnx",
     "https://drive.google.com/file/d/1pD210QTN1IL3CTA1D65ldKB7ooZ2hANl/view",
     _ARABIC_DIR, "Arabic voice - FastPitch multi-speaker (high quality)"),
    # Vocoders
    ("vocos22.onnx",
     "https://drive.google.com/file/d/1oNya-eTXB0_yqzHCIzPoHlJD1fwAUthI/view",
     _VOCODER_DIR, "Vocoder - Vocos 22 kHz (recommended)"),
    ("vocos44.onnx",
     "https://drive.google.com/file/d/1Ra0aNGYgD_j0jHs3rmytFEP0_JoIM2GJ/view",
     _VOCODER_DIR, "Vocoder - Vocos 44 kHz (higher fidelity)"),
    ("hifigan.onnx",
     "https://drive.google.com/file/d/1rZxulMhjrlQDheoGy7xnlWGjFYyjF9Gz/view",
     _VOCODER_DIR, "Vocoder - HiFi-GAN"),
    ("denoiser.onnx",
     "https://drive.google.com/file/d/1XWgV7F7eQdRy-KTvCteyXVXAQoNIRa7z/view",
     _VOCODER_DIR, "HiFi-GAN denoiser (used with HiFi-GAN)"),
    # Tashkeel / vowelizer libraries
    ("rawi_ensemble.onnx",
     "https://huggingface.co/TigreGotico/rawi-ensemble/resolve/main/"
     "rawi_ensemble.onnx",
     _VOWELIZER_DIR, "Tashkeel - Rawi ensemble"),
    ("catt_eo.onnx",
     "https://drive.google.com/file/d/15fUlilIt6hp_glYAOpSGlb4zWurErUjn/view",
     _VOWELIZER_DIR, "Tashkeel - CATT EO"),
    ("shakkala.onnx",
     "https://drive.google.com/file/d/1_BbfNj8fsSeGGSkws1tN6EB4zSmpTGp2/view",
     _VOWELIZER_DIR, "Tashkeel - Shakkala"),
    ("shakkelha.onnx",
     "https://drive.google.com/file/d/1scpaMnVLjrDkGBL239pWeb7QW76b15W1/view",
     _VOWELIZER_DIR, "Tashkeel - Shakkelha"),
]

# ---- Additional Arabic Piper voices (community / OVOS repos). These are
# ---- standard espeak-based Piper voices, so they work as PRIMARY voices
# ---- under the "Piper" engine. Each is stored with an ar_ prefixed file
# ---- name so ClaritySynth recognises it as Arabic. The companion .json
# ---- (phoneme map + config) is downloaded alongside the model.
def _ar_piper(stem, model_url, config_url, desc):
    return (stem + ".onnx", model_url, _PIPER_DIR, desc,
            [(config_url, stem + ".onnx.json")])


_ARABIC_PIPER_CATALOGUE = [
    # Kareem (Jordan) — the well-known Arabic Piper voice, hosted at rhasspy.
    _ar_piper(
        "ar_JO-kareem-low",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
        "ar/ar_JO/kareem/low/ar_JO-kareem-low.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
        "ar/ar_JO/kareem/low/ar_JO-kareem-low.onnx.json",
        "Arabic - Kareem (Jordan, low/fast; usable as primary)"),
    _ar_piper(
        "ar_JO-kareem-medium",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
        "ar/ar_JO/kareem/medium/ar_JO-kareem-medium.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
        "ar/ar_JO/kareem/medium/ar_JO-kareem-medium.onnx.json",
        "Arabic - Kareem (Jordan, medium/clearer; usable as primary)"),
    _ar_piper(
        "ar_AE-emirati-female-medium",
        "https://huggingface.co/vadimbelsky/arabic-emirati-female-piper/"
        "resolve/main/arabic-emirati-female-model.onnx",
        "https://huggingface.co/vadimbelsky/arabic-emirati-female-piper/"
        "resolve/main/arabic-emirati-female-model.onnx.json",
        "Arabic - Emirati female (UAE; usable as primary)"),
    _ar_piper(
        "ar_SA-miro-medium",
        "https://huggingface.co/OpenVoiceOS/phoonnx_ar_miro_espeak_V2/"
        "resolve/main/miro_ar.onnx",
        "https://huggingface.co/OpenVoiceOS/phoonnx_ar_miro_espeak_V2/"
        "resolve/main/miro_ar.piper.json",
        "Arabic - Miro (Saudi, diacritized, v2; usable as primary)"),
    _ar_piper(
        "ar_SA-miro-v1-medium",
        "https://huggingface.co/OpenVoiceOS/phoonnx_ar_miro_espeak/"
        "resolve/main/miro_ar.onnx",
        "https://huggingface.co/OpenVoiceOS/phoonnx_ar_miro_espeak/"
        "resolve/main/miro_ar.piper.json",
        "Arabic - Miro (Saudi, diacritized, v1; usable as primary)"),
    _ar_piper(
        "ar_SA-dii-medium",
        "https://huggingface.co/OpenVoiceOS/phoonnx_ar_dii_espeak/"
        "resolve/main/dii_ar.onnx",
        "https://huggingface.co/OpenVoiceOS/phoonnx_ar_dii_espeak/"
        "resolve/main/dii_ar.piper.json",
        "Arabic - Dii (Saudi, diacritized; usable as primary)"),
]
_CATALOGUE = _CATALOGUE + _ARABIC_PIPER_CATALOGUE

# ---------------------------------------------------------------------------
# Piper voice catalogue (secondary / non-Arabic voices, plus Arabic Piper).
# Source: rhasspy/piper-voices on Hugging Face (the canonical Piper voice
# repo, also used by the Sonata add-on). Each voice needs its .onnx and the
# matching .onnx.json. "Fast" = low / x_low quality (lowest latency,
# recommended); "Standard" = medium quality (clearer but SLOWER).
# ---------------------------------------------------------------------------
# Standard Piper voices (rhasspy) — quality tiers x_low/low/medium/high.
_HF = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
# The TRUE "fast" streaming variants: the Sonata developer (mush42) exports
# real-time (+RT) versions of Piper voices, hosted in a separate HF dataset.
# These are the ones Sonata calls "Fast". We fetch this catalogue at runtime
# (see _load_rt_catalogue) so we always point at the voices that actually
# exist, rather than hard-coding a list that could drift.
_RT_LIST_URL = ("https://huggingface.co/datasets/mush42/piper-rt/"
                "raw/main/voices.json")
_RT_PREFIX = ("https://huggingface.co/datasets/mush42/piper-rt/"
              "resolve/main")

# (stem, lang_path, quality, is_fast, human)  — url built below
_PIPER_CATALOGUE_RAW = [
    # English (US) — fast first
    ("en_US-lessac-low", "en/en_US/lessac/low", "low", True,
     "English US - Lessac (low quality, fast)"),
    ("en_US-amy-low", "en/en_US/amy/low", "low", True,
     "English US - Amy (low quality, fast)"),
    ("en_US-ryan-low", "en/en_US/ryan/low", "low", True,
     "English US - Ryan (low quality, fast)"),
    ("en_US-lessac-medium", "en/en_US/lessac/medium", "medium", False,
     "English US - Lessac (standard, slower)"),
    ("en_US-amy-medium", "en/en_US/amy/medium", "medium", False,
     "English US - Amy (standard, slower)"),
    # English (GB)
    ("en_GB-alan-low", "en/en_GB/alan/low", "low", True,
     "English UK - Alan (low quality, fast)"),
    ("en_GB-cori-medium", "en/en_GB/cori/medium", "medium", False,
     "English UK - Cori (standard, slower)"),
    # French
    ("fr_FR-siwis-low", "fr/fr_FR/siwis/low", "low", True,
     "French - Siwis (low quality, fast)"),
    ("fr_FR-upmc-medium", "fr/fr_FR/upmc/medium", "medium", False,
     "French - UPMC (standard, slower)"),
    # Spanish
    ("es_ES-davefx-medium", "es/es_ES/davefx/medium", "medium", False,
     "Spanish - DaveFX (standard, slower)"),
    ("es_ES-sharvard-medium", "es/es_ES/sharvard/medium", "medium", False,
     "Spanish - Sharvard (standard, slower)"),
    # German
    ("de_DE-thorsten-low", "de/de_DE/thorsten/low", "low", True,
     "German - Thorsten (low quality, fast)"),
    ("de_DE-thorsten-medium", "de/de_DE/thorsten/medium", "medium", False,
     "German - Thorsten (standard, slower)"),
    # Italian
    ("it_IT-riccardo-x_low", "it/it_IT/riccardo/x_low", "x_low", True,
     "Italian - Riccardo (low quality, fast)"),
]

# Build ONE download entry per Piper voice. The main file is the .onnx; the
# matching .onnx.json config is fetched automatically alongside it (5th tuple
# element = list of (url, filename) companions), so a voice is never left
# half-installed.
_PIPER_CATALOGUE = []
for _stem, _lp, _q, _fast, _desc in _PIPER_CATALOGUE_RAW:
    _url = "%s/%s/%s.onnx" % (_HF, _lp, _stem)
    _PIPER_CATALOGUE.append((
        _stem + ".onnx", _url, _PIPER_DIR, _desc,
        [(_url + ".json", _stem + ".onnx.json")]))

_CATALOGUE = _CATALOGUE + _PIPER_CATALOGUE


def _load_rt_catalogue():
    """Fetch Sonata's real-time (+RT) 'fast' voice list and return catalogue
    entries for them. These are the genuine streaming-optimised variants the
    Sonata developer (mush42) exports, hosted at mush42/piper-rt. Returns a
    list of (name, url, dest, desc, companions) tuples, or [] on any error."""
    import json as _json
    out = []
    try:
        op = _opener(_USER_AGENTS[0])
        with op.open(_RT_LIST_URL, timeout=30) as resp:
            data = _json.loads(resp.read().decode("utf-8", "ignore"))
    except Exception:
        return out
    # voices.json maps a voice key -> info incl. its files. Structure mirrors
    # rhasspy/piper-voices. We pull the .onnx and its .onnx.json.
    try:
        for key, info in data.items():
            files = (info or {}).get("files", {}) or {}
            onnx_path = None
            for fp in files:
                if fp.endswith(".onnx"):
                    onnx_path = fp
                    break
            if not onnx_path:
                continue
            stem = os.path.basename(onnx_path)
            url = "%s/%s" % (_RT_PREFIX, onnx_path)
            lang = (info.get("language", {}) or {}).get("code", "") or ""
            is_ar = str(lang).lower().startswith("ar")
            desc = "%s - RT streaming (fastest)%s" % (
                key, " [Arabic, usable as primary]" if is_ar else "")
            companions = [("%s/%s.json" % (_RT_PREFIX, onnx_path),
                           stem + ".json")]
            out.append((stem, url, _PIPER_DIR, desc, companions))
    except Exception:
        return []
    return out


def _gdrive_id(url):
    for pat in (r"/d/([a-zA-Z0-9_-]+)", r"[?&]id=([a-zA-Z0-9_-]+)"):
        m = re.search(pat, url)
        if m:
            return m.group(1)
    return url


def _gdrive_urls(file_id):
    base = "https://drive.usercontent.google.com/download"
    return [
        "%s?id=%s&export=download" % (base, file_id),
        "%s?id=%s&export=download&confirm=t" % (base, file_id),
        "https://drive.google.com/uc?id=%s&export=download" % file_id,
    ]


def _opener(ua):
    cj = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    op.addheaders = [("User-Agent", ua)]
    return op


def _extract_confirm(html, file_id):
    out = []
    m = re.search(
        r'href="(https://drive\.usercontent\.google\.com/download[^"]+)"',
        html)
    if m:
        out.append(m.group(1).replace("&amp;", "&"))
    m = re.search(r'name="confirm"\s+value="([^"]+)"', html)
    if m:
        out.append("https://drive.usercontent.google.com/download"
                   "?id=%s&export=download&confirm=%s" % (file_id, m.group(1)))
    return out


def _is_html(resp):
    return "text/html" in resp.headers.get("Content-Type", "")


def _bundled_dir_for(dest):
    """The add-on's bundled folder that corresponds to a download
    destination (the persistent dirs mirror the bundled lib/ layout)."""
    lib = os.path.join(_SYNTH_DIR, "lib")
    # match by the trailing sub-path of the destination
    dest_n = os.path.normpath(dest)
    if dest_n.endswith(os.path.join("tts_arabic", "data")):
        return os.path.join(lib, "tts_arabic", "data")
    if dest_n.endswith("vowelizers"):
        return os.path.join(lib, "vowelizers")
    if dest_n.endswith("piper_voices"):
        return os.path.join(lib, "piper_voices")
    return None


def _present_at(path, name):
    p = os.path.join(path, name)
    if not os.path.exists(p):
        return False
    # .json configs are small; models must be substantial
    if name.endswith(".json"):
        return os.path.getsize(p) > 100
    return os.path.getsize(p) > 100000


def installed(name, dest):
    # present if it was downloaded (persistent dest) OR ships bundled (lib/)
    if _present_at(dest, name):
        return True
    b = _bundled_dir_for(dest)
    if b and _present_at(b, name):
        return True
    return False


# ---------------------------------------------------------------------------
# Auto-update: check GitHub releases and show the changelog
# ---------------------------------------------------------------------------
_GITHUB_API = ("https://api.github.com/repos/riadassoum/claritysynth/"
               "releases/latest")


def _current_version():
    """Read the installed add-on version from manifest.ini."""
    try:
        mf = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "manifest.ini")
        with open(mf, encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("version"):
                    return line.split("=", 1)[1].strip().strip('"')
    except Exception:
        pass
    return "0"


def _ver_tuple(v):
    out = []
    for part in re.split(r"[.\-]", str(v)):
        try:
            out.append(int(part))
        except ValueError:
            out.append(0)
    return tuple(out)


def check_for_update(on_result):
    """Query GitHub for the latest release in a background thread. Calls
    on_result(latest_version, changelog, download_url) or on_result(None,
    ..., ...) if up to date / unavailable."""
    def _work():
        import json as _json
        try:
            op = _opener(_USER_AGENTS[0])
            op.addheaders += [("Accept", "application/vnd.github+json")]
            with op.open(_GITHUB_API, timeout=30) as resp:
                data = _json.loads(resp.read().decode("utf-8", "ignore"))
            latest = (data.get("tag_name") or data.get("name") or "").lstrip("v")
            body = data.get("body") or _("(no changelog provided)")
            url = data.get("html_url") or ""
            assets = data.get("assets") or []
            for a in assets:
                if (a.get("name") or "").endswith(".nvda-addon"):
                    url = a.get("browser_download_url") or url
                    break
            if latest and _ver_tuple(latest) > _ver_tuple(_current_version()):
                wx.CallAfter(on_result, latest, body, url)
            else:
                wx.CallAfter(on_result, None, body, url)
        except Exception:
            log.debugWarning("ClaritySynth update check failed", exc_info=True)
            wx.CallAfter(on_result, None, "", "")
    threading.Thread(target=_work, daemon=True).start()

class GlobalPlugin(globalPluginHandler.GlobalPlugin):

    def __init__(self, *args, **kwargs):
        super(GlobalPlugin, self).__init__(*args, **kwargs)
        self._item = None
        self._submenu = None
        self._submenuItem = None
        self._cancel = threading.Event()
        try:
            tools = gui.mainFrame.sysTrayIcon.toolsMenu
            # Group all ClaritySynth tools under a single "ClaritySynth"
            # submenu of the Tools menu, instead of three loose items.
            self._submenu = wx.Menu()
            self._item = self._submenu.Append(
                wx.ID_ANY, _("Voice && model &downloads..."),
                _("Download extra Arabic voices, vocoders and tashkeel "
                  "libraries for ClaritySynth"))
            gui.mainFrame.sysTrayIcon.Bind(
                wx.EVT_MENU, self._onOpen, self._item)
            self._diacItem = self._submenu.Append(
                wx.ID_ANY, _("Diacritize Arabic text (&tashkeel)..."),
                _("Add diacritics to Arabic text using a chosen tashkeel "
                  "library, whatever synthesizer is active"))
            gui.mainFrame.sysTrayIcon.Bind(
                wx.EVT_MENU, self._onDiacritize, self._diacItem)
            self._updItem = self._submenu.Append(
                wx.ID_ANY, _("Check for &updates..."),
                _("Check GitHub for a newer version of ClaritySynth"))
            gui.mainFrame.sysTrayIcon.Bind(
                wx.EVT_MENU, self._onCheckUpdate, self._updItem)
            # attach the submenu to the Tools menu
            self._submenuItem = tools.AppendSubMenu(
                self._submenu, _("Clarity&Synth"))
        except Exception:
            log.debugWarning("ClaritySynth downloader: menu add failed",
                             exc_info=True)
        # quiet automatic check shortly after startup (does not nag if the
        # network is down or the user is up to date)
        try:
            wx.CallLater(8000, self._autoCheck)
        except Exception:
            pass

    def terminate(self):
        try:
            # removing the submenu item removes the whole ClaritySynth submenu
            if getattr(self, "_submenuItem", None) is not None:
                gui.mainFrame.sysTrayIcon.toolsMenu.RemoveItem(
                    self._submenuItem)
                self._submenuItem = None
            self._submenu = None
            self._item = None
            self._updItem = None
            self._diacItem = None
        except Exception:
            pass

    def _onOpen(self, evt):
        gui.mainFrame.prePopup()
        d = _DownloadDialog(gui.mainFrame)
        d.Show()
        gui.mainFrame.postPopup()

    def _onDiacritize(self, evt):
        gui.mainFrame.prePopup()
        d = _DiacritizeDialog(gui.mainFrame)
        d.Show()
        gui.mainFrame.postPopup()

    def _onCheckUpdate(self, evt):
        self._manualCheck = True
        check_for_update(self._updateResult)

    def _autoCheck(self):
        self._manualCheck = False
        check_for_update(self._updateResult)

    def _updateResult(self, latest, changelog, url):
        if not latest:
            # only speak "up to date" when the user asked explicitly
            if getattr(self, "_manualCheck", False):
                gui.messageBox(
                    _("ClaritySynth is up to date."),
                    _("ClaritySynth update"), wx.OK | wx.ICON_INFORMATION)
            return
        can_direct = bool(url) and url.endswith(".nvda-addon")
        if can_direct:
            msg = _(
                "A new version of ClaritySynth is available: {ver}\n\n"
                "What's new:\n{changelog}\n\n"
                "Download and install it now?").format(
                    ver=latest, changelog=changelog)
        else:
            msg = _(
                "A new version of ClaritySynth is available: {ver}\n\n"
                "What's new:\n{changelog}\n\n"
                "Open the download page now?").format(
                    ver=latest, changelog=changelog)
        if gui.messageBox(msg, _("ClaritySynth update available"),
                          wx.YES_NO | wx.ICON_INFORMATION) != wx.YES:
            return
        if can_direct:
            self._download_and_install_update(url, latest)
        else:
            import webbrowser
            if url:
                webbrowser.open(url)

    def _download_and_install_update(self, url, version):
        """Download the .nvda-addon with a progress dialog and hand it to
        NVDA's add-on installer. Falls back to the browser on any error."""
        import tempfile
        dlg = wx.ProgressDialog(
            _("Updating ClaritySynth"),
            _("Downloading version %s ...") % version,
            maximum=100, parent=gui.mainFrame,
            style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE | wx.PD_CAN_ABORT)
        cancel = {"v": False}
        result = {"path": None, "error": None}

        def _work():
            tmp = os.path.join(tempfile.gettempdir(),
                               "claritySynth-%s.nvda-addon" % version)
            try:
                op = _opener(_USER_AGENTS[0])
                with op.open(url, timeout=120) as resp:
                    total = int(resp.headers.get("Content-Length", 0) or 0)
                    got = 0
                    with open(tmp, "wb") as f:
                        while True:
                            if cancel["v"]:
                                result["error"] = "cancelled"
                                return
                            chunk = resp.read(65536)
                            if not chunk:
                                break
                            f.write(chunk)
                            got += len(chunk)
                            pct = int(got * 100 / total) if total else 0
                            wx.CallAfter(_tick, pct, got, total)
                result["path"] = tmp
            except Exception as e:
                result["error"] = str(e)
            finally:
                wx.CallAfter(_finish)

        def _tick(pct, got, total):
            try:
                mb = got / (1024.0 * 1024.0)
                tmb = total / (1024.0 * 1024.0) if total else 0
                if total:
                    cont, _skip = dlg.Update(
                        min(pct, 99),
                        _("Downloaded %.1f of %.1f MB") % (mb, tmb))
                else:
                    cont, _skip = dlg.Pulse(
                        _("Downloaded %.1f MB") % mb)
                if not cont:
                    cancel["v"] = True
            except Exception:
                pass

        def _finish():
            try:
                dlg.Destroy()
            except Exception:
                pass
            if result["error"] == "cancelled":
                return
            if result["error"] or not result["path"]:
                # fall back to the browser
                import webbrowser
                webbrowser.open(url)
                return
            self._install_addon(result["path"])

        threading.Thread(target=_work, daemon=True).start()

    def _install_addon(self, path):
        """Install a downloaded .nvda-addon via NVDA's add-on API, prompting
        the restart NVDA needs to finish. Falls back to opening the file."""
        try:
            import addonHandler
            try:
                # newer NVDA: installAddon + prompt
                from gui import addonGui
                addonGui.installAddon(gui.mainFrame, path)
                return
            except Exception:
                pass
            bundle = addonHandler.AddonBundle(path)
            addonHandler.installAddonBundle(bundle)
            if gui.messageBox(
                    _("ClaritySynth has been updated. NVDA must restart to "
                      "finish. Restart now?"),
                    _("Update installed"),
                    wx.YES_NO | wx.ICON_INFORMATION) == wx.YES:
                import core
                core.restart()
        except Exception:
            log.debugWarning("ClaritySynth: direct install failed",
                             exc_info=True)
            try:
                os.startfile(path)
            except Exception:
                pass


class _DownloadDialog(wx.Dialog):

    def __init__(self, parent):
        super(_DownloadDialog, self).__init__(
            parent, title=_("ClaritySynth downloads"),
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        self._cancel = threading.Event()
        self._thread = None
        # start with the built-in catalogue; RT "fast" streaming voices are
        # fetched and appended in the background
        self._catalogue = list(_CATALOGUE)

        main = wx.BoxSizer(wx.VERTICAL)
        main.Add(wx.StaticText(self, label=_(
            "Download extra voices, vocoders and tashkeel libraries, grouped "
            "into tabs below. Voices marked \"RT streaming\" are the fastest. "
            "Installed items are marked. They take effect the next time you "
            "switch to ClaritySynth.")), 0, wx.ALL, 8)

        # Tabs group similar items so the window is less cluttered. Each tab
        # has its own list; the tab a catalogue entry lands in is decided by
        # _category_of(). self._lists maps a category key -> its ListBox, and
        # self._list always points at the ACTIVE tab's list so the rest of
        # the code (download, refresh) can stay category-agnostic.
        self._notebook = wx.Notebook(self)
        self._cats = [
            ("arabic", _("Arabic voices")),
            ("voices", _("Other-language voices")),
            ("vocoders", _("Vocoders")),
            ("tashkeel", _("Tashkeel libraries")),
        ]
        self._lists = {}
        for key, title in self._cats:
            panel = wx.Panel(self._notebook)
            ps = wx.BoxSizer(wx.VERTICAL)
            lb = wx.ListBox(panel, style=wx.LB_SINGLE, size=(560, 260))
            lb.Bind(wx.EVT_LISTBOX_DCLICK, self._onDownload)
            ps.Add(lb, 1, wx.EXPAND | wx.ALL, 4)
            panel.SetSizer(ps)
            self._notebook.AddPage(panel, title)
            self._lists[key] = lb
        self._notebook.Bind(
            wx.EVT_NOTEBOOK_PAGE_CHANGED, self._onTabChanged)
        # active list starts as the first tab's
        self._list = self._lists[self._cats[0][0]]
        main.Add(self._notebook, 1, wx.EXPAND | wx.ALL, 8)

        self._refresh_list()
        # fetch the RT (fast streaming) catalogue in the background
        threading.Thread(target=self._load_rt_bg, daemon=True).start()

        self._status = wx.StaticText(self, label=_("Ready."))
        main.Add(self._status, 0, wx.ALL, 8)

        # progress bar + timing detail (accessible: label is updated so NVDA
        # announces it, and the gauge shows visual progress)
        self._gauge = wx.Gauge(self, range=100, size=(560, 18))
        main.Add(self._gauge, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)
        self._detail = wx.StaticText(self, label="")
        main.Add(self._detail, 0, wx.ALL, 8)

        btns = wx.BoxSizer(wx.HORIZONTAL)
        self._dl = wx.Button(self, label=_("&Download selected"))
        self._dl.Bind(wx.EVT_BUTTON, self._onDownload)
        btns.Add(self._dl, 0, wx.RIGHT, 8)
        self._cancelBtn = wx.Button(self, label=_("&Cancel download"))
        self._cancelBtn.Bind(wx.EVT_BUTTON, self._onCancel)
        self._cancelBtn.Disable()
        btns.Add(self._cancelBtn, 0, wx.RIGHT, 8)
        close = wx.Button(self, id=wx.ID_CLOSE, label=_("Cl&ose"))
        close.Bind(wx.EVT_BUTTON, lambda e: self.Close())
        btns.Add(close, 0)
        main.Add(btns, 0, wx.ALL, 8)

        self.SetSizerAndFit(main)
        # Escape closes the window (via the standard cancel id) and the Close
        # button is the default cancel action.
        self.SetEscapeId(wx.ID_CLOSE)
        self.Bind(wx.EVT_CLOSE, self._onCloseEvt)
        # also bind Escape directly as a belt-and-braces accelerator
        self.Bind(wx.EVT_CHAR_HOOK, self._onCharHook)

    def _load_rt_bg(self):
        rt = _load_rt_catalogue()
        if rt:
            # RT voices first (they are the fastest), then the rest
            def _apply():
                existing = {e[0] for e in self._catalogue}
                merged = list(rt) + [e for e in self._catalogue
                                     if e[0] not in {r[0] for r in rt}]
                self._catalogue = merged
                self._refresh_list()
            try:
                wx.CallAfter(_apply)
            except Exception:
                pass

    def _category_of(self, name, dest, desc):
        """Which tab a catalogue entry belongs in."""
        d = (desc or "").lower()
        low = (name or "").lower()
        if dest == _VOWELIZER_DIR or "tashkeel" in d:
            return "tashkeel"
        # Arabic voices — check BEFORE vocoders, because the Arabic model dir
        # and the vocoder dir are the same folder (tts_arabic/data)
        if low.startswith("ar_") or low.startswith("ar-") \
                or "arabic" in d or "mixer" in d or "fastpitch" in d \
                or name in ("mixer128.onnx", "mixer80.onnx", "fp_ms.onnx"):
            return "arabic"
        if "vocoder" in d or "denoiser" in d \
                or name in ("vocos22.onnx", "vocos44.onnx", "hifigan.onnx",
                            "denoiser.onnx"):
            return "vocoders"
        return "voices"

    def _refresh_list(self):
        # distribute catalogue entries across the category tabs, and keep a
        # per-tab index list so a selection maps back to the right entry
        for lb in self._lists.values():
            lb.Clear()
        self._tab_entries = {key: [] for key, _t in self._cats}
        for entry in self._catalogue:
            name, url, dest, desc = entry[0], entry[1], entry[2], entry[3]
            cat = self._category_of(name, dest, desc)
            lb = self._lists.get(cat) or self._lists["voices"]
            mark = _("[installed] ") if installed(name, dest) else ""
            lb.Append("%s%s" % (mark, desc))
            self._tab_entries.setdefault(cat, []).append(entry)

    def _active_cat(self):
        try:
            return self._cats[self._notebook.GetSelection()][0]
        except Exception:
            return self._cats[0][0]

    def _onTabChanged(self, evt):
        # keep self._list pointing at the visible tab's list
        self._list = self._lists[self._active_cat()]
        try:
            evt.Skip()
        except Exception:
            pass

    def _onCharHook(self, evt):
        # Escape closes the window
        try:
            if evt.GetKeyCode() == wx.WXK_ESCAPE:
                self.Close()
                return
        except Exception:
            pass
        evt.Skip()

    def _onCloseEvt(self, evt):
        # abandon any in-progress download and close
        try:
            self._cancel.set()
        except Exception:
            pass
        self.Destroy()

    def _onDownload(self, evt):
        cat = self._active_cat()
        self._list = self._lists[cat]
        i = self._list.GetSelection()
        entries = getattr(self, "_tab_entries", {}).get(cat, [])
        if i < 0 or i >= len(entries):
            return
        entry = entries[i]
        name, url, dest, desc = entry[0], entry[1], entry[2], entry[3]
        companions = entry[4] if len(entry) > 4 else []
        if installed(name, dest):
            self._status.SetLabel(_("Already installed: %s") % desc)
            return
        self._cancel.clear()
        self._dl.Disable()
        self._cancelBtn.Enable()
        self._gauge.SetValue(0)
        self._detail.SetLabel("")
        self._status.SetLabel(_("Downloading %s ...") % desc)
        self._thread = threading.Thread(
            target=self._worker,
            args=(name, url, dest, desc, companions), daemon=True)
        self._thread.start()

    def _onCancel(self, evt):
        self._cancel.set()
        self._status.SetLabel(_("Cancelling..."))

    def _worker(self, name, url, dest, desc, companions=None):
        companions = companions or []
        tmp = os.path.join(dest, name + ".tmp")
        ok = False
        try:
            os.makedirs(dest, exist_ok=True)
            fid = _gdrive_id(url) if "drive.google" in url else None
            for ua in _USER_AGENTS:
                if ok or self._cancel.is_set():
                    break
                op = _opener(ua)
                cands = _gdrive_urls(fid) if fid else [url]
                tried = set()
                while cands and not self._cancel.is_set():
                    u = cands.pop(0)
                    if u in tried:
                        continue
                    tried.add(u)
                    try:
                        with op.open(u, timeout=120) as resp:
                            if _is_html(resp) and fid:
                                html = resp.read(32768).decode(
                                    "utf-8", "ignore")
                                for c in _extract_confirm(html, fid):
                                    if c not in tried:
                                        cands.append(c)
                                continue
                            self._stream(resp, tmp)
                            ok = True
                            break
                    except (urllib.error.HTTPError,
                            urllib.error.URLError):
                        continue
                    except Exception:
                        continue
            if self._cancel.is_set():
                self._rm(tmp)
                wx.CallAfter(self._done, _("Cancelled."))
                return
            if ok:
                os.replace(tmp, os.path.join(dest, name))
                # fetch companion files (e.g. a Piper voice's .onnx.json)
                for c_url, c_name in companions:
                    if self._cancel.is_set():
                        break
                    self._fetch_simple(c_url, os.path.join(dest, c_name))
                # Rawi needs its vocab.json alongside
                if name.startswith("rawi"):
                    self._fetch_rawi_vocab(dest)
                wx.CallAfter(self._done, _("Installed: %s") % desc, True)
            else:
                self._rm(tmp)
                wx.CallAfter(self._done, _(
                    "Download failed. Check your internet connection and "
                    "try again."))
        except Exception as e:
            self._rm(tmp)
            log.debugWarning("ClaritySynth download error", exc_info=True)
            wx.CallAfter(self._done, _("Error: %s") % e)

    def _stream(self, resp, tmp):
        # total size from Content-Length if the server provides it
        total = 0
        try:
            total = int(resp.headers.get("Content-Length", 0) or 0)
        except Exception:
            total = 0
        got = 0
        start = time.time()
        last_ui = 0.0
        with open(tmp, "wb") as f:
            while not self._cancel.is_set():
                chunk = resp.read(65536)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                now = time.time()
                # throttle UI updates to ~5/sec so we do not flood NVDA
                if now - last_ui >= 0.2:
                    last_ui = now
                    self._report_progress(got, total, now - start)
        # final 100% tick
        self._report_progress(got, total, time.time() - start, done=True)

    def _report_progress(self, got, total, elapsed, done=False):
        def _fmt_size(b):
            mb = b / (1024.0 * 1024.0)
            return "%.1f MB" % mb

        def _fmt_time(s):
            s = int(max(0, s))
            m, s = divmod(s, 60)
            return "%d:%02d" % (m, s)

        if total > 0:
            pct = min(100, int(got * 100 / total))
            speed = got / elapsed if elapsed > 0 else 0
            remaining = (total - got) / speed if speed > 0 else 0
            detail = _(
                "%(pct)d%%  -  %(got)s of %(total)s  -  "
                "elapsed %(el)s, remaining %(rem)s") % {
                "pct": pct, "got": _fmt_size(got), "total": _fmt_size(total),
                "el": _fmt_time(elapsed), "rem": _fmt_time(remaining)}
        else:
            # unknown total: show downloaded size + elapsed only
            pct = 0
            detail = _("%(got)s downloaded  -  elapsed %(el)s") % {
                "got": _fmt_size(got), "el": _fmt_time(elapsed)}

        def _apply():
            try:
                if total > 0:
                    self._gauge.SetValue(100 if done else pct)
                else:
                    # indeterminate: pulse
                    self._gauge.Pulse()
                self._detail.SetLabel(detail)
            except Exception:
                pass
        try:
            wx.CallAfter(_apply)
        except Exception:
            pass

    def _fetch_simple(self, url, path):
        """Download a small companion file (e.g. a Piper .onnx.json)."""
        try:
            op = _opener(_USER_AGENTS[0])
            with op.open(url, timeout=60) as resp:
                data = resp.read()
            with open(path, "wb") as f:
                f.write(data)
        except Exception:
            log.debugWarning("companion fetch failed: %s" % url,
                             exc_info=True)

    def _fetch_rawi_vocab(self, dest):
        vocab = os.path.join(dest, "vocab.json")
        if os.path.exists(vocab):
            return
        url = ("https://huggingface.co/TigreGotico/rawi-ensemble/"
               "resolve/main/vocab.json")
        try:
            op = _opener(_USER_AGENTS[0])
            with op.open(url, timeout=60) as resp:
                data = resp.read()
            with open(vocab, "wb") as f:
                f.write(data)
        except Exception:
            log.debugWarning("Rawi vocab fetch failed", exc_info=True)

    def _rm(self, p):
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

    def _done(self, msg, refresh=False):
        self._status.SetLabel(msg)
        self._dl.Enable()
        self._cancelBtn.Disable()
        try:
            # leave the bar full on success, clear it otherwise
            if refresh:
                self._gauge.SetValue(100)
            else:
                self._gauge.SetValue(0)
                self._detail.SetLabel("")
        except Exception:
            pass
        if refresh:
            self._refresh_list()


def _load_tashkeel_module():
    """Import ClaritySynth's ar_tashkeel module regardless of which
    synthesizer is currently active. _SYNTH_DIR (the ClaritySynth driver
    folder) is already on sys.path, so this works even when the user is on
    a different synth entirely."""
    import sys
    try:
        if _SYNTH_DIR not in sys.path:
            sys.path.append(_SYNTH_DIR)
        import ar_tashkeel
        return ar_tashkeel
    except Exception:
        log.debugWarning("ClaritySynth: ar_tashkeel import failed",
                         exc_info=True)
        return None


def _diac_split(text):
    """Split text into diacritization units small enough to process well,
    preserving the original delimiters (newlines and sentence punctuation)
    so the reassembled output matches the input layout exactly. Handles
    arbitrarily long input by yielding many small pieces."""
    import re
    # split on newlines first (keep them), then long lines on sentence marks
    out = []
    for line in re.split(r"(\n)", text):
        if line == "\n" or line == "":
            out.append(line)
            continue
        if len(line) <= 400:
            out.append(line)
            continue
        # break a very long line on sentence/clause punctuation, keeping it
        pieces = re.split(r"([.!?\u061f\u06d4:\u061b;,\u060c]+\s*)", line)
        buf = ""
        for p in pieces:
            if len(buf) + len(p) > 400 and buf:
                out.append(buf)
                buf = p
            else:
                buf += p
        if buf:
            out.append(buf)
    return out


class _DiacritizeDialog(wx.Dialog):
    """A Tools-menu window that diacritizes (adds tashkeel to) Arabic text of
    any length using a chosen tashkeel library. Works with any active
    synthesizer because it talks to ar_tashkeel directly rather than the
    speech pipeline."""

    def __init__(self, parent):
        super(_DiacritizeDialog, self).__init__(
            parent, title=_("Diacritize Arabic text"),
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        self._cancel = threading.Event()
        self._thread = None

        main = wx.BoxSizer(wx.VERTICAL)
        main.Add(wx.StaticText(self, label=_(
            "Enter or paste Arabic text of any length, choose a tashkeel "
            "library, and press Diacritize. The result appears below, ready "
            "to copy. This works no matter which synthesizer you are "
            "using.")), 0, wx.ALL, 8)

        # input
        main.Add(wx.StaticText(self, label=_("&Text to diacritize:")),
                 0, wx.LEFT | wx.TOP, 8)
        self._input = wx.TextCtrl(
            self, style=wx.TE_MULTILINE | wx.TE_RICH2, size=(620, 170))
        main.Add(self._input, 1, wx.EXPAND | wx.ALL, 8)

        # tashkeel library chooser
        row = wx.BoxSizer(wx.HORIZONTAL)
        row.Add(wx.StaticText(self, label=_("Tashkeel &library:")),
                0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6)
        self._lib = wx.Choice(self)
        self._libIds = []
        self._populate_libraries()
        row.Add(self._lib, 0, wx.RIGHT, 12)
        self._diacBtn = wx.Button(self, label=_("&Diacritize"))
        self._diacBtn.Bind(wx.EVT_BUTTON, self._onDiacritize)
        row.Add(self._diacBtn, 0, wx.RIGHT, 6)
        self._cancelBtn = wx.Button(self, label=_("&Cancel"))
        self._cancelBtn.Bind(wx.EVT_BUTTON, self._onCancel)
        self._cancelBtn.Disable()
        row.Add(self._cancelBtn, 0)
        main.Add(row, 0, wx.ALL, 8)

        # progress
        self._gauge = wx.Gauge(self, range=100, size=(620, 16))
        main.Add(self._gauge, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)
        self._status = wx.StaticText(self, label=_("Ready."))
        main.Add(self._status, 0, wx.ALL, 8)

        # output
        main.Add(wx.StaticText(self, label=_("&Result:")),
                 0, wx.LEFT | wx.TOP, 8)
        self._output = wx.TextCtrl(
            self, style=wx.TE_MULTILINE | wx.TE_RICH2 | wx.TE_READONLY,
            size=(620, 170))
        main.Add(self._output, 1, wx.EXPAND | wx.ALL, 8)

        # bottom buttons
        btns = wx.BoxSizer(wx.HORIZONTAL)
        self._copyBtn = wx.Button(self, label=_("C&opy result"))
        self._copyBtn.Bind(wx.EVT_BUTTON, self._onCopy)
        self._copyBtn.Disable()
        btns.Add(self._copyBtn, 0, wx.RIGHT, 8)
        close = wx.Button(self, wx.ID_CLOSE, _("Cl&ose"))
        close.Bind(wx.EVT_BUTTON, lambda e: self.Close())
        btns.Add(close, 0)
        main.Add(btns, 0, wx.ALL | wx.ALIGN_RIGHT, 8)

        self.SetSizerAndFit(main)
        self.Bind(wx.EVT_CLOSE, self._onClose)
        self._input.SetFocus()

    def _populate_libraries(self):
        labels = {
            "libtashkeel": _("Libtashkeel (recommended)"),
            "rawi": _("Rawi ensemble"),
            "catt": _("CATT"),
            "shakkelha": _("Shakkelha (neural)"),
            "shakkala": _("Shakkala (neural)"),
        }
        names = []
        mod = _load_tashkeel_module()
        if mod is not None:
            try:
                names = [n for n in mod.available() if n != "off"]
            except Exception:
                names = []
        if not names:
            names = ["libtashkeel"]
        self._libIds = names
        for n in names:
            self._lib.Append(labels.get(n, n))
        if self._lib.GetCount():
            self._lib.SetSelection(0)

    def _onDiacritize(self, evt):
        text = self._input.GetValue()
        if not text or not text.strip():
            self._status.SetLabel(_("Please enter some Arabic text first."))
            self._input.SetFocus()
            return
        sel = self._lib.GetSelection()
        if sel < 0 or sel >= len(self._libIds):
            backend = "libtashkeel"
        else:
            backend = self._libIds[sel]
        self._cancel.clear()
        self._diacBtn.Disable()
        self._cancelBtn.Enable()
        self._copyBtn.Disable()
        self._output.SetValue("")
        self._gauge.SetValue(0)
        self._status.SetLabel(_("Diacritizing..."))
        self._thread = threading.Thread(
            target=self._worker, args=(text, backend), daemon=True)
        self._thread.start()

    def _worker(self, text, backend):
        mod = _load_tashkeel_module()
        if mod is None:
            wx.CallAfter(
                self._done,
                _("The tashkeel libraries are unavailable. Make sure "
                  "ClaritySynth is installed."), None)
            return
        try:
            mod.set_backend(backend)
        except Exception:
            pass
        # prefer the strict, non-Arabic-preserving path if available: it uses
        # ONLY the chosen backend (so CATT means CATT) and never deletes
        # English/other words
        strict = getattr(mod, "diacritize_strict", None)
        units = _diac_split(text)
        total = max(1, sum(1 for u in units if u.strip()))
        done = 0
        out_parts = []
        for u in units:
            if self._cancel.is_set():
                wx.CallAfter(self._done, _("Cancelled."), None)
                return
            if not u.strip():
                out_parts.append(u)      # keep blank lines / spacing
                continue
            try:
                if strict is not None:
                    res = strict(u, backend)
                else:
                    res = mod.diacritize_text(u)
            except Exception:
                res = None
            out_parts.append(res if res else u)
            done += 1
            pct = int(done * 100 / total)
            # progressively reveal the result as it is produced
            partial = "".join(out_parts)
            wx.CallAfter(self._progress, pct, done, total, partial)
        final = "".join(out_parts)
        wx.CallAfter(self._done, _("Done."), final)

    def _progress(self, pct, done, total, partial):
        try:
            self._gauge.SetValue(min(100, pct))
            self._status.SetLabel(
                _("Diacritizing... %(done)d of %(total)d") % {
                    "done": done, "total": total})
            self._output.SetValue(partial)
        except Exception:
            pass

    def _done(self, msg, final):
        try:
            self._status.SetLabel(msg)
            self._diacBtn.Enable()
            self._cancelBtn.Disable()
            if final is not None:
                self._output.SetValue(final)
                self._gauge.SetValue(100)
                self._copyBtn.Enable()
                self._copyBtn.SetFocus()
            else:
                self._gauge.SetValue(0)
        except Exception:
            pass

    def _onCopy(self, evt):
        text = self._output.GetValue()
        if not text:
            return
        try:
            if wx.TheClipboard.Open():
                wx.TheClipboard.SetData(wx.TextDataObject(text))
                wx.TheClipboard.Close()
                self._status.SetLabel(_("Result copied to the clipboard."))
        except Exception:
            self._status.SetLabel(_("Could not copy to the clipboard."))

    def _onCancel(self, evt):
        self._cancel.set()
        self._status.SetLabel(_("Cancelling..."))

    def _onClose(self, evt):
        self._cancel.set()
        self.Destroy()
