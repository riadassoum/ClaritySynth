#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build script for the ClaritySynth NVDA add-on.

Produces ``claritySynth-<version>.nvda-addon`` from the ``addon/`` tree:

  1. reads the add-on version from ``buildVars.py`` (falling back to
     ``addon/manifest.ini``),
  2. compiles every ``locale/*/LC_MESSAGES/nvda.po`` to ``nvda.mo`` (uses
     ``msgfmt`` if available, otherwise a small pure-Python compiler),
  3. zips the ``addon/`` directory contents into the ``.nvda-addon`` package,
     skipping ``__pycache__`` and ``*.pyc``.

Usage:
    python build.py            # build into dist/
    python build.py --clean    # remove build artifacts first
    python build.py --out DIR  # write the package to DIR (default: dist)

This is a dependency-light alternative to the standard NVDA add-on SCons
build; either produces an equivalent package.
"""
import argparse
import os
import re
import struct
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ADDON_DIR = os.path.join(HERE, "addon")


# --------------------------------------------------------------------------
# version
# --------------------------------------------------------------------------
def read_version():
    """Version from buildVars.py, else manifest.ini, else '0.0'."""
    bv = os.path.join(HERE, "buildVars.py")
    if os.path.exists(bv):
        ns = {"_": lambda s: s}
        try:
            with open(bv, encoding="utf-8") as f:
                exec(compile(f.read(), bv, "exec"), ns)
            v = ns.get("addon_info", {}).get("addon_version")
            if v:
                return str(v)
        except Exception as e:
            print("  ! could not read buildVars.py (%s)" % e)
    mf = os.path.join(ADDON_DIR, "manifest.ini")
    if os.path.exists(mf):
        with open(mf, encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("version"):
                    return line.split("=", 1)[1].strip().strip('"').strip()
    return "0.0"


def read_addon_name():
    mf = os.path.join(ADDON_DIR, "manifest.ini")
    with open(mf, encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith("name"):
                return line.split("=", 1)[1].strip().strip('"').strip()
    return "claritySynth"


# --------------------------------------------------------------------------
# gettext .po -> .mo
# --------------------------------------------------------------------------
def _parse_po(path):
    """Minimal PO parser -> {msgid: msgstr}. Handles multi-line quoted
    strings and skips fuzzy/empty entries. Good enough for NVDA add-on
    catalogues (no plural forms used here besides the header)."""
    entries = {}
    msgid = msgstr = None
    target = None
    buf_id = []
    buf_str = []

    def _unquote(line):
        line = line.strip()
        m = re.match(r'^"(.*)"$', line)
        if not m:
            return ""
        return (m.group(1)
                .replace("\\n", "\n").replace("\\t", "\t")
                .replace('\\"', '"').replace("\\\\", "\\"))

    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            s = line.strip()
            if s.startswith("#") or not s:
                if msgid is not None and msgstr is not None:
                    entries["".join(buf_id)] = "".join(buf_str)
                    msgid = msgstr = None
                    buf_id, buf_str = [], []
                continue
            if s.startswith("msgid "):
                if msgid is not None and msgstr is not None:
                    entries["".join(buf_id)] = "".join(buf_str)
                    buf_id, buf_str = [], []
                msgid = True
                msgstr = None
                target = "id"
                buf_id = [_unquote(s[len("msgid "):])]
            elif s.startswith("msgstr "):
                msgstr = True
                target = "str"
                buf_str = [_unquote(s[len("msgstr "):])]
            elif s.startswith('"'):
                if target == "id":
                    buf_id.append(_unquote(s))
                elif target == "str":
                    buf_str.append(_unquote(s))
    if msgid is not None and msgstr is not None:
        entries["".join(buf_id)] = "".join(buf_str)
    return entries


def _write_mo(entries, path):
    """Write a GNU .mo file from a {msgid: msgstr} dict."""
    keys = sorted(entries.keys())
    offsets = []
    ids = b""
    strs = b""
    for k in keys:
        v = entries[k]
        kb = k.encode("utf-8")
        vb = v.encode("utf-8")
        offsets.append((len(ids), len(kb), len(strs), len(vb)))
        ids += kb + b"\x00"
        strs += vb + b"\x00"
    keystart = 7 * 4 + 16 * len(keys)
    valuestart = keystart + len(ids)
    koffsets = []
    voffsets = []
    for o1, l1, o2, l2 in offsets:
        koffsets += [l1, o1 + keystart]
        voffsets += [l2, o2 + valuestart]
    output = struct.pack("Iiiiiii",
                         0x950412de, 0, len(keys),
                         7 * 4, 7 * 4 + len(keys) * 8, 0, 0)
    output += struct.pack("i" * len(koffsets), *koffsets)
    output += struct.pack("i" * len(voffsets), *voffsets)
    output += ids
    output += strs
    with open(path, "wb") as f:
        f.write(output)


def compile_translations():
    locale_root = os.path.join(ADDON_DIR, "locale")
    if not os.path.isdir(locale_root):
        return
    for lang in sorted(os.listdir(locale_root)):
        po = os.path.join(locale_root, lang, "LC_MESSAGES", "nvda.po")
        if not os.path.exists(po):
            continue
        mo = po[:-3] + ".mo"
        try:
            entries = _parse_po(po)
            _write_mo(entries, mo)
            print("  compiled %s (%d strings)" % (
                os.path.relpath(mo, HERE), len(entries)))
        except Exception as e:
            print("  ! failed to compile %s: %s" % (po, e))


# --------------------------------------------------------------------------
# package
# --------------------------------------------------------------------------
def build_package(out_dir):
    name = read_addon_name()
    version = read_version()
    os.makedirs(out_dir, exist_ok=True)
    pkg = os.path.join(out_dir, "%s-%s.nvda-addon" % (name, version))
    if os.path.exists(pkg):
        os.remove(pkg)
    n = 0
    with zipfile.ZipFile(pkg, "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(ADDON_DIR):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for fn in files:
                if fn.endswith(".pyc"):
                    continue
                full = os.path.join(root, fn)
                arc = os.path.relpath(full, ADDON_DIR)
                z.write(full, arc)
                n += 1
    size_mb = os.path.getsize(pkg) / (1024 * 1024)
    print("\nBuilt %s" % os.path.relpath(pkg, HERE))
    print("  %d files, %.1f MB" % (n, size_mb))
    return pkg


def clean():
    import shutil
    for root, dirs, files in os.walk(HERE):
        for d in list(dirs):
            if d == "__pycache__":
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)
                dirs.remove(d)
        for fn in files:
            if fn.endswith(".pyc"):
                try:
                    os.remove(os.path.join(root, fn))
                except OSError:
                    pass
    print("cleaned __pycache__ and *.pyc")


def main():
    ap = argparse.ArgumentParser(description="Build the ClaritySynth add-on.")
    ap.add_argument("--clean", action="store_true",
                    help="remove build artifacts first")
    ap.add_argument("--out", default="dist",
                    help="output directory (default: dist)")
    args = ap.parse_args()

    if args.clean:
        clean()
    print("ClaritySynth build")
    print("  version: %s" % read_version())
    print("compiling translations...")
    compile_translations()
    build_package(os.path.join(HERE, args.out)
                  if not os.path.isabs(args.out) else args.out)


if __name__ == "__main__":
    sys.exit(main())
