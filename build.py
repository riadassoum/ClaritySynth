#!/usr/bin/env python3
"""Build ClaritySynth into a .nvda-addon package.

Zips the contents of the addon/ folder and renames to .nvda-addon, using
the version from addon/manifest.ini.
"""
import os
import re
import zipfile


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    addon_dir = os.path.join(here, "addon")
    manifest = os.path.join(addon_dir, "manifest.ini")

    version = "1.0"
    with open(manifest, encoding="utf-8") as f:
        m = re.search(r"version\s*=\s*(\S+)", f.read())
        if m:
            version = m.group(1)

    out = os.path.join(here, "claritySynth-%s.nvda-addon" % version)
    if os.path.exists(out):
        os.remove(out)

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(addon_dir):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for fn in files:
                if fn.endswith(".pyc"):
                    continue
                p = os.path.join(root, fn)
                z.write(p, os.path.relpath(p, addon_dir))

    size = os.path.getsize(out) // 1024 // 1024
    print("Built %s (%d MB)" % (os.path.basename(out), size))


if __name__ == "__main__":
    main()
