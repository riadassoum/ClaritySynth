# -*- coding: utf-8 -*-
"""Isolated loader for the bundled numpy / onnxruntime.

Other NVDA add-ons (EasySoundRecorder, etc.) put their OWN, often broken,
numpy on sys.path and even half-load it into sys.modules. A plain
`import numpy` then binds their broken copy. This loader forces OUR
bundled copy to win by:

1. Purging any already-cached numpy/onnxruntime (broken or not) from
   sys.modules so a fresh import can happen.
2. Putting our lib/ FIRST on sys.path and adding our numpy's DLL folder
   via os.add_dll_directory (Windows) so the C-extensions resolve.
3. Importing numpy and onnxruntime, then keeping our lib on the path so
   dependent imports (tts_arabic) also resolve.

The REAL exception is recorded in `last_error` and logged, so failures
are diagnosable instead of silent.
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_LIB = os.path.join(_here, "lib")

_booted = False
ok = False
last_error = None


def _purge(mod):
    for k in [m for m in list(sys.modules)
              if m == mod or m.startswith(mod + ".")]:
        try:
            del sys.modules[k]
        except Exception:
            pass


def boot():
    global _booted, ok, last_error
    if _booted:
        return ok
    _booted = True

    # 1. Is a WORKING numpy already loaded? If so, reuse it (a C-extension
    #    can load only once per process, so we cannot load a second copy).
    good_np = False
    if "numpy" in sys.modules:
        try:
            sys.modules["numpy"].array([0])
            good_np = True
        except Exception:
            _purge("numpy")   # broken cached copy -> remove it
    # onnxruntime: same idea
    good_ort = False
    if "onnxruntime" in sys.modules:
        try:
            sys.modules["onnxruntime"].get_available_providers()
            good_ort = True
        except Exception:
            _purge("onnxruntime")

    saved_path = list(sys.path)
    dll_ctx = None
    try:
        # 2. our lib/ first
        while _LIB in sys.path:
            sys.path.remove(_LIB)
        sys.path.insert(0, _LIB)
        # numpy's compiled DLLs live in numpy.libs / numpy/_core; make sure
        # Windows can find them
        if hasattr(os, "add_dll_directory"):
            for sub in ("numpy.libs", os.path.join("numpy", "_core"),
                        os.path.join("onnxruntime", "capi")):
                p = os.path.join(_LIB, sub)
                if os.path.isdir(p):
                    try:
                        os.add_dll_directory(p)
                    except Exception:
                        pass
        # 3. import ours explicitly
        if not good_np:
            _purge("numpy")
            import numpy  # noqa: F401
            numpy.array([0])   # sanity check it actually works
            # If we somehow bound another add-on's numpy, force ours by
            # explicit file path (only if that file is really present).
            bound_dir = os.path.dirname(getattr(numpy, "__file__", "") or "")
            our_init = os.path.join(_LIB, "numpy", "__init__.py")
            if _LIB not in bound_dir and os.path.exists(our_init):
                _purge("numpy")
                import importlib.util as _u
                spec = _u.spec_from_file_location("numpy", our_init)
                _np = _u.module_from_spec(spec)
                sys.modules["numpy"] = _np
                spec.loader.exec_module(_np)
                _np.array([0])
        if not good_ort:
            _purge("onnxruntime")
            import onnxruntime  # noqa: F401
        ok = True
        last_error = None
    except Exception as e:
        ok = False
        last_error = repr(e)
    finally:
        # keep our lib reachable (append) but restore prior ordering
        sys.path[:] = saved_path
        if _LIB not in sys.path:
            sys.path.append(_LIB)

    if not ok:
        try:
            from logHandler import log
            log.warning("ClaritySynth _libboot failed: %s" % last_error,
                        exc_info=True)
        except Exception:
            pass
    return ok
