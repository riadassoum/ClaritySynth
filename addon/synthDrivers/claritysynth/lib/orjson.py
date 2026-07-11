# Pure-Python shim for orjson (bundled build must run on NVDA's 32-bit
# Python where compiled wheels are unavailable). API-compatible subset.
import json as _json

OPT_INDENT_2 = 1
OPT_NON_STR_KEYS = 2
OPT_SERIALIZE_NUMPY = 4


def dumps(obj, option=0, default=None):
    if option & OPT_INDENT_2:
        s = _json.dumps(obj, ensure_ascii=False, indent=2, default=default)
    else:
        s = _json.dumps(obj, ensure_ascii=False, default=default)
    return s.encode("utf-8")


def loads(data):
    if isinstance(data, (bytes, bytearray)):
        data = data.decode("utf-8")
    return _json.loads(data)


class JSONDecodeError(ValueError):
    pass
