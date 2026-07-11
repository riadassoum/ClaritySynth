# Minimal pure-Python pickledb replacement for the bundled qalsadi cache.
# Implements only the API surface qalsadi uses; no async, no compiled deps.
import json
import os


class PickleDB(object):
    def __init__(self, location, auto_dump=True, sig=True):
        if hasattr(location, "name"):       # qalsadi passes a file handle
            try:
                location.close()
            except Exception:
                pass
            location = location.name
        self.location = os.path.expanduser(location)
        self.auto_dump = auto_dump
        self.db = {}
        if os.path.exists(self.location):
            try:
                with open(self.location, "rt", encoding="utf-8") as f:
                    self.db = json.load(f)
            except Exception:
                self.db = {}

    def dump(self):
        try:
            with open(self.location, "wt", encoding="utf-8") as f:
                json.dump(self.db, f, ensure_ascii=False)
            return True
        except Exception:
            return False

    save = dump

    def _autodump(self):
        if self.auto_dump:
            self.dump()

    def set(self, key, value):
        self.db[key] = value
        self._autodump()
        return True

    def get(self, key):
        return self.db.get(key, False)

    def exists(self, key):
        return key in self.db

    def rem(self, key):
        if key in self.db:
            del self.db[key]
            self._autodump()
        return True

    def getall(self):
        return self.db.keys()

    def deldb(self):
        self.db = {}
        self._autodump()
        return True


def load(location, auto_dump=True, sig=True):
    return PickleDB(location, auto_dump, sig)
