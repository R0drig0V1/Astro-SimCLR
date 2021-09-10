import os

# -----------------------------------------------------------------------------

# https://dev.to/0xbf/use-dot-syntax-to-access-dictionary-key-python-tips-10ec
class Args(dict):

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __repr__(self):
        return '<Args ' + dict.__repr__(self) + '>'

# -----------------------------------------------------------------------------
