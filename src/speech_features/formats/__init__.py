"""Built-in serialization formats for :class:`SpeechDocument`.

Each registered format maps a name to an ``{"encode", "decode"}`` callable
pair: ``encode(document) -> str`` and ``decode(text) -> SpeechDocument``. The
registry is a static built-in mapping; third-party discovery is deferred
until an actual third-party format exists.
"""

from .chat import decode_chat, encode_chat
from .json import decode_json, encode_json

FORMATS = {
    "chat": {"encode": encode_chat, "decode": decode_chat},
    "json": {"encode": encode_json, "decode": decode_json},
}

__all__ = ["FORMATS"]
