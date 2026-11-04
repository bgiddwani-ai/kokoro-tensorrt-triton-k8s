"""Content-addressed cache keys; engines are rebuilt for different GPU stacks."""

import hashlib
import json


def cache_key(identity):
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:24]
