"""Pinned RWKV World byte vocabulary for CPU preflight on remote gateways."""
import ast
from hashlib import sha256
from pathlib import Path


class WorldTokenizer:
    def __init__(self, path, expected_sha256):
        raw = Path(path).read_bytes()
        if not expected_sha256 or sha256(raw).hexdigest() != expected_sha256:
            raise ValueError('local RWKV vocabulary hash mismatch')
        self.sha256 = expected_sha256
        self.root = {}
        ids = set()
        values = set()
        for line in raw.decode('utf-8').splitlines():
            index, remainder = line.split(' ', 1)
            literal, length = remainder.rsplit(' ', 1)
            index, length = int(index), int(length)
            value = ast.literal_eval(literal)
            value = value.encode('utf-8') if isinstance(value, str) else value
            if (not isinstance(value, bytes) or not value or len(value) != length
                    or index < 1 or index in ids or value in values):
                raise ValueError('invalid local RWKV vocabulary')
            ids.add(index)
            values.add(value)
            node = self.root
            for byte in value:
                node = node.setdefault(byte, {})
            node[None] = index
        if not all(bytes([byte]) in values for byte in range(256)):
            raise ValueError('local RWKV vocabulary must cover every byte')

    def encode(self, text):
        raw = text.encode('utf-8')
        output = []
        offset = 0
        while offset < len(raw):
            node, end = self.root, offset
            best = None
            while end < len(raw) and raw[end] in node:
                node = node[raw[end]]
                end += 1
                if None in node:
                    best = (end, node[None])
            if best is None:
                raise ValueError('local RWKV vocabulary cannot encode input')
            offset, index = best
            output.append(index)
        return output
