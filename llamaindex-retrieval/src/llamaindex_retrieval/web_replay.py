"""Immutable, request-keyed provider captures. Replay never invokes a transport."""
import base64
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .direct_web import WebProviderError


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode('utf-8')


def sha(value):
    return hashlib.sha256(value).hexdigest()


class WebReplay:
    def __init__(self, *, mode='off', store=None):
        if mode not in {'off', 'record', 'replay'} or (mode != 'off' and not store):
            raise ValueError('invalid web_replay configuration')
        self.mode = mode
        self.store = Path(store) if store else None

    @staticmethod
    def key(provider, query, include_domains, max_results, full_snapshot, *, provider_options=None):
        # No query normalization or domain reordering: these are exact requests.
        value = {'provider': provider, 'query': query, 'include_domains': include_domains,
                 'max_results': max_results, 'full_snapshot': full_snapshot}
        if provider_options is not None:
            value['provider_options'] = provider_options
        return value

    def load(self, key):
        path = self.store / (sha(encode(key)) + '.json')
        try:
            envelope = json.loads(path.read_bytes())
            raw = base64.b64decode(envelope['payload_base64'], validate=True)
            if (envelope['format'] != 'web-replay-v1' or envelope['key'] != key
                    or envelope['payload_sha256'] != sha(raw)):
                raise ValueError('invalid capture')
            payload = json.loads(raw)
            if not isinstance(payload, dict) or payload['provider'] != key['provider']:
                raise ValueError('invalid payload')
            return payload, {'mode': self.mode, 'key_sha256': path.stem,
                             'payload_sha256': sha(raw), 'replayed': True}
        except FileNotFoundError:
            raise WebProviderError('web_replay_miss') from None
        except (ValueError, KeyError, TypeError):
            raise WebProviderError('web_replay_corrupt') from None
        except OSError:
            raise WebProviderError('web_replay_unavailable') from None

    def save(self, key, payload):
        raw = encode(payload)
        envelope = encode({'format': 'web-replay-v1', 'key': key,
                          'payload_sha256': sha(raw), 'payload_base64': base64.b64encode(raw).decode('ascii')})
        try:
            self.store.mkdir(parents=True, exist_ok=True, mode=0o700)
            destination = self.store / (sha(encode(key)) + '.json')
            # Link a completely written temporary file; never expose partial
            # captures or overwrite the first observation of an exact request.
            with tempfile.NamedTemporaryFile(dir=self.store) as temp:
                temp.write(envelope)
                temp.flush()
                os.fsync(temp.fileno())
                try:
                    os.link(temp.name, destination)
                except FileExistsError:
                    _, existing = self.load(key)
                    if existing['payload_sha256'] != sha(raw):
                        raise WebProviderError('web_replay_conflict')
            return {'mode': self.mode, 'key_sha256': destination.stem,
                    'payload_sha256': sha(raw), 'replayed': False}
        except OSError:
            raise WebProviderError('web_replay_unavailable') from None

    async def execute(self, key, live):
        if self.mode == 'off':
            return await live(), None
        try:
            return self.load(key)
        except WebProviderError as error:
            if self.mode == 'replay' or error.safe_code != 'web_replay_miss':
                raise
        payload = await live()
        return payload, self.save(key, payload)
