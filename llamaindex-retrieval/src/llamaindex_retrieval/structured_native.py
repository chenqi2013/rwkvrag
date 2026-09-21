"""Opt-in native JSON-schema transport; grammar constrains format, not meaning."""
from contextvars import ContextVar
from copy import deepcopy
from .native_rwkv import NativeRWKVClient

_schema = ContextVar('rwkv_completion_schema', default=None)


class StructuredNativeRWKVClient(NativeRWKVClient):
    async def complete(self, messages, *, structured_schema=None, **kwargs):
        binding = _schema.set(deepcopy(structured_schema))
        try:
            result = await super().complete(messages, **kwargs)
            if structured_schema is not None:
                result.trace.setdefault('parameters', {})['structured_outputs'] = {'json': deepcopy(structured_schema)}
            return result
        finally:
            _schema.reset(binding)

    async def _post(self, url, payload, stage, trace):
        schema = _schema.get()
        if stage == 'completion' and schema is not None:
            payload = {**payload, 'structured_outputs': {'json': deepcopy(schema)}}
            trace['parameters']['structured_outputs'] = deepcopy(payload['structured_outputs'])
        return await super()._post(url, payload, stage, trace)
