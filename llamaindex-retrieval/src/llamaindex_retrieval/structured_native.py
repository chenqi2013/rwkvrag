"""Opt-in native JSON-schema transport; grammar constrains format, not meaning."""
from contextvars import ContextVar
from copy import deepcopy
from .native_rwkv import NativeRWKVClient

_schema = ContextVar('rwkv_completion_schema', default=None)


def json_schema(schema, *, allow_delimiter=False):
    """Constrain accidental chat markers, preserving literals and user-supplied text."""
    result = deepcopy(schema)

    def visit(node):
        if isinstance(node, dict):
            for key in ('properties', 'patternProperties', '$defs', 'definitions', 'dependentSchemas'):
                for child in node.get(key, {}).values():
                    visit(child)
            for key in ('items', 'prefixItems', 'additionalProperties', 'contains', 'propertyNames',
                        'not', 'if', 'then', 'else', 'allOf', 'anyOf', 'oneOf'):
                visit(node.get(key))
            kind = node.get('type')
            is_string = kind == 'string' or isinstance(kind, list) and 'string' in kind
            if is_string and not allow_delimiter and 'enum' not in node and 'const' not in node:
                if 'pattern' in node:
                    node.setdefault('allOf', []).append({'pattern': node.pop('pattern')})
                node['pattern'] = '^[^✿]*$'
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(result)
    return result


class StructuredNativeRWKVClient(NativeRWKVClient):
    async def complete(self, messages, *, structured_schema=None, **kwargs):
        allow_delimiter = isinstance(messages, list) and any(
            isinstance(message, dict) and isinstance(message.get('content'), str)
            and '✿' in message['content'] for message in messages)
        schema = json_schema(structured_schema, allow_delimiter=allow_delimiter) if structured_schema is not None else None
        binding = _schema.set(schema)
        try:
            result = await super().complete(messages, **kwargs)
            if structured_schema is not None:
                result.trace.setdefault('parameters', {})['structured_outputs'] = {'json': deepcopy(schema)}
            return result
        finally:
            _schema.reset(binding)

    async def _post(self, url, payload, stage, trace):
        schema = _schema.get()
        if stage == 'completion' and schema is not None:
            payload = {**payload, 'stop': [], 'structured_outputs': {'json': deepcopy(schema)}}
            trace['parameters']['stop'] = []
            trace['parameters']['structured_outputs'] = deepcopy(payload['structured_outputs'])
            trace['structured_stop_policy'] = 'grammar_eos_without_chat_delimiter'
        return await super()._post(url, payload, stage, trace)
