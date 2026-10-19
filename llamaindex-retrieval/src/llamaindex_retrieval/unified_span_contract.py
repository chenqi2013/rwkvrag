"""Call-local ID choices guide generation; they do not establish factual support."""
from copy import deepcopy

from .unified_span_reading import SpanAnswer, SpanReading


def bound_span_schema(contract, allowed):
    """Keep the existing shape, binding only references to this call's namespace."""
    schema = deepcopy(contract.model_json_schema())
    ids = sorted(set(allowed))
    if contract is SpanReading:
        if ids:
            schema["$defs"]["SpanFinding"]["properties"]["span_id"]["enum"] = ids
        else:
            schema["properties"]["findings"]["maxItems"] = 0
    elif contract is SpanAnswer:
        if ids:
            schema["properties"]["span_ids"]["items"]["enum"] = ids
        else:
            schema["properties"]["span_ids"]["maxItems"] = 0
    else:
        raise ValueError("unsupported_span_contract")
    return schema
