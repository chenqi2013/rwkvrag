"""Decoding contracts for the ordinary pipeline, not semantic decision rules.

Schemas mirror the existing parsers. They contain no answers, entity lists or
reference evidence. Call sites select a purpose explicitly: history and query
planning share a transport stage but must not share a JSON contract.
"""


def json_output_options(settings, purpose: str) -> dict:
    if settings.native_json_constraints == "disabled":
        return {}
    if settings.native_json_constraints != "schema_v1":
        raise ValueError("unsupported native JSON constraint profile")
    if purpose == "current_question":
        properties = {"question": _text(8000)}
    elif purpose == "plan":
        properties = {"queries": {
            "type": "array", "minItems": 1, "maxItems": settings.native_max_queries,
            "items": _text(2000),
        }}
    elif purpose == "reader":
        properties = {"answer": {"type": "string", "enum": ["YES", "NO"]}}
    else:
        raise ValueError(f"unsupported native JSON purpose: {purpose}")
    return {"structured_schema": {
        "type": "object", "properties": properties,
        "required": list(properties), "additionalProperties": False,
    }}


def _text(max_length: int) -> dict:
    return {"type": "string", "minLength": 1, "maxLength": max_length, "pattern": r"\S"}
