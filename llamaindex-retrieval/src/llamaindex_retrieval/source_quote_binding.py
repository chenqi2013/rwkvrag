"""Bind model-selected literal quotes to the supplied source, with ID hints audited.

The quote is the primary selector in this protocol. Unit IDs are localization
hints, not an alternative source of text. Ambiguous occurrences require a
unique hinted location. Nothing edits the raw proposal or any final answer.
"""
import re
from .typed_funnel_contract import unique_subset


def bind_atomic(parsed, field, source, units):
    unique_subset(parsed.evidence_ids, units)
    quote = parsed.quote
    location = None
    hints_match = False
    if quote is not None:
        if not quote:
            raise ValueError('empty literal quote')
        occurrences = [m.start() for m in re.finditer('(?=' + re.escape(quote) + ')', source.snippet)]
        if not occurrences:
            raise ValueError('quote not verbatim in supplied source')
        hinted = [start for start in occurrences if any(
            units[i].start <= start and start + len(quote) <= units[i].end for i in parsed.evidence_ids)]
        if len(occurrences) == 1:
            location = occurrences[0]
        elif len(hinted) == 1:
            location = hinted[0]
        else:
            raise ValueError('ambiguous quote occurrence')
        # Every character must have appeared in the actual model input units.
        cursor = location
        for unit in sorted(units.values(), key=lambda item: item.start):
            if unit.start <= cursor < unit.end:
                cursor = unit.end
            if cursor >= location + len(quote):
                break
        if cursor < location + len(quote):
            raise ValueError('quote contains characters not supplied to the model')
        hints_match = location in hinted
    if parsed.source_scope is not None and (not parsed.source_scope.strip() or parsed.source_scope not in source.snippet):
        raise ValueError('scope not verbatim in supplied source')
    value, unit = parsed.value, None
    if value is not None:
        if quote is None:
            raise ValueError('known value needs a literal quote')
        if field['value_type'] == 'boolean':
            if type(value) is not bool:
                raise ValueError('boolean field requires boolean value')
        elif not isinstance(value, str) or not value.strip():
            raise ValueError('text/quantity requires a nonempty string')
        elif field['value_type'] == 'quantity':
            if value not in quote:
                raise ValueError('quantity must preserve its source spelling')
            match = re.fullmatch(r'([+-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+))\s*([^0-9\s].*)?', value)
            if not match:
                raise ValueError('invalid decimal scalar representation')
            value, unit = match.group(1), match.group(2)
    return {'observed': parsed.value is not None, 'unit_ids': parsed.evidence_ids,
            'quote': quote, 'value': value, 'unit': unit, 'scope': parsed.source_scope,
            'source_start': location, 'source_end': None if location is None else location + len(quote),
            'quote_binding': {'policy': 'literal_quote_primary_v1', 'unit_hints_match': hints_match,
                              'hint_ids': parsed.evidence_ids}}
