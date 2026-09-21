from types import SimpleNamespace
import pytest
from llamaindex_retrieval.typed_funnel_contract_v7 import Atomic
from llamaindex_retrieval.source_quote_binding import bind_atomic


def atom(quote, ids):
    return Atomic(evidence_ids=ids, quote=quote, value='18 W', source_scope=None)


def test_unique_literal_quote_is_bound_without_rewriting_wrong_unit_hints():
    source = SimpleNamespace(snippet='背景说明。额定功率18 W。')
    units = {'E1': SimpleNamespace(start=0,end=5,text=source.snippet[:5]),
             'E2': SimpleNamespace(start=5,end=len(source.snippet),text=source.snippet[5:])}
    proposal = atom('额定功率18 W。', ['E1'])
    result = bind_atomic(proposal, {'value_type':'quantity'}, source, units)
    assert result['source_start'] == 5 and result['unit_ids'] == ['E1']
    assert result['quote_binding']['unit_hints_match'] is False
    assert proposal.evidence_ids == ['E1']
    assert result['value'] == '18' and result['unit'] == 'W'


def test_quote_can_span_contiguous_supplied_units_but_never_unseen_characters():
    source = SimpleNamespace(snippet='额定功率18 W。')
    units = {'E1':SimpleNamespace(start=0,end=5,text=source.snippet[:5]),
             'E2':SimpleNamespace(start=5,end=len(source.snippet),text=source.snippet[5:])}
    assert bind_atomic(atom(source.snippet,['E1','E2']), {'value_type':'quantity'},source,units)['source_start']==0
    units['E2'].start = 6
    with pytest.raises(ValueError):
        bind_atomic(atom(source.snippet,['E1','E2']), {'value_type':'quantity'},source,units)


def test_duplicate_quote_requires_unambiguous_hint_and_never_chooses_first_silently():
    source = SimpleNamespace(snippet='18 W；18 W')
    units = {'E1':SimpleNamespace(start=0,end=4,text='18 W'), 'E2':SimpleNamespace(start=5,end=9,text='18 W')}
    with pytest.raises(ValueError):
        bind_atomic(atom('18 W',[]), {'value_type':'quantity'},source,units)
    result = bind_atomic(atom('18 W',['E2']), {'value_type':'quantity'},source,units)
    assert result['source_start'] == 5
