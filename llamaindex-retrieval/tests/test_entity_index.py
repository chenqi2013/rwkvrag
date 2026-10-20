import pytest

from llamaindex_retrieval.entity_index import EntityIndex, normalize, normalization_manifest


def row(fid, entities, aliases=()):
    return {'id': fid, 'title': fid, 'status': 'completed',
            'parsed': {'entities': list(entities), 'aliases': list(aliases)}}


def test_normalization_is_character_only_and_structural():
    assert normalize(' ＡＢＣ—臺灣 ') == normalize('abc台湾') == 'abc台湾'
    assert normalize('乾坤') == normalize('干坤')
    assert normalize('C++') != normalize('C')
    assert normalization_manifest()['conversion_unit'] == 'one Unicode codepoint'


def test_lookup_requires_complete_normalized_equality_without_fuzzy_or_overlap_rank():
    index = EntityIndex([row('a', ['New York']), row('b', ['New York', 'NYC']),
                         row('c', ['New York City'])])
    assert index.lookup(['York'])[0] == []
    assert index.lookup(['new-york'])[0] == ['a', 'b']
    ids, receipt = index.lookup(['NYC', 'New York'], ['ＮＹＣ'])
    assert ids == ['a', 'b']
    assert receipt['exact_matches']['nyc'] == ['b']
    assert index.lookup(['New York'], allowed_ids=['b', 'c'])[0] == ['b']


def test_invalid_or_failed_index_record_is_not_invented_or_repaired():
    failed = {**row('a', ['X']), 'status': 'length', 'raw_text': 'immutable raw'}
    invalid = {**row('b', ['X']), 'parsed': {'entities': 'X', 'aliases': []}}
    index = EntityIndex([failed, invalid])
    assert index.lookup(['X'])[0] == []
    assert failed['raw_text'] == 'immutable raw'
    with pytest.raises(ValueError, match='duplicate'):
        EntityIndex([row('a', ['X']), row('a', ['Y'])])
