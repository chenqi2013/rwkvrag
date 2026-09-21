from llamaindex_retrieval.funnel_task_v3 import user_clauses


def test_user_clause_offsets_are_exact_and_do_not_split_version_decimals():
    messages = {'U1':'比较白榆1.0和2.0，容量、离线支持都要。', 'U2':' Wiki先不做；资料不能传出去。'}
    clauses = list(user_clauses(messages))
    assert all(messages[c['message_id']][c['start']:c['end']] == c['quote'] for c in clauses)
    assert clauses[0]['quote'] == '比较白榆1.0和2.0，'
    assert any(c['quote'] == 'Wiki先不做；' for c in clauses)
    assert all('role' not in c and 'state' not in c for c in clauses)
