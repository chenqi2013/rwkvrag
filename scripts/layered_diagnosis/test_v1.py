"""Focused scorer/provenance tests for wrong entity and unreviewed Writer output."""
import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load('prepare_v1')
score = load('score_v1')


class DiagnosticTests(unittest.TestCase):
    def test_all_gold_decisions_consistent_with_source_status(self):
        prepare.validate(json.loads(prepare.SCENARIOS.read_text()))

    def test_swapped_project_is_error_even_with_same_value(self):
        gold = {'facts': [{'project': 'A', 'attribute': 'windows', 'value': 'native',
                           'source_id': 'S1', 'status': 'current'}]}
        wrong = {'facts': [{'project': 'B', 'attribute': 'windows', 'value': 'native',
                            'source_id': 'S1', 'status': 'current'}]}
        result = score.score_extract(json.dumps(wrong), gold)
        self.assertEqual((result['tp'], result['fp'], result['fn']), (0, 1, 1))
        self.assertEqual(result['swap_hints']['project'], 1)

    def test_duplicate_fact_rejected_instead_of_inflating_precision(self):
        item = {'project': 'A', 'attribute': 'windows', 'value': 'native',
                'source_id': 'S1', 'status': 'current'}
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            score.score_extract(json.dumps({'facts': [item, item]}), {'facts': [item]})

    def test_unknown_and_rejected_are_not_interchangeable(self):
        gold = {'eligible': ['A'], 'rejected': ['B'], 'unresolved': ['C'], 'recommended': ['A']}
        wrong = {'eligible': ['A'], 'rejected': ['B', 'C'], 'unresolved': [], 'recommended': ['A']}
        result = score.score_compare(json.dumps(wrong), gold, ['A', 'B', 'C'])
        self.assertFalse(result['exact'])
        self.assertEqual(result['status_correct'], 2)

    def test_writer_mechanics_never_claim_semantic_support(self):
        result = score.score_writer('推荐甲。[资料 9]', ['资料 1'], ['甲', '乙'], 'stop')
        self.assertEqual(result['invalid_citations'], ['资料 9'])
        self.assertEqual(result['missing_project_mentions'], ['乙'])
        self.assertFalse(result['semantic_support_reviewed'])


if __name__ == '__main__':
    unittest.main()
