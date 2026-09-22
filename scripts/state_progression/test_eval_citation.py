"""Regression for frozen cases that preserve nonconsecutive source labels."""
import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    'paired_diagnostics', Path(__file__).with_name('analyze_eval_v2.py'))
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


def test_single_source_preserves_original_source_number():
    source = [{'label': '资料 4', 'text': 'The project is an HTTP client.'}]
    assert diagnostics.cite_issue('客户端[资料 4]', source)['invalid_citations'] == []
    assert diagnostics.cite_issue('客户端[资料 1]', source)['invalid_citations'] == ['[资料 1]']


def test_unlabeled_historical_sources_keep_list_position():
    source = [{'text': 'one'}, {'text': 'two'}]
    assert diagnostics.cite_issue('甲[资料 2]', source)['invalid_citations'] == []
