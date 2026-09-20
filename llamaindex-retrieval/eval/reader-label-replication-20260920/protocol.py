"""Import the unchanged label-only intervention; no new semantic instructions."""
import importlib.util
from pathlib import Path

BASE = Path(__file__).parents[1] / "reader-label-20260920/protocol.py"
spec = importlib.util.spec_from_file_location("frozen_label_protocol", BASE)
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)
prompt = baseline.prompt
parse_label = baseline.parse_label
metrics = baseline.metrics
validate_cases = baseline.validate_cases
