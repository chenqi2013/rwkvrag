"""Single-variable ablation: change only output label words in the instruction."""
import importlib.util
import json
from pathlib import Path

BASE = Path(__file__).parents[1] / "evidence-support-20260920/protocol.py"
spec = importlib.util.spec_from_file_location("frozen_support_protocol", BASE)
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)
metrics = baseline.metrics
validate_cases = baseline.validate_cases


def prompt(case, protocol):
    original = baseline.prompt(case, "strict-v1")
    if protocol == "yes-no":
        return original
    if protocol != "neutral-labels":
        raise ValueError("unknown protocol")
    instruction, separator, data = original.partition("\n")
    assert separator and instruction.count("输出NO。") == 1
    assert instruction.count('"YES"') == 1 and instruction.count('"NO"') == 1
    instruction = instruction.replace("输出NO。", "输出INSUFFICIENT。")
    instruction = instruction.replace('"YES"', '"ANSWERABLE"').replace('"NO"', '"INSUFFICIENT"')
    return instruction + separator + data  # Evidence/query bytes are untouched.


def parse_label(text, protocol):
    value = json.loads(text, object_pairs_hook=list)
    labels = {"yes-no": ("YES", "NO"), "neutral-labels": ("ANSWERABLE", "INSUFFICIENT")}[protocol]
    if value == [("answer", labels[0])]:
        return True
    if value == [("answer", labels[1])]:
        return False
    raise ValueError("Output must contain exactly one answer key and the preregistered label")
