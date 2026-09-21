import json
from pathlib import Path
import runpy
import pytest

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "deploy/rwkv-fp32io16"
command = runpy.run_path(str(PROFILE / "launch.py"))["command"]


def test_fp32io16_means_fp32_state_with_fp16_io_not_fp16_state():
    profile = json.loads((PROFILE / "PROFILE.json").read_text())
    args = command(profile)
    assert args[args.index("--dtype") + 1] == "float16"
    assert args[args.index("--mamba-ssm-cache-dtype") + 1] == "float32"
    assert args.count("--mamba-ssm-cache-dtype") == 1


@pytest.mark.parametrize("field,value", [("recurrent_state_dtype", "float16"),
    ("kernel_mode", "fp16"), ("weights_io_dtype", "float32")])
def test_incompatible_precision_never_silently_falls_back(field, value):
    profile = json.loads((PROFILE / "PROFILE.json").read_text())
    profile[field] = value
    with pytest.raises(ValueError, match="requires FlashRWKV2 fp32io16"):
        command(profile)
