import importlib.util
from pathlib import Path

path=Path(__file__).parents[1]/'reader-presentation-20260920/protocol.py'
spec=importlib.util.spec_from_file_location('frozen_presentation_protocol',path)
presentation=importlib.util.module_from_spec(spec)
spec.loader.exec_module(presentation)
metrics=presentation.metrics
parse=presentation.parse
parameters=presentation.parameters
prompt=presentation.prompt
