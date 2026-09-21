import asyncio
from llamaindex_retrieval.funnel_task import build_task


class Runner:
    def __init__(self, bad_revision=False):
        self.calls = []
        self.bad_revision = bad_revision

    async def node(self, prompt, purpose, schema, validator=None, **kwargs):
        self.calls.append((purpose, prompt))
        if purpose == "funnel_task_overview":
            value = {"mode":"selection", "objects":["甲","乙"], "fields":["能否离线运行"], "requested_count":1}
        elif purpose.startswith("funnel_field_type:"):
            value = {"value_type":"boolean"}
        elif purpose == "funnel_requirement_quotes":
            value = {"quotes":[{"message_id":"U1","quote":"必须支持Wiki"}]}
        else:
            value = {"state":"withdrawn", "kind":"hard", "meaning":"Wiki功能要求暂缓", "field_names":[],
                     "lifecycle_message_id":"U2", "lifecycle_quote":"不存在的原话" if self.bad_revision else "Wiki先不做"}
        try:
            parsed = schema.model_validate(value)
            return validator(parsed) if validator else parsed.model_dump()
        except ValueError:
            return None


def test_task_funnel_separates_type_and_requirement_revision_and_binds_both_quotes():
    runner = Runner()
    task = asyncio.run(build_task(runner, {"U1":"必须支持Wiki", "U2":"Wiki先不做，只要离线运行"}))
    assert task["fields"][0]["value_type"] == "boolean"
    requirement = task["requirements"][0]
    assert requirement["state"] == "withdrawn"
    assert requirement["quote"] == "必须支持Wiki" and requirement["lifecycle_quote"] == "Wiki先不做"
    assert len(runner.calls) == 4


def test_fabricated_revision_fails_task_without_replacing_the_condition():
    assert asyncio.run(build_task(Runner(True), {"U1":"必须支持Wiki", "U2":"Wiki先不做"})) is None
