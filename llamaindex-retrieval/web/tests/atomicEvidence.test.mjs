import assert from "node:assert/strict";
import test from "node:test";
import { exactSpan } from "../src/atomicEvidence.ts";

test("source offsets count Unicode code points, including emoji", () => {
  assert.deepEqual(exactSpan("😀甲功率18 W。", 4, 8, "18 W"), { before: "😀甲功率", quote: "18 W", after: "。" });
});
test("changed source or invalid offsets never highlight different evidence", () => {
  assert.equal(exactSpan("甲25 W", 1, 5, "18 W"), undefined);
  for (const [start, end] of [[-1, 4], [0, 99], [2, 2], [0.5, 3]]) {
    assert.equal(exactSpan("18 W", start, end, "18 W"), undefined);
  }
});
