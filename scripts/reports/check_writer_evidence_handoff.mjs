import assert from "node:assert/strict";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const name = "writer-evidence-handoff-20260921";
const data = JSON.parse(await fs.readFile(path.join(root, `llamaindex-retrieval/web/public/experiments/${name}.json`), "utf8"));
const out = path.join(root, `data/quality-runs/${name}/browser`);
await fs.mkdir(out, { recursive: true });
const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
  const errors = [];
  page.on("pageerror", e => errors.push(String(e)));
  await page.goto("http://127.0.0.1:18446/admin/#/test-results");
  await page.getByText(data.title, { exact: true }).waitFor();
  let checked = 0;
  for (let i = 0; i < data.cases.length; i++) {
    const c = data.cases[i];
    await page.getByText(c.id, { exact: true }).waitFor();
    const expected = c.answers.slice().sort((a, b) => Number(b.role === "candidate") - Number(a.role === "candidate"));
    assert.deepEqual(await page.getByTestId("raw-answer").allTextContents(), expected.map(a => a.raw_text));
    checked += expected.length;
    if (i === 0) {
      const first = page.locator(".ant-card").filter({ has: page.getByTestId("raw-answer") }).first();
      const citation = first.locator(".comparison-citation").first();
      const id = (await citation.textContent()).match(/\d+/)[0];
      const source = expected[0].sources.find(s => s.label.replace(/\s/g, "") === `资料${Number(id)}`);
      assert.ok(source);
      await citation.click();
      await page.locator(".comparison-source").waitFor();
      assert.equal(await page.locator(".comparison-source").textContent(), source.text);
      await page.screenshot({ path: path.join(out, "citation.png") });
      await page.getByRole("button", { name: "Close", exact: true }).click();
    }
    if (i === 20) await page.screenshot({ path: path.join(out, "capacity-failure.png"), fullPage: true });
    if (i < data.cases.length - 1) await page.getByRole("button", { name: "下一题", exact: true }).click();
  }
  assert.deepEqual(errors, []);
  await fs.writeFile(path.join(out, "BROWSER-CHECK.json"), JSON.stringify({
    questions: data.cases.length, raw_answers_checked: checked,
    raw_text_unchanged: true, paired_rounds_visible: true, citation_drawer_exact: true,
    page_errors: errors,
  }, null, 2), { flag: "wx" });
  console.log({ questions: data.cases.length, raw_answers_checked: checked, page_errors: errors });
} finally {
  await browser.close();
}
