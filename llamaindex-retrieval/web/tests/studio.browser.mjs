// Real browser checks with synthetic HTTP fixtures only; no model or web calls.
import assert from "node:assert/strict";
import http from "node:http";
import fs from "node:fs/promises";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const require = createRequire(
  process.env.PLAYWRIGHT_MODULE_DIR
    ? path.resolve(process.env.PLAYWRIGHT_MODULE_DIR, "package.json")
    : import.meta.url,
);
const { chromium } = require("playwright-core");
const assets = fileURLToPath(
  new URL("../../src/llamaindex_retrieval/static/studio/", import.meta.url),
);
const original = "原始答案 <img src=x onerror=alert(1)>\n· 保留原始措辞。[1]";
const source = {
  id: 'source"1',
  title: "Synthetic original",
  source: "knowledge_base",
  uri: "javascript:alert(1)",
  snippet: "😀原文证据。",
  metadata: {},
};
const view = { body: source.snippet, body_sha256: "view-hash", snippet: "" };
const span = {
  id: "span-1",
  source_id: source.id,
  section: "body",
  span: { start: 1, end: 5, view_sha256: "view-hash" },
};
const result = {
  display_text: original,
  errors: [],
  graph: {
    source_snapshots: [source],
    reading_views: { [source.id]: view },
    spans: [span],
  },
  answer: {
    generation: {
      raw_model_answer: original,
      citation_bindings: { 1: [source.id] },
      parsed_answer: { source_ids: ["1"] },
    },
  },
};
const requests = [],
  jobs = new Map(),
  errors = [];
let state = { web: true, ready: true, events: true },
  next = 0,
  finishAfterRequests = 0;
const server = http.createServer(async (req, res) => {
  try {
    const url = new URL(req.url, "http://localhost");
    const json = (value) => {
      res.setHeader("Content-Type", "application/json");
      res.end(JSON.stringify(value));
    };
    if (url.pathname.startsWith("/studio/")) {
      const name = url.pathname.slice("/studio/".length) || "index.html";
      if (!["index.html", "style.css", "app.js"].includes(name)) {
        res.writeHead(404);
        res.end();
        return;
      }
      res.setHeader(
        "Content-Type",
        name.endsWith(".js")
          ? "text/javascript"
          : name.endsWith(".css")
            ? "text/css"
            : "text/html",
      );
      res.end(await fs.readFile(path.join(assets, name)));
      return;
    }
    if (url.pathname === "/v1/try/meta") {
      json({
        knowledge_bases: [{ id: "kb-1", name: "Selected knowledge base" }],
        default_knowledge_base_ids: ["kb-1"],
        live_progress: state.events,
        model_readiness: state.ready ? "available" : "unavailable",
        web_configured: state.web,
      });
      return;
    }
    if (url.pathname === "/v1/try/ask") {
      const chunks = [];
      for await (const chunk of req) chunks.push(chunk);
      const body = JSON.parse(Buffer.concat(chunks));
      requests.push(body);
      const id = String(++next).padStart(32, "0");
      jobs.set(id, { body, polls: 0 });
      json({
        id,
        question: body.question,
        status: "queued",
        phase: "等待处理",
      });
      return;
    }
    const id = url.pathname.split("/").at(-1),
      job = jobs.get(id);
    if (job) {
      job.polls++;
      const finished = job.polls > 1 && requests.length >= finishAfterRequests,
        failed = job.body.question === "fail";
      const events = state.events
        ? [
            { kind: "call_started", stage: "plan", operation: "p1" },
            {
              kind: "call_finished",
              stage: "plan",
              operation: "p1",
              status: "completed",
              valid_output: true,
            },
            { kind: "plan_ready", queries: ["synthetic query"] },
            {
              kind: "source_ready",
              source,
              reading_view: view,
              windows: [[0, 7]],
            },
            { kind: "call_started", stage: "read", operation: "r1" },
            ...(finished
              ? [
                  {
                    kind: "call_finished",
                    stage: "read",
                    operation: "r1",
                    status: failed ? "cancelled" : "completed",
                    valid_output: !failed,
                  },
                  { kind: "reading_finished" },
                  ...(!failed
                    ? [
                        {
                          kind: "binding_finished",
                          bound_quotes: 1,
                          spans: [span],
                        },
                        {
                          kind: "call_started",
                          stage: "final",
                          operation: "f1",
                        },
                        {
                          kind: "call_finished",
                          stage: "final",
                          operation: "f1",
                          status: "completed",
                          valid_output: true,
                        },
                      ]
                    : []),
                ]
              : []),
          ]
        : [];
      json({
        id,
        status: finished ? (failed ? "cancelled" : "completed") : "running",
        phase: finished ? (failed ? "已取消" : "回答完成") : "阅读资料",
        execution_events: events,
        ...(finished && !failed ? { result } : {}),
      });
      return;
    }
    res.writeHead(404);
    res.end();
  } catch (e) {
    errors.push(String(e));
    res.writeHead(500);
    res.end();
  }
});
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const browser = await chromium.launch({
  headless: true,
  ...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE
    ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE }
    : {}),
  args: ["--no-sandbox"],
});
const context = await browser.newContext({
  viewport: { width: 1920, height: 1080 },
});
const url = `http://127.0.0.1:${server.address().port}/studio/`;
const page = await context.newPage();
page.on("pageerror", (e) => errors.push(String(e)));
async function open() {
  await page.goto(url);
  await page.waitForFunction(
    () => document.querySelector("#readiness").textContent !== "连接中",
  );
}
async function submit(text) {
  await page.locator("#question").fill(text);
  await page.locator("#send").click();
  await page.waitForFunction(() => !document.querySelector("#send").disabled);
}
try {
  await open();
  await page.locator("#domains").fill("EXAMPLE.org., sub.example.org");
  await submit("first\nsecond");
  assert.equal(requests.length, 2);
  assert.deepEqual(requests[0].web_include_domains, [
    "example.org",
    "sub.example.org",
  ]);
  assert.deepEqual(requests[1].knowledge_base_ids, ["kb-1"]);
  assert.equal(await page.locator("#runs .run").count(), 2);
  assert.equal(await page.locator("#answer").textContent(), original);
  assert.equal(await page.locator("#answer img").count(), 0);
  assert.equal(await page.locator("#sources a[href]").count(), 0);
  assert.equal(
    await page.locator("#sources mark").first().textContent(),
    "原文证据",
  );
  await page.locator("#citations button").click(); // Handles quotes in source IDs safely.
  await page.locator("#runs .run").nth(1).click();
  assert.equal(await page.locator("#detail-question").textContent(), "second");
  await submit("third");
  assert.equal(requests.length, 3); // A completed batch does not lock the input.
  await page.locator("#domains").fill("example.org OR evil.org");
  await page.locator("#send").click();
  assert.equal(requests.length, 3);
  await page.locator("#domains").fill("");
  await submit("fail");
  assert.equal((await page.locator("#steps .bad").count()) > 0, true);
  assert.equal(
    await page.locator("#answer").textContent(),
    "本次请求未完成，保留实际状态。",
  );
  console.log(
    "PASS parallel requests, scopes, unchanged text, source binding, repeat batch and failure states",
  );

  // Completion is held until all 24 submissions arrive: a serial implementation
  // cannot finish this batch. Fixtures prove UI behavior, not model throughput.
  const beforeBatch = requests.length;
  finishAfterRequests = beforeBatch + 24;
  await submit(Array.from({ length: 24 }, (_, i) => `比较项目 ${i + 1} 的安装条件、能力与适用边界`).join("\n"));
  assert.equal(requests.length, finishAfterRequests);
  assert.equal(new Set(await page.locator("#runs .run").evaluateAll(
    (cards) => cards.map((card) => card.dataset.jobId),
  )).size, 24);
  assert.equal(await page.locator("body.overview").count(), 1);
  assert.equal(await page.locator(".detail").isVisible(), false);
  assert.equal(await page.locator("#runs .run").count(), 24);
  await page.setViewportSize({ width: 3840, height: 2160 });
  assert.equal(await page.locator("#runs").evaluate((grid) =>
    getComputedStyle(grid).gridTemplateColumns.split(" ").length), 6);
  assert.equal(await page.evaluate(() =>
    document.querySelector("#runs .run:last-child").getBoundingClientRect().bottom <= innerHeight), true);
  if (process.env.STUDIO_SCREENSHOT_PATH)
    await page.screenshot({ path: process.env.STUDIO_SCREENSHOT_PATH, fullPage: true });
  await page.locator("#runs .run").nth(23).click();
  assert.equal(await page.locator("body.overview").count(), 0);
  assert.match(await page.locator("#detail-question").textContent(), /项目 24/);
  assert.equal(await page.locator("#answer").textContent(), original);
  await page.locator("#overview").click();
  await page.setViewportSize({ width: 390, height: 844 });
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), true);
  await page.setViewportSize({ width: 1920, height: 1080 });
  await page.locator("#edit-input").click();
  await page.locator("#question").fill(Array(33).fill("超量问题").join("\n"));
  await page.locator("#send").click();
  assert.equal(requests.length, finishAfterRequests);
  assert.equal(await page.locator("#phase").textContent(), "请输入 1 至 32 行问题");
  finishAfterRequests = 0;
  console.log("PASS 24 concurrent submissions, 4K overview, full answer navigation, mobile layout and batch limit");

  state = { web: false, ready: true, events: false };
  await open();
  assert.equal(await page.locator("#send").isDisabled(), true);
  await page.locator("#retrieval-mode").selectOption("knowledge_base");
  await submit("knowledge only");
  assert.equal(await page.locator("#answer").textContent(), original);
  assert.equal(
    await page.locator("#sources mark").first().textContent(),
    "原文证据",
  );
  assert.equal(await page.locator("#steps .done").count(), 0);
  await page.setViewportSize({ width: 390, height: 844 });
  assert.equal(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth + 1,
    ),
    true,
  );
  console.log(
    "PASS knowledge-only readiness, snapshot fallback, no invented progress and mobile width",
  );

  state.ready = false;
  await open();
  assert.equal(await page.locator("#send").isDisabled(), true);
  assert.deepEqual(errors, []);
  console.log(
    "PASS unavailable model remains disabled; no browser or fixture errors",
  );
} finally {
  await browser.close();
  server.close();
}
