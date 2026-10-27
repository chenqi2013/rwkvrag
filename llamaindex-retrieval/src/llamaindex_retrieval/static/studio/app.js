const $ = (s) => document.querySelector(s),
  el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const MAX_QUESTIONS = 32;
const sceneInfo = {
  two: [
    "ORIGINAL SOURCES · SIDE BY SIDE",
    "资料与问题，一起读懂。",
    "并排查看知识库与网页原文，沿着引用追溯回答。",
  ],
  web: [
    "WEB SEARCH · ORIGINAL SOURCES",
    "连接网络，查阅新知。",
    "检索网页、阅读资料，并保留本次来源与原文。",
  ],
  parallel: [
    "PARALLEL QUESTIONS · INDEPENDENT ANSWERS",
    "多个问题，同时出发。",
    "每个问题独立检索、独立阅读、独立回答。",
  ],
};
let scene = new URLSearchParams(location.search).get("scene") || "parallel";
if (!Object.hasOwn(sceneInfo, scene)) scene = "parallel";
let info = sceneInfo[scene] || sceneInfo.two;
document.body.classList.add(scene);
$("#hero-kicker").textContent = info[0];
$("#hero-title").textContent = info[1];
$("#hero-description").textContent = info[2];
document.querySelector(`[data-scene="${scene}"]`)?.classList.add("active");
$("#send-label").textContent =
  scene === "web" ? "联网搜索" : scene === "parallel" ? "同时提问" : "开始阅读";
let runs = [],
  active = null,
  view = scene === "web" ? "web" : "knowledge_base",
  batch = null;
window.__rwkvLiveRuns = [];
window.__rwkvBatches = [];
async function api(path, body) {
  const r = await fetch(path, {
    method: body ? "POST" : "GET",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) throw Error(`HTTP ${r.status}`);
  return r.json();
}
function events(r) {
  return r?.job?.execution_events || [];
}
function sourceRows(r) {
  const observed = events(r)
    .filter((e) => e.kind === "source_ready")
    .map((e) => ({
      ...e.source,
      view: e.reading_view,
      windows: e.windows,
      received: true,
    }));
  if (observed.length) return observed;
  const graph = r?.job?.result?.graph;
  return (graph?.source_snapshots || []).map((s) => ({
    ...s,
    view: graph.reading_views?.[s.id],
    received: true,
  }));
}
function summaries(r) {
  const es = events(r),
    starts = es.filter((e) => e.kind === "call_started"),
    ends = es.filter((e) => e.kind === "call_finished"),
    reads = starts.filter((e) => e.stage === "read"),
    done = ends.filter((e) => e.stage === "read"),
    src = sourceRows(r);
  return {
    reads: reads.length,
    done: done.length,
    sources: r ? src.length : 0,
    quotes: Math.max(
      0,
      ...es
        .filter((e) => e.kind === "binding_finished")
        .map((e) => e.bound_quotes),
    ),
    active: starts.length - ends.length,
  };
}
function setOverview(enabled) {
  document.body.classList.toggle("overview", enabled);
  document.body.classList.remove("editing-input");
  $("#edit-input").setAttribute("aria-expanded", "false");
  $("#edit-input").textContent = "编辑问题与来源";
  $("#input-summary").textContent = runs.length
    ? `${runs.length} 个问题 · 点击问题查看完整答案与原文`
    : "选择来源，每行输入一个问题";
  $("#overview").setAttribute("aria-pressed", String(enabled));
  $("#overview").textContent = enabled ? "查看单题" : "并行总览";
}
function renderCards() {
  document.body.classList.toggle("has-runs", runs.length > 0);
  const box = $("#runs");
  box.replaceChildren();
  for (const r of runs) {
    const card = el(
      "button",
      "run" +
        (r === active ? " active" : "") +
        (r.finished ? " finished" : "") +
        (r.error || (r.finished && r.job?.status !== "completed") ? " failed" : ""),
    );
    card.dataset.jobId = r.job?.id || "";
    card.title = r.question;
    const head = el("div", "run-head");
    head.append(
      el("span", "", "问题 " + (runs.indexOf(r) + 1)),
      el(
        "span",
        "",
        r.finished
          ? r.job?.status === "completed"
            ? "已返回"
            : "未完成"
          : r.job?.status === "queued"
            ? "等待处理"
            : r.job ? "运行中" : "提交中",
      ),
    );
    card.append(head, el("h3", "", r.question));
    const scope = el("div", "run-scope", scopeDescription(r.request));
    scope.title = scope.textContent;
    card.append(scope);
    const stat = summaries(r);
    card.append(
      el(
        "div",
        "run-meta",
        `${r.job?.phase || "提交问题"} · ${stat.sources} 份来源\n阅读 ${stat.done}/${stat.reads} · ${stat.quotes} 处引用`,
      ),
    );
    card.append(
      el(
        "div",
        "run-time",
        (r.wall ?? (performance.now() - r.start) / 1000).toFixed(2) + " s",
      ),
    );
    const stages = el("div", "run-stages");
    for (const [name, status, note] of stateFor(r)) {
      const stage = el("span", status);
      stage.title = `${name}：${note}`;
      stage.setAttribute("aria-label", stage.title);
      stages.append(stage);
    }
    card.append(stages);
    card.onclick = () => {
      active = r;
      setOverview(false);
      view = r.request?.retrieval_mode === "web" ? "web" : "knowledge_base";
      renderCards();
      renderDetail();
    };
    box.append(card);
  }
  $("#batch-count").textContent =
    `${runs.filter((r) => r.finished).length} / ${runs.length} 已结束`;
}
function stateFor(r) {
  const es = events(r),
    has = (k) => es.some((e) => e.kind === k),
    calls = (s) => es.filter((e) => e.stage === s),
    cstate = (s) => {
      const cs = calls(s);
      if (
        cs.some(
          (e) =>
            e.kind === "call_started" &&
            !cs.some(
              (f) => f.kind === "call_finished" && f.operation === e.operation,
            ),
        )
      )
        return "active";
      if (
        cs.some(
          (e) =>
            e.kind === "call_finished" &&
            (e.status !== "completed" || !e.valid_output),
        )
      )
        return "bad";
      if (cs.some((e) => e.kind === "call_finished")) return "done";
      return cs.length ? "active" : "";
    },
    channel = (c) => {
      const a = es.filter(
          (e) => e.kind.startsWith("acquisition_") && e.channel === c,
        ),
        start = a.filter((e) => e.kind === "acquisition_started"),
        end = a.filter((e) => e.kind === "acquisition_finished");
      if (end.some((e) => e.status !== "completed" || e.file_errors?.length))
        return "bad";
      return start.length > end.length ? "active" : end.length ? "done" : "";
    },
    s = summaries(r),
    readEnds = calls("read").filter((e) => e.kind === "call_finished"),
    readFailed =
      readEnds.some((e) => e.status !== "completed" || !e.valid_output) ||
      (r?.job?.result?.errors || []).some(
        (e) => e.stage === "read" && e.status !== "windows_limited",
      ),
    ended = has("reading_finished") || r?.finished,
    readState = readFailed
      ? "bad"
      : ended
        ? s.reads && s.done === s.reads
          ? "done"
          : ""
        : s.reads
          ? "active"
          : "",
    readNote = readFailed
      ? "存在失败，保留状态"
      : !s.reads && ended
        ? "未执行 / 无可读来源"
        : `${s.done} / ${s.reads} 次已返回`;
  return [
    [
      "理解问题",
      cstate("plan"),
      has("plan_ready") ? "问题与信息项已生成" : "RWKV 规划",
    ],
    [
      "生成查询",
      has("plan_ready") ? "done" : cstate("plan"),
      has("plan_ready")
        ? `${(es.find((e) => e.kind === "plan_ready")?.queries || []).length} 条实际查询`
        : "与理解同次规划",
    ],
    [
      "知识库选材",
      channel("knowledge_base"),
      `${sourceRows(r).filter((x) => x.source === "knowledge_base").length} 份原文`,
    ],
    [
      "网络检索取材",
      channel("web"),
      `${sourceRows(r).filter((x) => x.source === "web").length} 个网页`,
    ],
    ["整理资料", has("source_ready") ? "done" : "", `${s.sources} 份进入阅读`],
    ["并行阅读", readState, readNote],
    [
      "引用定位",
      readFailed
        ? "bad"
        : has("reading_finished")
          ? s.quotes
            ? "done"
            : ""
          : has("binding_finished")
            ? "active"
            : "",
      s.quotes
        ? `${s.quotes} 处原文引句`
        : ended
          ? "未取得可绑定引用"
          : "等待阅读结果",
    ],
    [
      "生成回答",
      cstate("final"),
      cstate("final") === "done"
        ? "本次回答已返回"
        : ended && !calls("final").length
          ? "未执行 / 无直接证据"
          : "汇集选中证据",
    ],
    ...["calc", "review"]
      .filter((s) => calls(s).length)
      .map((s) => [
        s === "calc" ? "计算" : "回答核对",
        cstate(s),
        "本次实际调用",
      ]),
  ];
}
function highlighted(text, spans) {
  const chars = Array.from(text),
    fragment = document.createDocumentFragment(),
    ranges = spans
      .map((x) => [x.span.start, x.span.end])
      .sort((a, b) => a[0] - b[0]);
  let at = 0;
  for (const [a, b] of ranges) {
    if (a < at || b > chars.length) continue;
    fragment.append(
      document.createTextNode(chars.slice(at, a).join("")),
      el("mark", "", chars.slice(a, b).join("")),
    );
    at = b;
  }
  fragment.append(document.createTextNode(chars.slice(at).join("")));
  return fragment;
}
function renderSources(r) {
  const all = sourceRows(r),
    rows = all.filter((s) => view === "all" || s.source === view),
    box = $("#sources");
  const scrolls = new Map(
    [...box.querySelectorAll(".paper")].map((n) => [
      n.dataset.sourceId,
      {
        top: n.querySelector(".original").scrollTop,
        marks: n.querySelectorAll("mark").length,
      },
    ]),
  );
  box.replaceChildren();
  box.className =
    "sources" + (rows.length === 1 ? " one" : rows.length === 2 ? " two" : "");
  box.style.gridTemplateColumns =
    rows.length > 3 ? `repeat(${rows.length}, minmax(350px, 1fr))` : "";
  $("#kb-count").textContent = all.filter(
    (s) => s.source === "knowledge_base",
  ).length;
  $("#web-count").textContent = all.filter((s) => s.source === "web").length;
  $("#view-all").classList.toggle("active", view === "all");
  $("#view-kb").classList.toggle("active", view === "knowledge_base");
  $("#view-web").classList.toggle("active", view === "web");
  $("#source-title").textContent =
    view === "all"
      ? "知识库 + 网络 · 原文并排"
      : view === "knowledge_base"
        ? "知识库原文 · 并排阅读"
        : "网络来源 · 本次实际取材";
  $("#source-state").textContent = r
    ? "金色标记为逐字引用位置"
    : "等待本次取材";
  const bound = new Map(
    (r?.job?.result?.graph?.spans || []).map((s) => [s.id, s]),
  );
  for (const e of events(r).filter((e) => e.kind === "binding_finished"))
    for (const s of e.spans || []) bound.set(s.id, s);
  if (!rows.length) {
    box.append(
      el(
        "div",
        "empty-card",
        r
          ? view === "web"
            ? r.finished
              ? "本次未取得网页来源，请查看检索状态。"
              : "等待本次网络返回。查询与知识库取材可同时进行。"
            : r.finished
              ? "本次未取得知识库原文。"
              : "等待本次知识库选材。"
          : "提交问题后，显示本次实际取得的原文。",
      ),
    );
    return;
  }
  for (const s of rows) {
    const card = el("article", "paper");
    card.dataset.sourceId = s.id;
    const head = el("div", "paper-head");
    head.append(el("strong", "", s.metadata.original_source_title || s.title));
    const link = el("a", "", s.uri || "知识库原文");
    if (/^https?:\/\//.test(s.uri || "")) {
      link.href = s.uri;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
    }
    head.append(link);
    const spans = [...bound.values()].filter((x) => x.source_id === s.id),
      tag = el("div", "paper-tag"),
      label =
        s.source === "web"
          ? s.metadata.content_status === "fetched"
            ? "网络 · 已取得正文"
            : "网络 · 仅搜索摘要"
          : "知识库 · 原文";
    tag.append(
      el("span", "", label),
      el(
        "span",
        "",
        s.received
          ? `${spans.length} 引用 · ${s.source === "web" ? "实时获取" : "选入 " + (s.windows?.length || 0) + " 窗"}`
          : "已载入",
      ),
    );
    head.append(tag);
    const pre = el("pre", "original"),
      section = s.view?.body ? "body" : "snippet",
      text = s.view?.[section] || s.snippet || "";
    pre.append(
      highlighted(
        text,
        spans.filter(
          (x) =>
            x.section === section &&
            x.span.view_sha256 === s.view?.[section + "_sha256"],
        ),
      ),
    );
    if (s.view?.body && s.view?.snippet) {
      pre.append(
        el("div", "snippet-label", "搜索摘要 · 网页正文未核实"),
        highlighted(
          s.view.snippet,
          spans.filter(
            (x) =>
              x.section === "snippet" &&
              x.span.view_sha256 === s.view.snippet_sha256,
          ),
        ),
      );
    }
    if (s.source === "web" && s.metadata.retrieved_at) {
      const stamp = el(
        "span",
        "",
        "获取于 " +
          new Date(s.metadata.retrieved_at).toLocaleString("zh-CN", {
            hour12: false,
          }),
      );
      stamp.style.cssText = "font-size:10px;color:#888";
      head.append(stamp);
    }
    card.append(head, pre);
    box.append(card);
    const previous = scrolls.get(s.id);
    pre.scrollTop = previous?.top || (s.source === "knowledge_base" ? 110 : 0);
    if (spans.length && spans.length !== (previous?.marks || 0)) {
      const mark = pre.querySelector("mark");
      if (mark)
        pre.scrollTop +=
          (mark.getBoundingClientRect().top - pre.getBoundingClientRect().top) /
            (pre.getBoundingClientRect().width / pre.offsetWidth) -
          25;
    }
  }
}
function renderDetail() {
  const r = active;
  $("#detail-label").textContent = info[0];
  $("#detail-question").textContent = r?.question || "从原文出发。";
  $("#detail-scope").textContent = r ? scopeDescription(r.request) : "";
  $("#phase").textContent = r?.job?.phase || "等待提问";
  $("#steps").replaceChildren(
    ...stateFor(r).map(([name, state, note]) => {
      const n = el("div", "step " + state);
      n.append(
        el(
          "b",
          "",
          (state === "done" ? "✓ " : state === "active" ? "● " : "") + name,
        ),
        el("small", "", note),
      );
      return n;
    }),
  );
  const plan = events(r).find((e) => e.kind === "plan_ready");
  $("#queries").textContent = plan
    ? "实际查询：" + plan.queries.join("  |  ")
    : "将在模型完成规划后显示实际检索词。";
  renderSources(r);
  const answer = r?.job?.result?.display_text;
  $("#answer").replaceChildren(
    answer
      ? document.createTextNode(answer)
      : el(
          "span",
          "muted",
          r
            ? r.error ||
              (r.job?.status &&
                !["queued", "running", "completed"].includes(r.job.status))
              ? "本次请求未完成，保留实际状态。"
              : "正在处理本次问题，证据与阅读进度实时更新。"
            : "提交后，回答会显示在这里。",
        ),
  );
  $("#answer-state").textContent = answer
    ? `${r.job.result.answer?.generation?.raw_model_answer ? "本次实时回答" : "未找到依据 · 系统状态提示"} · ${(r.wall || (performance.now() - r.start) / 1000).toFixed(2)} 秒`
    : r?.finished
      ? "请求未完成"
      : "由本次选中证据生成";
  $("#citations").replaceChildren();
  for (const [n, ids] of Object.entries(
    r?.job?.result?.answer?.generation?.citation_bindings || {},
  ).filter(([n]) =>
    (
      r?.job?.result?.answer?.generation?.parsed_answer?.source_ids || []
    ).includes(n),
  ))
    for (const id of Array.isArray(ids) ? ids : [ids]) {
      const s = sourceRows(r).find((s) => s.id === id);
      if (!s) continue;
      const b = el(
        "button",
        "",
        `[${n}] ${s.metadata.original_source_title || s.title}`,
      );
      b.onclick = () => {
        view = s.source;
        renderSources(r);
        [...document.querySelectorAll("[data-source-id]")]
          .find((n) => n.dataset.sourceId === s.id)
          ?.querySelector("mark")
          ?.scrollIntoView({ block: "nearest" });
      };
      $("#citations").append(b);
    }
}
async function askOne(r) {
  try {
    r.job = await api("/v1/try/ask", r.request);
    renderCards();
    if (r === active) renderDetail();
    while (["queued", "running"].includes(r.job.status)) {
      await sleep(400);
      r.job = await api("/v1/try/jobs/" + r.job.id);
      renderCards();
      if (r === active) renderDetail();
    }
    r.wall = (performance.now() - r.start) / 1000;
    r.finished = true;
    window.__rwkvLiveRuns.push({
      question: r.question,
      job_id: r.job.id,
      status: r.job.status,
      client_wall_seconds: r.wall,
      request: r.request,
      raw_response: r.job,
    });
  } catch (e) {
    r.error = String(e);
    r.wall = (performance.now() - r.start) / 1000;
    r.finished = true;
  }
  renderCards();
  if (r === active) renderDetail();
}

let serviceMeta = null;
let importedRows = null, importRevision = 0, importProblem = false;
const modeLabels = {
  hybrid: "知识库 + 网络", knowledge_base: "知识库", web: "网络",
};
function scopeDescription(request) {
  const names = (request.knowledge_base_ids || []).map((id) =>
    serviceMeta?.knowledge_bases.find((kb) => kb.id === id)?.name || id,
  );
  const parts = [modeLabels[request.retrieval_mode]];
  if (request.retrieval_mode !== "web")
    parts.push(names.length ? names.join("、") : "未选择知识库");
  if (request.retrieval_mode !== "knowledge_base")
    parts.push(request.web_include_domains?.length
      ? request.web_include_domains.join("、") : Object.hasOwn(request, "web_include_domains")
        ? "未限定网络域名" : "域名沿用服务配置");
  return parts.join(" · ");
}
function validDomains(domains) {
  return domains.length <= 8 && domains.every((s) =>
    typeof s === "string" && s.length <= 253 &&
    /^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$/.test(s),
  );
}
function validateImport(rows) {
  if (!Array.isArray(rows) || !rows.length || rows.length > MAX_QUESTIONS)
    throw Error(`清单必须是 1 至 ${MAX_QUESTIONS} 项 JSON 数组`);
  if (!serviceMeta) throw Error("请等待服务连接后导入");
  const keys = ["question", "retrieval_mode", "knowledge_base_ids", "web_include_domains"];
  const known = new Set(serviceMeta.knowledge_bases.map((kb) => kb.id));
  const questions = new Set();
  return rows.map((row, i) => {
    const fail = (message) => { throw Error(`第 ${i + 1} 项：${message}`); };
    if (!row || typeof row !== "object" || Array.isArray(row) ||
      Object.keys(row).length !== keys.length || !keys.every((k) => Object.hasOwn(row, k)))
      fail("仅允许 question / retrieval_mode / knowledge_base_ids / web_include_domains 四个字段");
    if (typeof row.question !== "string" || !row.question.trim() || row.question.length > 10000 || /[\r\n]/.test(row.question))
      fail("问题须为非空单行文字，最多 10000 字符");
    const question = row.question.trim();
    if (questions.has(question)) fail("问题重复，请使用不同问题");
    questions.add(question);
    if (typeof row.retrieval_mode !== "string" || !Object.hasOwn(modeLabels, row.retrieval_mode))
      fail("检索模式无效");
    if (!Array.isArray(row.knowledge_base_ids) || row.knowledge_base_ids.some((id) => typeof id !== "string" || !known.has(id)) ||
      new Set(row.knowledge_base_ids).size !== row.knowledge_base_ids.length)
      fail("知识库 ID 不存在或重复");
    if (row.retrieval_mode === "knowledge_base" && !row.knowledge_base_ids.length)
      fail("知识库模式至少选择一份知识库");
    if (!Array.isArray(row.web_include_domains) || row.web_include_domains.some((s) => typeof s !== "string"))
      fail("网络域名须为数组");
    const domains = row.web_include_domains.map((s) => s.toLowerCase().replace(/\.$/, ""));
    if (!validDomains(domains) || new Set(domains).size !== domains.length)
      fail("最多 8 个不同域名，不包含网址路径或搜索指令");
    return {question, retrieval_mode: row.retrieval_mode,
      knowledge_base_ids: [...row.knowledge_base_ids], web_include_domains: domains};
  });
}
function clearImport(message) {
  importedRows = null;
  importProblem = false;
  importRevision++;
  $("#import-scopes").hidden = true;
  $("#import-scopes").open = false;
  $("#scope-list").replaceChildren();
  $("#import-status").textContent = message;
  readiness();
}
$("#question").addEventListener("input", () => {
  if (importedRows) clearImport("输入已修改：已清除全部逐题预设范围；本次将使用上方手选来源。重新导入可恢复逐题范围。");
  else {
    importRevision++;
    if (importProblem) clearImport("输入已修改：本次使用上方手选来源，逐题预设范围未启用。");
  }
});
$("#question-file").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  event.target.value = "";
  if (!file) return;
  clearImport("正在校验问题清单…");
  importProblem = true;
  readiness();
  const revision = importRevision;
  try {
    if (file.size > 1024 * 1024) throw Error("清单不能大于 1 MB");
    const rows = validateImport(JSON.parse(await file.text()));
    if (revision !== importRevision) return;
    importedRows = rows;
    importProblem = false;
    $("#question").value = rows.map((row) => row.question).join("\n");
    $("#scope-list").replaceChildren(...rows.map((row) => {
      const item = el("li");
      item.append(el("strong", "", row.question), el("span", "", scopeDescription(row)));
      return item;
    }));
    $("#import-scopes").hidden = false;
    $("#import-status").textContent = `已载入 ${rows.length} 题及各自资料范围 · 上方手选来源暂不作用于清单；修改问题会清除预设范围。`;
  } catch (error) {
    if (revision !== importRevision) return;
    importProblem = true;
    $("#import-status").textContent = "导入失败：" + error.message + "。请重新导入，或修改输入后使用手选来源。";
  }
  readiness();
});
function selectedKBs() {
  return [...document.querySelectorAll("#knowledge-bases input:checked")].map(
    (n) => n.value,
  );
}
function readiness() {
  const scopes = importedRows || [{retrieval_mode: $("#retrieval-mode").value, knowledge_base_ids: selectedKBs()}];
  const pending = runs.some((r) => !r.finished);
  const available = serviceMeta?.model_readiness === "available";
  $("#send").disabled =
    pending ||
    importProblem ||
    !available ||
    scopes.some((row) => (row.retrieval_mode !== "knowledge_base" && !serviceMeta?.web_configured) ||
      (row.retrieval_mode === "knowledge_base" && !row.knowledge_base_ids.length));
  $("#question-file").disabled = pending || !serviceMeta;
  for (const input of document.querySelectorAll(".settings input, .settings select, #knowledge-bases input"))
    input.disabled = Boolean(importedRows);
  $("#readiness").textContent = available ? "● RWKV 已连接" : "连接未就绪";
  $("#readiness").classList.toggle("ready", available);
}
$("#send").onclick = async () => {
  const questions = $("#question")
    .value.split("\n")
    .map((s) => s.trim())
    .filter(Boolean);
  if (!questions.length || questions.length > MAX_QUESTIONS) {
    setOverview(false);
    $("#phase").textContent = `请输入 1 至 ${MAX_QUESTIONS} 行问题`;
    return;
  }
  const mode = $("#retrieval-mode").value,
    ids = selectedKBs();
  const text = $("#domains").value.trim();
  const domains = text
    ? text
        .split(/[,，\s]+/)
        .filter(Boolean)
        .map((s) => s.toLowerCase().replace(/\.$/, ""))
    : [];
  if (!importedRows && !validDomains(domains)) {
    $("#phase").textContent = "请输入最多 8 个域名，不包含网址路径或搜索指令";
    return;
  }
  if (!importedRows && mode === "knowledge_base" && !ids.length) {
    $("#phase").textContent = "请选择知识库";
    return;
  }
  // Presets are explicit row scopes, never selected by question contents.
  // Recheck the exact textarea rather than silently binding a changed question.
  if (importedRows && $("#question").value !== importedRows.map((row) => row.question).join("\n")) {
    clearImport("输入已改变：已清除逐题预设范围，请确认上方来源后重新提交。");
    return;
  }
  const requests = importedRows || questions.map((question) => ({
    question, retrieval_mode: mode, knowledge_base_ids: ids,
    ...(domains.length ? {web_include_domains: domains} : {}),
  }));
  batch = { start: performance.now(), count: requests.length };
  runs = requests.map((request) => ({
    question: request.question,
    start: performance.now(),
    request: {
      ...request,
      knowledge_base_ids: [...request.knowledge_base_ids],
      ...(request.web_include_domains ? {web_include_domains: [...request.web_include_domains]} : {}),
      parent_job_id: null,
    },
  }));
  active = runs[0];
  view = runs[0].request.retrieval_mode === "web" ? "web" : "knowledge_base";
  document.body.classList.toggle(
    "parallel",
    runs.length > 1 || scene === "parallel",
  );
  setOverview(runs.length > 8);
  readiness();
  renderCards();
  renderDetail();
  await Promise.all(runs.map(askOne));
  batch.wall = (performance.now() - batch.start) / 1000;
  window.__rwkvBatches.push({
    count: batch.count,
    wall_seconds: batch.wall,
    job_ids: runs.map((r) => r.job?.id),
  });
  readiness();
};
$("#overview").onclick = () => {
  setOverview(!document.body.classList.contains("overview"));
};
$("#edit-input").onclick = () => {
  const expanded = document.body.classList.toggle("editing-input");
  $("#edit-input").setAttribute("aria-expanded", String(expanded));
  $("#edit-input").textContent = expanded ? "收起输入" : "编辑问题与来源";
};
$("#view-all").onclick = () => {
  view = "all";
  renderSources(active);
};
$("#view-kb").onclick = () => {
  view = "knowledge_base";
  renderSources(active);
};
$("#view-web").onclick = () => {
  view = "web";
  renderSources(active);
};
$("#retrieval-mode").value = scene === "web" ? "web" : "hybrid";
$("#retrieval-mode").onchange = readiness;
setInterval(() => {
  if (batch)
    $("#batch-time").textContent = (
      batch.wall ?? (performance.now() - batch.start) / 1000
    ).toFixed(2);
  document.querySelectorAll("#runs .run").forEach((n, i) => {
    const r = runs[i];
    if (r)
      n.querySelector(".run-time").textContent =
        (r.wall ?? (performance.now() - r.start) / 1000).toFixed(2) + " s";
  });
}, 100);
try {
  serviceMeta = await api("/v1/try/meta");
  for (const kb of serviceMeta.knowledge_bases) {
    const label = el("label"),
      input = el("input");
    input.type = "checkbox";
    input.value = kb.id;
    input.checked = (serviceMeta.default_knowledge_base_ids || []).includes(
      kb.id,
    );
    input.onchange = readiness;
    label.append(input, document.createTextNode(kb.name));
    $("#knowledge-bases").append(label);
  }
  if (!serviceMeta.live_progress)
    $("#input-hint").textContent = "当前服务未开启阶段事件；完成后显示原始回答";
  readiness();
  renderDetail();
} catch (e) {
  $("#readiness").textContent = "连接暂不可用";
  renderDetail();
}
