import { useEffect, useRef, useState } from "react";
import { Alert, Button, Checkbox, Drawer, Input, Select, Spin, Tag } from "antd";
import { ArrowUpOutlined, DownloadOutlined, FileTextOutlined, PlusOutlined, StopOutlined } from "@ant-design/icons";
import "../trial.css";

type Source = { id: string; title: string; source: string; uri?: string; snippet: string };
type Answer = { answer: string; sources: Source[]; generation: { status: string; citation_bindings?: Record<string, string[]>; parsed_answer?: { answer: string; source_ids: string[] } } };
type Result = { answer: Answer; response_type?: string; display_text?: string; not_found_needs?: { id: string; text: string }[];
  telemetry?: { total_ms: number; total_calls: number; critical_path_calls: number };
  graph?: { spans?: { id: string; text: string; source_id: string }[] };
  acquisition_incomplete?: boolean; stop_reason?: string; rounds?: { number: number; questions?: string[]; decision: { status: string; parsed?: { questions?: string[]; queries?: string[]; kb_queries?: string[]; web_queries?: string[] } } }[];
  acquisition?: { channel: string; query: string; status?: string; error?: string; trace?: { returned: number } }[] };
type Job = { id: string; question: string; status: string; phase: string; model_calls_finished: number; result?: Result; error_type?: string; context?: { resolved_question?: string; parent_job_id?: string } };
type Meta = { model: string; gpu: number; concurrency: number; expires_at?: string; default_pipeline?: string; pipelines?: string[];
  knowledge_bases: { id: string; name: string; description: string; file_count: number }[] };
const terminal = (job: Job) => ["completed", "failed", "timeout", "cancelled"].includes(job.status);
const STORAGE = "rwkvrag-funnel-trial-jobs-v2";
const responseLabels: Record<string, string> = { answered: "已回答", partial: "部分回答", not_found: "未找到依据", unverified: "未校验", error: "服务异常" };
const executionStatus = (status: string) => ({
  budget_exceeded: "所选资料超过模型可读长度，未生成回答",
  partial: "部分处理未完成", invalid_output: "模型输出格式未通过检查",
  length: "模型输出达到长度上限", timeout: "模型调用超时",
  no_evidence: "没有可用于回答的资料",
} as Record<string, string>)[status] || status;

async function request<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(path, body === undefined ? undefined : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const value = await response.json();
  if (!response.ok) throw new Error(typeof value.detail === "string" ? value.detail : `请求失败（${response.status}）`);
  return value;
}

export default function TrialPage() {
  const [meta, setMeta] = useState<Meta>();
  const [error, setError] = useState("");
  const [question, setQuestion] = useState("");
  const [scope, setScope] = useState("hybrid");
  const [pipeline, setPipeline] = useState<string>();
  const [kbIds, setKbIds] = useState<string[]>([]);
  const [supplementary, setSupplementary] = useState(true);
  const [job, setJob] = useState<Job>();
  const [history, setHistory] = useState<{ id: string; question: string }[]>([]);
  const [source, setSource] = useState<Source>();
  const [sending, setSending] = useState(false);
  const active = useRef("");
  const submitting = useRef(false);
  const input = useRef<React.ComponentRef<typeof Input.TextArea>>(null);
  const busy = sending || !!(job && !terminal(job));

  useEffect(() => {
    let live = true;
    request<Meta>("/v1/try/meta").then(value => {
      if (!live) return;
      setMeta(value);
      setPipeline(value.default_pipeline);
      // The existing encyclopedia is the general-purpose initial selection.
      // Other collections remain selectable, including explicitly named fixtures.
      const general = value.knowledge_bases.find(k => k.id === "wiki-5000-20260908");
      setKbIds(general ? [general.id] : []);
    }).catch(e => { if (live) setError(String(e.message)); });
    try {
      localStorage.removeItem("rwkvrag-direct-trial-jobs-v1");
      setHistory(JSON.parse(localStorage.getItem(STORAGE) || "[]"));
    } catch { /* Optional recent IDs only. */ }
    return () => { live = false; };
  }, []);

  useEffect(() => {
    if (!job || terminal(job)) return;
    let live = true;
    const id = job.id;
    const timer = window.setInterval(() => {
      void request<Job>(`/v1/try/jobs/${id}`).then(next => {
        if (live && active.current === id) setJob(next);
      }).catch(e => { if (live) setError(`读取进度失败：${e.message}。后台请求可能仍在运行。`); });
    }, 1200);
    return () => { live = false; window.clearInterval(timer); };
  }, [job?.id, job?.status]);

  async function send() {
    if (submitting.current || busy || !question.trim() || !meta) return;
    submitting.current = true;
    setSending(true);
    setError("");
    try {
      const next = await request<Job>("/v1/try/ask", { question, supplementary,
        ...(["lean", "needs", "spans"].includes(pipeline || "") ? { pipeline } : {}),
        parent_job_id: job?.status === "completed" ? job.id : null,
        retrieval_mode: scope, knowledge_base_ids: scope === "web" ? [] : kbIds });
      active.current = next.id;
      setJob(next);
      const rows = [{ id: next.id, question: next.question }, ...history.filter(h => h.id !== next.id)];
      setHistory(rows);
      try { localStorage.setItem(STORAGE, JSON.stringify(rows)); } catch { /* Server record remains available. */ }
      setQuestion("");
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { submitting.current = false; setSending(false); }
  }
  async function open(id: string) {
    if (busy) return;
    active.current = id;
    try { const value = await request<Job>(`/v1/try/jobs/${id}`); if (active.current === id) setJob(value); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  }
  async function stop() {
    if (!job) return;
    try { setJob(await request<Job>(`/v1/try/jobs/${job.id}/cancel`, {})); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  }
  function download() {
    if (!job) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(job, null, 2)], { type: "application/json" }));
    const link = document.createElement("a"); link.href = url; link.download = `rwkv-${job.id}.json`; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  function card(value: Answer, result: Result) {
    const parsed = value.generation.parsed_answer;
    const used = new Set((parsed?.source_ids || []).flatMap(id => value.generation.citation_bindings?.[id] || [id]));
    return <article className="trial-answer">
      <div className="trial-answer-heading"><strong>回答</strong><Tag color={result.response_type === "unverified" || result.response_type === "error" ? "orange" : "green"}>
        {responseLabels[result.response_type || ""] || "资料回答"}</Tag></div>
      {result.response_type === "unverified" && <Alert type="warning" showIcon message="以下为模型原始输出，尚未通过校验。" />}
      {!result.response_type && value.generation.status !== "completed" && <Alert type="warning" showIcon message={`${executionStatus(value.generation.status)}。可展开下方记录查看。`} />}
      <div className="trial-answer-body">{result.display_text !== undefined ? (result.display_text || "模型未返回正文。") : parsed ? parsed.answer : (value.answer || "本次没有生成可用回答。")}</div>
      {!!result.not_found_needs?.length && <div>未找到依据：{result.not_found_needs.map(n => <div key={n.id}>{n.text}</div>)}</div>}
      {(result.response_type === "not_found" || result.response_type === "error") && <div>
        <p>本次检索：{result.acquisition?.map(a => a.query).join("；") || "检索尚未完成"}</p>
        <div className="trial-source-list">{value.sources.slice(0, 5).map(s => <button key={s.id} onClick={() => setSource(s)}>{s.title || s.uri || s.id}</button>)}</div>
      </div>}
      {result.telemetry && <p className="trial-muted">耗时 {(result.telemetry.total_ms / 1000).toFixed(1)} 秒 · {result.telemetry.total_calls} 次模型调用</p>}
      <div className="trial-source-label">回答的来源链 · 可打开原文核对</div>
      <div className="trial-source-list">{value.sources.filter(s => used.has(s.id)).map(s =>
        <button key={s.id} onClick={() => setSource(s)}><FileTextOutlined /><span>{s.title || s.id}</span><small>{s.source === "web" ? "网络" : "知识库"}</small></button>)}
        {!used.size && <span className="trial-muted">模型未声明引用来源</span>}</div>
      <div className="trial-source-list">{result.graph?.spans?.filter(s => parsed?.source_ids.includes(s.id)).map(span => {
        const original = value.sources.find(s => s.id === span.source_id);
        return original && <button key={span.id} onClick={() => setSource({ ...original, snippet: span.text })}>[{span.id}] {original.title}</button>;
      })}</div>
      <details><summary>查看所选资料与原始回答</summary>
        <div className="trial-all-sources">{value.sources.map(s => <Button key={s.id} size="small" onClick={() => setSource(s)}>{s.title || s.id}</Button>)}</div>
        <pre className="trial-raw">{value.answer}</pre>
      </details>
    </article>;
  }

  return <div className="trial-shell">
    <aside className="trial-sidebar">
      <div className="trial-brand"><span>R</span><div><strong>RWKV</strong><small>资料问答 · 直接试用</small></div></div>
      <Button className="trial-new" icon={<PlusOutlined />} disabled={busy} onClick={() => { active.current = ""; setJob(undefined); setError(""); input.current?.focus(); }}>新的提问</Button>
      <div className="trial-label">查询范围</div>
      <Select aria-label="查询范围" value={scope} onChange={setScope} disabled={busy} options={[
        { value: "hybrid", label: "知识库 + 联网资料" }, { value: "knowledge_base", label: "仅知识库" }, { value: "web", label: "仅联网资料" }]} />
      {!!meta?.pipelines?.length && <><div className="trial-label">回答方式</div><Select aria-label="回答方式" value={pipeline} onChange={setPipeline} disabled={busy}
        options={meta.pipelines.map(value => ({ value, label: ({ lean: "快速检索", needs: "逐项核实", spans: "分段取证" } as Record<string, string>)[value] || value }))} /></>}
      {scope !== "web" && <><div className="trial-label">知识库</div><Select mode="multiple" aria-label="选择知识库" value={kbIds} onChange={setKbIds} disabled={busy}
        placeholder="选择资料库" options={meta?.knowledge_bases.map(k => ({ value: k.id, label: `${k.name} · ${k.file_count} 篇` }))} /></>}
      {pipeline !== "lean" && <div className="trial-comparison"><Checkbox checked={supplementary} disabled={busy} onChange={e => setSupplementary(e.target.checked)}>根据缺口继续补查</Checkbox>
        <p>在可用时间内继续寻找缺少的依据。</p></div>}
      <div className="trial-label">最近提问</div>
      <nav className="trial-history">{history.map(h => <button className={h.id === job?.id ? "active" : ""} key={h.id} disabled={busy} onClick={() => void open(h.id)} title={h.question}>{h.question}</button>)}</nav>
      <div className="trial-sidebar-foot">可以在回答后继续追问。点击“新的提问”会清空本次上下文。模型总结仍可能遗漏或出错，请结合原文核对。</div>
    </aside>
    <main className="trial-main">
      <header className="trial-top"><div><span className={`trial-dot${meta ? " ready" : ""}`} />{meta ? `${meta.model} · GPU${meta.gpu}` : "连接服务中"}</div>
        <span>{meta?.expires_at ? `试用至 ${new Date(meta.expires_at).toLocaleString("zh-CN", { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" })}` : "依据真实资料回答"}</span></header>
      <section className="trial-content">
        {!job && <div className="trial-welcome"><div className="trial-symbol">R</div><h1>把问题交给资料。</h1>
          <p>理解问题，检索资料，核对原文后回答。</p><p className="trial-muted">事实查询、方案比较、行程规划、文档综合，都可以直接提问。</p></div>}
        {error && <Alert className="trial-alert" type="error" message={error} showIcon closable onClose={() => setError("")} />}
        {job && <><div className="trial-question"><span>你的问题</span><h2>{job.question}</h2></div>
          {!terminal(job) && <div className="trial-progress" role="status"><Spin size="small" /><div>{job.phase}<small>已完成 {job.model_calls_finished} 次模型调用</small></div>
            <Button icon={<StopOutlined />} onClick={() => void stop()}>停止</Button></div>}
          {terminal(job) && job.status !== "completed" && <Alert type="warning" showIcon message={`${job.phase}${job.error_type ? `（${job.error_type}）` : ""}`} />}
          {job.context && <details className="trial-process"><summary>本次理解的问题{job.context.parent_job_id ? " · 承接上一问" : ""}</summary><div>{job.context.resolved_question || "上下文整理未完成"}</div></details>}
          {job.result && <>{job.result.acquisition_incomplete && <Alert type="warning" showIcon message="部分资料获取或目录选择未完成，本次回答依据可能不完整。可展开下方记录查看。" />}<div className="trial-answers">
            {card(job.result.answer, job.result)}</div>
            <details className="trial-process"><summary>查了什么资料 · {job.result.acquisition?.length || 0} 次获取</summary>
              {job.result.acquisition?.map((item, i) => <div key={i}><Tag>{item.channel === "web" ? "联网" : "知识库"}</Tag>{item.query}{item.status === "failed" && <strong> · 获取失败：{item.error}</strong>}</div>)}
              {job.result.rounds?.map(r => <div key={`r${r.number}`}>第 {r.number} 轮补查判断：{r.decision.parsed ?
                ((q => q.length ? q.join("；") : "模型决定无需继续补查")(r.decision.parsed.questions || r.decision.parsed.queries || [...(r.decision.parsed.kb_queries || []).map(q => `知识库：${q}`), ...(r.decision.parsed.web_queries || []).map(q => `联网：${q}`)])) : `未得到有效判断（${r.decision.status}）`}</div>)}
              {job.result.stop_reason && <div>补查结束状态：{job.result.stop_reason}</div>}
            </details><Button type="text" icon={<DownloadOutlined />} onClick={download}>导出本次回答与依据</Button></>}
        </>}
      </section>
      <form className="trial-composer" onSubmit={e => { e.preventDefault(); void send(); }}>
        <Input.TextArea ref={input} aria-label="输入问题" value={question} onChange={e => setQuestion(e.target.value)} autoSize={{ minRows: 2, maxRows: 7 }}
          maxLength={10000} placeholder="输入你想了解的问题…" disabled={!meta || busy}
          onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); void send(); } }} />
        <div className="trial-composer-bottom"><span>{scope === "web" ? "联网资料" : scope === "knowledge_base" ? "知识库" : "知识库 + 联网资料"} · {job?.status === "completed" ? "继续追问" : "新话题"}</span>
          <Button type="primary" htmlType="submit" aria-label="发送问题" icon={<ArrowUpOutlined />} loading={sending} disabled={!meta || busy || !question.trim()}>提问</Button></div>
      </form>
    </main>
    <Drawer title={source?.title || "来源原文"} open={!!source} onClose={() => setSource(undefined)} width="min(640px, 100vw)">
      {source && <><Tag>{source.source === "web" ? "联网资料" : "知识库原文"}</Tag><code>{source.id}</code>
        {source.uri && /^https?:\/\//.test(source.uri) && <p><a href={source.uri} target="_blank" rel="noreferrer">打开来源网页 ↗</a></p>}
        <pre className="trial-original">{source.snippet}</pre></>}
    </Drawer>
  </div>;
}
