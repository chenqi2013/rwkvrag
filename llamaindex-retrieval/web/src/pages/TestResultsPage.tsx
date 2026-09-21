import { useEffect, useState } from "react";
import { Alert, Button, Card, Drawer, Select, Space, Spin, Tag, Typography } from "antd";
import "./modelComparison.css";
import { evidenceFlowNotices, executionLabel, type EvidenceFlow } from "../evidenceFlow";

type Source = { label: string; text: string; url?: string };
type Answer = { role?: "candidate" | "baseline" | "historical"; evidence_flow?: EvidenceFlow; label: string; raw_text: string; finish_reason?: string; elapsed_s: number; notes: string; sources: Source[]; queries?: { query: string; status: string }[]; funnel?: { task?: unknown; facts?: unknown[]; cells?: unknown[]; field_summaries?: unknown[]; conditions?: unknown[]; candidates?: unknown[]; decision?: unknown; call_budget?: unknown; failures?: unknown[] }; trace?: unknown };
type Case = { id: string; question: string; answers: Answer[]; history?: {role: string; content: string}[]; input_sources?: Source[] };
type Dataset = { title: string; summary: string; paired?: boolean; cases: Case[] };

export default function TestResultsPage() {
  const [suite, setSuite] = useState("writer-evidence-handoff-20260921");
  const [showHistory, setShowHistory] = useState(false);
  const [data, setData] = useState<Dataset>();
  const [error, setError] = useState("");
  const [index, setIndex] = useState(0);
  const [source, setSource] = useState<Source>();
  useEffect(() => {
    const controller = new AbortController(); setData(undefined); setError(""); setIndex(0);
    fetch(`${import.meta.env.BASE_URL}experiments/${suite}.json`, {signal:controller.signal})
      .then(r => {if (!r.ok) throw new Error(`数据尚未发布或加载失败：HTTP ${r.status}`); return r.json();})
      .then(setData).catch(e => {if(e.name !== "AbortError") setError(String(e));});
    return () => controller.abort();
  }, [suite]);
  const current = data?.cases[index];
  const hasVersions = !data?.paired && current?.answers.some(a => a.role === "candidate");
  const visibleAnswers = current?.answers.filter(a => !hasVersions || showHistory || a.role === "candidate")
    .slice().sort((a,b) => Number(b.role === "candidate") - Number(a.role === "candidate"));
  useEffect(() => { setSource(undefined); }, [suite, index]);
  function answerText(a: Answer) {
    return a.raw_text.split(/(\[资料\s*\d+(?:\s*[,，、]\s*(?:资料\s*)?\d+)*\])/g).map((part, i) => {
      if (!/^\[资料/.test(part)) return part;
      const ids = part.match(/\d+/g) || [];
      const open = (id: string) => setSource(a.sources.find(s => s.label.replace(/\s/g, "") === `资料${Number(id)}`) || {label:`资料 ${id}`,text:"该编号没有对应的已选证据。保留原回答，未补齐引用。"});
      if (ids.length === 1) return <button key={i} className="comparison-citation" onClick={() => open(ids[0])}>{part}</button>;
      return <span key={i}>{part.split(/(\d+)/).map((piece, j) => /^\d+$/.test(piece) ? <button key={j} className="comparison-citation" onClick={() => open(piece)}>{piece}</button> : piece)}</span>;
    });
  }
  return <div className="model-comparison">
    <Typography.Title level={3}>真实检索与复读测试</Typography.Title>
    <Select aria-label="测试集合" value={suite} onChange={setSuite} style={{width:320}} options={[
      {value:"writer-evidence-handoff-20260921",label:"Writer处理状态 · 双轮原始回答对照"},
      {value:"typed-funnel-diagnostics-20260921",label:"上一阶段 · v10故障定位"},
      {value:"typed-funnel-20260921",label:"历史快照 · v4/v8/v10原始对照"},
      {value:"funnel-repairs-20260921",label:"真实比较修复 · 分层过程"},
      {value:"github-natural-comparison-20260921",label:"大型 GitHub 比较 · 自然问法"},
      {value:"github-project-comparison-paced-20260921",label:"大型 GitHub 比较 · 低频实时检索"},
      {value:"github-project-comparison-20260921",label:"大型 GitHub 比较 · 原联网失败记录"},
      {value:"live-comparison-20260921",label:"真实联网及知识库联合比较 · 12题"},
      {value:"writer-convergence-20260921",label:"Writer指令对照 · 188题双轮"}]} />
    {error && <Alert type="warning" title={error} />}
    {!data && !error && <Spin />}
    {data && <><h3>{data.title}</h3><Alert type="info" title={data.summary} />
      {hasVersions && <Space style={{marginTop:12}} wrap><Tag color="orange">实验候选 · 未部署正式服务</Tag><Button aria-pressed={showHistory} onClick={() => setShowHistory(!showHistory)}>{showHistory ? "只看最近实验结果" : "展开历史对照（v4、v8）"}</Button><span>历史回答用于对照；修复是否有效请看最近实验的具体诊断。</span></Space>}
      <Space wrap className="comparison-controls">
        <Button disabled={index === 0} onClick={() => setIndex(index-1)}>上一题</Button>
        <Select aria-label="测试题目" value={index} onChange={setIndex} style={{width:"min(760px,65vw)"}} options={data.cases.map((c,i)=>({value:i,label:`${i+1}. ${c.question}`}))} />
        <Button disabled={index >= data.cases.length-1} onClick={() => setIndex(index+1)}>下一题</Button>
        <a href={`${import.meta.env.BASE_URL}experiments/${suite}.json`} download>下载全部回答与证据</a>
      </Space>
      {current && <><Card title={current.id}><Typography.Title level={4}>{current.question}</Typography.Title>
          {!!current.history?.length && <details><summary>本题使用的原始对话历史</summary>{current.history.map((message, i) => <div key={i}><Tag>{message.role}</Tag><pre>{message.content}</pre></div>)}</details>}
          {current.input_sources && <details><summary>进入漏斗前的已选材料（{current.input_sources.length}份）</summary><p>这些材料用于检查在哪一层丢失了证据；回答内的引用仍按各版本实际使用的来源映射。</p>{current.input_sources.map((item, i) => <p key={i}><Button onClick={() => setSource(item)}>{i + 1}. 查看输入原文</Button></p>)}</details>}
        </Card>
        {visibleAnswers?.map((a,i)=><Card key={`${index}:${a.label}:${i}`} title={<Space wrap><Tag color={a.role === "candidate" ? "orange" : "default"}>{a.role === "candidate" ? "实验候选 · 未上线" : a.role === "baseline" ? "同轮基线" : a.role === "historical" ? "修复前历史版本" : "归档结果"}</Tag>{a.label}</Space>} style={{marginTop:16}}>
          <Space wrap><Tag>{executionLabel(a.finish_reason)}</Tag><Tag>{a.elapsed_s.toFixed(2)} 秒</Tag></Space>
          {evidenceFlowNotices(a.evidence_flow).map(([notice], n) => <Alert key={n} type="warning" title={notice} style={{marginTop:8}} />)}
          {(a.role === "candidate" || data.paired) && <Alert type="info" title={`本题诊断：${a.notes}`} style={{marginTop:8}} />}
          <pre className="comparison-raw" data-testid="raw-answer">{answerText(a)}</pre>
          {!a.raw_text && <Alert type="warning" title={a.finish_reason === "budget_exceeded"
            ? "已选证据超出模型输入预算，本次没有生成回答。下方保留检索证据和执行记录。"
            : a.finish_reason === "retrieval_failed"
              ? "检索失败，本次没有生成回答。失败原因见下方执行记录。"
              : "本次没有生成回答，请查看执行状态和记录。"} />}
          {a.role !== "candidate" && !data.paired && <details><summary>历史审读说明（不代表当前版本状态）</summary><p>{a.notes}</p></details>}
          {a.funnel && <details><summary>分层过程与失败记录</summary>
            <p>原子事实 {a.funnel.facts?.length || 0} 条 · 对象字段核验 {a.funnel.cells?.length || 0} 项 · 维度汇总 {a.funnel.field_summaries?.length || 0} 项 · 阶段失败 {a.funnel.failures?.length || 0} 项</p>
            <details><summary>有效任务</summary><pre>{JSON.stringify(a.funnel.task, null, 2)}</pre></details>
            <details><summary>逐字事实与来源</summary><pre>{JSON.stringify(a.funnel.facts, null, 2)}</pre></details>
            <details><summary>逐对象逐字段核验</summary><pre>{JSON.stringify(a.funnel.cells, null, 2)}</pre></details>
            <details><summary>同维度汇总</summary><pre>{JSON.stringify(a.funnel.field_summaries, null, 2)}</pre></details>
            {a.funnel.conditions && <details><summary>生效硬条件判断</summary><pre>{JSON.stringify(a.funnel.conditions, null, 2)}</pre></details>}
            {a.funnel.candidates && <details><summary>候选资格</summary><pre>{JSON.stringify(a.funnel.candidates, null, 2)}</pre></details>}
            {!!a.funnel.decision && <details><summary>最终选择依据</summary><pre>{JSON.stringify(a.funnel.decision, null, 2)}</pre></details>}
            {!!a.funnel.call_budget && <details><summary>调用预算与未检查材料</summary><pre>{JSON.stringify(a.funnel.call_budget, null, 2)}</pre></details>}
            {!!a.funnel.failures?.length && <details><summary>失败节点</summary><pre>{JSON.stringify(a.funnel.failures, null, 2)}</pre></details>}
          </details>}
          {a.queries && <details open><summary>检索词与执行状态（{a.queries.length}条）</summary>
            <ol>{a.queries.map((q, j) => <li key={j}><Tag>{q.status}</Tag>{q.query}</li>)}</ol>
          </details>}
          <details><summary>已选证据（{a.sources.length}份）</summary>{a.sources.map((s,j)=><p key={j}><Button onClick={()=>setSource(s)}>{s.label} · 查看原文</Button> {s.url && <a href={s.url} target="_blank" rel="noreferrer">访问来源</a>}</p>)}</details>
          <details><summary>检索与生成记录</summary><pre>{JSON.stringify(a.trace,null,2)}</pre></details>
        </Card>)}
      </>}
    </>}
    <Drawer title={source?.label} open={!!source} onClose={()=>setSource(undefined)} size="large">
      {source?.url && <a href={source.url} target="_blank" rel="noreferrer">访问原始网页</a>}
      <pre className="comparison-source">{source?.text}</pre>
    </Drawer>
  </div>;
}
