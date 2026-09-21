import { useEffect, useState } from "react";
import { Alert, Button, Card, Drawer, Select, Space, Spin, Tag, Typography } from "antd";
import "./modelComparison.css";

type Source = { label: string; text: string; url?: string };
type Answer = { label: string; raw_text: string; finish_reason?: string; elapsed_s: number; notes: string; sources: Source[]; queries?: { query: string; status: string }[]; trace?: unknown };
type Case = { id: string; question: string; answers: Answer[] };
type Dataset = { title: string; summary: string; cases: Case[] };

export default function TestResultsPage() {
  const [suite, setSuite] = useState("github-natural-comparison-20260921");
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
      {value:"github-natural-comparison-20260921",label:"大型 GitHub 比较 · 自然问法"},
      {value:"github-project-comparison-paced-20260921",label:"大型 GitHub 比较 · 低频实时检索"},
      {value:"github-project-comparison-20260921",label:"大型 GitHub 比较 · 原联网失败记录"},
      {value:"live-comparison-20260921",label:"真实联网及知识库联合比较 · 12题"},
      {value:"writer-convergence-20260921",label:"Writer指令对照 · 188题双轮"}]} />
    {error && <Alert type="warning" title={error} />}
    {!data && !error && <Spin />}
    {data && <><h3>{data.title}</h3><Alert type="info" title={data.summary} />
      <Space wrap className="comparison-controls">
        <Button disabled={index === 0} onClick={() => setIndex(index-1)}>上一题</Button>
        <Select aria-label="测试题目" value={index} onChange={setIndex} style={{width:"min(760px,65vw)"}} options={data.cases.map((c,i)=>({value:i,label:`${i+1}. ${c.question}`}))} />
        <Button disabled={index >= data.cases.length-1} onClick={() => setIndex(index+1)}>下一题</Button>
        <a href={`${import.meta.env.BASE_URL}experiments/${suite}.json`} download>下载全部回答与证据</a>
      </Space>
      {current && <><Card title={current.id}><Typography.Title level={4}>{current.question}</Typography.Title></Card>
        {current.answers.map((a,i)=><Card key={i} title={a.label} style={{marginTop:16}}>
          <Space><Tag>{a.finish_reason || "见执行记录"}</Tag><Tag>{a.elapsed_s.toFixed(2)} 秒</Tag></Space>
          <pre className="comparison-raw" data-testid="raw-answer">{answerText(a)}</pre>
          {!a.raw_text && <Alert type="warning" title={a.finish_reason === "budget_exceeded"
            ? "已选证据超出模型输入预算，本次没有生成回答。下方保留检索证据和执行记录。"
            : a.finish_reason === "retrieval_failed"
              ? "检索失败，本次没有生成回答。失败原因见下方执行记录。"
              : "本次没有生成回答，请查看执行状态和记录。"} />}
          <Alert type="info" title={a.notes} />
          {a.queries && <details open><summary>检索词与执行状态（{a.queries.length}条）</summary>
            <ol>{a.queries.map((q, j) => <li key={j}><Tag>{q.status}</Tag>{q.query}</li>)}</ol>
          </details>}
          <details><summary>已选证据（{a.sources.length}份）</summary>{a.sources.map((s,j)=><p key={j}><Button onClick={()=>setSource(s)}>{s.label} · 查看原文</Button> {s.url && <a href={s.url} target="_blank" rel="noreferrer">访问来源</a>}</p>)}</details>
          <details><summary>完整检索与生成记录</summary><pre>{JSON.stringify(a.trace,null,2)}</pre></details>
        </Card>)}
      </>}
    </>}
    <Drawer title={source?.label} open={!!source} onClose={()=>setSource(undefined)} size="large">
      {source?.url && <a href={source.url} target="_blank" rel="noreferrer">访问原始网页</a>}
      <pre className="comparison-source">{source?.text}</pre>
    </Drawer>
  </div>;
}
