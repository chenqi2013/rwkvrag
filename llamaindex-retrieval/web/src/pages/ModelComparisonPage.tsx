import { useEffect, useState } from "react";
import { Alert, Button, Card, Drawer, Input, Select, Space, Spin, Tag, Typography } from "antd";
import "./modelComparison.css";

type Answer = {
  arm: string; round: number; raw_text: string; finish_reason: string; elapsed_ms: number;
  raw_text_sha256: string; strict_pass: boolean;
  review: { facts_complete: boolean; citations_correct: boolean; repetition_confirmed: boolean; notes: string };
};
type Case = {
  ordinal: number; id: string; category: string; question: string; history: unknown[];
  evidence: { label: string; text: string }[]; answers: Answer[];
};
type Dataset = { cases: Case[] };
const labels: Record<string, string> = { comparison: "对比", choice: "选择", mixed: "混合", regression: "普通回归", ordinary: "普通问答" };
const categoryLabel = (value: string) => labels[value] || `历史类别 ${value}`;

export default function ModelComparisonPage() {
  const [data, setData] = useState<Dataset>();
  const [error, setError] = useState("");
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState("all");
  const [category, setCategory] = useState("all");
  const [round, setRound] = useState(1);
  const [ordinal, setOrdinal] = useState(5);
  const [source, setSource] = useState<{ label: string; text: string }>();
  useEffect(() => {
    const controller = new AbortController();
    fetch(`${import.meta.env.BASE_URL}experiments/model-size-paired-20260921.json`, { signal: controller.signal })
      .then(r => { if (!r.ok) throw new Error(`数据加载失败：HTTP ${r.status}`); return r.json(); })
      .then(setData).catch(e => { if (e.name !== "AbortError") setError(String(e)); });
    return () => controller.abort();
  }, []);
  if (error) return <Alert type="error" title={error} />;
  if (!data) return <Spin description="加载全部测试记录…" />;
  const visible = data.cases.filter(c => {
    const a = c.answers.filter(a => a.round === round);
    return (category === "all" || c.category === category) &&
      `${c.ordinal} ${c.id} ${c.question}`.toLowerCase().includes(query.toLowerCase()) &&
      (filter === "all" || (filter === "repeat" && a.some(a => a.review.repetition_confirmed)) ||
        (filter === "repeat72" && a.some(a => a.arm === "7.2b" && a.review.repetition_confirmed)) ||
        (filter === "fail" && a.some(a => !a.strict_pass)) ||
        (filter === "regressed" && a.find(a => a.arm === "2.9b")?.review.facts_complete && !a.find(a => a.arm === "7.2b")?.review.facts_complete));
  });
  const current = visible.find(c => c.ordinal === ordinal) || visible[0];
  const index = visible.findIndex(c => c.ordinal === current?.ordinal);
  function rawAnswer(a: Answer) {
    return a.raw_text.split(/(\[资料\s*\d+\])/g).map((part, i) => {
      const m = /^\[资料\s*(\d+)\]$/.exec(part);
      if (!m) return part;
      const evidence = current.evidence.find(e => e.label.replace(/\s/g, "") === `资料${Number(m[1])}`);
      return <button key={i} className={`comparison-citation ${evidence ? "" : "missing"}`}
        onClick={() => setSource(evidence || { label: part, text: "这条引用在本题输入中不存在。原始回答保留，未补齐来源。" })}>{part}</button>;
    });
  }
  return <div className="model-comparison">
    <Typography.Title level={3}>2.9B / 7.2B 模型对照</Typography.Title>
    <p>2026-09-21 · 已完成测试回放 · 188 条题目记录 · 两轮共 752 条原始回答</p>
    <Alert type="info" showIcon title="固定材料直接回答，两模型均零 State；不是当前在线检索结果。"
      description="包含 180 个不同输入和已见开发题。判读由本任务实现者完成，未经独立复核；事实正确、引用正确和持续复读分别记录。" />
    <div className="comparison-grid">
      {["2.9b", "7.2b"].map(arm => {
        const answers = data.cases.flatMap(c => c.answers.filter(a => a.arm === arm && a.round === round));
        const repeats = answers.filter(a => a.review.repetition_confirmed).length;
        return <Card key={arm} title={`G1j ${arm.toUpperCase()} · 第 ${round} 轮`}>
          <div className="comparison-metrics"><span>事实完整正确 <strong>{answers.filter(a => a.review.facts_complete).length}/188</strong></span>
            <span>严格通过 <strong>{answers.filter(a => a.strict_pass).length}/188</strong></span>
            <span>持续复读 <strong className="repeat-count">{repeats}/188（{(repeats / 188 * 100).toFixed(1)}%）</strong></span></div>
        </Card>;
      })}
    </div>
    <Space wrap className="comparison-controls">
      <Select aria-label="测试轮次" value={round} onChange={setRound} options={[{value:1,label:"第 1 轮"},{value:2,label:"第 2 轮"}]} />
      <Select aria-label="题目类型" value={category} onChange={setCategory} style={{width:140}}
        options={[{value:"all",label:"全部题型"}, ...Array.from(new Set(data.cases.map(c => c.category))).map(value => ({value,label:categoryLabel(value)}))]} />
      <Select aria-label="问题筛选" value={filter} onChange={setFilter} style={{width:190}} options={[
        {value:"all",label:"全部结果"},{value:"repeat",label:"任一模型持续复读"},{value:"repeat72",label:"7.2B 持续复读"},
        {value:"fail",label:"任一模型严格未通过"},{value:"regressed",label:"7.2B 事实判定退步"}]} />
      <Input aria-label="搜索题目" placeholder="搜索题目、ID 或编号" value={query} onChange={e => setQuery(e.target.value)} allowClear style={{width:260}} />
      <Tag>{visible.length} 条题目</Tag>
      <a href={`${import.meta.env.BASE_URL}experiments/model-size-paired-20260921.json`} download>下载全部展示数据</a>
    </Space>
    {!current ? <Alert title="没有匹配的题目" type="info" /> : <>
      <Space wrap className="comparison-controls">
        <Button disabled={index <= 0} onClick={() => setOrdinal(visible[index-1].ordinal)}>上一题</Button>
        <Select aria-label="选择题目" value={current.ordinal} onChange={setOrdinal} style={{width:"min(700px, 65vw)"}}
          options={visible.map(c => ({value:c.ordinal,label:`#${c.ordinal} · ${c.question}`}))} />
        <Button disabled={index >= visible.length-1} onClick={() => setOrdinal(visible[index+1].ordinal)}>下一题</Button>
      </Space>
      <Card title={`#${current.ordinal} · ${categoryLabel(current.category)}`}>
        <Typography.Title level={4}>{current.question}</Typography.Title>
        <Typography.Text type="secondary">{current.id} · 编号从 0 开始，与审读报告一致</Typography.Text>
        {current.history.length > 0 && <details><summary>完整历史上下文</summary><pre>{JSON.stringify(current.history, null, 2)}</pre></details>}
        <details><summary>输入材料（{current.evidence.length} 份，点击展开全文）</summary>
          {current.evidence.map((e, i) => <div key={i}><h4>{e.label}</h4><pre>{e.text}</pre></div>)}
          {!current.evidence.length && <p>本题没有输入材料。</p>}
        </details>
      </Card>
      <div className="comparison-grid">
        {["2.9b", "7.2b"].map(arm => {
          const a = current.answers.find(a => a.arm === arm && a.round === round)!;
          return <Card key={arm} title={`${arm.toUpperCase()} 原始回答`}>
            <Space wrap><Tag color={a.review.facts_complete ? "green" : "red"}>事实{a.review.facts_complete ? "通过" : "未通过"}</Tag>
              <Tag color={a.review.citations_correct ? "green" : "orange"}>引用{a.review.citations_correct ? "通过" : "未通过"}</Tag>
              <Tag color={a.review.repetition_confirmed ? "red" : "default"}>{a.review.repetition_confirmed ? "持续复读" : "无持续复读"}</Tag>
              <Tag>{a.finish_reason === "length" ? "到达输出上限" : "正常结束"}</Tag></Space>
            <pre className="comparison-raw" data-arm={arm}>{rawAnswer(a)}</pre>
            <p className="comparison-note">审读：{a.review.notes}</p>
            <small>模型调用 {(a.elapsed_ms/1000).toFixed(2)} 秒 · 点击原文中的引用查看材料；不代表引用语义正确</small>
            <details><summary>原文校验值</summary><code>{a.raw_text_sha256}</code></details>
          </Card>;
        })}
      </div>
    </>}
    <Drawer title={source?.label} open={!!source} onClose={() => setSource(undefined)} size="large"><pre className="comparison-source">{source?.text}</pre></Drawer>
  </div>;
}
