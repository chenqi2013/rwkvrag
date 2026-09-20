import { Alert, Button, Card, Drawer, Empty, Form, Input, Select, Space, Tag, Typography, message } from "antd";
import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { exactSpan } from "../atomicEvidence";
import type { AtomicClaim, AtomicRequest, AtomicRun, AtomicRunSummary } from "../atomicEvidence";
import type { KnowledgeBase } from "../types";
import { errorMessage } from "../utils";
import { useLanguage } from "../i18n";

export default function AtomicEvidencePage() {
  const { tr } = useLanguage();
  const [bases, setBases] = useState<KnowledgeBase[]>([]);
  const [kb, setKb] = useState<string>();
  const [available, setAvailable] = useState<boolean>();
  const [busy, setBusy] = useState(false);
  const [run, setRun] = useState<AtomicRun>();
  const [history, setHistory] = useState<AtomicRunSummary[]>([]);
  const [selected, setSelected] = useState<AtomicClaim>();
  const [form] = Form.useForm<AtomicRequest>();
  const epoch = useRef(0);
  useEffect(() => {
    let live = true;
    Promise.all([api.knowledgeBases(), api.atomicCapabilities()]).then(([b, c]) => {
      if (live) { setBases(b); setAvailable(c.available); }
    }).catch(e => { if (live) void message.error(errorMessage(e)); });
    return () => { live = false; epoch.current++; };
  }, []);
  useEffect(() => {
    const current = ++epoch.current;
    setRun(undefined); setSelected(undefined); setHistory([]);
    if (kb) api.atomicHistory(kb).then(h => { if (current === epoch.current) setHistory(h); })
      .catch(e => { if (current === epoch.current) void message.error(errorMessage(e)); });
  }, [kb]);
  const inspect = async () => {
    if (!kb) return;
    const values = await form.validateFields();
    const current = ++epoch.current;
    setBusy(true); setSelected(undefined); setRun(undefined);
    try {
      const next = await api.atomicInspect(kb, { ...values, conditions: values.conditions || "" });
      if (current === epoch.current) setRun(next);
      const h = await api.atomicHistory(kb);
      if (current === epoch.current) setHistory(h);
    } catch (e) { if (current === epoch.current) void message.error(errorMessage(e)); }
    finally { if (current === epoch.current) setBusy(false); }
  };
  const openHistory = async (id: string) => {
    if (!kb) return;
    const current = ++epoch.current;
    setBusy(true); setSelected(undefined);
    try { const next = await api.atomicDetail(kb, id); if (current === epoch.current) setRun(next); }
    catch (e) { if (current === epoch.current) void message.error(errorMessage(e)); }
    finally { if (current === epoch.current) setBusy(false); }
  };
  const source = run?.sources.find(s => s.id === selected?.binding.source_id);
  const position = selected?.statement_position;
  const parent = source?.metadata.context_spans;
  const parentText = position?.origin === "saved_source_metadata" && Array.isArray(parent)
    && typeof position.context_index === "number" ? parent[position.context_index]?.text : undefined;
  const referenceText = position?.origin ? (typeof parentText === "string" ? parentText : undefined) : source?.snippet;
  const span = referenceText !== undefined && position && selected ? exactSpan(referenceText, position.start, position.end, selected.statement_quote) : undefined;
  return <div className="page-stack atomic-evidence-page">
    <Typography.Title level={2}>{tr("单项证据核对", "Check one property")}</Typography.Title>
    <Typography.Paragraph>{tr("指定对象和一个属性，逐条查看来源中的记载。不同来源分别保留；保留完整短原文，不自动推断数值、未记载或最终结论。", "Choose an object and one property to inspect source claims. Sources stay separate; conflicts have not been adjudicated.")}</Typography.Paragraph>
    {available === false && <Alert type="info" showIcon title={tr("核对服务尚未启用，仍可查看已保存的记录。", "Checking is not enabled. Saved records remain available.")} />}
    <Card>
      <Form form={form} layout="vertical" initialValues={{ conditions: "" }} onFinish={() => void inspect()}>
        <Form.Item label={tr("知识库", "Knowledge base")} required><Select aria-label={tr("知识库", "Knowledge base")} value={kb} disabled={busy}
          onChange={value => { epoch.current++; setKb(value); }} options={bases.map(b => ({ value: b.id, label: b.name }))} /></Form.Item>
        <Form.Item name="object" label={tr("对象", "Object")} rules={[{ required: true, whitespace: true }]}><Input maxLength={120} placeholder={tr("例如：设备型号或产品版本", "A device model or product version")} /></Form.Item>
        <Form.Item name="attribute" label={tr("一个属性", "One property")} rules={[{ required: true, whitespace: true }]}><Input maxLength={120} placeholder={tr("例如：额定功率", "For example: rated power")} /></Form.Item>
        <Form.Item name="conditions" label={tr("适用条件（可选）", "Conditions (optional)")}><Input maxLength={300} placeholder={tr("例如：标准模式、2025 年", "For example: standard mode, 2025")} /></Form.Item>
        <Button htmlType="submit" type="primary" loading={busy} disabled={!kb || !available}>{tr("查找并核对证据", "Find and check evidence")}</Button>
      </Form>
    </Card>
    {run && <Card title={tr("属性相关的原文", "Property-related excerpts")}>
      <Space direction="vertical" style={{ width: "100%" }}>
        <Alert type={run.status === "completed" ? "info" : "warning"} showIcon
          title={run.status === "completed" ? tr("本次处理完成，语义仍待核验", "Processing completed; semantics are not independently verified") : tr("本次处理未完整完成", "Processing is incomplete")}
          description={tr("结果限于本次检索和检查的片段。未找到主张不代表不存在；处理失败不代表资料不足。", "Results cover only retrieved and examined excerpts. No claim found does not prove absence; a processing failure does not establish missing evidence.")} />
        {run.freshness && run.freshness !== "same_index_version" && <Alert type="warning" title={tr("当前索引已变化或无法核实，以下展示当时保存的原文。", "The index changed or could not be checked. The saved source text is shown below.")} />}
        {!run.claims.length && <Empty description={tr("本次未选中可展示的片段", "No displayable excerpts were selected")} />}
        {run.claims.map(claim => <Card size="small" key={claim.id} style={{ width: "100%" }}>
          <Space wrap><Tag>{claim.kind === "unconfirmed" ? tr("相关性待确认", "Relevance unconfirmed") : tr("Reader 选中的原文", "Reader-selected excerpt")}</Tag>
            <Typography.Text strong>{claim.target.object} · {claim.target.attribute}</Typography.Text></Space>
          {claim.value_quote !== null && <Typography.Paragraph className="result-snippet">{claim.value_quote}</Typography.Paragraph>}
          {claim.scope_quote && <Typography.Paragraph>{tr("原文条件", "Source conditions")}：{claim.scope_quote}</Typography.Paragraph>}
          {claim.value_quote === null && <Typography.Paragraph className="citation-full-text">{claim.statement_quote}</Typography.Paragraph>}
          <Button onClick={() => setSelected(claim)}>{tr("查看原文位置", "View source location")} · {run.sources.find(s => s.id === claim.binding.source_id)?.title || tr("来源", "Source")}</Button>
        </Card>)}
        <details><summary>{tr("检查范围与处理记录", "Coverage and processing records")}</summary><pre className="trace-json">{JSON.stringify({ coverage: run.coverage, issues: run.issues, calls: run.calls }, null, 2)}</pre></details>
      </Space>
    </Card>}
    <Card title={tr("已保存的核对", "Saved checks")}>
      {history.map(h => <div key={h.id}><Button type="link" disabled={busy} onClick={() => void openHistory(h.id)}>{h.request.object} · {h.request.attribute}</Button><Tag>{h.status}</Tag><Typography.Text type="secondary">{new Date(h.created_at).toLocaleString()}</Typography.Text></div>)}
      {!history.length && <Empty description={tr("暂无记录", "No saved checks")} />}
    </Card>
    <Drawer open={!!selected} onClose={() => setSelected(undefined)} width="min(760px, 100vw)" title={tr("原文位置与版本", "Source location and version")}>
      {selected && source && span ? <>
        <Typography.Title level={4}>{source.title}</Typography.Title>
        <Typography.Paragraph>{tr("这是本次核对保存的来源快照，标记处为该主张使用的原文。", "This is the source snapshot saved for this check. The excerpt used by this claim is highlighted.")}</Typography.Paragraph>
        <div className="citation-full-text" data-testid="atomic-source">{span.before}<mark>{span.quote}</mark>{span.after}</div>
        {selected.evidence.context.length > 0 && <><Typography.Title level={5}>{tr("随附上下文", "Attached context")}</Typography.Title>{selected.evidence.context.map((c, i) => <p className="citation-full-text" key={i}>{c.text}</p>)}</>}
        <details><summary>{tr("来源标识与版本", "Source identity and version")}</summary><pre className="trace-json">{JSON.stringify(selected.binding, null, 2)}</pre></details>
      </> : <Alert type="error" title={tr("原文位置校验失败，不能展示为已定位的证据。", "Source offset verification failed; this excerpt cannot be shown as verified.")} />}
    </Drawer>
  </div>;
}
