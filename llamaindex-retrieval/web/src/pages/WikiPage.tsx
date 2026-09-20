import { Alert, Button, Card, Collapse, Drawer, Empty, message, Select, Space, Table, Tag, Typography } from "antd";
import { useCallback, useEffect, useState } from "react";
import CitedAnswer from "../components/CitedAnswer";
import { api } from "../api";
import type { WikiVersion } from "../types";
import { errorMessage, formatDate } from "../utils";
import { useLanguage } from "../i18n";

export default function WikiPage() {
  const { tr } = useLanguage();
  const [pages, setPages] = useState<WikiVersion[]>([]);
  const [selected, setSelected] = useState<WikiVersion>();
  const [history, setHistory] = useState<WikiVersion[]>([]);
  const [loading, setLoading] = useState(false);
  const load = useCallback(async () => {
    setLoading(true);
    try { setPages(await api.wikiPages()); }
    catch (error) { void message.error(errorMessage(error)); }
    finally { setLoading(false); }
  }, []);
  useEffect(() => { void load(); const timer = window.setInterval(() => void load(), 10000); return () => window.clearInterval(timer); }, [load]);
  const open = async (item: WikiVersion) => {
    try {
      const [detail, versions] = await Promise.all([api.wikiVersion(item.id), api.wikiHistory(item.page_id)]);
      setSelected(detail); setHistory(versions);
    } catch (error) { void message.error(errorMessage(error)); }
  };
  const generate = async (item: WikiVersion) => {
    try { await api.generateWiki(item.page_id); void message.success(tr("生成任务已提交，可在导入任务查看进度", "Generation queued; see Imports for progress")); }
    catch (error) { void message.error(errorMessage(error)); }
  };
  const freshness = (item: WikiVersion) => item.freshness === "current"
    ? tr("来源未变化", "Sources unchanged") : item.freshness === "unknown"
      ? tr("来源状态待确认", "Source status unknown") : tr("来源已变化或不可用", "Sources changed or unavailable");
  return <div className="page-stack">
    <div className="page-heading"><Typography.Title level={2}>Wiki</Typography.Title><Button onClick={() => void load()}>{tr("刷新", "Refresh")}</Button></div>
    <Alert type="info" showIcon message={tr("自动生成的文档 Wiki 草稿", "Automatically generated document Wiki drafts")}
      description={tr("文档入库或更新后自动生成。草稿尚未人工审核；来源变化后需重新生成。Wiki 不会替代原文成为问答证据。", "Generated after documents are indexed or revised. Drafts are unreviewed and require regeneration when sources change. Original sources remain the evidence for answers.")} />
    <Card><Table rowKey="id" dataSource={pages} loading={loading} pagination={{pageSize: 10}}
      locale={{emptyText: <Empty description={tr("尚无 Wiki 页面，请先上传文档；生成失败可在任务列表查看原因。", "No Wiki pages yet. Upload a document; generation errors appear in Imports.")} />}}
      columns={[
        {title: tr("页面", "Page"), dataIndex: "title"},
        {title: tr("状态", "Status"), render: (_, item) => <Space direction="vertical"><Tag color={item.status === "draft" ? "gold" : "red"}>{item.status === "draft" ? tr("待审核草稿", "Unreviewed draft") : tr("生成失败", "Generation failed")}</Tag><Typography.Text type={item.freshness === "current" ? "secondary" : "warning"}>{freshness(item)}</Typography.Text></Space>},
        {title: tr("请求时间", "Requested"), dataIndex: "created_at", render: formatDate},
        {title: tr("操作", "Actions"), render: (_, item) => <Space><Button onClick={() => void open(item)}>{tr("查看与历史", "View and history")}</Button><Button onClick={() => void generate(item)}>{tr("重新生成", "Regenerate")}</Button></Space>},
      ]} /></Card>
    <Drawer open={Boolean(selected)} onClose={() => setSelected(undefined)} width="min(780px, 100vw)" title={selected?.title}>
      {selected && <Space direction="vertical" size="large" style={{width: "100%"}}>
        <Alert showIcon type={selected.freshness === "current" ? "warning" : "error"} message={freshness(selected)} description={tr("此页面是未经审核的模型输出，请对照来源核实。", "This model-generated page is unreviewed. Verify it against the sources.")} />
        <Select style={{width: "100%"}} value={selected.id} options={history.map(v => ({value: v.id, label: `${formatDate(v.created_at)} · ${v.status}`}))} onChange={id => { const version = history.find(v => v.id === id); if (version) void open(version); }} />
        {selected.response ? <CitedAnswer key={selected.id} text={selected.body || ""} response={selected.response} />
          : <Empty description={tr("没有保存来源记录", "No saved source record")} />}
        <Typography.Text type="secondary" copyable>{selected.binding.revision.source_sha256}</Typography.Text>
        <Collapse style={{width: "100%"}} items={[{key: "raw", label: tr("原始模型输出（未改写）", "Original model output"), children: <pre style={{whiteSpace: "pre-wrap"}}>{selected.response?.answer}</pre>}]} />
      </Space>}
    </Drawer>
  </div>;
}
