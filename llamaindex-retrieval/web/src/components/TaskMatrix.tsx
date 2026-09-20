import { Alert, Button, Table, Typography } from "antd";
import type { AskResponse } from "../types";
import { useLanguage } from "../i18n";

type Row = { id: string; object: string; dimension: string; question: string; status: string; source_ids: string[]; assessment_status?: string };
export default function TaskMatrix({ response, onSource }: { response: AskResponse; onSource: (id: string) => void }) {
  const { tr } = useLanguage();
  const raw = response.retrieval.task_matrix;
  if (!Array.isArray(raw)) return null;
  const rows = raw.filter((r): r is Row => r && typeof r === "object" && typeof r.id === "string"
    && typeof r.object === "string" && typeof r.dimension === "string" && typeof r.question === "string"
    && typeof r.status === "string" && Array.isArray(r.source_ids) && r.source_ids.every((id: unknown) => typeof id === "string"));
  const labels: Record<string, string> = { supported: tr("模型找到支持证据", "Model found supporting evidence"),
    missing: tr("证据仍不足", "Evidence still missing"), conflict: tr("来源存在冲突", "Sources conflict") };
  const review = response.retrieval.answer_review as { status?: string; cells?: { cell_id: string; valid: boolean | null }[]; model_judgment?: { valid?: boolean; issues?: unknown[] } } | undefined;
  const plan = response.retrieval.plan as { conditions?: unknown[] } | undefined;
  const conditions = Array.isArray(plan?.conditions) ? plan.conditions.filter((c): c is string => typeof c === "string") : [];
  return <section aria-label={tr("对比项目与证据", "Comparison items and evidence")}>
    <Typography.Title level={5}>{tr("对比项目与证据", "Comparison items and evidence")}</Typography.Title>
    <Typography.Paragraph type="secondary">{tr("以下状态来自模型检查，不代表事实已独立核验。点击来源可查看本次采用的原文。", "These are model judgments, not independent fact verification. Open sources to inspect the evidence used.")}</Typography.Paragraph>
    {conditions.length > 0 && <Typography.Paragraph>{tr("计划采用的条件：", "Planned conditions: ")}{conditions.join("；")}</Typography.Paragraph>}
    <Table size="small" pagination={false} rowKey="id" dataSource={rows} scroll={{ x: 580 }} columns={[
      { title: tr("对象／版本", "Object / version"), dataIndex: "object" },
      { title: tr("对比维度", "Dimension"), dataIndex: "dimension" },
      { title: tr("证据状态", "Evidence status"), dataIndex: "status", render: (s: string, row) => row.assessment_status === "failed"
        ? tr("证据检查失败", "Evidence assessment failed") : labels[s] ?? tr("状态未知", "Unknown") },
      { title: tr("回答核验", "Answer review"), render: (_, row) => {
        const check = Array.isArray(review?.cells) ? review.cells.find(c => c.cell_id === row.id) : undefined;
        return check?.valid === true ? tr("模型检查通过", "Passed model review") : check?.valid === false
          ? tr("模型检查未通过", "Failed model review") : tr("尚未完成", "Not completed");
      } },
      { title: tr("原文", "Sources"), render: (_, row) => row.source_ids.map(id => {
        const source = response.sources.find(s => s.id === id);
        return <Button key={id} type="link" disabled={!source} onClick={() => onSource(id)}>{source?.title || tr("来源不可用", "Source unavailable")}</Button>;
      }) },
    ]} />
    {review?.model_judgment?.valid === false && <Alert type="warning" showIcon
      title={tr("模型检查指出以下问题", "Model review reported issues")}
      description={<ul>{review.model_judgment.issues?.filter((i): i is string => typeof i === "string").map((issue, i) => <li key={i}>{issue}</li>)}</ul>} />}
  </section>;
}
