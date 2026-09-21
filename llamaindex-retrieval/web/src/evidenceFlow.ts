export type EvidenceFlow = {
  protocol: string; state: string; input_source_count: number | null; writer_source_count: number | null;
  failed_node_count: number; unexamined_job_count: number; failed_cell_count: number;
  verified_fact_count: number; verified_unselected_fact_ids: string[];
};

export function evidenceFlowNotices(flow?: EvidenceFlow): [string, string][] {
  if (!flow || flow.protocol !== "evidence-flow-v1") return [];
  const messages: [string, string][] = [];
  if (flow.state === "no_input_materials") messages.push([
    "本次没有材料进入分层处理；这不证明知识库或网络上没有答案。",
    "No materials entered this funnel; this does not establish absence in the knowledge base or web."]);
  if (flow.state === "no_selected_evidence") messages.push([
    `有 ${flow.input_source_count} 份材料进入处理，但没有形成最终可引用证据。不能据此断言原文未记载或产品不支持。`,
    `${flow.input_source_count} materials entered processing, but none reached the final evidence set. This does not mean the sources omit the answer or a product lacks the feature.`]);
  if (flow.state === "unrecorded" || flow.state === "inconsistent_trace") messages.push([
    "证据流转记录缺失或不一致，无法确认材料在哪个阶段丢失。",
    "Evidence flow is missing or inconsistent; the stage of evidence loss is unconfirmed."]);
  if (flow.failed_node_count || flow.failed_cell_count) messages.push([
    `处理记录包含 ${flow.failed_node_count} 个失败节点、${flow.failed_cell_count} 个失败字段。执行失败不等于事实未知。`,
    `Processing recorded ${flow.failed_node_count} failed nodes and ${flow.failed_cell_count} failed cells. Execution failure is not a factual unknown.`]);
  if (flow.unexamined_job_count) messages.push([
    `有 ${flow.unexamined_job_count} 项抽取任务未检查，比较覆盖不完整。`,
    `${flow.unexamined_job_count} extraction jobs were not examined; comparison coverage is incomplete.`]);
  if (flow.verified_unselected_fact_ids.length) messages.push([
    `有 ${flow.verified_unselected_fact_ids.length} 个通过节点核验的事实未被字段归并层选中；请检查是否遗漏不同记录。`,
    `${flow.verified_unselected_fact_ids.length} node-verified facts were not selected by the cell stage; check for omitted records.`]);
  return messages;
}

export function executionLabel(status?: string): string {
  return ({completed: "执行完成 · 不代表回答正确", length: "输出触顶 · 回答未完成",
    funnel_partial_failure: "部分处理失败 · 需核对来源", invalid_response: "响应格式无效",
    budget_exceeded: "输入超预算 · 未生成", retrieval_failed: "检索失败 · 未生成"} as Record<string,string>)[status || ""]
    || (status ? `执行状态：${status}` : "执行记录不完整");
}
