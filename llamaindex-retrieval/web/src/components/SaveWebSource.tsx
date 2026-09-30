import { Alert, Button, Modal, Select, Space, Spin, Typography } from "antd";
import { useId, useRef, useState } from "react";
import { api } from "../api";
import { externalSourceUrl } from "../citations";
import { useLanguage } from "../i18n";
import type { KnowledgeBase, SavedWebSnapshot, SearchResult, WebSnapshot } from "../types";
import { errorMessage } from "../utils";
import { isWebSource, snapshotSaveIssue, snapshotSaveRequest, webSnapshotId } from "../webSnapshots";

/** Opening this control only reads a receipt. Saving requires a separate user click. */
export default function SaveWebSource({ source }: { source: SearchResult }) {
  const { tr } = useLanguage();
  const selectId = useId();
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [snapshot, setSnapshot] = useState<WebSnapshot>();
  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBase[]>([]);
  const [knowledgeBaseId, setKnowledgeBaseId] = useState<string>();
  const [error, setError] = useState<string>();
  const [result, setResult] = useState<SavedWebSnapshot>();
  const requestVersion = useRef(0);
  const savingRef = useRef(false);
  const snapshotId = webSnapshotId(source);

  if (!isWebSource(source)) return null;
  if (!snapshotId) return <Typography.Text type="secondary">
    {tr("这条来源没有可核验的抓取快照，重新检索后才能保存到知识库。", "This source has no verifiable saved snapshot. Search again before saving it to a knowledge base.")}
  </Typography.Text>;

  const close = () => {
    if (savingRef.current) return;
    requestVersion.current += 1;
    setOpen(false);
    setSnapshot(undefined);
  };
  const preview = async () => {
    const version = ++requestVersion.current;
    setOpen(true);
    setLoading(true);
    setError(undefined);
    setResult(undefined);
    setSnapshot(undefined);
    setKnowledgeBaseId(undefined);
    setKnowledgeBases([]);
    try {
      const [receipt, bases] = await Promise.all([api.webSnapshot(snapshotId), api.knowledgeBases()]);
      if (requestVersion.current !== version) return;
      setSnapshot(receipt);
      setKnowledgeBases(bases);
    } catch (failure) {
      if (requestVersion.current === version) setError(errorMessage(failure));
    } finally {
      if (requestVersion.current === version) setLoading(false);
    }
  };
  const confirm = async () => {
    if (savingRef.current || !open || !snapshot || !knowledgeBases.some(kb => kb.id === knowledgeBaseId)) return;
    savingRef.current = true;
    setSaving(true);
    setError(undefined);
    try {
      // Send the receipt identity/hash only; neither answer nor editable text is accepted.
      const values = snapshotSaveRequest(source, snapshot, knowledgeBaseId, true);
      setResult(await api.saveWebSnapshot(snapshotId, values));
    } catch (failure) {
      setError(errorMessage(failure));
    } finally {
      savingRef.current = false;
      setSaving(false);
    }
  };
  const issue = snapshot ? snapshotSaveIssue(source, snapshot) : undefined;
  const link = externalSourceUrl(snapshot?.url);
  const domain = link ? new URL(link).hostname : "";
  const issueLabels: Record<string, [string, string]> = {
    identity_mismatch: ["快照与所选来源不一致，无法保存。", "The snapshot does not match the selected source and cannot be saved."],
    not_full_text: ["这条来源只有摘要或未成功抓取正文，不能保存为知识文档。", "This source is a snippet or has no successfully fetched body. It cannot be saved as a knowledge document."],
    not_allowed: ["服务器未允许保存这份快照。", "The server has not allowed this snapshot to be saved."],
    empty_text: ["快照没有正文，无法保存。", "The snapshot has no body and cannot be saved."],
    invalid_hash: ["快照缺少有效的内容校验，无法保存。", "The snapshot has no valid content hash and cannot be saved."],
    invalid_url: ["快照缺少有效的来源网址，无法保存。", "The snapshot has no valid source URL and cannot be saved."],
  };
  const failed = result?.status === "failed";
  const done = result?.status === "completed";
  return <>
    <Button size="small" onClick={() => void preview()}>{tr("保存到知识库", "Save to knowledge base")}</Button>
    <Modal open={open} onCancel={close} closable={!saving} maskClosable={!saving} keyboard={!saving}
      title={tr("预览并保存来源快照", "Preview and save source snapshot")} width={820}
      footer={<Space>
        <Button disabled={saving} onClick={close}>{result ? tr("关闭", "Close") : tr("取消", "Cancel")}</Button>
        {!result && <Button type="primary" loading={saving}
          disabled={loading || !snapshot || !!issue || !knowledgeBaseId}
          onClick={() => void confirm()}>{tr("确认保存抓取快照", "Confirm saving this snapshot")}</Button>}
      </Space>}>
      <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
        {loading && <Spin aria-label={tr("正在读取来源快照", "Loading source snapshot")} />}
        {error && <Alert type="error" showIcon title={tr("无法保存来源", "Cannot save source")} description={error} />}
        {snapshot && <>
          <Typography.Title level={5}>{snapshot.title || source.title}</Typography.Title>
          <Typography.Text>{tr("来源域名", "Source domain")} · {domain || tr("不可用", "Unavailable")}</Typography.Text>
          {link && <Typography.Link href={link} target="_blank" rel="noopener noreferrer">{link}</Typography.Link>}
          <Typography.Text>{tr("抓取时间", "Retrieved")} · {snapshot.retrieved_at || tr("未记录", "Not recorded")}</Typography.Text>
          <Typography.Text>{tr("检索提供方", "Search provider")} · {snapshot.provider || tr("未记录", "Not recorded")}</Typography.Text>
          {issue && <Alert type="warning" showIcon title={tr(...issueLabels[issue])} description={snapshot.reason} />}
          <Typography.Text strong>{snapshot.content_status === "fetched"
            ? tr("本次保存的抓取全文快照", "Full-text snapshot saved for this retrieval")
            : tr("已获取的来源内容（不可入库）", "Retrieved source content (cannot be saved)")}</Typography.Text>
          <Typography.Paragraph className="citation-full-text" style={{ maxHeight: 340, overflow: "auto", whiteSpace: "pre-wrap" }}>
            {snapshot.text}
          </Typography.Paragraph>
          <Typography.Text type="secondary">{tr("请核对正文是否完整、相关。保存的是当时抓取的网页文本，不包含模型答案，也不会重新检索网页。", "Check that the text is complete and relevant. This saves the captured page text, not the model answer, without searching again.")}</Typography.Text>
          {!result && <>
            <label htmlFor={selectId}>{tr("目标知识库", "Target knowledge base")}</label>
            <Select id={selectId} style={{ width: "100%" }} value={knowledgeBaseId} disabled={!!issue || saving}
              placeholder={tr("请选择要保存到的知识库", "Choose a knowledge base")}
              options={knowledgeBases.map(kb => ({ value: kb.id, label: kb.name }))} onChange={setKnowledgeBaseId} />
            {!knowledgeBases.length && <Typography.Text type="warning">{tr("没有可用知识库，请先创建知识库。", "No knowledge base is available. Create one first.")}</Typography.Text>}
          </>}
        </>}
        {result && <Alert type={failed ? "error" : done ? "success" : "info"} showIcon
          title={failed ? tr("入库任务失败，请查看任务记录", "Ingestion failed; check the task record")
            : result.existing ? tr("该快照已有入库记录", "This snapshot already has an ingestion record")
              : tr("入库任务已提交", "Ingestion task submitted")}
          description={<Space orientation="vertical">
            <Typography.Text>{tr("任务状态", "Task status")} · {result.status}</Typography.Text>
            <Typography.Text>{tr("请在任务页查看进度。Wiki 草稿需另行审核；提交任务不代表索引或草稿已经完成。", "Check progress on the task page. Wiki drafts require review; submitting a task does not mean indexing or drafting has completed.")}</Typography.Text>
            <Typography.Link href="#/imports">{tr("查看入库任务", "View ingestion tasks")}</Typography.Link>
          </Space>} />}
      </Space>
    </Modal>
  </>;
}
