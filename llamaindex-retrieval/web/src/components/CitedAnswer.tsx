import { Alert, Button, Card, Drawer, Empty, Space, Tag, Typography } from "antd";
import { useState } from "react";
import { useLanguage } from "../i18n";
import type { AskResponse, SearchResult } from "../types";
import { citationParts, citedSource, externalSourceUrl, savedContext, sourceLabels } from "../citations";
import TaskMatrix from "./TaskMatrix";

/** Presentation only: no answer changes, source renumbering or citation repair. */
export default function CitedAnswer({ text, response }: { text: string; response: AskResponse }) {
  const { tr } = useLanguage();
  const [selected, setSelected] = useState<{ id?: string; label?: number }>();
  const source = selected?.id ? response.sources.find(s => s.id === selected.id)
    : selected?.label !== undefined ? citedSource(response, selected.label) : undefined;
  const context = source ? savedContext(response, source) : undefined;
  const link = externalSourceUrl(source?.uri);
  const origin = (s: SearchResult) => s.metadata?.retrieval_origin === "web" || s.source === "web"
    ? tr("网络来源", "Web source") : tr("知识库来源", "Knowledge-base source");
  return <section className="cited-answer" aria-label={tr("答案与引用", "Answer and citations")}>
    <TaskMatrix response={response} onSource={id => setSelected({ id })} />
    {text ? <Typography.Paragraph className="result-snippet answer-body" copyable={{ text }}>
      {citationParts(text).map((part, i) => part.label === undefined ? part.text : <button
        type="button" key={i} className={`citation-link${citedSource(response, part.label) ? "" : " citation-missing"}`}
        aria-label={tr(`查看引用 ${part.label}`, `View citation ${part.label}`)}
        onClick={() => setSelected({ label: part.label })}>{part.text}</button>)}
    </Typography.Paragraph> : <Typography.Text type="secondary">{tr("未提供可显示的答案正文。", "No answer body is available.")}</Typography.Text>}
    <div className="citation-heading"><Typography.Title level={5}>{tr("引用原文", "Source evidence")} ({response.sources.length})</Typography.Title>
      <Typography.Text type="secondary">{tr("点击答案中的编号或下方来源，查看原文。编号对应关系不代表事实已核验。", "Click a citation or source to read its evidence. Citation mapping does not verify factual support.")}</Typography.Text></div>
    {!response.sources.length && <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={tr("本次没有返回来源，无法展示引用原文。", "No sources were returned for this answer.")} />}
    <div className="citation-source-list">{response.sources.map((item, index) => {
      const labels = sourceLabels(response, item, index);
      return <Card key={`${item.id}-${index}`} size="small" className="citation-source-card">
        <Button type="link" className="citation-source-title" onClick={() => setSelected({ id: item.id, label: labels[0] })}>
          {labels.length ? labels.map(n => `[${tr("资料", "Source")} ${n}]`).join(" ") : tr("未分配引用编号", "No citation label")} {item.title || tr("未命名资料", "Untitled source")}
        </Button>
        <Space wrap><Tag>{origin(item)}</Tag>{item.metadata?.content_status === "snippet_only" && <Tag color="orange">{tr("仅搜索摘要", "Search snippet only")}</Tag>}</Space>
        <Typography.Paragraph className="citation-preview" ellipsis={{ rows: 3, expandable: "collapsible", symbol: expanded => expanded ? tr("收起", "Less") : tr("展开原文", "Expand evidence") }}>{item.snippet}</Typography.Paragraph>
        <Button size="small" onClick={() => setSelected({ id: item.id, label: labels[0] })}>{tr("查看完整引用", "View full evidence")}</Button>
      </Card>;
    })}</div>
    <Drawer open={selected !== undefined} onClose={() => setSelected(undefined)} width="min(760px, 100vw)"
      title={selected?.label !== undefined ? tr(`引用原文 · 资料 ${selected.label}`, `Source evidence · Source ${selected.label}`) : tr("引用原文", "Source evidence")}>
      {!source ? <Alert type="error" showIcon title={tr("引用没有对应来源", "Citation has no matching source")}
        description={tr("这条回答未保存该编号对应的原文，不能将它作为可核对的依据。", "No source was saved for this citation. It cannot be checked against evidence.")} /> : <Space direction="vertical" size="middle" style={{ width: "100%" }}>
        <Typography.Title level={4}>{source.title || tr("未命名资料", "Untitled source")}</Typography.Title>
        <Space wrap><Tag>{origin(source)}</Tag>{typeof source.metadata?.retrieved_at === "string" && <Tag>{tr("抓取时间", "Retrieved")} · {source.metadata.retrieved_at}</Tag>}
          {typeof source.metadata?.published_date === "string" && source.metadata.published_date && <Tag>{tr("网页标注日期", "Page date")} · {source.metadata.published_date}</Tag>}</Space>
        {link ? <Typography.Link href={link} target="_blank" rel="noopener noreferrer">{tr("打开来源网页", "Open source page")} · {link}</Typography.Link>
          : <Typography.Text type="secondary">{tr("本地文档：以下显示本次回答保存的原文，不依赖当前文件是否已更新。", "Local document: the saved evidence below belongs to this answer, even if the current file has changed.")}</Typography.Text>}
        {source.metadata?.content_status === "snippet_only" && <Alert type="info" title={tr("此次只获取了搜索摘要，未获取网页全文。", "Only the search snippet was retrieved, not the full page.")} />}
        {source.metadata?.material_limited === true && <Alert type="info" title={tr("用于回答的材料受长度预算限制。", "Material used for the answer was limited by its length budget.")} />}
        <Typography.Title level={5}>{tr("回答使用的逐字证据", "Verbatim evidence used for this answer")}</Typography.Title>
        <Typography.Paragraph className="citation-full-text" copyable={{ text: source.snippet }}>{source.snippet || tr("来源记录中没有原文内容。", "No evidence text was saved.")}</Typography.Paragraph>
        {Array.isArray(source.metadata?.context_spans) && source.metadata.context_spans.length > 0 && <details><summary>{tr("文档上下文", "Document context")}</summary>
          {source.metadata.context_spans.map((span, i) => <p className="citation-full-text" key={i}>{typeof span?.text === "string" ? span.text : ""}</p>)}</details>}
        {context && context !== source.snippet && <details><summary>{tr("查看当时保存的来源材料", "View the saved source material")}</summary>
          <Typography.Paragraph className="citation-full-text" copyable={{ text: context }}>{context}</Typography.Paragraph></details>}
        <details><summary>{tr("来源标识与版本", "Source identity and version")}</summary><pre className="trace-json">{JSON.stringify({ id: source.id, uri: source.uri, metadata: source.metadata }, null, 2)}</pre></details>
      </Space>}
    </Drawer>
  </section>;
}
