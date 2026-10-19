import { Alert, Button, Card, Drawer, Empty, Space, Tag, Typography } from "antd";
import { useState } from "react";
import { useLanguage } from "../i18n";
import type { AskResponse, SearchResult } from "../types";
import { citationParts, citedSource, externalSourceUrl, savedContext, sourceLabels, originalSourceUrl } from "../citations";
import { formatAnswer } from "../answerFormat";
import TaskMatrix from "./TaskMatrix";
import SaveWebSource from "./SaveWebSource";
import { unifiedAnswer } from "../unifiedAnswer";

/** Keep the raw answer in the response; block unsafe display when format checks fail. */
export default function CitedAnswer({ text, response, compact = false }: { text: string; response: AskResponse; compact?: boolean }) {
  const { tr } = useLanguage();
  const formatted = formatAnswer(text, response);
  const unified = unifiedAnswer(response);
  const spanBindings = response.generation.span_bindings as Record<string, {
    source_id?: string; selected_original_text?: string; model_generated_quote?: boolean;
    span?: unknown; snapshot_id?: string;
  }> | undefined;
  const selectedSpans = (unified?.referenceIds ?? []).map(id => ({ id, binding: spanBindings?.[id] }));
  const [selected, setSelected] = useState<{ id?: string; label?: number }>();
  const [drawerOpen, setDrawerOpen] = useState(false);
  const openSource = (selection: NonNullable<typeof selected>) => {
    setSelected(selection);
    setDrawerOpen(true);
  };
  const source = selected?.id ? response.sources.find(s => s.id === selected.id)
    : selected?.label !== undefined ? citedSource(response, selected.label) : undefined;
  const context = source ? savedContext(response, source) : undefined;
  const link = externalSourceUrl(source?.uri);
  const original = source ? originalSourceUrl(source) : undefined;
  const origin = (s: SearchResult) => s.metadata?.retrieval_origin === "web" || s.source === "web"
    ? tr("网络来源", "Web source")
    : s.source === "provided_material" || s.source === "user_material"
      ? tr("提供的材料", "Provided material") : tr("资料来源", "Source material");
  return <section className="cited-answer" aria-label={tr("答案与引用", "Answer and citations")}>
    <TaskMatrix response={response} onSource={id => openSource({ id })} />
    {formatted.blocked && <Alert type="warning" showIcon
      title={tr("回答因引用或复读异常已从展示中隐藏", "Answer hidden from display due to citation or repetition errors")}
      description={tr("模型原文仍可在下方展开核对；这里没有自动补引用或修改事实。", "Open the raw model answer below to inspect it. No citation or fact was repaired.")} />}
    {!formatted.blocked && formatted.repeated && <Alert type="warning" showIcon
      title={tr("正文含重复片段，原文保持显示", "Repeated passages detected; text remains visible")}
      description={tr("这是形式提示，不代表内容错误；请按当前请求核对。", "This is a pattern notice, not a factual verdict. Check the current request.")} />}
    {unified?.valid && <>
      {!compact && unified.root?.kind === "final" && <Typography.Text type="secondary">{tr("正文与缺口是根JSON中模型字段的展示投影；完整原始输出保留在诊断记录中。", "Answer and gaps are display projections of model fields in the root JSON; the full raw output remains in the trace.")}</Typography.Text>}
      {selectedSpans.length > 0 && <Space wrap>
        <Typography.Text type="secondary">{tr("模型选择的来源片段（不代表逐句核实）：", "Model-selected source spans (not claim verification):")}</Typography.Text>
        {selectedSpans.map(({ id, binding }) => <Button key={id} size="small"
          disabled={!binding?.source_id} onClick={() => openSource({ id: binding?.source_id })}>{id}</Button>)}
      </Space>}
      {unified.missing.length > 0 && <Alert type="warning" showIcon
        title={tr("模型报告的缺口（未经独立核实）", "Model-reported gaps (not independently verified)")}
        description={<ul>{unified.missing.map((gap, i) => <li key={i}>{gap}</li>)}</ul>} />}
    </>}
    {!formatted.blocked && formatted.missingCitations && !selectedSpans.length && <Alert type="warning" showIcon
      title={tr("回答没有引用任何已保存的资料", "The answer cites none of its saved sources")}
      description={tr("下方资料只能说明本次提供了哪些材料，不能证明回答中的结论；请逐项核对原文。", "The source list shows what was provided, not which claims it supports. Check each claim against the original text.")} />}
    {formatted.text ? <Typography.Paragraph className="result-snippet answer-body" copyable={{ text: formatted.text }}>
      {citationParts(formatted.text, response.generation.protocol).map((part, i) => part.label === undefined ? part.text : <button
        type="button" key={i} className={`citation-link${citedSource(response, part.label) ? "" : " citation-missing"}`}
        aria-label={tr(`查看引用 ${part.label}`, `View citation ${part.label}`)}
        onClick={() => openSource({ label: part.label })}>{part.text}</button>)}
    </Typography.Paragraph> : !formatted.blocked && <Typography.Text type="secondary">{tr("未提供可显示的答案正文。", "No answer body is available.")}</Typography.Text>}
    {formatted.blocked && text && <details><summary>{tr("查看未经转换的答案", "View unchanged answer")}</summary>
      <Typography.Paragraph className="result-snippet raw-output" copyable={{ text }}>{text}</Typography.Paragraph>
    </details>}
    <details className={`citation-sources${compact ? " compact" : ""}`} open={!compact}>
    <summary style={compact ? undefined : { display: "none" }}>{tr("查看本次来源", "View sources")} ({response.sources.length})</summary>
    <div className="citation-heading"><Typography.Title level={5}>{tr("引用原文", "Source evidence")} ({response.sources.length})</Typography.Title>
      <Typography.Text type="secondary">{tr("点击答案中的编号或下方来源，查看原文。编号对应关系不代表事实已核验。", "Click a citation or source to read its evidence. Citation mapping does not verify factual support.")}</Typography.Text></div>
    {!response.sources.length && <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={tr("本次没有返回来源，无法展示引用原文。", "No sources were returned for this answer.")} />}
    <div className="citation-source-list">{response.sources.map((item, index) => {
      const labels = sourceLabels(response, item, index);
      return <Card key={`${item.id}-${index}`} size="small" className="citation-source-card">
        <Button type="link" className="citation-source-title" onClick={() => openSource({ id: item.id, label: labels[0] })}>
          {labels.length ? labels.map(n => `[${tr("资料", "Source")} ${n}]`).join(" ") : tr("未分配引用编号", "No citation label")} {item.title || tr("未命名资料", "Untitled source")}
        </Button>
        <Space wrap><Tag>{origin(item)}</Tag>{item.metadata?.content_status === "snippet_only" && <Tag color="orange">{tr("仅搜索摘要", "Search snippet only")}</Tag>}</Space>
        <Typography.Paragraph className="citation-preview" ellipsis={{ rows: 3, expandable: "collapsible", symbol: expanded => expanded ? tr("收起", "Less") : tr("展开原文", "Expand evidence") }}>{item.snippet}</Typography.Paragraph>
        <Space wrap>
          <Button size="small" onClick={() => openSource({ id: item.id, label: labels[0] })}>{tr("查看完整引用", "View full evidence")}</Button>
          <SaveWebSource source={item} />
        </Space>
      </Card>;
    })}</div>
    </details>
    <Drawer open={drawerOpen} onClose={() => setDrawerOpen(false)} width="min(760px, 100vw)"
      title={selected?.label !== undefined ? tr(`引用原文 · 资料 ${selected.label}`, `Source evidence · Source ${selected.label}`) : tr("引用原文", "Source evidence")}>
      {!source ? <Alert type="error" showIcon title={tr("引用没有对应来源", "Citation has no matching source")}
        description={tr("这条回答未保存该编号对应的原文，不能将它作为可核对的依据。", "No source was saved for this citation. It cannot be checked against evidence.")} /> : <Space direction="vertical" size="middle" style={{ width: "100%" }}>
        <Typography.Title level={4}>{source.title || tr("未命名资料", "Untitled source")}</Typography.Title>
        <Space wrap><Tag>{origin(source)}</Tag>{typeof source.metadata?.retrieved_at === "string" && <Tag>{tr("抓取时间", "Retrieved")} · {source.metadata.retrieved_at}</Tag>}
          {typeof source.metadata?.published_date === "string" && source.metadata.published_date && <Tag>{tr("网页标注日期", "Page date")} · {source.metadata.published_date}</Tag>}</Space>
        {link ? <Typography.Link href={link} target="_blank" rel="noopener noreferrer">{tr("打开来源网页", "Open source page")} · {link}</Typography.Link>
          : <Typography.Text type="secondary">{tr("本地文档：以下显示本次回答保存的原文，不依赖当前文件是否已更新。", "Local document: the saved evidence below belongs to this answer, even if the current file has changed.")}</Typography.Text>}
        <SaveWebSource key={source.id} source={source} />
        {original && <Typography.Link href={original} target="_blank" rel="noopener noreferrer">{tr("查看本次引用的原始文件", "Open this citation's original file")}{typeof source.metadata?.page === "number" ? ` · ${tr("第", "Page")} ${source.metadata.page} ${tr("页", "")}` : ""}</Typography.Link>}
        {typeof source.metadata?.extraction_method === "string" && source.metadata.extraction_method.includes("ocr") && <Alert type="info" title={tr("此页包含 OCR 识别文字，可打开原件核对。", "This page includes OCR text. Open the original to verify it.")} />}

        {source.metadata?.content_status === "snippet_only" && <Alert type="info" title={tr("此次只获取了搜索摘要，未获取网页全文。", "Only the search snippet was retrieved, not the full page.")} />}
        {source.metadata?.material_limited === true && <Alert type="info" title={tr("用于回答的材料受长度预算限制。", "Material used for the answer was limited by its length budget.")} />}
        {selectedSpans.filter(({ binding }) => binding?.source_id === source.id).map(({ id, binding }) =>
          binding?.model_generated_quote === false && typeof binding.selected_original_text === "string"
            ? <section key={id}>
              <Typography.Title level={5}>{id}</Typography.Title>
              <Typography.Text type="secondary">{tr("以下为按模型所选ID查找的原文，不是模型生成的quote；来源绑定不证明陈述正确。", "Original text looked up by the model-selected ID, not a model-generated quote; binding does not prove the statement.")}</Typography.Text>
              <Typography.Paragraph className="citation-full-text" copyable={{ text: binding.selected_original_text }}>{binding.selected_original_text}</Typography.Paragraph>
              <pre className="trace-json">{JSON.stringify({ snapshot_id: binding.snapshot_id, span: binding.span }, null, 2)}</pre>
            </section> : null)}
        <Typography.Title level={5}>{unified
          ? tr("该来源单元的完整原文（不代表全部被引用）", "Full source unit (not all of it is necessarily cited)")
          : tr("回答使用的逐字证据", "Verbatim evidence used for this answer")}</Typography.Title>
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
