import { DownOutlined, ReloadOutlined } from "@ant-design/icons";
import { Alert, Button, Card, Empty, List, message, Select, Space, Table, Tag, Typography } from "antd";
import type { ColumnsType } from "antd/es/table";
import { useCallback, useEffect, useState } from "react";

import { api } from "../api";
import type { FailureCategory, SearchAnswerStatus, SearchTestDetail, SearchTestItem, SearchTestRun } from "../types";
import { errorMessage, formatDate } from "../utils";
import { useLanguage } from "../i18n";
import { answerPresentation, evidenceWarnings } from "../answerPresentation";

function VersionResult({ run }: { run: SearchTestRun }) {
  const { tr } = useLanguage();
  const { response } = run;
  const presentation = answerPresentation(response);
  const warnings = evidenceWarnings(response);
  const model = response.generation.model;
  const failureCategory = response.generation.failure_category as FailureCategory | undefined;
  const failureReason = response.generation.failure_reason as string | undefined;
  const failureLabels: Record<FailureCategory, [string, string]> = {
    data_missing: ["数据缺失", "Data missing"],
    retrieval_failed: ["检索失败", "Retrieval failed"],
    evidence_extraction_failed: ["证据抽取失败", "Evidence extraction failed"],
    generation_failed: ["生成失败", "Generation failed"],
  };

  return (
    <Card size="small" className="history-run-card">
      <Space direction="vertical" size={5} style={{ width: "100%" }}>
        <Space wrap className="answer-status">
          <Tag color="blue">{tr(`第 ${run.run_number} 次`, `Run ${run.run_number}`)}</Tag>
          <Tag color={presentation.color}>
            {tr(...presentation.label)}
          </Tag>
          <Typography.Text type="secondary">{formatDate(run.created_at)}</Typography.Text>
          {model ? <Tag>{tr("模型", "Model")} · {String(model)}</Tag> : null}
          <Tag>{tr("证据", "Evidence")} {response.sources.length} {tr("条", "items")}</Tag>
          {failureCategory ? (
            <Tag color="red">
              {tr(failureLabels[failureCategory]?.[0] || failureCategory, failureLabels[failureCategory]?.[1] || failureCategory)}
            </Tag>
          ) : null}
        </Space>
        {warnings.map((warning) => <Alert key={warning[1]} type="warning" showIcon title={tr(...warning)} />)}
        {failureCategory && failureReason ? (
          <Typography.Text type="danger">
            {tr("失败原因：", "Failure reason: ")}{failureReason}
          </Typography.Text>
        ) : null}
        {presentation.answerText ? (
          <Typography.Paragraph className="result-snippet answer-body" copyable={{ text: presentation.answerText }}>
            {presentation.answerText}
          </Typography.Paragraph>
        ) : <Typography.Text type="secondary">{tr("未提供可显示的答案正文。", "No answer body is available.")}</Typography.Text>}
        {presentation.isNative && (
          <details className="answer-trace">
            <summary>{tr("原始模型输出与运行记录", "Raw model output and trace")}</summary>
            <Typography.Paragraph className="result-snippet raw-output" copyable={{ text: presentation.rawAnswer }}>
              {presentation.rawAnswer || tr("无原始输出", "No raw output")}
            </Typography.Paragraph>
            <details>
              <summary>{tr("完整运行记录", "Full trace")}</summary>
              <pre className="trace-json" tabIndex={0} aria-label={tr("完整运行记录", "Full trace")}>
                {JSON.stringify({ retrieval: response.retrieval, generation: response.generation }, null, 2)}
              </pre>
            </details>
          </details>
        )}
        <List
          size="small"
          dataSource={response.sources}
          locale={{ emptyText: tr("没有检索到证据", "No evidence retrieved") }}
          renderItem={(source, index) => (
            <List.Item className="history-evidence-item">
              <Typography.Text strong>[{tr("资料", "Source")} {index + 1}] {source.title || tr("未命名资料", "Untitled source")}</Typography.Text>
              <Typography.Text type="secondary" className="history-evidence-snippet">
                {source.snippet}
              </Typography.Text>
            </List.Item>
          )}
        />
      </Space>
    </Card>
  );
}

export default function SearchHistoryPage() {
  const { tr } = useLanguage();
  const [items, setItems] = useState<SearchTestItem[]>([]);
  const [details, setDetails] = useState<Record<string, SearchTestDetail>>({});
  const [expandedKeys, setExpandedKeys] = useState<readonly React.Key[]>([]);
  const [loading, setLoading] = useState(true);
  const [rerunningId, setRerunningId] = useState<string>();
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(20);
  const [total, setTotal] = useState(0);
  const [answerStatus, setAnswerStatus] = useState<"all" | SearchAnswerStatus>("all");
  const [failureCategory, setFailureCategory] = useState<"all" | FailureCategory>("all");

  const load = useCallback(async (
    nextPage: number,
    nextPageSize: number,
    nextAnswerStatus: "all" | SearchAnswerStatus,
    nextFailureCategory: "all" | FailureCategory,
  ) => {
    setLoading(true);
    try {
      const response = await api.searchHistory(
        nextPage,
        nextPageSize,
        nextAnswerStatus === "all" ? undefined : nextAnswerStatus,
        nextFailureCategory === "all" ? undefined : nextFailureCategory,
      );
      setItems(response.items);
      setTotal(response.total);
      setPage(response.page);
      setPageSize(response.page_size);
    } catch (error) {
      void message.error(errorMessage(error));
    } finally {
      setLoading(false);
    }
  }, []);

  const loadDetail = useCallback(async (id: string) => {
    try {
      const detail = await api.searchHistoryDetail(id);
      setDetails((current) => ({ ...current, [id]: detail }));
    } catch (error) {
      void message.error(errorMessage(error));
    }
  }, []);

  useEffect(() => {
    void load(1, 20, "all", "all");
  }, [load]);

  const rerun = async (item: SearchTestItem) => {
    setRerunningId(item.id);
    try {
      const run = await api.rerunSearchHistory(item.id);
      const detail = await api.searchHistoryDetail(item.id);
      setDetails((current) => ({ ...current, [item.id]: detail }));
      // Update the row in place so rerunning does not move it to the top of
      // the server's updated_at-sorted history list or reset the current page.
      setItems((current) => current.map((currentItem) => currentItem.id === item.id
        ? {
            ...currentItem,
            request: run.request,
            run_count: run.run_number,
            updated_at: run.created_at,
            latest_run_id: run.id,
            latest_run: run,
            latest_answer_status: detail.latest_answer_status,
            latest_failure_category: detail.latest_failure_category,
            latest_failure_reason: detail.latest_failure_reason,
          }
        : currentItem));
      if (answerStatus !== "all" || failureCategory !== "all") {
        await load(page, pageSize, answerStatus, failureCategory);
      }
      void message.success(tr("已追加一次重新检索生成结果", "A new search run has been added"));
    } catch (error) {
      void message.error(errorMessage(error));
    } finally {
      setRerunningId(undefined);
    }
  };

  const columns: ColumnsType<SearchTestItem> = [
    {
      title: tr("问题", "Question"),
      dataIndex: "question",
      width: 240,
      render: (question: string, item) => (
        <Space direction="vertical" size={1}>
          <Typography.Text strong>{question}</Typography.Text>
          <Typography.Text type="secondary">
            {item.knowledge_base_id ? `${tr("知识库：", "Knowledge base: ")}${item.knowledge_base_id}` : tr("全部知识库", "All knowledge bases")}
          </Typography.Text>
        </Space>
      ),
    },
    {
      title: tr("最新答案", "Latest Answer"),
      key: "answer",
      render: (_, item) => {
        const run = item.latest_run;
        if (!run) return "—";
        const presentation = answerPresentation(run.response);
        const preview = presentation.answerText || tr("未提供答案正文", "No answer body");
        return (
          <Typography.Paragraph className="history-answer" ellipsis={{ rows: 2, tooltip: preview }}>
            {preview}
          </Typography.Paragraph>
        );
      },
    },
    {
      title: tr("失败诊断", "Failure Diagnosis"),
      key: "failure",
      width: 180,
      render: (_, item) => {
        if (item.latest_run?.response.generation.pipeline === "rwkv") {
          const presentation = answerPresentation(item.latest_run.response);
          return <Tag color={presentation.color}>{tr(...presentation.label)}</Tag>;
        }
        if (!item.latest_failure_category) return <Tag color="green">{tr("无", "None")}</Tag>;
        const labels: Record<FailureCategory, [string, string]> = {
          data_missing: ["数据缺失", "Data missing"],
          retrieval_failed: ["检索失败", "Retrieval failed"],
          evidence_extraction_failed: ["证据抽取失败", "Evidence extraction failed"],
          generation_failed: ["生成失败", "Generation failed"],
        };
        const label = labels[item.latest_failure_category];
        return (
          <Space direction="vertical" size={0}>
            <Tag color="red">{tr(label[0], label[1])}</Tag>
            {item.latest_failure_reason ? <Typography.Text type="secondary">{item.latest_failure_reason}</Typography.Text> : null}
          </Space>
        );
      },
    },
    {
      title: tr("版本", "Versions"),
      key: "version",
      width: 120,
      render: (_, item) => <Tag color="blue">{item.run_count} {tr("次运行", "runs")}</Tag>,
    },
    {
      title: tr("最新运行", "Latest Run"),
      dataIndex: "updated_at",
      width: 150,
      render: formatDate,
    },
    {
      title: tr("操作", "Actions"),
      key: "actions",
      width: 145,
      render: (_, item) => (
        <Button
          type="primary"
          icon={<ReloadOutlined />}
          loading={rerunningId === item.id}
          onClick={() => void rerun(item)}
        >
          {tr("重新检索生成", "Run again")}
        </Button>
      ),
    },
  ];

  return (
    <div className="page-stack">
      <div className="page-heading">
        <div>
          <Typography.Text className="eyebrow">REGRESSION LAB</Typography.Text>
          <Typography.Title level={2}>{tr("历史检索测试", "Search History")}</Typography.Title>
          <Typography.Paragraph type="secondary">
            {tr("自动保存 `/v1/ask` 的问题、答案、证据和模型信息；重新检索会新增版本，方便比较改动前后的效果。", "Automatically save `/v1/ask` questions, answers, evidence, and model details. Reruns create versions for before-and-after comparison.")}
          </Typography.Paragraph>
        </div>
        <Space>
          <Select
            value={answerStatus}
            style={{ width: 150 }}
            options={[
              { value: "all", label: tr("全部状态", "All statuses") },
              { value: "completed", label: tr("生成完成（未核验语义）", "Completed (semantics unverified)") },
              { value: "partial", label: tr("部分步骤失败", "Some stages failed") },
              { value: "failed", label: tr("流程未完成", "Execution incomplete") },
              { value: "answered", label: tr("旧链路：标记已回答", "Legacy: marked answered") },
              { value: "refused", label: tr("旧链路：无法确定", "Legacy: unable to determine") },
            ]}
            onChange={(value) => {
              setAnswerStatus(value);
              setExpandedKeys([]);
            void load(1, pageSize, value, failureCategory);
          }}
        />
        <Select
          value={failureCategory}
          style={{ width: 180 }}
          options={[
            { value: "all", label: tr("全部失败类型", "All failure types") },
            { value: "data_missing", label: tr("数据缺失", "Data missing") },
            { value: "retrieval_failed", label: tr("检索失败", "Retrieval failed") },
            { value: "evidence_extraction_failed", label: tr("证据抽取失败", "Evidence extraction failed") },
            { value: "generation_failed", label: tr("生成失败", "Generation failed") },
          ]}
          onChange={(value) => {
            setFailureCategory(value);
            setExpandedKeys([]);
            void load(1, pageSize, answerStatus, value);
          }}
        />
          <Button icon={<ReloadOutlined />} loading={loading} onClick={() => void load(page, pageSize, answerStatus, failureCategory)}>{tr("刷新", "Refresh")}</Button>
        </Space>
      </div>
      <Alert
        type="info"
        showIcon
        message={tr("同一问题和知识库会归为同一测试项；展开一行即可按时间查看全部历史版本。", "The same question and knowledge base are grouped together. Expand a row to view all versions chronologically.")}
      />
      <Card title={tr("已记录问题", "Recorded Questions")}>
        <Table
          rowKey="id"
          size="small"
          loading={loading}
          columns={columns}
          dataSource={items}
          locale={{ emptyText: <Empty description={tr("还没有检索记录。请先在“检索测试”页面提交问题。", "No search history yet. Submit a question in Search Lab first.")} /> }}
          pagination={{
            current: page,
            pageSize,
            total,
            showSizeChanger: true,
            pageSizeOptions: [10, 20, 50, 100],
            showTotal: (count) => tr(`共 ${count} 条`, `${count} items`),
            onChange: (nextPage, nextPageSize) => {
              setExpandedKeys([]);
              void load(nextPageSize === pageSize ? nextPage : 1, nextPageSize, answerStatus, failureCategory);
            },
          }}
          expandable={{
            expandedRowKeys: expandedKeys,
            expandIcon: ({ expanded, onExpand, record }) => (
              <Button
                type="text"
                size="small"
                icon={<DownOutlined rotate={expanded ? 180 : 0} />}
                onClick={(event) => onExpand(record, event)}
              />
            ),
            onExpand: (expanded, record) => {
              setExpandedKeys((current) => expanded
                ? [...current, record.id]
                : current.filter((key) => key !== record.id));
              if (expanded) void loadDetail(record.id);
            },
            expandedRowRender: (item) => {
              const detail = details[item.id];
              if (!detail) return <Typography.Text type="secondary">{tr("正在加载历史版本…", "Loading versions…")}</Typography.Text>;
              return (
                <List
                  dataSource={[...detail.runs].reverse()}
                  renderItem={(run) => <List.Item><VersionResult run={run} /></List.Item>}
                />
              );
            },
          }}
        />
      </Card>
    </div>
  );
}
