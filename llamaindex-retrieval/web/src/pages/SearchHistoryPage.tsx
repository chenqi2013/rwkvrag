import { DownOutlined, ReloadOutlined } from "@ant-design/icons";
import { Alert, Button, Card, Empty, List, message, Space, Table, Tag, Typography } from "antd";
import type { ColumnsType } from "antd/es/table";
import { useCallback, useEffect, useState } from "react";

import { api } from "../api";
import type { SearchTestDetail, SearchTestItem, SearchTestRun } from "../types";
import { errorMessage, formatDate } from "../utils";
import { useLanguage } from "../i18n";

function VersionResult({ run }: { run: SearchTestRun }) {
  const { tr } = useLanguage();
  const { response } = run;
  const grounded = response.generation.evidence_grounded === true;
  const model = response.generation.model;

  return (
    <Card size="small" className="history-run-card">
      <Space direction="vertical" size={5} style={{ width: "100%" }}>
        <Space wrap>
          <Tag color="blue">{tr(`第 ${run.run_number} 次`, `Run ${run.run_number}`)}</Tag>
          <Tag color={grounded ? "green" : "orange"}>
            {grounded ? tr("证据校验通过", "Evidence verified") : tr("证据不足", "Insufficient evidence")}
          </Tag>
          <Typography.Text type="secondary">{formatDate(run.created_at)}</Typography.Text>
          {model ? <Tag>{tr("模型", "Model")} · {String(model)}</Tag> : null}
          <Tag>{tr("证据", "Evidence")} {response.sources.length} {tr("条", "items")}</Tag>
        </Space>
        <Typography.Paragraph className="result-snippet" copyable={{ text: response.answer }}>
          {response.answer}
        </Typography.Paragraph>
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

  const load = useCallback(async () => {
    setLoading(true);
    try {
      setItems(await api.searchHistory());
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
    void load();
  }, [load]);

  const rerun = async (item: SearchTestItem) => {
    setRerunningId(item.id);
    try {
      await api.rerunSearchHistory(item.id);
      await Promise.all([load(), loadDetail(item.id)]);
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
        return (
          <Typography.Paragraph className="history-answer" ellipsis={{ rows: 2, tooltip: run.response.answer }}>
            {run.response.answer}
          </Typography.Paragraph>
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
        <Button icon={<ReloadOutlined />} loading={loading} onClick={() => void load()}>{tr("刷新", "Refresh")}</Button>
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
          pagination={{ pageSize: 20, showSizeChanger: false }}
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
