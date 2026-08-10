import { DownOutlined, ReloadOutlined } from "@ant-design/icons";
import { Alert, Button, Card, Empty, List, message, Space, Table, Tag, Typography } from "antd";
import type { ColumnsType } from "antd/es/table";
import { useCallback, useEffect, useState } from "react";

import { api } from "../api";
import type { SearchTestDetail, SearchTestItem, SearchTestRun } from "../types";
import { errorMessage, formatDate } from "../utils";

function VersionResult({ run }: { run: SearchTestRun }) {
  const { response } = run;
  const grounded = response.generation.evidence_grounded === true;
  const model = response.generation.model;

  return (
    <Card size="small" className="history-run-card">
      <Space direction="vertical" size={5} style={{ width: "100%" }}>
        <Space wrap>
          <Tag color="blue">第 {run.run_number} 次</Tag>
          <Tag color={grounded ? "green" : "orange"}>
            {grounded ? "证据校验通过" : "证据不足"}
          </Tag>
          <Typography.Text type="secondary">{formatDate(run.created_at)}</Typography.Text>
          {model ? <Tag>模型 · {String(model)}</Tag> : null}
          <Tag>证据 {response.sources.length} 条</Tag>
        </Space>
        <Typography.Paragraph className="result-snippet" copyable={{ text: response.answer }}>
          {response.answer}
        </Typography.Paragraph>
        <List
          size="small"
          dataSource={response.sources}
          locale={{ emptyText: "没有检索到证据" }}
          renderItem={(source, index) => (
            <List.Item className="history-evidence-item">
              <Typography.Text strong>[资料 {index + 1}] {source.title || "未命名资料"}</Typography.Text>
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
      void message.success("已追加一次重新检索生成结果");
    } catch (error) {
      void message.error(errorMessage(error));
    } finally {
      setRerunningId(undefined);
    }
  };

  const columns: ColumnsType<SearchTestItem> = [
    {
      title: "问题",
      dataIndex: "question",
      width: 240,
      render: (question: string, item) => (
        <Space direction="vertical" size={1}>
          <Typography.Text strong>{question}</Typography.Text>
          <Typography.Text type="secondary">
            {item.knowledge_base_id ? `知识库：${item.knowledge_base_id}` : "全部知识库"}
          </Typography.Text>
        </Space>
      ),
    },
    {
      title: "最新答案",
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
      title: "版本",
      key: "version",
      width: 120,
      render: (_, item) => <Tag color="blue">{item.run_count} 次运行</Tag>,
    },
    {
      title: "最新运行",
      dataIndex: "updated_at",
      width: 150,
      render: formatDate,
    },
    {
      title: "操作",
      key: "actions",
      width: 145,
      render: (_, item) => (
        <Button
          type="primary"
          icon={<ReloadOutlined />}
          loading={rerunningId === item.id}
          onClick={() => void rerun(item)}
        >
          重新检索生成
        </Button>
      ),
    },
  ];

  return (
    <div className="page-stack">
      <div className="page-heading">
        <div>
          <Typography.Text className="eyebrow">REGRESSION LAB</Typography.Text>
          <Typography.Title level={2}>历史检索测试</Typography.Title>
          <Typography.Paragraph type="secondary">
            自动保存 `/v1/ask` 的问题、答案、证据和模型信息；重新检索会新增版本，方便比较改动前后的效果。
          </Typography.Paragraph>
        </div>
        <Button icon={<ReloadOutlined />} loading={loading} onClick={() => void load()}>刷新</Button>
      </div>
      <Alert
        type="info"
        showIcon
        message="同一问题和知识库会归为同一测试项；展开一行即可按时间查看全部历史版本。"
      />
      <Card title="已记录问题">
        <Table
          rowKey="id"
          size="small"
          loading={loading}
          columns={columns}
          dataSource={items}
          locale={{ emptyText: <Empty description="还没有检索记录。请先在“检索测试”页面提交问题。" /> }}
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
              if (!detail) return <Typography.Text type="secondary">正在加载历史版本…</Typography.Text>;
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
