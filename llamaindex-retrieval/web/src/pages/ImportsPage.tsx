import { ImportOutlined, ReloadOutlined } from "@ant-design/icons";
import {
  Alert,
  Button,
  Card,
  Checkbox,
  Col,
  Form,
  Input,
  InputNumber,
  message,
  Progress,
  Row,
  Select,
  Space,
  Table,
  Tag,
  Typography,
} from "antd";
import type { ColumnsType } from "antd/es/table";
import { useCallback, useEffect, useState } from "react";

import { api } from "../api";
import type { JobItem, KnowledgeBase } from "../types";
import { errorMessage, formatDate, formatNumber } from "../utils";

interface ImportForm {
  path: string;
  knowledge_base_id: string;
  source: string;
  titles: string;
  limit: number;
  batch_size: number;
  recreate: boolean;
}

const statusColors: Record<JobItem["status"], string> = {
  pending: "default",
  running: "processing",
  completed: "success",
  failed: "error",
};

export default function ImportsPage() {
  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBase[]>([]);
  const [jobs, setJobs] = useState<JobItem[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [form] = Form.useForm<ImportForm>();

  const load = useCallback(async () => {
    try {
      const [knowledgeBaseData, jobData] = await Promise.all([api.knowledgeBases(), api.jobs()]);
      setKnowledgeBases(knowledgeBaseData);
      setJobs(jobData);
    } catch (error) {
      void message.error(errorMessage(error));
    }
  }, []);

  useEffect(() => {
    void load();
    const timer = window.setInterval(() => void load(), 2500);
    return () => window.clearInterval(timer);
  }, [load]);

  const submit = async () => {
    const values = await form.validateFields();
    setSubmitting(true);
    try {
      const titles = values.titles
        ? values.titles.split("\n").map((item) => item.trim()).filter(Boolean)
        : [];
      await api.importFineWiki({ ...values, titles });
      void message.success("FineWiki 导入任务已提交");
      await load();
    } catch (error) {
      void message.error(errorMessage(error));
    } finally {
      setSubmitting(false);
    }
  };

  const columns: ColumnsType<JobItem> = [
    {
      title: "任务",
      key: "job",
      render: (_, item) => (
        <Space direction="vertical" size={0}>
          <Typography.Text strong>
            {item.kind === "finewiki_import" ? "FineWiki 导入" : item.kind === "file_reindex" ? "文档重建" : "文档入库"}
          </Typography.Text>
          <Typography.Text type="secondary" copyable={{ text: item.id }}>
            {item.id.slice(0, 12)}
          </Typography.Text>
        </Space>
      ),
    },
    {
      title: "状态",
      dataIndex: "status",
      width: 130,
      render: (status: JobItem["status"]) => <Tag color={statusColors[status]}>{status}</Tag>,
    },
    {
      title: "进度",
      key: "progress",
      width: 260,
      render: (_, item) => (
        <Space direction="vertical" size={3} style={{ width: "100%" }}>
          <Progress percent={item.progress} status={item.status === "failed" ? "exception" : undefined} />
          <Typography.Text type="secondary">{item.message}</Typography.Text>
        </Space>
      ),
    },
    {
      title: "处理量",
      key: "processed",
      width: 180,
      render: (_, item) => (
        <Typography.Text>
          {formatNumber(item.documents_processed)} 文档 / {formatNumber(item.nodes_processed)} 切片
        </Typography.Text>
      ),
    },
    {
      title: "时间",
      dataIndex: "created_at",
      width: 180,
      render: formatDate,
    },
    {
      title: "错误",
      dataIndex: "error",
      render: (value?: string) => value ? <Typography.Text type="danger">{value}</Typography.Text> : "—",
    },
  ];

  return (
    <div className="page-stack">
      <div className="page-heading">
        <div>
          <Typography.Text className="eyebrow">BULK INGESTION</Typography.Text>
          <Typography.Title level={2}>导入任务</Typography.Title>
          <Typography.Paragraph type="secondary">
            创建 FineWiki 异步任务并持续查看文档、切片数量和失败原因。
          </Typography.Paragraph>
        </div>
        <Button icon={<ReloadOutlined />} onClick={() => void load()}>刷新</Button>
      </div>
      <Row gutter={[8, 8]}>
        <Col xs={24} xl={9}>
          <Card title="新建 FineWiki 导入">
            <Form
              form={form}
              layout="vertical"
              initialValues={{
                path: "/Volumes/mark/rwkvrag/data/deploy-demo/finewiki-sample/train-00000-of-00026.parquet",
                knowledge_base_id: "default",
                source: "finewiki-zh",
                titles: "",
                limit: 0,
                batch_size: 8,
                recreate: false,
              }}
            >
              <Form.Item label="Parquet 文件或目录" name="path" rules={[{ required: true }]}>
                <Input />
              </Form.Item>
              <Form.Item label="知识库" name="knowledge_base_id" rules={[{ required: true }]}>
                <Select options={knowledgeBases.map((item) => ({ value: item.id, label: item.name }))} />
              </Form.Item>
              <Form.Item label="数据来源" name="source" rules={[{ required: true }]}>
                <Input />
              </Form.Item>
              <Form.Item label="指定标题（每行一个，可选）" name="titles">
                <Input.TextArea rows={4} placeholder={"首都\n秦始皇"} />
              </Form.Item>
              <Row gutter={8}>
                <Col span={12}>
                  <Form.Item label="限制文档数，0 为全部" name="limit">
                    <InputNumber min={0} style={{ width: "100%" }} />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item label="文档批次" name="batch_size">
                    <InputNumber min={1} max={128} style={{ width: "100%" }} />
                  </Form.Item>
                </Col>
              </Row>
              <Form.Item name="recreate" valuePropName="checked">
                <Checkbox>导入前重建当前 collection</Checkbox>
              </Form.Item>
              <Alert
                type="warning"
                showIcon
                message="重建 collection 会删除当前所有 Wiki 和上传文档向量，请谨慎使用。"
                className="form-alert"
              />
              <Button type="primary" icon={<ImportOutlined />} block loading={submitting} onClick={() => void submit()}>
                提交导入任务
              </Button>
            </Form>
          </Card>
        </Col>
        <Col xs={24} xl={15}>
          <Card title="任务记录">
            <Table rowKey="id" dataSource={jobs} columns={columns} pagination={{ pageSize: 10 }} scroll={{ x: 980 }} />
          </Card>
        </Col>
      </Row>
    </div>
  );
}
