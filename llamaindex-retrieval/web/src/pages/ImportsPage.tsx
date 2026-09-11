import {
  ArrowUpOutlined,
  FileOutlined,
  FolderOpenOutlined,
  FolderOutlined,
  ImportOutlined,
  ReloadOutlined,
} from "@ant-design/icons";
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
  Modal,
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
import type { FineWikiPathEntry, FineWikiPathPage, JobItem, KnowledgeBase } from "../types";
import { errorMessage, formatBytes, formatDate, formatNumber } from "../utils";
import { useLanguage } from "../i18n";

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
  const { tr } = useLanguage();
  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBase[]>([]);
  const [jobs, setJobs] = useState<JobItem[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [pathBrowserOpen, setPathBrowserOpen] = useState(false);
  const [pathBrowserLoading, setPathBrowserLoading] = useState(false);
  const [pathPage, setPathPage] = useState<FineWikiPathPage>();
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
      void message.success(tr("FineWiki 导入任务已提交", "FineWiki import job submitted"));
      await load();
    } catch (error) {
      void message.error(errorMessage(error));
    } finally {
      setSubmitting(false);
    }
  };

  const browsePath = async (path?: string) => {
    setPathBrowserLoading(true);
    try {
      setPathPage(await api.fineWikiPaths(path));
    } catch (error) {
      void message.error(errorMessage(error));
    } finally {
      setPathBrowserLoading(false);
    }
  };

  const openPathBrowser = async () => {
    setPathBrowserOpen(true);
    const currentPath = form.getFieldValue("path");
    try {
      setPathBrowserLoading(true);
      setPathPage(await api.fineWikiPaths(currentPath));
    } catch {
      await browsePath();
    } finally {
      setPathBrowserLoading(false);
    }
  };

  const selectPath = (path: string) => {
    form.setFieldValue("path", path);
    setPathBrowserOpen(false);
  };

  const columns: ColumnsType<JobItem> = [
    {
      title: tr("任务", "Job"),
      key: "job",
      render: (_, item) => (
        <Space direction="vertical" size={0}>
          <Typography.Text strong>
            {item.kind === "finewiki_import" ? tr("FineWiki 导入", "FineWiki import") : item.kind === "file_reindex" ? tr("文档重建", "Document reindex") : tr("文档入库", "Document ingestion")}
          </Typography.Text>
          <Typography.Text type="secondary" copyable={{ text: item.id }}>
            {item.id.slice(0, 12)}
          </Typography.Text>
        </Space>
      ),
    },
    {
      title: tr("状态", "Status"),
      dataIndex: "status",
      width: 130,
      render: (status: JobItem["status"]) => <Tag color={statusColors[status]}>{status}</Tag>,
    },
    {
      title: tr("进度", "Progress"),
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
      title: tr("处理量", "Processed"),
      key: "processed",
      width: 180,
      render: (_, item) => (
        <Typography.Text>
          {formatNumber(item.documents_processed)} {tr("文档", "documents")} / {formatNumber(item.nodes_processed)} {tr("切片", "chunks")}
        </Typography.Text>
      ),
    },
    {
      title: tr("时间", "Time"),
      dataIndex: "created_at",
      width: 180,
      render: formatDate,
    },
    {
      title: tr("错误", "Error"),
      dataIndex: "error",
      render: (value?: string) => value ? <Typography.Text type="danger">{value}</Typography.Text> : "—",
    },
  ];

  const pathColumns: ColumnsType<FineWikiPathEntry> = [
    {
      title: tr("名称", "Name"),
      dataIndex: "name",
      render: (name: string, item) => (
        <Button
          type="link"
          icon={item.type === "directory" ? <FolderOutlined /> : <FileOutlined />}
          onClick={() => item.type === "directory" ? void browsePath(item.path) : selectPath(item.path)}
        >
          {name}
        </Button>
      ),
    },
    {
      title: tr("类型", "Type"),
      dataIndex: "type",
      width: 90,
      render: (type: FineWikiPathEntry["type"]) => type === "directory" ? tr("目录", "Directory") : "Parquet",
    },
    {
      title: tr("大小", "Size"),
      dataIndex: "size",
      width: 100,
      render: (size?: number) => size == null ? "—" : formatBytes(size),
    },
  ];

  return (
    <div className="page-stack">
      <div className="page-heading">
        <div>
          <Typography.Text className="eyebrow">BULK INGESTION</Typography.Text>
          <Typography.Title level={2}>{tr("导入任务", "Import Jobs")}</Typography.Title>
          <Typography.Paragraph type="secondary">
            {tr("创建 FineWiki 异步任务并持续查看文档、切片数量和失败原因。", "Create asynchronous FineWiki jobs and monitor document counts, chunks, and failures.")}
          </Typography.Paragraph>
        </div>
        <Button icon={<ReloadOutlined />} onClick={() => void load()}>{tr("刷新", "Refresh")}</Button>
      </div>
      <Row gutter={[8, 8]}>
        <Col xs={24} xl={9}>
          <Card title={tr("新建 FineWiki 导入", "New FineWiki Import")}>
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
              <Form.Item label={tr("Parquet 文件或目录", "Parquet file or directory")} required>
                <Space.Compact block>
                  <Form.Item name="path" noStyle rules={[{ required: true }]}>
                    <Input />
                  </Form.Item>
                  <Button icon={<FolderOpenOutlined />} onClick={() => void openPathBrowser()}>
                    {tr("浏览", "Browse")}
                  </Button>
                </Space.Compact>
              </Form.Item>
              <Form.Item label={tr("知识库", "Knowledge base")} name="knowledge_base_id" rules={[{ required: true }]}>
                <Select options={knowledgeBases.map((item) => ({ value: item.id, label: item.name }))} />
              </Form.Item>
              <Form.Item label={tr("数据来源", "Data source")} name="source" rules={[{ required: true }]}>
                <Input />
              </Form.Item>
              <Form.Item label={tr("指定标题（每行一个，可选）", "Specific titles (one per line, optional)")} name="titles">
                <Input.TextArea rows={4} placeholder={tr("首都\n秦始皇", "Capital city\nQin Shi Huang")} />
              </Form.Item>
              <Row gutter={8}>
                <Col span={12}>
                  <Form.Item label={tr("限制文档数，0 为全部", "Document limit (0 for all)")} name="limit">
                    <InputNumber min={0} style={{ width: "100%" }} />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item label={tr("文档批次", "Batch size")} name="batch_size">
                    <InputNumber min={1} max={128} style={{ width: "100%" }} />
                  </Form.Item>
                </Col>
              </Row>
              <Form.Item name="recreate" valuePropName="checked">
                <Checkbox>{tr("导入前重建 OpenSearch 索引", "Recreate OpenSearch index before importing")}</Checkbox>
              </Form.Item>
              <Alert
                type="warning"
                showIcon
                message={tr("重建 OpenSearch 索引会删除当前 Wiki 和上传文档索引，请谨慎使用。", "Recreating the OpenSearch index deletes the current Wiki and uploaded-document indexes. Use with care.")}
                className="form-alert"
              />
              <Button type="primary" icon={<ImportOutlined />} block loading={submitting} onClick={() => void submit()}>
                {tr("提交导入任务", "Submit import job")}
              </Button>
            </Form>
          </Card>
        </Col>
        <Col xs={24} xl={15}>
          <Card title={tr("任务记录", "Job History")}>
            <Table rowKey="id" dataSource={jobs} columns={columns} pagination={{ pageSize: 10 }} scroll={{ x: 980 }} />
          </Card>
        </Col>
      </Row>
      <Modal
        open={pathBrowserOpen}
        title={tr("选择服务器上的 Parquet 文件或目录", "Select a Parquet file or directory on the server")}
        width={760}
        onCancel={() => setPathBrowserOpen(false)}
        footer={[
          <Button key="cancel" onClick={() => setPathBrowserOpen(false)}>{tr("取消", "Cancel")}</Button>,
          <Button
            key="directory"
            type="primary"
            icon={<FolderOpenOutlined />}
            disabled={!pathPage}
            onClick={() => pathPage && selectPath(pathPage.current)}
          >
            {tr("选择当前目录", "Select current directory")}
          </Button>,
        ]}
      >
        <Alert
          type="info"
          showIcon
          message={tr("这里显示的是 8090 后端服务器允许访问的 FineWiki 目录。", "These are the FineWiki directories accessible to the backend server on port 8090.")}
          className="form-alert"
        />
        <Space wrap style={{ marginBottom: 8 }}>
          <Button
            icon={<ArrowUpOutlined />}
            disabled={!pathPage?.parent}
            onClick={() => pathPage?.parent && void browsePath(pathPage.parent)}
          >
            {tr("上一级", "Up one level")}
          </Button>
          {pathPage?.roots.map((root) => (
            <Button key={root} onClick={() => void browsePath(root)}>{root}</Button>
          ))}
        </Space>
        <Typography.Paragraph copyable={{ text: pathPage?.current || "" }} ellipsis>
          {pathPage?.current || tr("正在读取目录...", "Loading directory...")}
        </Typography.Paragraph>
        <Table
          rowKey="path"
          size="small"
          loading={pathBrowserLoading}
          columns={pathColumns}
          dataSource={pathPage?.entries || []}
          pagination={false}
          scroll={{ y: 420 }}
        />
      </Modal>
    </div>
  );
}
