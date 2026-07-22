import {
  DeleteOutlined,
  DownloadOutlined,
  EyeOutlined,
  InboxOutlined,
  ReloadOutlined,
} from "@ant-design/icons";
import {
  Alert,
  Button,
  Card,
  Descriptions,
  Drawer,
  Empty,
  List,
  message,
  Popconfirm,
  Progress,
  Select,
  Space,
  Table,
  Tag,
  Typography,
  Upload,
} from "antd";
import type { ColumnsType } from "antd/es/table";
import { useCallback, useEffect, useState } from "react";

import { api } from "../api";
import type { ChunkItem, FileItem, KnowledgeBase } from "../types";
import { errorMessage, formatBytes, formatDate } from "../utils";

const statusMap: Record<FileItem["status"], { color: string; text: string }> = {
  pending: { color: "default", text: "等待处理" },
  processing: { color: "processing", text: "处理中" },
  ready: { color: "success", text: "已就绪" },
  failed: { color: "error", text: "失败" },
  deleting: { color: "warning", text: "删除中" },
};

export default function FilesPage() {
  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBase[]>([]);
  const [knowledgeBaseId, setKnowledgeBaseId] = useState("default");
  const [files, setFiles] = useState<FileItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [chunks, setChunks] = useState<ChunkItem[]>([]);
  const [chunkFile, setChunkFile] = useState<FileItem>();
  const [chunksLoading, setChunksLoading] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [knowledgeBaseData, fileData] = await Promise.all([
        api.knowledgeBases(),
        api.files(knowledgeBaseId),
      ]);
      setKnowledgeBases(knowledgeBaseData);
      setFiles(fileData);
    } catch (error) {
      void message.error(errorMessage(error));
    } finally {
      setLoading(false);
    }
  }, [knowledgeBaseId]);

  useEffect(() => void load(), [load]);

  useEffect(() => {
    if (!files.some((item) => ["pending", "processing"].includes(item.status))) return;
    const timer = window.setInterval(() => void load(), 2000);
    return () => window.clearInterval(timer);
  }, [files, load]);

  const upload = async (file: File) => {
    setUploading(true);
    try {
      await api.uploadFile(file, knowledgeBaseId);
      void message.success(`${file.name} 已进入处理队列`);
      await load();
    } catch (error) {
      void message.error(errorMessage(error));
    } finally {
      setUploading(false);
    }
  };

  const remove = async (id: string) => {
    try {
      await api.deleteFile(id);
      await load();
      void message.success("文件及对应索引已删除");
    } catch (error) {
      void message.error(errorMessage(error));
    }
  };

  const reindex = async (id: string) => {
    try {
      await api.reindexFile(id);
      await load();
      void message.success("重新索引任务已提交");
    } catch (error) {
      void message.error(errorMessage(error));
    }
  };

  const showChunks = async (file: FileItem) => {
    setChunkFile(file);
    setChunksLoading(true);
    try {
      setChunks((await api.fileChunks(file.id)).items);
    } catch (error) {
      void message.error(errorMessage(error));
    } finally {
      setChunksLoading(false);
    }
  };

  const columns: ColumnsType<FileItem> = [
    {
      title: "文件",
      dataIndex: "filename",
      render: (value: string, item) => (
        <Space direction="vertical" size={0}>
          <Typography.Text strong>{value}</Typography.Text>
          <Typography.Text type="secondary">
            {item.extension.toUpperCase()} · {formatBytes(item.size)}
          </Typography.Text>
        </Space>
      ),
    },
    {
      title: "状态",
      dataIndex: "status",
      width: 130,
      render: (status: FileItem["status"], item) => (
        <Space direction="vertical" size={3}>
          <Tag color={statusMap[status].color}>{statusMap[status].text}</Tag>
          {status === "processing" && <Progress percent={60} size="small" showInfo={false} status="active" />}
          {item.error && <Typography.Text type="danger">{item.error}</Typography.Text>}
        </Space>
      ),
    },
    {
      title: "切片",
      dataIndex: "node_count",
      width: 100,
      render: (value: number) => `${value} 个`,
    },
    {
      title: "上传时间",
      dataIndex: "created_at",
      width: 180,
      render: formatDate,
    },
    {
      title: "操作",
      key: "actions",
      width: 300,
      render: (_, item) => (
        <Space wrap>
          <Button type="text" icon={<EyeOutlined />} disabled={item.status !== "ready"} onClick={() => void showChunks(item)}>
            切片
          </Button>
          <Button type="text" icon={<DownloadOutlined />} href={`/v1/admin/files/${item.id}/download`}>
            下载
          </Button>
          <Button type="text" icon={<ReloadOutlined />} disabled={item.status !== "ready" && item.status !== "failed"} onClick={() => void reindex(item.id)}>
            重建
          </Button>
          <Popconfirm title="确认删除文件和全部索引？" onConfirm={() => void remove(item.id)}>
            <Button danger type="text" icon={<DeleteOutlined />} disabled={["pending", "processing"].includes(item.status)}>
              删除
            </Button>
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <div className="page-stack">
      <div className="page-heading">
        <div>
          <Typography.Text className="eyebrow">DOCUMENT PIPELINE</Typography.Text>
          <Typography.Title level={2}>文档管理</Typography.Title>
          <Typography.Paragraph type="secondary">
            上传 Markdown、PDF 或 DOCX，系统会自动解析、切片并写入 OpenSearch BM25 索引。
          </Typography.Paragraph>
        </div>
        <Select
          value={knowledgeBaseId}
          style={{ width: 180 }}
          options={knowledgeBases.map((item) => ({ value: item.id, label: item.name }))}
          onChange={setKnowledgeBaseId}
        />
      </div>
      <Alert
        showIcon
        type="info"
        message="PDF 说明"
        description="当前支持包含文字层的 PDF；扫描版 PDF 需要先完成 OCR。单文件最大 100MB。"
      />
      <Card>
        <Upload.Dragger
          accept=".md,.markdown,.mdx,.pdf,.docx"
          multiple
          showUploadList={false}
          disabled={uploading}
          beforeUpload={(file) => {
            void upload(file as File);
            return Upload.LIST_IGNORE;
          }}
        >
          <p className="ant-upload-drag-icon"><InboxOutlined /></p>
          <p className="ant-upload-text">点击或拖拽文件到这里上传</p>
          <p className="ant-upload-hint">支持 Markdown、PDF 和 DOCX，可一次选择多个文件</p>
        </Upload.Dragger>
      </Card>
      <Card>
        <Table
          rowKey="id"
          columns={columns}
          dataSource={files}
          loading={loading}
          pagination={{ pageSize: 10 }}
          locale={{ emptyText: <Empty description="当前知识库还没有上传文件" /> }}
          scroll={{ x: 1000 }}
        />
      </Card>
      <Drawer
        open={Boolean(chunkFile)}
        onClose={() => setChunkFile(undefined)}
        width={620}
        title={`${chunkFile?.filename || ""} · 切片内容`}
      >
        <List
          loading={chunksLoading}
          dataSource={chunks}
          renderItem={(item, index) => (
            <List.Item>
              <Card size="small" title={`切片 ${index + 1}`} className="chunk-card">
                <Typography.Paragraph className="chunk-text">{item.text}</Typography.Paragraph>
                <Descriptions
                  size="small"
                  column={1}
                  items={Object.entries(item.metadata).map(([key, value]) => ({
                    key,
                    label: key,
                    children: String(value),
                  }))}
                />
              </Card>
            </List.Item>
          )}
        />
      </Drawer>
    </div>
  );
}
