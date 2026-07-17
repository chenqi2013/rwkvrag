import {
  CloudDownloadOutlined,
  CloudUploadOutlined,
  CopyOutlined,
  DatabaseOutlined,
  DeleteOutlined,
  LinkOutlined,
  ReloadOutlined,
} from "@ant-design/icons";
import {
  Alert,
  Button,
  Card,
  Col,
  Form,
  Input,
  message,
  Modal,
  Popconfirm,
  Row,
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
import type { CollectionItem, SnapshotItem } from "../types";
import { errorMessage, formatBytes, formatDate, formatNumber } from "../utils";

interface AliasForm {
  alias_name: string;
  collection_name: string;
}

export default function CollectionsPage() {
  const [collections, setCollections] = useState<CollectionItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedCollection, setSelectedCollection] = useState<string>();
  const [snapshots, setSnapshots] = useState<SnapshotItem[]>([]);
  const [snapshotLoading, setSnapshotLoading] = useState(false);
  const [aliasForm] = Form.useForm<AliasForm>();

  const load = useCallback(async () => {
    setLoading(true);
    try {
      setCollections(await api.collections());
    } catch (error) {
      void message.error(errorMessage(error));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => void load(), [load]);

  const openSnapshots = async (collection: string) => {
    setSelectedCollection(collection);
    setSnapshotLoading(true);
    try {
      setSnapshots(await api.snapshots(collection));
    } catch (error) {
      void message.error(errorMessage(error));
    } finally {
      setSnapshotLoading(false);
    }
  };

  const createSnapshot = async () => {
    if (!selectedCollection) return;
    setSnapshotLoading(true);
    try {
      await api.createSnapshot(selectedCollection);
      setSnapshots(await api.snapshots(selectedCollection));
      void message.success("Snapshot 已创建");
    } catch (error) {
      void message.error(errorMessage(error));
    } finally {
      setSnapshotLoading(false);
    }
  };

  const deleteSnapshot = async (snapshot: string) => {
    if (!selectedCollection) return;
    try {
      await api.deleteSnapshot(selectedCollection, snapshot);
      setSnapshots(await api.snapshots(selectedCollection));
      void message.success("Snapshot 已删除");
    } catch (error) {
      void message.error(errorMessage(error));
    }
  };

  const restoreSnapshot = async (file: File) => {
    if (!selectedCollection) return;
    setSnapshotLoading(true);
    try {
      await api.restoreSnapshot(selectedCollection, file);
      void message.success("Snapshot 恢复完成");
      await load();
    } catch (error) {
      void message.error(errorMessage(error));
    } finally {
      setSnapshotLoading(false);
    }
  };

  const switchAlias = async () => {
    const values = await aliasForm.validateFields();
    try {
      await api.switchAlias(values.alias_name, values.collection_name);
      await load();
      void message.success("Alias 已原子切换");
    } catch (error) {
      void message.error(errorMessage(error));
    }
  };

  const columns: ColumnsType<CollectionItem> = [
    {
      title: "Collection",
      dataIndex: "name",
      render: (value: string, item) => (
        <Space direction="vertical" size={2}>
          <Typography.Text strong copyable>{value}</Typography.Text>
          <Space wrap>{item.aliases.map((alias) => <Tag icon={<LinkOutlined />} key={alias}>{alias}</Tag>)}</Space>
        </Space>
      ),
    },
    {
      title: "状态",
      dataIndex: "status",
      width: 110,
      render: (value: string) => <Tag color={value.toLowerCase().includes("green") ? "success" : "warning"}>{value}</Tag>,
    },
    {
      title: "节点",
      dataIndex: "points_count",
      width: 140,
      render: (value: number) => formatNumber(value),
    },
    {
      title: "Dense 维度",
      dataIndex: "dense_dimensions",
      width: 120,
      render: (value?: number) => value || "—",
    },
    {
      title: "操作",
      width: 160,
      render: (_, item) => (
        <Button icon={<CopyOutlined />} onClick={() => void openSnapshots(item.name)}>
          Snapshot
        </Button>
      ),
    },
  ];

  const snapshotColumns: ColumnsType<SnapshotItem> = [
    { title: "名称", dataIndex: "name", render: (value: string) => <Typography.Text copyable>{value}</Typography.Text> },
    { title: "大小", dataIndex: "size", width: 120, render: formatBytes },
    { title: "创建时间", dataIndex: "created_at", width: 180, render: formatDate },
    {
      title: "操作",
      width: 180,
      render: (_, item) => (
        <Space>
          <Button
            type="text"
            icon={<CloudDownloadOutlined />}
            href={`/v1/admin/collections/${encodeURIComponent(selectedCollection || "")}/snapshots/${encodeURIComponent(item.name)}/download`}
          >
            下载
          </Button>
          <Popconfirm title="确认删除 Snapshot？" onConfirm={() => void deleteSnapshot(item.name)}>
            <Button danger type="text" icon={<DeleteOutlined />} />
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <div className="page-stack">
      <div className="page-heading">
        <div>
          <Typography.Text className="eyebrow">VECTOR OPERATIONS</Typography.Text>
          <Typography.Title level={2}>向量数据库</Typography.Title>
          <Typography.Paragraph type="secondary">
            查看 collection、创建备份、恢复快照并通过 alias 原子切换生产索引。
          </Typography.Paragraph>
        </div>
        <Button icon={<ReloadOutlined />} onClick={() => void load()}>刷新</Button>
      </div>
      <Alert
        type="warning"
        showIcon
        message="恢复 Snapshot 会覆盖目标 collection；生产切换建议恢复到新 collection，验证后再切换 alias。"
      />
      <Row gutter={[8, 8]}>
        <Col xs={24} xl={17}>
          <Card title={<Space><DatabaseOutlined />Collections</Space>}>
            <Table rowKey="name" columns={columns} dataSource={collections} loading={loading} pagination={false} scroll={{ x: 760 }} />
          </Card>
        </Col>
        <Col xs={24} xl={7}>
          <Card title="Alias 原子切换">
            <Form form={aliasForm} layout="vertical" initialValues={{ alias_name: "rwkvrag-knowledge-current" }}>
              <Form.Item label="Alias 名称" name="alias_name" rules={[{ required: true }]}>
                <Input />
              </Form.Item>
              <Form.Item label="目标 Collection" name="collection_name" rules={[{ required: true }]}>
                <Select options={collections.map((item) => ({ value: item.name, label: item.name }))} />
              </Form.Item>
              <Button type="primary" block icon={<LinkOutlined />} onClick={() => void switchAlias()}>
                切换 Alias
              </Button>
            </Form>
          </Card>
        </Col>
      </Row>
      <Modal
        open={Boolean(selectedCollection)}
        onCancel={() => setSelectedCollection(undefined)}
        footer={null}
        width={720}
        title={`${selectedCollection || ""} · Snapshots`}
      >
        <Space style={{ marginBottom: 16 }} wrap>
          <Button type="primary" icon={<CopyOutlined />} loading={snapshotLoading} onClick={() => void createSnapshot()}>
            创建 Snapshot
          </Button>
          <Upload
            accept=".snapshot"
            showUploadList={false}
            beforeUpload={(file) => {
              void restoreSnapshot(file as File);
              return Upload.LIST_IGNORE;
            }}
          >
            <Button icon={<CloudUploadOutlined />} loading={snapshotLoading}>上传并恢复</Button>
          </Upload>
        </Space>
        <Table rowKey="name" columns={snapshotColumns} dataSource={snapshots} loading={snapshotLoading} pagination={false} />
      </Modal>
    </div>
  );
}
