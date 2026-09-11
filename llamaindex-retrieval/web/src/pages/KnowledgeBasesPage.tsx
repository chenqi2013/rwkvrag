import { DeleteOutlined, EditOutlined, PlusOutlined } from "@ant-design/icons";
import { Button, Card, Form, Input, List, message, Modal, Popconfirm, Space, Typography } from "antd";
import { useCallback, useEffect, useState } from "react";

import { api } from "../api";
import type { KnowledgeBase } from "../types";
import { errorMessage, formatDate } from "../utils";
import { useLanguage } from "../i18n";

interface FormValues {
  name: string;
  description: string;
}

export default function KnowledgeBasesPage() {
  const { tr } = useLanguage();
  const [items, setItems] = useState<KnowledgeBase[]>([]);
  const [loading, setLoading] = useState(true);
  const [open, setOpen] = useState(false);
  const [editing, setEditing] = useState<KnowledgeBase>();
  const [form] = Form.useForm<FormValues>();

  const load = useCallback(async () => {
    setLoading(true);
    try {
      setItems(await api.knowledgeBases());
    } catch (error) {
      void message.error(errorMessage(error));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => void load(), [load]);

  const showEditor = (item?: KnowledgeBase) => {
    setEditing(item);
    form.setFieldsValue({ name: item?.name || "", description: item?.description || "" });
    setOpen(true);
  };

  const save = async () => {
    const values = await form.validateFields();
    try {
      if (editing) await api.updateKnowledgeBase(editing.id, values);
      else await api.createKnowledgeBase(values);
      setOpen(false);
      await load();
      void message.success(editing ? tr("知识库已更新", "Knowledge base updated") : tr("知识库已创建", "Knowledge base created"));
    } catch (error) {
      void message.error(errorMessage(error));
    }
  };

  const remove = async (id: string) => {
    try {
      await api.deleteKnowledgeBase(id);
      await load();
      void message.success(tr("知识库及其索引已删除", "Knowledge base and its index deleted"));
    } catch (error) {
      void message.error(errorMessage(error));
    }
  };

  return (
    <div className="page-stack">
      <div className="page-heading">
        <div>
          <Typography.Text className="eyebrow">KNOWLEDGE SPACES</Typography.Text>
          <Typography.Title level={2}>{tr("知识库管理", "Knowledge Bases")}</Typography.Title>
          <Typography.Paragraph type="secondary">
            {tr("使用知识库隔离不同部门、产品或业务线的文档与检索结果。", "Use knowledge bases to isolate documents and search results by team, product, or business line.")}
          </Typography.Paragraph>
        </div>
        <Button type="primary" icon={<PlusOutlined />} onClick={() => showEditor()}>
          {tr("新建知识库", "New knowledge base")}
        </Button>
      </div>
      <Card>
        <List
          loading={loading}
          dataSource={items}
          renderItem={(item) => (
            <List.Item
              actions={[
                <Button key="edit" type="text" icon={<EditOutlined />} onClick={() => showEditor(item)}>
                  {tr("编辑", "Edit")}
                </Button>,
                <Popconfirm
                  key="delete"
                  title={tr("删除知识库", "Delete knowledge base")}
                  description={tr("将同时删除该知识库的文件记录和索引，且不可恢复。", "This permanently deletes its file records and index.")}
                  disabled={item.id === "default"}
                  onConfirm={() => void remove(item.id)}
                >
                  <Button danger type="text" icon={<DeleteOutlined />} disabled={item.id === "default"}>
                    {tr("删除", "Delete")}
                  </Button>
                </Popconfirm>,
              ]}
            >
              <List.Item.Meta
                title={
                  <Space>
                    <Typography.Text strong>{item.name}</Typography.Text>
                    <Typography.Text type="secondary">{item.file_count} {tr("个文件", "files")}</Typography.Text>
                  </Space>
                }
                description={
                  <Space direction="vertical" size={2}>
                    <Typography.Text type="secondary">{item.description || tr("暂无描述", "No description")}</Typography.Text>
                    <Typography.Text type="secondary">{tr("创建于", "Created")} {formatDate(item.created_at)}</Typography.Text>
                  </Space>
                }
              />
            </List.Item>
          )}
        />
      </Card>
      <Modal
        title={editing ? tr("编辑知识库", "Edit knowledge base") : tr("新建知识库", "New knowledge base")}
        open={open}
        onCancel={() => setOpen(false)}
        onOk={() => void save()}
        okText={tr("保存", "Save")}
      >
        <Form form={form} layout="vertical">
          <Form.Item label={tr("名称", "Name")} name="name" rules={[{ required: true, message: tr("请输入名称", "Please enter a name") }]}>
            <Input maxLength={100} />
          </Form.Item>
          <Form.Item label={tr("描述", "Description")} name="description">
            <Input.TextArea rows={4} maxLength={500} showCount />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
}
