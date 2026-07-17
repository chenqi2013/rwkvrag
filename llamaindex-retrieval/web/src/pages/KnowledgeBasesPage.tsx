import { DeleteOutlined, EditOutlined, PlusOutlined } from "@ant-design/icons";
import { Button, Card, Form, Input, List, message, Modal, Popconfirm, Space, Typography } from "antd";
import { useCallback, useEffect, useState } from "react";

import { api } from "../api";
import type { KnowledgeBase } from "../types";
import { errorMessage, formatDate } from "../utils";

interface FormValues {
  name: string;
  description: string;
}

export default function KnowledgeBasesPage() {
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
      void message.success(editing ? "知识库已更新" : "知识库已创建");
    } catch (error) {
      void message.error(errorMessage(error));
    }
  };

  const remove = async (id: string) => {
    try {
      await api.deleteKnowledgeBase(id);
      await load();
      void message.success("知识库及其向量已删除");
    } catch (error) {
      void message.error(errorMessage(error));
    }
  };

  return (
    <div className="page-stack">
      <div className="page-heading">
        <div>
          <Typography.Text className="eyebrow">KNOWLEDGE SPACES</Typography.Text>
          <Typography.Title level={2}>知识库管理</Typography.Title>
          <Typography.Paragraph type="secondary">
            使用知识库隔离不同部门、产品或业务线的文档与检索结果。
          </Typography.Paragraph>
        </div>
        <Button type="primary" icon={<PlusOutlined />} onClick={() => showEditor()}>
          新建知识库
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
                  编辑
                </Button>,
                <Popconfirm
                  key="delete"
                  title="删除知识库"
                  description="将同时删除该知识库的文件记录和向量，且不可恢复。"
                  disabled={item.id === "default"}
                  onConfirm={() => void remove(item.id)}
                >
                  <Button danger type="text" icon={<DeleteOutlined />} disabled={item.id === "default"}>
                    删除
                  </Button>
                </Popconfirm>,
              ]}
            >
              <List.Item.Meta
                title={
                  <Space>
                    <Typography.Text strong>{item.name}</Typography.Text>
                    <Typography.Text type="secondary">{item.file_count} 个文件</Typography.Text>
                  </Space>
                }
                description={
                  <Space direction="vertical" size={2}>
                    <Typography.Text type="secondary">{item.description || "暂无描述"}</Typography.Text>
                    <Typography.Text type="secondary">创建于 {formatDate(item.created_at)}</Typography.Text>
                  </Space>
                }
              />
            </List.Item>
          )}
        />
      </Card>
      <Modal
        title={editing ? "编辑知识库" : "新建知识库"}
        open={open}
        onCancel={() => setOpen(false)}
        onOk={() => void save()}
        okText="保存"
      >
        <Form form={form} layout="vertical">
          <Form.Item label="名称" name="name" rules={[{ required: true, message: "请输入名称" }]}>
            <Input maxLength={100} />
          </Form.Item>
          <Form.Item label="描述" name="description">
            <Input.TextArea rows={4} maxLength={500} showCount />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
}
