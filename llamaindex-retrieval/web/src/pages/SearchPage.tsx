import { LinkOutlined, SearchOutlined } from "@ant-design/icons";
import {
  Button,
  Card,
  Col,
  Empty,
  Form,
  Input,
  InputNumber,
  List,
  message,
  Row,
  Select,
  Space,
  Tag,
  Typography,
} from "antd";
import { useEffect, useState } from "react";

import { api } from "../api";
import type { AskResponse, KnowledgeBase } from "../types";
import { errorMessage } from "../utils";

interface SearchForm {
  question: string;
  knowledge_base_id?: string;
  top_k: number;
}

export default function SearchPage() {
  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBase[]>([]);
  const [response, setResponse] = useState<AskResponse>();
  const [loading, setLoading] = useState(false);
  const [form] = Form.useForm<SearchForm>();

  useEffect(() => {
    api.knowledgeBases().then(setKnowledgeBases).catch((error) => void message.error(errorMessage(error)));
  }, []);

  const search = async () => {
    const values = await form.validateFields();
    setLoading(true);
    try {
      setResponse(await api.ask(values));
    } catch (error) {
      void message.error(errorMessage(error));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page-stack">
      <div className="page-heading">
        <div>
          <Typography.Text className="eyebrow">RETRIEVAL LAB</Typography.Text>
          <Typography.Title level={2}>在线检索测试</Typography.Title>
          <Typography.Paragraph type="secondary">
            调用生产 `/v1/ask`：先用 BM25 检索证据，再由 RWKV 基于证据生成答案。
          </Typography.Paragraph>
        </div>
      </div>
      <Row gutter={[8, 8]}>
        <Col xs={24} xl={8}>
          <Card title="查询参数">
            <Form
              form={form}
              layout="vertical"
              initialValues={{ question: "中国人口最多的省份是哪个省", top_k: 5 }}
            >
              <Form.Item label="问题" name="question" rules={[{ required: true, message: "请输入问题" }]}>
                <Input.TextArea rows={6} placeholder="输入需要检索的问题" />
              </Form.Item>
              <Form.Item label="知识库过滤" name="knowledge_base_id">
                <Select
                  allowClear
                  placeholder="全部知识库"
                  options={knowledgeBases.map((item) => ({ value: item.id, label: item.name }))}
                />
              </Form.Item>
              <Form.Item label="返回数量" name="top_k">
                <InputNumber min={1} max={20} style={{ width: "100%" }} />
              </Form.Item>
              <Button type="primary" block icon={<SearchOutlined />} loading={loading} onClick={() => void search()}>
              检索并生成答案
              </Button>
            </Form>
          </Card>
        </Col>
        <Col xs={24} xl={16}>
          <Card
            title="生成答案"
            extra={response && <Tag color="green">RWKV · {String(response.generation.model)}</Tag>}
          >
            {!response ? (
              <Empty description="提交问题后查看生成答案和证据" />
            ) : (
              <Space direction="vertical" size={12} style={{ width: "100%" }}>
                <Typography.Paragraph className="result-snippet" copyable={{ text: response.answer }}>
                  {response.answer}
                </Typography.Paragraph>
                <Card
                  size="small"
                  title="检索证据"
                  extra={
                    <Space>
                      <Tag color="cyan">{String(response.retrieval.algorithm)}</Tag>
                      <Tag>{String(response.retrieval.mode)}</Tag>
                      <Tag>{String(response.retrieval.returned)} 条</Tag>
                    </Space>
                  }
                >
                  <List
                    dataSource={response.sources}
                    locale={{ emptyText: <Empty description="没有达到相关性阈值的结果" /> }}
                    renderItem={(item, index) => (
                      <List.Item>
                        <Card size="small" className="result-card">
                          <Space direction="vertical" size={10} style={{ width: "100%" }}>
                            <div className="result-title-row">
                              <Space>
                                <span className="rank-badge">{index + 1}</span>
                                <Typography.Title level={4}>{item.title}</Typography.Title>
                              </Space>
                              <Tag color="blue">{item.score.toFixed(4)}</Tag>
                            </div>
                            <Typography.Paragraph className="result-snippet">{item.snippet}</Typography.Paragraph>
                            <Space wrap>
                              <Tag>{item.source}</Tag>
                              {item.uri && (
                                <Typography.Link href={item.uri} target="_blank">
                                  <LinkOutlined /> 查看来源
                                </Typography.Link>
                              )}
                            </Space>
                          </Space>
                        </Card>
                      </List.Item>
                    )}
                  />
                </Card>
              </Space>
            )}
          </Card>
        </Col>
      </Row>
    </div>
  );
}
