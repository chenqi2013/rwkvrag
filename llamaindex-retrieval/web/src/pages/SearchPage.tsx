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
import type { AskResponse, FailureCategory, KnowledgeBase } from "../types";
import { errorMessage } from "../utils";
import { useLanguage } from "../i18n";

interface SearchForm {
  question: string;
  knowledge_base_id?: string;
  top_k: number;
}

export default function SearchPage() {
  const { tr } = useLanguage();
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

  const writerCalled = response?.generation.answer_strategy === "single_writer_call"
    || response?.generation.writer_trace != null;
  const evidenceVerified = writerCalled
    && response?.generation.grounding_valid === true
    && response?.generation.answer_support_passed === true;
  const queryNormalized = response?.retrieval.query_normalized === true;
  const normalizedQuestion = String(response?.retrieval.normalized_question || "");
  const failureCategory = response?.generation.failure_category as FailureCategory | undefined;
  const failureLabels: Record<FailureCategory, [string, string]> = {
    data_missing: ["数据缺失", "Data missing"],
    retrieval_failed: ["检索失败", "Retrieval failed"],
    evidence_extraction_failed: ["证据抽取失败", "Evidence extraction failed"],
    generation_failed: ["生成失败", "Generation failed"],
  };

  return (
    <div className="page-stack">
      <div className="page-heading">
        <div>
          <Typography.Text className="eyebrow">RETRIEVAL LAB</Typography.Text>
          <Typography.Title level={2}>{tr("在线检索测试", "Online Search Lab")}</Typography.Title>
          <Typography.Paragraph type="secondary">
            {tr("调用生产 `/v1/ask`：先用 BM25 检索证据，再由 RWKV 基于证据生成答案。", "Call the production `/v1/ask` endpoint: retrieve evidence with BM25, then let RWKV answer from that evidence.")}
          </Typography.Paragraph>
        </div>
      </div>
      <Row gutter={[8, 8]}>
        <Col xs={24} xl={8}>
          <Card title={tr("查询参数", "Query Parameters")}>
            <Form
              form={form}
              layout="vertical"
              initialValues={{ top_k: 1 }}
            >
              <Form.Item label={tr("问题", "Question")} name="question" rules={[{ required: true, message: tr("请输入问题", "Please enter a question") }]}>
                <Input.TextArea rows={6} placeholder={tr("输入需要检索的问题", "Enter a question to search")} />
              </Form.Item>
              <Form.Item label={tr("知识库过滤", "Knowledge base filter")} name="knowledge_base_id">
                <Select
                  allowClear
                  placeholder={tr("全部知识库", "All knowledge bases")}
                  options={knowledgeBases.map((item) => ({ value: item.id, label: item.name }))}
                />
              </Form.Item>
              <Form.Item
                label={tr("页面显示数量", "Displayed results")}
                name="top_k"
                extra={tr(
                  "仅控制页面展示；系统会按问题类型自动使用 5～12 条证据生成答案。",
                  "Controls display only; the system automatically uses 5–12 evidence items based on question type.",
                )}
              >
                <InputNumber min={1} max={20} style={{ width: "100%" }} />
              </Form.Item>
              <Button type="primary" block icon={<SearchOutlined />} loading={loading} onClick={() => void search()}>
              {tr("检索并生成答案", "Search and generate answer")}
              </Button>
            </Form>
          </Card>
        </Col>
        <Col xs={24} xl={16}>
          <Card
            title={tr("生成答案", "Generated Answer")}
            extra={
              response && (
                <Space size={4} wrap>
                  <Tag color={evidenceVerified ? "green" : writerCalled ? "blue" : "orange"}>
                    {evidenceVerified
                      ? tr("证据校验通过", "Evidence verified")
                      : writerCalled
                        ? tr("已生成，证据校验未通过", "Generated; evidence check failed")
                        : tr("证据不足，未生成", "Insufficient evidence; not generated")}
                  </Tag>
                  {response.generation.model ? (
                    <Tag color="green">{tr("生成模型", "Model")} · {String(response.generation.model)}</Tag>
                  ) : null}
                  {failureCategory ? (
                    <Tag color="red">{tr(failureLabels[failureCategory][0], failureLabels[failureCategory][1])}</Tag>
                  ) : null}
                </Space>
              )
            }
          >
            {!response ? (
              <Empty description={tr("提交问题后查看生成答案和证据", "Submit a question to view the answer and evidence")} />
            ) : (
              <Space direction="vertical" size={12} style={{ width: "100%" }}>
                <Space wrap>
                  {queryNormalized && <Tag color="blue">{tr("已纠正查询：", "Normalized query: ")}{normalizedQuestion}</Tag>}
                  <Tag>{tr("答案需标注资料编号", "Answer must cite source numbers")}</Tag>
                  {response.retrieval.evidence_top_k_policy ? (
                    <Tag color="purple">
                      {tr("自适应证据", "Adaptive evidence")} · {String(response.retrieval.answer_evidence_top_k)}
                    </Tag>
                  ) : null}
                  {failureCategory && response.generation.failure_reason ? (
                    <Typography.Text type="danger">
                      {tr("失败原因：", "Failure reason: ")}{String(response.generation.failure_reason)}
                    </Typography.Text>
                  ) : null}
                </Space>
                <Typography.Paragraph className="result-snippet" copyable={{ text: response.answer }}>
                  {response.answer}
                </Typography.Paragraph>
                <Card
                  size="small"
                  title={tr("检索证据", "Retrieved Evidence")}
                  extra={
                    <Space>
                      <Tag color="cyan">{String(response.retrieval.algorithm)}</Tag>
                      <Tag>{String(response.retrieval.mode)}</Tag>
                      <Tag>{String(response.retrieval.returned)} {tr("条", "results")}</Tag>
                    </Space>
                  }
                >
                  <List
                    dataSource={response.sources}
                    locale={{ emptyText: <Empty description={tr("没有达到相关性阈值的结果", "No results met the relevance threshold")} /> }}
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
                                  <LinkOutlined /> {tr("查看来源", "View source")}
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
