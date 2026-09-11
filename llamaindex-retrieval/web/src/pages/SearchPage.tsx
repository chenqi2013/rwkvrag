import { LinkOutlined, SearchOutlined } from "@ant-design/icons";
import {
  Alert,
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
import { answerPresentation, evidenceWarnings } from "../answerPresentation";

interface ConversationTurn {
  role: "user" | "assistant";
  content: string;
}

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
  const [history, setHistory] = useState<ConversationTurn[]>([]);
  const [form] = Form.useForm<SearchForm>();

  useEffect(() => {
    api.knowledgeBases().then(setKnowledgeBases).catch((error) => void message.error(errorMessage(error)));
  }, []);

  const search = async () => {
    const values = await form.validateFields();
    setLoading(true);
    try {
      const next = await api.ask({ ...values, history });
      setResponse(next);
      const display = answerPresentation(next);
      // Keep the raw response visible; carry only a completed answer body forward.
      const completed = display.isNative
        ? next.generation.status === "completed" && next.generation.writer_status === "completed"
        : next.generation.answer_strategy === "single_writer_call";
      if (completed && display.spanValid && display.answerText.trim()) {
        setHistory((previous) => [...previous,
          { role: "user", content: values.question },
          { role: "assistant", content: display.answerText },
        ]);
        if (form.getFieldValue("question") === values.question) {
          form.setFieldValue("question", "");
        }
      }
    } catch (error) {
      void message.error(errorMessage(error));
    } finally {
      setLoading(false);
    }
  };

  const presentation = answerPresentation(response);
  const warnings = evidenceWarnings(response);
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
      <Space wrap>
        <Button disabled={loading || history.length === 0} onClick={() => {
          setHistory([]);
          setResponse(undefined);
          form.setFieldValue("question", "");
        }}>{tr("开始新对话", "New conversation")}</Button>
        <Typography.Text type="secondary">
          {tr("追问会携带本页对话历史；刷新页面会开始新对话。", "Follow-up questions include this page’s conversation; reloading starts a new conversation.")}
        </Typography.Text>
        {history.length >= 64 && <Typography.Text type="warning">
          {tr("当前对话已达 32 轮，请开始新对话。", "This conversation has reached 32 turns. Start a new conversation.")}
        </Typography.Text>}
      </Space>
      {history.length > 0 && <details>
        <summary>{tr("对话历史", "Conversation history")} ({history.length / 2})</summary>
        {history.map((turn, index) => <Card key={index} size="small"
          title={turn.role === "user" ? tr("你", "You") : "RWKV"}>
          <Typography.Paragraph className="result-snippet">{turn.content}</Typography.Paragraph>
        </Card>)}
      </details>}
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
                  "实际证据范围由当前检索流程决定；原生 RWKV 会保留完整引用来源列表。",
                  "The active retrieval pipeline determines the evidence; native RWKV retains the complete citation source list.",
                )}
              >
                <InputNumber min={1} max={20} style={{ width: "100%" }} />
              </Form.Item>
              <Button type="primary" block icon={<SearchOutlined />} loading={loading} disabled={history.length >= 64} onClick={() => void search()}>
              {tr("检索并生成答案", "Search and generate answer")}
              </Button>
            </Form>
          </Card>
        </Col>
        <Col xs={24} xl={16}>
          <Card className="answer-card" title={tr("生成答案", "Generated Answer")}>
            {!response ? (
              <Empty description={tr("提交问题后查看生成答案和证据", "Submit a question to view the answer and evidence")} />
            ) : (
              <Space direction="vertical" size={12} style={{ width: "100%" }}>
                <Space size={4} wrap className="answer-status">
                  <Tag color={presentation.color}>
                    {tr(...presentation.label)}
                  </Tag>
                  {response.generation.model ? (
                    <Tag color="green">{tr("生成模型", "Model")} · {String(response.generation.model)}</Tag>
                  ) : null}
                  {failureCategory ? (
                    <Tag color="red">{tr(failureLabels[failureCategory][0], failureLabels[failureCategory][1])}</Tag>
                  ) : null}
                </Space>
                {warnings.map((warning) => <Alert key={warning[1]} type="warning" showIcon title={tr(...warning)} />)}
                {(queryNormalized || !!response.retrieval.evidence_top_k_policy || !!(failureCategory && response.generation.failure_reason)) && <Space wrap className="answer-status">
                  {queryNormalized && <Tag color="blue">{tr("已纠正查询：", "Normalized query: ")}{normalizedQuestion}</Tag>}
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
                </Space>}
                {presentation.answerText ? (
                  <Typography.Paragraph className="result-snippet answer-body" copyable={{ text: presentation.answerText }}>
                    {presentation.answerText}
                  </Typography.Paragraph>
                ) : <Typography.Text type="secondary">{tr("未提供可显示的答案正文。", "No answer body is available.")}</Typography.Text>}
                {presentation.isNative && (
                  <details className="answer-trace">
                    <summary>{tr("原始模型输出与运行记录", "Raw model output and trace")}</summary>
                    <Typography.Paragraph className="result-snippet raw-output" copyable={{ text: presentation.rawAnswer }}>
                      {presentation.rawAnswer || tr("无原始输出", "No raw output")}
                    </Typography.Paragraph>
                    <details>
                      <summary>{tr("完整运行记录", "Full trace")}</summary>
                      <pre className="trace-json" tabIndex={0} aria-label={tr("完整运行记录", "Full trace")}>
                        {JSON.stringify({ retrieval: response.retrieval, generation: response.generation }, null, 2)}
                      </pre>
                    </details>
                  </details>
                )}
                <Card
                  size="small"
                  title={tr("检索证据", "Retrieved Evidence")}
                >
                  <Space wrap className="answer-status">
                    {response.retrieval.algorithm ? <Tag color="cyan">{String(response.retrieval.algorithm)}</Tag> : null}
                    <Tag>{String(response.retrieval.mode)}</Tag>
                    <Tag>{response.sources.length} {tr("条", "results")}</Tag>
                  </Space>
                  <List
                    dataSource={response.sources}
                    locale={{ emptyText: <Empty description={tr("没有返回证据", "No evidence returned")} /> }}
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
