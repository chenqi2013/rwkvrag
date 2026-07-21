import {
  CheckCircleFilled,
  CloudServerOutlined,
  DatabaseOutlined,
  FileTextOutlined,
  ThunderboltOutlined,
} from "@ant-design/icons";
import { Alert, Card, Col, Descriptions, Row, Skeleton, Statistic, Tag, Typography } from "antd";
import { useCallback, useEffect, useState } from "react";

import { api } from "../api";
import type { AdminHealth, CollectionItem, FileItem, JobItem, KnowledgeBase } from "../types";
import { errorMessage, formatNumber } from "../utils";

export default function DashboardPage() {
  const [health, setHealth] = useState<AdminHealth>();
  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBase[]>([]);
  const [files, setFiles] = useState<FileItem[]>([]);
  const [jobs, setJobs] = useState<JobItem[]>([]);
  const [collections, setCollections] = useState<CollectionItem[]>([]);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    const results = await Promise.allSettled([
      api.health(),
      api.knowledgeBases(),
      api.files(),
      api.jobs(),
      api.collections(),
    ] as const);
    const [healthResult, knowledgeBaseResult, fileResult, jobResult, collectionResult] = results;
    if (healthResult.status === "fulfilled") setHealth(healthResult.value);
    if (knowledgeBaseResult.status === "fulfilled") setKnowledgeBases(knowledgeBaseResult.value);
    if (fileResult.status === "fulfilled") setFiles(fileResult.value);
    if (jobResult.status === "fulfilled") setJobs(jobResult.value);
    if (collectionResult.status === "fulfilled") setCollections(collectionResult.value);
    const failures = results
      .filter((result): result is PromiseRejectedResult => result.status === "rejected")
      .map((result) => errorMessage(result.reason));
    setError([...new Set(failures)].join("；"));
  }, []);

  useEffect(() => {
    void load();
    const timer = window.setInterval(() => void load(), 15000);
    return () => window.clearInterval(timer);
  }, [load]);

  const activeJobs = jobs.filter((job) => ["pending", "running"].includes(job.status)).length;
  const activeCollection = collections.find((item) => item.aliases.includes("rwkvrag-knowledge-current"));

  if (!health && !error) return <Skeleton active />;

  return (
    <div className="page-stack">
      <div className="page-heading">
        <div>
          <Typography.Text className="eyebrow">SYSTEM OVERVIEW</Typography.Text>
          <Typography.Title level={2}>运行概览</Typography.Title>
          <Typography.Paragraph type="secondary">
            实时检查 MongoDB、Qdrant 和 Embedding 服务状态。
          </Typography.Paragraph>
        </div>
        {health && (
          <Tag icon={<CheckCircleFilled />} color={health.status === "ok" ? "success" : "warning"}>
            {health.status === "ok" ? "全部服务正常" : "服务降级"}
          </Tag>
        )}
      </div>
      {error && <Alert type="error" showIcon message={error} />}
      <Row gutter={[8, 8]}>
        <Col xs={24} md={12} xl={6}>
          <Card className="metric-card">
            <Statistic title="知识库" value={knowledgeBases.length} prefix={<DatabaseOutlined />} />
          </Card>
        </Col>
        <Col xs={24} md={12} xl={6}>
          <Card className="metric-card">
            <Statistic title="管理文档" value={files.length} prefix={<FileTextOutlined />} />
          </Card>
        </Col>
        <Col xs={24} md={12} xl={6}>
          <Card className="metric-card">
            <Statistic title="进行中任务" value={activeJobs} prefix={<ThunderboltOutlined />} />
          </Card>
        </Col>
        <Col xs={24} md={12} xl={6}>
          <Card className="metric-card">
            <Statistic
              title="当前向量节点"
              value={activeCollection?.points_count || 0}
              formatter={(value) => formatNumber(Number(value))}
              prefix={<CloudServerOutlined />}
            />
          </Card>
        </Col>
      </Row>
      {health && (
        <Row gutter={[8, 8]}>
          <Col xs={24} xl={8}>
            <Card title="MongoDB" className="service-card">
              <Descriptions column={1} size="small">
                <Descriptions.Item label="状态">
                  <Tag color={health.mongodb.ok ? "success" : "error"}>
                    {health.mongodb.ok ? "正常" : "异常"}
                  </Tag>
                </Descriptions.Item>
                <Descriptions.Item label="延迟">
                  {String(health.mongodb.latency_ms ?? "—")} ms
                </Descriptions.Item>
              </Descriptions>
            </Card>
          </Col>
          <Col xs={24} xl={8}>
            <Card title="Qdrant" className="service-card">
              <Descriptions column={1} size="small">
                <Descriptions.Item label="状态">
                  <Tag color={health.qdrant.ok ? "success" : "error"}>
                    {health.qdrant.ok ? "正常" : "异常"}
                  </Tag>
                </Descriptions.Item>
                <Descriptions.Item label="Collections">
                  {String(health.qdrant.collections ?? "—")}
                </Descriptions.Item>
              </Descriptions>
            </Card>
          </Col>
          <Col xs={24} xl={8}>
            <Card title="Embedding" className="service-card">
              <Descriptions column={1} size="small">
                <Descriptions.Item label="模型">
                  {String(health.embedding.model ?? "—")}
                </Descriptions.Item>
                <Descriptions.Item label="向量维度">
                  {String(health.embedding.dimensions ?? "—")}
                </Descriptions.Item>
                <Descriptions.Item label="延迟">
                  {String(health.embedding.latency_ms ?? "—")} ms
                </Descriptions.Item>
              </Descriptions>
            </Card>
          </Col>
        </Row>
      )}
    </div>
  );
}
