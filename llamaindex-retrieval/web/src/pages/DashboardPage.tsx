import {
  CheckCircleFilled,
  DatabaseOutlined,
  FileTextOutlined,
  SearchOutlined,
  ThunderboltOutlined,
} from "@ant-design/icons";
import { Alert, Card, Col, Descriptions, Row, Skeleton, Statistic, Tag, Typography } from "antd";
import { useCallback, useEffect, useState } from "react";

import { api } from "../api";
import type { AdminHealth, FileItem, JobItem, KnowledgeBase } from "../types";
import { errorMessage, formatNumber } from "../utils";

export default function DashboardPage() {
  const [health, setHealth] = useState<AdminHealth>();
  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBase[]>([]);
  const [files, setFiles] = useState<FileItem[]>([]);
  const [jobs, setJobs] = useState<JobItem[]>([]);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    const results = await Promise.allSettled([
      api.health(),
      api.knowledgeBases(),
      api.files(),
      api.jobs(),
    ] as const);
    const [healthResult, knowledgeBaseResult, fileResult, jobResult] = results;
    if (healthResult.status === "fulfilled") setHealth(healthResult.value);
    if (knowledgeBaseResult.status === "fulfilled") setKnowledgeBases(knowledgeBaseResult.value);
    if (fileResult.status === "fulfilled") setFiles(fileResult.value);
    if (jobResult.status === "fulfilled") setJobs(jobResult.value);
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
  const coreServicesHealthy = Boolean(health?.mongodb.ok && health?.lexical.ok);

  if (!health && !error) return <Skeleton active />;

  return (
    <div className="page-stack">
      <div className="page-heading">
        <div>
          <Typography.Text className="eyebrow">SYSTEM OVERVIEW</Typography.Text>
          <Typography.Title level={2}>运行概览</Typography.Title>
          <Typography.Paragraph type="secondary">
            实时检查 MongoDB、OpenSearch BM25 索引和后台任务状态。
          </Typography.Paragraph>
        </div>
        {health && (
          <Tag icon={<CheckCircleFilled />} color={coreServicesHealthy ? "success" : "warning"}>
            {coreServicesHealthy ? "核心服务正常" : "核心服务异常"}
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
              title="BM25 索引切片"
              value={Number(health?.lexical?.documents ?? 0)}
              formatter={(value) => formatNumber(Number(value))}
              prefix={<SearchOutlined />}
            />
          </Card>
        </Col>
      </Row>
      {health && (
        <Row gutter={[8, 8]}>
          <Col xs={24} xl={12}>
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
          <Col xs={24} xl={12}>
            <Card title="OpenSearch BM25" className="service-card">
              <Descriptions column={1} size="small">
                <Descriptions.Item label="算法">
                  {String(health.lexical.algorithm ?? "—")}
                </Descriptions.Item>
                <Descriptions.Item label="索引切片">
                  {formatNumber(Number(health.lexical.documents ?? 0))}
                </Descriptions.Item>
                <Descriptions.Item label="索引">
                  {String(health.lexical.index ?? "—")}
                </Descriptions.Item>
                <Descriptions.Item label="服务地址">
                  {String(health.lexical.url ?? "—")}
                </Descriptions.Item>
              </Descriptions>
            </Card>
          </Col>
        </Row>
      )}
    </div>
  );
}
