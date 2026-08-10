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
import { useLanguage } from "../i18n";

export default function DashboardPage() {
  const { tr } = useLanguage();
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
          <Typography.Title level={2}>{tr("运行概览", "System Overview")}</Typography.Title>
          <Typography.Paragraph type="secondary">
            {tr("实时检查 MongoDB、OpenSearch BM25 索引和后台任务状态。", "Monitor MongoDB, the OpenSearch BM25 index, and background jobs in real time.")}
          </Typography.Paragraph>
        </div>
        {health && (
          <Tag icon={<CheckCircleFilled />} color={coreServicesHealthy ? "success" : "warning"}>
            {coreServicesHealthy ? tr("核心服务正常", "Core services healthy") : tr("核心服务异常", "Core services degraded")}
          </Tag>
        )}
      </div>
      {error && <Alert type="error" showIcon message={error} />}
      <Row gutter={[8, 8]}>
        <Col xs={24} md={12} xl={6}>
          <Card className="metric-card">
            <Statistic title={tr("知识库", "Knowledge bases")} value={knowledgeBases.length} prefix={<DatabaseOutlined />} />
          </Card>
        </Col>
        <Col xs={24} md={12} xl={6}>
          <Card className="metric-card">
            <Statistic title={tr("管理文档", "Documents")} value={files.length} prefix={<FileTextOutlined />} />
          </Card>
        </Col>
        <Col xs={24} md={12} xl={6}>
          <Card className="metric-card">
            <Statistic title={tr("进行中任务", "Active jobs")} value={activeJobs} prefix={<ThunderboltOutlined />} />
          </Card>
        </Col>
        <Col xs={24} md={12} xl={6}>
          <Card className="metric-card">
            <Statistic
              title={tr("BM25 索引切片", "BM25 indexed chunks")}
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
                <Descriptions.Item label={tr("状态", "Status")}>
                  <Tag color={health.mongodb.ok ? "success" : "error"}>
                    {health.mongodb.ok ? tr("正常", "Healthy") : tr("异常", "Error")}
                  </Tag>
                </Descriptions.Item>
                <Descriptions.Item label={tr("延迟", "Latency")}>
                  {String(health.mongodb.latency_ms ?? "—")} ms
                </Descriptions.Item>
              </Descriptions>
            </Card>
          </Col>
          <Col xs={24} xl={12}>
            <Card title="OpenSearch BM25" className="service-card">
              <Descriptions column={1} size="small">
                <Descriptions.Item label={tr("算法", "Algorithm")}>
                  {String(health.lexical.algorithm ?? "—")}
                </Descriptions.Item>
                <Descriptions.Item label={tr("索引切片", "Indexed chunks")}>
                  {formatNumber(Number(health.lexical.documents ?? 0))}
                </Descriptions.Item>
                <Descriptions.Item label={tr("索引", "Index")}>
                  {String(health.lexical.index ?? "—")}
                </Descriptions.Item>
                <Descriptions.Item label={tr("服务地址", "Service URL")}>
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
