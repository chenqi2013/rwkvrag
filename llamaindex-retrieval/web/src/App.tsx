import {
  ApiOutlined,
  DatabaseOutlined,
  FileTextOutlined,
  ImportOutlined,
  HistoryOutlined,
  SearchOutlined,
} from "@ant-design/icons";
import { Layout, Menu, Segmented, Space, Tag, Typography } from "antd";
import { Navigate, Route, Routes, useLocation, useNavigate } from "react-router-dom";

import DashboardPage from "./pages/DashboardPage";
import FilesPage from "./pages/FilesPage";
import ImportsPage from "./pages/ImportsPage";
import KnowledgeBasesPage from "./pages/KnowledgeBasesPage";
import SearchPage from "./pages/SearchPage";
import SearchHistoryPage from "./pages/SearchHistoryPage";
import { useLanguage } from "./i18n";

const { Header, Sider, Content } = Layout;

export default function App() {
  const location = useLocation();
  const navigate = useNavigate();
  const { language, setLanguage, tr } = useLanguage();
  const menuItems = [
    { key: "/dashboard", icon: <ApiOutlined />, label: tr("运行概览", "Overview") },
    { key: "/knowledge-bases", icon: <DatabaseOutlined />, label: tr("知识库", "Knowledge Bases") },
    { key: "/files", icon: <FileTextOutlined />, label: tr("文档管理", "Documents") },
    { key: "/imports", icon: <ImportOutlined />, label: tr("导入任务", "Imports") },
    { key: "/search", icon: <SearchOutlined />, label: tr("检索测试", "Search Lab") },
    { key: "/search-history", icon: <HistoryOutlined />, label: tr("历史测试", "History") },
  ];

  return (
    <Layout className="app-shell">
      <Sider width={172} className="app-sider" theme="light">
        <div className="brand">
          <div className="brand-mark">R</div>
          <div>
            <Typography.Title level={5}>RWKVRAG</Typography.Title>
            <Typography.Text type="secondary">Knowledge Console</Typography.Text>
          </div>
        </div>
        <Menu
          mode="inline"
          selectedKeys={[location.pathname]}
          items={menuItems}
          onClick={({ key }) => navigate(key)}
          className="nav-menu"
        />
      </Sider>
      <Layout>
        <Header className="app-header">
          <div>
            <Typography.Text className="eyebrow">LLAMAINDEX RETRIEVAL</Typography.Text>
            <Typography.Title level={4}>{tr("知识库管理后台", "Knowledge Base Console")}</Typography.Title>
          </div>
          <Space>
            <Tag color="cyan">OpenSearch BM25</Tag>
            <Tag color="green">{tr("经典检索", "Lexical Retrieval")}</Tag>
            <Segmented
              size="small"
              value={language}
              options={[{ label: "中文", value: "zh" }, { label: "EN", value: "en" }]}
              onChange={(value) => setLanguage(value as "zh" | "en")}
              aria-label={tr("切换语言", "Switch language")}
            />
          </Space>
        </Header>
        <Content className="app-content">
          <Routes>
            <Route path="/dashboard" element={<DashboardPage />} />
            <Route path="/knowledge-bases" element={<KnowledgeBasesPage />} />
            <Route path="/files" element={<FilesPage />} />
            <Route path="/imports" element={<ImportsPage />} />
            <Route path="/search" element={<SearchPage />} />
            <Route path="/search-history" element={<SearchHistoryPage />} />
            <Route path="*" element={<Navigate to="/dashboard" replace />} />
          </Routes>
        </Content>
      </Layout>
    </Layout>
  );
}
