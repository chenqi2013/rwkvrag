import {
  ApiOutlined,
  DatabaseOutlined,
  FileTextOutlined,
  ImportOutlined,
  HistoryOutlined,
  SearchOutlined,
} from "@ant-design/icons";
import { Layout, Menu, Space, Tag, Typography } from "antd";
import { Navigate, Route, Routes, useLocation, useNavigate } from "react-router-dom";

import DashboardPage from "./pages/DashboardPage";
import FilesPage from "./pages/FilesPage";
import ImportsPage from "./pages/ImportsPage";
import KnowledgeBasesPage from "./pages/KnowledgeBasesPage";
import SearchPage from "./pages/SearchPage";
import SearchHistoryPage from "./pages/SearchHistoryPage";

const { Header, Sider, Content } = Layout;

const menuItems = [
  { key: "/dashboard", icon: <ApiOutlined />, label: "运行概览" },
  { key: "/knowledge-bases", icon: <DatabaseOutlined />, label: "知识库" },
  { key: "/files", icon: <FileTextOutlined />, label: "文档管理" },
  { key: "/imports", icon: <ImportOutlined />, label: "导入任务" },
  { key: "/search", icon: <SearchOutlined />, label: "检索测试" },
  { key: "/search-history", icon: <HistoryOutlined />, label: "历史测试" },
];

export default function App() {
  const location = useLocation();
  const navigate = useNavigate();

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
            <Typography.Title level={4}>知识库管理后台</Typography.Title>
          </div>
          <Space>
            <Tag color="cyan">OpenSearch BM25</Tag>
            <Tag color="green">经典检索</Tag>
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
