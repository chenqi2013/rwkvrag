import React from "react";
import ReactDOM from "react-dom/client";
import { ConfigProvider, theme } from "antd";
import zhCN from "antd/locale/zh_CN";
import enUS from "antd/locale/en_US";
import { HashRouter } from "react-router-dom";

import App from "./App";
import { LanguageProvider, useLanguage } from "./i18n";
import "./styles.css";

function Root() {
  const { language } = useLanguage();
  return (
    <ConfigProvider
      locale={language === "zh" ? zhCN : enUS}
      theme={{
        algorithm: theme.defaultAlgorithm,
        token: {
          colorPrimary: "#1677ff",
          colorInfo: "#1677ff",
          colorSuccess: "#16a085",
          colorBgBase: "#f5f7fa",
          colorTextBase: "#1f2937",
          borderRadius: 4,
          fontSize: 11,
          controlHeight: 24,
          controlHeightLG: 28,
          controlHeightSM: 20,
          padding: 8,
          paddingSM: 6,
          paddingXS: 4,
          margin: 8,
          marginSM: 6,
          marginXS: 4,
          fontFamily: 'Inter, "PingFang SC", "Microsoft YaHei", sans-serif',
        },
        components: {
          Button: {
            contentFontSize: 11,
            contentFontSizeSM: 10,
            onlyIconSize: 11,
            iconGap: 3,
            paddingInline: 7,
          },
          Card: {
            bodyPadding: 9,
            bodyPaddingSM: 7,
            headerHeight: 34,
            headerFontSize: 12,
          },
          Menu: {
            itemHeight: 30,
            itemMarginBlock: 1,
            iconSize: 12,
          },
          Table: {
            cellFontSize: 11,
            cellPaddingBlock: 4,
            cellPaddingInline: 6,
            headerBg: "#f6f8fb",
          },
          Form: {
            itemMarginBottom: 8,
            labelFontSize: 11,
          },
          Modal: {
            titleFontSize: 13,
            headerBg: "#ffffff",
            contentBg: "#ffffff",
          },
        },
      }}
    >
      <HashRouter>
        <App />
      </HashRouter>
    </ConfigProvider>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <LanguageProvider>
      <Root />
    </LanguageProvider>
  </React.StrictMode>,
);
