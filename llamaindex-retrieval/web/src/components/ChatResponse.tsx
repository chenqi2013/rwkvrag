import { Alert, Typography } from "antd";
import { memo } from "react";
import type { AskResponse } from "../types";
import { answerPresentation, evidenceWarnings } from "../answerPresentation";
import { formatAnswer } from "../answerFormat";
import { useLanguage } from "../i18n";
import CitedAnswer from "./CitedAnswer";

export default memo(function ChatResponse({ response }: { response: AskResponse }) {
  const { tr } = useLanguage();
  const view = answerPresentation(response);
  const formatted = formatAnswer(view.answerText, response);
  const warnings = evidenceWarnings(response);
  return <div className="chat-response">
    <div className={`chat-answer-status ${formatted.blocked || !view.spanValid ? "has-error" : ""}`}>
      {formatted.blocked ? tr("格式检查未通过；原始输出保留在下方", "Format check failed; raw output is retained below") : tr(...view.label)}
    </div>
    <CitedAnswer text={view.answerText} response={response} compact />
    {warnings.length > 0 && <details className="chat-checks">
      <summary>{tr("执行与来源提示", "Execution and source notices")} ({warnings.length})</summary>
      {warnings.map(w => <Alert key={w[1]} type="warning" title={tr(...w)} showIcon />)}
    </details>}
    <details className="answer-trace">
      <summary>{tr("原始模型输出与运行记录", "Raw model output and trace")}</summary>
      <Typography.Paragraph className="result-snippet raw-output" copyable={{ text: response.answer }}>
        {response.answer || tr("API答案字段为空；未用中间结果替换。", "API answer is empty; no intermediate result was substituted.")}
      </Typography.Paragraph>
      <details><summary>{tr("完整响应（含来源与失败记录）", "Full response, including sources and failures")}</summary>
        <pre className="trace-json" tabIndex={0}>{JSON.stringify(response, null, 2)}</pre>
      </details>
    </details>
  </div>;
});
