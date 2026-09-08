import type { AskResponse } from "./types";

type Label = [string, string];

export function answerPresentation(response?: Pick<AskResponse, "answer" | "generation">) {
  const generation = response?.generation ?? {};
  const rawAnswer = response?.answer ?? "";
  const isNative = generation.pipeline === "rwkv";
  if (!isNative) {
    const writerAttempted = generation.answer_strategy === "single_writer_call"
      || generation.answer_strategy === "generation_failed";
    const writerCalled = generation.answer_strategy === "single_writer_call"
      || generation.writer_trace != null;
    const verified = writerCalled && generation.grounding_valid === true
      && generation.answer_support_passed === true;
    const label: Label = verified ? ["证据校验通过", "Evidence verified"]
      : generation.answer_strategy === "generation_failed" ? ["生成失败", "Generation failed"]
        : writerCalled ? ["已生成，证据校验未通过", "Generated; evidence check failed"]
          : ["证据不足，未生成", "Insufficient evidence; not generated"];
    return { isNative, rawAnswer, answerText: rawAnswer, spanValid: true,
      writerAttempted, writerCalled, label,
      color: verified ? "green" : writerAttempted ? "blue" : "orange" };
  }

  const calls = Array.isArray(generation.model_calls) ? generation.model_calls : [];
  const writers = calls.filter((call): call is Record<string, unknown> =>
    typeof call === "object" && call !== null && call.stage === "writer");
  const writer = writers[writers.length - 1];
  const writerStatus = generation.writer_status ?? writer?.status;
  const writerAttempted = writers.length > 0;
  const writerCalled = writer?.completion_attempted === true
    || writerStatus === "completed" || writerStatus === "length";
  const status = generation.status;

  // Python offsets count Unicode code points; String.slice counts UTF-16 units.
  // Never guess a boundary or fall back to showing thinking as the final answer.
  const characters = Array.from(rawAnswer);
  const span = generation.answer_span;
  const spanValid = Array.isArray(span) && span.length === 2
    && Number.isInteger(span[0]) && Number.isInteger(span[1])
    && span[0] >= 0 && span[0] <= span[1] && span[1] <= characters.length;
  const answerText = spanValid ? characters.slice(span[0], span[1]).join("") : "";
  let color = "orange";
  let label: Label;
  if (status === "resolver_partial_failure") {
    label = writerStatus === "completed"
      ? ["已生成，部分证据读取失败", "Generated; some evidence reads failed"]
      : ["部分证据读取失败，生成未完成", "Some evidence reads failed; generation incomplete"];
  } else if (status === "completed") {
    color = "blue";
    label = !spanValid ? ["流程报告完成，正文范围无效", "Completed status; invalid answer span"]
      : !answerText.trim() ? ["生成完成，正文为空", "Generation completed; answer is empty"]
        : !writerAttempted && !writerCalled
          ? ["流程报告完成，缺少生成记录", "Completed status; generation record missing"]
          : ["生成完成，尚未做语义核验", "Generation completed; semantic support not verified"];
  } else if (status === "length" || writerStatus === "length") {
    label = ["达到输出上限，生成未完成", "Output limit reached; generation incomplete"];
  } else if (status === "budget_exceeded" || writerStatus === "budget_exceeded") {
    label = ["输入超出模型预算，未生成", "Input exceeds model budget; not generated"];
  } else if (status === "planner_failed") {
    color = "red";
    label = ["规划失败，未进入生成", "Planning failed; generation not reached"];
  } else if (status === "retrieval_failed") {
    color = "red";
    label = ["检索失败，未进入生成", "Retrieval failed; generation not reached"];
  } else if (status === "invalid_materials") {
    color = "red";
    label = ["材料身份校验失败，未生成", "Material identity check failed; not generated"];
  } else {
    color = "red";
    label = writerCalled ? ["生成未完成，请查看运行记录", "Generation incomplete; see trace"]
      : writerAttempted ? ["生成请求未完成，请查看运行记录", "Generation request incomplete; see trace"]
        : ["未提供生成完成记录", "No completed generation record"];
  }
  return { isNative, rawAnswer, answerText, spanValid, writerAttempted, writerCalled, label, color };
}
