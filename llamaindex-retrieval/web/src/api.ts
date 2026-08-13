import type {
  AdminHealth,
  ChunkItem,
  FileItem,
  FineWikiPathPage,
  JobItem,
  KnowledgeBase,
  AskResponse,
  SearchTestDetail,
  SearchTestPage,
  SearchTestRun,
  SearchResponse,
} from "./types";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = (await response.json()) as { detail?: string };
      detail = body.detail || detail;
    } catch {
      // Keep the HTTP status when the response is not JSON.
    }
    throw new Error(detail);
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

const jsonHeaders = { "Content-Type": "application/json" };

export const api = {
  health: () => request<AdminHealth>("/v1/admin/health"),
  knowledgeBases: () => request<KnowledgeBase[]>("/v1/admin/knowledge-bases"),
  createKnowledgeBase: (values: { name: string; description: string }) =>
    request<KnowledgeBase>("/v1/admin/knowledge-bases", {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify(values),
    }),
  updateKnowledgeBase: (id: string, values: { name?: string; description?: string }) =>
    request<KnowledgeBase>(`/v1/admin/knowledge-bases/${id}`, {
      method: "PATCH",
      headers: jsonHeaders,
      body: JSON.stringify(values),
    }),
  deleteKnowledgeBase: (id: string) =>
    request<void>(`/v1/admin/knowledge-bases/${id}`, { method: "DELETE" }),
  files: (knowledgeBaseId?: string) =>
    request<FileItem[]>(
      `/v1/admin/files${knowledgeBaseId ? `?knowledge_base_id=${encodeURIComponent(knowledgeBaseId)}` : ""}`,
    ),
  uploadFile: (file: File, knowledgeBaseId: string) => {
    const form = new FormData();
    form.append("file", file);
    form.append("knowledge_base_id", knowledgeBaseId);
    return request<{ file_id: string; job_id: string; status: string }>("/v1/admin/files", {
      method: "POST",
      body: form,
    });
  },
  deleteFile: (id: string) => request<void>(`/v1/admin/files/${id}`, { method: "DELETE" }),
  reindexFile: (id: string) =>
    request<{ job_id: string; status: string }>(`/v1/admin/files/${id}/reindex`, {
      method: "POST",
    }),
  fileChunks: (id: string) =>
    request<{ items: ChunkItem[]; next_offset?: string }>(`/v1/admin/files/${id}/chunks`),
  jobs: () => request<JobItem[]>("/v1/admin/jobs"),
  importFineWiki: (values: object) =>
    request<{ job_id: string; status: string }>("/v1/admin/imports/finewiki", {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify(values),
    }),
  fineWikiPaths: (path?: string) =>
    request<FineWikiPathPage>(
      `/v1/admin/imports/finewiki/paths${path ? `?path=${encodeURIComponent(path)}` : ""}`,
    ),
  search: (values: object) =>
    request<SearchResponse>("/v1/search", {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify(values),
    }),
  ask: (values: object) =>
    request<AskResponse>("/v1/ask", {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify(values),
    }),
  searchHistory: (page = 1, pageSize = 20, answerStatus?: "answered" | "refused") =>
    request<SearchTestPage>(
      `/v1/admin/search-history?page=${page}&page_size=${pageSize}${answerStatus ? `&answer_status=${answerStatus}` : ""}`,
    ),
  searchHistoryDetail: (id: string) =>
    request<SearchTestDetail>(`/v1/admin/search-history/${id}`),
  rerunSearchHistory: (id: string) =>
    request<SearchTestRun>(`/v1/admin/search-history/${id}/rerun`, { method: "POST" }),
};
