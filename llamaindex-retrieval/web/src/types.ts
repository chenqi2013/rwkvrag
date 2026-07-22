export interface KnowledgeBase {
  id: string;
  name: string;
  description: string;
  created_at: string;
  updated_at: string;
  file_count: number;
}

export interface FileItem {
  id: string;
  knowledge_base_id: string;
  filename: string;
  content_type: string;
  extension: string;
  size: number;
  sha256: string;
  source: string;
  status: "pending" | "processing" | "ready" | "failed" | "deleting";
  node_count: number;
  error?: string;
  last_job_id?: string;
  created_at: string;
  updated_at: string;
}

export interface JobItem {
  id: string;
  kind: "file_ingest" | "file_reindex" | "finewiki_import";
  status: "pending" | "running" | "completed" | "failed";
  progress: number;
  stage: string;
  message: string;
  documents_processed: number;
  nodes_processed: number;
  error?: string;
  payload: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface FineWikiPathEntry {
  name: string;
  path: string;
  type: "directory" | "parquet";
  size?: number;
}

export interface FineWikiPathPage {
  current: string;
  parent?: string;
  roots: string[];
  entries: FineWikiPathEntry[];
}

export interface ChunkItem {
  id: string;
  document_id: string;
  text: string;
  metadata: Record<string, unknown>;
}

export interface SearchResult {
  id: string;
  document_id: string;
  source: string;
  title: string;
  uri?: string;
  score: number;
  snippet: string;
  metadata: Record<string, unknown>;
}

export interface SearchResponse {
  results: SearchResult[];
  retrieval: Record<string, unknown>;
}

export interface AdminHealth {
  status: "ok" | "degraded";
  mongodb: Record<string, unknown>;
  lexical: Record<string, unknown>;
}
