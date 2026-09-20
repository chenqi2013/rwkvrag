import type { AskResponse, SearchResult } from "./types.ts";

export function citationParts(text: string) {
  const pattern = /\[(?:(?:资料|Source)\s*)?(\d+)\](?!\()/g;
  const parts: Array<{ text: string; label?: number }> = [];
  let cursor = 0;
  for (const match of text.matchAll(pattern)) {
    if (match.index > cursor) parts.push({ text: text.slice(cursor, match.index) });
    parts.push({ text: match[0], label: Number(match[1]) });
    cursor = match.index + match[0].length;
  }
  if (cursor < text.length) parts.push({ text: text.slice(cursor) });
  return parts;
}

export function citedSource(response: AskResponse, label: number): SearchResult | undefined {
  if (!Number.isSafeInteger(label) || label < 1) return undefined;
  const map = response.generation.citation_map;
  if (map !== undefined && map !== null) {
    if (typeof map !== "object" || Array.isArray(map)) return undefined;
    const id = (map as Record<string, unknown>)[String(label)];
    const matches = response.sources.filter(source => source.id === id);
    return matches.length === 1 ? matches[0] : undefined;
  }
  return response.sources[label - 1];
}

export function sourceLabels(response: AskResponse, source: SearchResult, index: number): number[] {
  const map = response.generation.citation_map;
  if (map === undefined || map === null) return [index + 1];
  if (typeof map !== "object" || Array.isArray(map)) return [];
  return Object.keys(map).map(Number).filter(n => citedSource(response, n)?.id === source.id).sort((a, b) => a - b);
}

export function externalSourceUrl(uri?: string): string | undefined {
  try {
    const url = new URL(uri || "");
    if ((url.protocol === "http:" || url.protocol === "https:") && !url.username && !url.password) return url.href;
  } catch { /* Local paths and invalid URLs are displayed as text, never navigated. */ }
  return undefined;
}

export function savedContext(response: AskResponse, source: SearchResult): string | undefined {
  const parent = source.metadata?.parent_source_id || source.id;
  const searches = response.retrieval.web_search;
  if (Array.isArray(searches)) {
    for (const search of searches) {
      if (!search || !Array.isArray(search.snapshots)) continue;
      const snapshot = search.snapshots.find((s: Record<string, unknown>) => s.id === parent
        && s.sha256 === source.metadata?.snapshot_sha256 && typeof s.text === "string");
      if (snapshot) return snapshot.text;
    }
  }
  const candidates = response.retrieval.candidates;
  if (Array.isArray(candidates)) {
    const candidate = candidates.find(c => c?.id === parent && typeof c.snippet === "string");
    if (candidate) return candidate.snippet;
  }
  return undefined;
}
