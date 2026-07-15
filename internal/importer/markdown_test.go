package importer

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"rwkvrag/internal/rag"
)

func TestImportMarkdown(t *testing.T) {
	dir := t.TempDir()
	docs := filepath.Join(dir, "docs")
	if err := os.MkdirAll(docs, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(docs, "handbook.md"), []byte("# 员工手册\n\n报销流程需要提交发票和审批单。"), 0o644); err != nil {
		t.Fatal(err)
	}

	store, err := rag.NewStore(filepath.Join(dir, "index.jsonl"), rag.DefaultChunker())
	if err != nil {
		t.Fatal(err)
	}
	embedder := rag.HashEmbedder{Dimensions: 128}
	stats, err := ImportMarkdown(context.Background(), MarkdownOptions{Path: docs, Source: "company"}, store, embedder)
	if err != nil {
		t.Fatal(err)
	}
	if stats.Documents != 1 || stats.Chunks != 1 || stats.Files != 1 {
		t.Fatalf("unexpected stats: %+v", stats)
	}

	results, err := store.Search(context.Background(), embedder, "报销需要什么材料", 1, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 || results[0].Chunk.Title != "员工手册" {
		t.Fatalf("unexpected result: %+v", results)
	}
}
