package rag

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
)

func TestStoreSearchWithHashEmbedder(t *testing.T) {
	dir := t.TempDir()
	store, err := NewStore(filepath.Join(dir, "index.jsonl"), Chunker{Size: 80, Overlap: 10})
	if err != nil {
		t.Fatal(err)
	}
	embedder := HashEmbedder{Dimensions: 128}
	_, err = store.AddDocuments(context.Background(), []Document{
		{
			ID:      "doc-1",
			Source:  "test",
			Title:   "RWKV",
			Content: "RWKV 是一种循环神经网络架构，适合长上下文语言建模。",
		},
		{
			ID:      "doc-2",
			Source:  "test",
			Title:   "无关内容",
			Content: "今天的午餐包括米饭和蔬菜。",
		},
	}, embedder, 2)
	if err != nil {
		t.Fatal(err)
	}

	results, err := store.Search(context.Background(), embedder, "RWKV 架构", 1, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 {
		t.Fatalf("expected one result, got %d", len(results))
	}
	if !strings.Contains(results[0].Chunk.Content, "循环神经网络") {
		t.Fatalf("unexpected result: %+v", results[0])
	}
}

func TestChunkerSplitsLongText(t *testing.T) {
	chunker := Chunker{Size: 10, Overlap: 2}
	chunks := chunker.Split(Document{
		ID:      "doc",
		Source:  "test",
		Title:   "long",
		Content: strings.Repeat("知识库", 12),
	})
	if len(chunks) < 2 {
		t.Fatalf("expected multiple chunks, got %d", len(chunks))
	}
}
