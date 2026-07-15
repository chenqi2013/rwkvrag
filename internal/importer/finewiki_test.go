package importer

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/parquet-go/parquet-go"

	"rwkvrag/internal/rag"
)

type testFineWikiRow struct {
	ID       string `parquet:"id"`
	Title    string `parquet:"title"`
	Text     string `parquet:"text"`
	Language string `parquet:"language"`
	URL      string `parquet:"url"`
}

func TestImportFineWiki(t *testing.T) {
	dir := t.TempDir()
	parquetPath := filepath.Join(dir, "sample.parquet")
	err := parquet.WriteFile(parquetPath, []testFineWikiRow{
		{
			ID:       "1",
			Title:    "人工智能",
			Text:     "人工智能是计算机科学的一个领域，研究智能系统。",
			Language: "zh",
			URL:      "https://zh.wikipedia.org/wiki/人工智能",
		},
		{
			ID:       "2",
			Title:    "English",
			Text:     "This row should be filtered.",
			Language: "en",
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	store, err := rag.NewStore(filepath.Join(dir, "index.jsonl"), rag.DefaultChunker())
	if err != nil {
		t.Fatal(err)
	}
	embedder := rag.HashEmbedder{Dimensions: 128}
	stats, err := ImportFineWiki(context.Background(), FineWikiOptions{
		Path:     parquetPath,
		Language: "zh",
		Limit:    1,
	}, store, embedder)
	if err != nil {
		t.Fatal(err)
	}
	if stats.Documents != 1 || stats.Chunks != 1 {
		t.Fatalf("unexpected stats: %+v", stats)
	}

	results, err := store.Search(context.Background(), embedder, "人工智能", 1, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 || results[0].Chunk.Title != "人工智能" {
		t.Fatalf("unexpected result: %+v", results)
	}
}
