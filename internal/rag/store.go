package rag

import (
	"bufio"
	"container/heap"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
)

type Store struct {
	Path    string
	Chunker Chunker
	mu      sync.Mutex
	qdrant  *QdrantStore
}

type StoreStats struct {
	Path    string         `json:"path"`
	Bytes   int64          `json:"bytes"`
	Chunks  int            `json:"chunks"`
	Sources map[string]int `json:"sources"`
}

func NewStore(path string, chunker Chunker) (*Store, error) {
	if path == "" {
		path = filepath.Join("data", "index.jsonl")
	}
	path = filepath.Clean(path)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	if strings.EqualFold(strings.TrimSpace(os.Getenv("RWKVRAG_STORE_PROVIDER")), "qdrant") {
		qdrant, err := NewQdrantStore(QdrantConfigFromEnv())
		if err != nil {
			return nil, err
		}
		return &Store{
			Path:    qdrant.Endpoint(),
			Chunker: chunker.Normalize(),
			qdrant:  qdrant,
		}, nil
	}
	return &Store{Path: path, Chunker: chunker.Normalize()}, nil
}

func (s *Store) Close() error {
	return nil
}

func (s *Store) AddDocuments(ctx context.Context, docs []Document, embedder Embedder, batchSize int) (ImportStats, error) {
	var stats ImportStats
	if len(docs) == 0 {
		return stats, nil
	}
	if embedder == nil {
		return stats, errors.New("embedder is nil")
	}
	if batchSize <= 0 {
		batchSize = 32
	}

	var chunks []Chunk
	for _, doc := range docs {
		if err := ctx.Err(); err != nil {
			return stats, err
		}
		doc.Content = CleanText(doc.Content)
		if doc.ID == "" {
			doc.ID = HashID(doc.Source, doc.Title, doc.URI, doc.Content)
		}
		docChunks := s.Chunker.Split(doc)
		if len(docChunks) == 0 {
			stats.Skipped++
			continue
		}
		stats.Documents++
		chunks = append(chunks, docChunks...)
	}
	if len(chunks) == 0 {
		return stats, nil
	}

	for start := 0; start < len(chunks); start += batchSize {
		end := start + batchSize
		if end > len(chunks) {
			end = len(chunks)
		}
		texts := make([]string, end-start)
		for i := start; i < end; i++ {
			texts[i-start] = chunks[i].Title + "\n\n" + chunks[i].Content
		}
		vectors, err := embedder.Embed(ctx, texts)
		if err != nil {
			return stats, err
		}
		if len(vectors) != len(texts) {
			return stats, fmt.Errorf("embedder returned %d vectors for %d texts", len(vectors), len(texts))
		}
		for i := range vectors {
			chunks[start+i].Vector = vectors[i]
		}
	}

	var err error
	if s.qdrant != nil {
		err = s.qdrant.Upsert(ctx, chunks)
	} else {
		err = s.appendChunks(chunks)
	}
	if err != nil {
		return stats, err
	}
	stats.Chunks += len(chunks)
	return stats, nil
}

func (s *Store) Search(ctx context.Context, embedder Embedder, query string, topK int, minScore float64) ([]SearchResult, error) {
	query = CleanText(query)
	if query == "" {
		return nil, errors.New("query is empty")
	}
	if topK <= 0 {
		topK = 5
	}
	if s.qdrant != nil {
		return s.qdrant.Search(ctx, embedder, query, topK, minScore)
	}
	vectors, err := embedder.Embed(ctx, []string{query})
	if err != nil {
		return nil, err
	}
	if len(vectors) != 1 {
		return nil, fmt.Errorf("embedder returned %d query vectors", len(vectors))
	}
	queryVec := vectors[0]
	querySet := TokenSet(query)

	f, err := os.Open(s.Path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	defer f.Close()

	h := &resultHeap{}
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024*1024), 32*1024*1024)
	for scanner.Scan() {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		var chunk Chunk
		if err := json.Unmarshal(scanner.Bytes(), &chunk); err != nil {
			continue
		}
		vectorScore := Cosine(queryVec, chunk.Vector)
		lexicalScore := LexicalScore(querySet, chunk.Title+"\n"+chunk.Content)
		score := 0.85*vectorScore + 0.15*lexicalScore
		if score < minScore {
			continue
		}
		result := SearchResult{Chunk: chunk, Score: score}
		if h.Len() < topK {
			heap.Push(h, result)
			continue
		}
		if (*h)[0].Score < score {
			heap.Pop(h)
			heap.Push(h, result)
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}

	results := make([]SearchResult, h.Len())
	for i := len(results) - 1; i >= 0; i-- {
		results[i] = heap.Pop(h).(SearchResult)
	}
	sort.SliceStable(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})
	return results, nil
}

func (s *Store) Stats() (StoreStats, error) {
	if s.qdrant != nil {
		return s.qdrant.Stats(context.Background())
	}
	stats := StoreStats{
		Path:    s.Path,
		Sources: map[string]int{},
	}
	info, err := os.Stat(s.Path)
	if errors.Is(err, os.ErrNotExist) {
		return stats, nil
	}
	if err != nil {
		return stats, err
	}
	stats.Bytes = info.Size()
	f, err := os.Open(s.Path)
	if err != nil {
		return stats, err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024*1024), 32*1024*1024)
	for scanner.Scan() {
		var chunk Chunk
		if err := json.Unmarshal(scanner.Bytes(), &chunk); err != nil {
			continue
		}
		stats.Chunks++
		stats.Sources[chunk.Source]++
	}
	if err := scanner.Err(); err != nil {
		return stats, err
	}
	return stats, nil
}

func (s *Store) appendChunks(chunks []Chunk) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if err := os.MkdirAll(filepath.Dir(s.Path), 0o755); err != nil {
		return err
	}
	f, err := os.OpenFile(s.Path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	writer := bufio.NewWriterSize(f, 1024*1024)
	enc := json.NewEncoder(writer)
	for _, chunk := range chunks {
		if err := enc.Encode(chunk); err != nil {
			return err
		}
	}
	if err := writer.Flush(); err != nil {
		return err
	}
	return f.Sync()
}

func CopyIndex(dst io.Writer, path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = io.Copy(dst, f)
	return err
}

type resultHeap []SearchResult

func (h resultHeap) Len() int           { return len(h) }
func (h resultHeap) Less(i, j int) bool { return h[i].Score < h[j].Score }
func (h resultHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *resultHeap) Push(x any) {
	*h = append(*h, x.(SearchResult))
}

func (h *resultHeap) Pop() any {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[:n-1]
	return item
}
