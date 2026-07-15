package rag

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

type fixedEmbedder struct {
	vector []float32
}

func (e fixedEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	vectors := make([][]float32, len(texts))
	for i := range vectors {
		vectors[i] = append([]float32(nil), e.vector...)
	}
	return vectors, nil
}

func TestQdrantStoreCreatesUpsertsAndSearches(t *testing.T) {
	created := false
	upserted := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/collections/test":
			if !created {
				http.Error(w, `{"status":"not found"}`, http.StatusNotFound)
				return
			}
			writeTestJSON(w, map[string]any{
				"result": map[string]any{
					"config": map[string]any{
						"params": map[string]any{
							"vectors": map[string]any{
								"dense": map[string]any{"size": 3, "distance": "Cosine"},
							},
							"sparse_vectors": map[string]any{"text": map[string]any{}},
						},
					},
				},
			})
		case r.Method == http.MethodPut && r.URL.Path == "/collections/test":
			created = true
			writeTestJSON(w, map[string]any{"result": true, "status": "ok"})
		case r.Method == http.MethodPut && r.URL.Path == "/collections/test/points":
			var body struct {
				Points []json.RawMessage `json:"points"`
			}
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Fatal(err)
			}
			upserted += len(body.Points)
			writeTestJSON(w, map[string]any{"result": map[string]any{"status": "completed"}, "status": "ok"})
		case r.Method == http.MethodPut && r.URL.Path == "/collections/test/index":
			writeTestJSON(w, map[string]any{"result": map[string]any{"status": "completed"}, "status": "ok"})
		case r.Method == http.MethodPost && r.URL.Path == "/collections/test/points/query":
			writeTestJSON(w, map[string]any{
				"result": map[string]any{
					"points": []any{
						map[string]any{
							"score": 0.91,
							"payload": map[string]any{
								"id": "chunk-1", "document_id": "doc-1", "source": "test",
								"title": "中国首都", "content": "北京是中国的首都。", "chunk_index": 0,
							},
						},
					},
				},
			})
		default:
			http.Error(w, "unexpected request", http.StatusNotFound)
		}
	}))
	defer server.Close()

	store, err := NewQdrantStore(QdrantConfig{
		URL: server.URL, Collection: "test", Timeout: time.Second, SearchEF: 64,
	})
	if err != nil {
		t.Fatal(err)
	}
	chunk := Chunk{
		ID: "chunk-1", DocumentID: "doc-1", Source: "test", Title: "中国首都",
		Content: "北京是中国的首都。", Vector: []float32{1, 0, 0},
	}
	if err := store.Upsert(context.Background(), []Chunk{chunk}); err != nil {
		t.Fatal(err)
	}
	if !created || upserted != 1 {
		t.Fatalf("created=%v upserted=%d", created, upserted)
	}

	results, err := store.Search(context.Background(), fixedEmbedder{vector: []float32{1, 0, 0}}, "中国首都", 1, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 || results[0].Chunk.Title != "中国首都" || results[0].Score <= 0 {
		t.Fatalf("unexpected results: %+v", results)
	}
}

func writeTestJSON(w http.ResponseWriter, value any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(value)
}
