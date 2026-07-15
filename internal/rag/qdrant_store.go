package rag

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"hash/fnv"
	"io"
	"math"
	"net/http"
	"net/url"
	"os"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode/utf8"
)

type QdrantConfig struct {
	URL        string
	APIKey     string
	Collection string
	Timeout    time.Duration
	SearchEF   int
}

type QdrantStore struct {
	config       QdrantConfig
	client       *http.Client
	mu           sync.Mutex
	dimension    int
	indexesReady bool
}

type qdrantPayload struct {
	ID         string            `json:"id"`
	DocumentID string            `json:"document_id"`
	Source     string            `json:"source"`
	Title      string            `json:"title"`
	URI        string            `json:"uri,omitempty"`
	Content    string            `json:"content"`
	ChunkIndex int               `json:"chunk_index"`
	Metadata   map[string]string `json:"metadata,omitempty"`
}

type qdrantSparseVector struct {
	Indices []uint32  `json:"indices"`
	Values  []float32 `json:"values"`
}

func QdrantConfigFromEnv() QdrantConfig {
	return QdrantConfig{
		URL:        qdrantEnvString("RWKVRAG_QDRANT_URL", "http://127.0.0.1:6333"),
		APIKey:     strings.TrimSpace(os.Getenv("RWKVRAG_QDRANT_API_KEY")),
		Collection: qdrantEnvString("RWKVRAG_QDRANT_COLLECTION", "rwkvrag"),
		Timeout:    time.Duration(envInt("RWKVRAG_QDRANT_TIMEOUT_SECONDS", 120)) * time.Second,
		SearchEF:   envInt("RWKVRAG_QDRANT_SEARCH_EF", 128),
	}
}

func NewQdrantStore(config QdrantConfig) (*QdrantStore, error) {
	config.URL = strings.TrimRight(strings.TrimSpace(config.URL), "/")
	config.Collection = strings.TrimSpace(config.Collection)
	if config.URL == "" {
		return nil, errors.New("RWKVRAG_QDRANT_URL is required")
	}
	parsed, err := url.Parse(config.URL)
	if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" {
		return nil, fmt.Errorf("invalid Qdrant URL %q", config.URL)
	}
	if config.Collection == "" {
		return nil, errors.New("RWKVRAG_QDRANT_COLLECTION is required")
	}
	if config.Timeout <= 0 {
		config.Timeout = 120 * time.Second
	}
	if config.SearchEF <= 0 {
		config.SearchEF = 128
	}
	return &QdrantStore{
		config: config,
		client: &http.Client{Timeout: config.Timeout},
	}, nil
}

func (q *QdrantStore) Endpoint() string {
	return q.config.URL + "/collections/" + url.PathEscape(q.config.Collection)
}

func (q *QdrantStore) Upsert(ctx context.Context, chunks []Chunk) error {
	if len(chunks) == 0 {
		return nil
	}
	dimension := len(chunks[0].Vector)
	if dimension == 0 {
		return errors.New("cannot store empty vectors in Qdrant")
	}
	for _, chunk := range chunks[1:] {
		if len(chunk.Vector) != dimension {
			return fmt.Errorf("vector dimension mismatch: got %d want %d", len(chunk.Vector), dimension)
		}
	}
	if err := q.ensureCollection(ctx, dimension); err != nil {
		return err
	}

	points := make([]struct {
		ID      string         `json:"id"`
		Vector  map[string]any `json:"vector"`
		Payload qdrantPayload  `json:"payload"`
	}, len(chunks))
	for i, chunk := range chunks {
		points[i].ID = qdrantPointID(chunk.ID)
		points[i].Vector = map[string]any{
			"dense": chunk.Vector,
			"text":  qdrantSparse(strings.Repeat(chunk.Title+"\n", 3) + chunk.Content),
		}
		points[i].Payload = payloadFromChunk(chunk)
	}
	status, err := q.request(ctx, http.MethodPut, q.collectionPath("/points?wait=true"), map[string]any{"points": points}, nil)
	if err != nil {
		return err
	}
	if status < 200 || status >= 300 {
		return fmt.Errorf("Qdrant upsert returned HTTP %d", status)
	}
	return nil
}

func (q *QdrantStore) Search(ctx context.Context, embedder Embedder, query string, topK int, minScore float64) ([]SearchResult, error) {
	vectors, err := embedder.Embed(ctx, []string{query})
	if err != nil {
		return nil, err
	}
	if len(vectors) != 1 || len(vectors[0]) == 0 {
		return nil, fmt.Errorf("embedder returned %d query vectors", len(vectors))
	}

	candidates := topK * 20
	if candidates < 100 {
		candidates = 100
	}
	body := map[string]any{
		"prefetch": []any{
			map[string]any{
				"query": vectors[0],
				"using": "dense",
				"limit": candidates,
				"params": map[string]any{
					"hnsw_ef": q.config.SearchEF,
					"exact":   false,
				},
			},
			map[string]any{
				"query": qdrantSparse(query),
				"using": "text",
				"limit": candidates,
			},
		},
		"query":        map[string]string{"fusion": "rrf"},
		"limit":        candidates,
		"with_payload": true,
	}
	var response struct {
		Result struct {
			Points []struct {
				Score   float64       `json:"score"`
				Payload qdrantPayload `json:"payload"`
			} `json:"points"`
		} `json:"result"`
	}
	status, err := q.request(ctx, http.MethodPost, q.collectionPath("/points/query"), body, &response)
	if err != nil {
		return nil, err
	}
	if status == http.StatusNotFound {
		return nil, nil
	}
	if status < 200 || status >= 300 {
		return nil, fmt.Errorf("Qdrant search returned HTTP %d", status)
	}

	querySet := TokenSet(query)
	results := make([]SearchResult, 0, len(response.Result.Points))
	for _, result := range response.Result.Points {
		chunk := result.Payload.chunk()
		titleScore := LexicalScore(querySet, chunk.Title)
		contentScore := LexicalScore(querySet, chunk.Title+"\n"+chunk.Content)
		score := 0.7*result.Score + 0.2*titleScore + 0.1*contentScore
		if minScore > 0 && score < minScore {
			continue
		}
		results = append(results, SearchResult{Chunk: chunk, Score: score})
	}
	sort.SliceStable(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})
	if len(results) > topK {
		results = results[:topK]
	}
	return results, nil
}

func (q *QdrantStore) Stats(ctx context.Context) (StoreStats, error) {
	stats := StoreStats{Path: q.Endpoint(), Sources: map[string]int{}}
	var response struct {
		Result struct {
			PointsCount int `json:"points_count"`
		} `json:"result"`
	}
	status, err := q.request(ctx, http.MethodGet, q.collectionPath(""), nil, &response)
	if err != nil {
		return stats, err
	}
	if status == http.StatusNotFound {
		return stats, nil
	}
	if status < 200 || status >= 300 {
		return stats, fmt.Errorf("Qdrant collection info returned HTTP %d", status)
	}
	stats.Chunks = response.Result.PointsCount

	var facet struct {
		Result struct {
			Hits []struct {
				Value any `json:"value"`
				Count int `json:"count"`
			} `json:"hits"`
		} `json:"result"`
	}
	status, err = q.request(ctx, http.MethodPost, q.collectionPath("/facet"), map[string]any{
		"key":   "source",
		"limit": 100,
		"exact": true,
	}, &facet)
	if err == nil && status >= 200 && status < 300 {
		for _, hit := range facet.Result.Hits {
			stats.Sources[fmt.Sprint(hit.Value)] = hit.Count
		}
	}
	return stats, nil
}

func (q *QdrantStore) ensureCollection(ctx context.Context, dimension int) error {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.dimension != 0 {
		if q.dimension != dimension {
			return fmt.Errorf("Qdrant collection vector size is %d, embedder returned %d", q.dimension, dimension)
		}
		return q.ensurePayloadIndexes(ctx)
	}

	var info struct {
		Result struct {
			Config struct {
				Params struct {
					Vectors map[string]struct {
						Size int `json:"size"`
					} `json:"vectors"`
					SparseVectors map[string]json.RawMessage `json:"sparse_vectors"`
				} `json:"params"`
			} `json:"config"`
		} `json:"result"`
	}
	status, err := q.request(ctx, http.MethodGet, q.collectionPath(""), nil, &info)
	if err != nil {
		return err
	}
	if status == http.StatusNotFound {
		status, err = q.request(ctx, http.MethodPut, q.collectionPath(""), map[string]any{
			"vectors": map[string]any{
				"dense": map[string]any{
					"size":     dimension,
					"distance": "Cosine",
				},
			},
			"sparse_vectors": map[string]any{
				"text": map[string]any{
					"index":    map[string]any{"on_disk": true},
					"modifier": "idf",
				},
			},
			"hnsw_config": map[string]any{
				"m":            16,
				"ef_construct": 128,
			},
			"optimizers_config": map[string]any{
				"indexing_threshold": 20000,
			},
			"on_disk_payload": true,
		}, nil)
		if err != nil {
			return err
		}
		if status < 200 || status >= 300 {
			return fmt.Errorf("create Qdrant collection returned HTTP %d", status)
		}
		q.dimension = dimension
		return q.ensurePayloadIndexes(ctx)
	}
	if status < 200 || status >= 300 {
		return fmt.Errorf("Qdrant collection info returned HTTP %d", status)
	}
	dense, ok := info.Result.Config.Params.Vectors["dense"]
	if !ok || dense.Size == 0 {
		return errors.New("Qdrant collection does not contain the dense vector")
	}
	if _, ok := info.Result.Config.Params.SparseVectors["text"]; !ok {
		return errors.New("Qdrant collection does not contain the text sparse vector")
	}
	q.dimension = dense.Size
	if q.dimension != dimension {
		return fmt.Errorf("Qdrant collection vector size is %d, embedder returned %d", q.dimension, dimension)
	}
	return q.ensurePayloadIndexes(ctx)
}

func (q *QdrantStore) ensurePayloadIndexes(ctx context.Context) error {
	if q.indexesReady {
		return nil
	}
	status, err := q.request(ctx, http.MethodPut, q.collectionPath("/index?wait=true"), map[string]any{
		"field_name":   "source",
		"field_schema": "keyword",
	}, nil)
	if err != nil {
		return err
	}
	if status < 200 || status >= 300 {
		return fmt.Errorf("create Qdrant payload index returned HTTP %d", status)
	}
	q.indexesReady = true
	return nil
}

func (q *QdrantStore) request(ctx context.Context, method, path string, body, target any) (int, error) {
	var reader io.Reader
	if body != nil {
		payload, err := json.Marshal(body)
		if err != nil {
			return 0, err
		}
		reader = bytes.NewReader(payload)
	}
	req, err := http.NewRequestWithContext(ctx, method, q.config.URL+path, reader)
	if err != nil {
		return 0, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if q.config.APIKey != "" {
		req.Header.Set("api-key", q.config.APIKey)
	}
	resp, err := q.client.Do(req)
	if err != nil {
		return 0, fmt.Errorf("Qdrant request failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		message, _ := io.ReadAll(io.LimitReader(resp.Body, 64*1024))
		if resp.StatusCode == http.StatusNotFound {
			return resp.StatusCode, nil
		}
		return resp.StatusCode, fmt.Errorf("Qdrant %s %s failed: %s: %s", method, path, resp.Status, strings.TrimSpace(string(message)))
	}
	if target != nil {
		if err := json.NewDecoder(resp.Body).Decode(target); err != nil {
			return resp.StatusCode, err
		}
	}
	return resp.StatusCode, nil
}

func (q *QdrantStore) collectionPath(suffix string) string {
	return "/collections/" + url.PathEscape(q.config.Collection) + suffix
}

func payloadFromChunk(chunk Chunk) qdrantPayload {
	return qdrantPayload{
		ID:         chunk.ID,
		DocumentID: chunk.DocumentID,
		Source:     chunk.Source,
		Title:      chunk.Title,
		URI:        chunk.URI,
		Content:    chunk.Content,
		ChunkIndex: chunk.ChunkIndex,
		Metadata:   chunk.Metadata,
	}
}

func (p qdrantPayload) chunk() Chunk {
	return Chunk{
		ID:         p.ID,
		DocumentID: p.DocumentID,
		Source:     p.Source,
		Title:      p.Title,
		URI:        p.URI,
		Content:    p.Content,
		ChunkIndex: p.ChunkIndex,
		Metadata:   p.Metadata,
	}
}

func qdrantPointID(chunkID string) string {
	sum := sha256.Sum256([]byte(chunkID))
	encoded := hex.EncodeToString(sum[:16])
	return strings.Join([]string{
		encoded[0:8],
		encoded[8:12],
		encoded[12:16],
		encoded[16:20],
		encoded[20:32],
	}, "-")
}

func qdrantSparse(text string) qdrantSparseVector {
	tokens := Tokens(text)
	preferLong := false
	for _, token := range tokens {
		if utf8.RuneCountInString(token) > 1 {
			preferLong = true
			break
		}
	}
	counts := map[uint32]int{}
	for _, token := range tokens {
		if preferLong && utf8.RuneCountInString(token) == 1 {
			continue
		}
		hasher := fnv.New32a()
		_, _ = hasher.Write([]byte(token))
		counts[hasher.Sum32()]++
	}
	indices := make([]uint32, 0, len(counts))
	for index := range counts {
		indices = append(indices, index)
	}
	sort.Slice(indices, func(i, j int) bool { return indices[i] < indices[j] })
	values := make([]float32, len(indices))
	for i, index := range indices {
		count := counts[index]
		if count > 4 {
			count = 4
		}
		values[i] = float32(1 + math.Log(float64(count)))
	}
	return qdrantSparseVector{Indices: indices, Values: values}
}

func qdrantEnvString(key, fallback string) string {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return fallback
	}
	return value
}
