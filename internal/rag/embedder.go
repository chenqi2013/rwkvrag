package rag

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"hash/fnv"
	"math"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
)

type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
}

type EmbeddingConfig struct {
	Provider   string
	BaseURL    string
	APIKey     string
	Model      string
	Dimensions int
	BatchSize  int
	Timeout    time.Duration
}

func EmbeddingConfigFromEnv() EmbeddingConfig {
	cfg := EmbeddingConfig{
		Provider:   strings.ToLower(strings.TrimSpace(os.Getenv("RWKVRAG_EMBEDDING_PROVIDER"))),
		BaseURL:    strings.TrimSpace(os.Getenv("RWKVRAG_EMBEDDING_BASE_URL")),
		APIKey:     strings.TrimSpace(os.Getenv("RWKVRAG_EMBEDDING_API_KEY")),
		Model:      strings.TrimSpace(os.Getenv("RWKVRAG_EMBEDDING_MODEL")),
		Dimensions: envInt("RWKVRAG_HASH_DIM", 384),
		BatchSize:  envInt("RWKVRAG_EMBEDDING_BATCH_SIZE", 32),
		Timeout:    time.Duration(envInt("RWKVRAG_EMBEDDING_TIMEOUT_SECONDS", 60)) * time.Second,
	}
	if cfg.Provider == "" {
		if cfg.BaseURL != "" || cfg.APIKey != "" || cfg.Model != "" {
			cfg.Provider = "openai"
		} else {
			cfg.Provider = "hash"
		}
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.openai.com/v1"
	}
	return cfg
}

func NewEmbedder(cfg EmbeddingConfig) (Embedder, error) {
	switch strings.ToLower(cfg.Provider) {
	case "", "hash", "local":
		if cfg.Dimensions <= 0 {
			cfg.Dimensions = 384
		}
		return HashEmbedder{Dimensions: cfg.Dimensions}, nil
	case "openai", "openai-compatible":
		if cfg.Model == "" {
			return nil, errors.New("RWKVRAG_EMBEDDING_MODEL is required for openai embeddings")
		}
		if cfg.BatchSize <= 0 {
			cfg.BatchSize = 32
		}
		if cfg.Timeout <= 0 {
			cfg.Timeout = 60 * time.Second
		}
		return &OpenAIEmbedder{
			BaseURL:   strings.TrimRight(cfg.BaseURL, "/"),
			APIKey:    cfg.APIKey,
			Model:     cfg.Model,
			BatchSize: cfg.BatchSize,
			Client:    &http.Client{Timeout: cfg.Timeout},
		}, nil
	default:
		return nil, fmt.Errorf("unknown embedding provider %q", cfg.Provider)
	}
}

type HashEmbedder struct {
	Dimensions int
}

func (h HashEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	if h.Dimensions <= 0 {
		h.Dimensions = 384
	}
	out := make([][]float32, len(texts))
	for i, text := range texts {
		vec := make([]float32, h.Dimensions)
		for _, token := range Tokens(text) {
			index, sign := hashToken(token, h.Dimensions)
			vec[index] += sign
		}
		normalize(vec)
		out[i] = vec
	}
	return out, nil
}

type OpenAIEmbedder struct {
	BaseURL   string
	APIKey    string
	Model     string
	BatchSize int
	Client    *http.Client
}

func (e *OpenAIEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	if e.Client == nil {
		e.Client = &http.Client{Timeout: 60 * time.Second}
	}
	if e.BatchSize <= 0 {
		e.BatchSize = 32
	}

	all := make([][]float32, 0, len(texts))
	for start := 0; start < len(texts); start += e.BatchSize {
		end := start + e.BatchSize
		if end > len(texts) {
			end = len(texts)
		}
		batch, err := e.embedBatch(ctx, texts[start:end])
		if err != nil {
			return nil, err
		}
		all = append(all, batch...)
	}
	return all, nil
}

func (e *OpenAIEmbedder) embedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	body := map[string]any{
		"model": e.Model,
		"input": texts,
	}
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.BaseURL+"/embeddings", bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if e.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+e.APIKey)
	}

	resp, err := e.Client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		var errBody bytes.Buffer
		_, _ = errBody.ReadFrom(resp.Body)
		return nil, fmt.Errorf("embedding request failed: %s: %s", resp.Status, errBody.String())
	}

	var decoded struct {
		Data []struct {
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		return nil, err
	}
	if len(decoded.Data) != len(texts) {
		return nil, fmt.Errorf("embedding response length mismatch: got %d want %d", len(decoded.Data), len(texts))
	}
	out := make([][]float32, len(texts))
	for i, item := range decoded.Data {
		idx := item.Index
		if idx < 0 || idx >= len(texts) {
			idx = i
		}
		vec := item.Embedding
		normalize(vec)
		out[idx] = vec
	}
	return out, nil
}

func Cosine(a, b []float32) float64 {
	if len(a) == 0 || len(b) == 0 || len(a) != len(b) {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		av := float64(a[i])
		bv := float64(b[i])
		dot += av * bv
		na += av * av
		nb += bv * bv
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

func LexicalScore(querySet map[string]struct{}, text string) float64 {
	if len(querySet) == 0 {
		return 0
	}
	chunkSet := TokenSet(text)
	if len(chunkSet) == 0 {
		return 0
	}
	var overlap int
	for token := range querySet {
		if _, ok := chunkSet[token]; ok {
			overlap++
		}
	}
	return float64(overlap) / math.Sqrt(float64(len(querySet)*len(chunkSet)))
}

func normalize(vec []float32) {
	var sum float64
	for _, v := range vec {
		sum += float64(v * v)
	}
	if sum == 0 {
		return
	}
	scale := float32(1 / math.Sqrt(sum))
	for i := range vec {
		vec[i] *= scale
	}
}

func hashToken(token string, dimensions int) (int, float32) {
	h := fnv.New64a()
	_, _ = h.Write([]byte(token))
	sum := h.Sum64()
	idx := int(sum % uint64(dimensions))
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], sum)
	if b[0]&1 == 0 {
		return idx, 1
	}
	return idx, -1
}

func envInt(key string, fallback int) int {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return fallback
	}
	n, err := strconv.Atoi(value)
	if err != nil {
		return fallback
	}
	return n
}
