package rag

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"
)

type Generator interface {
	Generate(ctx context.Context, question string, contexts []SearchResult) (string, error)
}

type LLMConfig struct {
	Provider    string
	BaseURL     string
	APIKey      string
	Model       string
	Temperature float64
	Timeout     time.Duration
}

func LLMConfigFromEnv() LLMConfig {
	cfg := LLMConfig{
		Provider:    strings.ToLower(strings.TrimSpace(os.Getenv("RWKVRAG_LLM_PROVIDER"))),
		BaseURL:     strings.TrimSpace(os.Getenv("RWKVRAG_LLM_BASE_URL")),
		APIKey:      strings.TrimSpace(os.Getenv("RWKVRAG_LLM_API_KEY")),
		Model:       strings.TrimSpace(os.Getenv("RWKVRAG_LLM_MODEL")),
		Temperature: envFloat("RWKVRAG_LLM_TEMPERATURE", 0.2),
		Timeout:     time.Duration(envInt("RWKVRAG_LLM_TIMEOUT_SECONDS", 120)) * time.Second,
	}
	if cfg.Provider == "" {
		if cfg.BaseURL != "" || cfg.APIKey != "" || cfg.Model != "" {
			cfg.Provider = "openai"
		} else {
			cfg.Provider = "extractive"
		}
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.openai.com/v1"
	}
	return cfg
}

func NewGenerator(cfg LLMConfig) (Generator, error) {
	switch strings.ToLower(cfg.Provider) {
	case "", "extractive", "none":
		return ExtractiveGenerator{}, nil
	case "openai", "openai-compatible", "rwkv":
		if cfg.Model == "" {
			return nil, errors.New("RWKVRAG_LLM_MODEL is required for openai-compatible generation")
		}
		if cfg.Timeout <= 0 {
			cfg.Timeout = 120 * time.Second
		}
		return &OpenAIChatGenerator{
			BaseURL:     strings.TrimRight(cfg.BaseURL, "/"),
			APIKey:      cfg.APIKey,
			Model:       cfg.Model,
			Temperature: cfg.Temperature,
			Client:      &http.Client{Timeout: cfg.Timeout},
		}, nil
	default:
		return nil, fmt.Errorf("unknown llm provider %q", cfg.Provider)
	}
}

type ExtractiveGenerator struct{}

func (ExtractiveGenerator) Generate(_ context.Context, question string, contexts []SearchResult) (string, error) {
	if len(contexts) == 0 {
		return "没有在知识库中检索到足够相关的资料。", nil
	}
	var b strings.Builder
	b.WriteString("未配置生成模型，先返回检索到的相关资料摘要。\n\n")
	b.WriteString("问题：")
	b.WriteString(question)
	b.WriteString("\n\n")
	for i, item := range contexts {
		fmt.Fprintf(&b, "[%d] %s\n%s\n\n", i+1, item.Chunk.Title, Snippet(item.Chunk.Content, 260))
	}
	return strings.TrimSpace(b.String()), nil
}

type OpenAIChatGenerator struct {
	BaseURL     string
	APIKey      string
	Model       string
	Temperature float64
	Client      *http.Client
}

func (g *OpenAIChatGenerator) Generate(ctx context.Context, question string, contexts []SearchResult) (string, error) {
	if g.Client == nil {
		g.Client = &http.Client{Timeout: 120 * time.Second}
	}
	contextText := BuildContext(contexts, 1800)
	payload := map[string]any{
		"model":       g.Model,
		"temperature": g.Temperature,
		"messages": []map[string]string{
			{
				"role":    "system",
				"content": "你是企业知识库问答助手。只根据用户提供的资料回答问题；资料不足时直接说明不足。回答要简洁、准确，并在关键结论后用 [1]、[2] 这样的编号标注来源。",
			},
			{
				"role":    "user",
				"content": fmt.Sprintf("问题：%s\n\n资料：\n%s", question, contextText),
			},
		},
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return "", err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, g.BaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	if g.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+g.APIKey)
	}

	resp, err := g.Client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		var errBody bytes.Buffer
		_, _ = errBody.ReadFrom(resp.Body)
		return "", fmt.Errorf("chat completion failed: %s: %s", resp.Status, errBody.String())
	}
	var decoded struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		return "", err
	}
	if len(decoded.Choices) == 0 {
		return "", errors.New("chat completion returned no choices")
	}
	return strings.TrimSpace(decoded.Choices[0].Message.Content), nil
}

func BuildContext(results []SearchResult, snippetRunes int) string {
	var b strings.Builder
	for i, result := range results {
		chunk := result.Chunk
		fmt.Fprintf(&b, "[%d] title=%q source=%q uri=%q score=%.4f\n", i+1, chunk.Title, chunk.Source, chunk.URI, result.Score)
		b.WriteString(Snippet(chunk.Content, snippetRunes))
		b.WriteString("\n\n")
	}
	return strings.TrimSpace(b.String())
}

func envFloat(key string, fallback float64) float64 {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return fallback
	}
	var out float64
	if _, err := fmt.Sscanf(value, "%f", &out); err != nil {
		return fallback
	}
	return out
}
