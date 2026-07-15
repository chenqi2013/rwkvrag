package api

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"time"

	"rwkvrag/internal/importer"
	"rwkvrag/internal/rag"
)

type Server struct {
	Store     *rag.Store
	Embedder  rag.Embedder
	Generator rag.Generator
	Logger    *slog.Logger
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", s.health)
	mux.HandleFunc("GET /v1/stats", s.stats)
	mux.HandleFunc("POST /v1/search", s.search)
	mux.HandleFunc("POST /v1/ask", s.ask)
	mux.HandleFunc("POST /v1/import/markdown", s.importMarkdown)
	mux.HandleFunc("POST /v1/import/finewiki", s.importFineWiki)
	return s.withMiddleware(mux)
}

type askRequest struct {
	Question string  `json:"question"`
	TopK     int     `json:"top_k,omitempty"`
	MinScore float64 `json:"min_score,omitempty"`
}

type askResponse struct {
	Answer  string       `json:"answer"`
	Sources []sourceItem `json:"sources"`
}

type searchResponse struct {
	Results []sourceItem `json:"results"`
}

type sourceItem struct {
	ID         string            `json:"id"`
	DocumentID string            `json:"document_id"`
	Source     string            `json:"source"`
	Title      string            `json:"title"`
	URI        string            `json:"uri,omitempty"`
	Score      float64           `json:"score"`
	Snippet    string            `json:"snippet"`
	Metadata   map[string]string `json:"metadata,omitempty"`
}

func (s *Server) health(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (s *Server) stats(w http.ResponseWriter, _ *http.Request) {
	stats, err := s.Store.Stats()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, stats)
}

func (s *Server) search(w http.ResponseWriter, r *http.Request) {
	var req askRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	results, err := s.searchResults(r.Context(), req)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	writeJSON(w, http.StatusOK, searchResponse{Results: toSources(results)})
}

func (s *Server) ask(w http.ResponseWriter, r *http.Request) {
	var req askRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	results, err := s.searchResults(r.Context(), req)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	answer, err := s.Generator.Generate(r.Context(), req.Question, results)
	if err != nil {
		writeError(w, http.StatusBadGateway, err)
		return
	}
	writeJSON(w, http.StatusOK, askResponse{Answer: answer, Sources: toSources(results)})
}

func (s *Server) searchResults(ctx context.Context, req askRequest) ([]rag.SearchResult, error) {
	req.Question = strings.TrimSpace(req.Question)
	if req.Question == "" {
		return nil, errors.New("question is required")
	}
	if req.TopK <= 0 {
		req.TopK = 5
	}
	return s.Store.Search(ctx, s.Embedder, req.Question, req.TopK, req.MinScore)
}

type markdownImportRequest struct {
	Path      string `json:"path"`
	Source    string `json:"source,omitempty"`
	BatchSize int    `json:"batch_size,omitempty"`
}

func (s *Server) importMarkdown(w http.ResponseWriter, r *http.Request) {
	var req markdownImportRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	if strings.TrimSpace(req.Path) == "" {
		writeError(w, http.StatusBadRequest, errors.New("path is required"))
		return
	}
	start := time.Now()
	stats, err := importer.ImportMarkdown(r.Context(), importer.MarkdownOptions{
		Path:      req.Path,
		Source:    req.Source,
		BatchSize: req.BatchSize,
	}, s.Store, s.Embedder)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"stats": stats, "elapsed": time.Since(start).String()})
}

type fineWikiImportRequest struct {
	Path      string `json:"path"`
	Source    string `json:"source,omitempty"`
	Language  string `json:"language,omitempty"`
	Limit     int    `json:"limit,omitempty"`
	MaxFiles  int    `json:"max_files,omitempty"`
	BatchSize int    `json:"batch_size,omitempty"`
}

func (s *Server) importFineWiki(w http.ResponseWriter, r *http.Request) {
	var req fineWikiImportRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	if strings.TrimSpace(req.Path) == "" {
		writeError(w, http.StatusBadRequest, errors.New("path is required"))
		return
	}
	start := time.Now()
	stats, err := importer.ImportFineWiki(r.Context(), importer.FineWikiOptions{
		Path:      req.Path,
		Source:    req.Source,
		Language:  req.Language,
		Limit:     req.Limit,
		MaxFiles:  req.MaxFiles,
		BatchSize: req.BatchSize,
	}, s.Store, s.Embedder)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"stats": stats, "elapsed": time.Since(start).String()})
}

func (s *Server) withMiddleware(next http.Handler) http.Handler {
	logger := s.Logger
	if logger == nil {
		logger = slog.New(slog.NewTextHandler(os.Stdout, nil))
	}
	origin := os.Getenv("RWKVRAG_CORS_ORIGIN")
	if origin == "" {
		origin = "*"
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", origin)
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		start := time.Now()
		next.ServeHTTP(w, r)
		logger.Info("request", "method", r.Method, "path", r.URL.Path, "elapsed", time.Since(start))
	})
}

func decodeJSON(r *http.Request, target any) error {
	defer r.Body.Close()
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	return dec.Decode(target)
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

func writeError(w http.ResponseWriter, status int, err error) {
	writeJSON(w, status, map[string]string{"error": err.Error()})
}

func toSources(results []rag.SearchResult) []sourceItem {
	out := make([]sourceItem, len(results))
	for i, result := range results {
		chunk := result.Chunk
		out[i] = sourceItem{
			ID:         chunk.ID,
			DocumentID: chunk.DocumentID,
			Source:     chunk.Source,
			Title:      chunk.Title,
			URI:        chunk.URI,
			Score:      result.Score,
			Snippet:    rag.Snippet(chunk.Content, 360),
			Metadata:   chunk.Metadata,
		}
	}
	return out
}
