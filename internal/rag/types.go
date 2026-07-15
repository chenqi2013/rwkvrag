package rag

type Document struct {
	ID       string            `json:"id"`
	Source   string            `json:"source"`
	Title    string            `json:"title"`
	URI      string            `json:"uri,omitempty"`
	Content  string            `json:"content"`
	Metadata map[string]string `json:"metadata,omitempty"`
}

type Chunk struct {
	ID         string            `json:"id"`
	DocumentID string            `json:"document_id"`
	Source     string            `json:"source"`
	Title      string            `json:"title"`
	URI        string            `json:"uri,omitempty"`
	Content    string            `json:"content"`
	ChunkIndex int               `json:"chunk_index"`
	Metadata   map[string]string `json:"metadata,omitempty"`
	Vector     []float32         `json:"vector,omitempty"`
}

type SearchResult struct {
	Chunk Chunk   `json:"chunk"`
	Score float64 `json:"score"`
}

type ImportStats struct {
	Documents int `json:"documents"`
	Chunks    int `json:"chunks"`
	Skipped   int `json:"skipped"`
	Files     int `json:"files"`
}

func (s *ImportStats) Add(other ImportStats) {
	s.Documents += other.Documents
	s.Chunks += other.Chunks
	s.Skipped += other.Skipped
	s.Files += other.Files
}
