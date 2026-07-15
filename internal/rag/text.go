package rag

import (
	"crypto/sha256"
	"encoding/hex"
	"path/filepath"
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"
)

const (
	DefaultChunkSize    = 1200
	DefaultChunkOverlap = 180
)

var blankLines = regexp.MustCompile(`\n{3,}`)

type Chunker struct {
	Size    int
	Overlap int
}

func DefaultChunker() Chunker {
	return Chunker{Size: DefaultChunkSize, Overlap: DefaultChunkOverlap}
}

func (c Chunker) Normalize() Chunker {
	if c.Size <= 0 {
		c.Size = DefaultChunkSize
	}
	if c.Overlap < 0 {
		c.Overlap = 0
	}
	if c.Overlap >= c.Size {
		c.Overlap = c.Size / 5
	}
	return c
}

func (c Chunker) Split(doc Document) []Chunk {
	c = c.Normalize()
	content := CleanText(doc.Content)
	if content == "" {
		return nil
	}

	parts := splitText(content, c.Size, c.Overlap)
	chunks := make([]Chunk, 0, len(parts))
	for i, part := range parts {
		chunkID := HashID(doc.ID, doc.Source, doc.Title, part)
		chunks = append(chunks, Chunk{
			ID:         chunkID,
			DocumentID: doc.ID,
			Source:     doc.Source,
			Title:      doc.Title,
			URI:        doc.URI,
			Content:    part,
			ChunkIndex: i,
			Metadata:   cloneMap(doc.Metadata),
		})
	}
	return chunks
}

func CleanText(s string) string {
	s = strings.ReplaceAll(s, "\r\n", "\n")
	s = strings.ReplaceAll(s, "\r", "\n")
	s = strings.TrimSpace(s)
	s = blankLines.ReplaceAllString(s, "\n\n")
	return s
}

func FirstMarkdownTitle(content, fallbackPath string) string {
	for _, line := range strings.Split(content, "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "# ") {
			title := strings.TrimSpace(strings.TrimPrefix(line, "# "))
			if title != "" {
				return title
			}
		}
	}
	base := filepath.Base(fallbackPath)
	ext := filepath.Ext(base)
	if ext != "" {
		base = strings.TrimSuffix(base, ext)
	}
	return base
}

func HashID(parts ...string) string {
	h := sha256.New()
	for _, part := range parts {
		h.Write([]byte(part))
		h.Write([]byte{0})
	}
	sum := h.Sum(nil)
	return hex.EncodeToString(sum[:])[:24]
}

func Snippet(text string, maxRunes int) string {
	text = strings.TrimSpace(CleanText(text))
	if maxRunes <= 0 || utf8.RuneCountInString(text) <= maxRunes {
		return text
	}
	runes := []rune(text)
	return strings.TrimSpace(string(runes[:maxRunes])) + "..."
}

func Tokens(text string) []string {
	text = strings.ToLower(text)
	var tokens []string
	var word []rune
	var han []rune

	flushWord := func() {
		if len(word) > 0 {
			tokens = append(tokens, string(word))
			word = word[:0]
		}
	}
	flushHan := func() {
		if len(han) == 0 {
			return
		}
		for _, r := range han {
			tokens = append(tokens, string(r))
		}
		for i := 0; i+1 < len(han); i++ {
			tokens = append(tokens, string(han[i:i+2]))
		}
		han = han[:0]
	}

	for _, r := range text {
		switch {
		case unicode.Is(unicode.Han, r):
			flushWord()
			han = append(han, r)
		case unicode.IsLetter(r) || unicode.IsDigit(r):
			flushHan()
			word = append(word, r)
		default:
			flushWord()
			flushHan()
		}
	}
	flushWord()
	flushHan()
	return tokens
}

func TokenSet(text string) map[string]struct{} {
	tokens := Tokens(text)
	set := make(map[string]struct{}, len(tokens))
	for _, token := range tokens {
		if token != "" {
			set[token] = struct{}{}
		}
	}
	return set
}

func splitText(text string, size, overlap int) []string {
	if len([]rune(text)) <= size {
		return []string{text}
	}

	paragraphs := strings.Split(text, "\n\n")
	var chunks []string
	var current strings.Builder
	currentLen := 0

	flush := func() {
		part := strings.TrimSpace(current.String())
		if part != "" {
			chunks = append(chunks, part)
		}
		current.Reset()
		currentLen = 0
	}

	for _, p := range paragraphs {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		pLen := utf8.RuneCountInString(p)
		if pLen > size {
			flush()
			chunks = append(chunks, splitRunes(p, size, overlap)...)
			continue
		}
		if currentLen > 0 && currentLen+2+pLen > size {
			flush()
		}
		if currentLen > 0 {
			current.WriteString("\n\n")
			currentLen += 2
		}
		current.WriteString(p)
		currentLen += pLen
	}
	flush()
	return chunks
}

func splitRunes(text string, size, overlap int) []string {
	runes := []rune(text)
	var parts []string
	step := size - overlap
	if step <= 0 {
		step = size
	}
	for start := 0; start < len(runes); start += step {
		end := start + size
		if end > len(runes) {
			end = len(runes)
		}
		parts = append(parts, strings.TrimSpace(string(runes[start:end])))
		if end == len(runes) {
			break
		}
	}
	return parts
}

func cloneMap(in map[string]string) map[string]string {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]string, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}
