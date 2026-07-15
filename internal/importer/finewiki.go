package importer

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/parquet-go/parquet-go"

	"rwkvrag/internal/rag"
)

type FineWikiOptions struct {
	Path      string
	Source    string
	Language  string
	Limit     int
	MaxFiles  int
	BatchSize int
}

func ImportFineWiki(ctx context.Context, opts FineWikiOptions, store *rag.Store, embedder rag.Embedder) (rag.ImportStats, error) {
	var stats rag.ImportStats
	if opts.Source == "" {
		opts.Source = "finewiki-zh"
	}
	if opts.Language == "" {
		opts.Language = "zh"
	}
	if opts.BatchSize <= 0 {
		opts.BatchSize = 32
	}
	files, err := parquetFiles(opts.Path)
	if err != nil {
		return stats, err
	}
	if opts.MaxFiles > 0 && opts.MaxFiles < len(files) {
		files = files[:opts.MaxFiles]
	}
	stats.Files = len(files)

	for _, path := range files {
		if err := ctx.Err(); err != nil {
			return stats, err
		}
		fileStats, err := importFineWikiFile(ctx, path, opts, store, embedder, stats.Documents)
		if err != nil {
			return stats, err
		}
		stats.Documents += fileStats.Documents
		stats.Chunks += fileStats.Chunks
		stats.Skipped += fileStats.Skipped
		if opts.Limit > 0 && stats.Documents >= opts.Limit {
			break
		}
	}
	return stats, nil
}

func importFineWikiFile(ctx context.Context, path string, opts FineWikiOptions, store *rag.Store, embedder rag.Embedder, alreadyImported int) (rag.ImportStats, error) {
	var stats rag.ImportStats
	f, err := os.Open(path)
	if err != nil {
		return stats, err
	}
	defer f.Close()

	info, err := f.Stat()
	if err != nil {
		return stats, err
	}
	parquetFile, err := parquet.OpenFile(f, info.Size())
	if err != nil {
		return stats, fmt.Errorf("open parquet %s: %w", path, err)
	}
	reader := parquet.NewReader(parquetFile)
	defer reader.Close()

	columns := reader.Schema().Columns()
	rows := make([]parquet.Row, 64)
	batch := make([]rag.Document, 0, opts.BatchSize)

	flush := func() error {
		if len(batch) == 0 {
			return nil
		}
		added, err := store.AddDocuments(ctx, batch, embedder, 0)
		if err != nil {
			return err
		}
		stats.Add(added)
		batch = batch[:0]
		return nil
	}

	rowNumber := 0
	accepted := 0
	for {
		n, err := reader.ReadRows(rows)
		if n > 0 {
			for i := 0; i < n; i++ {
				if err := ctx.Err(); err != nil {
					return stats, err
				}
				if opts.Limit > 0 && alreadyImported+accepted >= opts.Limit {
					if err := flush(); err != nil {
						return stats, err
					}
					return stats, nil
				}
				doc, ok := fineWikiRowToDocument(rows[i], columns, opts.Source, opts.Language, path, rowNumber)
				rowNumber++
				if !ok {
					stats.Skipped++
					continue
				}
				batch = append(batch, doc)
				accepted++
				if len(batch) >= opts.BatchSize {
					if err := flush(); err != nil {
						return stats, err
					}
				}
			}
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return stats, fmt.Errorf("read parquet %s: %w", path, err)
		}
	}
	if err := flush(); err != nil {
		return stats, err
	}
	return stats, nil
}

func fineWikiRowToDocument(row parquet.Row, columns [][]string, source, language, filePath string, rowNumber int) (rag.Document, bool) {
	values := map[string]string{}
	row.Range(func(columnIndex int, columnValues []parquet.Value) bool {
		if columnIndex < 0 || columnIndex >= len(columns) {
			return true
		}
		key := canonicalColumn(columns[columnIndex])
		if key == "" {
			return true
		}
		text := valuesToString(columnValues)
		if text != "" {
			values[key] = text
		}
		return true
	})

	rowLanguage := firstValue(values, "language", "lang")
	if rowLanguage != "" && language != "" && !strings.EqualFold(rowLanguage, language) && !strings.HasPrefix(strings.ToLower(rowLanguage), strings.ToLower(language)) {
		return rag.Document{}, false
	}

	content := firstValue(values, "text", "markdown", "content", "article", "plain_text", "body")
	if strings.TrimSpace(content) == "" {
		return rag.Document{}, false
	}
	title := firstValue(values, "title", "name")
	if title == "" {
		title = rag.Snippet(content, 60)
	}
	externalID := firstValue(values, "id", "page_id", "wiki_id", "wikidata_id")
	if externalID == "" {
		externalID = filepath.Base(filePath) + ":" + strconv.Itoa(rowNumber)
	}
	uri := firstValue(values, "url", "uri", "source_url")
	if uri == "" && title != "" {
		uri = "https://zh.wikipedia.org/wiki/" + strings.ReplaceAll(title, " ", "_")
	}

	metadata := map[string]string{
		"kind": "finewiki",
		"file": filepath.Base(filePath),
	}
	for k, v := range values {
		if isLargeTextColumn(k) {
			continue
		}
		metadata[k] = rag.Snippet(v, 240)
	}

	return rag.Document{
		ID:       rag.HashID(source, externalID, title),
		Source:   source,
		Title:    title,
		URI:      uri,
		Content:  content,
		Metadata: metadata,
	}, true
}

func parquetFiles(path string) ([]string, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	if !info.IsDir() {
		return []string{path}, nil
	}
	matches, err := filepath.Glob(filepath.Join(path, "*.parquet"))
	if err != nil {
		return nil, err
	}
	sort.Strings(matches)
	return matches, nil
}

func canonicalColumn(path []string) string {
	if len(path) == 0 {
		return ""
	}
	name := strings.ToLower(path[len(path)-1])
	name = strings.ReplaceAll(name, "-", "_")
	name = strings.ReplaceAll(name, " ", "_")
	return name
}

func valuesToString(values []parquet.Value) string {
	parts := make([]string, 0, len(values))
	for _, value := range values {
		if value.IsNull() {
			continue
		}
		var s string
		switch value.Kind() {
		case parquet.Boolean:
			s = strconv.FormatBool(value.Boolean())
		case parquet.Int32:
			s = strconv.FormatInt(int64(value.Int32()), 10)
		case parquet.Int64:
			s = strconv.FormatInt(value.Int64(), 10)
		case parquet.Float:
			s = strconv.FormatFloat(float64(value.Float()), 'f', -1, 32)
		case parquet.Double:
			s = strconv.FormatFloat(value.Double(), 'f', -1, 64)
		case parquet.ByteArray, parquet.FixedLenByteArray:
			s = string(value.ByteArray())
		default:
			s = value.String()
		}
		if s != "" {
			parts = append(parts, s)
		}
	}
	return strings.Join(parts, " ")
}

func firstValue(values map[string]string, keys ...string) string {
	for _, key := range keys {
		if value := strings.TrimSpace(values[key]); value != "" {
			return value
		}
	}
	return ""
}

func isLargeTextColumn(key string) bool {
	switch key {
	case "text", "markdown", "content", "article", "plain_text", "body", "html":
		return true
	default:
		return false
	}
}
