package importer

import (
	"context"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"rwkvrag/internal/rag"
)

type MarkdownOptions struct {
	Path      string
	Source    string
	BatchSize int
}

func ImportMarkdown(ctx context.Context, opts MarkdownOptions, store *rag.Store, embedder rag.Embedder) (rag.ImportStats, error) {
	var stats rag.ImportStats
	if opts.Source == "" {
		opts.Source = "company-markdown"
	}
	if opts.BatchSize <= 0 {
		opts.BatchSize = 64
	}

	files, err := markdownFiles(opts.Path)
	if err != nil {
		return stats, err
	}
	stats.Files = len(files)

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

	root, _ := filepath.Abs(opts.Path)
	for _, path := range files {
		if err := ctx.Err(); err != nil {
			return stats, err
		}
		contentBytes, err := os.ReadFile(path)
		if err != nil {
			stats.Skipped++
			continue
		}
		content := string(contentBytes)
		rel := path
		if r, err := filepath.Rel(root, path); err == nil && !strings.HasPrefix(r, "..") {
			rel = r
		}
		doc := rag.Document{
			ID:      rag.HashID(opts.Source, rel, content),
			Source:  opts.Source,
			Title:   rag.FirstMarkdownTitle(content, path),
			URI:     path,
			Content: content,
			Metadata: map[string]string{
				"kind": "markdown",
				"path": rel,
			},
		}
		batch = append(batch, doc)
		if len(batch) >= opts.BatchSize {
			if err := flush(); err != nil {
				return stats, err
			}
		}
	}
	if err := flush(); err != nil {
		return stats, err
	}
	return stats, nil
}

func markdownFiles(path string) ([]string, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	var files []string
	if !info.IsDir() {
		if isMarkdown(path) {
			return []string{path}, nil
		}
		return files, nil
	}
	err = filepath.WalkDir(path, func(p string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			switch d.Name() {
			case ".git", "node_modules", "vendor":
				return filepath.SkipDir
			}
			return nil
		}
		if isMarkdown(p) {
			files = append(files, p)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	sort.Strings(files)
	return files, nil
}

func isMarkdown(path string) bool {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".md", ".markdown", ".mdx":
		return true
	default:
		return false
	}
}
