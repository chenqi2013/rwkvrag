package importer

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

const FineWikiDataset = "HuggingFaceFW/finewiki"

type DownloadFineWikiOptions struct {
	Config   string
	OutDir   string
	MaxFiles int
	Token    string
	Logf     func(format string, args ...any)
}

func DownloadFineWiki(ctx context.Context, opts DownloadFineWikiOptions) error {
	if opts.Config == "" {
		opts.Config = "zh"
	}
	if opts.OutDir == "" {
		opts.OutDir = filepath.Join("data", "finewiki", opts.Config)
	}
	if opts.Token == "" {
		opts.Token = os.Getenv("HF_TOKEN")
	}
	if err := os.MkdirAll(opts.OutDir, 0o755); err != nil {
		return err
	}

	files, err := ListFineWikiFiles(ctx, opts.Config, opts.Token)
	if err != nil {
		return err
	}
	if opts.MaxFiles > 0 && opts.MaxFiles < len(files) {
		files = files[:opts.MaxFiles]
	}
	client := &http.Client{Timeout: 0}
	for _, file := range files {
		if err := ctx.Err(); err != nil {
			return err
		}
		name := filepath.Base(file.Path)
		dest := filepath.Join(opts.OutDir, name)
		if info, err := os.Stat(dest); err == nil && info.Size() == file.Size {
			logf(opts.Logf, "skip %s; already downloaded", name)
			continue
		}
		logf(opts.Logf, "download %s (%.2f GB)", name, float64(file.Size)/(1024*1024*1024))
		if err := downloadHFFile(ctx, client, opts.Token, file.Path, dest); err != nil {
			return err
		}
	}
	return nil
}

type FineWikiFile struct {
	Path string
	Size int64
}

func ListFineWikiFiles(ctx context.Context, config, token string) ([]FineWikiFile, error) {
	dataPath := fmt.Sprintf("data/%swiki", config)
	url := fmt.Sprintf("https://huggingface.co/api/datasets/%s/tree/main/%s?recursive=false", FineWikiDataset, dataPath)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("list FineWiki files failed: %s: %s", resp.Status, string(body))
	}
	var entries []struct {
		Type string `json:"type"`
		Path string `json:"path"`
		Size int64  `json:"size"`
		LFS  struct {
			Size int64 `json:"size"`
		} `json:"lfs"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&entries); err != nil {
		return nil, err
	}
	files := make([]FineWikiFile, 0, len(entries))
	for _, entry := range entries {
		if entry.Type != "file" || !strings.HasSuffix(entry.Path, ".parquet") {
			continue
		}
		size := entry.Size
		if entry.LFS.Size > 0 {
			size = entry.LFS.Size
		}
		files = append(files, FineWikiFile{Path: entry.Path, Size: size})
	}
	return files, nil
}

func downloadHFFile(ctx context.Context, client *http.Client, token, repoPath, dest string) error {
	url := fmt.Sprintf("https://huggingface.co/datasets/%s/resolve/main/%s", FineWikiDataset, repoPath)
	tmp := dest + ".tmp"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("download %s failed: %s: %s", repoPath, resp.Status, string(body))
	}
	out, err := os.Create(tmp)
	if err != nil {
		return err
	}
	_, copyErr := io.Copy(out, resp.Body)
	closeErr := out.Close()
	if copyErr != nil {
		_ = os.Remove(tmp)
		return copyErr
	}
	if closeErr != nil {
		_ = os.Remove(tmp)
		return closeErr
	}
	return os.Rename(tmp, dest)
}

func logf(fn func(string, ...any), format string, args ...any) {
	if fn != nil {
		fn(format, args...)
	}
}
