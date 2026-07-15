package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"rwkvrag/internal/api"
	"rwkvrag/internal/importer"
	"rwkvrag/internal/rag"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		log.Fatal(err)
	}
}

func run(args []string) error {
	if len(args) == 0 {
		usage()
		return nil
	}
	switch args[0] {
	case "serve":
		return serve(args[1:])
	case "import-markdown":
		return importMarkdown(args[1:])
	case "import-finewiki":
		return importFineWiki(args[1:])
	case "download-finewiki":
		return downloadFineWiki(args[1:])
	case "search":
		return search(args[1:])
	case "ask":
		return ask(args[1:])
	case "stats":
		return stats(args[1:])
	case "help", "-h", "--help":
		usage()
		return nil
	default:
		return fmt.Errorf("unknown command %q", args[0])
	}
}

func serve(args []string) error {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	addr := fs.String("addr", envString("RWKVRAG_ADDR", ":8080"), "HTTP listen address")
	db := fs.String("db", envString("RWKVRAG_DB", "data/index.jsonl"), "local JSONL index path")
	chunkSize := fs.Int("chunk-size", envInt("RWKVRAG_CHUNK_SIZE", rag.DefaultChunkSize), "chunk size in runes")
	chunkOverlap := fs.Int("chunk-overlap", envInt("RWKVRAG_CHUNK_OVERLAP", rag.DefaultChunkOverlap), "chunk overlap in runes")
	if err := fs.Parse(args); err != nil {
		return err
	}
	store, embedder, generator, err := runtime(*db, *chunkSize, *chunkOverlap)
	if err != nil {
		return err
	}
	server := &api.Server{
		Store:     store,
		Embedder:  embedder,
		Generator: generator,
		Logger:    slog.New(slog.NewTextHandler(os.Stdout, nil)),
	}
	httpServer := &http.Server{
		Addr:              *addr,
		Handler:           server.Handler(),
		ReadHeaderTimeout: 10 * time.Second,
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	errCh := make(chan error, 1)
	go func() {
		log.Printf("rwkvrag listening on %s", *addr)
		if err := httpServer.ListenAndServe(); !errors.Is(err, http.ErrServerClosed) {
			errCh <- err
		}
		close(errCh)
	}()

	select {
	case <-ctx.Done():
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		return httpServer.Shutdown(shutdownCtx)
	case err := <-errCh:
		return err
	}
}

func importMarkdown(args []string) error {
	fs := flag.NewFlagSet("import-markdown", flag.ExitOnError)
	path := fs.String("path", "", "markdown file or directory")
	source := fs.String("source", "company-markdown", "source name")
	db := fs.String("db", envString("RWKVRAG_DB", "data/index.jsonl"), "local JSONL index path")
	batchSize := fs.Int("batch-size", 64, "documents per indexing batch")
	chunkSize := fs.Int("chunk-size", envInt("RWKVRAG_CHUNK_SIZE", rag.DefaultChunkSize), "chunk size in runes")
	chunkOverlap := fs.Int("chunk-overlap", envInt("RWKVRAG_CHUNK_OVERLAP", rag.DefaultChunkOverlap), "chunk overlap in runes")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if strings.TrimSpace(*path) == "" {
		return errors.New("--path is required")
	}
	store, embedder, _, err := runtime(*db, *chunkSize, *chunkOverlap)
	if err != nil {
		return err
	}
	stats, err := importer.ImportMarkdown(context.Background(), importer.MarkdownOptions{
		Path:      *path,
		Source:    *source,
		BatchSize: *batchSize,
	}, store, embedder)
	if err != nil {
		return err
	}
	printJSON(stats)
	return nil
}

func importFineWiki(args []string) error {
	fs := flag.NewFlagSet("import-finewiki", flag.ExitOnError)
	path := fs.String("path", "", "FineWiki parquet file or directory")
	source := fs.String("source", "finewiki-zh", "source name")
	language := fs.String("language", "zh", "language filter")
	limit := fs.Int("limit", 0, "maximum documents to import")
	maxFiles := fs.Int("max-files", 0, "maximum parquet files to import")
	batchSize := fs.Int("batch-size", 32, "documents per indexing batch")
	db := fs.String("db", envString("RWKVRAG_DB", "data/index.jsonl"), "local JSONL index path")
	chunkSize := fs.Int("chunk-size", envInt("RWKVRAG_CHUNK_SIZE", rag.DefaultChunkSize), "chunk size in runes")
	chunkOverlap := fs.Int("chunk-overlap", envInt("RWKVRAG_CHUNK_OVERLAP", rag.DefaultChunkOverlap), "chunk overlap in runes")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if strings.TrimSpace(*path) == "" {
		return errors.New("--path is required")
	}
	store, embedder, _, err := runtime(*db, *chunkSize, *chunkOverlap)
	if err != nil {
		return err
	}
	stats, err := importer.ImportFineWiki(context.Background(), importer.FineWikiOptions{
		Path:      *path,
		Source:    *source,
		Language:  *language,
		Limit:     *limit,
		MaxFiles:  *maxFiles,
		BatchSize: *batchSize,
	}, store, embedder)
	if err != nil {
		return err
	}
	printJSON(stats)
	return nil
}

func downloadFineWiki(args []string) error {
	fs := flag.NewFlagSet("download-finewiki", flag.ExitOnError)
	config := fs.String("config", "zh", "FineWiki config/language")
	out := fs.String("out", "data/finewiki/zh", "output directory")
	maxFiles := fs.Int("max-files", 0, "maximum parquet files to download")
	if err := fs.Parse(args); err != nil {
		return err
	}
	return importer.DownloadFineWiki(context.Background(), importer.DownloadFineWikiOptions{
		Config:   *config,
		OutDir:   *out,
		MaxFiles: *maxFiles,
		Logf: func(format string, args ...any) {
			log.Printf(format, args...)
		},
	})
}

func search(args []string) error {
	fs := flag.NewFlagSet("search", flag.ExitOnError)
	query := fs.String("q", "", "query")
	topK := fs.Int("top-k", 5, "number of results")
	db := fs.String("db", envString("RWKVRAG_DB", "data/index.jsonl"), "local JSONL index path")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if strings.TrimSpace(*query) == "" {
		return errors.New("--q is required")
	}
	store, embedder, _, err := runtime(*db, rag.DefaultChunkSize, rag.DefaultChunkOverlap)
	if err != nil {
		return err
	}
	results, err := store.Search(context.Background(), embedder, *query, *topK, 0)
	if err != nil {
		return err
	}
	printJSON(results)
	return nil
}

func ask(args []string) error {
	fs := flag.NewFlagSet("ask", flag.ExitOnError)
	question := fs.String("q", "", "question")
	topK := fs.Int("top-k", 5, "number of context chunks")
	db := fs.String("db", envString("RWKVRAG_DB", "data/index.jsonl"), "local JSONL index path")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if strings.TrimSpace(*question) == "" {
		return errors.New("--q is required")
	}
	store, embedder, generator, err := runtime(*db, rag.DefaultChunkSize, rag.DefaultChunkOverlap)
	if err != nil {
		return err
	}
	results, err := store.Search(context.Background(), embedder, *question, *topK, 0)
	if err != nil {
		return err
	}
	answer, err := generator.Generate(context.Background(), *question, results)
	if err != nil {
		return err
	}
	printJSON(map[string]any{"answer": answer, "sources": results})
	return nil
}

func stats(args []string) error {
	fs := flag.NewFlagSet("stats", flag.ExitOnError)
	db := fs.String("db", envString("RWKVRAG_DB", "data/index.jsonl"), "local JSONL index path")
	if err := fs.Parse(args); err != nil {
		return err
	}
	store, err := rag.NewStore(*db, rag.DefaultChunker())
	if err != nil {
		return err
	}
	stats, err := store.Stats()
	if err != nil {
		return err
	}
	printJSON(stats)
	return nil
}

func runtime(db string, chunkSize, chunkOverlap int) (*rag.Store, rag.Embedder, rag.Generator, error) {
	store, err := rag.NewStore(db, rag.Chunker{Size: chunkSize, Overlap: chunkOverlap})
	if err != nil {
		return nil, nil, nil, err
	}
	embedder, err := rag.NewEmbedder(rag.EmbeddingConfigFromEnv())
	if err != nil {
		return nil, nil, nil, err
	}
	generator, err := rag.NewGenerator(rag.LLMConfigFromEnv())
	if err != nil {
		return nil, nil, nil, err
	}
	return store, embedder, generator, nil
}

func usage() {
	fmt.Println(`rwkvrag

Commands:
  serve                 Start HTTP API
  download-finewiki     Download HuggingFaceFW/finewiki parquet files
  import-finewiki       Import FineWiki parquet files
  import-markdown       Import company Markdown files
  search                Search the local index
  ask                   Search and answer using configured LLM
  stats                 Show index stats

Examples:
  rwkvrag serve --addr :8080
  rwkvrag download-finewiki --config zh --out data/finewiki/zh --max-files 1
  rwkvrag import-finewiki --path data/finewiki/zh --limit 10000
  rwkvrag import-markdown --path ./company-docs --source company
  rwkvrag ask --q "RWKV 是什么？"`)
}

func printJSON(value any) {
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	_ = encoder.Encode(value)
}

func envString(key, fallback string) string {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return fallback
	}
	return value
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
