package rag

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"regexp"
	"strings"
	"time"
	"unicode/utf8"
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
	if answer, ok := extractCapitalAnswer(question, contexts); ok {
		return answer, nil
	}
	if answer, ok := extractPopulationProvinceAnswer(question, contexts); ok {
		return answer, nil
	}
	if answer, ok := bestEvidenceSentence(question, contexts); ok {
		return answer, nil
	}
	var b strings.Builder
	b.WriteString("根据检索到的资料：\n\n")
	for i, item := range contexts {
		fmt.Fprintf(&b, "[%d] %s\n%s\n\n", i+1, item.Chunk.Title, Snippet(item.Chunk.Content, 260))
	}
	return strings.TrimSpace(b.String()), nil
}

var (
	capitalQuestionPatterns = []*regexp.Regexp{
		regexp.MustCompile(`^(.+?)的首都(?:是)?(?:哪里|哪儿|在哪(?:里)?|什么|哪(?:一)?(?:个|座)?城市)[？?]?$`),
		regexp.MustCompile(`^(.+?)首都(?:是)?(?:哪里|哪儿|在哪(?:里)?|什么|哪(?:一)?(?:个|座)?城市)[？?]?$`),
	}
	populationProvinceQuestionPatterns = []*regexp.Regexp{
		regexp.MustCompile(`^(.+?)(?:的)?人口(最多|最少)的(?:省份|省级行政区)(?:是)?(?:哪个|哪一个|哪|什么)(?:省|省份|地区)?[？?]?$`),
		regexp.MustCompile(`^(.+?)(?:的)?(?:哪个|哪一个|哪|什么)(?:省|省份|省级行政区)(?:的)?人口(?:是)?(最多|最少)[？?]?$`),
	}
	populationProvinceEvidencePatterns = buildPopulationProvinceEvidencePatterns()
	sentenceSeparator                  = regexp.MustCompile(`[。！？!?；;\n]+`)
)

type populationProvinceQuestion struct {
	Scope     string
	Direction string
}

type populationProvinceEvidencePattern struct {
	Canonical string
	Pattern   *regexp.Regexp
}

var provinceNames = []struct {
	Canonical string
	Aliases   []string
}{
	{Canonical: "北京市", Aliases: []string{"北京市", "北京"}},
	{Canonical: "天津市", Aliases: []string{"天津市", "天津"}},
	{Canonical: "河北省", Aliases: []string{"河北省", "河北"}},
	{Canonical: "山西省", Aliases: []string{"山西省", "山西"}},
	{Canonical: "内蒙古自治区", Aliases: []string{"内蒙古自治区", "内蒙古"}},
	{Canonical: "辽宁省", Aliases: []string{"辽宁省", "辽宁"}},
	{Canonical: "吉林省", Aliases: []string{"吉林省", "吉林"}},
	{Canonical: "黑龙江省", Aliases: []string{"黑龙江省", "黑龙江"}},
	{Canonical: "上海市", Aliases: []string{"上海市", "上海"}},
	{Canonical: "江苏省", Aliases: []string{"江苏省", "江苏"}},
	{Canonical: "浙江省", Aliases: []string{"浙江省", "浙江"}},
	{Canonical: "安徽省", Aliases: []string{"安徽省", "安徽"}},
	{Canonical: "福建省", Aliases: []string{"福建省", "福建"}},
	{Canonical: "江西省", Aliases: []string{"江西省", "江西"}},
	{Canonical: "山东省", Aliases: []string{"山东省", "山东"}},
	{Canonical: "河南省", Aliases: []string{"河南省", "河南"}},
	{Canonical: "湖北省", Aliases: []string{"湖北省", "湖北"}},
	{Canonical: "湖南省", Aliases: []string{"湖南省", "湖南"}},
	{Canonical: "广东省", Aliases: []string{"广东省", "广东"}},
	{Canonical: "广西壮族自治区", Aliases: []string{"广西壮族自治区", "广西"}},
	{Canonical: "海南省", Aliases: []string{"海南省", "海南"}},
	{Canonical: "重庆市", Aliases: []string{"重庆市", "重庆"}},
	{Canonical: "四川省", Aliases: []string{"四川省", "四川"}},
	{Canonical: "贵州省", Aliases: []string{"贵州省", "贵州"}},
	{Canonical: "云南省", Aliases: []string{"云南省", "云南"}},
	{Canonical: "西藏自治区", Aliases: []string{"西藏自治区", "西藏"}},
	{Canonical: "陕西省", Aliases: []string{"陕西省", "陕西"}},
	{Canonical: "甘肃省", Aliases: []string{"甘肃省", "甘肃"}},
	{Canonical: "青海省", Aliases: []string{"青海省", "青海"}},
	{Canonical: "宁夏回族自治区", Aliases: []string{"宁夏回族自治区", "宁夏"}},
	{Canonical: "新疆维吾尔自治区", Aliases: []string{"新疆维吾尔自治区", "新疆"}},
	{Canonical: "台湾省", Aliases: []string{"台湾省", "台湾"}},
	{Canonical: "香港特别行政区", Aliases: []string{"香港特别行政区", "香港"}},
	{Canonical: "澳门特别行政区", Aliases: []string{"澳门特别行政区", "澳门"}},
}

func extractCapitalAnswer(question string, contexts []SearchResult) (string, bool) {
	subject, ok := capitalQuestionSubject(question)
	if !ok {
		return "", false
	}
	aliases := capitalSubjectAliases(subject)
	for contextIndex, item := range contexts {
		if answer, ok := capitalAnswerInText(aliases, item.Chunk.Content); ok {
			return fmt.Sprintf("%s的首都是%s。[%d]", subject, answer, contextIndex+1), true
		}
	}
	return "", false
}

func capitalAnswerInText(aliases []string, text string) (string, bool) {
	for _, alias := range aliases {
		pattern := regexp.MustCompile(regexp.QuoteMeta(alias) + `(?:的)?首都(?:为|是)\s*([\p{Han}·]{1,12})`)
		match := pattern.FindStringSubmatch(text)
		if len(match) != 2 {
			continue
		}
		answer := strings.TrimSpace(match[1])
		if answer != "" {
			return answer, true
		}
	}
	return "", false
}

func capitalSubjectAliases(subject string) []string {
	if subject == "中国" || subject == "中华人民共和国" {
		return []string{"中华人民共和国", "中国"}
	}
	return []string{subject}
}

func capitalQuestionSubject(question string) (string, bool) {
	question = strings.TrimSpace(question)
	for _, pattern := range capitalQuestionPatterns {
		match := pattern.FindStringSubmatch(question)
		if len(match) == 2 {
			subject := strings.TrimSpace(match[1])
			return subject, subject != ""
		}
	}
	return "", false
}

func parsePopulationProvinceQuestion(question string) (populationProvinceQuestion, bool) {
	question = strings.TrimSpace(question)
	for _, pattern := range populationProvinceQuestionPatterns {
		match := pattern.FindStringSubmatch(question)
		if len(match) != 3 {
			continue
		}
		parsed := populationProvinceQuestion{
			Scope:     strings.TrimSpace(match[1]),
			Direction: strings.TrimSpace(match[2]),
		}
		return parsed, parsed.Scope != "" && parsed.Direction != ""
	}
	return populationProvinceQuestion{}, false
}

func extractPopulationProvinceAnswer(question string, contexts []SearchResult) (string, bool) {
	parsed, ok := parsePopulationProvinceQuestion(question)
	if !ok {
		return "", false
	}
	for contextIndex, item := range contexts {
		if province, ok := populationProvinceAnswerInText(parsed.Direction, item.Chunk.Content); ok {
			return fmt.Sprintf("%s人口%s的省份是%s。[%d]", parsed.Scope, parsed.Direction, province, contextIndex+1), true
		}
	}
	return "", false
}

func populationProvinceAnswerInText(direction, text string) (string, bool) {
	patterns, ok := populationProvinceEvidencePatterns[direction]
	if !ok {
		return "", false
	}
	for _, sentence := range sentenceSeparator.Split(text, -1) {
		sentence = strings.TrimSpace(sentence)
		for _, evidence := range patterns {
			if evidence.Pattern.MatchString(sentence) {
				return evidence.Canonical, true
			}
		}
	}
	return "", false
}

func buildPopulationProvinceEvidencePatterns() map[string][]populationProvinceEvidencePattern {
	patterns := make(map[string][]populationProvinceEvidencePattern, 2)
	terms := map[string]string{
		"最多": `(?:最多|最大)`,
		"最少": `(?:最少|最小)`,
	}
	ordinal := map[string]string{
		"最多": `第一大`,
		"最少": `最小`,
	}
	for direction, term := range terms {
		for _, province := range provinceNames {
			aliases := make([]string, 0, len(province.Aliases))
			for _, alias := range province.Aliases {
				aliases = append(aliases, regexp.QuoteMeta(alias))
			}
			entity := `(?:` + strings.Join(aliases, `|`) + `)`
			relation := `(?:中国)?人口` + term + `的省(?:份)?`
			pattern := `(?:` +
				entity + `(?:的)?(?:常住)?人口[^。！？；]{0,48}(?:是|为)` + relation + `|` +
				entity + `(?:是|为)` + relation + `|` +
				entity + `(?:是|为)(?:中国)?人口` + ordinal[direction] + `省|` +
				relation + `(?:是|为)` + entity + `|` +
				`(?:中国)?人口` + ordinal[direction] + `省(?:是|为)` + entity + `|` +
				entity + `(?:的)?(?:常住)?人口` + term + `(?:[，,。]|$)` +
				`)`
			patterns[direction] = append(patterns[direction], populationProvinceEvidencePattern{
				Canonical: province.Canonical,
				Pattern:   regexp.MustCompile(pattern),
			})
		}
	}
	return patterns
}

func normalizeRetrievalQuery(question string) string {
	if subject, ok := capitalQuestionSubject(question); ok {
		return subject + " 首都"
	}
	if parsed, ok := parsePopulationProvinceQuestion(question); ok {
		return strings.Join([]string{parsed.Scope, "人口", parsed.Direction, "省份"}, " ")
	}
	return question
}

func bestEvidenceSentence(question string, contexts []SearchResult) (string, bool) {
	querySet := TokenSet(question)
	bestScore := 0.0
	bestSentence := ""
	bestContext := 0
	for contextIndex, item := range contexts {
		for _, sentence := range sentenceSeparator.Split(item.Chunk.Content, -1) {
			sentence = strings.TrimSpace(sentence)
			length := utf8.RuneCountInString(sentence)
			if length < 4 || length > 240 {
				continue
			}
			score := LexicalScore(querySet, sentence)
			if strings.Contains(sentence, item.Chunk.Title) {
				score += 0.05
			}
			if score > bestScore {
				bestScore = score
				bestSentence = sentence
				bestContext = contextIndex
			}
		}
	}
	if bestSentence == "" || bestScore <= 0 {
		return "", false
	}
	return fmt.Sprintf("%s。[%d]", strings.TrimRight(bestSentence, "。"), bestContext+1), true
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
