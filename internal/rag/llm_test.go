package rag

import (
	"context"
	"testing"
)

func TestExtractiveGeneratorAnswersCapitalQuestion(t *testing.T) {
	generator := ExtractiveGenerator{}
	answer, err := generator.Generate(context.Background(), "中国的首都是哪里？", []SearchResult{
		{Chunk: Chunk{
			Title:   "首都",
			Content: "美国的首都为华盛顿特区，中华人民共和国的首都为北京，经济中心为上海。",
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if answer != "中国的首都是北京。[1]" {
		t.Fatalf("unexpected answer: %q", answer)
	}
}

func TestExtractiveGeneratorAnswersCapitalQuestionParaphrase(t *testing.T) {
	generator := ExtractiveGenerator{}
	answer, err := generator.Generate(context.Background(), "中国的首都是哪一个城市", []SearchResult{
		{Chunk: Chunk{Title: "首都", Content: "中华人民共和国的首都为北京，经济中心为上海。"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if answer != "中国的首都是北京。[1]" {
		t.Fatalf("unexpected answer: %q", answer)
	}
	if query := normalizeRetrievalQuery("中国的首都是哪一个城市"); query != "中国 首都" {
		t.Fatalf("unexpected retrieval query: %q", query)
	}
}

func TestExtractiveGeneratorReturnsBestEvidenceSentence(t *testing.T) {
	generator := ExtractiveGenerator{}
	answer, err := generator.Generate(context.Background(), "报销需要什么材料？", []SearchResult{
		{Chunk: Chunk{Title: "报销流程", Content: "员工应及时提交申请。报销需要提交发票和审批单。"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if answer != "报销需要提交发票和审批单。[1]" {
		t.Fatalf("unexpected answer: %q", answer)
	}
}

func TestExtractiveGeneratorAnswersPopulationProvinceQuestion(t *testing.T) {
	generator := ExtractiveGenerator{}
	contexts := []SearchResult{
		{Chunk: Chunk{
			Title:   "河南省",
			Content: "河南人口位居中国人口最多的省份前列。但如果计算流动人口，广东人口超过1亿，是人口最多的省。",
		}},
	}

	for _, question := range []string{
		"中国人口最多的省份是哪个省",
		"中国哪个省人口最多",
		"中国哪个省的人口最多？",
	} {
		answer, err := generator.Generate(context.Background(), question, contexts)
		if err != nil {
			t.Fatal(err)
		}
		if answer != "中国人口最多的省份是广东省。[1]" {
			t.Fatalf("question %q returned %q", question, answer)
		}
		if query := normalizeRetrievalQuery(question); query != "中国 人口 最多 省份" {
			t.Fatalf("question %q normalized to %q", question, query)
		}
	}
}

func TestPopulationProvinceAnswerRequiresDefiniteEvidence(t *testing.T) {
	if answer, ok := populationProvinceAnswerInText("最多", "河南人口位居中国人口最多的省份前列。"); ok {
		t.Fatalf("unexpected answer from ambiguous evidence: %q", answer)
	}
	answer, ok := populationProvinceAnswerInText("最多", "根据人口普查结果，河南省为人口最大的省份。")
	if !ok || answer != "河南省" {
		t.Fatalf("unexpected answer: %q, ok=%v", answer, ok)
	}
}
