"""Data admission checks must fail closed before any State optimizer update."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_candidates import audit, digest, normalize


def row(identity, split, role="plan", family="repo:new/project", question="比较甲和乙的部署条件"):
    prompt = "User: " + question + "\n\nAssistant: <think></think>"
    target = '{"objects":["甲","乙"]}'
    return {"id": identity, "split": split, "role": role, "kind": "comparison",
            "question": question, "prompt": prompt, "target": target,
            "prompt_sha256": digest(prompt), "target_sha256": digest(target),
            "source_families": [family], "source_hashes": [digest(identity)],
            "review": {"accepted": True, "independent_of_author": True,
                       "author": "teacher", "reviewer": "human", "reason": "checked"}}


MINIMUMS = {"train": 1, "roles": {"plan": 1}, "kinds": {"comparison": 1},
            "train_families": 1, "max_family_fraction": 1,
            "min_unique_question_fraction": 0.5, "max_question_repetitions": 3}


class AdmissionAuditTest(unittest.TestCase):
    def test_clean_source_separated_rows(self):
        rows = [row("a", "train"), row("b", "dev", family="repo:other/project", question="乙的来源在哪里")]
        report = audit(rows, {}, MINIMUMS)
        self.assertTrue(report["admitted"], report["issues"])

    def test_evaluation_overlap_and_cross_split_source_blocked(self):
        rows = [row("a", "train"), row("b", "heldout", question="其他问题")]
        registry = {"question_hashes": [digest(normalize(rows[0]["question"]))]}
        report = audit(rows, registry, MINIMUMS)
        self.assertFalse(report["admitted"])
        self.assertTrue(any("evaluation material" in issue for issue in report["issues"]))
        self.assertTrue(any("source family crosses" in issue for issue in report["issues"]))

    def test_unreviewed_hash_mismatch_and_empty_evidence_blocked(self):
        bad = row("a", "train", role="evidence")
        bad["source_hashes"] = []
        bad["prompt_sha256"] = "0" * 64
        bad["review"]["reviewer"] = bad["review"]["author"]
        report = audit([bad], {}, {**MINIMUMS, "roles": {"evidence": 1}})
        self.assertFalse(report["admitted"])
        self.assertGreaterEqual(len(report["issues"]), 3)

    def test_repeated_intent_is_not_counted_as_diversity(self):
        rows = [row(str(i), "train", family=f"repo:new/{i}") for i in range(4)]
        report = audit(rows, {}, {**MINIMUMS, "min_unique_question_fraction": 0.9})
        self.assertFalse(report["admitted"])
        self.assertIn("normalized question uniqueness below minimum", report["issues"])
        self.assertTrue(any("repeated too often" in issue for issue in report["issues"]))


if __name__ == "__main__":
    unittest.main()
