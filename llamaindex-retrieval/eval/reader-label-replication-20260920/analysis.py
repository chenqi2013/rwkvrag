"""Preregistered paired scoring; repeats are stability checks, not extra samples."""
from collections import defaultdict
import random
from statistics import mean

from protocol import metrics


def correct(row):
    return row["status"] == "valid" and row["prediction"] == row["expected"]


def summarize(rows, cases, bootstrap_draws=10000):
    expected_keys = {(r, c["id"], arm) for r in (1, 2) for c in cases
                     for arm in ("yes-no", "neutral-labels")}
    keyed = {(r["round"], r["case_id"], r["arm"]): r for r in rows}
    if len(keyed) != len(rows) or set(keyed) != expected_keys:
        raise ValueError("incomplete_or_duplicate_scheduled_results")
    rounds = {}
    families = defaultdict(list)
    cases_by_id = {c["id"]: c for c in cases}
    for c in cases:
        families[c["family"]].append(c["id"])
    for round_id in (1, 2):
        base = [keyed[round_id, c["id"], "yes-no"] for c in cases]
        candidate = [keyed[round_id, c["id"], "neutral-labels"] for c in cases]
        deltas = {c["id"]: int(correct(b)) - int(correct(a)) for c, a, b in zip(cases, base, candidate)}
        family_deltas = {f: mean(deltas[i] for i in ids) for f, ids in families.items()}
        rng = random.Random(20260920)
        names = sorted(families)
        # All families have the same size; resampling whole families keeps variants together.
        assert len({len(ids) for ids in families.values()}) == 1
        samples = sorted(mean(family_deltas[rng.choice(names)] for _ in names) for _ in range(bootstrap_draws))
        interval = [samples[int(.025 * (bootstrap_draws - 1))], samples[int(.975 * (bootstrap_draws - 1))]]
        rounds[str(round_id)] = {
            "baseline": metrics(base), "candidate": metrics(candidate),
            "accuracy_delta": mean(deltas.values()),
            "recovered": sum(d == 1 for d in deltas.values()),
            "regressed": sum(d == -1 for d in deltas.values()),
            "cluster_bootstrap_95_percentile_interval": interval,
            "family_accuracy_deltas": family_deltas,
            "leave_one_family_out_min_delta": min(mean(v for f, v in family_deltas.items() if f != leave) for leave in names),
            "categories": {category: {
                "baseline": metrics([r for r in base if cases_by_id[r["case_id"]]["category"] == category]),
                "candidate": metrics([r for r in candidate if cases_by_id[r["case_id"]]["category"] == category])}
                for category in sorted({c["category"] for c in cases})}}
    stability = {}
    for arm in ("yes-no", "neutral-labels"):
        changed = []
        for c in cases:
            a, b = keyed[1, c["id"], arm], keyed[2, c["id"], arm]
            if (a["status"], a["prediction"]) != (b["status"], b["prediction"]):
                changed.append(c["id"])
        stability[arm] = {"n": len(cases), "changed_cases": changed, "agreement": 1 - len(changed) / len(cases)}
    first = rounds["1"]
    second = rounds["2"]
    criteria = {
        "round1_accuracy_gain_at_least_5_points": first["accuracy_delta"] >= .05,
        "round1_cluster_interval_above_zero": first["cluster_bootstrap_95_percentile_interval"][0] > 0,
        "round1_not_driven_by_one_family": first["leave_one_family_out_min_delta"] > 0,
        "no_more_false_positives_in_either_round": all(r["candidate"]["fp"] <= r["baseline"]["fp"] for r in rounds.values()),
        "no_more_invalid_outputs_in_either_round": all(r["candidate"]["invalid"] <= r["baseline"]["invalid"] for r in rounds.values()),
        "repeat_accuracy_gain_positive": second["accuracy_delta"] > 0,
        "each_arm_repeat_agreement_at_least_99_percent": all(s["agreement"] >= .99 for s in stability.values()),
    }
    return {"unique_cases": len(cases), "template_families": len(families), "calls": len(rows),
            "rounds": rounds, "stability": stability, "criteria": criteria,
            "limited_experimental_improvement_gate": all(criteria.values()),
            "independent_label_review": False, "production_readiness_claim": False,
            "uncertainty_scope": "conditional on these synthetic template families; not population generalization"}
