import pytest
from llamaindex_retrieval.funnel_call_budget import reserve_atomic_jobs


def test_many_atomic_jobs_cannot_starve_verification_or_final_stages():
    jobs = [(source, oid, fid) for source in range(34) for oid in ("O1", "O2", "O3", "O4", "O5") for fid in ("F1", "F2", "F3", "F4")]
    selected, audit = reserve_atomic_jobs(jobs, remaining_calls=476, downstream_calls=60)
    assert len(selected) * 3 + 60 <= 476
    assert len({job[1] for job in selected[:5]}) == 5
    assert {job[1:] for job in selected[:20]} == {job[1:] for job in jobs}
    assert len(selected) + len(audit["unexamined_jobs"]) == len(jobs)
    assert audit["downstream_reservation_fits"]


def test_small_case_preserves_all_jobs_and_explicitly_records_insufficient_budget():
    jobs = [(0, "O1", "F1"), (1, "O1", "F1")]
    assert reserve_atomic_jobs(jobs, remaining_calls=20, downstream_calls=10)[0] == jobs
    selected, audit = reserve_atomic_jobs(jobs, remaining_calls=9, downstream_calls=10)
    assert selected == [] and not audit["downstream_reservation_fits"]
    assert len(audit["unexamined_jobs"]) == 2


def test_duplicate_jobs_rejected_without_silently_changing_work():
    with pytest.raises(ValueError):
        reserve_atomic_jobs([(0,"O1","F1")]*2, remaining_calls=20, downstream_calls=10)
