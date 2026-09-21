"""Physical call scheduling; never remove task objects or infer missing facts."""
from collections import OrderedDict


def reserve_atomic_jobs(jobs, *, remaining_calls, downstream_calls, calls_per_job=3):
    """Reserve the complete observation path and downstream closure first.

    A job is (source_index, object_id, field_id). Rotate object/field groups so a
    long early source list cannot consume every slot. Omitted work is explicit.
    Three calls cover extraction, optional scalar binding and verification.
    """
    if min(remaining_calls, downstream_calls) < 0 or calls_per_job < 1:
        raise ValueError("invalid call budget")
    groups = OrderedDict()
    seen = set()
    for job in jobs:
        if job in seen:
            raise ValueError("duplicate atomic job")
        seen.add(job)
        groups.setdefault(job[1:], []).append(job)
    by_object = OrderedDict()
    for key in groups:
        by_object.setdefault(key[0], []).append(key)
    keys = [object_fields[rank] for rank in range(max(map(len, by_object.values()), default=0))
            for object_fields in by_object.values() if rank < len(object_fields)]
    ordered = [groups[key][rank] for rank in range(max(map(len, groups.values()), default=0))
               for key in keys if rank < len(groups[key])]
    capacity = max(0, remaining_calls - downstream_calls) // calls_per_job
    selected, omitted = ordered[:capacity], ordered[capacity:]
    return selected, {"remaining_calls": remaining_calls, "downstream_calls": downstream_calls,
        "calls_per_atomic_job": calls_per_job, "requested_jobs": len(jobs), "scheduled_jobs": len(selected),
        "unexamined_jobs": [{"source_index": i, "object_id": oid, "field_id": fid} for i, oid, fid in omitted],
        "downstream_reservation_fits": remaining_calls >= downstream_calls}
