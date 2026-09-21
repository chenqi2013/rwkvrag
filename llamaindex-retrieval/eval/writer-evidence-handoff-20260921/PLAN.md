# Writer evidence handoff — fixed upstream, paired two rounds

User authorized next-stage implementation and model verification. This experiment changes only the Writer execution-provenance handoff, on12/36 fixed-upstream v10 cases with input materials but no final evidence. All24 controls retain byte-identical prompts. Existing raw responses and snapshots remain untouched.

36 cases × baseline/candidate ×2 rounds =144 new calls. Round1 baseline→candidate, round2 candidate→baseline. Same7.2B zero State, audited local FlashRWKV2 0.1.0a13 fp32io16 kernel (FP32 recurrent State,FP16 weights/IO), same engine/model/config/top-1/seed11/2K generation ceiling. Native prompt wrapper and baseline token IDs must match the saved input. Serial calls; no training, retrieval or production changes. The source version and service profile are pinned before running.

Predeclared gates: all144 records saved; all12 target cases in both candidate rounds acknowledge material-processing/evidence-selection limitations without false source-absence, invented facts/citations or repetition/length termination. All24 identical-prompt controls must have no new output drift or semantic regression. Successful abstention is a narrow provenance improvement, NOT successful comparison/retrieval. Human diagnostics by implementer, not independent accuracy. Record baseline vs historical drift and cross-round changes; drift limits causal interpretation. Do not retune in place or suppress failures.

If the gate passes, separately connect the tested handoff to the live application call path and validate integration. If not, keep it experimental and identify the next bottleneck with original evidence.
