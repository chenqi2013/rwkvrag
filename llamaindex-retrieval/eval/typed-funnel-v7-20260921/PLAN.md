# V7 single typed value protocol

V6 completed all13 pairs: 8/13 exact. Boolean false was correct but redundant known_value flag copied false, rejecting valid negatives; count units omitted, explicit unrecorded evidence failed the empty-provenance constraint. Previous versions and outputs remain unchanged.

Only atomic protocol changes: remove redundant truth flag; nullable typed value is the fact. Quantity copies a single verbatim scalar including unit; code only parses decimal syntax into number and unit without arithmetic/conversion. Null may retain evidence explicitly stating missing data. Same 13 inputs/gold, v4 paired baseline, model/State/budgets512/seed/top1/concurrency1. Same gate >=12/13, both zero and allboolean cases correct, unknown not invented. This is one protocol bundle, not a single-word attribution.

Save all traces and wrong outputs. No production promotion based on this seen node regression.
