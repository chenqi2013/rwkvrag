# Clause query-last controlled test

V3 raw traces show clause interpretation repeatedly summarizes the whole conversation rather than the specified fragment. Its prompt places the target fragment before the full conversation.

Change exactly the order of those two data blocks in the clause prompt: whole conversation first, target fragment last. All instructions, schemas, message bytes, splitting, validators, model/State/decoding and36 cases remain unchanged. No claim of a proven RWKV internal mechanism; this tests a recency hypothesis. Preserve all failures and metadata. The overview/intent/type calls remain the same protocol. Run after v3 completes, no overlapping traffic. Review targets vs whole-conversation contamination, hard/withdrawn states and ordinary-question fields.
