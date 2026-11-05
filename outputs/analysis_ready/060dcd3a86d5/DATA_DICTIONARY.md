# Analysis Input Dictionary

One row is one recorded graph-generation attempt, not an API request.
All paths are relative to the project root. Raw evidence stays in its original runs folder.

- primary_member: original fixed study, including its six failures.
- eligible_primary_graph: a valid original graph; use primary_attempts.csv to retain missingness.
- valid_graph: frozen validator accepted the observation; empty graphs can be valid.
- selected_for_896_view: descriptive inventory only, never a pooled primary sample.
- original_run_id: links every supplemental draw to the unchanged primary failed record.
- failure: recorded validation failure, not an unattempted or zero-valued graph.
- *_sha256: byte identities for receipts and artifacts.
- request_receipts: received provider replies used by the observation, not graph count.
- Metrics are copied from verified terminal records, not recalculated statistically here.
  Empty CSV cells mean missing/undefined, never zero-fill. density and clustering are [0,1].
  Primary endpoints: density for global; avg_clustering_coef for the other three methods.
- Supplemental observations are selectively completed and excluded from primary inference.
