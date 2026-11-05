# legacy/

These are original scripts from the private evaluation repository. They are copied unchanged except that they were checked for local paths, credentials and embedded contest data, and none was found. **They will not run on their own**, because they expect the contest corpus layout (`videos/`, `features_*/`, `semantic-index/`, `Official-Queries/`), which is not distributed.

| File | What it did |
|---|---|
| `tools/extract_asr_ocr_metadata.py` | Decoded the organizers' per-keyframe ASR/OCR arrays into readable text spans with timestamps |
| `tools/stack_video_embeddings.py` | Stacked per-frame features into `[N, D]` matrices for indexing |
| `tools/semantic_index_pipeline.py`, `tools/semantic_states.py` | Validated and merged worker-written semantic records (scene → claims) into the semantic index |
| `tools/build_corpus_video_hours.py` | ffprobe-based corpus duration inventory (873 videos, 130.7 h) |
| `scripts/benchmark_semantic_index.py` | BM25 benchmark over the semantic index (the clause-decomposition baseline) |
| `scripts/validate_query_source_identity.py`, `scripts/query_validation/` | Checked that each query's source and corpus membership were consistent |
| `reranker_fusion/evaluate_reranker_fusion.py` (+ test) | The frozen 113-query scorer behind the reranker audit. It asserts the expected control numbers. |

Scripts that held hand-written decompositions of official queries, or transcripts, were left out.
