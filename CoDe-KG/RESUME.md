# API recovery and fixed-sample continuation

Run commands from the repository root. No training is required.

```sh
.venv/bin/python CoDe-KG/run_pipeline.py --input_csv data/evaluation/pipeline_sample_5.csv --id_col doc_id --out_csv outputs/pipeline_sample_5_consolidated/triples.csv --accept_legacy_provenance
```

The prepared consolidated directory contains copies of complete traces for 168
and 313 and a CSV reconstructed offline. Original files are unchanged.
`provenance.json` records source hashes, the inspected attempt directories and
the reported (not independently logged) timeout on 315. No additional complete
traces or failure logs were found in the other named attempt directories.
Legacy records lack gateway and token-limit metadata. Their exact configuration
cannot be verified. Relation extraction reportedly used 512 and later 2048 tokens;
no per-record attribution is asserted. `--accept_legacy_provenance` explicitly
accepts reuse of these historical results, preserving their original metadata.
It does not override text conflicts or differing modern configuration hashes.
Multiple differing complete records for one ID cause an error.

New records fingerprint prompts, stage code, adapter, classifier files, seed,
gateway and actual model. Complete documents are skipped only after validating
input and configuration. New traces are atomically replaced and fsynced after
each document. Each successful API response is saved before the next request.
An interrupted document replays its computation using cached responses; local
BERT classification may run again. Previously unsaved API responses cannot be
recovered. Gold annotations are never read by this workflow.

Cache files are under `outputs/llm_cache` (override with `LLM_CACHE_DIR`). Keys
cover schema version, actual model, gateway, complete messages and every sent
generation parameter. API credentials are excluded. Only nonempty `stop`
responses are cached. This deliberately reuses sampled responses for recovery;
it is not an independent experimental replicate. Use a new output directory and
`--bypass_cache` for an independent run. Bypass neither reads nor writes cache.
`top_k` and `repetition_penalty` remain unsent; batches remain sequential.

The adapter owns retries (SDK retries disabled): at most five attempts per
request, seven seconds before each network call, exponential transient backoff,
and 65 seconds for 429 unless valid Retry-After seconds/date are supplied.
Retry-After above one hour fails the document rather than retrying too early.
Timeout retries may repeat work already processed and may incur additional cost.
Authentication/configuration errors terminate the run. Truncation is a distinct
non-retried document error; token limits are unchanged. Other non-stop or empty
responses fail visibly. Content errors in regularly terminated responses remain
visible to the existing parsers; prompts, parsers and weights are unchanged.

All three generative stages are covered by the document error boundary. Exhausted
transient failures and invalid/truncated responses produce `.failures.jsonl`
records containing ID, stage, error type, UTC timestamp and a sanitized message.
They do not produce successful empty results. Unexpected programming errors
propagate. Counts of completed, failed and skipped documents are printed.
Rerunning retries failed documents; successful requests use cache.

Rebuild the CSV without constructing an API client or loading BERT:

```sh
.venv/bin/python CoDe-KG/run_pipeline.py --input_csv data/evaluation/pipeline_sample_5.csv --id_col doc_id --out_csv outputs/pipeline_sample_5_consolidated/triples.csv --rebuild_csv_only
```

To import other complete traces, repeat `--import_trace PATH` with an explicit
`--accept_legacy_provenance` for old metadata. Identical repeated inputs are
idempotent. Different complete records conflict. Legitimate repeated triples
within a model response are preserved; CSV reconstruction introduces no copies.
Keep source traces and failure logs as provenance. CSV is a derived file and is
atomically regenerated. Use one writer per output/cache directory.

Offline checks (no model download or API call):

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q CoDe-KG tests
```
