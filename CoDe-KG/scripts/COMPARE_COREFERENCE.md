# Coreference-only comparison

From the repository root, generate the offline report without credentials, API
access, model loading, or downloads:

```sh
.venv/bin/python CoDe-KG/scripts/compare_coreference.py
```

Explicitly enable the five-document Qwen comparison:

```sh
.venv/bin/python CoDe-KG/scripts/compare_coreference.py --run
```

The script fixes the inputs and gold to `data/evaluation/pipeline_sample_5.*`.
It prefers the preserved baseline trace in `results/baseline-8b/pipeline/`, with
`outputs/pipeline_sample_5_consolidated/triples.trace.jsonl` as fallback.
All new artifacts are stored in `outputs/coreference_qwen35/`:
`coreference.trace.jsonl`, `failures.jsonl` (when needed), and `comparison.json`.
A later offline invocation rebuilds the report from saved traces.

The existing adapter replaces its constructor's model argument with `LLM_MODEL`.
The script explicitly sets that process environment variable to
`gwdg.qwen3.5-397b-a17b-no-think` before construction, leaving `.env` untouched.
It preserves the baseline prompt, tokenizer, parser, replacement function, and
settings: 1024 maximum new tokens, temperature 0.7, top_p 0.9, do_sample=True.
The existing adapter translates these into max_tokens=1024, temperature=0.7,
and top_p=0.9; top_k and do_sample are not separate gateway request fields.
No local random seed is applied: the API request has no seed parameter.

Completed document traces are skipped on resume after input, settings, model,
and dependency-hash validation. Successful adapter calls use its existing
request-addressed cache, even when a later processing step fails. Cache bypass
is explicitly disabled for this script. Failures are recorded by exception type
without exception messages to avoid leaking credentials; processing continues.
Failed documents are retried on the next --run invocation. Parsed references
and raw answers are retained if replacement fails. No predictions are deduplicated,
filtered, or repaired before the original replacement function.

Diagnostics use zero-based inclusive spans. Span/Expression matching collapses
whitespace and strips leading/trailing non-word characters from each whitespace
token, so `(ADC)` and `ADC,` match `ADC`. Case and internal punctuation are
preserved. RefersTo uses the same normalization. Duplicate full annotations and
duplicate diagnostic tuples are reported separately.

The strict diagnostic compares unique `(StartToken, EndToken, normalized RefersTo)`
sets and reports predicted/gold counts and TP/FP/FN. It is **not the paper's
coreference metric**. Structurally invalid or out-of-range annotations are flagged
and excluded only from diagnostic tuple sets; Expression mismatches remain in
those sets. Gold defects are reported separately. Totals cover completed documents
only, so compare individual documents or the paired completed IDs if coverage
differs. Missing baseline document 394 has no score, rather than a successful
empty prediction.

Offline verification:

```sh
.venv/bin/python -m unittest discover -s tests -p test_compare_coreference.py
.venv/bin/python -m unittest discover -s tests -p test_api_resume.py
```
