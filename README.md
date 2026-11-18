# CoDe-KG Reproduction

Reproduction of [CoDe-KG (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.783/) under limited hardware and API-budget constraints.

Based on the [original implementation](https://github.com/KaushikMahmud/CoDe-KG_EMNLP_2025).

## Approach

The pipeline performs coreference resolution, BERT sentence classification, sentence simplification, and relation extraction.

Generative stages use a university-hosted, OpenAI-compatible API instead of local language models. All generative roles use `LLM_MODEL`, while retaining their stage-specific prompts and generation settings. BERT runs locally.

The API adapter supports caching, bounded retries, and response checks. Requests run sequentially. `top_k` and `repetition_penalty` are not transmitted.

## Results and limitations

- BERT test accuracy: **88.72%**
- BERT test macro-F1: **87.60%**
- Test set: **5,223 sentences**
- Four of five selected abstracts completed the full pipeline across retained attempts, yielding 184 triple rows and 150 distinct within-document triples.
- Historical configuration gaps are documented in the repository.
- These results are a partial reproduction, not a full validation of the paper's reported performance.

A separate workflow compares coreference resolution with a larger Qwen model. It evaluates coreference only, not the complete pipeline. The comparison script and offline tests are included. Qwen outputs were not included in the previously reviewed repository snapshot.

## Installation

```bash
git clone --branch reporduction-api https://github.com/ugurates57/code-kg-reproduction.git
cd code-kg-reproduction
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-mac-api.txt
```

Create `.env` in the repository root:

```dotenv
LLM_API_BASE=YOUR_GATEWAY_BASE_URL
LLM_API_KEY=YOUR_API_KEY
LLM_MODEL=YOUR_MODEL_ID
```

Do not commit API credentials. University network or VPN access may be required.

## Running

Inspect the pipeline arguments:

```bash
python CoDe-KG/run_pipeline.py --help
```

BERT training is handled separately by `CoDe-KG/scripts/train_bert.py`. The training script requires Apple MPS.

For the larger-model coreference comparison, see:

- `CoDe-KG/scripts/compare_coreference.py`
- `CoDe-KG/scripts/COMPARE_COREFERENCE.md`

Run the offline comparison tests:

```bash
python -m unittest discover -s tests -p 'test_compare_coreference.py'
```

## Repository structure

- `CoDe-KG/`: pipeline, API adapter, and experiment scripts
- `data/` and `datasets/`: data files
- `results/` and `outputs/`: saved experiment artifacts and traces
- `tests/`: automated checks
- `requirements-mac-api.txt`: reproduction dependencies

## Attribution

The original method, prompts, and upstream implementation belong to the CoDe-KG authors. This repository documents the reproduction adaptations and available experimental evidence. See `LICENSE` for license terms.