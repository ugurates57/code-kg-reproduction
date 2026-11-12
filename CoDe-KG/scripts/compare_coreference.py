"""Coreference-only comparison; offline by default, API access requires --run."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'CoDe-KG'))
from tokenize_paper import tokens_with_indices, format_pairs
from coref_stage import build_coref_prompt, parse_coref_output, apply_coref_switch

MODEL = 'gwdg.qwen3.5-397b-a17b-no-think'
BASELINE_MODEL = 'gwdg.meta-llama-3.1-8b-instruct'
SETTINGS = dict(max_new_tokens=1024, temperature=0.7, top_p=0.9,
                do_sample=True, top_k=40, return_full_text=False, batch_size=1)


def normalize(value):
    """Collapse whitespace and strip punctuation at each whitespace token's edges.

    Preserve case and internal punctuation (e.g. disability-adjusted). This
    accommodates source tokens such as '(ADC)' versus the expression 'ADC'.
    """
    return ' '.join(filter(None, (re.sub(r'^\W+|\W+$', '', t) for t in value.split())))


def diagnostics(annotations, text):
    tokens = [t for t, _ in tokens_with_indices(text, keep_punct=True)]
    invalid, mismatches, tuples = [], [], []
    identities = []
    for index, annotation in enumerate(annotations):
        identities.append(json.dumps(annotation, sort_keys=True, ensure_ascii=False))
        start, end = annotation.get('StartToken'), annotation.get('EndToken')
        if (type(start) is not int or type(end) is not int or
                not 0 <= start <= end < len(tokens) or
                not isinstance(annotation.get('Expression'), str) or
                not isinstance(annotation.get('RefersTo'), str)):
            invalid.append(dict(index=index, annotation=annotation))
            continue
        source = ' '.join(tokens[start:end + 1])
        if normalize(source) != normalize(annotation['Expression']):
            mismatches.append(dict(index=index, source=source, annotation=annotation))
        tuples.append((start, end, normalize(annotation['RefersTo'])))
    return dict(annotation_count=len(annotations),
                duplicate_annotations=sum(n - 1 for n in Counter(identities).values()),
                duplicate_tuples=len(tuples) - len(set(tuples)),
                invalid_annotations=invalid, source_expression_mismatches=mismatches), set(tuples)


def load_records(path):
    records = {}
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            record = json.loads(line)
            key = str(record['doc_id'])
            if key in records and records[key] != record:
                raise ValueError(f'Conflicting traces for document {key}')
            records[key] = record
    return records


def compare(inputs, gold, baseline, comparison):
    report = dict(
        metric='Strict unique (start, end, normalized RefersTo) diagnostic; NOT the paper coreference metric.',
        normalization=normalize.__doc__,
        span_policy='Zero-based inclusive indices. Invalid spans are excluded only from diagnostic tuple sets. Expression mismatches remain in tuple sets. Predictions passed unchanged to apply_coref_switch.',
        missing_baseline_394='394' not in baseline,
        gold={}, models={})
    gold_sets = {}
    for key, text in inputs.items():
        report['gold'][key], gold_sets[key] = diagnostics(gold[key], text)
    for name, records in [('baseline', baseline), ('comparison', comparison)]:
        documents = {}
        totals = Counter()
        for key, text in inputs.items():
            if key not in records:
                documents[key] = dict(status='missing', strict_diagnostic=None)
                continue
            record = records[key]
            if record['original_text'] != text:
                raise ValueError(f'Trace/input mismatch for {key}')
            diag, predicted = diagnostics(record['coreference']['parsed_references'], text)
            target = gold_sets[key]
            counts = dict(predicted_unique=len(predicted), gold_unique=len(target),
                          true_positive=len(predicted & target), false_positive=len(predicted - target),
                          false_negative=len(target - predicted))
            totals.update(counts)
            documents[key] = dict(status='completed', diagnostics=diag, strict_diagnostic=counts,
                                  configuration=record.get('configuration'))
        report['models'][name] = dict(completed_documents=sum(k in records for k in inputs),
            missing_documents=[k for k in inputs if k not in records], documents=documents,
            strict_diagnostic_completed_documents_only=dict(totals))
    common = [k for k in inputs if k in baseline and k in comparison]
    report['paired_completed_document_ids'] = common
    report['aggregation_note'] = 'Totals cover completed documents only; compare document-level counts or paired IDs when coverage differs. Missing documents are not empty predictions.'
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Explicitly enable API calls for all five documents')
    args = parser.parse_args(argv)
    with (ROOT / 'data/evaluation/pipeline_sample_5.csv').open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    inputs = {r['doc_id']: r['abstract'] for r in rows}
    if len(inputs) != 5 or len(rows) != 5:
        raise ValueError('Expected five unique input documents')
    gold = {str(d['doc_id']): d['annotations'] for d in json.loads(
        (ROOT / 'data/evaluation/pipeline_sample_5_gold.json').read_text())['documents']}
    baseline_path = ROOT / 'results/baseline-8b/pipeline/triples.trace.jsonl'
    if not baseline_path.exists():
        baseline_path = ROOT / 'outputs/pipeline_sample_5_consolidated/triples.trace.jsonl'
    baseline = load_records(baseline_path)
    for record in baseline.values():
        if record['configuration']['coreference_model'] != BASELINE_MODEL:
            raise ValueError('Unexpected baseline model')
    output = ROOT / 'outputs/coreference_qwen35'
    output.mkdir(parents=True, exist_ok=True)
    trace = output / 'coreference.trace.jsonl'
    records = load_records(trace)
    # Fingerprint scientific dependencies to reject incompatible resumes.
    configuration = dict(model=MODEL, settings=SETTINGS, files={name: hashlib.sha256(
        (ROOT / 'CoDe-KG' / name).read_bytes()).hexdigest() for name in
        ['coref_stage.py', 'tokenize_paper.py', 'prompts.py', 'llm_runtime.py']})
    for key, record in records.items():
        if record['configuration'] != configuration or inputs.get(key) != record['original_text']:
            raise ValueError(f'Incompatible resume for document {key}')
    if args.run:
        from llm_runtime import LLM
        from pipeline_resume import append_record
        # LLM.__init__ overrides its argument with LLM_MODEL; load_dotenv does
        # not override these explicit process settings. Never modify .env.
        os.environ['LLM_MODEL'] = MODEL
        os.environ['LLM_CACHE_BYPASS'] = '0'
        llm = LLM(MODEL)
        if llm.model_name != MODEL:
            raise ValueError('Effective model differs from requested comparison model')
        for key, text in inputs.items():
            if key in records:
                continue
            tokenized = format_pairs(tokens_with_indices(text, keep_punct=True))
            record = dict(doc_id=key, original_text=text, tokenized_text=tokenized,
                          configuration=configuration, gateway=llm.base_url)
            try:
                raw = llm.generate_batch([build_coref_prompt(tokenized)], **SETTINGS)[0][0]['generated_text'].strip()
                references = parse_coref_output(raw)
                record['coreference'] = dict(raw_output=raw, parsed_references=references)
                # Preserve the original processing method, including duplicates
                # and invalid predictions; diagnostics never alter this input.
                record['coreference']['resolved_text'] = apply_coref_switch(tokenized, references)
                append_record(trace, record)
                records[key] = record
            except Exception as exc:
                # Avoid persisting exception messages that could contain secrets.
                append_record(output / 'failures.jsonl', dict(record, error_type=type(exc).__name__))
                print(f'Document {key} failed: {type(exc).__name__}; continuing')
    report = compare(inputs, gold, baseline, records)
    report['baseline_trace'] = str(baseline_path.relative_to(ROOT))
    # Offline reporting uses only the standard library and existing stage code.
    (output / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for name, result in report['models'].items():
        print(f"{name}: completed={result['completed_documents']}/5; missing={result['missing_documents']}")
    print(f"Strict diagnostic (not paper metric): {output / 'comparison.json'}")


if __name__ == '__main__':
    main()
