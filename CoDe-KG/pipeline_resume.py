"""Durable document storage and deterministic CSV reconstruction."""
import csv
import hashlib
import json
import os
from pathlib import Path

FIELDS = ['doc_id', 'original_text', 'tokenized_text', 'coreference', 'classification',
          'simplification', 'extraction_input', 'relation_extraction', 'configuration']


def read_traces(paths):
    records = {}
    for path in paths:
        for line in Path(path).read_text().splitlines():
            record = json.loads(line)
            if not all(field in record for field in FIELDS):
                raise ValueError(f'Incomplete trace in {path}')
            key = str(record['doc_id'])
            if key in records and records[key] != record:
                raise ValueError(f'Conflicting complete traces for document {key}')
            records[key] = record
    return records


def append_record(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import tempfile
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.trace-')
    try:
        with os.fdopen(fd, 'wb') as handle:
            if path.exists():
                handle.write(path.read_bytes())
            handle.write((json.dumps(record, ensure_ascii=False) + '\n').encode())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)



def reconstruct_csv(records, output):
    """Emit each stored document once, preserving legitimate model duplicates."""
    fields = ['doc_id', 'simple_sentence_index', 'simple_sentence', 'entity_1', 'relationship', 'entity_2']
    temporary = Path(str(output) + '.pending')
    with temporary.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in sorted(records):
            for index, row in enumerate(records[key]['relation_extraction']):
                for triple in row.get('triples', []):
                    writer.writerow(dict(doc_id=key, simple_sentence_index=index,
                        simple_sentence=row['sentence'], entity_1=triple.get('Entity 1'),
                        relationship=triple.get('Relationship'), entity_2=triple.get('Entity 2')))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)


def configuration(args):
    """Fingerprint scientific code and classifier contents without loading models."""
    root = Path(__file__).parent
    paths = [root / name for name in ['prompts.py', 'coref_stage.py', 'simplify_stage.py',
             'relation_extract_stage.py', 'run_pipeline.py', 'llm_runtime.py', 'tokenize_paper.py', 'sentence_split_stage.py', 'bert_sentence_classifier.py']]
    paths += sorted(Path(args.bert_out_dir).rglob('*'))
    hashes = {}
    for path in paths:
        if path.is_file():
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(block)
            hashes[str(path)] = digest.hexdigest()
    return dict(schema=1, seed=args.seed, model=os.environ['LLM_MODEL'],
                gateway=os.environ['LLM_API_BASE'].rstrip('/'), files=hashes,
                coreference_max_tokens=1024, bert_max_length=256)
