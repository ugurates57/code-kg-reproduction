# run_pipeline.py
import os
import argparse
import pandas as pd

import json
from pathlib import Path

from tokenize_paper import tokens_with_indices, format_pairs
from llm_runtime import LLM, set_seed, GenerationError
from pipeline_resume import read_traces, append_record, reconstruct_csv, configuration
from datetime import datetime, timezone
from coref_stage import build_coref_prompt, parse_coref_output, apply_coref_switch
from sentence_split_stage import split_sentences
from bert_sentence_classifier import BertSentenceClassifier, BertClassifierConfig
from simplify_stage import simplify_sentences_for_class
from relation_extract_stage import extract_relationships

MIXTRAL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
LLAMA_31_8B = "meta-llama/Llama-3.1-8B-Instruct"
LLAMA_33_70B = "meta-llama/Llama-3.3-70B-Instruct"

def read_abstracts(args) -> pd.DataFrame:
    if args.input_csv:
        df = pd.read_csv(args.input_csv)
        if args.text_col not in df.columns:
            raise ValueError(f"--text_col '{args.text_col}' not found in {args.input_csv}. Columns: {list(df.columns)}")
        df = df.reset_index(drop=True)
        if args.id_col and args.id_col in df.columns:
            df["doc_id"] = df[args.id_col].astype(str)
        else:
            df["doc_id"] = [f"doc_{i}" for i in range(len(df))]
        df["abstract"] = df[args.text_col].astype(str)
        return df[["doc_id", "abstract"]]

    if args.abstract:
        return pd.DataFrame([{"doc_id": "doc_0", "abstract": args.abstract}])

    # stdin
    text = input("Enter abstract: ").strip()
    return pd.DataFrame([{"doc_id": "doc_0", "abstract": text}])

def main():
    # Read arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, default=None, help="CSV containing abstracts")
    parser.add_argument("--text_col", type=str, default="abstract", help="Column containing abstract text")
    parser.add_argument("--id_col", type=str, default=None, help="Optional doc id column")

    parser.add_argument("--abstract", type=str, default=None, help="Single abstract (if not using --input_csv)")
    parser.add_argument("--out_csv", type=str, required=True, help="Output CSV path")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=4000)

    # BERT
    parser.add_argument("--bert_out_dir", type=str, default="models/bert_sentence_classifier")
    parser.add_argument("--train_bert_if_missing", action="store_true")
    parser.add_argument("--bert_train_csv", type=str, default="train.csv")
    parser.add_argument("--bert_text_col", type=str, default="sentence")
    parser.add_argument("--bert_label_col", type=str, default="label")
    parser.add_argument("--bert_epochs", type=int, default=3)
    parser.add_argument("--bert_train_batch_size", type=int, default=8)
    parser.add_argument("--bert_lr", type=float, default=2e-5)

    parser.add_argument('--import_trace', action='append', default=[])
    parser.add_argument('--accept_legacy_provenance', action='store_true')
    parser.add_argument('--rebuild_csv_only', action='store_true')
    parser.add_argument('--bypass_cache', action='store_true')
    args = parser.parse_args()
    # Prepare output paths before making API requests.
    output_path = Path(args.out_csv)
    trace_path = output_path.with_suffix(".trace.jsonl")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read abstracts from CSV
    df_abs = read_abstracts(args)

    if df_abs['doc_id'].duplicated().any():
        raise ValueError('Duplicate document IDs in input')
    paths = ([trace_path] if trace_path.exists() else []) + args.import_trace
    completed = read_traces(paths)
    inputs = dict(zip(df_abs.doc_id.astype(str), df_abs.abstract))
    for key, record in completed.items():
        if key not in inputs or inputs[key] != record['original_text']:
            raise ValueError(f'Trace/input conflict for {key}')
    if args.rebuild_csv_only:
        reconstruct_csv(completed, output_path)
        return
    if args.bypass_cache:
        os.environ['LLM_CACHE_BYPASS'] = '1'

    # Initialize LLMs
    mixtral_llm = LLM(MIXTRAL)
    llama8_llm = LLM(LLAMA_31_8B)
    llama70_llm = LLM(LLAMA_33_70B)

    # Check for configuration conflicts
    run_configuration = configuration(args)
    for key, record in completed.items():
        metadata = record.get('resume_configuration')
        if metadata is not None and metadata != run_configuration:
            raise ValueError(f'Configuration conflict for {key}')
        if metadata is None and not args.accept_legacy_provenance:
            raise ValueError('Legacy trace requires --accept_legacy_provenance')
    existing = read_traces([trace_path]) if trace_path.exists() else {}
    for key, record in completed.items():
        if key not in existing:
            append_record(trace_path, record)
    set_seed(args.seed)

    # BERT classifier (train once if needed)
    bert_cfg = BertClassifierConfig(
        out_dir=args.bert_out_dir,
        max_length=256,
    )
    bert = BertSentenceClassifier(bert_cfg)
    if bert.exists():
        bert.load()
    else:
        if args.train_bert_if_missing:
            bert.train_if_missing(
                train_csv=args.bert_train_csv,
                text_col=args.bert_text_col,
                label_col=args.bert_label_col,
                epochs=args.bert_epochs,
                batch_size=args.bert_train_batch_size,
                lr=args.bert_lr,
            )
        else:
            raise ValueError(f"No finetuned BERT found at {args.bert_out_dir}. Use --train_bert_if_missing.")

    succeeded = failed = skipped = 0

    # Process each abstract
    for _, row in df_abs.iterrows():
        doc_id = row["doc_id"]
        abstract = row["abstract"]

        if str(doc_id) in completed:
            skipped += 1
            continue
        try:
            stage = "coreference"
            # 1) Tokenize as in paper
            pairs = tokens_with_indices(abstract, keep_punct=True)
            tokenized_text = format_pairs(pairs)

            # 2) Coref with Mixtral
            coref_prompt = build_coref_prompt(tokenized_text)
            coref_out = mixtral_llm.generate_batch(
                prompts=[coref_prompt], max_new_tokens=1024, batch_size=1,
                temperature=0.7, top_k=40, top_p=0.9, do_sample=True,
                return_full_text=False,
            )[0][0]['generated_text'].strip()

            # LLM returns something like: {
            #    "Expression": "She",
            #    "StartToken": 4,
            #    "EndToken": 4,
            #    "RefersTo": "Anna"
            #    }

            # read the coref output into a structured format
            corefs = parse_coref_output(coref_out)

            # 3) Switch coref into text (PBC -> Primary biliary cirrhosis at indices)
            switched = apply_coref_switch(tokenized_text, corefs)

            # 4) Sentence split (NLTK)
            sentences = split_sentences(switched)

            # 5) Use BERT to classify sentences into Simple, Compound, Complex, Compound-Complex or Incomplete.
            # Locally trained model is used, up to 4 sentences per batch for efficiency.
            preds = bert.predict_batch(sentences, batch_size=4)
            cls_df = pd.DataFrame(preds)
            print("\nSentence classification:")
            print(cls_df.to_string(index=False))

            # classifier returns something like this: {
            #    "sentence": "Anna is a researcher.",
            #    "predicted_label": "Simple Sentence",
            #    "confidence": 0.97
            # }

            # sort sentences into lists by predicted label for simplification
            simple_sents = cls_df[cls_df["predicted_label"] == "Simple Sentence"]["sentence"].tolist()
            compound_sents = cls_df[cls_df["predicted_label"] == "Compound Sentence"]["sentence"].tolist()
            complex_sents = cls_df[cls_df["predicted_label"] == "Complex Sentence"]["sentence"].tolist()
            cc_sents = cls_df[cls_df["predicted_label"] == "Compound-Complex Sentence"]["sentence"].tolist()

            print(f"[DOC {doc_id}] Label counts → "
            f"Simple={len(simple_sents)} | "
            f"Compound={len(compound_sents)} | "
            f"Complex={len(complex_sents)} | "
            f"Compound-Complex={len(cc_sents)}")

            # 6) Simplify (route to correct LLM)
            stage = "simplification"
            simplified_rows = []
            # Choose the appropriate LLM for each sentence type and simplify in batches
            # In my reproduction: only prompts differ, same Mixtral model is used for all sentence types
            # Simple sentences skip this step
            if compound_sents:
                simplified_rows += simplify_sentences_for_class(llama8_llm, "Compound Sentence", compound_sents, args.batch_size)
            if complex_sents:
                simplified_rows += simplify_sentences_for_class(llama70_llm, "Complex Sentence", complex_sents, args.batch_size)
            if cc_sents:
                simplified_rows += simplify_sentences_for_class(mixtral_llm, "Compound-Complex Sentence", cc_sents, args.batch_size)

            converted_simple = []
            # Combine the simplified sentences with the original simple sentences for relationship extraction
            for r in simplified_rows:
                converted_simple.extend(r.get("simplified_sentences") or [])
            final_simple_sentences = simple_sents + converted_simple

            print(f"[DOC {doc_id}] Simplify input counts: "
            f"Compound={len(compound_sents)} | "
            f"Complex={len(complex_sents)} | "
            f"Compound-Complex={len(cc_sents)}")

            print(f"[DOC {doc_id}] Relation extractor input simple sentences: {len(final_simple_sentences)}")



            # 7) Relationship extraction with Mixtral
            stage = "relation_extraction"
            # Build extraction prompt for every simple sentence and run in batches. The LLM returns a JSON-like string for each sentence.
            rel_rows = extract_relationships(mixtral_llm, final_simple_sentences, batch_size=args.batch_size)
            # Save intermediate results for later inspection.
            trace = {
                "doc_id": str(doc_id),
                "original_text": abstract,
                "tokenized_text": tokenized_text,
                "coreference": {
                    "raw_output": coref_out,
                    "parsed_references": corefs,
                    "resolved_text": switched,
                },
                "classification": preds,
                "simplification": simplified_rows,
                "extraction_input": final_simple_sentences,
                "relation_extraction": rel_rows,
                "configuration": {
                    "seed": args.seed,
                    "bert_directory": str(args.bert_out_dir),
                    "bert_max_length": bert_cfg.max_length,
                    "coreference_model": mixtral_llm.model_name,
                    "compound_model": llama8_llm.model_name,
                    "complex_model": llama70_llm.model_name,
                    "compound_complex_model": mixtral_llm.model_name,
                    "relation_model": mixtral_llm.model_name,
                },
            }

            trace['resume_configuration'] = run_configuration
            append_record(trace_path, trace)
            completed[str(doc_id)] = trace
            succeeded += 1
            print(f"Saved document trace: {trace_path}")

        except GenerationError as exc:
            failed += 1
            append_record(output_path.with_suffix('.failures.jsonl'), {
                'doc_id': str(doc_id), 'stage': stage,
                'error_type': type(exc).__name__, 'timestamp': datetime.now(timezone.utc).isoformat(),
                'message': str(exc),
            })
            print(f'[DOC {doc_id}] {stage}: {exc}; continuing', flush=True)
            
    # Write the final CSV with all completed records
    reconstruct_csv(completed, output_path)
    print(f'Completed={succeeded} failed={failed} skipped={skipped}; CSV: {output_path}')

if __name__ == "__main__":
    main()
