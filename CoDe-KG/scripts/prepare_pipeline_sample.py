import ast
import json
import random
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from tokenize_paper import tokens_with_indices

SOURCE = REPO_DIR / "datasets/coref_human_gold_annotation.json"
OUT_DIR = REPO_DIR / "data/evaluation"
INPUT_PATH = OUT_DIR / "pipeline_sample_5.csv"
GOLD_PATH = OUT_DIR / "pipeline_sample_5_gold.json"

SEED = 4000
SAMPLE_SIZE = 5


def main():
    if INPUT_PATH.exists() or GOLD_PATH.exists():
        raise FileExistsError(
            "Sample files already exist. Keep the existing fixed sample."
        )

    documents = json.loads(SOURCE.read_text(encoding="utf-8"))

    # Select documents before inspecting model outputs.
    rng = random.Random(SEED)
    indices = sorted(rng.sample(range(len(documents)), SAMPLE_SIZE))

    rows = []
    gold = []

    for index in indices:
        document = documents[index]

        # Parse token pairs without executing their contents.
        pairs = ast.literal_eval("[" + document["new_text"] + "]")

        if not all(
            isinstance(pair, tuple)
            and len(pair) == 2
            and isinstance(pair[0], str)
            and type(pair[1]) is int
            for pair in pairs
        ):
            raise ValueError("Unexpected token format.")

        if [position for _, position in pairs] != list(range(len(pairs))):
            raise ValueError("Token indices are not consecutive.")

        text = " ".join(token for token, _ in pairs)

        # Preserve token boundaries so gold indices remain meaningful.
        reconstructed = tokens_with_indices(text, keep_punct=True)
        if list(reconstructed) != list(pairs):
            raise ValueError(
                f"Token reconstruction differs for {document['doc_id']}."
            )

        rows.append({
            "doc_id": str(document["doc_id"]),
            "abstract": text,
        })
        gold.append(document)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(INPUT_PATH, index=False)
    GOLD_PATH.write_text(
        json.dumps(
            {
                "source": str(SOURCE.relative_to(REPO_DIR)),
                "seed": SEED,
                "source_indices": indices,
                "documents": gold,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved input: {INPUT_PATH}")
    print(f"Saved gold annotations: {GOLD_PATH}")

    for row in rows:
        print(f"\nDocument {row['doc_id']}:")
        print(row["abstract"][:250])


if __name__ == "__main__":
    main()