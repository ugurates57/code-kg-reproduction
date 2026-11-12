import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_DIR / "models" / "bert_sentence_classifier"
TEST_PATH = REPO_DIR / "data" / "evaluation" / "bert_test_clean.csv"
OUT_DIR = REPO_DIR / "outputs" / "bert_test"

MAX_LENGTH = 256
BATCH_SIZE = 4


def main():
    if not TEST_PATH.exists():
        raise FileNotFoundError("Create bert_test_clean.csv first.")

    if OUT_DIR.exists():
        raise FileExistsError(
            "Test output already exists. Rename it before another run."
        )

    df = pd.read_csv(TEST_PATH)
    label_map = None

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR, local_files_only=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        attn_implementation="eager",
    )

    label_map = model.config.label2id
    names = sorted(label_map, key=label_map.get)

    if not set(df["label"]).issubset(label_map):
        raise ValueError("Test data contains unknown labels.")

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model.to(device)
    model.eval()

    print("Device:", device)
    print("Test rows:", len(df))

    predictions = []
    truncated_rows = 0

    with torch.inference_mode():
        for start in range(0, len(df), BATCH_SIZE):
            texts = df["Sentence"].iloc[
                start:start + BATCH_SIZE
            ].astype(str).tolist()

            # Count sentences exceeding the training sequence limit.
            untruncated = tokenizer(
                texts, truncation=False, padding=False
            )
            truncated_rows += sum(
                len(ids) > MAX_LENGTH
                for ids in untruncated["input_ids"]
            )

            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(device)

            logits = model(**inputs).logits
            predictions.extend(
                logits.argmax(dim=-1).cpu().tolist()
            )

            if start % 400 == 0:
                print(
                    f"Processed {len(predictions)}/{len(df)}",
                    flush=True,
                )

    gold = df["label"].map(label_map).to_numpy()
    predicted = np.asarray(predictions)

    # Rows are reference classes; columns are predicted classes.
    confusion = np.zeros((len(names), len(names)), dtype=int)
    np.add.at(confusion, (gold, predicted), 1)

    rows = []
    for name in names:
        index = label_map[name]
        tp = int(confusion[index, index])
        support = int(confusion[index, :].sum())
        predicted_count = int(confusion[:, index].sum())

        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        rows.append({
            "class": name,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        })

    report = pd.DataFrame(rows)
    summary = {
        "test_rows": len(df),
        "accuracy": float(np.mean(gold == predicted)),
        "macro_f1": float(report["f1"].mean()),
        "truncated_rows": int(truncated_rows),
        "max_length": MAX_LENGTH,
    }

    OUT_DIR.mkdir(parents=True)
    report.to_csv(OUT_DIR / "per_class_metrics.csv", index=False)
    pd.DataFrame(
        confusion, index=names, columns=names
    ).to_csv(OUT_DIR / "confusion_matrix.csv")

    id_to_label = {value: key for key, value in label_map.items()}
    df["predicted_label"] = [
        id_to_label[index] for index in predictions
    ]
    df["correct"] = gold == predicted
    df.to_csv(OUT_DIR / "predictions.csv", index=False)

    (OUT_DIR / "metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nTest results:")
    print(json.dumps(summary, indent=2))
    print("\nPer-class results:")
    print(report.to_string(index=False))
    print("\nSaved results:", OUT_DIR)


if __name__ == "__main__":
    main()