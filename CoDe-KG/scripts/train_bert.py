import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from bert_sentence_classifier import CSVDataset, LABEL_MAP_DEFAULT

SEED = 4000
MODEL_NAME = "bert-large-uncased"
MAX_LENGTH = 256

RUN_DIR = REPO_DIR / "outputs" / "bert_training"
MODEL_DIR = REPO_DIR / "models" / "bert_sentence_classifier"


def compute_metrics(prediction):
    logits, labels = prediction
    predicted = np.argmax(logits, axis=-1)

    metrics = {
        "accuracy": float(np.mean(predicted == labels)),
    }
    f1_scores = []

    for name, class_id in LABEL_MAP_DEFAULT.items():
        tp = np.sum((predicted == class_id) & (labels == class_id))
        fp = np.sum((predicted == class_id) & (labels != class_id))
        fn = np.sum((predicted != class_id) & (labels == class_id))

        denominator = 2 * tp + fp + fn
        f1 = float(2 * tp / denominator) if denominator else 0.0
        f1_scores.append(f1)
        metrics[f"f1_{name.replace(' ', '_')}"] = f1

    metrics["macro_f1"] = float(np.mean(f1_scores))
    return metrics


def main():
    set_seed(SEED)

    if MODEL_DIR.exists() or RUN_DIR.exists():
        raise FileExistsError(
            "The output directory already exists. "
            "Rename it before starting another run."
        )

    try:
        df = pd.read_csv(PROJECT_DIR / "train.csv", encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(PROJECT_DIR / "train.csv", encoding="latin-1")

    columns = ["PMID", "Sentence", "label"]
    if df[columns].isna().any().any():
        raise ValueError("The dataset contains missing values.")

    if not set(df["label"]).issubset(LABEL_MAP_DEFAULT):
        raise ValueError("The dataset contains unknown labels.")

    # Normalize whitespace to detect duplicate sentences.
    df["_text_key"] = df["Sentence"].astype(str).str.split().str.join(" ")

    if df["_text_key"].eq("").any():
        raise ValueError("The dataset contains empty sentences.")

    # Exclude every occurrence of sentences with conflicting labels.
    label_counts = df.groupby("_text_key")["label"].nunique()
    conflicting_keys = label_counts[label_counts > 1].index
    conflict_mask = df["_text_key"].isin(conflicting_keys)

    print("Excluded conflicting rows:", int(conflict_mask.sum()))
    df = df.loc[~conflict_mask].copy()

    original_count = len(df)
    df = df.drop_duplicates("_text_key").reset_index(drop=True)
    print("Removed duplicate sentences:", original_count - len(df))

    # This split assumes one sentence per abstract, as verified beforehand.
    if df["PMID"].duplicated().any():
        raise ValueError("Repeated PMIDs require a grouped split.")

    validation = pd.concat(
        group.sample(frac=0.2, random_state=SEED)
        for _, group in df.groupby("label")
    )
    training = df.drop(index=validation.index)

    assert set(training["PMID"]).isdisjoint(validation["PMID"])
    assert set(training["_text_key"]).isdisjoint(validation["_text_key"])

    print("\nClass distribution:")
    print(pd.DataFrame({
        "training": training["label"].value_counts(),
        "validation": validation["label"].value_counts(),
    }))

    RUN_DIR.mkdir(parents=True)

    # Preserve the exact split for later reproduction.
    training[columns].to_csv(RUN_DIR / "train_split.csv", index=False)
    validation[columns].to_csv(RUN_DIR / "validation_split.csv", index=False)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_MAP_DEFAULT),
        id2label={v: k for k, v in LABEL_MAP_DEFAULT.items()},
        label2id=LABEL_MAP_DEFAULT,
        attn_implementation="eager",
    )

    def make_dataset(frame):
        return CSVDataset(
            df=frame,
            tokenizer=tokenizer,
            text_col="Sentence",
            label_col="label",
            label_map=LABEL_MAP_DEFAULT,
            max_length=MAX_LENGTH,
        )

    args = TrainingArguments(
        output_dir=str(RUN_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=1,
        learning_rate=2e-5,
        optim="adamw_torch",
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=25,
        report_to=[],
        seed=SEED,
        data_seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=make_dataset(training),
        eval_dataset=make_dataset(validation),
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    print("\nTraining device:", trainer.args.device, flush=True)
    if trainer.args.device.type != "mps":
        raise RuntimeError("Expected the Apple GPU device 'mps'.")

    # Record the configuration before training starts.
    # Save the training configuration as JSON.
    (RUN_DIR / "training_arguments.json").write_text(
        args.to_json_string(),
        encoding="utf-8",
    )
    (RUN_DIR / "experiment.json").write_text(
        json.dumps({
            "model": MODEL_NAME,
            "seed": SEED,
            "max_length": MAX_LENGTH,
            "training_rows": len(training),
            "validation_rows": len(validation),
            "torch_version": torch.__version__,
        }, indent=2),
        encoding="utf-8",
    )

    trainer.train()

    metrics = trainer.evaluate()
    trainer.save_metrics("validation", metrics)

    # Save the final model and tokenizer for the CoDe-KG pipeline.
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))

    (RUN_DIR / "training_history.json").write_text(
        json.dumps(trainer.state.log_history, indent=2),
        encoding="utf-8",
    )

    print("\nValidation results:")
    print(json.dumps(metrics, indent=2))
    print("\nSaved classifier:", MODEL_DIR)


if __name__ == "__main__":
    main()