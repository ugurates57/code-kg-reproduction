import sys
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)

sys.path.insert(0, str(Path("CoDe-KG").resolve()))

from bert_sentence_classifier import (
    CSVDataset,
    LABEL_MAP_DEFAULT,
)

set_seed(4000)

# Zwei Beispiele je Klasse, ausschließlich für den technischen Test.
try:
    df = pd.read_csv("CoDe-KG/train.csv", encoding="utf-8-sig")
except UnicodeDecodeError:
    df = pd.read_csv("CoDe-KG/train.csv", encoding="latin-1")
sample = pd.concat(
    group.sample(n=2, random_state=4000)
    for _, group in df.groupby("label")
).reset_index(drop=True)

model_name = "bert-large-uncased"

print("Lade Tokenizer und BERT-Large ...", flush=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=5,
    id2label={v: k for k, v in LABEL_MAP_DEFAULT.items()},
    label2id=LABEL_MAP_DEFAULT,
    attn_implementation="eager",
)

dataset = CSVDataset(
    df=sample,
    tokenizer=tokenizer,
    text_col="Sentence",
    label_col="label",
    label_map=LABEL_MAP_DEFAULT,
    max_length=128,
)

args = TrainingArguments(
    output_dir="outputs/bert_smoke",
    max_steps=2,
    per_device_train_batch_size=1,
    learning_rate=2e-5,
    gradient_checkpointing=True,
    dataloader_pin_memory=False,
    logging_steps=1,
    save_strategy="no",
    report_to=[],
    dataloader_num_workers=0,
    seed=4000,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=DataCollatorWithPadding(tokenizer),
)

print("Trainingsgerät:", trainer.args.device, flush=True)

if trainer.args.device.type != "mps":
    raise RuntimeError(
        "Apple-GPU wurde nicht ausgewählt. Bitte Ausgabe prüfen."
    )

start = time.perf_counter()
trainer.train()
torch.mps.synchronize()

print(f"\nTest erfolgreich. Trainingsdauer: {time.perf_counter()-start:.1f} s")
print("Es wurde kein fertiger Klassifikator gespeichert.")