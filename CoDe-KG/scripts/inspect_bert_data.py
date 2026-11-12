from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


# Funktioniert unabhängig vom aktuellen Terminalordner.
PROJECT_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_DIR / "train.csv"

try:
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
except UnicodeDecodeError:
    df = pd.read_csv(CSV_PATH, encoding="latin-1")

print("=== Datenqualität ===")
print("Zeilen:", len(df))
print("Fehlende Werte:")
print(df[["PMID", "Sentence", "label"]].isna().sum())

print("\nUnterschiedliche Abstracts:", df["PMID"].nunique())
print("Exakt doppelte Sätze:", df["Sentence"].duplicated().sum())

print("\nKlassenverteilung:")
print(df["label"].value_counts())

if df[["PMID", "Sentence", "label"]].isna().any().any():
    raise ValueError("Fehlende Werte zuerst prüfen.")

tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")

encoded = tokenizer(
    df["Sentence"].astype(str).tolist(),
    add_special_tokens=True,
    truncation=False,
    padding=False,
)

lengths = np.array([len(ids) for ids in encoded["input_ids"]])

print("\n=== Satzlängen inklusive Spezialtokens ===")
for percentile in (50, 90, 95, 99):
    print(
        f"{percentile}. Perzentil: "
        f"{np.percentile(lengths, percentile):.0f} Tokens"
    )

print("Maximum:", lengths.max())

for limit in (128, 256, 512):
    count = int((lengths > limit).sum())
    print(
        f"Länger als {limit} Tokens: "
        f"{count}/{len(df)} ({count / len(df):.1%})"
    )