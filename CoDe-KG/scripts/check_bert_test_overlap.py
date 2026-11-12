from pathlib import Path

import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[2]


def read_csv(path):
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def normalize_text(series):
    # Normalize case and whitespace for a conservative overlap check.
    return series.astype(str).str.casefold().str.split().str.join(" ")


run_dir = REPO_DIR / "outputs" / "bert_training"

training = read_csv(run_dir / "train_split.csv")
validation = read_csv(run_dir / "validation_split.csv")
test = read_csv(
    REPO_DIR / "datasets" / "Sentence_Classifier_Dataset" / "test.csv"
)

# Treat validation data as used data too, keeping the final test independent.
used = pd.concat([training, validation], ignore_index=True)

used_texts = set(normalize_text(used["Sentence"]))
used_pmids = set(used["PMID"].astype(str).str.strip())

test_keys = normalize_text(test["Sentence"])
same_text = test_keys.isin(used_texts)
same_abstract = test["PMID"].astype(str).str.strip().isin(used_pmids)
overlap = same_text | same_abstract

print("Original test rows:", len(test))
print("Matching sentence text:", int(same_text.sum()))
print("Matching PMID:", int(same_abstract.sum()))
print("Rows with either overlap:", int(overlap.sum()))
print("Remaining rows:", int((~overlap).sum()))

print("\nDuplicate sentence rows within the test set:")
print(int(test_keys.duplicated().sum()))

label_counts = test.assign(_key=test_keys).groupby("_key")["label"].nunique()
print("Test sentences with conflicting labels:", int((label_counts > 1).sum()))

print("\nClass distribution after excluding overlaps:")
print(test.loc[~overlap, "label"].value_counts())

# Inspect labels only; do not run predictions or alter the training data.
# Build an independent test set with unique normalized sentences.
clean_test = test.loc[~overlap].copy()
clean_test["_text_key"] = normalize_text(clean_test["Sentence"])

before_deduplication = len(clean_test)
clean_test = clean_test.drop_duplicates("_text_key").copy()
removed_duplicates = before_deduplication - len(clean_test)

# Verify independence from both training and validation data.
assert set(clean_test["_text_key"]).isdisjoint(used_texts)
assert set(
    clean_test["PMID"].astype(str).str.strip()
).isdisjoint(used_pmids)

output_dir = REPO_DIR / "data" / "evaluation"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "bert_test_clean.csv"

# Prevent accidental replacement of an already frozen test set.
if output_path.exists():
    raise FileExistsError(f"Test set already exists: {output_path}")

clean_test = clean_test.drop(columns="_text_key")
clean_test.to_csv(output_path, index=False)

print("\nRemoved duplicates after overlap exclusion:", removed_duplicates)
print("Final test rows:", len(clean_test))
print("Saved test set:", output_path)
print("\nFinal class distribution:")
print(clean_test["label"].value_counts())