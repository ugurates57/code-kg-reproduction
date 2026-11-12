import json
from pathlib import Path

import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_DIR / "datasets"

for path in sorted(DATA_DIR.rglob("*.csv")):
    print(f"\n=== {path.relative_to(REPO_DIR)} ===")

    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")

    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())
    print("Missing values:", int(df.isna().sum().sum()))

    # Show a short preview without flooding the terminal.
    if not df.empty:
        print("First row:")
        for column, value in df.iloc[0].items():
            print(f"  {column}: {str(value)[:200]}")

path = DATA_DIR / "coref_human_gold_annotation.json"

if path.exists():
    print(f"\n=== {path.relative_to(REPO_DIR)} ===")
    data = json.loads(path.read_text(encoding="utf-8"))
    print("Top-level type:", type(data).__name__)
    print("Top-level entries:", len(data))

    if isinstance(data, list) and data:
        first = data[0]
        print(
            "First entry keys:",
            list(first) if isinstance(first, dict) else type(first).__name__,
        )
    elif isinstance(data, dict):
        print("First keys:", list(data)[:10])