# bert_sentence_classifier.py
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


@dataclass
class BertClassifierConfig:
    model_name: str = "bert-large-uncased"
    num_labels: int = 5
    out_dir: str = "models/bert_sentence_classifier"
    max_length: int = 256


LABEL_MAP_DEFAULT = {
    "Simple Sentence": 0,
    "Compound Sentence": 1,
    "Complex Sentence": 2,
    "Compound-Complex Sentence": 3,
    "Incomplete Sentence": 4,
}

INV_LABEL_MAP_DEFAULT = {
    value: key for key, value in LABEL_MAP_DEFAULT.items()
}


class CSVDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        text_col: str,
        label_col: str,
        label_map: Dict[str, int],
        max_length: int,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.label_col = label_col
        self.label_map = label_map
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.df.loc[idx, self.text_col])
        label_str = str(self.df.loc[idx, self.label_col])

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
        )
        enc["labels"] = self.label_map[label_str]
        return enc


class BertSentenceClassifier:
    def __init__(self, cfg: BertClassifierConfig):
        self.cfg = cfg
        self.tokenizer = None
        self.model = None

    def exists(self) -> bool:
        return (
            os.path.isdir(self.cfg.out_dir)
            and os.path.isfile(
                os.path.join(self.cfg.out_dir, "config.json")
            )
        )

    def load(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.out_dir,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.cfg.out_dir,
            attn_implementation="eager",
        )

    def train_if_missing(
        self,
        train_csv: Optional[str],
        text_col: str,
        label_col: str,
        label_map: Dict[str, int] = LABEL_MAP_DEFAULT,
        epochs: int = 3,
        batch_size: int = 8,
        lr: float = 2e-5,
    ) -> None:
        if self.exists():
            self.load()
            return

        # Use the dedicated script to preserve our training setup.
        raise RuntimeError(
            "No trained BERT classifier found at "
            f"{self.cfg.out_dir}. "
            "Run CoDe-KG/scripts/train_bert.py first, or provide "
            "the correct model directory."
        )

    @torch.inference_mode()
    def predict_batch(
        self,
        sentences: List[str],
        batch_size: int = 4,
    ) -> List[Dict[str, object]]:
        if self.model is None or self.tokenizer is None:
            if self.exists():
                self.load()
            else:
                raise ValueError(
                    f"BERT model missing at {self.cfg.out_dir}."
                )

        if not sentences:
            return []

        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        # Place the model and all input tensors on the same device.
        self.model.to(device)
        self.model.eval()

        print(f"BERT device: {next(self.model.parameters()).device}")

        out = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]

            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.cfg.max_length,
            )
            enc = {
                name: tensor.to(device)
                for name, tensor in enc.items()
            }

            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1)

            predictions = probs.argmax(dim=-1).cpu().tolist()
            confidences = probs.max(dim=-1).values.cpu().tolist()

            for sentence, prediction, confidence in zip(
                batch, predictions, confidences
            ):
                out.append({
                    "sentence": sentence,
                    "predicted_label": self.model.config.id2label[
                        int(prediction)
                    ],
                    "confidence": float(confidence),
                })

        return out