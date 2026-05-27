from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from src.modeling import DeBERTaDetector


class TextDataset(Dataset):
    def __init__(self, texts: Iterable[str], tokenizer, max_length: int = 512):
        self.texts = list(texts)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        return {key: value.squeeze(0) for key, value in enc.items()}


def load_detector(
    checkpoint_path: Path,
    model_name: str = "microsoft/deberta-v3-base",
    pooling: str = "mean",
    dropout: float = 0.1,
    device: Optional[torch.device] = None,
) -> DeBERTaDetector:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeBERTaDetector(model_name=model_name, pooling=pooling, dropout=dropout).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def predict_proba(
    texts: Iterable[str],
    checkpoint_path: Path,
    model_name: str = "microsoft/deberta-v3-base",
    max_length: int = 512,
    batch_size: int = 32,
    pooling: str = "mean",
    dropout: float = 0.1,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Return P(AI) for each input text."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = load_detector(
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        pooling=pooling,
        dropout=dropout,
        device=device,
    )
    dataset = TextDataset(texts, tokenizer=tokenizer, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    probs: List[float] = []
    for batch in tqdm(loader, desc="DeBERTa inference", ncols=90):
        batch = {key: value.to(device) for key, value in batch.items()}
        logits = model(**batch)
        probs.extend(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy().tolist())
    return np.asarray(probs, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DeBERTa AI-text detection inference.")
    parser.add_argument("--input", required=True, type=Path, help="CSV file containing a text column.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to DeBERTa checkpoint.")
    parser.add_argument("--output", required=True, type=Path, help="Output CSV path.")
    parser.add_argument("--text-col", default="text_clean", help="Input text column name.")
    parser.add_argument("--model-name", default="microsoft/deberta-v3-base")
    parser.add_argument("--max-length", default=512, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--threshold", default=0.5, type=float)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.text_col not in df.columns:
        raise ValueError(f"Column '{args.text_col}' not found in {args.input}")

    probs = predict_proba(
        texts=df[args.text_col].fillna("").astype(str).tolist(),
        checkpoint_path=args.checkpoint,
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    out = df.copy()
    out["prob_ai"] = probs
    out["pred"] = (out["prob_ai"] >= args.threshold).astype(int)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()

