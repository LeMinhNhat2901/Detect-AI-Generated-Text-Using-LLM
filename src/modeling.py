from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class MeanPooling(nn.Module):
    """Mean pooling over non-padding tokens."""

    def forward(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts


class DeBERTaDetector(nn.Module):
    """DeBERTa-v3 classifier with mean/CLS pooling and a linear head."""

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        num_labels: int = 2,
        dropout: float = 0.1,
        pooling: str = "mean",
    ):
        super().__init__()
        if pooling not in {"mean", "cls"}:
            raise ValueError("pooling must be either 'mean' or 'cls'")

        self.pooling = pooling
        self.config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
        )
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        self.mean_pooler = MeanPooling()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        self._init_classifier()

    def _init_classifier(self) -> None:
        nn.init.normal_(self.classifier.weight, mean=0.0, std=self.config.initializer_range)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        if self.pooling == "cls":
            pooled = outputs.last_hidden_state[:, 0]
        else:
            pooled = self.mean_pooler(outputs.last_hidden_state, attention_mask)
        return self.classifier(self.dropout(pooled))

