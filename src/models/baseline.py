"""Vanilla DeBERTa + GRU baseline (no persuasion fine-tuning).

Uses frozen DeBERTa-v3-base embeddings directly fed to GRU classifier.
Serves as the control condition to measure persuasion-grounding benefit.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from .gru_classifier import GRUClassifier


class BaselineModel(nn.Module):
    """Frozen DeBERTa-v3-base encoder + GRU classifier (end-to-end inference)."""

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, dtype=torch.float32)
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.classifier = GRUClassifier(
            input_dim=self.config.hidden_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    @torch.no_grad()
    def encode_turns(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]

    def forward(self, embeddings: torch.Tensor, lengths: torch.Tensor):
        return self.classifier(embeddings, lengths)
