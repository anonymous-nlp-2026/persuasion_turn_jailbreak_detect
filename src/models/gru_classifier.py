"""GRU sequence classifier for conversation-level jailbreak detection.

Input: sequence of turn embeddings (from DeBERTa CLS), variable length
Output: binary classification (jailbreak vs benign)
Architecture: Bidirectional GRU -> attention pooling -> MLP -> sigmoid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """Learned attention pooling over GRU hidden states."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, hidden_states: torch.Tensor, lengths: torch.Tensor):
        """
        Args:
            hidden_states: (batch, max_seq, hidden_dim)
            lengths: (batch,) actual lengths
        Returns:
            pooled: (batch, hidden_dim)
        """
        scores = self.attention(hidden_states).squeeze(-1)  # (batch, max_seq)
        mask = torch.arange(hidden_states.size(1), device=hidden_states.device)
        mask = mask.unsqueeze(0) < lengths.unsqueeze(1)  # (batch, max_seq)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=1).unsqueeze(-1)  # (batch, max_seq, 1)
        pooled = (hidden_states * weights).sum(dim=1)  # (batch, hidden_dim)
        return pooled


class GRUClassifier(nn.Module):
    """Bidirectional GRU with attention pooling for binary classification."""

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = AttentionPooling(hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, embeddings: torch.Tensor, lengths: torch.Tensor):
        """
        Args:
            embeddings: (batch, max_turns, input_dim)
            lengths: (batch,) number of actual turns
        Returns:
            logits: (batch, num_classes)
        """
        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        gru_out, _ = self.gru(packed)
        gru_out, _ = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)
        pooled = self.attention(gru_out, lengths)
        logits = self.classifier(pooled)
        return logits
