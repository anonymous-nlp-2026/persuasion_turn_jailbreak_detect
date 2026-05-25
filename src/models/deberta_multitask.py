"""DeBERTa multi-task model for persuasion strategy + jailbreak intent classification.

Input: tokenized user turn
Output: persuasion logits (9-class), intent logits (binary), CLS embedding
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class DeBERTaMultiTask(nn.Module):
    """DeBERTa-v3-base with two classification heads.

    Head A: 9-class persuasion strategy (0=none + 8 strategies)
    Head B: binary jailbreak intent per turn
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        num_persuasion_classes: int = 9,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.deberta = AutoModel.from_pretrained(model_name, dtype=torch.float32)
        hidden_size = self.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.persuasion_head = nn.Linear(hidden_size, num_persuasion_classes)
        self.intent_head = nn.Linear(hidden_size, 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        persuasion_labels: torch.Tensor = None,
        intent_labels: torch.Tensor = None,
        alpha: float = 0.3,
    ):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # CLS token
        cls_emb = self.dropout(cls_emb)

        persuasion_logits = self.persuasion_head(cls_emb)
        intent_logits = self.intent_head(cls_emb)

        loss = None
        if persuasion_labels is not None and intent_labels is not None:
            ce = nn.CrossEntropyLoss()
            loss_persuasion = ce(persuasion_logits, persuasion_labels)
            loss_intent = ce(intent_logits, intent_labels)
            loss = loss_persuasion + alpha * loss_intent

        return {
            "loss": loss,
            "persuasion_logits": persuasion_logits,
            "intent_logits": intent_logits,
            "cls_embedding": cls_emb.detach(),
        }

    def get_embedding(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Extract CLS embedding without classification heads."""
        with torch.no_grad():
            outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state[:, 0, :]
