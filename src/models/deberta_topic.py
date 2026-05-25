# Plan 016v2: DeBERTa with 5-class topic head only (no intent head).
# Control experiment to test whether generic classification fine-tuning
# (as opposed to persuasion-specific) drives cross-attack generalization.
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class DeBERTaTopic(nn.Module):
    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        num_topic_classes: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.deberta = AutoModel.from_pretrained(model_name, dtype=torch.float32)
        hidden_size = self.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.topic_head = nn.Linear(hidden_size, num_topic_classes)

    def forward(
        self,
        input_ids,
        attention_mask,
        topic_labels=None,
    ):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        cls_emb = self.dropout(cls_emb)

        topic_logits = self.topic_head(cls_emb)

        loss = None
        if topic_labels is not None:
            ce = nn.CrossEntropyLoss()
            loss = ce(topic_logits, topic_labels)

        return {
            "loss": loss,
            "topic_logits": topic_logits,
            "cls_embedding": cls_emb.detach(),
        }
