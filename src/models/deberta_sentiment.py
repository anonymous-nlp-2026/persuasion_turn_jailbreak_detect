# Plan 016 control: 3-class sentiment + binary intent (replaces 9-class persuasion head)
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class DeBERTaSentiment(nn.Module):
    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        num_sentiment_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.deberta = AutoModel.from_pretrained(model_name, dtype=torch.float32)
        hidden_size = self.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.sentiment_head = nn.Linear(hidden_size, num_sentiment_classes)
        self.intent_head = nn.Linear(hidden_size, 2)

    def forward(
        self,
        input_ids,
        attention_mask,
        sentiment_labels=None,
        intent_labels=None,
        alpha=0.3,
    ):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        cls_emb = self.dropout(cls_emb)

        sentiment_logits = self.sentiment_head(cls_emb)
        intent_logits = self.intent_head(cls_emb)

        loss = None
        if sentiment_labels is not None and intent_labels is not None:
            ce = nn.CrossEntropyLoss()
            loss_sentiment = ce(sentiment_logits, sentiment_labels)
            loss_intent = ce(intent_logits, intent_labels)
            loss = loss_sentiment + alpha * loss_intent

        return {
            "loss": loss,
            "sentiment_logits": sentiment_logits,
            "intent_logits": intent_logits,
            "cls_embedding": cls_emb.detach(),
        }
