import os
os.environ["HF_HOME"] = "~/.cache/huggingface"
from transformers import AutoModel, AutoTokenizer
print("Downloading roberta-base...")
tok = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModel.from_pretrained("roberta-base")
print(f"Done. Hidden size: {model.config.hidden_size}")
print(f"Model type: {type(model).__name__}")
