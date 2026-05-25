import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import AutoModel, AutoTokenizer

save_path = './models/roberta-large'
if not os.path.exists(save_path):
    print('Downloading roberta-large...')
    m = AutoModel.from_pretrained('roberta-large')
    t = AutoTokenizer.from_pretrained('roberta-large')
    m.save_pretrained(save_path)
    t.save_pretrained(save_path)
    print(f'Saved to {save_path}')
else:
    print(f'{save_path} already exists')

m = AutoModel.from_pretrained(save_path)
print(f'hidden_size={m.config.hidden_size}, num_params={sum(p.numel() for p in m.parameters())/1e6:.1f}M')
print('DONE')
