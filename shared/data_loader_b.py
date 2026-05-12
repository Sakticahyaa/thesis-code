"""Approach B data loader: RoBERTa tokenizer + captions (experiments 158, 162–164)."""
import os, json
import pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer

_HERE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CSV = os.path.join(_HERE, 'data', 'lampiran_3', 'data.csv')


def load_captions(path, exclude_types=None):
    """Load captions dict from JSON, optionally skipping chart types in exclude_types."""
    exclude_types = exclude_types or set()
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    captions = {}
    for k, v in raw.items():
        entry = v if isinstance(v, dict) else {'caption': v, 'type': ''}
        if entry.get('type') not in exclude_types:
            captions[k] = entry['caption']
    return captions


class T2TDataset(Dataset):
    def __init__(self, df, captions, tokenizer, score_columns, max_length=512):
        self.df            = df.reset_index(drop=True)
        self.captions      = captions
        self.tokenizer     = tokenizer
        self.score_columns = score_columns
        self.max_length    = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row     = self.df.iloc[idx]
        caption = self.captions[f'image_{int(row["original_idx"])}']
        cap_enc = self.tokenizer(caption, max_length=self.max_length,
                                 padding=False, truncation=True, return_tensors='pt')
        ess_enc = self.tokenizer(str(row['Essay']), max_length=self.max_length,
                                 padding=False, truncation=True, return_tensors='pt')
        scores  = torch.tensor([row[c] for c in self.score_columns], dtype=torch.float32)
        return {
            'caption_input_ids':      cap_enc['input_ids'].squeeze(0),
            'caption_attention_mask': cap_enc['attention_mask'].squeeze(0),
            'essay_input_ids':        ess_enc['input_ids'].squeeze(0),
            'essay_attention_mask':   ess_enc['attention_mask'].squeeze(0),
            'scores': scores,
        }


def _collate(batch):
    pad = torch.nn.utils.rnn.pad_sequence
    return {
        'caption_input_ids':      pad([b['caption_input_ids'] for b in batch],      batch_first=True, padding_value=1),
        'caption_attention_mask': pad([b['caption_attention_mask'] for b in batch], batch_first=True, padding_value=0),
        'essay_input_ids':        pad([b['essay_input_ids'] for b in batch],        batch_first=True, padding_value=1),
        'essay_attention_mask':   pad([b['essay_attention_mask'] for b in batch],   batch_first=True, padding_value=0),
        'scores': torch.stack([b['scores'] for b in batch]),
    }


def get_loaders(captions, score_columns, batch_size=4, seed=42):
    """captions: pre-built dict {image_id: caption_str} from load_captions()."""
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    df = pd.read_csv(DATA_CSV, encoding='latin-1')
    df = df.reset_index(drop=False).rename(columns={'index': 'original_idx'})
    df['image_id'] = [f'image_{int(i)}' for i in df['original_idx']]
    df = df[df['image_id'].isin(captions)].reset_index(drop=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    n = len(df)
    n_train, n_val = int(0.70 * n), int(0.15 * n)
    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df   = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test_df  = df.iloc[n_train + n_val:].reset_index(drop=True)
    print(f'Split: {len(train_df)} / {len(val_df)} / {len(test_df)}', flush=True)

    kw   = dict(collate_fn=_collate, num_workers=0)
    make = lambda d, s: DataLoader(T2TDataset(d, captions, tokenizer, score_columns),
                                   batch_size=batch_size, shuffle=s, **kw)
    return make(train_df, True), make(val_df, False), make(test_df, False), train_df
