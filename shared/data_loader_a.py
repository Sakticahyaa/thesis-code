"""Approach A data loader: BERT tokenizer + image (experiments 046–056)."""
import os
import urllib.request
import pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import BertTokenizer
from torchvision import transforms

_HERE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CSV  = os.path.join(_HERE, 'data', 'data.csv')
IMG_CACHE = 'C:/Outpost/Skripsi/Research 3/shared_data/image_cache'  # images not copied (too large)
MAX_LEN   = 464

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _fetch_image(url, idx):
    os.makedirs(IMG_CACHE, exist_ok=True)
    path = os.path.join(IMG_CACHE, f'image_{idx}.jpg')
    if not os.path.exists(path):
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as e:
            print(f'Image download failed (idx={idx}): {e}')
            return None
    return path


def load_splits(seed=42):
    df = pd.read_csv(DATA_CSV, encoding='latin-1')
    df = df.reset_index(drop=False).rename(columns={'index': 'original_idx'})
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_train, n_val = int(0.70 * n), int(0.15 * n)
    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df   = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test_df  = df.iloc[n_train + n_val:].reset_index(drop=True)
    print(f'Split: {len(train_df)} / {len(val_df)} / {len(test_df)}', flush=True)
    return train_df, val_df, test_df


class FusionDataset(Dataset):
    def __init__(self, df, tokenizer, score_columns):
        self.df, self.tokenizer, self.score_columns = df, tokenizer, score_columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        enc  = self.tokenizer(str(row['Essay']), max_length=MAX_LEN,
                              truncation=True, padding=False, return_tensors='pt')
        path = _fetch_image(row['graph'], int(row['original_idx']))
        try:
            image = TRANSFORM(Image.open(path).convert('RGB')) if path else torch.zeros(3, 224, 224)
        except Exception:
            image = torch.zeros(3, 224, 224)
        labels = torch.tensor([float(row[c]) for c in self.score_columns], dtype=torch.float32)
        return enc['input_ids'].squeeze(0), enc['attention_mask'].squeeze(0), image, labels


def _collate(batch):
    ids, masks, images, labels = zip(*batch)
    max_len   = max(x.size(0) for x in ids)
    pad_ids   = torch.zeros(len(ids), max_len, dtype=torch.long)
    pad_masks = torch.zeros(len(ids), max_len, dtype=torch.long)
    for i, (id_, mask) in enumerate(zip(ids, masks)):
        pad_ids[i, :id_.size(0)]    = id_
        pad_masks[i, :mask.size(0)] = mask
    return pad_ids, pad_masks, torch.stack(images), torch.stack(labels)


def get_loaders(score_columns, batch_size=4, seed=42):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_df, val_df, test_df = load_splits(seed)
    kw   = dict(collate_fn=_collate, num_workers=0)
    make = lambda df, s: DataLoader(FusionDataset(df, tokenizer, score_columns),
                                    batch_size=batch_size, shuffle=s, **kw)
    return make(train_df, True), make(val_df, False), make(test_df, False), train_df
