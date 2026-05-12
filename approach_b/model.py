"""Approach B model: dual RoBERTa + cross-attention (experiments 158, 162–164)."""
import torch, torch.nn as nn
from transformers import RobertaModel

NUM_TRAITS  = 10
NUM_CLASSES = 11  # score 0.0–5.0 in 0.5 steps → class index 0..10


class DualRobertaCrossAttnCls(nn.Module):
    """Essay CLS [B,1,768] attends to caption sequence [B,m,768] via cross-attention, CE."""
    def __init__(self, d1=0.30, d2=0.25):
        super().__init__()
        self.essay_enc   = RobertaModel.from_pretrained('roberta-base')
        self.caption_enc = RobertaModel.from_pretrained('roberta-base')
        self.cross_attn  = nn.MultiheadAttention(768, num_heads=8, batch_first=True)
        self.classifier  = nn.Sequential(
            nn.Linear(1536, 768), nn.ReLU(), nn.Dropout(d1),
            nn.Linear(768, 256),  nn.ReLU(), nn.Dropout(d2),
            nn.Linear(256, NUM_TRAITS * NUM_CLASSES),
        )

    def forward(self, essay_ids, essay_mask, caption_ids, caption_mask):
        essay_out   = self.essay_enc(input_ids=essay_ids,   attention_mask=essay_mask)
        caption_out = self.caption_enc(input_ids=caption_ids, attention_mask=caption_mask)

        essay_cls  = essay_out.last_hidden_state[:, 0:1, :]   # [B,1,768] — query
        cap_seq    = caption_out.last_hidden_state             # [B,m,768] — key/value

        # key_padding_mask: True where padding (MHA convention)
        key_pad = (caption_mask == 0)
        ca, _ = self.cross_attn(query=essay_cls, key=cap_seq, value=cap_seq,
                                key_padding_mask=key_pad)
        fused = torch.cat([essay_cls.squeeze(1), ca.squeeze(1)], dim=1)  # [B,1536]
        return self.classifier(fused).view(-1, NUM_TRAITS, NUM_CLASSES)
