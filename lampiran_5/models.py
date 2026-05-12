"""Approach A model variants (experiments 046–056)."""
import os, urllib.request
import torch, torch.nn as nn
from transformers import BertModel
import torchvision.models as tv_models

NUM_TRAITS  = 10
NUM_CLASSES = 11  # score 0.0–5.0 in 0.5 steps → class index 0..10

PLACES365_URL     = 'http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar'
PLACES365_WEIGHTS = 'C:/Outpost/Skripsi/Research 3/Experiment/046_BRCAF-Classification/code/resnet50_places365.pth.tar'


def _load_places365():
    if not os.path.exists(PLACES365_WEIGHTS):
        print('Downloading Places365 weights (~99MB)...', flush=True)
        urllib.request.urlretrieve(PLACES365_URL, PLACES365_WEIGHTS)
    model = tv_models.resnet50(pretrained=False)
    model.fc = nn.Linear(2048, 365)
    ckpt = torch.load(PLACES365_WEIGHTS, map_location='cpu')
    model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['state_dict'].items()})
    return model


def _resnet(name):
    return {'resnet18': tv_models.resnet18,
            'resnet50': tv_models.resnet50,
            'resnet152': tv_models.resnet152}[name](pretrained=True)


def _fc_cls(in_dim, d1=0.30, d2=0.25):
    """Three-layer FC head → NUM_TRAITS × NUM_CLASSES outputs."""
    return nn.Sequential(
        nn.Linear(in_dim, 768), nn.ReLU(), nn.Dropout(d1),
        nn.Linear(768, 256),    nn.ReLU(), nn.Dropout(d2),
        nn.Linear(256, NUM_TRAITS * NUM_CLASSES),
    )


def _cross_attn_fuse(H, h_cls, vis_kv, attention_mask, cross_attn):
    """essay sequence H attends to vis_kv; returns fused [B,1536]."""
    ca, _  = cross_attn(query=H, key=vis_kv, value=vis_kv)
    mask   = attention_mask.unsqueeze(-1).float()
    ca_mean = (ca * mask).sum(1) / mask.sum(1).clamp(min=1)
    return torch.cat([vis_kv.squeeze(1), h_cls * ca_mean], dim=1)


class BRCAFDualCls(nn.Module):
    """Exp046: dual ResNet (ImageNet ResNet152 + Places365 ResNet50) + BERT + cross-attention, CE."""
    def __init__(self, d1=0.30, d2=0.25):
        super().__init__()
        self.bert            = BertModel.from_pretrained('bert-base-uncased')
        self.resnet152       = tv_models.resnet152(pretrained=True)
        self.resnet50_places = _load_places365()
        self.vis_proj        = nn.Linear(1365, 768)
        self.cross_attn      = nn.MultiheadAttention(768, num_heads=8, batch_first=True)
        self.classifier      = _fc_cls(1536, d1, d2)

    def forward(self, input_ids, attention_mask, image):
        out  = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        H, h_cls = out.last_hidden_state, out.last_hidden_state[:, 0, :]
        vis  = torch.cat([self.resnet152(image), self.resnet50_places(image)], dim=1)
        vis_kv = self.vis_proj(vis).unsqueeze(1)
        fused  = _cross_attn_fuse(H, h_cls, vis_kv, attention_mask, self.cross_attn)
        return self.classifier(fused).view(-1, NUM_TRAITS, NUM_CLASSES)


class BRCAFSingleCls(nn.Module):
    """Exp047/048/049: single ImageNet ResNet + BERT + cross-attention, CE.
    resnet_type: 'resnet18' | 'resnet50' | 'resnet152'."""
    def __init__(self, resnet_type='resnet50', d1=0.30, d2=0.25):
        super().__init__()
        self.bert       = BertModel.from_pretrained('bert-base-uncased')
        self.cnn        = _resnet(resnet_type)
        self.vis_proj   = nn.Linear(1000, 768)
        self.cross_attn = nn.MultiheadAttention(768, num_heads=8, batch_first=True)
        self.classifier = _fc_cls(1536, d1, d2)

    def forward(self, input_ids, attention_mask, image):
        out  = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        H, h_cls = out.last_hidden_state, out.last_hidden_state[:, 0, :]
        vis_kv = self.vis_proj(self.cnn(image)).unsqueeze(1)
        fused  = _cross_attn_fuse(H, h_cls, vis_kv, attention_mask, self.cross_attn)
        return self.classifier(fused).view(-1, NUM_TRAITS, NUM_CLASSES)


class BRCAFSingleReg(nn.Module):
    """Exp050: single ResNet50 + BERT + cross-attention, MSE regression (sigmoid × 5)."""
    def __init__(self, d1=0.30, d2=0.25):
        super().__init__()
        self.bert       = BertModel.from_pretrained('bert-base-uncased')
        self.cnn        = tv_models.resnet50(pretrained=True)
        self.vis_proj   = nn.Linear(1000, 768)
        self.cross_attn = nn.MultiheadAttention(768, num_heads=8, batch_first=True)
        self.regressor  = nn.Sequential(
            nn.Linear(1536, 768), nn.ReLU(), nn.Dropout(d1),
            nn.Linear(768, 256),  nn.ReLU(), nn.Dropout(d2),
            nn.Linear(256, NUM_TRAITS),
        )

    def forward(self, input_ids, attention_mask, image):
        out  = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        H, h_cls = out.last_hidden_state, out.last_hidden_state[:, 0, :]
        vis_kv = self.vis_proj(self.cnn(image)).unsqueeze(1)
        fused  = _cross_attn_fuse(H, h_cls, vis_kv, attention_mask, self.cross_attn)
        return torch.sigmoid(self.regressor(fused)) * 5.0


class BRCAFEarlyFusionCls(nn.Module):
    """Exp055: image token prepended to essay tokens → joint BERT encoding, CE."""
    def __init__(self, d1=0.30, d2=0.25):
        super().__init__()
        self.bert       = BertModel.from_pretrained('bert-base-uncased')
        self.cnn        = tv_models.resnet50(pretrained=True)
        self.vis_proj   = nn.Linear(1000, 768)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(d1),
            nn.Linear(256, NUM_TRAITS * NUM_CLASSES),
        )

    def forward(self, input_ids, attention_mask, image):
        img_tok  = self.vis_proj(self.cnn(image)).unsqueeze(1)
        combined = torch.cat([img_tok, self.bert.embeddings.word_embeddings(input_ids)], dim=1)
        img_mask = torch.ones(attention_mask.size(0), 1,
                              dtype=attention_mask.dtype, device=attention_mask.device)
        out = self.bert(inputs_embeds=combined,
                        attention_mask=torch.cat([img_mask, attention_mask], dim=1))
        return self.classifier(out.last_hidden_state[:, 1, :]).view(-1, NUM_TRAITS, NUM_CLASSES)


class BRCAFLateFusionCls(nn.Module):
    """Exp056: BERT and ResNet50 encode independently; concatenated → FC, CE."""
    def __init__(self, d1=0.30, d2=0.25):
        super().__init__()
        self.bert       = BertModel.from_pretrained('bert-base-uncased')
        self.cnn        = tv_models.resnet50(pretrained=True)
        self.vis_proj   = nn.Linear(1000, 768)
        self.classifier = _fc_cls(1536, d1, d2)

    def forward(self, input_ids, attention_mask, image):
        cls_out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        img_out = self.vis_proj(self.cnn(image))
        return self.classifier(torch.cat([cls_out, img_out], dim=1)).view(-1, NUM_TRAITS, NUM_CLASSES)
