"""Exp 163 — T2T dual RoBERTa + cross-attention, Qwen7B captions, excl bar+pie+composite."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from shared.common import Tee, set_seed, SCORE_COLUMNS
from shared.data_loader_b import load_captions, get_loaders
from approach_b.model import DualRobertaCrossAttnCls
from approach_b.trainer import run

CAPTIONS_PATH  = os.path.join(ROOT, 'data', 'captions', 'qwen7b_captions_full1054.json')
EXCLUDE_TYPES  = {'pie_chart', 'bar_chart', 'composite_chart'}

RESULTS_DIR = os.path.join(ROOT, 'results', '163', 'results')
MODELS_DIR  = os.path.join(ROOT, 'results', '163', 'models')

CFG = {
    'exp': '163_T2T-DualRoBERTa-CrossAttn-Qwen7B-ExclBarPieComposite',
    'lr': 1e-4, 'weight_decay': 0.01,
    'dropout1': 0.30, 'dropout2': 0.25,
    'batch_size': 4, 'accum_steps': 8,
    'max_epochs': 30, 'patience': 10,
    'scheduler_factor': 0.5, 'scheduler_patience': 3,
    'seed': 42,
}

if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)
    Tee(os.path.join(RESULTS_DIR, 'train_log.txt'))
    set_seed(CFG['seed'])
    print(f"Exp: {CFG['exp']}", flush=True)
    captions = load_captions(CAPTIONS_PATH, exclude_types=EXCLUDE_TYPES)
    train_loader, val_loader, test_loader, train_df = get_loaders(
        captions, SCORE_COLUMNS, CFG['batch_size'], CFG['seed'])
    model = DualRobertaCrossAttnCls(d1=CFG['dropout1'], d2=CFG['dropout2'])
    run(model, train_loader, val_loader, test_loader, train_df, CFG, RESULTS_DIR, MODELS_DIR)
