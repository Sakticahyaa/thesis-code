"""Exp 050 — single ResNet50 + BERT + cross-attention, MSE regression."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from shared.common import Tee, set_seed, SCORE_COLUMNS
from shared.data_loader_a import get_loaders
from lampiran_5.models import BRCAFSingleReg
from lampiran_5.trainer import run

RESULTS_DIR = os.path.join(ROOT, 'results', '050', 'results')
MODELS_DIR  = os.path.join(ROOT, 'results', '050', 'models')

CFG = {
    'exp': '050_BRCAFSingleReg_ResNet50',
    'lr': 2e-5, 'weight_decay': 0.01,
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
    train_loader, val_loader, test_loader, train_df = get_loaders(
        SCORE_COLUMNS, CFG['batch_size'], CFG['seed'])
    model = BRCAFSingleReg(d1=CFG['dropout1'], d2=CFG['dropout2'])
    run(model, train_loader, val_loader, test_loader, train_df, CFG, RESULTS_DIR, MODELS_DIR, mode='reg')
