"""Shared training loop for Approach B (experiments 158, 162–164). CE classification."""
import os, json, time
import numpy as np, torch, torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from shared.common import (NUM_TRAITS, NUM_CLASSES, TRAITS,
                            compute_class_weights, compute_qwk_cls)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _epoch(model, loader, ce_losses, accum_steps, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, all_pred, all_label = 0.0, [], []
    if training:
        optimizer.zero_grad()
    with torch.set_grad_enabled(training):
        for i, batch in enumerate(loader):
            essay_ids   = batch['essay_input_ids'].to(DEVICE)
            essay_mask  = batch['essay_attention_mask'].to(DEVICE)
            caption_ids = batch['caption_input_ids'].to(DEVICE)
            caption_mask = batch['caption_attention_mask'].to(DEVICE)
            labels      = batch['scores'].to(DEVICE)

            label_idx = torch.round(labels * 2).long().clamp(0, NUM_CLASSES - 1)
            logits    = model(essay_ids, essay_mask, caption_ids, caption_mask)
            loss = sum(ce_losses[t](logits[:, t, :], label_idx[:, t])
                       for t in range(NUM_TRAITS)) / NUM_TRAITS

            if training:
                (loss / accum_steps).backward()
                if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step(); optimizer.zero_grad()

            total_loss += loss.item()
            all_pred.append(logits.argmax(-1).detach().cpu().numpy())
            all_label.append(label_idx.cpu().numpy())

    preds, labels = np.concatenate(all_pred), np.concatenate(all_label)
    avg_qwk, per_trait = compute_qwk_cls(preds, labels)
    return total_loss / len(loader), avg_qwk, per_trait


def run(model, train_loader, val_loader, test_loader, train_df, cfg, results_dir, models_dir):
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    model = model.to(DEVICE)
    accum = cfg.get('accum_steps', 8)

    cw = compute_class_weights(train_df)
    ce_losses = [nn.CrossEntropyLoss(weight=torch.tensor(cw[t], dtype=torch.float32).to(DEVICE))
                 for t in range(NUM_TRAITS)]

    optimizer = AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='max',
                                  factor=cfg['scheduler_factor'],
                                  patience=cfg['scheduler_patience'])

    best_val_qwk, best_epoch, patience_ctr = -1.0, 0, 0
    best_ckpt = os.path.join(models_dir, 'best_model.pt')
    history, t_start = [], time.time()

    for epoch in range(1, cfg['max_epochs'] + 1):
        t0 = time.time()
        tr_loss, tr_qwk, _ = _epoch(model, train_loader, ce_losses, accum, optimizer)
        vl_loss, vl_qwk, _ = _epoch(model, val_loader,   ce_losses, accum)
        scheduler.step(vl_qwk)
        print(f'Epoch {epoch:02d} | train={tr_loss:.4f}/{tr_qwk:.4f} '
              f'val={vl_loss:.4f}/{vl_qwk:.4f} | {time.time()-t0:.1f}s', flush=True)
        history.append({'epoch': epoch, 'train_loss': tr_loss,
                        'train_qwk': tr_qwk, 'val_qwk': vl_qwk})

        if vl_qwk > best_val_qwk:
            best_val_qwk, best_epoch, patience_ctr = vl_qwk, epoch, 0
            torch.save(model.state_dict(), best_ckpt)
            print(f'  -> best val QWK {best_val_qwk:.4f} saved', flush=True)
        else:
            patience_ctr += 1
            if patience_ctr >= cfg['patience']:
                print(f'Early stop at epoch {epoch}', flush=True); break

    model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))
    _, test_qwk, per_trait = _epoch(model, test_loader, ce_losses, accum)
    elapsed_min = (time.time() - t_start) / 60.0

    print(f'\nTest QWK: {test_qwk:.4f}', flush=True)
    for t, q in zip(TRAITS, per_trait):
        print(f'  {t[:20]:<20}: {q:.4f}', flush=True)
    print(f'Best val: {best_val_qwk:.4f} (epoch {best_epoch}) | {elapsed_min:.1f} min', flush=True)

    with open(os.path.join(results_dir, 'training_results.json'), 'w') as f:
        json.dump({
            'exp': cfg['exp'], 'best_val_qwk': round(best_val_qwk, 4),
            'best_epoch': best_epoch, 'test_qwk': round(test_qwk, 4),
            'per_trait_test': {t: round(q, 4) for t, q in zip(TRAITS, per_trait)},
            'training_min': round(elapsed_min, 1), 'config': cfg, 'history': history,
        }, f, indent=2)
