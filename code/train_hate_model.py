# Train XLM-R Victim Model on Hate Speech
# Same architecture as sentiment model

import torch
import numpy as np
import pandas as pd
import os
import json
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report
)
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

CONFIG = {
    'model_name':    'xlm-roberta-base',
    'num_labels':    2,
    'label_map':     {'hate': 0, 'not_hate': 1},
    'max_length':    128,
    'batch_size':    32,
    'learning_rate': 2e-5,
    'num_epochs':    5,
    'warmup_ratio':  0.1,
    'weight_decay':  0.01,
    'seed':          42,
    'data_path':     '/workspace/comixattack/data/processed',
    'save_path':     '/workspace/comixattack/models/xlmr_hinglish_hate'
}

os.makedirs(CONFIG['save_path'], exist_ok=True)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(CONFIG['seed'])
device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

print("Hate Speech Victim Model Training")
print("=" * 55)
print(f"Device: {device}")
print(f"Model:  {CONFIG['model_name']}")


# ─────────────────────────────────────────────
# DATASET CLASS
# ─────────────────────────────────────────────

class HateSpeechDataset(Dataset):
    def __init__(self, df, tokenizer, label_map, max_length):
        self.texts     = df['text'].tolist()
        self.labels    = [label_map[l] for l in df['label']]
        self.tokenizer = tokenizer
        self.max_len   = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'label':          torch.tensor(
                self.labels[idx], dtype=torch.long
            )
        }


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

print("\nLoading datasets...")
train_df = pd.read_csv(f"{CONFIG['data_path']}/hate_train.csv").dropna()
dev_df   = pd.read_csv(f"{CONFIG['data_path']}/hate_dev.csv").dropna()
test_df  = pd.read_csv(f"{CONFIG['data_path']}/hate_test.csv").dropna()

print(f"Train: {len(train_df)}  Dev: {len(dev_df)}  Test: {len(test_df)}")


# ─────────────────────────────────────────────
# TOKENIZER AND MODEL
# ─────────────────────────────────────────────

print("\nLoading XLM-R...")
tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
model     = AutoModelForSequenceClassification.from_pretrained(
    CONFIG['model_name'],
    num_labels=CONFIG['num_labels']
).to(device)
print("✓ Model loaded")


# ─────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────

train_dataset = HateSpeechDataset(
    train_df, tokenizer,
    CONFIG['label_map'], CONFIG['max_length']
)
dev_dataset = HateSpeechDataset(
    dev_df, tokenizer,
    CONFIG['label_map'], CONFIG['max_length']
)
test_dataset = HateSpeechDataset(
    test_df, tokenizer,
    CONFIG['label_map'], CONFIG['max_length']
)

train_loader = DataLoader(
    train_dataset, batch_size=CONFIG['batch_size'],
    shuffle=True
)
dev_loader = DataLoader(
    dev_dataset,
    batch_size=CONFIG['batch_size'] * 2,
    shuffle=False
)
test_loader = DataLoader(
    test_dataset,
    batch_size=CONFIG['batch_size'] * 2,
    shuffle=False
)

print(f"Train batches: {len(train_loader)}")


# ─────────────────────────────────────────────
# OPTIMIZER AND SCHEDULER
# ─────────────────────────────────────────────

optimizer     = AdamW(
    model.parameters(),
    lr=CONFIG['learning_rate'],
    weight_decay=CONFIG['weight_decay']
)
total_steps   = len(train_loader) * CONFIG['num_epochs']
warmup_steps  = int(total_steps * CONFIG['warmup_ratio'])
scheduler     = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)


# ─────────────────────────────────────────────
# EVALUATION FUNCTION
# ─────────────────────────────────────────────

def evaluate(model, loader):
    model.eval()
    all_preds  = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            ids    = batch['input_ids'].to(device)
            mask   = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(
                input_ids=ids,
                attention_mask=mask,
                labels=labels
            )
            total_loss += outputs.loss.item()
            preds = torch.argmax(
                outputs.logits, dim=-1
            ).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, average='macro')
    return avg_loss, acc, f1, all_preds, all_labels


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("STARTING TRAINING")
print("=" * 55)

best_f1    = 0.0
best_epoch = 0

for epoch in range(CONFIG['num_epochs']):
    model.train()
    total_loss = 0
    correct    = 0
    total      = 0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}"
    )

    for batch in progress:
        optimizer.zero_grad()
        ids    = batch['input_ids'].to(device)
        mask   = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            labels=labels
        )
        loss  = outputs.loss
        preds = torch.argmax(outputs.logits, dim=-1)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0
        )
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

        progress.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc':  f'{correct/total:.3f}'
        })

    dev_loss, dev_acc, dev_f1, _, _ = evaluate(
        model, dev_loader
    )
    avg_train_loss = total_loss / len(train_loader)

    print(f"\nEpoch {epoch+1}:")
    print(f"  Train Loss: {avg_train_loss:.4f} "
          f"Acc: {correct/total:.4f}")
    print(f"  Dev   Loss: {dev_loss:.4f} "
          f"Acc: {dev_acc:.4f} F1: {dev_f1:.4f}")

    if dev_f1 > best_f1:
        best_f1    = dev_f1
        best_epoch = epoch + 1
        model.save_pretrained(CONFIG['save_path'])
        tokenizer.save_pretrained(CONFIG['save_path'])
        print(f"  ✓ Best model saved (F1: {best_f1:.4f})")


# ─────────────────────────────────────────────
# TEST EVALUATION
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("TEST SET EVALUATION")
print("=" * 55)

from transformers import AutoModelForSequenceClassification
best_model = AutoModelForSequenceClassification.from_pretrained(
    CONFIG['save_path']
).to(device)

_, test_acc, test_f1, preds, labels = evaluate(
    best_model, test_loader
)

label_names = ['hate', 'not_hate']
report = classification_report(
    labels, preds, target_names=label_names
)

print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test F1 Macro: {test_f1:.4f}")
print(f"\n{report}")

results = {
    'model':         CONFIG['model_name'],
    'task':          'hinglish_hate_speech',
    'best_epoch':    best_epoch,
    'best_dev_f1':   round(best_f1, 4),
    'test_accuracy': round(test_acc, 4),
    'test_f1_macro': round(test_f1, 4),
}

with open(f"{CONFIG['save_path']}/results.json", 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Model saved to: {CONFIG['save_path']}")
print(f"\nHATE SPEECH VICTIM MODEL TRAINING COMPLETE")
print(f"Best epoch: {best_epoch} | F1: {test_f1:.4f}")
