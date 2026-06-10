import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Config
MODEL_NAME = "bert-base-multilingual-cased"
DATA_DIR = "/workspace/comixattack/data/processed"
SAVE_DIR = "/workspace/comixattack/models/mbert_hinglish_hate"
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 5
LR = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL2ID = {"hate": 0, "not_hate": 1}
ID2LABEL = {0: "hate", 1: "not_hate"}

print(f"Device: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Dataset
class HinglishDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts = df['text'].tolist()
        self.labels = [LABEL2ID[l] for l in df['label'].tolist()]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Load data
print("Loading data...")
train_df = pd.read_csv(f"{DATA_DIR}/hate_train.csv")
dev_df   = pd.read_csv(f"{DATA_DIR}/hate_dev.csv")
test_df  = pd.read_csv(f"{DATA_DIR}/hate_test.csv")

print(f"Train: {len(train_df)} | Dev: {len(dev_df)} | Test: {len(test_df)}")
print(f"Label distribution:\n{train_df['label'].value_counts()}")

# Tokenizer
print(f"\nLoading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset = HinglishDataset(train_df, tokenizer)
dev_dataset   = HinglishDataset(dev_df, tokenizer)
test_dataset  = HinglishDataset(test_df, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
dev_loader   = DataLoader(dev_dataset,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

# Model
print(f"Loading model: {MODEL_NAME}")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label=ID2LABEL,
    label2id=LABEL2ID
)
model.to(DEVICE)

# Optimizer
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

# Eval function
def evaluate(loader):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    f1 = f1_score(all_labels, all_preds, average='macro')
    return f1, total_loss / len(loader)

# Training loop
print("\nStarting training...")
best_f1 = 0
os.makedirs(SAVE_DIR, exist_ok=True)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for step, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

        if (step + 1) % 50 == 0:
            print(f"Epoch {epoch+1} | Step {step+1}/{len(train_loader)} | Loss: {total_loss/(step+1):.4f}")

    dev_f1, dev_loss = evaluate(dev_loader)
    print(f"\nEpoch {epoch+1} complete | Dev F1: {dev_f1:.4f} | Dev Loss: {dev_loss:.4f}")

    if dev_f1 > best_f1:
        best_f1 = dev_f1
        model.save_pretrained(SAVE_DIR)
        tokenizer.save_pretrained(SAVE_DIR)
        print(f"✅ Best model saved (F1={best_f1:.4f})")

# Final test evaluation
print("\nLoading best model for test evaluation...")
model = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR)
model.to(DEVICE)
test_f1, test_loss = evaluate(test_loader)
print(f"\n{'='*50}")
print(f"FINAL TEST F1 (macro): {test_f1:.4f}")
print(f"{'='*50}")

# Detailed report
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch['labels'].numpy())

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=["hate","not_hate"]))
