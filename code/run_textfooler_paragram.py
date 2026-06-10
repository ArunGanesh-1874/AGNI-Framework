# TextFooler Implementation using Paragram Embeddings
# Faithful reimplementation of Jin et al. 2020
# USE replaced with LaBSE (valid for multilingual text)
# Paragram-CF embeddings from TextAttack cache

import sys
sys.path.append('/workspace/comixattack/code')

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random
import json
import os
import time
import jsonlines
import pickle
from datetime import datetime
from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("TextFooler (Paragram+LaBSE) Implementation")
print(f"Random seed: {SEED}")
print("=" * 55)

MODEL_PATH    = '/workspace/comixattack/models/xlmr_hinglish_sentiment'
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'
LABEL_MAP     = {0: 'positive', 1: 'negative', 2: 'neutral'}
LABEL_MAP_REV = {'positive': 0, 'negative': 1, 'neutral': 2}

print(f"Device: {DEVICE}")
tokenizer    = AutoTokenizer.from_pretrained(MODEL_PATH)
victim_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
).to(DEVICE)
victim_model.eval()
print("✓ Victim model loaded")

print("Loading LaBSE...")
labse = SentenceTransformer(
    'sentence-transformers/LaBSE', device='cpu'
)
print("✓ LaBSE loaded")

def predict(text):
    inputs = tokenizer(
        text, return_tensors='pt',
        max_length=128, truncation=True, padding=True
    ).to(DEVICE)
    with torch.no_grad():
        outputs  = victim_model(**inputs)
        probs    = F.softmax(outputs.logits, dim=-1)
        pred_idx = torch.argmax(probs).item()
        conf     = probs[0][pred_idx].item()
    return {
        'label':      LABEL_MAP[pred_idx],
        'confidence': round(conf, 4),
        'probs':      probs[0].tolist()
    }

def get_conf(text, label):
    inputs = tokenizer(
        text, return_tensors='pt',
        max_length=128, truncation=True, padding=True
    ).to(DEVICE)
    with torch.no_grad():
        outputs = victim_model(**inputs)
        probs   = F.softmax(outputs.logits, dim=-1)
    return round(probs[0][LABEL_MAP_REV[label]].item(), 4)

def labse_sim(t1, t2):
    emb = labse.encode(
        [t1, t2], show_progress_bar=False, device='cpu'
    )
    return float(cosine_similarity([emb[0]], [emb[1]])[0][0])

# ─────────────────────────────────────────────
# LOAD PARAGRAM EMBEDDINGS
# Already downloaded by TextAttack
# ─────────────────────────────────────────────

print("\nLoading Paragram-CF embeddings...")
paragram_path = '/root/.cache/textattack/word_embeddings/paragramcf'

# Load word to index mapping and embeddings
try:
    # TextAttack stores paragram as numpy files
    import glob
    files = glob.glob(paragram_path + '/**/*', recursive=True)
    print(f"Files in paragram cache: {len(files)}")
    for f in files[:10]:
        print(f"  {f}")
except Exception as e:
    print(f"Error: {e}")

print("\nChecking paragram structure...")
import os
if os.path.exists(paragram_path):
    for item in os.listdir(paragram_path):
        full = os.path.join(paragram_path, item)
        size = os.path.getsize(full) if os.path.isfile(full) else 0
        print(f"  {item} ({size/1e6:.1f}MB)")
