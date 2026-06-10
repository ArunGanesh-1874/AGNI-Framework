# CoMixAttack — Main Research Script
# PhD Research: Adversarial Attacks on Code-Mixed NLP
# Scholar: Arun Kumar G (24PHD0774)

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import jsonlines
import json
import os
import time
from tqdm import tqdm
from typing import List, Dict, Tuple

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

CONFIG = {
    'model_path':    '/workspace/comixattack/models/xlmr_hinglish_sentiment',
    'data_path':     '/workspace/comixattack/data/processed',
    'output_path':   '/workspace/comixattack/data/adversarial',
    'results_path':  '/workspace/comixattack/results',
    'language_pair': 'hi-en',
    'task':          'sentiment',
    'budget':        0.30,
    'labse_threshold': 0.80,
    'device':        'cuda' if torch.cuda.is_available() else 'cpu'
}

LABEL_MAP     = {0: 'positive', 1: 'negative', 2: 'neutral'}
LABEL_MAP_REV = {'positive': 0, 'negative': 1, 'neutral': 2}

print("CoMixAttack — Initializing")
print("=" * 55)
print(f"Device: {CONFIG['device']}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────

print("\nLoading victim model...")
tokenizer    = AutoTokenizer.from_pretrained(CONFIG['model_path'])
victim_model = AutoModelForSequenceClassification.from_pretrained(
    CONFIG['model_path']
).to(CONFIG['device'])
victim_model.eval()
print("✓ Victim model loaded")

print("Loading LaBSE on CPU...")
labse = SentenceTransformer(
    'sentence-transformers/LaBSE',
    device='cpu'
)
print("✓ LaBSE loaded")

# ─────────────────────────────────────────────
# PREDICTION FUNCTION
# ─────────────────────────────────────────────

def predict(text: str) -> Dict:
    inputs = tokenizer(
        text, return_tensors='pt',
        max_length=128, truncation=True, padding=True
    ).to(CONFIG['device'])
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

def get_label_confidence(text: str, label: str) -> float:
    inputs = tokenizer(
        text, return_tensors='pt',
        max_length=128, truncation=True, padding=True
    ).to(CONFIG['device'])
    with torch.no_grad():
        outputs = victim_model(**inputs)
        probs   = F.softmax(outputs.logits, dim=-1)
    return round(probs[0][LABEL_MAP_REV[label]].item(), 4)

print("\n✓ All models ready")
print("=" * 55)
