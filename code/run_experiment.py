import sys
sys.path.append('/workspace/comixattack/code')

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import pandas as pd
import jsonlines
import json
import os
import re
import time
from tqdm import tqdm
from modules import (
    InputProcessor, PerturbationEngine,
    ValidationModule, SearchController, OutputGenerator
)

MODEL_PATH = '/workspace/comixattack/models/xlmr_hinglish_sentiment'
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Loading models...")
tokenizer    = AutoTokenizer.from_pretrained(MODEL_PATH)
victim_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
).to(DEVICE)
victim_model.eval()

LABEL_MAP     = {0: 'positive', 1: 'negative', 2: 'neutral'}
LABEL_MAP_REV = {'positive': 0, 'negative': 1, 'neutral': 2}

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

def get_label_confidence(text, label):
    inputs = tokenizer(
        text, return_tensors='pt',
        max_length=128, truncation=True, padding=True
    ).to(DEVICE)
    with torch.no_grad():
        outputs = victim_model(**inputs)
        probs   = F.softmax(outputs.logits, dim=-1)
    return round(probs[0][LABEL_MAP_REV[label]].item(), 4)

print("Loading LaBSE...")
labse = SentenceTransformer('sentence-transformers/LaBSE', device='cpu')
print("All models ready")

EXPANDED_HINDI_LEXICON = {
    'yaar','bhai','kya','nahi','nahin','haan','han',
    'achha','acha','theek','thik','bahut','bohot',
    'main','mein','mai','tum','aap','woh','hum',
    'hai','hain','tha','thi','the','hoga','hogi',
    'karo','karna','karta','karti','jana','aana',
    'aaj','kal','phir','ab','abhi','sirf','bas',
    'dost','ghar','log','desh','pyaar','pyar','dil',
    'mera','tera','uska','hamara','apna','unka',
    'khush','udaas','sach','acchi','buri','bura',
    'ke','ki','ka','ko','se','ne','pe','toh','bhi',
    'hi','wala','wali','wale','mujhe','tumhe','usse',
    'zyada','thoda','bilkul','zaroor','ekdum',
    'sahi','galat','bara','chota','maza','maja',
    'woh','wo','usne','unhe','chahiye','karega',
    'toh','to','tou','fir','kuch','koi','sab',
    'mere','tere','uske','hamare','isko','usko',
    'yeh','ye','kaafi','bohot','bohat','bahot',
    'achhe','acche','bure','bare','chote','naya',
    'nayi','purana','mast','zabardast','kamaal',
    'hua','hui','hue','raha','rahi','gaya','gayi',
    'liya','diya','kiya','dekha','suna','batao',
    'pata','maloom','samajh','logo','logon','dosto',
    'kaam','paisa','paise','waqt','raat','subah',
    'khana','paani','chai','dekho','suno','bolo',
    'jao','aao','ruko','mat','pakka','shayad',
    'hamesha','kabhi','accha','bekar','bakwaas',
    'sundar','khubsoorat','jabardast','aya','ayi',
}

def clean_text(text):
    tokens = text.split()
    clean  = []
    for t in tokens:
        if t.startswith('http') or t.startswith('//'):
            continue
        if t.startswith('@') or t.startswith('#'):
            continue
        if t.isdigit() or len(t) <= 1:
            continue
        if all(not c.isalnum() for c in t):
            continue
        clean.append(t)
    return ' '.join(clean)

processor = InputProcessor(predict, get_label_confidence, 'hi-en')
processor.lang_det.HINDI_LEXICON = EXPANDED_HINDI_LEXICON
engine    = PerturbationEngine('hi-en')
validator = ValidationModule(labse, predict, 0.80, 'sentiment')
searcher  = SearchController(predict, get_label_confidence, validator)

EXPERIMENT_ID = 'E21_v2_hi_en_sentiment_xlmr'
output_dir    = '/workspace/comixattack/data/adversarial/hi_en_v2'
os.makedirs(output_dir, exist_ok=True)

output_gen = OutputGenerator(
    output_dir, EXPERIMENT_ID,
    'hi-en', 'sentiment', 'xlmr-hinglish'
)

test_df = pd.read_csv(
    '/workspace/comixattack/data/processed/hinglish_test_final.csv'
).head(500)

print(f"Test set: {len(test_df)} sentences")
print("Starting attack...")
print("=" * 55)

stats            = {'total':0,'skipped':0,'attacked':0,'successful':0}
experiment_start = time.time()

for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
    text       = str(row['text'])
    gold_label = str(row['label'])
    stats['total'] += 1

    clean = clean_text(text)
    if len(clean.split()) < 3:
        stats['skipped'] += 1
        continue

    orig = predict(text)
    if orig['label'] != gold_label:
        stats['skipped'] += 1
        continue

    try:
        processed = processor.process(clean, budget=0.40)
        processed['original_text']       = text
        processed['original_prediction'] = orig['label']
        processed['original_confidence'] = orig['confidence']
        result = searcher.search(processed, engine, budget=0.40)
        result['original_text'] = text
    except Exception:
        stats['skipped'] += 1
        continue

    output_gen.save_example(result, idx)
    stats['attacked'] += 1
    if result['attack_successful']:
        stats['successful'] += 1

    if stats['attacked'] % 50 == 0:
        attacked = stats['attacked']
        asr      = stats['successful']/attacked if attacked>0 else 0
        elapsed  = time.time() - experiment_start
        tqdm.write(
            f"[{attacked}] ASR={asr:.1%} "
            f"Success={stats['successful']} "
            f"Time={elapsed/60:.1f}min"
        )

total_time = time.time() - experiment_start
attacked   = stats['attacked']
asr        = stats['successful']/attacked if attacked>0 else 0

print("\n" + "=" * 55)
print("EXPERIMENT COMPLETE")
print("=" * 55)
print(f"Total:      {stats['total']}")
print(f"Skipped:    {stats['skipped']}")
print(f"Attacked:   {attacked}")
print(f"Successful: {stats['successful']}")
print(f"ASR:        {asr:.4f} ({asr:.1%})")
print(f"Time:       {total_time/60:.1f} minutes")
output_gen.save_summary()
