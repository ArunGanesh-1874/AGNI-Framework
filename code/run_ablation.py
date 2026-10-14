import os, sys, random, json, time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

sys.path.insert(0, '/workspace/comixattack/code')
from modules_v2 import (InputProcessor, PerturbationEngine,
                         ValidationModule, SearchController, OutputGenerator)

SEED       = 42
OUTPUT_DIR = '/workspace/comixattack/results/ablation'
MODEL_DIR  = '/workspace/comixattack/models'
DATA_DIR   = '/workspace/comixattack/data/processed'

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ABLATION_CONDITIONS = [
    ('P2_only', ['P2'],                               'agni_p2only_xlmr_sentiment'),
    ('No_P2',   ['P1','P3','P4','P5','P6','P7','PC'], 'agni_nop2_xlmr_sentiment'),
    ('Full',    ['P1','P2','P3','P4','P5','P6','P7','PC'], 'agni_full_xlmr_sentiment'),
]

BUDGET     = 0.60
THRESHOLD  = 0.75
MODEL_PATH = f'{MODEL_DIR}/xlmr_hinglish_sentiment'
DATA_FILE  = 'hinglish_test_final.csv'
LABEL_MAP  = {'positive':0,'negative':1,'neutral':2}

labse  = SentenceTransformer('sentence-transformers/LaBSE', device='cuda')
device = torch.device('cuda')

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
vic_model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_PATH).to(device).eval()
id2label  = {v:k for k,v in LABEL_MAP.items()}

def predict(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True,
                      max_length=128, padding=True).to(device)
    with torch.no_grad():
        logits = vic_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    pred  = int(probs.argmax())
    return {'label': id2label[pred], 'confidence': float(probs[pred])}

def get_confidence(text, label):
    inputs = tokenizer(text, return_tensors='pt', truncation=True,
                      max_length=128, padding=True).to(device)
    with torch.no_grad():
        logits = vic_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    return float(probs[LABEL_MAP[label]])

df      = pd.read_csv(f'{DATA_DIR}/{DATA_FILE}').dropna(subset=['text'])
correct = [(t, predict(t)) for t in df['text'].tolist()]
correct = [(t, p) for t, p in correct if p['confidence'] > 0.5][:1000]
print(f'Selected {len(correct)} examples for ablation')

master_summary = []

for condition_name, active_types, exp_id in ABLATION_CONDITIONS:
    print(f'\n{"="*60}')
    print(f'ABLATION: {condition_name} — {active_types}')
    print(f'{"="*60}')

    processor  = InputProcessor(predict, get_confidence,
                                language_pair='hi-en')
    engine     = PerturbationEngine(language_pair='hi-en',
                                    active_types=active_types)
    validator  = ValidationModule(labse, predict,
                                  threshold=THRESHOLD, task='sentiment')
    searcher   = SearchController(predict, get_confidence, validator)
    output_gen = OutputGenerator(OUTPUT_DIR, exp_id, 'hi-en',
                                 'sentiment', 'xlmr')

    for i, (text, orig_pred) in enumerate(correct):
        processed = processor.process(text, budget=BUDGET)
        result    = searcher.search(processed, engine, budget=BUDGET)
        output_gen.save_example(result, i)
        if (i+1) % 100 == 0:
            s   = output_gen.stats
            asr = s['successful'] / max(s['total'], 1)
            avgt = s['total_time'] / max(s['total'], 1)
            print(f'  [{i+1}] ASR={asr:.1%} AvgT={avgt:.2f}s')

    output_gen.save_summary()
    s    = output_gen.stats
    tot  = s['total']
    asr  = s['successful'] / max(tot, 1)
    avgq = s['total_queries'] / max(tot, 1)

    summary = {
        'experiment_id': exp_id,
        'condition':     condition_name,
        'active_types':  active_types,
        'results': {
            'total': tot, 'successful': s['successful'],
            'asr':   round(asr, 4),
            'avg_queries': round(avgq, 2),
        }
    }
    master_summary.append(summary)
    print(f'\n✅ {condition_name} DONE — ASR={asr:.4f} AvgQ={avgq:.1f}')

    with open(f'{OUTPUT_DIR}/master_summary.json', 'w') as f:
        json.dump(master_summary, f, indent=2)

print(f'\n{"="*60}')
print('ABLATION RESULTS')
print(f'{"="*60}')
print(f'{"Condition":<12} {"Active Types":<50} {"ASR":>8} {"AvgQ":>8}')
print('-'*80)
for s in master_summary:
    r = s['results']
    print(f'{s["condition"]:<12} {str(s["active_types"]):<50} {r["asr"]:>8.4f} {r["avg_queries"]:>8.1f}')
