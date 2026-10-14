import os, sys, random, json, time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

sys.path.insert(0, '/workspace/comixattack/code')
from modules_v2 import (InputProcessor, PerturbationEngine,
                         ValidationModule, SearchController, OutputGenerator)

SEED          = 42
OUTPUT_DIR    = '/workspace/comixattack/results/comixattack_controlled'
MODEL_DIR     = '/workspace/comixattack/models'
DATA_DIR      = '/workspace/comixattack/data/processed'
LANGUAGE_PAIR = 'hi-en'
BUDGET        = 0.40
THRESHOLD     = 0.80

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load existing sentiment results
master_summary = []
summary_path = f'{OUTPUT_DIR}/master_summary.json'
if os.path.exists(summary_path):
    with open(summary_path) as f:
        master_summary = json.load(f)
    print(f'Loaded {len(master_summary)} existing results')

EXPERIMENTS = [
    ('hate', 'xlmr',  f'{MODEL_DIR}/xlmr_hinglish_hate',
     'hate_test.csv', {'hate':0,'not_hate':1}),
    ('hate', 'muril', f'{MODEL_DIR}/muril_hinglish_hate',
     'hate_test.csv', {'hate':0,'not_hate':1}),
    ('hate', 'mbert', f'{MODEL_DIR}/mbert_hinglish_hate',
     'hate_test.csv', {'hate':0,'not_hate':1}),
]

labse = SentenceTransformer('sentence-transformers/LaBSE', device='cuda')

for task, model_name, model_path, data_file, label_map in EXPERIMENTS:
    exp_id = f'controlled_{model_name}_{task}'
    print(f'\n{"="*60}')
    print(f'EXPERIMENT: {exp_id}')
    print(f'Budget={BUDGET}, Threshold={THRESHOLD}')
    print(f'{"="*60}')

    device    = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    vic_model = AutoModelForSequenceClassification.from_pretrained(
                    model_path).to(device).eval()
    id2label  = {v:k for k,v in label_map.items()}

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
        return float(probs[label_map[label]])

    df      = pd.read_csv(f'{DATA_DIR}/{data_file}').dropna(subset=['text'])
    correct = [(t, predict(t)) for t in df['text'].tolist()]
    correct = [(t, p) for t, p in correct if p['confidence'] > 0.5][:1000]
    print(f'Selected {len(correct)} examples')

    processor  = InputProcessor(predict, get_confidence,
                                language_pair=LANGUAGE_PAIR)
    engine     = PerturbationEngine(language_pair=LANGUAGE_PAIR,
                                    active_types=['P1','P2','P3','P4',
                                                  'P5','P6','P7','PC'])
    validator  = ValidationModule(labse, predict,
                                  threshold=THRESHOLD, task=task)
    searcher   = SearchController(predict, get_confidence, validator)
    output_gen = OutputGenerator(OUTPUT_DIR, exp_id, LANGUAGE_PAIR,
                                 task, model_name)

    for i, (text, orig_pred) in enumerate(correct):
        processed = processor.process(text, budget=BUDGET)
        result    = searcher.search(processed, engine, budget=BUDGET)
        output_gen.save_example(result, i)
        if (i+1) % 100 == 0:
            s    = output_gen.stats
            asr  = s['successful'] / max(s['total'], 1)
            avgt = s['total_time'] / max(s['total'], 1)
            print(f'  [{i+1}] ASR={asr:.1%} AvgT={avgt:.2f}s')

    output_gen.save_summary()
    s    = output_gen.stats
    tot  = s['total']
    asr  = s['successful'] / max(tot, 1)
    avgq = s['total_queries'] / max(tot, 1)

    summary = {
        'experiment_id': exp_id,
        'task': task, 'victim_model': model_name,
        'budget': BUDGET, 'threshold': THRESHOLD,
        'results': {
            'total': tot, 'successful': s['successful'],
            'failed': s['failed'],
            'asr': round(asr, 4),
            'avg_queries': round(avgq, 2),
        }
    }
    master_summary.append(summary)
    print(f'\n✅ {exp_id} DONE — ASR={asr:.4f} AvgQ={avgq:.1f}')

    with open(summary_path, 'w') as f:
        json.dump(master_summary, f, indent=2)

print(f'\n{"="*60}')
print('FINAL RESULTS — Controlled Hate Experiments')
print(f'{"="*60}')
for s in master_summary:
    r = s['results']
    print(f'{s["experiment_id"]:<35} ASR={r["asr"]:.4f} AvgQ={r["avg_queries"]:.1f}')
