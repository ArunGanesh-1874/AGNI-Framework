"""
MATCHED-POPULATION controlled comparison (beta=0.40, tau=0.80).
Re-attacks EXACTLY the examples used in the original AGNI-Full (v4c) run, so ASR is
comparable. Fixes the MuRIL-hate anomaly (previous controlled run used all test rows,
including misclassified ones, vs. correctly-predicted subset in the original run).

    python code/run_controlled_matched.py hate        # ~1.5-2 h
    python code/run_controlled_matched.py sentiment   # ~3-4 h
    python code/run_controlled_matched.py all
Output: results/comixattack_controlled_matched/
"""
import os, sys, random, json, jsonlines
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
sys.path.insert(0, '/workspace/comixattack/code')
from modules_v2 import (InputProcessor, PerturbationEngine,
                        ValidationModule, SearchController, OutputGenerator)

SEED, BUDGET, THRESHOLD, LANGUAGE_PAIR = 42, 0.40, 0.80, 'hi-en'
BASE = '/workspace/comixattack'
MODEL_DIR, RES = f'{BASE}/models', f'{BASE}/results'
SRC = f'{RES}/comixattack_v4c_full'
OUTPUT_DIR = f'{RES}/comixattack_controlled_matched'
os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

LM = {'sentiment': {'positive': 0, 'negative': 1, 'neutral': 2},
      'hate': {'hate': 0, 'not_hate': 1}}
which = sys.argv[1] if len(sys.argv) > 1 else 'hate'
tasks = ['hate', 'sentiment'] if which == 'all' else [which]
labse = SentenceTransformer('sentence-transformers/LaBSE', device='cuda')
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
master = []

for task in tasks:
    label_map = LM[task]; id2label = {v: k for k, v in label_map.items()}
    for model_name in ['xlmr', 'muril', 'mbert']:
        exp_id = f'matched_{model_name}_{task}'
        src = f'{SRC}/comixattack_{model_name}_{task}_corpus.jsonl'
        if not os.path.exists(src):
            print('MISSING source corpus', src); continue
        with jsonlines.open(src) as f:
            texts = [r['original_text'] for r in f]      # same examples, same order
        print(f'\n=== {exp_id}: {len(texts)} examples from {src}')
        path = f'{MODEL_DIR}/{model_name}_hinglish_{task}'
        tok = AutoTokenizer.from_pretrained(path)
        mdl = AutoModelForSequenceClassification.from_pretrained(path).to(dev).eval()

        def probs(text):
            x = tok(text, return_tensors='pt', truncation=True, max_length=128, padding=True).to(dev)
            with torch.no_grad():
                return torch.softmax(mdl(**x).logits, dim=-1)[0]
        def predict(text):
            p = probs(text); i = int(p.argmax())
            return {'label': id2label[i], 'confidence': float(p[i])}
        def get_confidence(text, label):
            return float(probs(text)[label_map[label]])

        processor = InputProcessor(predict, get_confidence, language_pair=LANGUAGE_PAIR)
        engine = PerturbationEngine(language_pair=LANGUAGE_PAIR,
                                    active_types=['P1','P2','P3','P4','P5','P6','P7','PC'])
        validator = ValidationModule(labse, predict, threshold=THRESHOLD, task=task)
        searcher = SearchController(predict, get_confidence, validator)
        out = OutputGenerator(OUTPUT_DIR, exp_id, LANGUAGE_PAIR, task, model_name)

        n_correct = 0
        for i, text in enumerate(texts):
            n_correct += int(predict(text)['label'] in label_map)   # sanity only
            res = searcher.search(processor.process(text, budget=BUDGET), engine, budget=BUDGET)
            out.save_example(res, i)
            if (i + 1) % 100 == 0:
                s = out.stats; print(f'  [{i+1}] ASR={s["successful"]/max(s["total"],1):.1%}')
        out.save_summary()
        s = out.stats; tot = s['total']
        summ = dict(experiment_id=exp_id, task=task, victim_model=model_name, budget=BUDGET,
                    threshold=THRESHOLD, source=src,
                    results=dict(total=tot, successful=s['successful'], failed=s['failed'],
                                 asr=round(s['successful']/max(tot,1), 4),
                                 avg_queries=round(s['total_queries']/max(tot,1), 2)))
        master.append(summ)
        print('DONE', exp_id, summ['results'])
        json.dump(master, open(f'{OUTPUT_DIR}/master_summary_{which}.json', 'w'), indent=2)
        del mdl; torch.cuda.empty_cache()
