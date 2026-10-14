import os, sys, random, json, time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

sys.path.insert(0, '/workspace/comixattack/code')
from modules_v2 import (InputProcessor, PerturbationEngine,
                         ValidationModule, OutputGenerator)
from search_v2 import SearchControllerV2

# ─── Reproducibility ───
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR   = "/workspace/comixattack/data/processed"
MODEL_DIR  = "/workspace/comixattack/models"
OUTPUT_DIR = "/workspace/comixattack/results/comixattack_v4_full"
N_EXAMPLES = 1000
LANGUAGE_PAIR = 'hi-en'

os.makedirs(OUTPUT_DIR, exist_ok=True)

EXPERIMENTS = [
    ('sentiment', 'xlmr',  f'{MODEL_DIR}/xlmr_hinglish_sentiment',
     'hinglish_test_final.csv',
     {'positive':0,'negative':1,'neutral':2}),
    ('sentiment', 'muril', f'{MODEL_DIR}/muril_hinglish_sentiment',
     'hinglish_test_final.csv',
     {'positive':0,'negative':1,'neutral':2}),
    ('sentiment', 'mbert', f'{MODEL_DIR}/mbert_hinglish_sentiment',
     'hinglish_test_final.csv',
     {'positive':0,'negative':1,'neutral':2}),
    ('hate',      'xlmr',  f'{MODEL_DIR}/xlmr_hinglish_hate',
     'hate_test.csv',
     {'hate':0,'not_hate':1}),
    ('hate',      'muril', f'{MODEL_DIR}/muril_hinglish_hate',
     'hate_test.csv',
     {'hate':0,'not_hate':1}),
    ('hate',      'mbert', f'{MODEL_DIR}/mbert_hinglish_hate',
     'hate_test.csv',
     {'hate':0,'not_hate':1}),
]

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(DEVICE).eval()
    return tokenizer, model

def make_predict_fn(tokenizer, model, id2label):
    def predict(text):
        enc = tokenizer(text, return_tensors='pt',
                        max_length=128, truncation=True,
                        padding=True).to(DEVICE)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred  = int(np.argmax(probs))
        return {'label': id2label[pred],
                'confidence': float(probs[pred]),
                'probs': probs.tolist()}
    return predict

def make_conf_fn(tokenizer, model, label2id):
    def get_conf(text, label):
        enc = tokenizer(text, return_tensors='pt',
                        max_length=128, truncation=True,
                        padding=True).to(DEVICE)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        return float(probs[label2id.get(label, 0)])
    return get_conf

print("Loading LaBSE model...")
labse = SentenceTransformer('sentence-transformers/LaBSE', device='cpu')
print("LaBSE loaded ✅")

all_summaries = []

for (task, model_name, model_path, test_file, label2id) in EXPERIMENTS:
    exp_id = f"comixattack_v4_{model_name}_{task}"
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {exp_id}")
    print(f"{'='*60}")

    tokenizer, model = load_model(model_path)
    id2label    = {v: k for k, v in label2id.items()}
    predict_fn  = make_predict_fn(tokenizer, model, id2label)
    get_conf_fn = make_conf_fn(tokenizer, model, label2id)

    df = pd.read_csv(f"{DATA_DIR}/{test_file}")
    df = df.dropna(subset=['text','label'])
    df = df[df['text'].str.strip().str.len() > 10]

    print(f"Filtering correctly predicted examples from {len(df)}...")
    correct_rows = []
    for _, row in df.iterrows():
        if len(correct_rows) >= N_EXAMPLES * 2:
            break
        result = predict_fn(str(row['text']))
        if result['label'] == row['label']:
            correct_rows.append(row)

    random.seed(SEED)
    random.shuffle(correct_rows)
    selected = correct_rows[:N_EXAMPLES]
    print(f"Selected {len(selected)} examples")

    # Setup modules
    engine    = PerturbationEngine(
        language_pair=LANGUAGE_PAIR,
        active_types=['P1','P2','P3','P4','P5','P6','P7','PC']
    )
    validator = ValidationModule(
        labse, predict_fn, threshold=0.75, task=task
    )
    processor = InputProcessor(
        predict_fn, get_conf_fn, language_pair=LANGUAGE_PAIR
    )

    # Use new SearchControllerV2
    searcher  = SearchControllerV2(
        predict_fn, get_conf_fn, validator,
        beam_size=3,
        max_iterations=3,
        confidence_threshold=0.40
    )

    output_gen = OutputGenerator(
        OUTPUT_DIR, exp_id,
        LANGUAGE_PAIR, task, model_name
    )

    print(f"Running CoMixAttack v4 on {len(selected)} examples...")

    for idx, row in enumerate(selected):
        try:
            text      = str(row['text'])
            processed = processor.process(text, budget=0.60)
            result    = searcher.search(processed, engine, budget=0.60)
            output_gen.save_example(result, idx)
            output_gen.print_progress(interval=100)
        except Exception as e:
            print(f"  [SKIP] idx={idx} error: {e}", flush=True)
            continue

    summary = output_gen.save_summary()
    all_summaries.append(summary)

    r = summary['results']
    print(f"\n✅ {exp_id} DONE")
    print(f"   ASR     : {r['asr']:.4f} ({r['successful']}/{r['total']})")
    print(f"   AvgQ    : {r['avg_queries']:.1f}")
    print(f"   AvgTime : {r['avg_time']:.3f}s")

    del model
    torch.cuda.empty_cache()

# ─── Final table ───
print(f"\n{'='*60}")
print("FINAL RESULTS — CoMixAttack v4 (Multi-word + Iterative)")
print(f"{'='*60}")
print("Experiment                          ASR    AvgQ    AvgT")
print("-"*60)
for s in all_summaries:
    r    = s['results']
    name = s['victim_model'] + '_' + s['task']
    print(name.ljust(35),
          str(round(r['asr'],4)).rjust(6),
          str(round(r['avg_queries'],1)).rjust(6),
          str(round(r['avg_time'],3)).rjust(8)+'s')

master_path = f"{OUTPUT_DIR}/master_summary.json"
with open(master_path, 'w') as f:
    json.dump(all_summaries, f, indent=2)
print(f"\n✅ Master summary: {master_path}")
