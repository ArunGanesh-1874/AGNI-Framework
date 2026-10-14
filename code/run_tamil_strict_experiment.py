import os, sys, random, json, time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

sys.path.insert(0, '/workspace/comixattack/code')
from modules_v2 import (InputProcessor, PerturbationEngine,
                     ValidationModule, SearchController, OutputGenerator)

# ─── Reproducibility ───
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR   = "/workspace/comixattack/data/processed"
MODEL_DIR  = "/workspace/comixattack/models"
OUTPUT_DIR = "/workspace/comixattack/results/comixattack_tamil_strict_full"
N_EXAMPLES = 1000
LANGUAGE_PAIR = 'ta-en'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Model registry ───
EXPERIMENTS = [
    ('sentiment', 'xlmr',  f'{MODEL_DIR}/xlmr_tamil_sentiment',
     'tamil_sentiment_test.csv', {'positive':0,'negative':1,'neutral':2}),
    ('sentiment', 'muril', f'{MODEL_DIR}/muril_tamil_sentiment',
     'tamil_sentiment_test.csv', {'positive':0,'negative':1,'neutral':2}),
    ('sentiment', 'mbert', f'{MODEL_DIR}/mbert_tamil_sentiment',
     'tamil_sentiment_test.csv', {'positive':0,'negative':1,'neutral':2}),
    ('offensive', 'xlmr',  f'{MODEL_DIR}/xlmr_tamil_offensive',
     'tamil_offensive_test.csv', {'offensive':0,'not_offensive':1}),
    ('offensive', 'muril', f'{MODEL_DIR}/muril_tamil_offensive',
     'tamil_offensive_test.csv', {'offensive':0,'not_offensive':1}),
    ('offensive', 'mbert', f'{MODEL_DIR}/mbert_tamil_offensive',
     'tamil_offensive_test.csv', {'offensive':0,'not_offensive':1}),
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

def make_conf_fn(predict_fn):
    def get_conf(text, label):
        result = predict_fn(text)
        for k, v in result.items():
            pass
        probs = result['probs']
        label2id = {}
        r2 = predict_fn(text)
        return r2['confidence'] if r2['label'] == label else \
               1.0 - r2['confidence']
    return get_conf

def make_conf_fn_v2(tokenizer, model, label2id):
    def get_conf(text, label):
        enc = tokenizer(text, return_tensors='pt',
                        max_length=128, truncation=True,
                        padding=True).to(DEVICE)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        idx   = label2id.get(label, 0)
        return float(probs[idx])
    return get_conf

print("Loading LaBSE model...")
labse = SentenceTransformer('sentence-transformers/LaBSE',
                             device='cpu')
print("LaBSE loaded ✅")

# ─── Main loop ───
all_summaries = []

for (task, model_name, model_path, test_file, label2id) in EXPERIMENTS:
    exp_id = f"comixattack_{model_name}_{task}"
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {exp_id}")
    print(f"{'='*60}")

    # Load model
    tokenizer, model = load_model(model_path)
    id2label = {v:k for k,v in label2id.items()}

    predict_fn  = make_predict_fn(tokenizer, model, id2label)
    get_conf_fn = make_conf_fn_v2(tokenizer, model, label2id)

    # Load test data — only correctly predicted examples
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
    print(f"Selected {len(selected)} correctly predicted examples")

    # Setup modules
    engine    = PerturbationEngine(language_pair=LANGUAGE_PAIR,
                                   active_types=['P1','P2','P3','P4','P5','P6','P7','PC'])
    validator = ValidationModule(labse, predict_fn,
                                 threshold=0.80, task=task)
    processor = InputProcessor(predict_fn, get_conf_fn,
                               language_pair=LANGUAGE_PAIR)
    searcher  = SearchController(predict_fn, get_conf_fn, validator)
    output_gen = OutputGenerator(OUTPUT_DIR, exp_id,
                                 LANGUAGE_PAIR, task, model_name)

    # Run attack
    print(f"Running CoMixAttack on {len(selected)} examples...")
    start_total = time.time()

    for idx, row in enumerate(selected):
        try:
            text = str(row['text'])
            processed = processor.process(text, budget=0.60)
            result    = searcher.search(processed, engine, budget=0.60)
            output_gen.save_example(result, idx)
            output_gen.print_progress(interval=100)
        except Exception as e:
            print(f"  [SKIP] idx={idx} error: {e}")
            continue

    total_time = time.time() - start_total
    summary    = output_gen.save_summary()
    summary['total_wall_time'] = round(total_time, 1)
    all_summaries.append(summary)

    r = summary['results']
    print(f"\n✅ {exp_id} DONE")
    print(f"   ASR     : {r['asr']:.4f} ({r['successful']}/{r['total']})")
    print(f"   AvgQ    : {r['avg_queries']:.1f}")
    print(f"   AvgTime : {r['avg_time']:.3f}s")
    print(f"   P-types : P1={r['p1_used']} P2={r['p2_used']} "
          f"P3={r['p3_used']} P4={r['p4_used']} P5={r['p5_used']}")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

# ─── Final summary table ───
print(f"\n{'='*60}")
print("FINAL RESULTS — CoMixAttack P1-P5")
print(f"{'='*60}")
print(f"{'Experiment':<35} {'ASR':>6} {'AvgQ':>6} {'AvgT':>6}")
print("-"*60)
for s in all_summaries:
    r = s['results']
    name = f"{s['victim_model']}_{s['task']}"
    print(f"{name:<35} {r['asr']:>6.4f} {r['avg_queries']:>6.1f} {r['avg_time']:>6.3f}s")

# Save master summary
master_path = f"{OUTPUT_DIR}/master_summary.json"
with open(master_path, 'w') as f:
    json.dump(all_summaries, f, indent=2)
print(f"\n✅ Master summary saved: {master_path}")
