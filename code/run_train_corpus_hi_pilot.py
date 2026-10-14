import os, sys, random, json, time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
sys.path.insert(0, '/workspace/comixattack/code')
from modules_v2 import (InputProcessor, PerturbationEngine,
                     ValidationModule, SearchController, OutputGenerator)

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR   = "/workspace/comixattack/data/processed"
MODEL_DIR  = "/workspace/comixattack/models"
OUTPUT_DIR = "/workspace/comixattack/results/pilot_defense_hi_train"
N_EXAMPLES = 1000
LANGUAGE_PAIR = 'hi-en'
os.makedirs(OUTPUT_DIR, exist_ok=True)

EXPERIMENTS = [
    ('sentiment', 'xlmr', f'{MODEL_DIR}/xlmr_hinglish_sentiment',
     'hinglish_train_final.csv',
     {'positive':0,'negative':1,'neutral':2}),
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
labse = SentenceTransformer('sentence-transformers/LaBSE', device='cpu')
print("LaBSE loaded")

all_summaries = []
for (task, model_name, model_path, data_file, label2id) in EXPERIMENTS:
    exp_id = f"comixattack_{model_name}_{task}_TRAIN_hi"
    print(f"\n{'='*60}\nEXPERIMENT: {exp_id}\n{'='*60}")

    tokenizer, model = load_model(model_path)
    id2label = {v:k for k,v in label2id.items()}
    predict_fn  = make_predict_fn(tokenizer, model, id2label)
    get_conf_fn = make_conf_fn_v2(tokenizer, model, label2id)

    df = pd.read_csv(f"{DATA_DIR}/{data_file}")
    df = df.dropna(subset=['text','label'])
    df = df[df['text'].str.strip().str.len() > 10]
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

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

    engine    = PerturbationEngine(language_pair=LANGUAGE_PAIR,
                                   active_types=['P1','P2','P3','P4','P5','P6','P7','PC'])
    validator = ValidationModule(labse, predict_fn, threshold=0.80, task=task)
    processor = InputProcessor(predict_fn, get_conf_fn, language_pair=LANGUAGE_PAIR)
    searcher  = SearchController(predict_fn, get_conf_fn, validator)
    output_gen = OutputGenerator(OUTPUT_DIR, exp_id, LANGUAGE_PAIR, task, model_name)

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
    print(f"\nDONE: {exp_id}")
    print(f"   ASR     : {r['asr']:.4f} ({r['successful']}/{r['total']})")
    print(f"   AvgQ    : {r['avg_queries']:.1f}")
    print(f"   AvgTime : {r['avg_time']:.3f}s")

    del model
    torch.cuda.empty_cache()

master_path = f"{OUTPUT_DIR}/master_summary.json"
with open(master_path, 'w') as f:
    json.dump(all_summaries, f, indent=2)
print(f"\nMaster summary saved: {master_path}")
