import os, sys, random, json, time
import numpy as np
import pandas as pd
import torch
import jsonlines
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# ─── Reproducibility ───
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR   = "/workspace/comixattack/data/processed"
MODEL_DIR  = "/workspace/comixattack/models"
OUTPUT_DIR = "/workspace/comixattack/results/advcodemix_labse_baseline"
N_EXAMPLES = 1000

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Device: {DEVICE}")
print("AdvCodeMix baseline — WITH LaBSE filter (controlled comparison)")

# ─── Load LaBSE ───
SIM_THRESH = 0.80
print("Loading LaBSE...")
labse = SentenceTransformer('sentence-transformers/LaBSE', device='cpu')
print("LaBSE loaded ✅")

def labse_sim_batch(original_text, candidate_texts):
    all_texts  = [original_text] + candidate_texts
    embeddings = labse.encode(all_texts, show_progress_bar=False,
                               batch_size=32, device='cpu')
    sims = cosine_similarity([embeddings[0]], embeddings[1:])[0]
    return sims

# ─── Character perturbations (same as AdvCodeMix paper) ───
def swap_chars(word):
    if len(word) < 3: return []
    results = []
    for i in range(len(word)-1):
        w = list(word)
        w[i], w[i+1] = w[i+1], w[i]
        p = ''.join(w)
        if p != word:
            results.append(p)
    return results[:3]

def delete_char(word):
    if len(word) < 4: return []
    results = []
    for i in range(1, len(word)-1):
        results.append(word[:i] + word[i+1:])
    return results[:3]

def insert_char(word):
    if len(word) < 2: return []
    chars = 'abcdefghijklmnopqrstuvwxyz'
    results = []
    for i in range(1, len(word)):
        for c in random.sample(chars, 2):
            results.append(word[:i] + c + word[i:])
    return results[:3]

def substitute_char(word):
    if len(word) < 2: return []
    chars = 'abcdefghijklmnopqrstuvwxyz'
    results = []
    for i in range(1, len(word)-1):
        for c in random.sample(chars, 2):
            if c != word[i]:
                results.append(word[:i] + c + word[i+1:])
    return results[:3]

def generate_advcodemix_candidates(token):
    """Generate character-level candidates — no language restriction."""
    if len(token) < 3:
        return []
    candidates = []
    token_lower = token.lower()
    for p in swap_chars(token_lower):
        candidates.append({'perturbed_text': p, 'rule': 'swap'})
    for p in delete_char(token_lower):
        candidates.append({'perturbed_text': p, 'rule': 'delete'})
    for p in insert_char(token_lower):
        candidates.append({'perturbed_text': p, 'rule': 'insert'})
    for p in substitute_char(token_lower):
        candidates.append({'perturbed_text': p, 'rule': 'substitute'})
    return candidates

# ─── Model registry ───
EXPERIMENTS = [
    # Hinglish
    ('sentiment', 'xlmr',  f'{MODEL_DIR}/xlmr_hinglish_sentiment',
     'hinglish_test_final.csv', {'positive':0,'negative':1,'neutral':2}),
    ('sentiment', 'muril', f'{MODEL_DIR}/muril_hinglish_sentiment',
     'hinglish_test_final.csv', {'positive':0,'negative':1,'neutral':2}),
    ('sentiment', 'mbert', f'{MODEL_DIR}/mbert_hinglish_sentiment',
     'hinglish_test_final.csv', {'positive':0,'negative':1,'neutral':2}),
    ('hate',      'xlmr',  f'{MODEL_DIR}/xlmr_hinglish_hate',
     'hate_test.csv', {'hate':0,'not_hate':1}),
    ('hate',      'muril', f'{MODEL_DIR}/muril_hinglish_hate',
     'hate_test.csv', {'hate':0,'not_hate':1}),
    ('hate',      'mbert', f'{MODEL_DIR}/mbert_hinglish_hate',
     'hate_test.csv', {'hate':0,'not_hate':1}),
    # Tamil
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

def advcodemix_attack(text, predict_fn, get_conf_fn,
                      original_label, original_conf):
    """
    AdvCodeMix attack — faithful reimplementation of Das et al. (2022).
    Key difference from DeepWordBug: NO similarity filter applied.
    Uses leave-one-out importance scoring + character perturbations.
    """
    tokens = text.strip().split()
    if not tokens:
        return text, False, 1, []

    query_count = 0

    # Step 1: Word importance ranking (leave-one-out)
    scores = []
    for i, token in enumerate(tokens):
        if len(token) < 3:
            scores.append((i, 0.0))
            continue
        masked      = tokens.copy()
        masked[i]   = '[MASK]'
        conf        = get_conf_fn(' '.join(masked), original_label)
        query_count += 1
        scores.append((i, original_conf - conf))
    scores.sort(key=lambda x: x[1], reverse=True)

    current_tokens = tokens.copy()
    attack_success = False
    perturbations  = []
    max_changes    = max(1, int(len(tokens) * 0.40))
    tokens_changed = 0

    # Step 2: Greedy substitution — NO similarity filter
    for pos, importance in scores:
        if tokens_changed >= max_changes:
            break

        original_token = current_tokens[pos]
        candidates     = generate_advcodemix_candidates(original_token)
        if not candidates:
            continue

        # Build candidate sentences
        candidate_sentences = []
        for cand in candidates:
            ct      = current_tokens.copy()
            ct[pos] = cand['perturbed_text']
            candidate_sentences.append(' '.join(ct))

        # Batch LaBSE similarity — consistent with other baselines
        current_text = ' '.join(current_tokens)
        sims = labse_sim_batch(text, candidate_sentences)

        # Filter by similarity threshold
        valid_candidates = [
            (cand, candidate_sentences[i], float(sims[i]))
            for i, cand in enumerate(candidates)
            if sims[i] >= SIM_THRESH
        ]

        if not valid_candidates:
            continue

        best_sub  = None
        best_drop = -float('inf')

        for cand, cand_text, sim in valid_candidates:
            # WITH LaBSE filter — controlled comparison
            new_conf    = get_conf_fn(cand_text, original_label)
            query_count += 1
            drop        = original_conf - new_conf

            if drop > best_drop:
                best_drop = drop
                best_sub  = {
                    'position':        pos,
                    'original_token':  original_token,
                    'perturbed_token': cand['perturbed_text'],
                    'rule':            cand['rule'],
                    'conf_drop':       float(drop),
                    'similarity':      sim
                }

        if best_sub:
            current_tokens[pos] = best_sub['perturbed_token']
            tokens_changed      += 1
            perturbations.append(best_sub)

            result      = predict_fn(' '.join(current_tokens))
            query_count += 1
            if result['label'] != original_label:
                attack_success = True
                break

    return ' '.join(current_tokens), attack_success, query_count, perturbations

# ─── Main loop ───
all_summaries = []

for (task, model_name, model_path, test_file, label2id) in EXPERIMENTS:
    lang_tag = 'hi' if ('hinglish' in test_file or 'hate' in test_file) else 'ta'
    exp_id = f"advcodemix_{model_name}_{task}_{lang_tag}"
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

    print(f"Filtering correctly predicted from {len(df)}...")
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

    stats = {
        'total':0,'successful':0,'failed':0,
        'total_queries':0,'total_time':0.0
    }
    corpus_path = f"{OUTPUT_DIR}/{exp_id}_corpus.jsonl"
    if os.path.exists(corpus_path):
        os.remove(corpus_path)

    print(f"Running AdvCodeMix on {len(selected)} examples...")

    for idx, row in enumerate(selected):
        try:
            text       = str(row['text'])
            start      = time.time()
            result     = predict_fn(text)
            orig_label = result['label']
            orig_conf  = result['confidence']

            adv_text, success, query_count, perturbations = advcodemix_attack(
                text, predict_fn, get_conf_fn,
                orig_label, orig_conf
            )
            elapsed      = time.time() - start
            final_result = predict_fn(adv_text)

            record = {
                'id':                   f"{exp_id}_{idx:05d}",
                'original_text':        text,
                'adversarial_text':     adv_text,
                'original_label':       orig_label,
                'adversarial_label':    final_result['label'],
                'attack_successful':    success,
                'n_perturbations':      len(perturbations),
                'perturbations':        perturbations,
                'model_queries':        query_count,
                'generation_time_secs': round(elapsed, 3),
                'task':                 task,
                'victim_model':         model_name,
                'language':             'hinglish' if 'hinglish' in test_file or 'hate' in test_file else 'tamil'
            }
            with jsonlines.open(corpus_path, mode='a') as w:
                w.write(record)

            stats['total']         += 1
            stats['total_queries'] += query_count
            stats['total_time']    += elapsed
            if success:
                stats['successful'] += 1
            else:
                stats['failed']     += 1

            if stats['total'] % 100 == 0:
                asr  = stats['successful'] / stats['total']
                avgq = stats['total_queries'] / stats['total']
                avgt = stats['total_time'] / stats['total']
                print(f"  [{stats['total']}] ASR={asr:.1%} "
                      f"AvgQ={avgq:.1f} AvgT={avgt:.2f}s", flush=True)

        except Exception as e:
            print(f"  [SKIP] idx={idx} error: {e}", flush=True)
            continue

    tot = stats['total']
    summary = {
        'experiment_id': exp_id,
        'task':          task,
        'victim_model':  model_name,
        'completed_at':  datetime.now().isoformat(),
        'results': {
            'total':       tot,
            'successful':  stats['successful'],
            'failed':      stats['failed'],
            'asr':         round(stats['successful']/max(tot,1), 4),
            'avg_queries': round(stats['total_queries']/max(tot,1), 2),
            'avg_time':    round(stats['total_time']/max(tot,1), 3),
        }
    }
    summary_path = f"{OUTPUT_DIR}/{exp_id}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    r = summary['results']
    print(f"\n✅ {exp_id} DONE")
    print(f"   ASR     : {r['asr']:.4f} ({r['successful']}/{r['total']})")
    print(f"   AvgQ    : {r['avg_queries']:.1f}")
    print(f"   AvgTime : {r['avg_time']:.3f}s")
    all_summaries.append(summary)

    del model
    torch.cuda.empty_cache()

# ─── Final table ───
print(f"\n{'='*60}")
print("FINAL RESULTS — AdvCodeMix Baseline")
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
