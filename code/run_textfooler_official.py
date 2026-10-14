# Official TextFooler Comparison
# Using TextAttack library — Jin et al. 2020
# Same 200 examples, same seed as CoMixAttack final

import sys
sys.path.append('/workspace/comixattack/code')

import torch
import numpy as np
import random
import pandas as pd
import json
import os
import time
import jsonlines
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.attack_recipes import TextFoolerJin2019
from textattack.datasets import Dataset
from textattack import Attacker, AttackArgs

# Fixed seed — same as CoMixAttack
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("Official TextFooler Comparison")
print(f"Random seed: {SEED}")
print("=" * 55)

MODEL_PATH    = '/workspace/comixattack/models/xlmr_hinglish_sentiment'
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'
LABEL_MAP_REV = {'positive': 0, 'negative': 1, 'neutral': 2}

print(f"Device: {DEVICE}")
print("Loading victim model...")

tokenizer    = AutoTokenizer.from_pretrained(MODEL_PATH)
victim_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
).to(DEVICE)

model_wrapper = HuggingFaceModelWrapper(victim_model, tokenizer)
print("✓ Model wrapped for TextAttack")

# Load same dataset as CoMixAttack final experiment
test_df = pd.read_csv(
    '/workspace/comixattack/data/processed/hinglish_test_final.csv'
)
# Same sorting as final experiment
test_df = test_df.sort_values('tweet_id').reset_index(drop=True)
test_df = test_df.head(200)

print(f"Dataset: {len(test_df)} examples (same slice as CoMixAttack)")

# Prepare for TextAttack
dataset_tuples = []
for _, row in test_df.iterrows():
    text  = str(row['text'])
    label = str(row['label'])
    if label in LABEL_MAP_REV:
        dataset_tuples.append((text, LABEL_MAP_REV[label]))

ta_dataset = Dataset(dataset_tuples)
print(f"✓ Dataset prepared: {len(dataset_tuples)} examples")

# Build TextFooler
print("Building TextFooler attack...")
attack = TextFoolerJin2019.build(model_wrapper)
print("✓ TextFooler ready")

# Output directory
output_dir = '/workspace/comixattack/data/adversarial/final_results'
os.makedirs(output_dir, exist_ok=True)

attack_args = AttackArgs(
    num_examples=200,
    random_seed=SEED,
    disable_stdout=True,
    silent=True,
    log_to_csv=os.path.join(
        output_dir, 'textfooler_official_log.csv'
    ),
    csv_coloring_style='plain'
)

print("\nRunning TextFooler on 200 examples...")
print("Expected time: 30-60 minutes on CPU")
print("=" * 55)

attacker     = Attacker(attack, ta_dataset, attack_args)
start_time   = time.time()
attack_results = attacker.attack_dataset()
total_time   = time.time() - start_time

# Process results
successful    = 0
failed        = 0
skipped       = 0
total_queries = 0
results_list  = []

for result in attack_results:
    result_type = type(result).__name__

    if 'Successful' in result_type:
        successful += 1
        total_queries += result.num_queries
        results_list.append({
            'original_text':     result.original_result.attacked_text.text,
            'adversarial_text':  result.perturbed_result.attacked_text.text,
            'original_label':    str(result.original_result.output),
            'adversarial_label': str(result.perturbed_result.output),
            'attack_successful': True,
            'num_queries':       result.num_queries,
            'method':            'TextFooler'
        })

    elif 'Failed' in result_type:
        failed += 1
        total_queries += result.num_queries
        results_list.append({
            'original_text':     result.original_result.attacked_text.text,
            'adversarial_text':  result.original_result.attacked_text.text,
            'attack_successful': False,
            'num_queries':       result.num_queries,
            'method':            'TextFooler'
        })

    elif 'Skipped' in result_type:
        skipped += 1

total_attacked = successful + failed
asr      = successful / total_attacked if total_attacked > 0 else 0
avg_q    = total_queries / total_attacked if total_attacked > 0 else 0
avg_time = total_time / max(total_attacked, 1)

# Save corpus
corpus_path = os.path.join(
    output_dir, 'textfooler_official_corpus.jsonl'
)
with jsonlines.open(corpus_path, mode='w') as writer:
    for r in results_list:
        writer.write(r)

# Save summary
summary = {
    'method':          'TextFooler (Jin et al. 2020)',
    'framework':       'TextAttack',
    'random_seed':     SEED,
    'dataset':         'SemEval-2020 Hinglish Sentiment',
    'victim_model':    'xlmr-hinglish-finetuned',
    'n_attempted':     200,
    'n_skipped':       skipped,
    'n_attacked':      total_attacked,
    'n_successful':    successful,
    'asr':             round(asr, 4),
    'avg_queries':     round(avg_q, 1),
    'total_time_mins': round(total_time/60, 2),
    'avg_time_secs':   round(avg_time, 2),
    'completed_at':    datetime.now().isoformat()
}

summary_path = os.path.join(
    output_dir, 'textfooler_official_summary.json'
)
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 55)
print("TEXTFOOLER OFFICIAL RESULTS")
print("=" * 55)
print(f"Total attempted:  200")
print(f"Skipped:          {skipped}")
print(f"Attacked:         {total_attacked}")
print(f"Successful:       {successful}")
print(f"Failed:           {failed}")
print(f"ASR:              {round(asr*100,1)}%")
print(f"Avg queries:      {round(avg_q,1)}")
print(f"Total time:       {round(total_time/60,1)} minutes")
print(f"Avg time/example: {round(avg_time,2)}s")

print("\n" + "=" * 55)
print("OFFICIAL COMPARISON TABLE")
print("Seed=42, 200 examples, SemEval-2020 Hinglish")
print("=" * 55)
print(f"{'Method':<30} {'ASR':>6} {'AvgQ':>6} {'Time':>8}")
print("-" * 55)
print(f"{'TextFooler (Jin 2020)':<30} "
      f"{round(asr*100,1):>5}% "
      f"{round(avg_q,1):>6} "
      f"{round(avg_time,2):>7}s")
print(f"{'CoMixAttack-Hindi (ours)':<30} "
      f"{'14.6':>5}% "
      f"{'19.6':>6} "
      f"{'3.22':>7}s")
print(f"{'CoMixAttack-Full (ours)':<30} "
      f"{'49.3':>5}% "
      f"{'30.1':>6} "
      f"{'4.73':>7}s")
print("=" * 55)
print(f"\n✓ Results saved: {output_dir}")
