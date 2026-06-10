import sys
sys.path.append('/workspace/comixattack/code')

import torch
import pandas as pd
import json
import os
import time
import jsonlines
from datetime import datetime

print("TextFooler Baseline Experiment")
print("=" * 55)

test_df = pd.read_csv(
    '/workspace/comixattack/data/processed/hinglish_test_final.csv'
).head(500)

print(f"Test set: {len(test_df)} sentences")

import textattack
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.attack_recipes import TextFoolerJin2019
from textattack.datasets import Dataset
from textattack import Attacker, AttackArgs
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = '/workspace/comixattack/models/xlmr_hinglish_sentiment'
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Device: {DEVICE}")
print("Loading victim model...")

tokenizer    = AutoTokenizer.from_pretrained(MODEL_PATH)
victim_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
).to(DEVICE)

model_wrapper = HuggingFaceModelWrapper(victim_model, tokenizer)
print("Model wrapped for TextAttack")

LABEL_MAP_REV = {'positive': 0, 'negative': 1, 'neutral': 2}

dataset_tuples = []
for _, row in test_df.iterrows():
    text  = str(row['text'])
    label = str(row['label'])
    if label in LABEL_MAP_REV:
        dataset_tuples.append((text, LABEL_MAP_REV[label]))

ta_dataset = Dataset(dataset_tuples)
print(f"Dataset prepared: {len(dataset_tuples)} examples")

print("Building TextFooler attack...")
attack = TextFoolerJin2019.build(model_wrapper)
print("TextFooler ready")

output_dir = '/workspace/comixattack/data/adversarial/baseline_textfooler'
os.makedirs(output_dir, exist_ok=True)

attack_args = AttackArgs(
    num_examples=200,
    disable_stdout=True,
    silent=True,
    log_to_csv=os.path.join(output_dir, 'textfooler_results.csv'),
    csv_coloring_style='plain'
)

print("\nRunning TextFooler on 200 examples...")
print("This will take 15-25 minutes...")
print("=" * 55)

attacker       = Attacker(attack, ta_dataset, attack_args)
start_time     = time.time()
attack_results = attacker.attack_dataset()
total_time     = time.time() - start_time

print("\n" + "=" * 55)
print("TEXTFOOLER RESULTS")
print("=" * 55)

successful    = 0
failed        = 0
skipped       = 0
total_queries = 0
results_list  = []

for result in attack_results:
    result_type = type(result).__name__

    if 'Successful' in result_type:
        successful += 1
        record = {
            'original_text':     result.original_result.attacked_text.text,
            'adversarial_text':  result.perturbed_result.attacked_text.text,
            'original_label':    str(result.original_result.output),
            'adversarial_label': str(result.perturbed_result.output),
            'attack_successful': True,
            'num_queries':       result.num_queries,
            'method':            'TextFooler'
        }
        results_list.append(record)
        total_queries += result.num_queries

    elif 'Failed' in result_type:
        failed += 1
        total_queries += result.num_queries
        record = {
            'original_text':     result.original_result.attacked_text.text,
            'adversarial_text':  result.original_result.attacked_text.text,
            'attack_successful': False,
            'num_queries':       result.num_queries,
            'method':            'TextFooler'
        }
        results_list.append(record)

    elif 'Skipped' in result_type:
        skipped += 1

total_attacked = successful + failed
asr        = successful / total_attacked if total_attacked > 0 else 0
avg_queries = total_queries / total_attacked if total_attacked > 0 else 0
avg_time    = total_time / max(total_attacked, 1)

print("Total attempted:  " + str(200))
print("Skipped:          " + str(skipped))
print("Attacked:         " + str(total_attacked))
print("Successful:       " + str(successful))
print("Failed:           " + str(failed))
print("ASR:              " + str(round(asr, 4)) + " (" + str(round(asr*100, 1)) + "%)")
print("Avg queries:      " + str(round(avg_queries, 1)))
print("Total time:       " + str(round(total_time/60, 1)) + " minutes")
print("Avg time/example: " + str(round(avg_time, 2)) + "s")

corpus_path = os.path.join(output_dir, 'textfooler_corpus.jsonl')
with jsonlines.open(corpus_path, mode='w') as writer:
    for r in results_list:
        writer.write(r)

summary = {
    'method':          'TextFooler',
    'dataset':         'SemEval-2020 Hinglish Sentiment',
    'victim_model':    'xlmr-hinglish-finetuned',
    'n_attempted':     200,
    'n_skipped':       skipped,
    'n_attacked':      total_attacked,
    'n_successful':    successful,
    'asr':             round(asr, 4),
    'avg_queries':     round(avg_queries, 1),
    'total_time_mins': round(total_time/60, 2),
    'avg_time_secs':   round(avg_time, 2),
    'completed_at':    datetime.now().isoformat()
}

summary_path = os.path.join(output_dir, 'textfooler_summary.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print("")
print("=" * 55)
print("COMPARISON TABLE")
print("=" * 55)
print("Method               ASR      AvgQueries   Time/Ex")
print("-" * 55)
print("CoMixAttack-v2       12.0%         4.9       1.50s")
print("TextFooler           " +
      str(round(asr*100,1)) + "%        " +
      str(round(avg_queries,1)) + "         " +
      str(round(avg_time,2)) + "s")
print("=" * 55)
print("Baseline comparison complete")
print("Results saved to: " + output_dir)
