import csv
import json
import numpy as np
from collections import defaultdict

HUMAN_EVAL_DIR = '/home/arun_24phd0774/comixattack/human_eval'

# Load keyed JSONs (has pair_id + method for each pair in order)
with open(f'{HUMAN_EVAL_DIR}/annotation_hinglish_keyed_clean.json') as f:
    hinglish_keyed = json.load(f)
with open(f'{HUMAN_EVAL_DIR}/annotation_tamil_keyed_clean.json') as f:
    tamil_keyed = json.load(f)

def load_responses(csv_path, n_pairs):
    """Load ratings from CSV by position. Returns list of dicts per annotator."""
    annotators = []
    with open(csv_path, encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        # Find start of pair ratings (after background questions)
        # Background: Timestamp, consent, lang background, familiarity,
        # academic, practice1(3), practice2(3) = 11 columns before pair ratings
        # Each pair has 3 columns: Naturalness, Meaning, CodeMix
        start_col = 11  # 0-indexed: columns 0-10 are background
        for row in reader:
            if not row: continue
            ratings = []
            for i in range(n_pairs):
                col = start_col + (i * 3)
                if col + 2 < len(row):
                    try:
                        nat  = int(row[col])
                        mean = int(row[col+1])
                        cmq  = int(row[col+2])
                        ratings.append({'naturalness': nat,
                                       'meaning': mean,
                                       'codemix': cmq})
                    except (ValueError, IndexError):
                        ratings.append(None)
                else:
                    ratings.append(None)
            annotators.append(ratings)
    return annotators

def krippendorff_alpha(data, metric='ordinal'):
    """
    Compute Krippendorff's Alpha for ordinal data.
    data: list of lists — data[annotator][item] = value or None
    """
    # Flatten to coincidence matrix approach
    n_annotators = len(data)
    n_items = len(data[0])
    
    # Collect all values
    all_values = []
    for ann in data:
        for v in ann:
            if v is not None:
                all_values.append(v)
    
    if not all_values:
        return None
    
    categories = sorted(set(all_values))
    
    # Build coincidence matrix
    coincidences = defaultdict(float)
    n_o = 0  # total observed coincidences
    
    for item in range(n_items):
        item_ratings = [data[ann][item] for ann in range(n_annotators)
                       if data[ann][item] is not None]
        m = len(item_ratings)
        if m < 2:
            continue
        for i in range(len(item_ratings)):
            for j in range(len(item_ratings)):
                if i != j:
                    coincidences[(item_ratings[i], item_ratings[j])] += 1.0 / (m - 1)
                    n_o += 1.0 / (m - 1)
    
    if n_o == 0:
        return None
    
    # Expected disagreement
    n_k = defaultdict(float)
    for (k, l), v in coincidences.items():
        n_k[k] += v
    
    total = sum(n_k.values())
    
    # Observed disagreement (ordinal metric)
    D_o = 0
    for (k, l), v in coincidences.items():
        D_o += v * (k - l) ** 2
    D_o /= n_o
    
    # Expected disagreement
    D_e = 0
    cats = list(n_k.keys())
    for k in cats:
        for l in cats:
            D_e += n_k[k] * n_k[l] * (k - l) ** 2
    D_e /= (total * (total - 1))
    
    if D_e == 0:
        return 1.0
    
    alpha = 1 - D_o / D_e
    return round(alpha, 4)

def analyze_forms(form_files, keyed_data, lang, pair_offsets, n_pairs_per_form):
    """Analyze all forms for a language."""
    all_method_scores = defaultdict(lambda: defaultdict(list))
    all_nat_ratings = []   # for Krippendorff
    all_mean_ratings = []
    all_cmq_ratings = []
    
    for form_idx, (csv_file, offset, n_pairs) in enumerate(
            zip(form_files, pair_offsets, n_pairs_per_form)):
        
        annotators = load_responses(csv_file, n_pairs)
        n_ann = len(annotators)
        print(f"  {csv_file.split('/')[-1]}: {n_ann} annotators, {n_pairs} pairs")
        
        # For Krippendorff — one list per annotator
        nat_data  = [[] for _ in range(n_ann)]
        mean_data = [[] for _ in range(n_ann)]
        cmq_data  = [[] for _ in range(n_ann)]
        
        for pair_local_idx in range(n_pairs):
            pair_global_idx = offset + pair_local_idx
            if pair_global_idx >= len(keyed_data):
                break
            
            method = keyed_data[pair_global_idx]['method']
            
            for ann_idx, ann_ratings in enumerate(annotators):
                if ann_ratings[pair_local_idx] is not None:
                    r = ann_ratings[pair_local_idx]
                    all_method_scores[method]['naturalness'].append(r['naturalness'])
                    all_method_scores[method]['meaning'].append(r['meaning'])
                    all_method_scores[method]['codemix'].append(r['codemix'])
                    nat_data[ann_idx].append(r['naturalness'])
                    mean_data[ann_idx].append(r['meaning'])
                    cmq_data[ann_idx].append(r['codemix'])
                else:
                    nat_data[ann_idx].append(None)
                    mean_data[ann_idx].append(None)
                    cmq_data[ann_idx].append(None)
        
        all_nat_ratings.extend(nat_data)
        all_mean_ratings.extend(mean_data)
        all_cmq_ratings.extend(cmq_data)
    
    # Compute Krippendorff's Alpha
    alpha_nat  = krippendorff_alpha(all_nat_ratings)
    alpha_mean = krippendorff_alpha(all_mean_ratings)
    alpha_cmq  = krippendorff_alpha(all_cmq_ratings)
    
    print(f"\n  Krippendorff Alpha ({lang}):")
    print(f"    Naturalness:        {alpha_nat}")
    print(f"    Meaning Preserv.:   {alpha_mean}")
    print(f"    Code-Mix Quality:   {alpha_cmq}")
    
    # Mean scores per method
    print(f"\n  Mean scores per method ({lang}):")
    print(f"  {'Method':<14} {'Naturalness':>12} {'Meaning':>10} {'CodeMix':>10} {'N pairs':>8}")
    print(f"  {'-'*56}")
    
    results = {}
    for method in ['AGNI-Full','AGNI-Strict','DeepWordBug',
                   'TextFooler','Random','AdvCodeMix']:
        if method in all_method_scores:
            nat  = np.mean(all_method_scores[method]['naturalness'])
            mean = np.mean(all_method_scores[method]['meaning'])
            cmq  = np.mean(all_method_scores[method]['codemix'])
            n    = len(all_method_scores[method]['naturalness'])
            print(f"  {method:<14} {nat:>12.3f} {mean:>10.3f} {cmq:>10.3f} {n:>8}")
            results[method] = {'naturalness': round(nat,3),
                               'meaning': round(mean,3),
                               'codemix': round(cmq,3),
                               'n': n}
    
    return results, {
        'alpha_naturalness': alpha_nat,
        'alpha_meaning': alpha_mean,
        'alpha_codemix': alpha_cmq
    }

HE = HUMAN_EVAL_DIR

print("="*60)
print("HUMAN EVALUATION ANALYSIS")
print("="*60)

# Hinglish: H1=pairs 0-84, H2=pairs 85-169, H3=pairs 170-254
print("\nHINGLISH:")
h_files   = [f'{HE}/responses_h1.csv',
             f'{HE}/responses_h2.csv',
             f'{HE}/responses_h3.csv']
h_offsets = [0, 85, 170]
h_npairs  = [85, 85, 85]

h_results, h_alpha = analyze_forms(
    h_files, hinglish_keyed, 'Hinglish', h_offsets, h_npairs)

# Tamil: T1=pairs 0-99, T2=pairs 100-199, T3=pairs 200-299
print("\nTAMIL:")
t_files   = [f'{HE}/responses_t1.csv',
             f'{HE}/responses_t2.csv',
             f'{HE}/responses_t3.csv']
t_offsets = [0, 100, 200]
t_npairs  = [100, 100, 100]

t_results, t_alpha = analyze_forms(
    t_files, tamil_keyed, 'Tamil', t_offsets, t_npairs)

# Save results
output = {
    'hinglish': {'results': h_results, 'alpha': h_alpha},
    'tamil':    {'results': t_results, 'alpha': t_alpha}
}
with open(f'{HE}/human_eval_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nSaved: {HE}/human_eval_results.json")
print("="*60)
