"""
Rebuild Tables 2, 3, 6 and the AQI JSON from summaries + per-experiment similarity.
    python code/build_tables.py            (server)
    python code/build_tables.py --base /path/to/repo   (any machine with the repo)
Writes results/aqi_final_v2.json and prints table rows.
"""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_map import experiments

ap = argparse.ArgumentParser()
ap.add_argument('--base', default=os.environ.get('AGNI_BASE', '/workspace/comixattack'))
ap.add_argument('--sim', default=None, help='similarity json (default results/labse_similarity_per_experiment.json)')
args = ap.parse_args()
RES = f'{args.base}/results'
sim_path = args.sim or f'{RES}/labse_similarity_per_experiment.json'
sims = {(r['lang'], r['method'], r['task'], r['model']): r['avg_sim_all'] for r in json.load(open(sim_path))}

rows = []
for lang, method, task, model, folder, exp_id in experiments():
    p = f'{RES}/{folder}/{exp_id}_summary.json'
    if not os.path.exists(p):
        print('MISSING summary', p); continue
    r = json.load(open(p))['results']
    key = (lang, method, task, model)
    if key not in sims:
        print('MISSING similarity for', key); continue
    rows.append(dict(lang=lang, method=method, task=task, model=model,
                     total=r['total'], asr=r['asr'], avgq=r['avg_queries'], sim=sims[key]))

QMAX = max(r['avgq'] for r in rows)
print(f'Qmax (max avg queries over all {len(rows)} experiments) = {QMAX:.2f}   [paper says 99.3]')
for r in rows:
    r['aqi'] = r['asr'] * r['sim'] * (1 - r['avgq'] / QMAX)

def agg(sel):
    a = np.mean([r['asr'] for r in sel]); s = np.mean([r['sim'] for r in sel]); q = np.mean([r['avgq'] for r in sel])
    return dict(asr=a, sim=s, avgq=q,
                aqi_mean_of_experiments=float(np.mean([r['aqi'] for r in sel])),
                aqi_from_averaged_components=float(a * s * (1 - q / QMAX)))

summary, methods = {}, list(dict.fromkeys(r['method'] for r in rows))
for m in methods:
    for lang in ('hinglish', 'tamil'):
        sel = [r for r in rows if r['method'] == m and r['lang'] == lang]
        if sel: summary[f'{m}|{lang}'] = agg(sel)
    langs = [l for l in ('hinglish', 'tamil') if f'{m}|{l}' in summary]
    summary[f'{m}|overall'] = {
        'aqi_mean_of_experiments': float(np.mean([summary[f'{m}|{l}']['aqi_mean_of_experiments'] for l in langs])),
        # Convention used in the paper (Sec 3.8, Table 6): AQI from averaged components, overall = mean of the two languages
        'aqi_paper_convention': float(np.mean([summary[f'{m}|{l}']['aqi_from_averaged_components'] for l in langs]))}

print('\nMETHOD        Hinglish AQI (mean-of-exp / avg-components)   Tamil AQI    Overall(exp-mean)  [Hin sim | Tam sim]   <- paper uses the second number of each pair, overall = aqi_paper_convention')
for m in sorted(methods, key=lambda x: -summary[f'{x}|overall']['aqi_mean_of_experiments']):
    h, t = summary.get(f'{m}|hinglish'), summary.get(f'{m}|tamil')
    print(f'{m:12s}  {h["aqi_mean_of_experiments"]:.4f} / {h["aqi_from_averaged_components"]:.4f}          '
          f'{t["aqi_mean_of_experiments"]:.4f} / {t["aqi_from_averaged_components"]:.4f}     '
          f'{summary[m+"|overall"]["aqi_mean_of_experiments"]:.4f}   [{h["sim"]:.4f} | {t["sim"]:.4f}]')
json.dump(dict(qmax=QMAX, per_experiment=rows, summary=summary), open(f'{RES}/aqi_final_v2.json', 'w'), indent=2)
print('\nPAPER CONVENTION (Table 6): Hinglish | Tamil | Overall')
for m in sorted(methods, key=lambda x: -summary[f'{x}|overall']['aqi_paper_convention']):
    print(f'{m:12s} {summary[m+"|hinglish"]["aqi_from_averaged_components"]:.4f} | {summary[m+"|tamil"]["aqi_from_averaged_components"]:.4f} | {summary[m+"|overall"]["aqi_paper_convention"]:.4f}')
print(f'\nSaved {RES}/aqi_final_v2.json')
