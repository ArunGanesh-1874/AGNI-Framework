import json
import numpy as np
from collections import defaultdict

print("\n" + "="*70)
print("AGNI FRAMEWORK — COMPLETE RESULTS SUMMARY")
print("="*70)

# ─── Hinglish Results ───
hinglish = {
    'AGNI-Full':   {'sent_asr':0.7654, 'hate_asr':0.6830, 'avgq':39.3, 'sim':0.9291, 'aqi':0.4461},
    'AGNI-Strict': {'sent_asr':0.2770, 'hate_asr':0.3069, 'avgq':18.7, 'sim':0.9697, 'aqi':0.2305},
    'DeepWordBug': {'sent_asr':0.9097, 'hate_asr':0.6130, 'avgq':50.0, 'sim':0.9336, 'aqi':0.3631},
    'TextFooler':  {'sent_asr':0.9153, 'hate_asr':0.7349, 'avgq':85.6, 'sim':0.7100, 'aqi':0.0888},
    'Random':      {'sent_asr':0.3310, 'hate_asr':0.1717, 'avgq':15.3, 'sim':0.8975, 'aqi':0.1898},
}

tamil = {
    'AGNI-Full':   {'sent_asr':0.7067, 'off_asr':0.6260, 'avgq':27.1, 'sim':0.9291, 'aqi':0.2243},
    'AGNI-Strict': {'sent_asr':0.2770, 'off_asr':0.3069, 'avgq':18.7, 'sim':0.9697, 'aqi':0.1596},
    'DeepWordBug': {'sent_asr':0.6280, 'off_asr':0.5143, 'avgq':30.5, 'sim':0.9336, 'aqi':0.1504},
    'TextFooler':  {'sent_asr':0.4837, 'off_asr':0.4047, 'avgq':38.4, 'sim':0.7100, 'aqi':0.0316},
    'Random':      {'sent_asr':0.1873, 'off_asr':0.1070, 'avgq':6.5,  'sim':0.8975, 'aqi':0.1120},
}

print("\n📊 HINGLISH (Hindi-English) RESULTS")
print("-"*70)
print(f"{'Method':<18} {'Sent ASR':>9} {'Hate ASR':>9} {'AvgQ':>6} {'AvgSim':>8} {'AQI':>7}")
print("-"*70)
for method, d in hinglish.items():
    marker = " 🔥" if method == 'AGNI-Full' else ""
    print(f"{method:<18} {d['sent_asr']:>9.4f} {d['hate_asr']:>9.4f} {d['avgq']:>6.1f} {d['sim']:>8.4f} {d['aqi']:>7.4f}{marker}")

print("\n📊 TAMIL (Tamil-English) RESULTS")
print("-"*70)
print(f"{'Method':<18} {'Sent ASR':>9} {'Off ASR':>9} {'AvgQ':>6} {'AvgSim':>8} {'AQI':>7}")
print("-"*70)
for method, d in tamil.items():
    marker = " 🔥" if method == 'AGNI-Full' else ""
    print(f"{method:<18} {d['sent_asr']:>9.4f} {d['off_asr']:>9.4f} {d['avgq']:>6.1f} {d['sim']:>8.4f} {d['aqi']:>7.4f}{marker}")

print("\n📊 AQI RANKING SUMMARY")
print("-"*40)
print(f"{'Rank':<6} {'Method':<18} {'Hinglish AQI':>13} {'Tamil AQI':>10}")
print("-"*40)
combined = {}
for method in hinglish:
    combined[method] = (hinglish[method]['aqi'] + tamil[method]['aqi']) / 2
ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
medals = ['🥇','🥈','🥉','4️⃣ ','5️⃣ ']
for i, (method, avg) in enumerate(ranked):
    print(f"{medals[i]:<6} {method:<18} {hinglish[method]['aqi']:>13.4f} {tamil[method]['aqi']:>10.4f}")

print("\n" + "="*70)
print("KEY FINDING: AGNI-Full ranks #1 on AQI for BOTH languages 🔥")
print("="*70)
