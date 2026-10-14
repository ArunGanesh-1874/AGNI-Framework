# AGNI: Adversarial Generation for Native Indian Language Texts

A linguistically-motivated black-box adversarial attack framework for code-mixed NLP.

## Overview

AGNI is an adversarial attack framework specifically designed for code-mixed Indian language text (Hinglish and Tamil-English). It implements 7 linguistically motivated perturbation types (P1-P7) and a combined perturbation module (PC).

## Perturbation Types

| Type | Description | Example |
|------|-------------|---------|
| P1 | Phonetic Variation | bhai to bhay |
| P2 | Character-level | bahut to baut |
| P3 | Cross-script | tum to devanagari |
| P4 | Transliteration Variant | pyaar to pyar |
| P5 | Morphological Variation | achha to acche |
| P6 | Code-switching | good to achha |
| P7 | Hindi Synonym | achha to badhiya |
| PC | Combined Perturbation | good to achha to badhiya |

## Attack Variants

- AGNI-Full: All perturbation types, budget=0.60
- AGNI-Strict: P1-P7 only, budget=0.40, higher linguistic validity

## Results Summary

AGNI-Full achieves highest AQI on both Hinglish (0.4461) and Tamil-English (0.2243).
## Results

### Hinglish (Hindi-English)

| Method | Sent ASR | Hate ASR | AvgQ | AvgSim | AQI |
|--------|----------|----------|------|--------|-----|
| **AGNI-Full** | **88.6%** | **63.6%** | **39.3** | **0.9291** | **0.4297** |
| AGNI-Strict | 25.2% | 33.1% | 16.9 | 0.9754 | 0.2414 |
| DeepWordBug | 90.9% | 61.3% | 50.0 | 0.9336 | 0.3530 |
| TextFooler | 91.5% | 73.5% | 85.6 | 0.7100 | 0.0809 |
| Random | 33.1% | 17.2% | 15.3 | 0.8975 | 0.1907 |
| AdvCodeMix | 92.6% | 76.1% | 56.4 | 0.9070 | 0.3041 |

### Tamil-English

| Method | Sent ASR | Off ASR | AvgQ | AvgSim | AQI |
|--------|----------|---------|------|--------|-----|
| **AGNI-Full** | **70.6%** | **62.7%** | **27.1** | **0.9291** | **0.4501** |
| AGNI-Strict | 22.0% | 14.0% | 10.9 | 0.9541 | 0.1526 |
| DeepWordBug | 62.8% | 51.4% | 30.5 | 0.9336 | 0.3697 |
| TextFooler | 48.4% | 40.5% | 38.4 | 0.7100 | 0.1935 |
| Random | 18.7% | 10.7% | 6.5 | 0.8975 | 0.1235 |
| AdvCodeMix | 65.5% | 56.1% | 36.3 | 0.9072 | 0.3497 |

### AQI Rankings

| Rank | Method | Hinglish AQI | Tamil AQI | Overall |
|------|--------|--------------|-----------|---------|
| 🥇 1 | **AGNI-Full** | **0.4297** | **0.4501** | **0.4399** |
| 🥈 2 | DeepWordBug | 0.3530 | 0.3697 | 0.3614 |
| 🥉 3 | AdvCodeMix | 0.3041 | 0.3497 | 0.3269 |
| 4 | AGNI-Strict | 0.2414 | 0.1526 | 0.1970 |
| 5 | Random | 0.1907 | 0.1235 | 0.1571 |
| 6 | TextFooler | 0.0809 | 0.1935 | 0.1372 |

## Citation

Paper under review.

## License

Private until paper acceptance.
