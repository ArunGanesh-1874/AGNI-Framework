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
| **AGNI-Full** | **88.6%** | **69.8%** | **39.3** | **0.9291** | **0.4461** |
| AGNI-Strict | 27.7% | 30.7% | 18.7 | 0.9697 | 0.2305 |
| DeepWordBug | 90.9% | 61.3% | 50.0 | 0.9336 | 0.3631 |
| TextFooler | 91.5% | 73.5% | 85.6 | 0.7100 | 0.0888 |
| Random | 33.1% | 17.2% | 15.3 | 0.8975 | 0.1898 |

### Tamil-English

| Method | Sent ASR | Off ASR | AvgQ | AvgSim | AQI |
|--------|----------|---------|------|--------|-----|
| **AGNI-Full** | **70.7%** | **62.6%** | **27.1** | **0.9291** | **0.2243** |
| AGNI-Strict | 27.7% | 30.7% | 18.7 | 0.9697 | 0.1596 |
| DeepWordBug | 62.8% | 51.4% | 30.5 | 0.9336 | 0.1504 |
| TextFooler | 48.4% | 40.5% | 38.4 | 0.7100 | 0.0316 |
| Random | 18.7% | 10.7% | 6.5 | 0.8975 | 0.1120 |

### AQI Rankings

| Rank | Method | Hinglish AQI | Tamil AQI |
|------|--------|--------------|-----------|
| 🥇 1 | **AGNI-Full** | **0.4461** | **0.2243** |
| 🥈 2 | DeepWordBug | 0.3631 | 0.1504 |
| 🥉 3 | AGNI-Strict | 0.2305 | 0.1596 |
| 4 | Random | 0.1898 | 0.1120 |
| 5 | TextFooler | 0.0888 | 0.0316 |

## Citation

Paper under review.

## License

Private until paper acceptance.
