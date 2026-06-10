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

## Citation

Paper under review.

## License

Private until paper acceptance.
