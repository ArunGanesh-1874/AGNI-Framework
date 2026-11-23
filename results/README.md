# results/ — which folder feeds which paper number

Per-experiment `*_summary.json` files are included. The large per-example `*_corpus.jsonl` files are NOT in the repository
(see `.gitignore`); `code/compute_labse_similarity_all.py` needs them to regenerate similarities, so the output of that
script is stored here as `labse_similarity_per_experiment.json`.

## Final folders (used in the paper)
| Method | Hinglish | Tanglish |
|---|---|---|
| AGNI-Full | comixattack_v4c_full | comixattack_tamil_v2_full |
| AGNI-Strict | comixattack_v3_full | comixattack_tamil_strict_v2_full |
| DeepWordBug | deepwordbug_baseline | deepwordbug_tamil_baseline |
| TextFooler | textfooler_baseline | textfooler_tamil_baseline |
| Random | random_baseline | random_tamil_baseline |
| AdvCodeMix | advcodemix_labse_baseline (`*_hi`) | advcodemix_labse_baseline (`*_ta`) |

## Superseded folders (kept for history, NOT used in the paper)
comixattack_full, comixattack_v2_full, comixattack_tamil_full, comixattack_tamil_strict_full, advcodemix_baseline

## Result files
- `labse_similarity_per_experiment.json` — LaBSE similarity for all 72 experiments (mean over all attempted examples).
- `aqi_final_v2.json` — AQI per experiment and per method/language, from `code/build_tables.py` (replaces `aqi_final.json`, `aqi_tamil.json`, which used pooled/borrowed similarity constants).
- `comixattack_controlled/` — AGNI-Full under beta=0.40, tau=0.80 (Table 5). **Hate runs attacked all 498 test examples; the original runs attacked only correctly classified examples (n = 339/325/337), so the hate rows are not on matched populations.** `code/run_controlled_matched.py` re-attacks exactly the original examples; its results are not part of the paper.
- `ablation/`, `mcnemar_results.json`, `bootstrap_ci.json`, `aqi_sensitivity.json`, `language_id_validation.json`, `wer_analysis.json`, `perplexity_analysis.json`, `human_eval/` — supporting analyses.

## Known caveats
- AGNI-Strict and AdvCodeMix on MuRIL hate speech used 266 examples; all other MuRIL hate runs used 325.
- AGNI-Full uses tau=0.75 (beta=0.60); baselines use tau=0.80. A share of Tanglish AGNI-Full examples (74–84%) has similarity >= 0.80.
- `bootstrap_ci.json` and `aqi_sensitivity.json` were computed before the similarity recomputation and have not been regenerated.
