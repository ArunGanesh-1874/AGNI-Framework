import time
import torch
import numpy as np
from typing import List, Dict

class SearchControllerV2:
    """
    Improved search with:
    1. Multi-word simultaneous perturbation
    2. Iterative deepening until confidence < threshold
    3. Beam search over top-K candidates
    """
    def __init__(self, predict_fn, get_conf_fn, validator,
                 beam_size=3, max_iterations=3,
                 confidence_threshold=0.40):
        self.predict              = predict_fn
        self.get_confidence       = get_conf_fn
        self.validator            = validator
        self.beam_size            = beam_size
        self.max_iterations       = max_iterations
        self.confidence_threshold = confidence_threshold
        self.query_count          = 0

    def _get_importance_scores(self, tokens, original_label,
                                original_conf):
        scores = []
        for i, token in enumerate(tokens):
            if len(token) <= 1:
                scores.append((i, 0.0))
                continue
            masked      = tokens.copy()
            masked[i]   = '[MASK]'
            conf        = self.get_confidence(
                ' '.join(masked), original_label
            )
            self.query_count += 1
            scores.append((i, original_conf - conf))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def _get_best_perturbation(self, token_dict, engine,
                                current_tokens, original_text,
                                original_label, original_conf):
        """Find best single perturbation for one token."""
        candidates = engine.generate_candidates(token_dict)
        if not candidates:
            return None

        current_text = ' '.join(current_tokens)
        valid_cands  = self.validator.validate_batch(
            current_text, candidates
        )
        if not valid_cands:
            return None

        best_cand = None
        best_drop = -float('inf')

        for cand in valid_cands:
            perturbed_sent = cand['perturbed_sentence']
            new_conf       = self.get_confidence(
                perturbed_sent, original_label
            )
            self.query_count += 1
            drop = original_conf - new_conf

            if drop > best_drop:
                best_drop = drop
                best_cand = cand
                best_cand['conf_after'] = new_conf
                best_cand['conf_drop']  = drop

        return best_cand

    def search(self, processed_input, engine, budget=0.60):
        start            = time.time()
        self.query_count = 0

        original_text  = processed_input['original_text']
        original_label = processed_input['original_prediction']
        original_conf  = processed_input['original_confidence']
        tokens         = processed_input['tokens']

        current_tokens = [t['text'] for t in tokens]
        applied        = []
        attack_success = False
        max_changes    = max(1, int(len(tokens) * budget))

        # ── Phase 1: Importance scoring ──
        scores = self._get_importance_scores(
            current_tokens, original_label, original_conf
        )

        # ── Phase 2: Multi-word simultaneous attack ──
        # Select top-K tokens and perturb ALL simultaneously
        top_k = min(self.beam_size, max_changes,
                    len([s for s in scores if s[1] > 0]))

        if top_k > 0:
            top_positions = [pos for pos, _ in scores[:top_k]]
            simultaneous_tokens = current_tokens.copy()
            simultaneous_applied = []

            for pos in top_positions:
                token_dict = tokens[pos]
                best_cand  = self._get_best_perturbation(
                    token_dict, engine,
                    simultaneous_tokens,
                    original_text, original_label, original_conf
                )
                if best_cand:
                    simultaneous_tokens[pos] = best_cand['perturbed_text']
                    simultaneous_applied.append({
                        'position':          pos,
                        'original_token':    token_dict['text'],
                        'perturbed_token':   best_cand['perturbed_text'],
                        'perturbation_type': best_cand['perturbation_type'],
                        'rule_applied':      best_cand['rule_applied'],
                        'confidence_drop':   best_cand['conf_drop'],
                        'labse_similarity':  best_cand.get('labse_similarity')
                    })

            if simultaneous_applied:
                sim_text   = ' '.join(simultaneous_tokens)
                sim_result = self.predict(sim_text)
                self.query_count += 1

                if sim_result['label'] != original_label:
                    # Simultaneous attack succeeded
                    attack_success = True
                    current_tokens = simultaneous_tokens
                    applied        = simultaneous_applied
                else:
                    # Use simultaneous as starting point
                    # then continue iterating
                    current_tokens = simultaneous_tokens
                    applied        = simultaneous_applied

        # ── Phase 3: Iterative deepening ──
        # Keep attacking until confidence < threshold
        # or max iterations reached
        iteration = 0
        tokens_changed = len(applied)

        while not attack_success and \
              iteration < self.max_iterations and \
              tokens_changed < max_changes:

            iteration += 1
            current_text = ' '.join(current_tokens)
            current_result = self.predict(current_text)
            self.query_count += 1
            current_conf = current_result['confidence']

            # If already close to boundary, push harder
            if current_conf < self.confidence_threshold:
                attack_success = current_result['label'] != original_label
                break

            # Re-score remaining tokens
            remaining_scores = []
            for i, token in enumerate(tokens):
                if i in [a['position'] for a in applied]:
                    continue  # already perturbed
                if len(token['text']) <= 1:
                    continue
                masked    = current_tokens.copy()
                masked[i] = '[MASK]'
                conf      = self.get_confidence(
                    ' '.join(masked), original_label
                )
                self.query_count += 1
                remaining_scores.append((i, current_conf - conf))

            if not remaining_scores:
                break

            remaining_scores.sort(key=lambda x: x[1], reverse=True)

            # Try top remaining token
            improved = False
            for pos, importance in remaining_scores[:3]:
                token_dict = tokens[pos]
                best_cand  = self._get_best_perturbation(
                    token_dict, engine,
                    current_tokens,
                    original_text, original_label, original_conf
                )
                if best_cand and best_cand['conf_drop'] > 0.02:
                    current_tokens[pos] = best_cand['perturbed_text']
                    tokens_changed += 1
                    applied.append({
                        'position':          pos,
                        'original_token':    token_dict['text'],
                        'perturbed_token':   best_cand['perturbed_text'],
                        'perturbation_type': best_cand['perturbation_type'],
                        'rule_applied':      best_cand['rule_applied'],
                        'confidence_drop':   best_cand['conf_drop'],
                        'labse_similarity':  best_cand.get('labse_similarity')
                    })

                    final = self.predict(' '.join(current_tokens))
                    self.query_count += 1

                    if final['label'] != original_label:
                        attack_success = True
                        break
                    improved = True
                    break

            if not improved:
                break

        # ── Phase 4: Also try perturbing already-perturbed tokens again ──
        # Stack perturbations: pyaar → pyar → pyarr
        if not attack_success and applied:
            for app in applied[:3]:
                pos        = app['position']
                new_token  = app['perturbed_token']
                script     = 'ROMAN'
                lang       = 'hi'
                new_dict   = {
                    'text': new_token,
                    'language': lang,
                    'script': script,
                    'position': pos,
                    'importance_score': 1.0,
                    'is_candidate': True
                }
                best_cand = self._get_best_perturbation(
                    new_dict, engine,
                    current_tokens,
                    original_text, original_label, original_conf
                )
                if best_cand and best_cand['conf_drop'] > 0.02:
                    current_tokens[pos] = best_cand['perturbed_text']
                    applied.append({
                        'position':          pos,
                        'original_token':    app['perturbed_token'],
                        'perturbed_token':   best_cand['perturbed_text'],
                        'perturbation_type': best_cand['perturbation_type'] + '_stacked',
                        'rule_applied':      best_cand['rule_applied'],
                        'confidence_drop':   best_cand['conf_drop'],
                        'labse_similarity':  best_cand.get('labse_similarity')
                    })

                    final = self.predict(' '.join(current_tokens))
                    self.query_count += 1

                    if final['label'] != original_label:
                        attack_success = True
                        break

        final_text   = ' '.join(current_tokens)
        final_result = self.predict(final_text)
        self.query_count += 1

        return {
            'original_text':          original_text,
            'adversarial_text':       final_text,
            'original_label':         original_label,
            'original_confidence':    original_conf,
            'adversarial_label':      final_result['label'],
            'adversarial_confidence': final_result['confidence'],
            'attack_successful':      attack_success,
            'perturbations':          applied,
            'n_perturbations':        len(applied),
            'budget_used':            len(applied) / max(len(tokens), 1),
            'model_queries':          self.query_count,
            'generation_time_secs':   round(time.time() - start, 3)
        }


# ─────────────────────────────────────────────
# SUBWORD RARITY SCORER
# ─────────────────────────────────────────────

class SubwordRarityScorer:
    """
    Scores perturbation candidates by subword rarity.
    Higher rarity = more likely to confuse transformer models.
    Rarity = average vocabulary ID of subword tokens.
    Higher ID = rarer token in training data.
    """
    def __init__(self, tokenizer, vocab_size=250002):
        self.tokenizer  = tokenizer
        self.vocab_size = vocab_size

    def rarity_score(self, word):
        """Returns normalized rarity score 0-1. Higher = rarer."""
        try:
            ids = self.tokenizer.encode(
                word, add_special_tokens=False
            )
            if not ids:
                return 0.0
            avg_id = sum(ids) / len(ids)
            return avg_id / self.vocab_size
        except:
            return 0.0

    def score_candidates(self, candidates, original_token,
                          alpha=0.6, beta=0.4):
        """
        Score each candidate combining:
        - alpha: subword rarity (higher = better)
        - beta:  edit distance from original (lower = more natural)

        Returns candidates sorted by combined score descending.
        """
        scored = []
        orig_rarity = self.rarity_score(original_token)

        for cand in candidates:
            pert_text   = cand['perturbed_text']
            pert_rarity = self.rarity_score(pert_text)

            # Rarity gain — how much rarer is the perturbation
            rarity_gain = max(0, pert_rarity - orig_rarity)

            # Edit distance penalty — prefer minimal changes
            edit_dist   = self._edit_distance(
                original_token.lower(), pert_text.lower()
            )
            max_len     = max(len(original_token), len(pert_text), 1)
            edit_penalty = edit_dist / max_len

            # Combined score
            score = alpha * rarity_gain - beta * edit_penalty

            scored.append({**cand, 'rarity_score': score,
                           'rarity': pert_rarity})

        # Sort by rarity score descending
        scored.sort(key=lambda x: x['rarity_score'], reverse=True)
        return scored

    def _edit_distance(self, s1, s2):
        """Simple Levenshtein distance."""
        m, n = len(s1), len(s2)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m+1):
            dp[i][0] = i
        for j in range(n+1):
            dp[0][j] = j
        for i in range(1, m+1):
            for j in range(1, n+1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],
                        dp[i][j-1],
                        dp[i-1][j-1]
                    )
        return dp[m][n]


class SearchControllerV3(SearchControllerV2):
    """
    Extends V2 with subword rarity-guided candidate selection.
    Queries model only for top-N rarity-scored candidates.
    """
    def __init__(self, predict_fn, get_conf_fn, validator,
                 tokenizer, beam_size=3, max_iterations=3,
                 confidence_threshold=0.40, top_n_rarity=5):
        super().__init__(predict_fn, get_conf_fn, validator,
                         beam_size, max_iterations,
                         confidence_threshold)
        self.rarity_scorer = SubwordRarityScorer(tokenizer)
        self.top_n_rarity  = top_n_rarity

    def _get_best_perturbation(self, token_dict, engine,
                                current_tokens, original_text,
                                original_label, original_conf):
        candidates = engine.generate_candidates(token_dict)
        if not candidates:
            return None

        current_text = ' '.join(current_tokens)
        valid_cands  = self.validator.validate_batch(
            current_text, candidates
        )
        if not valid_cands:
            return None

        # Score by subword rarity — query model only for top-N
        scored_cands = self.rarity_scorer.score_candidates(
            valid_cands, token_dict['text']
        )
        top_cands = scored_cands[:self.top_n_rarity]

        best_cand = None
        best_drop = -float('inf')

        for cand in top_cands:
            perturbed_sent = cand['perturbed_sentence']
            new_conf       = self.get_confidence(
                perturbed_sent, original_label
            )
            self.query_count += 1
            drop = original_conf - new_conf

            if drop > best_drop:
                best_drop = drop
                best_cand = cand
                best_cand['conf_after'] = new_conf
                best_cand['conf_drop']  = drop

        return best_cand
