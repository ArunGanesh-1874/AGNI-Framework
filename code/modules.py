# CoMixAttack — All 5 Modules
# Module 1: Input Processor
# Module 2: Perturbation Engine  
# Module 3: Validation Module
# Module 4: Search Controller
# Module 5: Output Generator

import torch
import torch.nn.functional as F
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import jsonlines
import json
import os
import time
from typing import List, Dict, Tuple
from datetime import datetime

# ─────────────────────────────────────────────
# MODULE 1 — INPUT PROCESSOR
# ─────────────────────────────────────────────

class ScriptDetector:
    def detect(self, token: str) -> str:
        counts = {
            'DEVANAGARI': 0, 'TAMIL': 0,
            'BENGALI': 0, 'ROMAN': 0, 'OTHER': 0
        }
        for char in token:
            code = ord(char)
            if 0x0900 <= code <= 0x097F:
                counts['DEVANAGARI'] += 1
            elif 0x0B80 <= code <= 0x0BFF:
                counts['TAMIL'] += 1
            elif 0x0980 <= code <= 0x09FF:
                counts['BENGALI'] += 1
            elif (0x0041 <= code <= 0x005A or
                  0x0061 <= code <= 0x007A):
                counts['ROMAN'] += 1
            elif char.strip():
                counts['OTHER'] += 1
        dominant = max(counts, key=counts.get)
        return dominant if counts[dominant] > 0 else 'OTHER'


class LanguageDetector:
    HINDI_LEXICON = {
        'yaar','bhai','kya','nahi','nahin','haan','han',
        'achha','acha','theek','thik','bahut','bohot',
        'main','mein','mai','tum','aap','woh','hum',
        'hai','hain','tha','thi','the','hoga','hogi',
        'karo','karna','karta','karti','jana','aana',
        'aaj','kal','phir','ab','abhi','sirf','bas',
        'dost','ghar','log','desh','pyaar','pyar','dil',
        'mera','tera','uska','hamara','apna','unka',
        'khush','udaas','sach','jhooth','acchi','buri',
        'ke','ki','ka','ko','se','ne','pe','toh','bhi',
        'hi','wala','wali','wale','mujhe','tumhe','usse',
        'zyada','thoda','bilkul','zaroor','ekdum',
        'sahi','galat','bara','chota','lamba',
        'maza','maja','movie','film','gana',
        'subah','shaam','raat','din','paise','kaam',
        'naukri','padhai','school','college','woh','wo',
        'usne','unhe','humne','tumne','maine',
        'chahiye','karega','karegi','hona','rehna',
        'bolna','sunna','padhna','likhna',
        'abhi','toh','to','tou','phir','fir',
        'bohot','bahot','bhaut','achi','bura','buraa',
        'accha','achhi','theek','thik','teek',
    }

    TAMIL_LEXICON = {
        'vanakkam','romba','nalla','enna','paaru',
        'iruku','illai','inge','ange','epdi','sollu',
        'machan','da','di','nga','seri','ponga',
        'naan','nee','avan','aval','unna','eppo',
        'konjam','adhigam','supera','semma','mass',
        'vera','level','boss','dei','machi','theriuma',
    }

    def detect(self, token: str, script: str,
               language_pair: str) -> str:
        if script == 'DEVANAGARI': return 'hi'
        if script == 'TAMIL':      return 'ta'
        if script == 'BENGALI':    return 'bn'
        token_lower = token.lower()
        if 'hi' in language_pair:
            if token_lower in self.HINDI_LEXICON:
                return 'hi'
        if 'ta' in language_pair:
            if token_lower in self.TAMIL_LEXICON:
                return 'ta'
        return 'en'


class InputProcessor:
    def __init__(self, predict_fn, get_conf_fn,
                 language_pair='hi-en'):
        self.predict        = predict_fn
        self.get_confidence = get_conf_fn
        self.language_pair  = language_pair
        self.script_det     = ScriptDetector()
        self.lang_det       = LanguageDetector()

    def process(self, text: str,
                budget: float = 0.30) -> Dict:
        tokens      = text.strip().split()
        orig_result = self.predict(text)

        processed_tokens = []
        for i, token in enumerate(tokens):
            script   = self.script_det.detect(token)
            language = self.lang_det.detect(
                token, script, self.language_pair
            )
            processed_tokens.append({
                'text': token, 'position': i,
                'script': script, 'language': language,
                'importance_score': 0.0,
                'is_candidate': False
            })

        # Importance ranking
        scores = []
        for i, token in enumerate(tokens):
            if len(token) <= 1:
                scores.append((i, 0.0))
                continue
            masked        = tokens.copy()
            masked[i]     = '[MASK]'
            masked_text   = ' '.join(masked)
            masked_conf   = self.get_confidence(
                masked_text, orig_result['label']
            )
            importance = orig_result['confidence'] - masked_conf
            scores.append((i, importance))

        scores.sort(key=lambda x: x[1], reverse=True)

        for pos, score in scores:
            processed_tokens[pos]['importance_score'] = score

        n_candidates  = max(1, int(len(tokens) * budget))
        top_positions = {
            pos for pos, _ in scores[:n_candidates]
        }
        for tok in processed_tokens:
            if tok['position'] in top_positions:
                tok['is_candidate'] = True

        return {
            'original_text':       text,
            'tokens':              processed_tokens,
            'original_prediction': orig_result['label'],
            'original_confidence': orig_result['confidence'],
            'n_tokens':            len(tokens),
            'n_candidates':        n_candidates
        }


# ─────────────────────────────────────────────
# MODULE 2 — PERTURBATION ENGINE
# ─────────────────────────────────────────────

class PhoneticGenerator:
    HINDI_RULES = {
        'bh':['b','v'], 'kh':['k','q'], 'gh':['g'],
        'sh':['s'], 'ph':['f','p'], 'dh':['d'],
        'th':['t'], 'ch':['c'], 'aa':['a','ah'],
        'ai':['ay','ae'], 'oo':['u','uu'],
        'ee':['i','y'], 'ou':['o','ow'],
        'j':['z'], 'v':['w','b'], 'q':['k'],
    }
    TAMIL_RULES = {
        'zh':['l','z','r'], 'ng':['n'],
        'tt':['t','dd'], 'kk':['k','g'],
        'pp':['p','b'], 'rr':['r'],
        'aa':['a','ah'], 'ii':['i','ee'],
        'uu':['u','oo'], 'ai':['ay','ei'],
    }

    def __init__(self, language_pair):
        self.rules = (self.HINDI_RULES
                      if 'hi' in language_pair
                      else self.TAMIL_RULES)

    def generate(self, token_dict):
        token    = token_dict['text']
        language = token_dict['language']
        if language == 'en' or len(token) < 3:
            return []
        candidates  = []
        token_lower = token.lower()
        seen        = set()
        for pattern, replacements in self.rules.items():
            if pattern in token_lower:
                for rep in replacements:
                    perturbed = token_lower.replace(
                        pattern, rep, 1
                    )
                    if perturbed != token_lower and \
                       perturbed not in seen:
                        seen.add(perturbed)
                        candidates.append({
                            'perturbed_text':    perturbed,
                            'perturbation_type': 'P1',
                            'rule_applied':      f'{pattern}→{rep}',
                            'original_token':    token
                        })
        return candidates


class CharacterGenerator:
    def generate(self, token_dict):
        token    = token_dict['text']
        language = token_dict['language']
        if language == 'en' or len(token) < 3:
            return []
        candidates  = []
        token_lower = token.lower()

        # Repetition
        for i in range(1, len(token_lower)-1):
            if token_lower[i].isalpha():
                p = token_lower[:i]+token_lower[i]*2+token_lower[i+1:]
                candidates.append({
                    'perturbed_text':    p,
                    'perturbation_type': 'P2',
                    'rule_applied':      f'repeat_{i}',
                    'original_token':    token
                })

        # Deletion
        for i in range(1, len(token_lower)-1):
            if token_lower[i].isalpha():
                p = token_lower[:i]+token_lower[i+1:]
                if len(p) >= 3:
                    candidates.append({
                        'perturbed_text':    p,
                        'perturbation_type': 'P2',
                        'rule_applied':      f'delete_{i}',
                        'original_token':    token
                    })

        # Swap
        for i in range(1, len(token_lower)-1):
            chars       = list(token_lower)
            chars[i], chars[i+1] = chars[i+1], chars[i]
            p           = ''.join(chars)
            if p != token_lower:
                candidates.append({
                    'perturbed_text':    p,
                    'perturbation_type': 'P2',
                    'rule_applied':      f'swap_{i}',
                    'original_token':    token
                })
        return candidates[:6]


class ScriptConverter:
    SCRIPT_MAP = {
        'hi': sanscript.DEVANAGARI,
        'ta': sanscript.TAMIL,
        'bn': sanscript.BENGALI
    }

    def generate(self, token_dict):
        token    = token_dict['text']
        language = token_dict['language']
        script   = token_dict['script']
        if language == 'en' or language not in self.SCRIPT_MAP:
            return []
        if len(token) < 3 or not token[0].isalpha():
            return []
        try:
            if script == 'ROMAN':
                converted = transliterate(
                    token.lower(),
                    sanscript.ITRANS,
                    self.SCRIPT_MAP[language]
                )
                if converted and converted != token and \
                   len(converted) >= 2:
                    return [{
                        'perturbed_text':    converted,
                        'perturbation_type': 'P3',
                        'rule_applied':      f'roman_to_{language}',
                        'original_token':    token
                    }]
        except Exception:
            pass
        return []


class TransliterationVariantGenerator:
    HINDI_VARIANTS = {
        'pyaar':['pyar','piyar','piyaar'],
        'pyar':['pyaar','piyar'],
        'bhai':['bhaai','bhay'],
        'kya':['kyaa','kia'],
        'nahi':['nahin','nai','ni'],
        'nahin':['nahi','nai'],
        'haan':['han','ha','haa'],
        'achha':['acha','accha'],
        'acha':['achha','accha'],
        'bahut':['bohot','bahot','bhaut'],
        'bohot':['bahut','bahot'],
        'main':['mein','mai'],
        'mein':['main','mai'],
        'aaj':['aaz','aj'],
        'yaar':['yar','yaaar'],
        'theek':['thik','teek'],
        'thik':['theek','teek'],
        'bilkul':['bilcul','bilqul'],
        'mujhe':['muje','mujhey'],
        'achhi':['achi','acchi'],
        'achi':['achhi','acchi'],
        'phir':['fir'],
        'fir':['phir'],
        'woh':['wo','voh','vo'],
        'wo':['woh','voh'],
        'toh':['to','tou'],
        'to':['toh','tou'],
        'khush':['kush','khus'],
        'udaas':['udas'],
        'maza':['maja','mazaa'],
        'maja':['maza'],
        'sach':['saach'],
        'ghar':['gher'],
        'karo':['kro'],
        'jana':['jaana'],
        'bas':['bus'],
        'bus':['bas'],
        'dost':['doast'],
        'abhi':['abi'],
        'bhi':['bi','bhii'],
        'zaroor':['zarur','zarour'],
    }

    TAMIL_VARIANTS = {
        'vanakkam':['vanakam','vanagam'],
        'romba':['rombha','rumba'],
        'nalla':['nala','nallah'],
        'paaru':['paro','paaro'],
        'iruku':['irukku'],
        'illai':['ilai','illa'],
        'machan':['machen','machaan'],
        'sollu':['solu'],
        'enna':['yenna','ena'],
        'seri':['sari'],
    }

    def __init__(self, language_pair):
        self.variants = (self.HINDI_VARIANTS
                         if 'hi' in language_pair
                         else self.TAMIL_VARIANTS)

    def generate(self, token_dict):
        token    = token_dict['text']
        language = token_dict['language']
        if language == 'en':
            return []
        token_lower = token.lower()
        if token_lower not in self.variants:
            return []
        return [{
            'perturbed_text':    v,
            'perturbation_type': 'P4',
            'rule_applied':      'translit_variant',
            'original_token':    token
        } for v in self.variants[token_lower]
          if v != token_lower]


class MorphologicalGenerator:
    HINDI_PARADIGMS = {
        'achha':['acche','acchi'],
        'accha':['acche','acchi'],
        'bura':['bure','buri'],
        'bara':['bare','bari'],
        'chota':['chote','choti'],
        'khaana':['khaane','khaata','khaati'],
        'jaana':['jaane','jaata','jaati'],
        'aana':['aane','aata','aati'],
        'bolta':['bolte','bolti'],
        'karta':['karte','karti'],
        'chahiye':['chahie'],
        'chahie':['chahiye'],
        'karega':['karegi','karenge'],
        'karegi':['karega','karenge'],
        'hoga':['hogi','honge'],
        'hogi':['hoga','honge'],
        'dost':['doston','dosto'],
        'ghar':['gharon'],
        'baat':['baaton'],
        'log':['logon'],
    }

    TAMIL_PARADIGMS = {
        'irukken':['irukkeaan','irukkean'],
        'paakuren':['paakureen'],
        'illai':['illa','illaiye'],
        'illa':['illai','illaiye'],
    }

    def __init__(self, language_pair):
        self.paradigms = (self.HINDI_PARADIGMS
                          if 'hi' in language_pair
                          else self.TAMIL_PARADIGMS)

    def generate(self, token_dict):
        token    = token_dict['text']
        language = token_dict['language']
        if language == 'en':
            return []
        token_lower = token.lower()
        if token_lower not in self.paradigms:
            return []
        return [{
            'perturbed_text':    v,
            'perturbation_type': 'P5',
            'rule_applied':      'morph_variant',
            'original_token':    token
        } for v in self.paradigms[token_lower]
          if v != token_lower]


class PerturbationEngine:
    def __init__(self, language_pair='hi-en',
                 active_types=None):
        self.active_types = active_types or [
            'P1','P2','P3','P4','P5'
        ]
        self.generators = {
            'P1': PhoneticGenerator(language_pair),
            'P2': CharacterGenerator(),
            'P3': ScriptConverter(),
            'P4': TransliterationVariantGenerator(language_pair),
            'P5': MorphologicalGenerator(language_pair)
        }

    def generate_candidates(self, token_dict):
        all_candidates = []
        for ptype in self.active_types:
            if ptype in self.generators:
                all_candidates.extend(
                    self.generators[ptype].generate(token_dict)
                )
        return all_candidates


# ─────────────────────────────────────────────
# MODULE 3 — VALIDATION MODULE
# ─────────────────────────────────────────────

SENTIMENT_ANTONYMS = set()
for w1, w2 in [
    ('khush','udaas'), ('achha','bura'), ('achhi','buri'),
    ('accha','bura'), ('pyaar','nafrat'), ('maza','takleef'),
    ('sahi','galat'), ('sundar','badsurat'),
    ('good','bad'), ('great','terrible'), ('amazing','awful'),
    ('excellent','horrible'), ('love','hate'),
    ('happy','sad'), ('best','worst'),
    ('wonderful','dreadful'), ('fantastic','pathetic'),
]:
    SENTIMENT_ANTONYMS.add((w1, w2))
    SENTIMENT_ANTONYMS.add((w2, w1))


class ValidationModule:
    def __init__(self, labse_model, predict_fn,
                 threshold=0.80, task='sentiment'):
        self.labse     = labse_model
        self.predict   = predict_fn
        self.threshold = threshold
        self.task      = task

    def _labse_similarity(self, text1, text2):
        emb = self.labse.encode(
            [text1, text2],
            show_progress_bar=False,
            device='cpu'
        )
        return float(
            cosine_similarity([emb[0]], [emb[1]])[0][0]
        )

    def validate(self, original_text, perturbed_text,
                 perturbation_type,
                 original_token=None,
                 perturbed_token=None):

        # P3 — task consistency check
        if perturbation_type == 'P3':
            orig = self.predict(original_text)
            pert = self.predict(perturbed_text)
            same = orig['label'] == pert['label']
            return {'is_valid': same,
                    'validator': 'B_TaskConsistency'}

        # All others — LaBSE check
        sim = self._labse_similarity(
            original_text, perturbed_text
        )
        if sim < self.threshold:
            return {'is_valid': False,
                    'similarity': sim,
                    'validator': 'A_LaBSE'}

        # Sentiment — antonym check
        if self.task == 'sentiment' and \
           original_token and perturbed_token:
            pair = (original_token.lower(),
                    perturbed_token.lower())
            if pair in SENTIMENT_ANTONYMS:
                return {'is_valid': False,
                        'validator': 'C_Antonym'}

        return {'is_valid': True,
                'similarity': sim,
                'validator': 'A_LaBSE'}

    def validate_batch(self, original_text, candidates):
        valid = []
        for c in candidates:
            tokens = original_text.split()
            for i, t in enumerate(tokens):
                if t.lower() == c['original_token'].lower():
                    tokens[i] = c['perturbed_text']
                    break
            perturbed_text = ' '.join(tokens)
            result = self.validate(
                original_text, perturbed_text,
                c['perturbation_type'],
                c['original_token'],
                c['perturbed_text']
            )
            if result['is_valid']:
                c['perturbed_sentence'] = perturbed_text
                c['labse_similarity']   = result.get(
                    'similarity'
                )
                valid.append(c)
        return valid


# ─────────────────────────────────────────────
# MODULE 4 — SEARCH CONTROLLER
# ─────────────────────────────────────────────

class SearchController:
    def __init__(self, predict_fn, get_conf_fn,
                 validator):
        self.predict        = predict_fn
        self.get_confidence = get_conf_fn
        self.validator      = validator
        self.query_count    = 0

    def search(self, processed_input, engine,
               budget=0.30):
        start          = time.time()
        self.query_count = 0

        original_text  = processed_input['original_text']
        original_label = processed_input['original_prediction']
        original_conf  = processed_input['original_confidence']
        tokens         = processed_input['tokens']

        current_tokens   = [t['text'] for t in tokens]
        applied          = []
        tokens_changed   = 0
        max_changes      = max(1, int(len(tokens) * budget))
        attack_success   = False
        current_pred     = original_label
        current_conf     = original_conf

        sorted_tokens = sorted(
            tokens,
            key=lambda x: x['importance_score'],
            reverse=True
        )

        for token_dict in sorted_tokens:
            if tokens_changed >= max_changes:
                break
            if not token_dict['is_candidate']:
                continue

            position   = token_dict['position']
            candidates = engine.generate_candidates(token_dict)
            if not candidates:
                continue

            current_text   = ' '.join(current_tokens)
            valid_cands    = self.validator.validate_batch(
                current_text, candidates
            )
            if not valid_cands:
                continue

            best_cand     = None
            best_drop     = -float('inf')

            for cand in valid_cands:
                perturbed_sent = cand['perturbed_sentence']
                new_conf       = self.get_confidence(
                    perturbed_sent, original_label
                )
                self.query_count += 1
                drop = original_conf - new_conf
                if drop > best_drop:
                    best_drop              = drop
                    best_cand              = cand
                    best_cand['conf_after'] = new_conf
                    best_cand['conf_drop']  = drop

            if best_cand:
                current_tokens[position] = (
                    best_cand['perturbed_text']
                )
                tokens_changed += 1
                applied.append({
                    'position':          position,
                    'original_token':    token_dict['text'],
                    'perturbed_token':   best_cand['perturbed_text'],
                    'perturbation_type': best_cand['perturbation_type'],
                    'rule_applied':      best_cand['rule_applied'],
                    'confidence_drop':   best_cand['conf_drop'],
                    'labse_similarity':  best_cand.get('labse_similarity')
                })

                current_text = ' '.join(current_tokens)
                final        = self.predict(current_text)
                self.query_count += 1
                current_pred     = final['label']
                current_conf     = final['confidence']

                if current_pred != original_label:
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
            'budget_used':            tokens_changed / max(len(tokens), 1),
            'model_queries':          self.query_count,
            'generation_time_secs':   round(time.time() - start, 3)
        }


# ─────────────────────────────────────────────
# MODULE 5 — OUTPUT GENERATOR
# ─────────────────────────────────────────────

class OutputGenerator:
    def __init__(self, output_dir, experiment_id,
                 language_pair, task, victim_model_name):
        self.output_dir    = output_dir
        self.experiment_id = experiment_id
        self.language_pair = language_pair
        self.task          = task
        self.victim_model  = victim_model_name
        os.makedirs(output_dir, exist_ok=True)

        self.corpus_path = os.path.join(
            output_dir, f'{experiment_id}_corpus.jsonl'
        )
        self.stats = {
            'total': 0, 'successful': 0, 'failed': 0,
            'total_queries': 0, 'total_time': 0.0,
            'p1_used': 0, 'p2_used': 0, 'p3_used': 0,
            'p4_used': 0, 'p5_used': 0,
        }

    def save_example(self, result, idx):
        record = {
            'id':                     f"{self.experiment_id}_{idx:05d}",
            'original_text':          result['original_text'],
            'adversarial_text':       result['adversarial_text'],
            'original_label':         result['original_label'],
            'adversarial_label':      result['adversarial_label'],
            'original_confidence':    result['original_confidence'],
            'adversarial_confidence': result['adversarial_confidence'],
            'attack_successful':      result['attack_successful'],
            'n_perturbations':        result['n_perturbations'],
            'budget_used':            result['budget_used'],
            'perturbations':          result['perturbations'],
            'model_queries':          result['model_queries'],
            'generation_time_secs':   result['generation_time_secs'],
            'language_pair':          self.language_pair,
            'task':                   self.task,
            'victim_model':           self.victim_model,
            'timestamp':              datetime.now().isoformat()
        }
        with jsonlines.open(self.corpus_path, mode='a') as w:
            w.write(record)

        self.stats['total']         += 1
        self.stats['total_queries'] += result['model_queries']
        self.stats['total_time']    += result['generation_time_secs']
        if result['attack_successful']:
            self.stats['successful'] += 1
        else:
            self.stats['failed']     += 1
        for p in result['perturbations']:
            key = f"{p['perturbation_type'].lower()}_used"
            if key in self.stats:
                self.stats[key] += 1

    def print_progress(self, interval=100):
        s   = self.stats
        tot = s['total']
        if tot == 0 or tot % interval != 0:
            return
        asr = s['successful'] / tot
        avg_q = s['total_queries'] / tot
        avg_t = s['total_time'] / tot
        print(
            f"  [{tot}] ASR={asr:.1%} "
            f"AvgQ={avg_q:.1f} "
            f"AvgT={avg_t:.2f}s"
        )

    def save_summary(self):
        s   = self.stats
        tot = s['total']
        summary = {
            'experiment_id': self.experiment_id,
            'language_pair': self.language_pair,
            'task':          self.task,
            'victim_model':  self.victim_model,
            'completed_at':  datetime.now().isoformat(),
            'results': {
                'total':      tot,
                'successful': s['successful'],
                'failed':     s['failed'],
                'asr':        round(s['successful']/max(tot,1), 4),
                'avg_queries':round(s['total_queries']/max(tot,1), 2),
                'avg_time':   round(s['total_time']/max(tot,1), 3),
                'p1_used':    s['p1_used'],
                'p2_used':    s['p2_used'],
                'p3_used':    s['p3_used'],
                'p4_used':    s['p4_used'],
                'p5_used':    s['p5_used'],
            }
        }
        path = self.corpus_path.replace(
            '_corpus.jsonl', '_summary.json'
        )
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Summary saved: {path}")
        return summary

print("✓ All modules defined successfully")
