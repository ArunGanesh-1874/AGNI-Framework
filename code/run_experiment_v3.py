import sys
sys.path.append('/workspace/comixattack/code')

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import pandas as pd
import jsonlines
import json
import os
import time
import unicodedata
from tqdm import tqdm
from modules import (
    InputProcessor, PerturbationEngine,
    ValidationModule, SearchController, OutputGenerator
)

MODEL_PATH = '/workspace/comixattack/models/xlmr_hinglish_sentiment'
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

print("CoMixAttack v3 — Loading models...")
tokenizer    = AutoTokenizer.from_pretrained(MODEL_PATH)
victim_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
).to(DEVICE)
victim_model.eval()

LABEL_MAP     = {0: 'positive', 1: 'negative', 2: 'neutral'}
LABEL_MAP_REV = {'positive': 0, 'negative': 1, 'neutral': 2}

def predict(text):
    inputs = tokenizer(
        text, return_tensors='pt',
        max_length=128, truncation=True, padding=True
    ).to(DEVICE)
    with torch.no_grad():
        outputs  = victim_model(**inputs)
        probs    = F.softmax(outputs.logits, dim=-1)
        pred_idx = torch.argmax(probs).item()
        conf     = probs[0][pred_idx].item()
    return {
        'label':      LABEL_MAP[pred_idx],
        'confidence': round(conf, 4),
        'probs':      probs[0].tolist()
    }

def get_label_confidence(text, label):
    inputs = tokenizer(
        text, return_tensors='pt',
        max_length=128, truncation=True, padding=True
    ).to(DEVICE)
    with torch.no_grad():
        outputs = victim_model(**inputs)
        probs   = F.softmax(outputs.logits, dim=-1)
    return round(probs[0][LABEL_MAP_REV[label]].item(), 4)

print("Loading LaBSE...")
labse = SentenceTransformer(
    'sentence-transformers/LaBSE', device='cpu'
)
print("All models ready")


# ─────────────────────────────────────────────
# FIX 1 — COMMON ENGLISH WORDS
# P2 character operations must not create
# real English words from Hindi tokens
# ─────────────────────────────────────────────

COMMON_ENGLISH_WORDS = {
    'tree','free','see','bee','fee','lee','tea','sea',
    'are','ore','ire','ere','ace','ice','age','ape',
    'the','and','for','not','but','can','had','has',
    'was','his','her','she','they','what','with','this',
    'that','from','have','will','been','when','who',
    'how','all','one','two','get','got','put','set',
    'bar','car','far','jar','tar','war','par','bat',
    'cat','hat','mat','rat','sat','bit','fit','hit',
    'kit','sit','wit','bot','dot','got','hot','lot',
    'pot','rot','cut','gut','hut','nut','put','rut',
    'bad','dad','had','lad','mad','sad','bed','fed',
    'led','red','bid','did','hid','kid','rid','cod',
    'god','mod','rod','bud','dud','mud','big','dig',
    'fig','gig','jig','pig','rig','wig','bog','cog',
    'dog','fog','hog','jog','log','bus','cup','pun',
    'run','sun','fun','gun','bun','nun','tin','win',
    'bin','din','fin','gin','kin','pin','sin','hip',
    'dip','lip','nip','rip','sip','tip','zip','hop',
    'cop','mop','pop','top','map','cap','gap','lap',
    'nap','rap','sap','tap','yap','mix','fix','six',
    'box','fox','pox','wax','tax','lax','max','bay',
    'day','gay','hay','jay','lay','may','pay','ray',
    'say','way','boy','coy','joy','toy','buy','guy',
    'try','dry','fly','fry','sky','spy','cry','why',
}


# ─────────────────────────────────────────────
# FIX 2 — BETTER TEXT QUALITY FILTER
# Removes broken encoding, non-Latin scripts
# that are not your target languages
# ─────────────────────────────────────────────

def is_valid_hinglish_text(text):
    """
    Check if text is valid Hinglish
    Reject: Gurmukhi, Telugu, Malayalam scripts
    Reject: Broken UTF-8 encoding
    Accept: Roman + Devanagari mix
    """
    # Check for broken encoding
    if 'à' in text or 'â€' in text or 'ð' in text:
        return False

    # Check for Gurmukhi script (Punjabi)
    gurmukhi_count = sum(
        1 for c in text if 0x0A00 <= ord(c) <= 0x0A7F
    )
    if gurmukhi_count > 2:
        return False

    # Check for Telugu script
    telugu_count = sum(
        1 for c in text if 0x0C00 <= ord(c) <= 0x0C7F
    )
    if telugu_count > 2:
        return False

    # Check for Malayalam script
    malayalam_count = sum(
        1 for c in text if 0x0D00 <= ord(c) <= 0x0D7F
    )
    if malayalam_count > 2:
        return False

    return True


def clean_text(text):
    """Remove noise tokens from text"""
    tokens = text.split()
    clean  = []
    for t in tokens:
        if t.startswith('http') or t.startswith('//'):
            continue
        if t.startswith('@') or t.startswith('#'):
            continue
        if t.startswith('RT'):
            continue
        if t.isdigit() or len(t) <= 1:
            continue
        if all(not c.isalnum() for c in t):
            continue
        # Skip tokens with broken encoding
        if 'à' in t or 'â' in t or 'ð' in t:
            continue
        clean.append(t)
    return ' '.join(clean)


# ─────────────────────────────────────────────
# FIX 3 — IMPROVED CANDIDATE VALIDATOR
# Rejects P2 candidates that accidentally
# create common English words from Hindi tokens
# ─────────────────────────────────────────────

def is_valid_candidate(
    original_token, perturbed_token,
    original_language, perturbation_type
):
    """
    Additional candidate quality check
    Prevents nonsensical perturbations
    """
    # P2 check — did character operation create English word?
    if perturbation_type == 'P2':
        if original_language == 'hi':
            if perturbed_token.lower() in COMMON_ENGLISH_WORDS:
                return False

    # Reject if perturbed token is too short
    if len(perturbed_token) < 2:
        return False

    # Reject if perturbed token is same as original
    if perturbed_token.lower() == original_token.lower():
        return False

    return True


# ─────────────────────────────────────────────
# EXPANDED HINDI LEXICON
# ─────────────────────────────────────────────

EXPANDED_HINDI_LEXICON = {
    'yaar','bhai','kya','nahi','nahin','haan','han',
    'achha','acha','theek','thik','bahut','bohot',
    'main','mein','mai','tum','aap','woh','hum',
    'hai','hain','tha','thi','the','hoga','hogi',
    'karo','karna','karta','karti','jana','aana',
    'aaj','kal','phir','ab','abhi','sirf','bas',
    'dost','ghar','log','desh','pyaar','pyar','dil',
    'mera','tera','uska','hamara','apna','unka',
    'khush','udaas','sach','acchi','buri','bura',
    'ke','ki','ka','ko','se','ne','pe','toh','bhi',
    'hi','wala','wali','wale','mujhe','tumhe','usse',
    'zyada','thoda','bilkul','zaroor','ekdum',
    'sahi','galat','bara','chota','maza','maja',
    'woh','wo','usne','unhe','chahiye','karega',
    'toh','to','tou','fir','kuch','koi','sab',
    'mere','tere','uske','hamare','isko','usko',
    'yeh','ye','kaafi','bohot','bohat','bahot',
    'achhe','acche','bure','bare','chote','naya',
    'nayi','purana','mast','zabardast','kamaal',
    'hua','hui','hue','raha','rahi','gaya','gayi',
    'liya','diya','kiya','dekha','suna','batao',
    'pata','maloom','samajh','logo','logon','dosto',
    'kaam','paisa','paise','waqt','raat','subah',
    'khana','paani','chai','dekho','suno','bolo',
    'jao','aao','ruko','mat','pakka','shayad',
    'hamesha','kabhi','accha','bekar','bakwaas',
    'sundar','khubsoorat','jabardast','aya','ayi',
    'naya','baar','sath','saath','milke','milna',
    'bolna','sunna','dekhna','likhna','padhna',
    'khana','peena','sona','uthna','baithna',
    'chalna','daudna','khelnа','gana','nachna',
    'izzat','respect','pyaar','mohabbat','dosti',
    'nafrat','gussa','khushi','dukh','dard',
    'umeed','bharosa','yakeen','shak','darr',
    'himmat','honsla','hausla','sahas','dil',
    'zindagi','maut','duniya','aasman','zameen',
    'pani','aag','hawa','mitti','phool','patta',
    'rasta','safar','manzil','mushkil','aasaan',
}


# ─────────────────────────────────────────────
# INITIALIZE ALL MODULES
# ─────────────────────────────────────────────

processor = InputProcessor(predict, get_label_confidence, 'hi-en')
processor.lang_det.HINDI_LEXICON = EXPANDED_HINDI_LEXICON

engine    = PerturbationEngine('hi-en')
validator = ValidationModule(labse, predict, 0.80, 'sentiment')
searcher  = SearchController(predict, get_label_confidence, validator)

print("All modules initialized")


# ─────────────────────────────────────────────
# PATCH SEARCH CONTROLLER
# Add candidate quality check before scoring
# ─────────────────────────────────────────────

original_search = searcher.search

def patched_search(processed_input, engine, budget=0.40):
    result = original_search(processed_input, engine, budget)

    # Post-filter perturbations
    # Remove any that created English words from Hindi
    clean_perturbations = []
    for p in result['perturbations']:
        if is_valid_candidate(
            p['original_token'],
            p['perturbed_token'],
            'hi',
            p['perturbation_type']
        ):
            clean_perturbations.append(p)

    result['perturbations']   = clean_perturbations
    result['n_perturbations'] = len(clean_perturbations)
    return result

searcher.search = patched_search
print("Search controller patched with quality filter")


# ─────────────────────────────────────────────
# EXPERIMENT CONFIGURATION
# ─────────────────────────────────────────────

EXPERIMENT_ID = 'E21_v3_hi_en_sentiment_xlmr'
output_dir    = '/workspace/comixattack/data/adversarial/hi_en_v3'
os.makedirs(output_dir, exist_ok=True)

output_gen = OutputGenerator(
    output_dir, EXPERIMENT_ID,
    'hi-en', 'sentiment', 'xlmr-hinglish'
)

test_df = pd.read_csv(
    '/workspace/comixattack/data/processed/hinglish_test_final.csv'
).head(500)

print(f"\nExperiment: {EXPERIMENT_ID}")
print(f"Test set: {len(test_df)} sentences")
print("Starting attack...")
print("=" * 55)

stats = {
    'total': 0, 'skipped': 0,
    'skipped_encoding': 0,
    'attacked': 0, 'successful': 0
}
experiment_start = time.time()

for idx, row in tqdm(
    test_df.iterrows(),
    total=len(test_df)
):
    text       = str(row['text'])
    gold_label = str(row['label'])
    stats['total'] += 1

    # FIX 2 — Reject broken encoding / wrong scripts
    if not is_valid_hinglish_text(text):
        stats['skipped_encoding'] += 1
        stats['skipped']          += 1
        continue

    # Clean noise tokens
    clean = clean_text(text)
    if len(clean.split()) < 3:
        stats['skipped'] += 1
        continue

    # Check model predicts correctly
    orig = predict(text)
    if orig['label'] != gold_label:
        stats['skipped'] += 1
        continue

    try:
        processed = processor.process(clean, budget=0.40)
        processed['original_text']       = text
        processed['original_prediction'] = orig['label']
        processed['original_confidence'] = orig['confidence']

        result = searcher.search(processed, engine, budget=0.40)
        result['original_text'] = text

    except Exception as e:
        stats['skipped'] += 1
        continue

    output_gen.save_example(result, idx)
    stats['attacked'] += 1

    if result['attack_successful']:
        stats['successful'] += 1

    if stats['attacked'] % 50 == 0:
        attacked = stats['attacked']
        asr      = stats['successful']/attacked if attacked>0 else 0
        elapsed  = time.time() - experiment_start
        tqdm.write(
            f"[{attacked}] ASR={asr:.1%} "
            f"Success={stats['successful']} "
            f"SkipEnc={stats['skipped_encoding']} "
            f"Time={elapsed/60:.1f}min"
        )

total_time = time.time() - experiment_start
attacked   = stats['attacked']
asr        = stats['successful']/attacked if attacked>0 else 0

print("\n" + "=" * 55)
print("EXPERIMENT V3 COMPLETE")
print("=" * 55)
print(f"Total:            {stats['total']}")
print(f"Skipped encoding: {stats['skipped_encoding']}")
print(f"Skipped total:    {stats['skipped']}")
print(f"Attacked:         {attacked}")
print(f"Successful:       {stats['successful']}")
print(f"ASR:              {asr:.4f} ({asr:.1%})")
print(f"Time:             {total_time/60:.1f} minutes")

output_gen.save_summary()
print("Done.")
