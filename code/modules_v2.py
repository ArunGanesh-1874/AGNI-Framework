# CoMixAttack v2 — All 6 Modules
# Module 1: Input Processor
# Module 2: Perturbation Engine (P1-P6)
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
        # Pronouns
        'yaar','bhai','kya','nahi','nahin','haan','han',
        'main','mein','mai','tum','aap','woh','hum','wo',
        'mera','tera','uska','hamara','apna','unka','tumhara',
        'mujhe','tumhe','usse','unhe','humne','tumne','maine',
        'usne','inhe','jinhe','jisko','jisne','kisne','kisko',
        # Common verbs
        'hai','hain','tha','thi','the','hoga','hogi','honge',
        'karo','karna','karta','karti','karte','karein',
        'jana','aana','jaana','aaana','jao','aao','gaya','gayi',
        'bolna','sunna','padhna','likhna','dekhna','khana',
        'pina','sona','uthna','baithna','chalna','daudna',
        'lena','dena','milna','khelna','hasna','rona',
        'samajhna','batana','poochna','sochna','banana',
        'karega','karegi','karenge','karunga','karungi',
        'hona','rehna','raho','rehe','raha','rahi',
        'hua','hui','hue','ho','hoga','hogi',
        'chahiye','chahie','chahta','chahti','chahte',
        # Adjectives
        'achha','acha','accha','achhi','achi','acchi',
        'theek','thik','teek','bura','bure','buri',
        'bahut','bohot','bahot','bhaut','zyada','thoda',
        'bilkul','zaroor','ekdum','sirf','bas','hi',
        'sahi','galat','bara','bada','chota','lamba','chhota',
        'sundar','khubsoorat','ganda','saaf','purana','naya',
        'khush','udaas','pareshan','thaka','bored','excited',
        'mazedaar','bekar','zabardast','kamaal','shandar',
        'mushkil','aasaan','seedha','ulta','tez','dheema',
        'gehri','halki','bhari','khali','poora','adha',
        'pehla','doosra','teesra','aakhri','agla','pichhla',
        # Nouns
        'dost','ghar','log','desh','pyaar','pyar','dil',
        'subah','shaam','raat','din','paise','kaam',
        'naukri','padhai','school','college','university',
        'movie','film','gana','gaana','khana','paani',
        'duniya','zindagi','jindgi','life','time','jagah',
        'baat','cheez','kaam','naam','kaam','saal','mahina',
        'insaan','aadmi','aurat','bachha','bacha','ladka','ladki',
        'maa','baap','papa','mama','didi','bhaiya','bhai',
        'shahar','gaon','rasta','safar','mulk','watan',
        'dil','dimaag','haath','aankhein','chehara',
        'awaaz','khwaab','sapna','yaad','waqt','pal',
        'rishta','pyaar','mohabbat','nafrat','dosti','dushman',
        # Adverbs/particles
        'ke','ki','ka','ko','se','ne','pe','toh','bhi',
        'wala','wali','wale','phir','fir','ab','abhi',
        'kal','aaj','parso','kabhi','kab','kahan','kyun',
        'isliye','lekin','magar','par','aur','ya','agar',
        'tab','jab','jab','to','tou','toh','kyunki',
        'matlab','yani','seedha','directly','actually',
        'sach','jhooth','sacchi','pakka','bilkul','hamesha',
        'kabhi','rarely','aksar','often','roz','daily',
        'jaldi','dheere','achanak','suddenly','finally','aakhir',
        # Emotions/expressions
        'maza','maja','mazaa','dard','takleef','khushi',
        'ghussa','nafrat','pyaar','mohabbat','ishq',
        'uff','arrey','yaar','oye','arre','hey','haye',
        'wah','waah','shabash','shukria','shukriya','thanks',
        'sorry','maafi','please','please','help','madad',
        'haha','lol','lmao','omg','wtf','damn',
        # Common Hinglish words
        'matlab','scene','setting','jugaad','timepass',
        'bakwaas','faltu','solid','mast','bindaas',
        'chill','tension','stress','pressure','fun',
        'party','trip','plan','idea','problem','solution',
        'feeling','mood','vibe','energy','attitude',
        'boss','yaar','dude','bro','sis','guys',
        'kuch','koi','sab','sabhi','kaafi','thodi',
        'ek','do','teen','char','paanch','das','sau',
        'pehle','baad','saath','bina','liye','waste',
        'kyunki','isliye','warna','tabhi','tabhi',
        'dekho','suno','bolo','jao','aao','ruko',
        'rehne','jaane','aane','karne','bolne','sunne',
        'unka','inका','jinका','kisका','apneaap',
        'seedhi','ulti','sidhi','sachi','pakki',
        'chhod','chhor','bhool','yaad','pata','malum',
        'pata','maloom','samajh','samjha','samjhi',
        'zyaada','thodi','kaafi','bahot','itna','utna',
        'aisa','waisa','kaisa','jaisa','aise','waise',
        'idhar','udhar','kahan','yahan','wahan','neeche',
        'upar','andar','bahar','paas','door','saamne',
        'peeche','dono','teeno','sabka','sabke','sabki',
    }

    TAMIL_LEXICON = {
        # Greetings/expressions
        'vanakkam','vanakam','super','superaa','supera',
        'romba','rombha','rumba','konjam','adhigam',
        'nalla','nallaa','nallave','semma','mass','vera','level',
        # Pronouns
        'naan','nee','avan','aval','naanga','neenga',
        'avanga','inga','avaru','ivaru','yaar','yaaru',
        'unna','enna','avana','avala','namma','ungaluku',
        'enaku','avanuku','avaluku','namaku','ellorum',
        'avanugaluku','avalugaluku','naangaluku','neengaluku',
        'avan','aval','avanga','ivan','ival','ivanga',
        # Verbs
        'paaru','paaren','paakuren','paakureen','paarunga',
        'iruku','iruken','irukken','irukkean','illai','illa',
        'sollu','solunga','solluren','ketku','kelvi',
        'vaa','vaanga','po','ponga','iru','irunga',
        'saapu','saapidu','kudu','edu','vidu','podu',
        'paesu','pesu','pesunga','kelu','kelunga',
        'theriyum','theriyaadu','puriguthu','puriyala',
        'pannuven','panren','panuvom','seiven','seivom',
        'varuvaan','varuvaal','varuvaanga','poovaan','poovaal',
        'solluvaen','solluvaanga','ketpaen','ketpaanga',
        'paakuvaen','paakuvaanga','pannuvaen','pannuvaanga',
        'tharuvaen','tharuvaanga','vaanguvaen','vaanguvaanga',
        'padikiren','padikireen','padikaama','padichuten',
        'paarkirein','paarkiren','paarkiraanga','paarthuten',
        'pesikireen','pesikiren','pesikiraanga','pesichuten',
        'seikireen','seikiren','seikitraanga','seithuten',
        # Adjectives - expanded
        'nallavan','nallavana','kettavan','periya','sinna',
        'azhaga','azhagana','mosama','mosaman','bayangara',
        'arumai','arumaiyana','mukkiyam','mukkiyamana',
        'nalladhu','kettadhu','periadhu','sinnadhu',
        'azhagana','mosam','bayangaram','arumaiyana',
        'mudiyum','mudiyaadhu','vennum','vaenum','venda',
        'santhosham','sandhosham','dukham','kovam','aasai',
        'bayam','nambikai','unmai','poi','nalla','ketta',
        'mukkiyam','saadharana','asamaanya','therinja','theriyaadha',
        # Sentiment words
        'nalla','nalladhu','nallavan','nallavana','nallaa',
        'ketta','kettadhu','kettavan','kettavana','kettaa',
        'super','superaa','supera','semma','mass','vera level',
        'mokka','mokkaa','worst','worsta','pathetic','pathetica',
        'happy','happya','sad','sada','angry','angrya',
        'love','lovea','hate','hatea','like','likea',
        'beautiful','beautifula','ugly','uglya','cute','cutea',
        'santhosham','sandhosham','dukham','kovam','aasai',
        'payam','bayam','nambikai','niraisaa','erichal',
        'kalakuran','kalakkiraan','mass pannuran','semma pannuran',
        # Common expressions
        'machan','machen','machaan','da','di','nga','pa','ma',
        'seri','saeri','otay','okay','sari','dei','dei',
        'epdi','eppadi','eppo','eppO','enge','enga',
        'enna','yenna','ena','ean','yen','yaen',
        'inge','inga','ange','anga','here','there',
        'theriuma','theriyuma','theruma','teriuma',
        'aama','aaamaa','illaya','illaya','solluda',
        'poda','podanga','mokka','mokkaya','loose','loosa',
        'scene','setting','matter','problem','solution',
        'kaduppu','kaduppe','bore','borea','interest','interestaa',
        'feel','feelaa','mood','moodaa','vibe','vibea',
        'kadhal','anbu','piriya','virumbu','pidi','pidika',
        'veruppu','erchu','erichal','kovam','thappu','sari',
        'unmai','poi','nambikkai','nambikai','thelivu','sandegam',
        # Social media expressions
        'lol','lola','haha','hahaha','omg','omga','wow','wowa',
        'bro','broa','da','di','pa','ma','nga','ra',
        'illa','illai','ama','aama','seri','sari','okay','ok',
        'yellam','ellam','onnum','onum','yaarum','aarum',
        'epovum','eppoovum','ennavum','onnavum','evanum','evanum',
        # Time/place - expanded
        'ippo','ippO','appo','appO','naalaiku','inniku',
        'mudila','munnaadi','pinnaadi','mela','kela',
        'vaitu','veetla','schoolla','collegela','officela',
        'kaalaila','maalaila','raathiri','pakal','neram',
        'inniki','naalaiku','netchathiram','vaaram','maasam',
        'idatha','adatha','yedathula','engeyum','engum',
        'veetula','schoolla','collegela','officela','hospitalla',
        # Intensifiers
        'romba','rombha','rumba','konjam','adhigam','oru vaela',
        'super','semma','mass','vera level','kalakkal',
        'mokka','bekar','worst','pathetic','useless',
        'thaan','dhaan','kuda','kooda','mattum','matum',
        'ellam','yellam','onnum','onum','vera','veru',
        # Hate speech related Tamil words
        'thevdiya','ommala','punda','otha','koothi',
	'sunni','oombu','olu','poolu','ooka','soothu', 
        'paiyan','ponna','aalu','naaye','kazhudhai',
        'mudarikku','thayoli','loosu','paithiyam','mental',
        'vekkam','maanam','mosam','kevalama','azhukku',
        'racist','casteist','communal','violence','attack',
        'hate','hateful','offensive','abuse','abusive',
        # Discourse markers
        'aana','aanaa','aadha','aadhaan','aaganaa','aagadhu',
        'adhukku','idhukkku','idhukku','andha','indha',
        'appuram','ipuram','mudichitu','mudinja','mudiyala',
        'therinja','theriyaama','pakkaama','paathitten','illana',
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
                budget: float = 0.40,
                restrict_to_target_language: bool = False) -> Dict:
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

        # English sentiment words that should always be candidates
        ENGLISH_SENTIMENT = {
            'good','bad','great','nice','awesome','amazing',
            'wonderful','terrible','horrible','awful','excellent',
            'happy','sad','angry','love','hate','like','dislike',
            'beautiful','ugly','best','worst','better','worse',
            'true','false','right','wrong','clean','dirty',
            'funny','boring','interesting','serious','stupid',
            'smart','strong','weak','fast','slow','big','small',
            'hot','cold','sweet','bitter','rich','poor','free',
            'fake','real','original','corrupt','honest','violent',
            'racist','offensive','abusive','toxic','positive',
            'negative','neutral','support','oppose','attack',
            'defend','promote','spread','share','post','comment',
        }

        n_candidates  = max(1, int(len(tokens) * budget))
        top_positions = {
            pos for pos, _ in scores[:n_candidates]
        }
        # Force-include English sentiment words as candidates
        # — only when NOT restricted to target-language tokens (AGNI-Full behavior)
        for tok in processed_tokens:
            if tok['position'] in top_positions:
                tok['is_candidate'] = True
            elif (not restrict_to_target_language) and tok['text'].lower() in ENGLISH_SENTIMENT:
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
# MODULE 2 — PERTURBATION ENGINE (P1-P6)
# ─────────────────────────────────────────────

class PhoneticGenerator:
    HINDI_RULES = {
        'bh':['b','v'], 'kh':['k','q'], 'gh':['g'],
        'sh':['s','ch'], 'ph':['f','p'], 'dh':['d'],
        'th':['t'], 'ch':['c','chh'], 'aa':['a','ah'],
        'ai':['ay','ae','ei'], 'oo':['u','uu','ou'],
        'ee':['i','y','ie'], 'ou':['o','ow','au'],
        'j':['z','dj'], 'v':['w','b'], 'q':['k','qu'],
        'ng':['n'], 'ck':['k'], 'tt':['t'],
        'nn':['n'], 'rr':['r'], 'll':['l'],
        'iya':['ia','ya'], 'iye':['ie','ye'],
        'wa':['va','waa'], 'wi':['vi','wee'],
        'uu':['u','oo'], 'ii':['i','ee'],
    }
    TAMIL_RULES = {
        'zh':['l','z','r','rl'], 'ng':['n','ng'],
        'tt':['t','dd','d'], 'kk':['k','g','gg'],
        'pp':['p','b','bb'], 'rr':['r','rr'],
        'aa':['a','ah','a'], 'ii':['i','ee','y'],
        'uu':['u','oo','oo'], 'ai':['ay','ei','ae'],
        'ck':['k','kk'], 'nn':['n'],
        'nd':['nt','n'], 'mb':['m','nb'],
        'ng':['n','nk'], 'la':['la','laa'],
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
                    perturbed = token_lower.replace(pattern, rep, 1)
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
    def generate(self, token_dict, restrict_to_target_language=False):
        token    = token_dict['text']
        language = token_dict['language']
        if restrict_to_target_language and language == 'en':
            return []
        if len(token) < 3:
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

        # Swap adjacent
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

        return candidates[:8]


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
        # Core variants - expanded
        'pyaar':['pyar','piyar','piyaar','piaar'],
        'pyar':['pyaar','piyar','piyaar'],
        'bhai':['bhaai','bhay','bhaiya','bhayia'],
        'kya':['kyaa','kia','kiya','kyaa'],
        'nahi':['nahin','nai','ni','nahee','nahe'],
        'nahin':['nahi','nai','naheen','nahe'],
        'haan':['han','ha','haa','hanji','haaan'],
        'achha':['acha','accha','achcha','achchha'],
        'acha':['achha','accha','achcha'],
        'bahut':['bohot','bahot','bhaut','bahoot','bohoot'],
        'bohot':['bahut','bahot','bhaut','bohoot'],
        'main':['mein','mai','maii','maiin'],
        'mein':['main','mai','maii'],
        'aaj':['aaz','aj','aaji','aaje'],
        'yaar':['yar','yaaar','yaare','yare'],
        'theek':['thik','teek','theeke','thike'],
        'thik':['theek','teek','thike'],
        'bilkul':['bilcul','bilqul','bilkull','bilkul'],
        'mujhe':['muje','mujhey','mujhe','mujhey'],
        'achhi':['achi','acchi','achchhi','achee'],
        'achi':['achhi','acchi','achee'],
        'phir':['fir','phirr','firr'],
        'fir':['phir','phirr'],
        'woh':['wo','voh','vo','woh','wohi'],
        'wo':['woh','voh','vo'],
        'toh':['to','tou','toh','toe'],
        'to':['toh','tou','toe'],
        'khush':['kush','khus','khushi','khoosh'],
        'udaas':['udas','udaas','udaasi'],
        'maza':['maja','mazaa','maje','mazey'],
        'maja':['maza','mazaa'],
        'sach':['saach','sachi','sachchi'],
        'ghar':['gher','ghur','gharr'],
        'karo':['kro','karro','karro'],
        'jana':['jaana','jaanaa','janaa'],
        'bas':['bus','buss','baas'],
        'bus':['bas','buss'],
        'dost':['doast','dostt','dost'],
        'abhi':['abi','abhii','abhi'],
        'bhi':['bi','bhii','bhee'],
        'zaroor':['zarur','zarour','zaroorr'],
        'zyada':['jyada','zyaada','jyaada','ziyada'],
        'thoda':['thora','thorra','thoraa','thoda'],
        'raat':['raat','raaat','rat','raati'],
        'din':['dinn','deen','dina'],
        'subah':['subha','suba','subaha'],
        'shaam':['sham','shaam','shaami'],
        'pyaara':['pyara','piyara','piyaara'],
        'mohabbat':['mohabbat','muhabbt','mohabbatt'],
        'zindagi':['jindagi','zindagee','jindgi'],
        'jindgi':['zindagi','jindagi','zindagee'],
        'khana':['khaana','khanaa','khaanaa'],
        'khaana':['khana','khanaa'],
        'paani':['pani','paanee','panee'],
        'pani':['paani','paanee'],
        'baat':['baath','baate','baati'],
        'log':['logg','logon','logo'],
        'kaam':['kam','kaaam','kaam'],
        'paise':['paisa','paisey','paise'],
        'dil':['dill','dill','dill'],
        'duniya':['dunia','duniyaa','dunia'],
        'aur':['or','auur','aurr'],
        'lekin':['lekinn','lakin','lekin'],
        'magar':['mger','magar','mgarr'],
        'kyun':['kyoon','kyon','kyunki'],
        'isliye':['islie','isliey','isliye'],
        'matlab':['matalab','matlaab','matlab'],
        'samajh':['samajhh','samajha','samjha'],
        'pata':['pataa','patta','patah'],
        'malum':['maloom','maalum','maaloom'],
        'dekho':['dekho','dekhoo','dekhou'],
        'suno':['sunoo','sunou','sunno'],
        'ruko':['rukoo','rukou','rukko'],
        'chhod':['chhor','chod','chhod'],
        'yaad':['yaadh','yaade','yaadi'],
        'khwaab':['khwab','kwab','khwaab'],
        'sapna':['sapnaa','sapne','sapnee'],
        'dard':['dardd','darde','dardi'],
        'khushi':['khushee','kushi','kushee'],
        'ghussa':['ghusa','ghussaa','ghusaa'],
        'uff':['ufff','uuff','ufh'],
        'arrey':['arey','arre','arree'],
        'oye':['oee','oyee','oi'],
        'wah':['waah','waahh','vah'],
        'shabash':['shabaash','shaabash','shaabaash'],
        'shukriya':['shukria','shukriyaa','shukrya'],
        'maafi':['maafee','mafi','mafee'],
        'bakwaas':['bakbas','bakwaas','bakwaasi'],
        'faltu':['faltoo','faltu','faltuu'],
        'mast':['mastt','masst','maast'],
        'bindaas':['bindas','bindaas','bindaasi'],
        'jugaad':['jugad','jugaadd','jugaadi'],
        'timepass':['time pass','timepaas','taimpass'],
        'scene':['sceen','seene','scen'],
        'setting':['setng','settng','seting'],
    }

    TAMIL_VARIANTS = {
        # Greetings
        'vanakkam':['vanakam','vanagam','vanakkum'],
        'romba':['rombha','rombaa','rumba'],
        'nalla':['nala','nallah','nallaa'],
        'super':['superaa','supera','supper'],
        'semma':['semmaa','sema','semba'],
        'mass':['maas','massaa','maass'],
        'vera':['veraa','verraa','verra'],
        # Pronouns
        'naan':['naaan','naanu','nan'],
        'nee':['neee','neeye','ni'],
        'avan':['avaan','avann','avane'],
        'aval':['avaal','avall','avale'],
        'naanga':['naangaa','naangga','nanga'],
        'neenga':['neengaa','neengga','nenga'],
        'enna':['yenna','ena','ennada'],
        'unna':['unnaa','unnda','una'],
        # Common words
        'machan':['machen','machaan','macha'],
        'seri':['sari','seeri','seriya'],
        'okay':['okayaa','okaa','okey'],
        'aama':['aamaa','aamaaa','aamma'],
        'illai':['ilai','illa','illaiye'],
        'illa':['illai','illaiye','ila'],
        'ippo':['ippoo','ippou','ippa'],
        'epdi':['eppadi','eppdi','epadi'],
        'enge':['engey','engaa','enga'],
        'eppo':['eppoo','eppov','eppa'],
        # Verbs
        'paaru':['paro','paaro','parunga'],
        'iruku':['irukku','irruku','irukuu'],
        'sollu':['solu','solluu','sollunga'],
        'vaa':['vaaa','vaang','vaanga'],
        'po':['poo','pooo','ponga'],
        'podu':['poduu','podunga','podung'],
        'saapu':['saapdu','saapidu','saapitu'],
        'paesu':['pesu','pesuu','pesunga'],
        'kelu':['keluu','kelunga','keling'],
        'pannuven':['pannuvaen','panniduven','pannidren'],
        'seiven':['seikireen','seivaen','seiyaen'],
        'padikiren':['padikireen','padikiraen','padichen'],
        # Adjectives/sentiment
        'nallavan':['nallavaan','nallavaane','nallavane'],
        'kettavan':['kettavaan','kettavaane','kettavane'],
        'azhaga':['azhagaa','azhakaa','azhagana'],
        'mosama':['mosamaa','mosamaaa','mosam'],
        'periya':['periyaa','periyaaa','peria'],
        'sinna':['sinnaa','sinnaaa','china'],
        'mukkiyam':['mukkiyamaa','mukkiam','mukiyam'],
        # Expressions
        'dei':['dey','deii','deyyy'],
        'da':['daa','daaa','dah'],
        'nga':['ngaa','ngaaa','nnga'],
        'poda':['podaa','podaaa','pottaa'],
        'mokka':['mokkaa','mokkah','moka'],
        'kaduppu':['kaduppe','kadupuu','kadupu'],
        'naalaiku':['nalaiku','naalaikku','nalaikku'],
        'inniku':['innikku','innikkuu','inikku'],
        # Sentiment expressions
        'santhosham':['sandhosham','sandosham','santhosam'],
        'dukham':['dukam','dukkam','dhukkam'],
        'kovam':['kovaam','kobam','koopam'],
        'aasai':['asai','aasaai','aasaay'],
        'kadhal':['kaadhal','kaadal','kadal'],
        'anbu':['anbuu','anpuu','anpu'],
        'veruppu':['verupu','verruppuu','verrupu'],
        'nambikai':['nambikkai','nambikay','nambikkai'],
        # Discourse markers
        'theriyum':['theriyumm','theriyuma','teriuma'],
        'puriguthu':['puriyuthu','puriyithu','puriyuthu'],
        'konjam':['konjem','konjum','konjaam'],
        'mudinja':['mudincha','mudinjaa','mudinjaa'],
        'therinja':['therincha','therinjaa','therinjaa'],
        # Social media
        'kalakkal':['kalakkalaa','kalakal','kalakkalaa'],
        'paithiyam':['paithiam','paidhiyam','paithiyamaa'],
        'loosu':['loosuu','loosaa','lusuu'],
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
        # Adjectives
        'achha':['acche','acchi','achha'],
        'accha':['acche','acchi'],
        'bura':['bure','buri','buraa'],
        'bara':['bare','bari','baraa'],
        'bada':['bade','badi','badaa'],
        'chota':['chote','choti','chotaa'],
        'chhota':['chhote','chhoti'],
        'lamba':['lambe','lambi'],
        'ganda':['gande','gandi'],
        'saaf':['saafe','saafi'],
        'purana':['purane','purani'],
        'naya':['naye','nayi','naee'],
        'sundar':['sundare','sundari'],
        'khubsoorat':['khubsoorati'],
        'mazedaar':['mazedaari'],
        'bekar':['bekari'],
        'mushkil':['mushkile','mushkili'],
        'aasaan':['aasaani'],
        # Verbs - infinitive/conjugation
        'khaana':['khaane','khaata','khaati','khaate'],
        'jaana':['jaane','jaata','jaati','jaate'],
        'aana':['aane','aata','aati','aate'],
        'bolta':['bolte','bolti','bolo','bola'],
        'karta':['karte','karti','karo','kiya'],
        'chahiye':['chahie','chahta','chahti','chahte'],
        'chahie':['chahiye','chahta','chahti'],
        'karega':['karegi','karenge','karunga'],
        'karegi':['karega','karenge','karungi'],
        'hoga':['hogi','honge','hua','hui'],
        'hogi':['hoga','honge','hui'],
        'dost':['doston','dosto','dostana'],
        'ghar':['gharon','gharo','gharwale'],
        'baat':['baaton','baato','baatein'],
        'log':['logon','logo','logon'],
        'kaam':['kaamon','kaami','kaamgar'],
        'raat':['raaton','raate','raatein'],
        'din':['dinon','dino','dinbhar'],
        'saal':['saalon','saalo','saalana'],
        'paise':['paison','paisa','paisewala'],
        'zindagi':['zindagiyon','zindagiyo'],
        'duniya':['duniyaon','duniyao'],
        'dil':['dilon','dilo','dilwala'],
        'bhai':['bhaiyon','bhaiyo','bhaijaan'],
        'dost':['doston','yaarana','dostana'],
        'ladka':['ladke','ladkon','ladkaa'],
        'ladki':['ladkiyan','ladkiyo','ladkee'],
        'bachha':['bachche','bachchon','baccha'],
        'khwaab':['khwabon','khwabo','khwaabee'],
        'yaad':['yaadon','yaado','yaadein'],
        'baat':['baatein','baaton','baato'],
        'khushi':['khushiyan','khushiyo','kushiyan'],
        'dard':['dardon','dardo','dardein'],
        'mohabbat':['mohabbatein','mohabbaton'],
        'sapna':['sapne','sapnon','sapno'],
    }

    TAMIL_PARADIGMS = {
        # Verb conjugations
        'irukken':['irukkeaan','irukkean','irukkaan','irukkiren'],
        'paakuren':['paakureen','paakuraan','paakurean','paakiren'],
        'illai':['illa','illaiye','illaiyaa','illena'],
        'illa':['illai','illaiye','illaiyaa','illena'],
        'nalla':['nallavan','nallavana','nalladu','nalladhu'],
        'paaru':['paaren','paarunga','paarttaan','paarthaan'],
        'sollu':['sollunga','solluvaan','solunga','solluvaal'],
        'vaa':['vaanga','varuvan','varuvaan','varuvaal'],
        'po':['ponga','poovaan','poovaal','povaanga'],
        'saapu':['saapidu','saapiduvaan','saapituvaanga','saapiduvaal'],
        'iruku':['irukkaan','irukaal','irukkaanga','irukkiral'],
        'puriguthu':['puriyuthu','puriyala','purinjuchu','puriyuthu'],
        'theriyum':['theriyaadu','therinjuchu','theriyumaa','theriyaadhu'],
        # Additional verb paradigms
        'pannuran':['pannuraan','pannuvaan','pannuvaal','pannuvaanga'],
        'paakiran':['paakiraen','paakiiraan','paakiraanga','paakiraal'],
        'pesiran':['pesiraen','pesiraan','pesiraanga','pesiraal'],
        'seiran':['seiraen','seiraan','seiraanga','seiraal'],
        'varuvan':['varuvaan','varuvaal','varuvaanga','varuvom'],
        'poovan':['poovaan','poovaal','poovaanga','poovom'],
        'solluvaan':['solluvaanga','solluvaal','solluvom','solluvaen'],
        'paarthaan':['paarthaanga','paarthaal','paarthom','paarthaen'],
        # Adjective paradigms
        'nallavan':['nallavaanga','nallavaal','nallavanukku','nallavalukku'],
        'kettavan':['kettavaanga','kettavaal','kettavanukku','kettavalukku'],
        'periyavan':['periyavaanga','periyavaal','periyavanukku','periyavalukku'],
        'sinnavan':['sinnavaanga','sinnavaal','sinnavanukku','sinnavalukku'],
        # Noun paradigms
        'veedu':['veetula','veetukku','veetta','veettukku'],
        'ooru':['oorula','oorukku','oorda','oordukku'],
        'paeru':['paerul','paerukku','paeroda','paerodukku'],
        'kaadhal':['kaadhalula','kaadhalkku','kaadhala','kaadhalukku'],
        'nanbam':['nanbamula','nanbamkku','nanbama','nanbamukku'],
        'kovam':['kovamula','kovamkku','kovama','kovamukku'],
        'santhosham':['santhoshamula','santhoshamkku','santhoshama','santhoshamukku'],
        # Tense variations
        'vandhen':['vandhaen','vandhom','vandhaanga','vandhaal'],
        'ponaen':['ponom','ponaanga','ponaal','ponaan'],
        'paarthenn':['paarthenom','paarthaanga','paarthaal','paarthaan'],
        'sonnean':['sonnom','sonnaanga','sonnaal','sonnaan'],
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


# ─────────────────────────────────────────────
# P6 — CODE SWITCHING GENERATOR
# ─────────────────────────────────────────────

class CodeSwitchingGenerator:
    # English → Hindi substitution dictionary
    EN_TO_HI = {
        # Sentiment words
        'good':'achha', 'bad':'bura', 'great':'zabardast',
        'nice':'badhiya', 'awesome':'kamaal', 'amazing':'shandar',
        'wonderful':'adbhut', 'terrible':'bekar', 'horrible':'ganda',
        'happy':'khush', 'sad':'udaas', 'angry':'gusse',
        'love':'pyaar', 'hate':'nafrat', 'like':'pasand',
        'beautiful':'sundar', 'ugly':'badsurat',
        'fast':'tez', 'slow':'dheema', 'big':'bada', 'small':'chota',
        'easy':'aasaan', 'difficult':'mushkil', 'hard':'mushkil',
        'true':'sach', 'false':'jhooth', 'right':'sahi', 'wrong':'galat',
        'new':'naya', 'old':'purana', 'clean':'saaf', 'dirty':'ganda',
        'hot':'garam', 'cold':'thanda', 'sweet':'meetha', 'bitter':'kadwa',
        'strong':'mazboot', 'weak':'kamzor', 'rich':'ameer', 'poor':'gareeb',
        'smart':'samajhdaar', 'stupid':'bewakoof', 'funny':'mazedaar',
        'boring':'boring', 'interesting':'dilchasp', 'important':'zaruri',
        'serious':'gambhir', 'busy':'vyast', 'free':'azaad',
        # Common words
        'friend':'dost', 'enemy':'dushman', 'brother':'bhai',
        'sister':'behen', 'mother':'maa', 'father':'baap',
        'home':'ghar', 'school':'pathshala', 'work':'kaam',
        'money':'paise', 'food':'khana', 'water':'paani',
        'time':'waqt', 'day':'din', 'night':'raat',
        'morning':'subah', 'evening':'shaam',
        'heart':'dil', 'mind':'dimaag', 'life':'zindagi',
        'world':'duniya', 'country':'desh', 'city':'shahar',
        'people':'log', 'person':'insaan', 'man':'aadmi', 'woman':'aurat',
        'boy':'ladka', 'girl':'ladki', 'child':'bachha',
        'dream':'sapna', 'hope':'umeed', 'fear':'darr',
        'pain':'dard', 'joy':'khushi', 'peace':'sukoon',
        'problem':'samasya', 'solution':'hal', 'help':'madad',
        'truth':'sach', 'lie':'jhooth', 'trust':'bharosa',
        'feeling':'ehsaas', 'thought':'soch', 'idea':'khayal',
        # Verbs
        'go':'jao', 'come':'aao', 'see':'dekho', 'look':'dekho',
        'hear':'suno', 'listen':'suno', 'speak':'bolo', 'say':'kaho',
        'think':'socho', 'know':'jano', 'understand':'samjho',
        'want':'chahte', 'need':'chahiye', 'like':'pasand',
        'eat':'khao', 'drink':'peeyo', 'sleep':'soo', 'wake':'utho',
        'sit':'baitho', 'stand':'khade', 'run':'daudho', 'walk':'chalo',
        'stop':'ruko', 'wait':'intezaar', 'leave':'jao', 'stay':'raho',
        'remember':'yaad', 'forget':'bhool', 'learn':'seekho',
        'play':'khelo', 'laugh':'haso', 'cry':'roo',
        # Expressions
        'yes':'haan', 'no':'nahi', 'okay':'theek', 'fine':'theek',
        'sorry':'maafi', 'thanks':'shukriya', 'please':'please',
        'again':'phir', 'always':'hamesha', 'never':'kabhi',
        'now':'abhi', 'later':'baad', 'today':'aaj', 'tomorrow':'kal',
        'here':'yahan', 'there':'wahan', 'where':'kahan',
        'why':'kyun', 'how':'kaise', 'what':'kya', 'who':'kaun',
        'very':'bahut', 'too':'bhi', 'also':'bhi', 'only':'sirf',
        'just':'bas', 'really':'sach', 'actually':'actually',
        'maybe':'shayad', 'definitely':'zaroor', 'probably':'shayad',
    }

    # Hindi → English substitution dictionary (reverse)
    HI_TO_EN = {v: k for k, v in EN_TO_HI.items()}

    def __init__(self, language_pair='hi-en'):
        self.language_pair = language_pair
        self.en_to_hi = self.EN_TO_HI
        self.hi_to_en = self.HI_TO_EN

    def generate(self, token_dict):
        token    = token_dict['text']
        language = token_dict['language']
        token_lower = token.lower()
        candidates = []

        # English token → substitute with Hindi
        if language == 'en' and token_lower in self.en_to_hi:
            hindi_word = self.en_to_hi[token_lower]
            candidates.append({
                'perturbed_text':    hindi_word,
                'perturbation_type': 'P6',
                'rule_applied':      f'en_to_hi:{token_lower}→{hindi_word}',
                'original_token':    token
            })

        # Hindi token → substitute with English
        elif language == 'hi' and token_lower in self.hi_to_en:
            eng_word = self.hi_to_en[token_lower]
            candidates.append({
                'perturbed_text':    eng_word,
                'perturbation_type': 'P6',
                'rule_applied':      f'hi_to_en:{token_lower}→{eng_word}',
                'original_token':    token
            })

        return candidates


class PerturbationEngine:
    def __init__(self, language_pair='hi-en',
                 active_types=None,
                 restrict_to_target_language=False):
        # Default includes P7
        self.active_types = active_types or [
            'P1','P2','P3','P4','P5','P6','P7','PC'
        ]
        self.restrict_to_target_language = restrict_to_target_language
        self.generators = {
            'P1': PhoneticGenerator(language_pair),
            'P2': CharacterGenerator(),
            'P3': CuratedScriptConverter(),
            'P4': TransliterationVariantGenerator(language_pair),
            'P5': MorphologicalGenerator(language_pair),
            'P6': CodeSwitchingGenerator(language_pair),
            'P7': HindiSynonymGenerator(language_pair),
            'PC': CombinedPerturbationEngine(language_pair),
        }

    def generate_candidates(self, token_dict):
        all_candidates = []
        for ptype in self.active_types:
            if ptype == 'P2' and 'P2' in self.generators:
                all_candidates.extend(
                    self.generators['P2'].generate(
                        token_dict,
                        restrict_to_target_language=self.restrict_to_target_language
                    )
                )
                continue
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
    ('true','false'), ('right','wrong'), ('fast','slow'),
    ('easy','difficult'), ('strong','weak'), ('rich','poor'),
    ('smart','stupid'), ('clean','dirty'), ('hot','cold'),
    ('new','old'), ('big','small'), ('sweet','bitter'),
]:
    SENTIMENT_ANTONYMS.add((w1, w2))
    SENTIMENT_ANTONYMS.add((w2, w1))


class ValidationModule:
    def __init__(self, labse_model, predict_fn,
                 threshold=0.75, task='sentiment'):
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
                c['labse_similarity']   = result.get('similarity')
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
               budget=0.40):
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
                current_tokens[position] = best_cand['perturbed_text']
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
            'p4_used': 0, 'p5_used': 0, 'p6_used': 0, 'p7_used': 0, 'pc_used': 0,
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
            pt = p['perturbation_type'].lower()
            key = 'pc_used' if pt in ['pc','p67','p72','p42','p12'] else f'{pt}_used'
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
                'p6_used':    s['p6_used'],
                'p7_used':    s['p7_used'],
                'pc_used':    s['pc_used'],
            }
        }
        path = self.corpus_path.replace(
            '_corpus.jsonl', '_summary.json'
        )
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Summary saved: {path}")
        return summary

print("✓ modules_v2 loaded successfully — P1-P6 active")


# ─────────────────────────────────────────────
# P7 — HINDI SYNONYM SUBSTITUTION
# ─────────────────────────────────────────────

class HindiSynonymGenerator:
    HINDI_SYNONYMS = {
        # Positive sentiment words
        'achha':  ['badhiya','umda','shandar','laajawab','behtareen'],
        'accha':  ['badhiya','umda','shandar','laajawab','behtareen'],
        'achhi':  ['badhiya','umda','shandar','laajawab','behtareen'],
        'acchi':  ['badhiya','umda','shandar','laajawab'],
        'acha':   ['badhiya','umda','shandar','behtareen'],
        'badhiya':['achha','umda','shandar','laajawab'],
        'shandar':['achha','badhiya','umda','laajawab'],
        'zabardast':['shandar','badhiya','kamaal','laajawab'],
        'kamaal': ['shandar','badhiya','zabardast','umda'],
        'mast':   ['badhiya','shandar','zabardast','kamaal'],
        'behtareen':['achha','badhiya','umda','shandar'],
        'umda':   ['achha','badhiya','shandar','behtareen'],
        'laajawab':['shandar','badhiya','zabardast','kamaal'],
        'adbhut': ['kamaal','shandar','zabardast','hairaan'],
        # Negative sentiment words
        'bura':   ['kharab','ghatiya','bekar','wahiyat','forma'],
        'bure':   ['kharab','ghatiya','bekar','wahiyat'],
        'buri':   ['kharab','ghatiya','bekar','wahiyat'],
        'kharab': ['bura','ghatiya','bekar','wahiyat'],
        'ghatiya':['bura','kharab','bekar','wahiyat'],
        'bekar':  ['bura','kharab','ghatiya','faltu'],
        'wahiyat':['bura','kharab','ghatiya','bekar'],
        'faltu':  ['bekar','wahiyat','ghatiya','fijool'],
        'ganda':  ['kharab','bekar','ghatiya','wahiyat'],
        'bakwaas':['bekar','faltu','wahiyat','ghatiya'],
        # Emotions
        'khush':  ['prasann','anand','magan','harshit','fullaaye'],
        'prasann':['khush','anand','magan','harshit'],
        'anand':  ['khush','prasann','magan','khushi'],
        'udaas':  ['dukhi','ghamgin','pareshaan','majboor'],
        'dukhi':  ['udaas','ghamgin','pareshaan','bezaar'],
        'ghamgin':['udaas','dukhi','pareshaan','bezaar'],
        'pareshaan':['udaas','dukhi','ghamgin','bezaar'],
        'gussa':  ['naraaz','krodh','gazab','taish'],
        'naraaz': ['gussa','krodh','bekhush','nakhush'],
        'darr':   ['bhay','khauf','dread','sahm'],
        'khauf':  ['darr','bhay','dread','sahm'],
        # Love/hate
        'pyaar':  ['mohabbat','ishq','prem','ulfat','chahat'],
        'mohabbat':['pyaar','ishq','prem','ulfat'],
        'ishq':   ['pyaar','mohabbat','prem','chahat'],
        'prem':   ['pyaar','mohabbat','ishq','ulfat'],
        'nafrat': ['ghrina','dwesh','bair','naphrat'],
        'ghrina': ['nafrat','dwesh','bair'],
        'pasand': ['achha','pasandida','manzoor','priya'],
        'napasand':['nafrat','ghrina','bekar','na-pasand'],
        # People/relations
        'dost':   ['yaar','mitra','saathi','humsafar','rafiq'],
        'yaar':   ['dost','mitra','saathi','bhai','rafiq'],
        'mitra':  ['dost','yaar','saathi','humsafar'],
        'bhai':   ['yaar','dost','bhaijaan','sahab'],
        'dushman':['dushmaan','virodhee','bair','shatru'],
        # Common nouns
        'ghar':   ['makaan','aashiyana','niwaas','dar','grih'],
        'makaan': ['ghar','aashiyana','niwaas','dar'],
        'zindagi':['hayaat','jeevan','umr','aayush'],
        'jindgi': ['hayaat','jeevan','umr','zindagi'],
        'jeevan': ['zindagi','hayaat','umr','aayush'],
        'duniya': ['jahan','aalam','kainaat','jagat'],
        'desh':   ['mulk','watan','qaum','rashtra'],
        'mulk':   ['desh','watan','qaum','rashtra'],
        'log':    ['insaan','aadam','banda','janta','awaam'],
        'insaan': ['log','aadam','banda','insan'],
        'aadmi':  ['mard','purush','banda','insaan'],
        'aurat':  ['mahila','stri','khatoon','begum'],
        'bachha': ['bacha','child','ladka','larka'],
        # Actions/verbs (base forms)
        'kaam':   ['karya','kaarj','mehnat','kaarobaar'],
        'karya':  ['kaam','kaarj','mehnat'],
        'padhai': ['taleem','vidya','shiksha','seekh'],
        'taleem': ['padhai','vidya','shiksha'],
        'khana':  ['bhojan','khaana','aahar','rizq'],
        'khaana': ['bhojan','khana','aahar','rizq'],
        'bhojan': ['khana','khaana','aahar'],
        'paani':  ['jal','aab','neer','pani'],
        'pani':   ['jal','aab','neer','paani'],
        # Truth/lies
        'sach':   ['satya','haqeeqat','sachai','vastavik'],
        'satya':  ['sach','haqeeqat','sachai'],
        'jhooth': ['asatya','fareb','dhoka','miथ्या'],
        'fareb':  ['jhooth','dhoka','makr','chal'],
        'dhoka':  ['fareb','jhooth','makr','farebi'],
        # Qualities
        'mushkil':['kathin','kasta','dushwar','aaukha'],
        'kathin': ['mushkil','kasta','dushwar'],
        'aasaan': ['sahel','saral','aasaani','suhana'],
        'sahel':  ['aasaan','saral','suhana'],
        'mazboot':['taaqatwar','shakti','bali','qawi'],
        'kamzor': ['nirabal','durbaal','halka','bedam'],
        'ameer':  ['daulatmand','maaldar','waali','sahib'],
        'gareeb': ['muflis','faqeer','nirgat','bekas'],
        'tez':    ['jaldi','fast','chaalaak','hoshiyaar'],
        'dheema': ['slow','aahista','dhire','manda'],
        # Size
        'bada':   ['bara','vishaal','azeem','khabeer'],
        'bara':   ['bada','vishaal','azeem','bhari'],
        'chota':  ['chhota','nano','laghu','sukshm'],
        'chhota': ['chota','nano','laghu','sukshm'],
        # Time
        'jaldi':  ['shigraf','jald','tezi','speed'],
        'abhi':   ['iss waqt','filhaal','haal','tatkal'],
        'hamesha':['sada','nityam','sadaiv','har waqt'],
        'kabhi':  ['kisi waqt','kisi din','kadaachit'],
        # Intensifiers
        'bahut':  ['kaafi','zyada','atyant','ghani','nihayat'],
        'kaafi':  ['bahut','zyada','atyant','paryapt'],
        'zyada':  ['adhik','bahut','kaafi','atyant'],
        'thoda':  ['kam','thori','kuch','alp'],
        'bilkul': ['poori tarah','hargiz','qatai','mutlaq'],
        # Expressions
        'sahi':   ['theek','durust','sahih','vaajib'],
        'galat':  ['ghalat','asahi','bhool','adurust'],
        'theek':  ['sahi','durust','thik','okay'],
        'pakka':  ['zaroor','nischit','tay','qatai'],
        'shayad': ['maybe','sambhav','mumkin','lagta'],
        'zaroor': ['avashya','zaroorat','pakka','albatta'],
        # Hate speech related
        'nalla':  ['bekar','faltu','ghatiya','wahiyat'],
        'bewakoof':['murkh','jahil','gadha','nalayak'],
        'murkh':  ['bewakoof','jahil','gadha','nalayak'],
        'nalayak':['bewakoof','bekar','nikamma','faltu'],
        'nikamma':['bekar','faltu','nalayak','nikamma'],
        'kamina': ['neech','kameena','zaleel','razil'],
        'kameena':['kamina','neech','zaleel','razil'],
        'neech':  ['kamina','zaleel','razil','adna'],
        'zaleel': ['neech','kamina','razil','adna'],
        'gaddaar':['bewafaa','khayin','namak-haraam'],
        'bewafaa':['gaddaar','khayin','bure','dhokhebaaz'],
    }

    def __init__(self, language_pair='hi-en'):
        self.language_pair = language_pair

    def generate(self, token_dict):
        token    = token_dict['text']
        language = token_dict['language']
        if language == 'en':
            return []
        token_lower = token.lower()
        if token_lower not in self.HINDI_SYNONYMS:
            return []
        return [{
            'perturbed_text':    syn,
            'perturbation_type': 'P7',
            'rule_applied':      f'hindi_synonym:{token_lower}→{syn}',
            'original_token':    token
        } for syn in self.HINDI_SYNONYMS[token_lower]
          if syn != token_lower]


# ─────────────────────────────────────────────
# COMBINED PERTURBATION GENERATOR
# ─────────────────────────────────────────────

class CombinedPerturbationEngine:
    """
    Applies two perturbations on the same token sequentially.
    Example: P6(good→achha) then P7(achha→badhiya)
             P4(pyaar→pyar) then P2(pyar→pyarr)
    """
    def __init__(self, language_pair='hi-en'):
        self.language_pair = language_pair
        self.p1 = PhoneticGenerator(language_pair)
        self.p2 = CharacterGenerator()
        self.p4 = TransliterationVariantGenerator(language_pair)
        self.p6 = CodeSwitchingGenerator(language_pair)
        self.p7 = HindiSynonymGenerator(language_pair)
        self.lang_det = LanguageDetector()
        self.script_det = ScriptDetector()

    def generate(self, token_dict):
        combined = []

        # Combo 1: P6 then P7 — code-switch then synonym
        # good → achha → badhiya
        p6_results = self.p6.generate(token_dict)
        for p6_cand in p6_results:
            new_token = p6_cand['perturbed_text']
            script    = self.script_det.detect(new_token)
            lang      = self.lang_det.detect(new_token, script, self.language_pair)
            new_dict  = {'text': new_token, 'language': lang, 'script': script}
            p7_results = self.p7.generate(new_dict)
            for p7_cand in p7_results[:2]:
                combined.append({
                    'perturbed_text':    p7_cand['perturbed_text'],
                    'perturbation_type': 'P67',
                    'rule_applied':      f"P6+P7:{token_dict['text']}→{new_token}→{p7_cand['perturbed_text']}",
                    'original_token':    token_dict['text']
                })

        # Combo 2: P7 then P2 — synonym then character variation
        # achha → badhiya → badhiyaa
        p7_results = self.p7.generate(token_dict)
        for p7_cand in p7_results[:3]:
            new_token = p7_cand['perturbed_text']
            script    = self.script_det.detect(new_token)
            lang      = self.lang_det.detect(new_token, script, self.language_pair)
            new_dict  = {'text': new_token, 'language': lang, 'script': script}
            p2_results = self.p2.generate(new_dict)
            for p2_cand in p2_results[:2]:
                combined.append({
                    'perturbed_text':    p2_cand['perturbed_text'],
                    'perturbation_type': 'P72',
                    'rule_applied':      f"P7+P2:{token_dict['text']}→{new_token}→{p2_cand['perturbed_text']}",
                    'original_token':    token_dict['text']
                })

        # Combo 3: P4 then P2 — transliteration then character
        # pyaar → pyar → pyarr
        p4_results = self.p4.generate(token_dict)
        for p4_cand in p4_results[:2]:
            new_token = p4_cand['perturbed_text']
            script    = self.script_det.detect(new_token)
            lang      = self.lang_det.detect(new_token, script, self.language_pair)
            new_dict  = {'text': new_token, 'language': lang, 'script': script}
            p2_results = self.p2.generate(new_dict)
            for p2_cand in p2_results[:2]:
                combined.append({
                    'perturbed_text':    p2_cand['perturbed_text'],
                    'perturbation_type': 'P42',
                    'rule_applied':      f"P4+P2:{token_dict['text']}→{new_token}→{p2_cand['perturbed_text']}",
                    'original_token':    token_dict['text']
                })

        # Combo 4: P1 then P2 — phonetic then character
        # bhai → bhay → bhaay
        p1_results = self.p1.generate(token_dict)
        for p1_cand in p1_results[:2]:
            new_token = p1_cand['perturbed_text']
            script    = self.script_det.detect(new_token)
            lang      = self.lang_det.detect(new_token, script, self.language_pair)
            new_dict  = {'text': new_token, 'language': lang, 'script': script}
            p2_results = self.p2.generate(new_dict)
            for p2_cand in p2_results[:2]:
                combined.append({
                    'perturbed_text':    p2_cand['perturbed_text'],
                    'perturbation_type': 'P12',
                    'rule_applied':      f"P1+P2:{token_dict['text']}→{new_token}→{p2_cand['perturbed_text']}",
                    'original_token':    token_dict['text']
                })

        return combined


# ─────────────────────────────────────────────
# P3 FIXED — CURATED CROSS-SCRIPT GENERATOR
# ─────────────────────────────────────────────

class CuratedScriptConverter:
    """
    Manual curated Roman→Devanagari dictionary.
    Only verified correct conversions included.
    Replaces automatic ITRANS conversion.
    """
    ROMAN_TO_DEVANAGARI = {
        # Common Hindi words
        'hai':   'है',
        'hain':  'हैं',
        'tha':   'था',
        'thi':   'थी',
        'the':   'थे',
        'hoga':  'होगा',
        'hogi':  'होगी',
        'tum':   'तुम',
        'main':  'मैं',
        'mein':  'मैं',
        'aap':   'आप',
        'woh':   'वो',
        'wo':    'वो',
        'hum':   'हम',
        'yeh':   'यह',
        'ye':    'ये',
        'kya':   'क्या',
        'nahi':  'नहीं',
        'nahin': 'नहीं',
        'haan':  'हाँ',
        'han':   'हाँ',
        'bhai':  'भाई',
        'yaar':  'यार',
        'dost':  'दोस्त',
        'pyaar': 'प्यार',
        'pyar':  'प्यार',
        'dil':   'दिल',
        'ghar':  'घर',
        'log':   'लोग',
        'aaj':   'आज',
        'kal':   'कल',
        'subah': 'सुबह',
        'raat':  'रात',
        'din':   'दिन',
        'khush': 'खुश',
        'udaas': 'उदास',
        'achha': 'अच्छा',
        'accha': 'अच्छा',
        'bura':  'बुरा',
        'bahut': 'बहुत',
        'bohot': 'बहुत',
        'thoda': 'थोड़ा',
        'zyada': 'ज़्यादा',
        'sach':  'सच',
        'jhooth':'झूठ',
        'desh':  'देश',
        'zindagi':'ज़िंदगी',
        'mohabbat':'मोहब्बत',
        'nafrat':'नफ़रत',
        'insaan':'इंसान',
        'duniya':'दुनिया',
        'waqt':  'वक्त',
        'paise': 'पैसे',
        'kaam':  'काम',
        'naam':  'नाम',
        'baat':  'बात',
        'dard':  'दर्द',
        'khushi':'खुशी',
        'ghussa':'गुस्सा',
        'sapna': 'सपना',
        'yaad':  'याद',
        'umeed': 'उम्मीद',
        'bharosa':'भरोसा',
        'izzat': 'इज़्ज़त',
        'kismat':'किस्मत',
        'takdir':'तक़दीर',
        'jannat':'जन्नत',
        'dua':   'दुआ',
        'khuda': 'ख़ुदा',
        'Allah': 'अल्लाह',
        'maa':   'माँ',
        'baap':  'बाप',
        'behen': 'बहन',
        'ladka': 'लड़का',
        'ladki': 'लड़की',
        'bachha':'बच्चा',
        'dosti': 'दोस्ती',
        'rishta':'रिश्ता',
        'shaadi':'शादी',
        'khana': 'खाना',
        'paani': 'पानी',
        'chai':  'चाय',
        'film':  'फ़िल्म',
        'gaana': 'गाना',
        'khel':  'खेल',
        'school':'स्कूल',
        'padhai':'पढ़ाई',
        'naukri':'नौकरी',
        'sarkar':'सरकार',
        'desh':  'देश',
        'mulk':  'मुल्क',
        'awaam': 'अवाम',
        'neta':  'नेता',
        'sarkaar':'सरकार',
        'police':'पुलिस',
        'court': 'कोर्ट',
        'media': 'मीडिया',
    }

    def generate(self, token_dict):
        token    = token_dict['text']
        language = token_dict['language']
        script   = token_dict['script']

        if script != 'ROMAN':
            return []
        if language == 'en':
            return []
        if len(token) < 2:
            return []

        token_lower = token.lower()
        if token_lower not in self.ROMAN_TO_DEVANAGARI:
            return []

        converted = self.ROMAN_TO_DEVANAGARI[token_lower]
        return [{
            'perturbed_text':    converted,
            'perturbation_type': 'P3',
            'rule_applied':      f'roman_to_devanagari:{token_lower}→{converted}',
            'original_token':    token
        }]
