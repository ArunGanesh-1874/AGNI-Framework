"""Shared map: which results folder / experiment id feeds each paper number."""
MODELS = ['xlmr', 'muril', 'mbert']
LANGS = {
    'hinglish': {
        'tasks': ['sentiment', 'hate'],
        'methods': {
            'AGNI-Full':   ('comixattack_v4c_full',        'comixattack_{m}_{t}'),
            'AGNI-Strict': ('comixattack_v3_full',         'comixattack_{m}_{t}'),
            'DeepWordBug': ('deepwordbug_baseline',        'deepwordbug_{m}_{t}'),
            'TextFooler':  ('textfooler_baseline',         'textfooler_{m}_{t}'),
            'Random':      ('random_baseline',             'random_{m}_{t}'),
            'AdvCodeMix':  ('advcodemix_labse_baseline',   'advcodemix_{m}_{t}_hi'),
        },
    },
    'tamil': {
        'tasks': ['sentiment', 'offensive'],
        'methods': {
            'AGNI-Full':   ('comixattack_tamil_v2_full',        'comixattack_{m}_{t}'),
            'AGNI-Strict': ('comixattack_tamil_strict_v2_full', 'comixattack_{m}_{t}'),
            'DeepWordBug': ('deepwordbug_tamil_baseline',       'deepwordbug_{m}_{t}'),
            'TextFooler':  ('textfooler_tamil_baseline',        'textfooler_{m}_{t}'),
            'Random':      ('random_tamil_baseline',            'random_{m}_{t}'),
            'AdvCodeMix':  ('advcodemix_labse_baseline',        'advcodemix_{m}_{t}_ta'),
        },
    },
}
def experiments():
    for lang, cfg in LANGS.items():
        for method, (folder, pattern) in cfg['methods'].items():
            for t in cfg['tasks']:
                for m in MODELS:
                    yield lang, method, t, m, folder, pattern.format(m=m, t=t)
