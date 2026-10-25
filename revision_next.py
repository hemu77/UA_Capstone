"""Prepare and exercise revised prompts without an API client or paid command.

V5 remains immutable. These fixtures test software contracts, not model behavior.
The translated wording below is an AI-authored candidate awaiting human review.
"""
import argparse
import contextlib
import copy
import hashlib
import io
import itertools
import json
import random
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import networkx as nx
import numpy as np

import generate_networks as engine
import revision224 as frozen
import revision224_prompts as previous

ROOT = Path(__file__).resolve().parent
DESTINATION = ROOT / 'outputs/offline_revision_v6'
TEXT = copy.deepcopy(previous.TEXT)
COUNTRIES = copy.deepcopy(previous.COUNTRIES)
COUNTRIES['portuguese'] = dict(us='os Estados Unidos', india='a Índia', japan='o Japão', brazil='o Brasil')
TEXT['hindi']['context'] = 'सामाजिक परिवेश {country} है। ये लोग काल्पनिक वयस्क हैं। उनके गुण अपरिवर्तित रखें। उनकी बोली जाने वाली भाषा निर्दिष्ट नहीं है। सभी स्थितियों में उम्मीदवारों के डेटा के फ़ील्ड नाम और मान अंग्रेज़ी में ही रहते हैं और बदले नहीं जाते। '
SINGULAR = {
    'english': 'Choose exactly one friend from the eligible candidates. Output only one integer ID. No self-links, explanation or other text.',
    'portuguese': 'Escolha exatamente um amigo entre os candidatos elegíveis. Retorne apenas um ID inteiro. Não inclua ligações consigo mesmo, explicações ou outro texto.',
    'hindi': 'पात्र उम्मीदवारों में से ठीक एक मित्र चुनें। केवल एक पूर्णांक ID दें। स्वयं से संबंध, स्पष्टीकरण या अन्य पाठ न दें।',
    'japanese': '選択可能な候補者から友人をちょうど1人選んでください。整数IDを1つだけ返してください。自分自身への関係、説明、その他の文章は含めないでください。',
}
INTRO = {
    'english': ('You are persona {actor}. ', 'You are persona {actor}, joining a network. Candidate degree gives their current number of friends. '),
    'portuguese': ('Você é a pessoa {actor}. ', 'Você é a pessoa {actor}, entrando em uma rede. O campo degree do candidato informa seu número atual de amigos. '),
    'hindi': ('आप व्यक्ति {actor} हैं। ', 'आप व्यक्ति {actor} हैं और एक नेटवर्क में शामिल हो रहे हैं। उम्मीदवार की degree उसके वर्तमान मित्रों की संख्या है। '),
    'japanese': ('あなたは人物{actor}です。', 'あなたはネットワークに参加する人物{actor}です。候補者のdegreeは現在の友人数です。'),
}
TEXT['japanese']['local'] = 'あなたは人物{actor}です。候補者から重複しないように友人をちょうど{count}人選んでください。整数IDだけをコンマで区切って返してください。自分自身への関係、説明、その他の文章は含めないでください。'
TEXT['japanese']['sequential'] = 'あなたはネットワークに参加する人物{actor}です。候補者のdegreeは現在の友人数です。候補者から重複しないように友人をちょうど{count}人選んでください。整数IDだけをコンマで区切って返してください。自分自身への関係、説明、その他の文章は含めないでください。'
GLOBAL = {
    'english': 'Choose all friendship pairs that together specify the entire network among the listed people, not a one-to-one pairing. Each person may have zero, one or multiple friends. There is no required number of friendships; disconnected components are allowed. Friendship has no direction: list each pair only once. Output one pair of integer IDs per line as ID1, ID2, with ID1 < ID2. If the whole network has no friendships, output only NONE. No self-links, duplicate pairs, explanation or other text.',
    'portuguese': 'Escolha todos os pares de amizade que, juntos, definem a rede inteira entre as pessoas listadas, não um emparelhamento um a um. Cada pessoa pode ter zero, um ou vários amigos. Não há uma quantidade obrigatória de amizades; são permitidos componentes desconectados. A amizade não tem direção: liste cada par apenas uma vez. Escreva um par de IDs inteiros por linha como ID1, ID2, com ID1 < ID2. Se a rede inteira não tiver amizades, retorne apenas NONE. Não inclua ligações consigo mesmo, pares duplicados, explicações ou outro texto.',
    'hindi': 'सूचीबद्ध लोगों के पूरे नेटवर्क को परिभाषित करने वाली सभी मित्रता जोड़ियाँ चुनें, न कि केवल एक-से-एक जोड़ियाँ। प्रत्येक व्यक्ति के शून्य, एक या कई मित्र हो सकते हैं। मित्रताओं की कोई निर्धारित संख्या नहीं है; अलग-अलग असंबद्ध समूह स्वीकार्य हैं। मित्रता में कोई दिशा नहीं होती: प्रत्येक जोड़ी केवल एक बार दें। हर पंक्ति में पूर्णांक ID की एक जोड़ी ID1, ID2 के रूप में दें, जहाँ ID1 < ID2 हो। यदि पूरे नेटवर्क में कोई मित्रता नहीं है, तो केवल NONE दें। स्वयं से संबंध, दोहराई गई जोड़ियाँ, स्पष्टीकरण या अन्य पाठ न दें।',
    'japanese': '一覧の人物間のネットワーク全体を定義するすべての友人ペアを選んでください。1対1の組み合わせだけを作る課題ではありません。各人物の友人数は0人、1人、複数人のいずれでも構いません。友人関係の総数は指定しません。互いにつながっていないグループも許されます。友人関係に方向はありません。各ペアは1回だけ記載してください。各行に整数IDのペアをID1, ID2として記載し、ID1 < ID2を守ってください。ネットワーク全体に友人関係がない場合はNONEだけを返してください。自分自身への関係、重複するペア、説明、その他の文章は含めないでください。',
}


def config():
    value = json.loads((ROOT / 'study_protocol_next.json').read_text(encoding='utf-8'))
    if value['generation_authorized'] is not False or value['status'] != 'OFFLINE_ONLY_NOT_FROZEN_FOR_COLLECTION':
        raise ValueError('This preparation tool cannot grant generation authorization.')
    roster = json.loads((ROOT / value['persona_file']).read_text(encoding='utf-8'))
    if roster != previous.adult_roster() or value['expected_personas'] != len(roster):
        raise ValueError('Declared roster differs from the offline candidate implementation.')
    return value


def stream_seed(seed, purpose):
    if type(seed) is not int or seed < 0:
        raise ValueError('Seed must be a nonnegative integer, not population size.')
    return int.from_bytes(hashlib.sha256(f'v6:{seed}:{purpose}'.encode()).digest()[:4], 'big')


def schedule(seed):
    ids = list(previous.adult_roster())
    display = ids.copy()
    random.Random(stream_seed(seed, 'display')).shuffle(display)
    actors = np.random.RandomState(stream_seed(seed, 'actors')).choice(ids, len(ids), replace=False).tolist()
    rng = np.random.RandomState(stream_seed(seed, 'counts'))
    quotas = {actor: int(min(max(rng.exponential(5), 1), 20, len(ids)-1)) for actor in actors}
    return dict(seed=seed, display_order=display, actor_order=actors, nomination_quotas=quotas)


def system_prompt(method, language, country, actor=None, count=None):
    # Reuse the strict V5 argument validation without mutating its text catalog.
    previous.system_prompt(method, previous.adult_roster(), previous.DEMOS,
                           curr_pid=actor, num_choices=count,
                           culture_context=country, prompt_language=language)
    context = TEXT[language]['context'].format(country=COUNTRIES[language][country])
    if method == 'global':
        return context + GLOBAL[language]
    if method in {'local', 'sequential'} and count == 1:
        return context + INTRO[language][method == 'sequential'].format(actor=actor) + SINGULAR[language]
    return context + TEXT[language][method].format(actor=actor, count=count)


def candidate_payload(method, language, actor, graph, display_order):
    roster = previous.adult_roster()
    if language not in TEXT or method not in {'global', 'local', 'sequential', 'iterative-add', 'iterative-drop'}:
        raise ValueError('Unsupported candidate condition.')
    if len(display_order) != len(roster) or set(display_order) != set(roster):
        raise ValueError('Display order must be a complete roster permutation.')
    if set(graph) != set(roster) or graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph):
        raise ValueError('Graph must use the exact undirected roster.')
    if method != 'global' and actor not in roster:
        raise ValueError('Missing actor.')
    friends = set(graph[actor]) if actor is not None else set()
    ids = [pid for pid in display_order if method == 'global' or pid != actor]
    if method.startswith('iterative'):
        ids = [pid for pid in ids if (pid in friends) == (method == 'iterative-drop')]
    fields = ['id', *previous.DEMOS]
    if method not in {'global', 'local'}:
        fields.append('degree')
    if method.startswith('iterative'):
        fields.append('mutual')
    rows = []
    for pid in ids:
        row = [pid, *[roster[pid][key] for key in previous.DEMOS]]
        if 'degree' in fields:
            row.append(graph.degree(pid))
        if 'mutual' in fields:
            row.append(len(set(graph[pid]) & friends))
        rows.append(row)
    return json.dumps(dict(fields=fields, actor_fields=['id', *previous.DEMOS],
                           actor=None if actor is None else [actor, *[roster[actor][d] for d in previous.DEMOS]],
                           candidates=rows), ensure_ascii=False, separators=(',', ':'))


def parse_response(method, response, G, **kwargs):
    if method == 'global' and isinstance(response, str) and response.strip() == 'NONE':
        if G.number_of_edges() or kwargs.get('required_global_edges'):
            raise ValueError('NONE cannot discard an existing or required friendship set.')
        return G
    return engine.update_graph_from_response(method, response, G, **kwargs)


def retry_prompt(method, language, country, actor=None, count=None, preserving=False):
    prefix = {'english': 'Invalid response. ', 'portuguese': 'Resposta inválida. ',
              'hindi': 'अमान्य उत्तर। ', 'japanese': '無効な回答です。'}[language]
    repair = {'english': 'Return exactly the original valid friendship set, with no missing or new friendships. ',
              'portuguese': 'Retorne exatamente o conjunto original de amizades válidas, sem omitir nem adicionar amizades. ',
              'hindi': 'मूल वैध मित्रताओं का वही पूरा समूह लौटाएँ; कोई मित्रता न छोड़ें और न जोड़ें। ',
              'japanese': '元の有効な友人ペアをすべてそのまま返してください。削除や追加はしないでください。'}
    return prefix + (repair[language] if preserving else '') + system_prompt(method, language, country, actor, count)


def dry_run(method, country, language, seed, reply=None):
    """Exercise the real graph engine using local fixture replies, never an API.

    Module overrides follow the existing runner pattern; this function is serial,
    not thread-safe. Separate RNG instances prevent display changes altering quotas.
    """
    if method not in config()['methods']:
        raise ValueError('Unknown method.')
    roster, plan, events, requests = previous.adult_roster(), schedule(seed), [], []
    actors = np.random.RandomState(stream_seed(seed, 'actors'))
    counts = np.random.RandomState(stream_seed(seed, 'counts'))
    local_np = SimpleNamespace(isfinite=np.isfinite, random=SimpleNamespace(choice=actors.choice, exponential=counts.exponential))

    def ask(model, system, user, parser, parse_args, **kwargs):
        args = dict(parse_args)
        actual_method = kwargs.get('prompt_method', args['method'])
        for attempt in range(1, 4):
            ids = [row[0] for row in json.loads(user)['candidates']]
            default = '\n'.join(', '.join(sorted(pair, key=int)) for pair in zip(ids[:-1], ids[1:])) if args['method'] == 'global' else ', '.join(ids[:args.get('num_choices') or 1])
            response = default if reply is None else reply(system, user, args, attempt)
            audit = dict(method=args['method'], actor=args.get('curr_pid'), count=args.get('num_choices'),
                         candidate_order=ids, attempt=attempt, valid=False)
            requests.append(audit)
            try:
                result = parse_response(response=response, **args)
            except ValueError:
                if args['method'] == 'global' and response.strip() and 'required_global_edges' not in args:
                    pairs = [line.replace(',', ' ').split() for line in response.splitlines()]
                    required = [p for p in pairs if len(p) == 2 and p[0] != p[1] and set(p) <= set(roster)]
                    if not required:
                        raise ValueError('No valid original pairs to repair.')
                    args['required_global_edges'] = required
                system = retry_prompt(actual_method, language, country, args.get('curr_pid'), args.get('num_choices'), 'required_global_edges' in args)
                # A repair request must carry its target, not assume hidden chat history.
                if 'required_global_edges' in args:
                    system += '\n' + json.dumps({'required_pairs': sorted({tuple(sorted(p, key=int)) for p in args['required_global_edges']})})
                audit['retry_prompt'] = system
            else:
                audit['valid'] = True
                return result, response, attempt
        raise ValueError('Fixture exhausted the three-attempt limit.')

    def system_adapter(method, personas, demos, curr_pid=None, num_choices=None, **kwargs):
        return system_prompt(method, language, country, curr_pid, num_choices)

    def user_adapter(method, personas, order, demos, curr_pid=None, G=None, **kwargs):
        return candidate_payload(method, language, curr_pid, G if G is not None else nx.empty_graph(roster), plan['display_order'])

    with patch.object(engine, 'np', local_np), patch.object(engine, 'get_system_prompt', system_adapter), \
            patch.object(engine, 'get_user_prompt', user_adapter), patch.object(engine, 'repeat_prompt_until_parsed', ask), \
            patch.object(frozen.shared, 'OpenAI', side_effect=AssertionError('Offline only')), contextlib.redirect_stdout(io.StringIO()):
        graph, *_ = engine.generate_network(method, previous.DEMOS, roster, list(roster), 'offline-fixture',
            mean_choices=5, num_iter=3, culture_context=country, prompt_language=language, events=events)
    frozen.verify_graph(graph, events, roster)
    return graph, events, requests, plan


def prepare():
    protocol = config()
    roster = previous.adult_roster()
    graph = nx.path_graph(list(roster))
    plan = schedule(21000)
    catalog = []
    for language, country, method in itertools.product(protocol['languages'], protocol['cultures'], ['global', 'local', 'sequential', 'iterative-add', 'iterative-drop']):
        for count in ([1, 2, 8] if method in {'local', 'sequential'} else [None]):
            actor = None if method == 'global' else '0'
            catalog.append(dict(language=language, country=country, method=method, count=count,
                system=system_prompt(method, language, country, actor, count),
                user=candidate_payload(method, language, actor, graph, plan['display_order']),
                retry=retry_prompt(method, language, country, actor, count, method == 'global'),
                evidence_type='offline_prompt_fixture_not_model_output'))
    probe = []
    for language, count, actor, wording in itertools.product(protocol['probe_scope']['languages_initial'], protocol['probe_scope']['counts'], protocol['probe_scope']['actors'], protocol['probe_scope']['wordings']):
        # Same shuffled data in both arms isolates wording, not candidate position.
        text = system_prompt('local', language, 'us', actor, count) if wording == 'v6_candidate' else previous.system_prompt('local', roster, previous.DEMOS, curr_pid=actor, num_choices=count, culture_context='us', prompt_language=language)
        probe.append(dict(language=language, count=count, actor=actor, wording=wording, system=text,
                          user=candidate_payload('local', language, actor, graph, plan['display_order']),
                          evidence_type='UNEXECUTED_COMPLIANCE_PROBE'))
    assert len(probe) == protocol['probe_scope']['first_attempts']
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for name, value in [('prompt_catalog.json', catalog), ('probe_manifest.json', probe), ('schedule_example.json', plan)]:
        (DESTINATION / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    report = dict(status='OFFLINE_CANDIDATE', paid_calls=0, prompt_cases=len(catalog), probe_first_attempts=len(probe),
                  protocol_sha256=frozen.digest(protocol), roster_sha256=frozen.digest(roster),
                  source_sha256={**frozen.source_hashes(), 'revision_next.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                  human_bilingual_validation=False, behavioral_probe_completed=False, main_authorized=False)
    (DESTINATION / 'preparation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--prepare', action='store_true', required=True)
    cli.parse_args()
    print(json.dumps(prepare(), indent=2))
