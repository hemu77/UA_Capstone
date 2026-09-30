"""Fresh instruction-language treatments with identical persona data.

The English field names and values in the candidate data are deliberately fixed.
Only instructions and the translated country label vary. Independent AI review
is recorded in TRANSLATION_REVIEW.md; human bilingual signoff remains separate.
"""
import json
import random

COUNTRIES = {
    'english': {'us': 'the United States', 'india': 'India', 'japan': 'Japan', 'brazil': 'Brazil'},
    'hindi': {'us': 'संयुक्त राज्य अमेरिका', 'india': 'भारत', 'japan': 'जापान', 'brazil': 'ब्राज़ील'},
    'japanese': {'us': '米国', 'india': 'インド', 'japan': '日本', 'brazil': 'ブラジル'},
    'portuguese': {'us': 'Estados Unidos', 'india': 'Índia', 'japan': 'Japão', 'brazil': 'Brasil'},
}
TEXT = {
    'english': {
        'context': 'The social setting is {country}. The people are fictional adults. Keep their attributes unchanged. Their spoken language is unspecified. The candidate data uses fixed English field names and values in every condition. ',
        'global': 'Choose friendships among the listed people. Friendships are undirected. Output one pair of integer IDs per line as ID1, ID2, with ID1 < ID2. No self-links, duplicate pairs, explanation or other text.',
        'local': 'You are persona {actor}. Choose exactly {count} distinct friends from the candidates. Output only their integer IDs separated by commas. No self-links, explanation or other text.',
        'sequential': 'You are persona {actor}, joining a network. Candidate degree gives their current number of friends. Choose exactly {count} distinct friends from the candidates. Output only their integer IDs separated by commas. No self-links, explanation or other text.',
        'iterative-add': 'You are persona {actor}. Choose one new friend from the eligible candidates. Candidate degree is their current number of friends; mutual is the number of current friends shared with you. Output only one integer ID; no explanation or other text.',
        'iterative-drop': 'You are persona {actor}. Choose one existing friend to disconnect from among the eligible candidates. Candidate degree is their current number of friends; mutual is the number of current friends shared with you. Output only one integer ID; no explanation or other text.',
    },
    'hindi': {
        'context': 'सामाजिक परिवेश {country} है। ये लोग काल्पनिक वयस्क हैं। उनके गुण अपरिवर्तित रखें। उनकी बोली जाने वाली भाषा निर्दिष्ट नहीं है। हर स्थिति में उम्मीदवारों के डेटा में अंग्रेज़ी के समान फ़ील्ड नाम और मान हैं। ',
        'global': 'सूचीबद्ध लोगों के बीच मित्रता चुनें। मित्रता में कोई दिशा नहीं होती। हर पंक्ति में पूर्णांक ID की एक जोड़ी ID1, ID2 के रूप में दें, जहां ID1 < ID2 हो। स्वयं से संबंध, दोहराई गई जोड़ी, स्पष्टीकरण या अन्य पाठ न दें।',
        'local': 'आप व्यक्ति {actor} हैं। उम्मीदवारों में से ठीक {count} अलग मित्र चुनें। केवल उनकी पूर्णांक ID अल्पविराम से अलग करके दें। स्वयं से संबंध, स्पष्टीकरण या अन्य पाठ न दें।',
        'sequential': 'आप व्यक्ति {actor} हैं और एक नेटवर्क में शामिल हो रहे हैं। उम्मीदवार की degree उसके वर्तमान मित्रों की संख्या है। उम्मीदवारों में से ठीक {count} अलग मित्र चुनें। केवल उनकी पूर्णांक ID अल्पविराम से अलग करके दें। स्वयं से संबंध, स्पष्टीकरण या अन्य पाठ न दें।',
        'iterative-add': 'आप व्यक्ति {actor} हैं। पात्र उम्मीदवारों में से एक नया मित्र चुनें। उम्मीदवार की degree उसके वर्तमान मित्रों की संख्या है; mutual वर्तमान में आपके और उम्मीदवार के साझा मित्रों की संख्या है। केवल एक पूर्णांक ID दें; स्पष्टीकरण या अन्य पाठ न दें।',
        'iterative-drop': 'आप व्यक्ति {actor} हैं। पात्र उम्मीदवारों में से एक वर्तमान मित्र से संबंध हटाने के लिए चुनें। उम्मीदवार की degree उसके वर्तमान मित्रों की संख्या है; mutual वर्तमान में आपके और उम्मीदवार के साझा मित्रों की संख्या है। केवल एक पूर्णांक ID दें; स्पष्टीकरण या अन्य पाठ न दें।',
    },
    'japanese': {
        'context': '社会的な舞台は{country}です。登場人物は架空の成人です。属性を変更しないでください。登場人物が話す言語は指定されていません。候補者データの英語の項目名と値は、すべての条件で同じです。',
        'global': '一覧の人物間の友人関係を選んでください。友人関係は無向です。各行に整数IDのペアをID1, ID2として記載し、ID1 < ID2を守ってください。自分自身への関係、重複するペア、説明、その他の文章は含めないでください。',
        'local': 'あなたは人物{actor}です。候補者から異なる友人をちょうど{count}人選んでください。整数IDだけをコンマで区切って返してください。自分自身への関係、説明、その他の文章は含めないでください。',
        'sequential': 'あなたはネットワークに参加する人物{actor}です。候補者のdegreeは現在の友人数です。候補者から異なる友人をちょうど{count}人選んでください。整数IDだけをコンマで区切って返してください。自分自身への関係、説明、その他の文章は含めないでください。',
        'iterative-add': 'あなたは人物{actor}です。適格な候補者から新しい友人を1人選んでください。候補者のdegreeは現在の友人数で、mutualは現在あなたと候補者に共通する友人の数です。整数IDを1つだけ返してください。説明やその他の文章は含めないでください。',
        'iterative-drop': 'あなたは人物{actor}です。適格な候補者に含まれる現在の友人から、関係を削除する人物を1人選んでください。候補者のdegreeは現在の友人数で、mutualは現在あなたと候補者に共通する友人の数です。整数IDを1つだけ返してください。説明やその他の文章は含めないでください。',
    },
    'portuguese': {
        'context': 'O cenário social é {country}. As pessoas são adultos fictícios. Mantenha seus atributos inalterados. O idioma falado por elas não foi especificado. Os dados dos candidatos usam os mesmos nomes de campos e valores em inglês em todas as condições. ',
        'global': 'Escolha amizades entre as pessoas listadas. As amizades não são direcionadas. Escreva um par de IDs inteiros por linha como ID1, ID2, com ID1 < ID2. Não inclua ligações consigo mesmo, pares duplicados, explicações ou outro texto.',
        'local': 'Você é a pessoa {actor}. Escolha exatamente {count} amigos distintos entre os candidatos. Retorne apenas seus IDs inteiros separados por vírgulas. Não inclua ligações consigo mesmo, explicações ou outro texto.',
        'sequential': 'Você é a pessoa {actor}, entrando em uma rede. O campo degree do candidato informa seu número atual de amigos. Escolha exatamente {count} amigos distintos entre os candidatos. Retorne apenas seus IDs inteiros separados por vírgulas. Não inclua ligações consigo mesmo, explicações ou outro texto.',
        'iterative-add': 'Você é a pessoa {actor}. Escolha um novo amigo entre os candidatos elegíveis. O campo degree do candidato é seu número atual de amigos; mutual é o número atual de amigos em comum com você. Retorne apenas um ID inteiro, sem explicações ou outro texto.',
        'iterative-drop': 'Você é a pessoa {actor}. Escolha um amigo atual para remover entre os candidatos elegíveis. O campo degree do candidato é seu número atual de amigos; mutual é o número atual de amigos em comum com você. Retorne apenas um ID inteiro, sem explicações ou outro texto.',
    },
}
DEMOS = ['gender', 'age', 'religion', 'political orientation']


def adult_roster():
    """Designed marginals, independently shuffled; no census or LLM fabrication."""
    rng = random.Random(224)
    columns = {
        'gender': ['Man'] * 20 + ['Woman'] * 20 + ['Nonbinary'] * 10,
        'age': list(range(18, 68)),
        'religion': ['Christian', 'Muslim', 'Hindu', 'Buddhist', 'Nonreligious'] * 10,
        'political orientation': ['Left-leaning'] * 17 + ['Centrist'] * 16 + ['Right-leaning'] * 17,
    }
    for values in columns.values():
        rng.shuffle(values)
    return {str(i): {key: values[i] for key, values in columns.items()} for i in range(50)}


def system_prompt(method, personas, demos_to_include, curr_pid=None, G=None,
                  only_degree=True, num_choices=None, include_reason=False,
                  all_demos=False, culture_context=None, prompt_language='english'):
    if prompt_language not in TEXT or culture_context not in COUNTRIES[prompt_language]:
        raise ValueError('Unsupported instruction language or country framing.')
    if method not in TEXT[prompt_language] or method == 'context':
        raise ValueError('Unsupported prompt method.')
    if include_reason or all_demos or not only_degree or demos_to_include != DEMOS:
        raise ValueError('Requested prompt settings differ from the fresh protocol.')
    if method != 'global' and curr_pid not in personas:
        raise ValueError('Missing focal persona.')
    if method in {'local', 'sequential'} and (type(num_choices) is not int or not 1 <= num_choices < len(personas)):
        raise ValueError('Invalid choice count.')
    text = TEXT[prompt_language]
    return text['context'].format(country=COUNTRIES[prompt_language][culture_context]) + text[method].format(actor=curr_pid, count=num_choices)


def user_prompt(method, personas, order, demos_to_include, curr_pid=None,
                G=None, only_degree=True, prompt_language='english'):
    if prompt_language not in TEXT or demos_to_include != DEMOS or not only_degree:
        raise ValueError('Unsupported candidate presentation.')
    if method not in {'global', 'local', 'sequential', 'iterative-add', 'iterative-drop'}:
        raise ValueError('Unknown candidate method.')
    ids = list(order) if order is not None else sorted(personas, key=int)
    if method.startswith('iterative'):
        friends = set(G.neighbors(curr_pid))
        ids = [pid for pid in ids if pid != curr_pid and ((pid in friends) == (method == 'iterative-drop'))]
        random.shuffle(ids)
    else:
        ids = [pid for pid in ids if method == 'global' or pid != curr_pid]
    fields = ['id', *DEMOS]
    if method not in {'global', 'local'}:
        fields.append('degree')
    if method.startswith('iterative'):
        fields.append('mutual')
    rows = []
    for pid in ids:
        row = {'id': pid, **{key: personas[pid][key] for key in DEMOS}}
        if method != 'global' and method != 'local':
            row['degree'] = G.degree(pid)
        if method.startswith('iterative'):
            row['mutual'] = len(set(G.neighbors(pid)) & friends)
        rows.append([row[field] for field in fields])
    # This payload is byte-identical across language treatments given the same
    # roster, candidate order and graph state. JSON ID strings remain canonical.
    # One header names the columns; repeating field names fifty times wastes
    # input tokens without adding information. Name actor columns explicitly:
    # unlike candidates, that row does not include degree or mutual counts.
    payload = {'fields': fields, 'actor_fields': ['id', *DEMOS], 'actor': ([curr_pid, *[personas[curr_pid][key] for key in DEMOS]] if curr_pid is not None else None), 'candidates': rows}
    return json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
