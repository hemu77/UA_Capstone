"""Versioned repair treatments. Never mutate or relabel frozen V6 evidence."""
import revision_next as v6

VERSION = 'v7-diagnostic-prompts-1'
PORTUGUESE_CONTEXT = {
    'us': 'O contexto social é o dos Estados Unidos.',
    'india': 'O contexto social é o da Índia.',
    'japan': 'O contexto social é o do Japão.',
    'brazil': 'O contexto social é o do Brasil.',
}


def _repair_context(prompt, language, country):
    if language != 'portuguese':
        return prompt
    original = f"O cenário social é {v6.COUNTRIES[language][country]}."
    if prompt.count(original) != 1:
        raise ValueError('Expected exactly one frozen Portuguese context sentence.')
    return prompt.replace(original, PORTUGUESE_CONTEXT[country], 1)


def system_prompt(method, language, country, actor=None, count=None):
    return _repair_context(v6.system_prompt(method, language, country, actor, count), language, country)


def retry_prompt(method, language, country, actor=None, count=None, preserving=False):
    return _repair_context(v6.retry_prompt(method, language, country, actor, count, preserving), language, country)


# The response grammar stays identical across arms, including valid empty output.
# B tests an additional empty-network reminder, not permission to return NONE.
GLOBAL_CLAUSES = {
    'english': (
        'There is no required number of friendships; disconnected components are allowed.',
        'Disconnected components are allowed.',
        'If the whole network has no friendships, output only NONE.',
        'Response syntax: NONE, or one ID1, ID2 pair per line.'),
    'portuguese': (
        'Não há uma quantidade obrigatória de amizades; são permitidos componentes desconectados.',
        'São permitidos componentes desconectados.',
        'Se a rede inteira não tiver amizades, retorne apenas NONE.',
        'Formato da resposta: NONE ou um par ID1, ID2 por linha.'),
    'hindi': (
        'मित्रताओं की कोई निर्धारित संख्या नहीं है; अलग-अलग असंबद्ध समूह स्वीकार्य हैं।',
        'अलग-अलग असंबद्ध समूह स्वीकार्य हैं।',
        'यदि पूरे नेटवर्क में कोई मित्रता नहीं है, तो केवल NONE दें।',
        'उत्तर का प्रारूप: NONE या प्रत्येक पंक्ति में ID1, ID2 की एक जोड़ी।'),
    'japanese': (
        '友人関係の総数は指定しません。互いにつながっていないグループも許されます。',
        '互いにつながっていないグループも許されます。',
        'ネットワーク全体に友人関係がない場合はNONEだけを返してください。',
        '回答形式：NONE、または各行にID1, ID2のペアを1組。'),
}

# These are AI-authored alternative local-task phrasings, not human translations
# or independently authored full-context catalogs. Context and persona data stay fixed.
LOCAL_VARIANTS = {
    'english': {
        'alternative_1': (
            'Act as persona {actor}. From the candidates, select exactly one friend. Respond with that person\'s integer ID only. Do not select yourself or include explanations.',
            'Act as persona {actor}. From the candidates, select exactly {count} different friends. Respond only with their integer IDs, separated by commas. Do not select yourself, repeat an ID, or include explanations.'),
        'alternative_2': (
            'For persona {actor}, identify one candidate to be a friend. Exactly one selection is required. Return only the selected candidate\'s integer ID, without explanation or a self-link.',
            'For persona {actor}, identify {count} candidates to be friends. Exactly {count} distinct selections are required. Return only their integer IDs, separated by commas, without explanations, repeated IDs or self-links.'),
    },
    'portuguese': {
        'alternative_1': (
            'Aja como a pessoa {actor}. Entre os candidatos, selecione exatamente um amigo. Responda apenas com o ID inteiro dessa pessoa. Não selecione a si mesmo nem inclua explicações.',
            'Aja como a pessoa {actor}. Entre os candidatos, selecione exatamente {count} amigos diferentes. Responda apenas com seus IDs inteiros, separados por vírgulas. Não selecione a si mesmo, repita um ID ou inclua explicações.'),
        'alternative_2': (
            'Para a pessoa {actor}, identifique um candidato para ser seu amigo. É obrigatória exatamente uma escolha. Retorne somente o ID inteiro do candidato escolhido, sem explicações ou ligação consigo mesmo.',
            'Para a pessoa {actor}, identifique {count} candidatos para serem seus amigos. São obrigatórias exatamente {count} escolhas distintas. Retorne somente seus IDs inteiros, separados por vírgulas, sem explicações, IDs repetidos ou ligações consigo mesmo.'),
    },
    'hindi': {
        'alternative_1': (
            'व्यक्ति {actor} की भूमिका निभाएँ। उम्मीदवारों में से ठीक एक मित्र चुनें। उत्तर में केवल उस व्यक्ति की पूर्णांक ID दें। स्वयं को न चुनें और कोई स्पष्टीकरण न दें।',
            'व्यक्ति {actor} की भूमिका निभाएँ। उम्मीदवारों में से ठीक {count} अलग-अलग मित्र चुनें। उत्तर में केवल उनकी पूर्णांक ID अल्पविराम से अलग करके दें। स्वयं को न चुनें, कोई ID न दोहराएँ और कोई स्पष्टीकरण न दें।'),
        'alternative_2': (
            'व्यक्ति {actor} के लिए मित्र बनने हेतु एक उम्मीदवार चुनें। ठीक एक चयन आवश्यक है। केवल चुने गए उम्मीदवार की पूर्णांक ID लौटाएँ; स्वयं को न चुनें और कोई स्पष्टीकरण न दें।',
            'व्यक्ति {actor} के लिए मित्र बनने हेतु {count} उम्मीदवार चुनें। ठीक {count} अलग-अलग चयन आवश्यक हैं। केवल उनकी पूर्णांक ID अल्पविराम से अलग करके लौटाएँ; स्वयं को न चुनें, कोई ID न दोहराएँ और कोई स्पष्टीकरण न दें।'),
    },
    'japanese': {
        'alternative_1': (
            '人物{actor}として選んでください。候補者から友人をちょうど1人選び、その人物の整数IDだけを回答してください。自分自身を選ばず、説明を含めないでください。',
            '人物{actor}として選んでください。候補者から異なる友人をちょうど{count}人選び、その整数IDだけをコンマで区切って回答してください。自分自身を選ばず、同じIDを繰り返さず、説明を含めないでください。'),
        'alternative_2': (
            '人物{actor}の友人となる候補者を1人選んでください。選ぶ人数は必ず1人です。選んだ候補者の整数IDだけを返してください。自分自身を選ばず、説明を含めないでください。',
            '人物{actor}の友人となる候補者を{count}人選んでください。必ず異なる{count}人を選んでください。その整数IDだけをコンマで区切って返してください。自分自身を選ばず、IDの重複や説明を含めないでください。'),
    },
}


def treatment_prompt(method, language, country, actor=None, count=None,
                     wording='reference', global_arm=None):
    original = system_prompt(method, language, country, actor, count)
    if method != 'global' and global_arm is not None:
        raise ValueError('Global clauses cannot be applied to another method.')
    if method == 'global' and global_arm is not None:
        if global_arm not in {'a0_b0', 'a0_b1', 'a1_b0', 'a1_b1'} or wording != 'reference':
            raise ValueError('Unknown global treatment.')
        quantity, disconnected, empty, grammar = GLOBAL_CLAUSES[language]
        if original.count(quantity) != 1 or original.count(empty) != 1:
            raise ValueError('Global clause source drift.')
        if global_arm.startswith('a0'):
            original = original.replace(quantity, disconnected, 1)
        if global_arm.endswith('b0'):
            original = original.replace(empty, '', 1)
        return original + ' ' + grammar
    if wording == 'reference':
        return original
    if method != 'local' or wording not in LOCAL_VARIANTS[language] or global_arm is not None:
        raise ValueError('Alternative wording is supported only for local decisions.')
    context = _repair_context(v6.TEXT[language]['context'].format(
        country=v6.COUNTRIES[language][country]), language, country)
    return context + LOCAL_VARIANTS[language][wording][count != 1].format(actor=actor, count=count)


def treatment_retry(method, language, country, actor=None, count=None,
                    preserving=False, wording='reference', global_arm=None):
    if wording == 'reference' and global_arm is None:
        return retry_prompt(method, language, country, actor, count, preserving)
    if preserving:
        raise ValueError('Diagnostic treatments do not support global correction retries.')
    prefix = {'english': 'Invalid response. ', 'portuguese': 'Resposta inválida. ',
              'hindi': 'अमान्य उत्तर। ', 'japanese': '無効な回答です。'}[language]
    return prefix + treatment_prompt(method, language, country, actor, count, wording, global_arm)
