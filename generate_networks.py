"""
Network generation engine.

This is the main file that turns personas into a graph:
1. choose a prompting method
2. build prompts for the model
3. parse chosen IDs from the reply
4. add those choices as edges in a NetworkX graph
5. save the graph and basic run statistics
"""

from constants_and_utils import *
from generate_personas import *
import argparse
import json
import numpy as np
import pandas as pd
import time

# The project uses culture and prompt language as two different knobs:
# - culture_context changes the simulated social setting
# - prompt_language changes the language the instructions are written in
# Keeping those separate is what lets the capstone ask RQ1-RQ3 and RQ4 cleanly.
SUPPORTED_PROMPT_LANGUAGES = {'english', 'spanish', 'hindi', 'japanese'}
LANGUAGE_NAME_BY_CODE = {
    'english': 'English',
    'spanish': 'Spanish',
    'hindi': 'Hindi',
    'japanese': 'Japanese',
}
DEMO_LABEL_TRANSLATIONS = {
    'english': {
        'name': 'Name',
        'gender': 'Gender',
        'age': 'Age',
        'race/ethnicity': 'Race/ethnicity',
        'religion': 'Religion',
        'political affiliation': 'Political affiliation',
        'interests': 'Interests',
    },
    'spanish': {
        'name': 'Nombre',
        'gender': 'Genero',
        'age': 'Edad',
        'race/ethnicity': 'Raza/etnia',
        'religion': 'Religion',
        'political affiliation': 'Afiliacion politica',
        'interests': 'Intereses',
    },
    'hindi': {
        'name': 'नाम',
        'gender': 'लिंग',
        'age': 'उम्र',
        'race/ethnicity': 'जाति/जातीयता',
        'religion': 'धर्म',
        'political affiliation': 'राजनीतिक संबद्धता',
        'interests': 'रुचियां',
    },
    'japanese': {
        'name': '名前',
        'gender': '性別',
        'age': '年齢',
        'race/ethnicity': '人種・民族',
        'religion': '宗教',
        'political affiliation': '政治的立場',
        'interests': '興味',
    },
}
VALUE_TRANSLATIONS = {
    'spanish': {
        'Man': 'Hombre',
        'Woman': 'Mujer',
        'Nonbinary': 'No binario',
        'White': 'Blanco',
        'Black': 'Negro',
        'American Indian/Alaska Native': 'Indigena americano/Nativo de Alaska',
        'Asian': 'Asiatico',
        'Native Hawaiian/Pacific Islander': 'Nativo hawaiano/Isleno del Pacifico',
        'Hispanic': 'Hispano',
        'Protestant': 'Protestante',
        'Catholic': 'Catolico',
        'Jewish': 'Judio',
        'Buddhist': 'Budista',
        'Unreligious': 'Sin religion',
        'Muslim': 'Musulman',
        'Hindu': 'Hindu',
        'Christian': 'Cristiano',
        'Republican': 'Republicano',
        'Democrat': 'Democrata',
        'Independent': 'Independiente',
    },
    'hindi': {
        'Man': 'पुरुष',
        'Woman': 'महिला',
        'Nonbinary': 'नॉन-बाइनरी',
        'White': 'श्वेत',
        'Black': 'काला',
        'American Indian/Alaska Native': 'अमेरिकी मूल निवासी/अलास्का मूल निवासी',
        'Asian': 'एशियाई',
        'Native Hawaiian/Pacific Islander': 'मूल हवाईयन/प्रशांत द्वीपीय',
        'Hispanic': 'हिस्पैनिक',
        'Protestant': 'प्रोटेस्टेंट',
        'Catholic': 'कैथोलिक',
        'Jewish': 'यहूदी',
        'Buddhist': 'बौद्ध',
        'Unreligious': 'अधार्मिक',
        'Muslim': 'मुस्लिम',
        'Hindu': 'हिंदू',
        'Christian': 'ईसाई',
        'Republican': 'रिपब्लिकन',
        'Democrat': 'डेमोक्रेट',
        'Independent': 'निर्दलीय',
    },
    'japanese': {
        'Man': '男性',
        'Woman': '女性',
        'Nonbinary': 'ノンバイナリー',
        'White': '白人',
        'Black': '黒人',
        'American Indian/Alaska Native': 'アメリカ先住民/アラスカ先住民',
        'Asian': 'アジア系',
        'Native Hawaiian/Pacific Islander': 'ハワイ先住民/太平洋諸島系',
        'Hispanic': 'ヒスパニック',
        'Protestant': 'プロテスタント',
        'Catholic': 'カトリック',
        'Jewish': 'ユダヤ教徒',
        'Buddhist': '仏教徒',
        'Unreligious': '無宗教',
        'Muslim': 'イスラム教徒',
        'Hindu': 'ヒンドゥー教徒',
        'Christian': 'キリスト教徒',
        'Republican': '共和党支持',
        'Democrat': '民主党支持',
        'Independent': '無党派',
    },
}
PROMPT_TEXT = {
    'english': {
        'culture_statement': (
            'The social network is set in a {culture} cultural context, and everyone communicates in {language}. '
            'Base friendship decisions on norms and expectations that plausibly fit that cultural context while keeping the same people and demographics fixed. '
        ),
        'persona_format_wrapper': 'where each person is described as "{persona_format}"',
        'prompt_extra': 'Do not include any other text in your response. Do not include any people who are not listed below.',
        'valid_ids_note': 'The only valid persona IDs are: {valid_ids}. Use only those IDs. Never use ages, counts, or any other numbers as IDs. ',
        'prompt_all_prefix': 'Pay attention to all demographics. ',
        'you_are': 'You are this person: {persona}.',
        'joining_network': 'You are joining a social network.',
        'list_people_intro': 'You will be provided a list of people in the network, {persona_format}',
        'followed_by': 'followed by ',
        'current_friend_count': 'their current number of friends',
        'current_friend_ids': "their current friends' IDs",
        'which_friends': 'Which of these people will you become friends with? ',
        'choose_people': 'Choose {num} {people_word}. ',
        'people_word_singular': 'person',
        'people_word_plural': 'people',
        'provide_friend_list': 'Provide a list of YOUR friends in the format ID, ID, ID, etc. ',
        'provide_friend_list_with_reason': 'Provide a list of YOUR friends and a short reason for why you are befriending them, in the format:\nID, reason\nID, reason\n...\n\n',
        'global_task': 'Your task is to create a realistic social network. You will be provided a list of people in the network, {persona_format}. Provide a list of friendship pairs in the format ID, ID with each pair separated by a newline. {prompt_extra}',
        'iterative_add_intro': 'You are part of a social network and you want to make a new friend.',
        'iterative_add_people': 'You will be provided a list of potential new friends, {persona_format}, followed by their total number of friends and number of mutual friends with you. ',
        'iterative_existing_friends': 'Keep in mind that you are already friends with IDs {friend_ids}.',
        'iterative_add_question': 'Which person in this list are you likeliest to befriend? ',
        'iterative_add_json': 'Provide your answer in JSON form: {{"new friend": ID, "reason": reason for adding friend}}. ',
        'iterative_only_id': "Answer by providing ONLY this person's ID. ",
        'iterative_drop_intro': 'Unfortunately, you are busy with work and unable to keep up all your friendships.',
        'iterative_drop_people': 'You will be provided a list of your current friends, {persona_format}, followed by their total number of friends and number of mutual friends with you.',
        'iterative_drop_question': 'Which friend in this list are you likeliest to drop? ',
        'iterative_drop_json': 'Provide your answer in JSON form: {{"dropped friend": ID, "reason": reason for dropping friend}}. ',
        'candidate_has_friends': 'has {num} friends',
        'candidate_no_friends': 'no friends yet',
        'candidate_friend_ids': 'friends with IDs {friend_ids}',
        'candidate_stats': '# friends: {num_friends}, # mutual friends: {num_mutual}',
        'iterative_choice_line': 'Which person ID out of {id_list} are you likeliest to {action}?',
        'iterative_action_befriend': 'befriend',
        'iterative_action_drop': 'drop',
        'age_label': 'age',
        'interests_label': 'interests include:',
    },
    'spanish': {
        'culture_statement': (
            'La red social se desarrolla en un contexto cultural de {culture}, y todas las personas se comunican en {language}. '
            'Basa las decisiones de amistad en normas y expectativas que encajen razonablemente con ese contexto cultural, manteniendo fijas a las mismas personas y demografias. '
        ),
        'persona_format_wrapper': 'donde cada persona se describe como "{persona_format}"',
        'prompt_extra': 'No incluyas ningun otro texto en tu respuesta. No incluyas a personas que no aparezcan en la lista.',
        'valid_ids_note': 'Los unicos IDs validos de personas son: {valid_ids}. Usa solo esos IDs. Nunca uses edades, cantidades u otros numeros como IDs. ',
        'prompt_all_prefix': 'Presta atencion a todas las demografias. ',
        'you_are': 'Tu eres esta persona: {persona}.',
        'joining_network': 'Te unes a una red social.',
        'list_people_intro': 'Se te proporcionara una lista de personas de la red, {persona_format}',
        'followed_by': 'seguida de ',
        'current_friend_count': 'su numero actual de amistades',
        'current_friend_ids': 'los IDs de sus amistades actuales',
        'which_friends': 'Con cuales de estas personas te haras amigo? ',
        'choose_people': 'Elige {num} {people_word}. ',
        'people_word_singular': 'persona',
        'people_word_plural': 'personas',
        'provide_friend_list': 'Proporciona una lista de TUS amistades en el formato ID, ID, ID, etc. ',
        'provide_friend_list_with_reason': 'Proporciona una lista de TUS amistades y una razon corta de por que te haces amigo, en el formato:\nID, razon\nID, razon\n...\n\n',
        'global_task': 'Tu tarea es crear una red social realista. Recibiras una lista de personas en la red, {persona_format}. Proporciona una lista de pares de amistad en el formato ID, ID con cada par separado por una nueva linea. {prompt_extra}',
        'iterative_add_intro': 'Formas parte de una red social y quieres hacer una nueva amistad.',
        'iterative_add_people': 'Se te proporcionara una lista de posibles nuevas amistades, {persona_format}, seguida de su numero total de amistades y el numero de amistades en comun contigo. ',
        'iterative_existing_friends': 'Ten en cuenta que ya eres amigo de los IDs {friend_ids}.',
        'iterative_add_question': 'Que persona de esta lista es la mas probable que agregues como amistad? ',
        'iterative_add_json': 'Proporciona tu respuesta en formato JSON: {{"new friend": ID, "reason": razon para agregar la amistad}}. ',
        'iterative_only_id': 'Responde proporcionando SOLO el ID de esa persona. ',
        'iterative_drop_intro': 'Lamentablemente, estas ocupado con el trabajo y no puedes mantener todas tus amistades.',
        'iterative_drop_people': 'Se te proporcionara una lista de tus amistades actuales, {persona_format}, seguida de su numero total de amistades y el numero de amistades en comun contigo.',
        'iterative_drop_question': 'Que amistad de esta lista es la mas probable que elimines? ',
        'iterative_drop_json': 'Proporciona tu respuesta en formato JSON: {{"dropped friend": ID, "reason": razon para eliminar la amistad}}. ',
        'candidate_has_friends': 'tiene {num} amistades',
        'candidate_no_friends': 'todavia no tiene amistades',
        'candidate_friend_ids': 'es amigo de los IDs {friend_ids}',
        'candidate_stats': '# amistades: {num_friends}, # amistades mutuas: {num_mutual}',
        'iterative_choice_line': 'Que ID de persona de entre {id_list} es el mas probable que {action}?',
        'iterative_action_befriend': 'agregues como amistad',
        'iterative_action_drop': 'elimines',
        'age_label': 'edad',
        'interests_label': 'sus intereses incluyen:',
    },
    'hindi': {
        'culture_statement': (
            'यह सामाजिक नेटवर्क {culture} सांस्कृतिक संदर्भ में स्थित है, और सभी लोग {language} में संवाद करते हैं। '
            'दोस्ती के निर्णय ऐसे मानदंडों और अपेक्षाओं पर आधारित हों जो उस सांस्कृतिक संदर्भ के अनुरूप हों, जबकि वही लोग और जनसांख्यिकी स्थिर रहें। '
        ),
        'persona_format_wrapper': 'जहां प्रत्येक व्यक्ति का वर्णन "{persona_format}" के रूप में किया गया है',
        'prompt_extra': 'अपने उत्तर में कोई और पाठ शामिल न करें। नीचे सूचीबद्ध नहीं किए गए किसी भी व्यक्ति को शामिल न करें।',
        'valid_ids_note': 'केवल यही व्यक्ति ID मान्य हैं: {valid_ids}। केवल इन्हीं IDs का उपयोग करें। उम्र, गिनती या किसी अन्य संख्या को ID के रूप में कभी उपयोग न करें। ',
        'prompt_all_prefix': 'सभी जनसांख्यिकीय गुणों पर ध्यान दें। ',
        'you_are': 'आप यह व्यक्ति हैं: {persona}.',
        'joining_network': 'आप एक सामाजिक नेटवर्क में शामिल हो रहे हैं।',
        'list_people_intro': 'आपको नेटवर्क में लोगों की एक सूची दी जाएगी, {persona_format}',
        'followed_by': 'जिसके बाद होगा ',
        'current_friend_count': 'उनके वर्तमान दोस्तों की संख्या',
        'current_friend_ids': 'उनके वर्तमान दोस्तों के ID',
        'which_friends': 'इनमें से आप किन लोगों से दोस्ती करेंगे? ',
        'choose_people': '{num} {people_word} चुनिए। ',
        'people_word_singular': 'व्यक्ति',
        'people_word_plural': 'लोग',
        'provide_friend_list': 'अपने दोस्तों की सूची ID, ID, ID आदि के प्रारूप में दें। ',
        'provide_friend_list_with_reason': 'अपने दोस्तों की सूची और हर दोस्ती का एक छोटा कारण इस प्रारूप में दें:\nID, कारण\nID, कारण\n...\n\n',
        'global_task': 'आपका कार्य एक यथार्थवादी सामाजिक नेटवर्क बनाना है। आपको नेटवर्क में लोगों की एक सूची दी जाएगी, {persona_format}। मित्रता की जोडियां ID, ID प्रारूप में दें और प्रत्येक जोड़ी नई पंक्ति में हो। {prompt_extra}',
        'iterative_add_intro': 'आप एक सामाजिक नेटवर्क का हिस्सा हैं और एक नया दोस्त बनाना चाहते हैं।',
        'iterative_add_people': 'आपको संभावित नए दोस्तों की एक सूची दी जाएगी, {persona_format}, और उनके कुल दोस्तों की संख्या तथा आपके साथ साझा दोस्तों की संख्या भी दी जाएगी। ',
        'iterative_existing_friends': 'ध्यान रखें कि आप पहले से IDs {friend_ids} के दोस्त हैं।',
        'iterative_add_question': 'इस सूची में से आप किस व्यक्ति से सबसे अधिक दोस्ती करना चाहेंगे? ',
        'iterative_add_json': 'अपना उत्तर JSON रूप में दें: {{"new friend": ID, "reason": दोस्त जोड़ने का कारण}}. ',
        'iterative_only_id': 'केवल उस व्यक्ति का ID देकर उत्तर दें। ',
        'iterative_drop_intro': 'दुर्भाग्य से, आप काम में व्यस्त हैं और अपनी सभी दोस्तियों को निभा नहीं पा रहे हैं।',
        'iterative_drop_people': 'आपको अपने वर्तमान दोस्तों की एक सूची दी जाएगी, {persona_format}, और उनके कुल दोस्तों की संख्या तथा आपके साथ साझा दोस्तों की संख्या भी दी जाएगी।',
        'iterative_drop_question': 'इस सूची में से आप किस दोस्त को छोड़ने की सबसे अधिक संभावना रखते हैं? ',
        'iterative_drop_json': 'अपना उत्तर JSON रूप में दें: {{"dropped friend": ID, "reason": दोस्ती छोड़ने का कारण}}. ',
        'candidate_has_friends': 'इसके {num} दोस्त हैं',
        'candidate_no_friends': 'अभी तक कोई दोस्त नहीं है',
        'candidate_friend_ids': 'यह IDs {friend_ids} का दोस्त है',
        'candidate_stats': '# दोस्त: {num_friends}, # साझा दोस्त: {num_mutual}',
        'iterative_choice_line': 'IDs {id_list} में से आप किस व्यक्ति को सबसे अधिक {action}?',
        'iterative_action_befriend': 'दोस्त बनाएंगे',
        'iterative_action_drop': 'छोड़ेंगे',
        'age_label': 'उम्र',
        'interests_label': 'रुचियां शामिल हैं:',
    },
    'japanese': {
        'culture_statement': (
            'この社会的ネットワークは{culture}の文化的文脈に設定されており、全員が{language}でやり取りします。'
            '同じ人々と属性を固定したまま、その文化的文脈にもっともらしく合う規範や期待に基づいて友人関係を判断してください。'
        ),
        'persona_format_wrapper': '各人物は"{persona_format}"として記述されています',
        'prompt_extra': '回答にはそれ以外の文章を含めないでください。以下に listed されていない人物を含めないでください。',
        'valid_ids_note': '有効な人物IDは次のみです: {valid_ids}。必ずそのIDだけを使ってください。年齢や人数などの他の数字をIDとして使わないでください。 ',
        'prompt_all_prefix': 'すべての属性に注意してください。 ',
        'you_are': 'あなたはこの人物です: {persona}。',
        'joining_network': 'あなたは社会的ネットワークに参加します。',
        'list_people_intro': 'ネットワーク内の人物一覧が与えられます。{persona_format}',
        'followed_by': 'その後に続くのは',
        'current_friend_count': '現在の友人数',
        'current_friend_ids': '現在の友人ID',
        'which_friends': 'この中の誰と友達になりますか。 ',
        'choose_people': '{num} {people_word}選んでください。 ',
        'people_word_singular': '人',
        'people_word_plural': '人',
        'provide_friend_list': 'あなたの友人を ID, ID, ID などの形式で答えてください。 ',
        'provide_friend_list_with_reason': 'あなたの友人一覧とその理由を次の形式で答えてください:\nID, 理由\nID, 理由\n...\n\n',
        'global_task': 'あなたの仕事は現実的な社会的ネットワークを作ることです。ネットワーク内の人物一覧が与えられます。{persona_format}。友人関係のペアを ID, ID の形式で、各ペアを改行区切りで答えてください。{prompt_extra}',
        'iterative_add_intro': 'あなたは社会的ネットワークの一員で、新しい友人を作りたいと思っています。',
        'iterative_add_people': '候補となる新しい友人の一覧が与えられます。{persona_format}。さらに、その人の総友人数とあなたとの共通友人数も与えられます。 ',
        'iterative_existing_friends': 'あなたはすでに IDs {friend_ids} と友達であることを覚えておいてください。',
        'iterative_add_question': 'この一覧の中で、最も友達になりそうな人は誰ですか。 ',
        'iterative_add_json': '回答は JSON 形式で答えてください: {{"new friend": ID, "reason": 友人を追加する理由}}. ',
        'iterative_only_id': 'その人物の ID のみで答えてください。 ',
        'iterative_drop_intro': '残念ながら、あなたは仕事で忙しく、すべての友人関係を維持できません。',
        'iterative_drop_people': '現在の友人一覧が与えられます。{persona_format}。さらに、その人の総友人数とあなたとの共通友人数も与えられます。',
        'iterative_drop_question': 'この一覧の中で、最も関係を切りそうな友人は誰ですか。 ',
        'iterative_drop_json': '回答は JSON 形式で答えてください: {{"dropped friend": ID, "reason": 友人関係を切る理由}}. ',
        'candidate_has_friends': '友人数は{num}人',
        'candidate_no_friends': 'まだ友人がいません',
        'candidate_friend_ids': '友人IDは {friend_ids}',
        'candidate_stats': '友人数: {num_friends}, 共通友人数: {num_mutual}',
        'iterative_choice_line': 'IDs {id_list} の中で、最も {action} しそうな人物IDはどれですか。',
        'iterative_action_befriend': '友達に',
        'iterative_action_drop': '関係を切る',
        'age_label': '年齢',
        'interests_label': '興味には次が含まれます:',
    },
}


def normalize_condition_token(value):
    return value.lower().replace(' ', '-').replace('/', '-')


def get_prompt_language(prompt_language):
    if prompt_language is None:
        return 'english'
    prompt_language = prompt_language.lower()
    if prompt_language not in SUPPORTED_PROMPT_LANGUAGES:
        raise ValueError(f'Unsupported prompt language: {prompt_language}')
    return prompt_language


def translate_prompt_text(prompt_language, key, **kwargs):
    prompt_language = get_prompt_language(prompt_language)
    return PROMPT_TEXT[prompt_language][key].format(**kwargs)


def translate_demo_label(demo, prompt_language):
    prompt_language = get_prompt_language(prompt_language)
    return DEMO_LABEL_TRANSLATIONS[prompt_language][demo]


def translate_value(value, prompt_language):
    prompt_language = get_prompt_language(prompt_language)
    if prompt_language == 'english':
        return value
    return VALUE_TRANSLATIONS.get(prompt_language, {}).get(str(value), str(value))


def localize_persona_string(persona, demos_to_include, pid=None, prompt_language='english'):
    """
    Convert a persona into the surface language used by the prompt.
    """
    prompt_language = get_prompt_language(prompt_language)
    if pid is None:
        s = ''
    else:
        s = f'{pid}. '
    if 'name' in demos_to_include:
        name = ' '.join(persona['name'])
        s += f'{name} - '
    for pos, demo in enumerate(demos_to_include):
        if demo == 'name':
            continue
        value = persona[demo]
        if demo != 'age':
            value = translate_value(value, prompt_language)
        if demo == 'age':
            s += f"{translate_prompt_text(prompt_language, 'age_label')} {value}, "
        elif demo == 'interests' and pos > 0:
            s += f"{translate_prompt_text(prompt_language, 'interests_label')} {value}, "
        else:
            s += f'{value}, '
    return s[:-2]


def assign_persona_to_prompt(persona, demos_to_include, prompt_language='english'):
    persona_str = localize_persona_string(persona, demos_to_include, prompt_language=prompt_language)
    return translate_prompt_text(prompt_language, 'you_are', persona=persona_str)


def get_culture_statement(culture_context, prompt_language='english'):
    if not culture_context:
        return ''
    language_name = LANGUAGE_NAME_BY_CODE[get_prompt_language(prompt_language)]
    return translate_prompt_text(prompt_language, 'culture_statement', culture=culture_context, language=language_name)


def get_persona_format(demos_to_include, prompt_language='english'):
    """
    Define persona format for GPT: eg, "ID. Name - Gender, Age, Race/ethnicity, Religion, Political Affiliation". 
    """
    persona_format = 'ID. '
    if 'name' in demos_to_include:
        persona_format += translate_demo_label('name', prompt_language) + ' - '
    for demo in demos_to_include:
        if demo != 'name':
            persona_format += f'{translate_demo_label(demo, prompt_language)}, '
    persona_format = persona_format[:-2]  # remove trailing ', '
    return persona_format


def get_system_prompt(method, personas, demos_to_include, curr_pid=None, G=None, 
                      only_degree=True, num_choices=None, include_reason=False, all_demos=False,
                      culture_context=None, prompt_language='english'):
    """
    Get content for system message.
    """
    assert method in {'global', 'local', 'sequential', 'iterative-add', 'iterative-drop'}
    if G is not None:
        assert 'iterative' in method 
    if (curr_pid is not None) or include_reason:
        assert method != 'global'
    if num_choices is not None:
        assert method in {'local', 'sequential'}
        assert num_choices >= 1

    # Build reusable text fragments first so every prompting method stays consistent.
    prompt_language = get_prompt_language(prompt_language)
    persona_format = get_persona_format(demos_to_include, prompt_language=prompt_language)
    persona_format = translate_prompt_text(prompt_language, 'persona_format_wrapper', persona_format=persona_format)
    prompt_extra = translate_prompt_text(prompt_language, 'prompt_extra')
    if all_demos:
        prompt_extra = translate_prompt_text(prompt_language, 'prompt_all_prefix') + prompt_extra
    culture_statement = get_culture_statement(culture_context, prompt_language=prompt_language)
    valid_ids = ', '.join(sorted(personas.keys(), key=int))
    valid_ids_note = translate_prompt_text(prompt_language, 'valid_ids_note', valid_ids=valid_ids)
    if curr_pid is not None:
        prompt_personal = assign_persona_to_prompt(personas[curr_pid], demos_to_include, prompt_language=prompt_language)
    
    if method == 'global':
        prompt = culture_statement + translate_prompt_text(
            prompt_language,
            'global_task',
            persona_format=persona_format,
            prompt_extra=prompt_extra,
        ) + ' ' + valid_ids_note
    
    elif method in {'local', 'sequential'}:
        # In these methods one persona is making friendship decisions.
        prompt = (
            culture_statement
            + prompt_personal
            + ' '
            + translate_prompt_text(prompt_language, 'joining_network')
            + '\n\n'
            + translate_prompt_text(prompt_language, 'list_people_intro', persona_format=persona_format)
        )
        if method == 'sequential':
            prompt += ', ' + translate_prompt_text(prompt_language, 'followed_by')
            if only_degree:
                prompt += translate_prompt_text(prompt_language, 'current_friend_count')
            else:
                prompt += translate_prompt_text(prompt_language, 'current_friend_ids')
        prompt += '.\n\n' + translate_prompt_text(prompt_language, 'which_friends')
        if num_choices is not None:
            people_word = translate_prompt_text(
                prompt_language,
                'people_word_plural' if num_choices > 1 else 'people_word_singular',
            )
            prompt += translate_prompt_text(prompt_language, 'choose_people', num=num_choices, people_word=people_word)
        if include_reason:
            prompt += translate_prompt_text(prompt_language, 'provide_friend_list_with_reason')
        else:
            prompt += translate_prompt_text(prompt_language, 'provide_friend_list')
        prompt += valid_ids_note + prompt_extra
    
    elif method == 'iterative-add':
        prompt = (
            culture_statement
            + prompt_personal
            + ' '
            + translate_prompt_text(prompt_language, 'iterative_add_intro')
            + '\n\n'
            + translate_prompt_text(prompt_language, 'iterative_add_people', persona_format=persona_format)
        )
        curr_friends = ', '.join(list(G.neighbors(curr_pid)))
        prompt += (
            translate_prompt_text(prompt_language, 'iterative_existing_friends', friend_ids=curr_friends)
            + '\n\n'
            + translate_prompt_text(prompt_language, 'iterative_add_question')
        )
        if include_reason:
            prompt += translate_prompt_text(prompt_language, 'iterative_add_json')
        else:
            prompt += translate_prompt_text(prompt_language, 'iterative_only_id')
        prompt += valid_ids_note + prompt_extra
    
    else:  # iterative-drop
        prompt = (
            culture_statement
            + prompt_personal
            + ' '
            + translate_prompt_text(prompt_language, 'iterative_drop_intro')
            + '\n\n'
            + translate_prompt_text(prompt_language, 'iterative_drop_people', persona_format=persona_format)
        )
        prompt += '\n\n' + translate_prompt_text(prompt_language, 'iterative_drop_question')
        if include_reason:
            prompt += translate_prompt_text(prompt_language, 'iterative_drop_json')
        else:
            prompt += translate_prompt_text(prompt_language, 'iterative_only_id')
        prompt += valid_ids_note + prompt_extra
    return prompt 


def get_user_prompt(method, personas, order, demos_to_include, curr_pid=None, 
                    G=None, only_degree=True, prompt_language='english'):
    """
    Get content for user message.
    """
    assert method in {'global', 'local', 'sequential', 'iterative-add', 'iterative-drop'}        
    prompt_language = get_prompt_language(prompt_language)
    lines = []
    if method == 'global':
        for pid in order:
            lines.append(localize_persona_string(personas[pid], demos_to_include, pid=pid, prompt_language=prompt_language))
    
    elif method == 'local':
        assert curr_pid is not None 
        for pid in order:
            if pid != curr_pid:
                lines.append(localize_persona_string(personas[pid], demos_to_include, pid=pid, prompt_language=prompt_language))
        assert len(lines) == (len(order)-1)
    
    elif method == 'sequential':
        assert curr_pid is not None 
        assert G is not None
        # Sequential mode exposes lightweight graph state for each candidate.
        for pid in order:
            if pid != curr_pid:
                persona = localize_persona_string(personas[pid], demos_to_include, pid=pid, prompt_language=prompt_language)
                cand_friends = set(G.neighbors(pid))  # candidate's friends
                if only_degree:
                    persona += '; ' + translate_prompt_text(prompt_language, 'candidate_has_friends', num=len(cand_friends))
                else:
                    if len(cand_friends) == 0:
                        persona += '; ' + translate_prompt_text(prompt_language, 'candidate_no_friends')
                    else:
                        persona += '; ' + translate_prompt_text(
                            prompt_language,
                            'candidate_friend_ids',
                            friend_ids=', '.join(cand_friends),
                        )
                lines.append(persona)
        assert len(lines) == (len(order)-1)
        
    else:  # iterative
        assert curr_pid is not None 
        assert G is not None
        friends = list(G.neighbors(curr_pid))
        if method == 'iterative-add':
            id_list = list(set(G.nodes()) - set(friends) - {curr_pid})  # non-friends
            action = 'befriend'
        else:
            id_list = friends  # current friends
            action = 'drop'
        random.shuffle(id_list)
        for pid in id_list:
            persona = localize_persona_string(personas[pid], demos_to_include, pid=pid, prompt_language=prompt_language)
            cand_friends = set(G.neighbors(pid))  # candidate's friends
            mutuals = set(friends).intersection(cand_friends)
            lines.append(
                persona
                + '; '
                + translate_prompt_text(
                    prompt_language,
                    'candidate_stats',
                    num_friends=len(cand_friends),
                    num_mutual=len(mutuals),
                )
            )
        id_list = ', '.join(id_list)
        lines.append(
            translate_prompt_text(
                prompt_language,
                'iterative_choice_line',
                id_list=id_list,
                action=translate_prompt_text(prompt_language, f'iterative_action_{action}'),
            )
        )
    
    prompt = '\n'.join(lines)
    return prompt 
    

def update_graph_from_response(method, response, G, curr_pid=None, include_reason=False, num_choices=None):
    """
    Parse response from LLM and update graph based on edges found.
    Expectation:
    - 'global' response should list all edges in the graph
    - 'local' and 'sequential' should list all new edges for curr_pid
    - 'iterative-add' should list one new edge to add for curr_pid
    - 'iterative-drop' should list one existing edge to drop for curr_pid
    """
    assert method in {'global', 'local', 'sequential', 'iterative-add', 'iterative-drop'}
    if num_choices is not None:
        assert method in {'local', 'sequential'}
    if include_reason:
        assert method != 'global' and curr_pid is not None
        reasons = {}
    edges_found = []
    
    # The parser is intentionally strict so malformed model output does not
    # silently create bad graph structure.
    lines = response.split('\n')
    if method == 'global':
        for line in lines:
            cleaned = line.strip().replace(',', ' ')
            if not cleaned:
                continue
            parts = cleaned.split()
            assert len(parts) == 2, 'Each friendship pair must contain exactly two IDs'
            id1, id2 = parts
            edges_found.append((id1.strip(), id2.strip()))
    
    elif method == 'local' or method == 'sequential':
        assert curr_pid is not None, f'{method} method needs curr_pid to parse response'
        new_edges = []
        if include_reason:
            for line in lines:
                pid, reason = line.strip('.').split(',', 1)
                new_edges.append((curr_pid, pid.strip()))
                reasons[pid] = reason.strip()
        else:
            assert len(lines) == 1, f'Response should not be more than one line'
            line = lines[0].replace(',', ' ').replace('.', ' ')
            ids = line.split()
            for pid in ids:
                assert pid.isnumeric(), f'Response should contain ONLY the ID(s)'
                new_edges.append((curr_pid, pid.strip()))
        if num_choices is not None:
            pp = 'people' if num_choices > 1 else 'person'
            assert len(new_edges) == num_choices, f'Choose {num_choices} {pp}'
        edges_found.extend(new_edges)
    
    else:  # iterative-add or iterative-drop
        assert curr_pid is not None, f'{method} method needs curr_pid to parse response'
        if include_reason:
            resp = json.loads(response.strip())
            key = 'new friend' if method == 'iterative-add' else 'dropped friend'
            assert key in resp, f'Missing "{key}" in response'
            pid = str(resp[key])
            action = method.split('-')[1]
            reasons[(pid, action)] = reason
        else:
            assert len(lines) == 1, f'Response should not be more than one line'
            pid = lines[0].strip('.')
            assert len(pid.split()) == 1 and pid.isnumeric(), f'Response should contain only the ID of the person you\'re choosing'
        assert pid.lower() != 'none', 'You must choose one of the IDs in the list'
        edges_found.append((curr_pid, pid))
    
    orig_len = len(edges_found)
    edges_found = set(edges_found)
    if len(edges_found) < orig_len:
        print(f'Warning: {orig_len} edges were returned, {len(edges_found)} are unique')
    
    # check all valid
    valid_nodes = set(G.nodes())
    curr_edges = set(G.edges())
    for id1, id2 in edges_found:
        assert id1 in valid_nodes, f'{id1} is not a node in the network'
        assert id2 in valid_nodes, f'{id2} is not a node in the network'
        if method == 'iterative-drop':
            assert ((id1, id2) in curr_edges) or ((id2, id1) in curr_edges), f'{id2} is not an existing friend'

    # only modify graph at the end
    if method == 'iterative-drop':
        G.remove_edges_from(edges_found)
    else:
        G.add_edges_from(edges_found)
    if include_reason:
        return G, reasons 
    return G
    
    
def generate_network(method, demos_to_include, personas, order, model, mean_choices=None, include_reason=False, 
                     all_demos=False, only_degree=True, num_iter=3, temp=None, verbose=False,
                     culture_context=None, prompt_language='english'):
    """
    Generate entire network.
    """
    assert method in {'global', 'local', 'sequential', 'iterative'}
    G = nx.Graph()
    G.add_nodes_from(order)
    reasons = {}
    total_num_tries = 0
    total_input_toks = 0
    total_output_toks = 0
    
    if method == 'global':
        system_prompt = get_system_prompt(method, personas, demos_to_include, all_demos=all_demos,
                                          culture_context=culture_context, prompt_language=prompt_language)
        user_prompt = get_user_prompt(method, personas, order, demos_to_include, prompt_language=prompt_language)
        parse_args = {'method': method, 'G': G}
        G, response, num_tries = repeat_prompt_until_parsed(model, system_prompt, user_prompt, update_graph_from_response,
                                                            parse_args, temp=temp, verbose=verbose)
        total_num_tries += num_tries
        total_input_toks += len(system_prompt.split()) + len(user_prompt.split())
        total_output_toks += len(response.split())
    
    elif method == 'local' or method == 'sequential':
        order2 = np.random.choice(order, size=len(order), replace=False)  # order of adding nodes
        print('Order of assigning:', order2[:10])
        for node_num, pid in enumerate(order2):
            if mean_choices is None:
                num_choices = None 
            else:
                num_choices = int(min(max(np.random.exponential(mean_choices), 1), 20))
            # Early in the process the graph is almost empty, so the first few
            # people use local mode before switching into full sequential mode.
            if node_num < 3:  # for first three nodes, use local
                system_prompt = get_system_prompt('local', personas, demos_to_include, curr_pid=pid,
                                    num_choices=num_choices, include_reason=include_reason, all_demos=all_demos,
                                    culture_context=culture_context, prompt_language=prompt_language)
                user_prompt = get_user_prompt('local', personas, order, demos_to_include, curr_pid=pid,
                                              prompt_language=prompt_language)
            else:  # otherwise, allow local or sequential
                system_prompt = get_system_prompt(method, personas, demos_to_include, curr_pid=pid, 
                    num_choices=num_choices, include_reason=include_reason, all_demos=all_demos,
                    only_degree=only_degree, culture_context=culture_context, prompt_language=prompt_language)
                user_prompt = get_user_prompt(method, personas, order, demos_to_include, curr_pid=pid,
                                               G=G, only_degree=only_degree, prompt_language=prompt_language)
            parse_args = {'method': method, 'G': G, 'curr_pid': pid, 'num_choices': num_choices, 'include_reason': include_reason}
            G, response, num_tries = repeat_prompt_until_parsed(model, system_prompt, user_prompt, 
                    update_graph_from_response, parse_args, temp=temp, verbose=verbose)
            if include_reason:
                G, pid_reasons = G 
                print(pid, pid_reasons)
                reasons[pid] = pid_reasons
            total_num_tries += num_tries
            total_input_toks += len(system_prompt.split()) + len(user_prompt.split())
            total_output_toks += len(response.split())
            
    else:  # iterative
        # construct local network first 
        order2 = np.random.choice(order, size=len(order), replace=False)  # order of adding nodes
        for pid in order2:
            if mean_choices is None:
                num_choices = None 
            else:
                num_choices = int(max(np.random.exponential(mean_choices), 1))
            system_prompt = get_system_prompt('local', personas, demos_to_include, curr_pid=pid,
                                num_choices=num_choices, include_reason=include_reason, all_demos=all_demos,
                                culture_context=culture_context, prompt_language=prompt_language)
            user_prompt = get_user_prompt('local', personas, order, demos_to_include, curr_pid=pid,
                                          prompt_language=prompt_language)
            parse_args = {'method': 'local', 'G': G, 'curr_pid': pid, 'num_choices': num_choices, 'include_reason': include_reason}
            G, response, num_tries = repeat_prompt_until_parsed(model, system_prompt, user_prompt, 
                    update_graph_from_response, parse_args, temp=temp, verbose=verbose)
            if include_reason:
                G, pid_reasons = G 
                reasons[pid] = pid_reasons
            total_num_tries += num_tries
            total_input_toks += len(system_prompt.split()) + len(user_prompt.split())
            total_output_toks += len(response.split())
        print('Constructed initial network using local method')
        
        for it in range(num_iter):
            print(f'========= ITERATION {it} =========')
            order3 = np.random.choice(order2, size=len(order2), replace=False)  # order of rewiring nodes
            for pid in order3:  # iterate through nodes and rewire
                system_prompt = get_system_prompt('iterative-add', personas, demos_to_include, 
                        curr_pid=pid, G=G, include_reason=include_reason, all_demos=all_demos,
                        culture_context=culture_context, prompt_language=prompt_language)
                user_prompt = get_user_prompt('iterative-add', personas, None, demos_to_include, 
                                              curr_pid=pid, G=G, prompt_language=prompt_language)
                parse_args = {'method': 'iterative-add', 'G': G, 'curr_pid': pid, 'include_reason': include_reason}
                G, response_add, num_tries = repeat_prompt_until_parsed(model, system_prompt, user_prompt, 
                        update_graph_from_response, parse_args, temp=temp, verbose=verbose)
                if include_reason:
                    G, pid_reasons = G 
                    reasons[pid] = pid_reasons
                total_num_tries += num_tries
                total_input_toks += len(system_prompt.split()) + len(user_prompt.split())
                total_output_toks += len(response_add.split())
                
                friends = list(G.neighbors(pid))
                if len(friends) > 1:
                    system_prompt = get_system_prompt('iterative-drop', personas, demos_to_include, 
                            curr_pid=pid, G=G, include_reason=include_reason, all_demos=all_demos,
                            culture_context=culture_context, prompt_language=prompt_language)
                    user_prompt = get_user_prompt('iterative-drop', personas, None, demos_to_include, 
                                                  curr_pid=pid, G=G, prompt_language=prompt_language)
                    parse_args = {'method': 'iterative-drop', 'G': G, 'curr_pid': pid, 'include_reason': include_reason}
                    G, response_drop, num_tries = repeat_prompt_until_parsed(model, system_prompt, user_prompt, 
                            update_graph_from_response, parse_args, temp=temp, verbose=verbose)
                    if include_reason:
                        G, pid_reasons = G 
                        reasons[pid] = pid_reasons
                    total_num_tries += num_tries
                    total_input_toks += len(system_prompt.split()) + len(user_prompt.split())
                    total_output_toks += len(response_drop.split())
                else:  
                    assert len(friends) == 1  # must be at least 1 because we just added
                    G.remove_edge(pid, friends[0])
                print(pid, response_add, response_drop)
                
    return G, reasons, total_num_tries, total_input_toks, total_output_toks
   

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('method', type=str, choices=['global', 'local', 'sequential', 'iterative'])
    parser.add_argument('--persona_fn', type=str, default='us_50_gpt4o_w_interests.json')
    parser.add_argument('--mean_choices', type=int, default=-1)
    parser.add_argument('--include_names', action='store_true')
    parser.add_argument('--include_interests', action='store_true')
    parser.add_argument('--only_interests', action='store_true')
    parser.add_argument('--shuffle_all', action='store_true')
    parser.add_argument('--shuffle_interests', action='store_true')
    parser.add_argument('--include_friend_list', action='store_true')
    parser.add_argument('--include_reason', action='store_true')
    parser.add_argument('--prompt_all', action='store_true')

    parser.add_argument('--model', type=str, default='gpt-4.1-mini')
    parser.add_argument('--num_networks', type=int, default=1)
    parser.add_argument('--start_seed', type=int, default=0)  # set start seed to continue with new seeds
    parser.add_argument('--temp', type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument('--num_iter', type=int, default=3)  # only used when method is iterative
    parser.add_argument('--culture_context', type=str, default=None)
    parser.add_argument('--prompt_language', type=str, default=None)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args

def get_save_prefix_and_demos(args):
    """
    Get save prefix and demos to include based on args.
    """
    # The save prefix acts like the experiment ID and is reused for filenames.
    save_prefix = f'{args.method}_{args.model}'
    demos_to_include = []
    if args.mean_choices != -1:
        assert args.mean_choices > 0
        save_prefix += '_n' + str(args.mean_choices)
    if args.only_interests:
        save_prefix += '_only_interests'
        demos_to_include.append('interests')
    else:
        if args.include_names:
            save_prefix += '_w_names'
            demos_to_include.append('name')        
        demos_to_include.extend(['gender', 'age', 'race/ethnicity', 'religion', 'political affiliation'])
        if args.include_interests:
            save_prefix += '_w_interests'
            demos_to_include.append('interests')
    if args.shuffle_interests:
        assert args.include_interests or args.only_interests
        assert '_INTERESTS_SHUFFLED' in args.persona_fn
        save_prefix += '_INTERESTS_SHUFFLED'
    if args.shuffle_all:
        assert '_ALL_SHUFFLED' in args.persona_fn
        save_prefix += '_ALL_SHUFFLED'
    if args.include_friend_list:
        save_prefix += '_w_list'  # list of friends
    if args.include_reason:
        save_prefix += '_w_reason'
    if args.prompt_all:
        save_prefix += '_prompt_all'
    if args.culture_context:
        save_prefix += '_culture_' + normalize_condition_token(args.culture_context)
    if args.prompt_language is not None:
        save_prefix += '_lang_' + normalize_condition_token(args.prompt_language)
    if args.temp != DEFAULT_TEMPERATURE:
        temp_str = str(args.temp).replace('.', '')
        save_prefix += f'_temp{temp_str}'
    return save_prefix, demos_to_include


if __name__ == '__main__':
    args = parse_args()
    save_prefix, demos_to_include = get_save_prefix_and_demos(args)
    print('save prefix:', save_prefix)
    fn = os.path.join(PATH_TO_TEXT_FILES, args.persona_fn)
    with open(fn) as f:
        personas = json.load(f)
    pids = list(personas.keys())
    print(f'Loaded {len(pids)} personas from {args.persona_fn}')

    stats = []    
    end_seed = args.start_seed+args.num_networks
    for seed in range(args.start_seed, end_seed):
        ts = time.time()
        np.random.seed(seed)
        order = np.random.choice(pids, size=len(pids), replace=False)  # order of printing personas
        print('Order of printing:', order[:10])
        G, reasons, num_tries, input_toks, output_toks = generate_network(
            args.method, demos_to_include, personas, order, args.model, 
            mean_choices=args.mean_choices if args.mean_choices > 0 else None,
            include_reason=args.include_reason, all_demos=args.prompt_all, 
            only_degree=not args.include_friend_list, temp=args.temp, num_iter=args.num_iter,
            verbose=args.verbose, culture_context=args.culture_context,
            prompt_language=get_prompt_language(args.prompt_language))
        
        save_network(G, f'{save_prefix}_{seed}')
        draw_and_save_network_plot(G, f'{save_prefix}_{seed}')
        duration = time.time()-ts
        print(f'Seed {seed}: {len(G.edges())} edges, num tries={num_tries}, input toks={input_toks}, output toks={output_toks} [time={duration:.2f}s]')
        stats.append({'seed': seed, 'duration': duration, 'num_tries': num_tries, 
                      'num_input_toks': input_toks, 'num_output_toks': output_toks})
        if args.include_reason:
            fn = os.path.join(PATH_TO_TEXT_FILES, f'{save_prefix}_{seed}_reasons.json')
            with open(fn, 'w') as f:
                json.dump(reasons, f)
    
    stats_df = pd.DataFrame(stats, columns=['seed', 'duration', 'num_tries', 'num_input_toks', 'num_output_toks'])
    save_dir = os.path.join(PATH_TO_STATS_FILES, save_prefix)
    if not os.path.exists(save_dir):
        print('Making directory:', save_dir)
        os.makedirs(save_dir)
    stats_fn = os.path.join(PATH_TO_STATS_FILES, save_prefix, f'cost_stats_s{args.start_seed}-{end_seed-1}.csv')
    stats_df.to_csv(stats_fn, index=False)
