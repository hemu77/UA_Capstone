"""Offline Portuguese repair candidate, NOT integrated into a paid runner.

Global ablation and wording variants remain undecided. V6 is immutable because
its receipts bind exact source hashes; this module must never relabel V6 data.
"""
import revision_next as v6

VERSION = 'v7-portuguese-context-candidate-only'
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
