"""Unadopted global parser/Hindi candidates. No API client or execution command.

This is deliberately separate from the paid V8 contract. Reinterpreting a saved
reply can test a parser, but cannot turn an old failed observation into new data.
"""
import re


def canonical_global(response, roster):
    """Validate every nonblank line, then collapse identical undirected pairs only.

    No ID repair, inferred tie, extra LLM request or mutation of the input graph.
    Keep raw text and compliance flags in any future receipt using this policy.
    """
    if not isinstance(response, str) or not response.strip():
        raise ValueError('Expected a nonblank global response.')
    if response.strip() == 'NONE':
        return dict(raw_response=response, canonical_response='NONE', edges=[],
                    repeated_pairs=0, reversed_pairs=0, interior_blank_lines=0)
    valid_ids = set(roster)
    edges, seen, repeated, reversed_pairs, blanks = [], set(), 0, 0, 0
    for line in response.strip().splitlines():
        if not line.strip():
            blanks += 1  # V8 also permits blank lines; retain this compliance detail.
            continue
        match = re.fullmatch(r'\s*(0|[1-9][0-9]*)\s*(?:,\s*|\s+)(0|[1-9][0-9]*)\s*', line)
        if not match:
            raise ValueError('Malformed pair; do not repair or discard a line.')
        a, b = match.groups()
        if a not in valid_ids or b not in valid_ids or a == b:
            raise ValueError('Unknown ID or self-link; the whole response remains invalid.')
        pair = tuple(sorted((a, b), key=int))
        reversed_pairs += int(int(a) > int(b))
        if pair in seen:
            repeated += 1
        else:
            seen.add(pair)
            edges.append(list(pair))
    if not edges:
        raise ValueError('An empty edge list must be explicitly NONE.')
    return dict(raw_response=response, canonical_response='\n'.join(', '.join(pair) for pair in edges),
                edges=edges, repeated_pairs=repeated, reversed_pairs=reversed_pairs, interior_blank_lines=blanks)


def candidate_catalog():
    import repair_v8 as v8
    old = 'मित्रताओं का एक संभावित विन्यास उत्पन्न करें।'
    new = 'मित्रताओं का ऐसा विन्यास उत्पन्न करें जो यथार्थसंगत और विश्वसनीय हो।'
    rows = []
    changed = 0
    for row in v8.catalog():
        value = dict(row)
        if value['method'] == 'global' and value['language'] == 'hindi':
            if old not in value['system']:
                raise ValueError('The reviewed Hindi source phrase changed.')
            value['system'] = value['system'].replace(old, new)
            changed += 1
        rows.append(value)
    if changed != 16:
        raise ValueError('Expected four countries times four global diagnostic arms.')
    return dict(status='OFFLINE_CANDIDATE_NOT_ADOPTED', main_authorized=False,
        human_bilingual_validation=False, modified_prompt_cases=changed,
        english_gloss='Generate a configuration of friendships that is realistic and plausible.',
        limitation='AI-authored semantic clarification, not a certified translation; live behavior untested.',
        parser_policy='Exact undirected duplicate pairs collapse deterministically; other malformed output remains invalid; raw text and compliance flags retained.',
        rows=rows)
