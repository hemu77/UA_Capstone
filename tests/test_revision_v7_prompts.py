"""Candidate-only checks; passing these does not validate translated meaning."""
import copy
import hashlib
import itertools
import json
from pathlib import Path
import unittest

import revision_next as v6
import revision_v7_prompts as v7


class CandidateTests(unittest.TestCase):
    def test_portuguese_context_and_retry_for_every_country_and_method(self):
        for country, phrase in v7.PORTUGUESE_CONTEXT.items():
            for method in ['global', 'local', 'sequential', 'iterative-add', 'iterative-drop']:
                for count in [1, 2, 8]:
                    actor = None if method == 'global' else '0'
                    choices = None if method.startswith('iterative') or method == 'global' else count
                    for prompt in [v7.system_prompt(method, 'portuguese', country, actor, choices),
                                   v7.retry_prompt(method, 'portuguese', country, actor, choices)]:
                        self.assertIn(phrase, prompt)
                        self.assertNotIn('O cenário social é', prompt)
                        self.assertNotIn('{country}', prompt)

    def test_other_languages_and_frozen_catalog_are_unchanged(self):
        before = copy.deepcopy(v6.TEXT)
        for language, country in itertools.product(['english', 'hindi', 'japanese'], v7.PORTUGUESE_CONTEXT):
            self.assertEqual(v7.system_prompt('local', language, country, '0', 1),
                             v6.system_prompt('local', language, country, '0', 1))
        self.assertEqual(v6.TEXT, before)
        self.assertIn('O cenário social é os Estados Unidos.',
                      v6.system_prompt('local', 'portuguese', 'us', '0', 1))

    def test_repair_preservation_and_argument_validation_remain(self):
        self.assertIn('sem omitir nem adicionar amizades',
                      v7.retry_prompt('global', 'portuguese', 'us', preserving=True))
        for args in [('unknown', 'english', 'us'), ('global', 'spanish', 'us'),
                     ('global', 'portuguese', 'unknown')]:
            with self.assertRaises((ValueError, KeyError)):
                v7.system_prompt(*args)

    def test_unexpected_template_drift_fails_closed(self):
        with self.assertRaises(ValueError):
            v7._repair_context('Different context', 'portuguese', 'us')
        original = 'O cenário social é os Estados Unidos.'
        with self.assertRaises(ValueError):
            v7._repair_context(original + original, 'portuguese', 'us')

    def test_v6_source_hashes_still_match_saved_contract(self):
        root = Path(__file__).resolve().parents[1]
        contract = json.loads((root / 'outputs/calibration_v6/5fb3715550db/contract.json').read_text())
        for name, expected in contract['sources'].items():
            self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), expected, name)


if __name__ == '__main__':
    unittest.main()
