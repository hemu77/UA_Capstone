"""Offline prompt/parser checks, not evidence of model or human compliance."""
import unittest

import networkx as nx

import revision_hindi_candidate as candidate
import revision_next as protocol
import revision_v7_prompts as frozen


class HindiCandidateTests(unittest.TestCase):
    def test_same_context_and_selection_with_explicit_serialization(self):
        for country in ('us', 'india', 'japan', 'brazil'):
            for count in (1, 2, 49):
                for retry in (False, True):
                    with self.subTest(country=country, count=count, retry=retry):
                        render = frozen.treatment_retry if retry else frozen.treatment_prompt
                        original = render('local', 'hindi', country, '30', count,
                                          wording='alternative_2')
                        text = candidate.hindi_alternative_2(country, '30', count, retry=retry)
                        self.assertTrue(text.startswith(original + ' '))
                        self.assertIn('केवल एक पंक्ति', text)
                        self.assertIn('मित्रता की जोड़ियाँ नहीं', text)
                        self.assertIn('नई पंक्ति', text)

    def test_strict_parser_rejects_pairs_without_graph_mutation(self):
        graph = nx.empty_graph([str(i) for i in range(50)])
        for response in ('6,29\n6,41', '6,29,6,41', '6,6', '6,30', '6,99', '6'):
            with self.subTest(response=response), self.assertRaises(ValueError):
                protocol.parse_response('local', response, graph, curr_pid='30', num_choices=2)
            self.assertEqual(graph.number_of_edges(), 0)
        protocol.parse_response('local', '6,29', graph, curr_pid='30', num_choices=2)
        self.assertEqual(set(graph['30']), {'6', '29'})

    def test_invalid_counts_are_not_silently_rephrased(self):
        for count in (True, 0, -1, 50, 2.0, '2', None):
            with self.subTest(count=count), self.assertRaises(ValueError):
                candidate.hindi_alternative_2('us', '30', count)


if __name__ == '__main__':
    unittest.main()
