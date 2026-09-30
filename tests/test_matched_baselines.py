import unittest
import networkx as nx
from make_matched_baselines import matched_baselines


class BaselineTests(unittest.TestCase):
    def test_controls_match_size_density_and_degree_contract(self):
        graph = nx.cycle_graph([str(i) for i in range(8)])
        personas = {node: dict(gender='a', age=20+int(node), religion='b',
                              **{'race/ethnicity':'c', 'political affiliation':'d'}) for node in graph}
        for name, baseline, mixed in matched_baselines(graph, personas):
            self.assertEqual(set(baseline), set(graph))
            self.assertEqual(baseline.number_of_edges(), graph.number_of_edges())
            self.assertEqual(nx.number_of_selfloops(baseline), 0)
            if name == 'degree_preserving_rewire':
                self.assertEqual(dict(graph.degree()), dict(baseline.degree()))
        with self.assertRaises(ValueError):
            matched_baselines(graph.to_directed(), personas)


if __name__ == '__main__':
    unittest.main()
