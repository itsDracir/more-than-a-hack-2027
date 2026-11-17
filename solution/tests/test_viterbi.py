import unittest

import numpy as np

from solution.viterbi import viterbi


class ViterbiTest(unittest.TestCase):

    def test_short_blip_does_not_switch(self):
        emissions = np.zeros((100, 2))
        emissions[:, 0] = 1.0
        emissions[50:53, 1] = 3.0          # a 3-frame touch by the other state
        path = viterbi(emissions, switch_cost=10.0)
        self.assertTrue((path == 0).all())

    def test_sustained_evidence_switches_where_it_starts(self):
        emissions = np.zeros((100, 2))
        emissions[:40, 0] = 1.0
        emissions[40:, 1] = 1.0
        path = viterbi(emissions, switch_cost=10.0)
        np.testing.assert_array_equal(path, [0] * 40 + [1] * 60)

    def test_switch_cost_can_be_a_matrix(self):
        # Leaving state 2 is free, entering it is free, 0 <-> 1 directly is prohibitive.
        cost = np.array([[0, 1e6, 0], [1e6, 0, 0], [0, 0, 0]], dtype=float)
        emissions = np.zeros((20, 3))
        emissions[:10, 0] = 1.0
        emissions[10:, 1] = 1.0
        path = viterbi(emissions, switch_cost=cost)
        # Has to pass through state 2 rather than jump 0 -> 1, and it costs nothing to do so on a
        # frame whose emission it gives up: the only way to avoid 1e6 is a 1-frame detour.
        self.assertNotIn((0, 1), set(zip(path[:-1].tolist(), path[1:].tolist())))

    def test_empty_input(self):
        self.assertEqual(len(viterbi(np.zeros((0, 2)), switch_cost=1.0)), 0)


if __name__ == "__main__":
    unittest.main()
