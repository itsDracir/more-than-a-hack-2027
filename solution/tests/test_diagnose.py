import unittest

import numpy as np

from solution.diagnose import half_report
from solution.tests.synthetic import FPS, PITCH, synthetic_match


class HalfReportTest(unittest.TestCase):

    def test_report_holds_only_aggregates(self):
        features, events, blocks, _ = synthetic_match()
        report = half_report(features, events, blocks[0], PITCH, FPS)
        for key in ("speed_threshold", "speed_quantiles", "dead_share", "dead_ball_moving_share",
                    "alive_ball_moving_share", "restarts_found", "set_piece_windows"):
            self.assertIn(key, report)
        # Al partit sintètic les aturades són de veritat: la pilota gairebé no s'hi mou.
        self.assertLess(report["dead_ball_moving_share"], report["alive_ball_moving_share"])
        for value in report.values():
            self.assertFalse(isinstance(value, np.ndarray) and value.size > 20)


if __name__ == "__main__":
    unittest.main()
