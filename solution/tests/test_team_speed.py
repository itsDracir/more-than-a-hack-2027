import unittest
from io import StringIO

import numpy as np

from solution.team_speed import team_mean_speeds


class TeamMeanSpeedsTest(unittest.TestCase):

    def test_mean_speed_per_team_with_missing_slots(self):
        raw = StringIO(
            "1:0.1,0.1,2.0;0.2,0.2,4.0;0.8,0.8,1.0;NaN:0.5,0.5,3.0\n"
            "2:NaN;NaN;0.8,0.8,1.5;0.7,0.7,2.5:NaN\n"
        )
        speeds = team_mean_speeds(raw, team_by_slot=[0, 0, 1, 1], n_frames=2)
        np.testing.assert_allclose(speeds[0], [3.0, 1.0])
        self.assertTrue(np.isnan(speeds[1, 0]))
        np.testing.assert_allclose(speeds[1, 1], 2.0)

    def test_frame_ids_place_rows(self):
        raw = StringIO("2:0.1,0.1,2.0;0.8,0.8,1.0:NaN\n")
        speeds = team_mean_speeds(raw, team_by_slot=[0, 1], n_frames=3)
        self.assertTrue(np.isnan(speeds[0]).all())
        np.testing.assert_allclose(speeds[1], [2.0, 1.0])

    def test_implausible_speeds_are_clipped(self):
        raw = StringIO("1:0.1,0.1,80.0;0.8,0.8,1.0:NaN\n")
        speeds = team_mean_speeds(raw, team_by_slot=[0, 1], n_frames=1)
        self.assertLessEqual(speeds[0, 0], 12.0)


if __name__ == "__main__":
    unittest.main()
