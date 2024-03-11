import numpy as np
import pandas as pd
import unittest

from events_package.config import FIVE_LAYERS
from events_package.Experiment import Experiment


class ExperimentTest(unittest.TestCase):
    def __init__(self, methodName="runTest"):
        super().__init__(methodName=methodName)
        self.example_df = pd.read_csv("example_df.csv")
        self.expected_tot_layers_et = np.array(
            [
                35412.74120516001,
                20649.991099360002,
                39266.697150570006,
                48459.51081149,
                36653.324367179994,
                11562.17414904,
                3000.5171057700004,
                25180.052962479996,
                34426.49757961,
                10716.471099829998,
                11768.08898671,
                45959.34261785,
                12386.42750955,
                13160.41585887,
                21800.22897859,
                19855.255351600004,
                46073.694318140006,
                31902.20432428999,
                24624.151211829994,
                48529.58530065,
                39812.487332660006,
                7158.523111360001,
                40759.673651400015,
                23577.99202559,
                45586.59705697,
            ]
        )

    def setUp(self):
        self.example_experiment = Experiment(self.example_df, config=FIVE_LAYERS)

    def test_length_property(self):
        # check if the method outputs correct length of the dataset
        self.assertEqual(self.example_experiment.length, 25, "")

    def test_z_attribute(self):
        # check if the method outputs correct z values
        np.testing.assert_array_equal(
            self.example_experiment.z.values, self.example_df["z"].values
        )

    def test_tot_layers_et(self):
        # check for two example indices
        self.assertAlmostEqual(
            self.example_experiment.tot_layers_et(index=0), 35412.74120516001, places=4
        )
        self.assertAlmostEqual(
            self.example_experiment.tot_layers_et(index=1), 20649.991099360002, places=4
        )

        # test if without index it returns correct array
        self.assertEqual(len(self.example_experiment.tot_layers_et(index=None)), 25)
        np.testing.assert_almost_equal(
            self.example_experiment.tot_layers_et(),
            self.expected_tot_layers_et,
            decimal=4,
        )

    def test_remove_duplicates(self):
        # check if the remove_duplicates method works as expected:
        # it should look at which events (rows) have the same z , et and eta,
        # of those events that have the same values, only the one with highest total energy in cells should be kept
        # in example_df, this happens for event_no == 13, which is at row 5 and 6
        # in this example, row 6 has lower total energy in layers so it should be removed
        self.assertEqual(len(np.where(self.example_experiment.event_no == 13)[0]), 2)
        self.example_experiment.remove_duplicates(shuffle=False)

        # check if after removing:
        # there is only one row with event_no == 13
        # there are 24 rows in dataset (1 less than the original 25)
        # the value of total energy from deleted event does not occur in data
        self.assertEqual(len(np.where(self.example_experiment.event_no == 13)[0]), 1)
        self.assertEqual(self.example_experiment.length, 24)

        not_close = ~np.isclose(
            self.example_experiment.tot_layers_et(), 3000.51710577, atol=1e-4
        )
        self.assertTrue(np.all(not_close))


if __name__ == "__main__":
    unittest.main(verbosity=2)
