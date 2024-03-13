import numpy as np
import unittest

import events_package.utils as utils


class UtilsTest(unittest.TestCase):
    """
    Test cases for the utility functions in the utils module.
    """

    def __init__(self, methodName="runTest"):
        super().__init__(methodName)
        # example data
        self.values_y = np.array([8, 14, 9, 7, 11, 11, 10, 7, 10, 15, 14, 12])
        self.values_x = np.arange(len(self.values_y))

    def test_plot_avg(self):

        # the code below should split data into batches and from each calculate RMS
        # for interval=5, there should be 3 batches: two first containing 5 values each and
        # the last one containing 2 last values

        x_e, y_e, x_u_e, u_e = utils.plot_avg(
            x_values=self.values_x,
            y_values=self.values_y,
            interval=5,
            rms=True,
            return_values=True,
            ylabel="rms",
            return_x_u=True,
            plot=False,
        )

        self.assertAlmostEqual(y_e[0], 10.10940156488009, places=7)
        self.assertAlmostEqual(y_e[1], 10.908712114635714, places=7)
        self.assertAlmostEqual(y_e[2], 13.038404810405298, places=7)

        self.assertAlmostEqual(x_e[0], 2.0, places=7)
        self.assertAlmostEqual(x_e[1], 7.0, places=7)
        self.assertAlmostEqual(x_e[2], 10.5, places=7)

        # in the example below, interval larger than entire data is chosen,
        # there are 12 data points but interval (batch size) is 15
        # the function should include all data and calculate RMS from all of it

        x_e, y_e, x_u_e, u_e = utils.plot_avg(
            x_values=self.values_x,
            y_values=self.values_y,
            interval=15,
            rms=True,
            return_values=True,
            ylabel="rms",
            return_x_u=True,
            plot=False,
        )

        self.assertAlmostEqual(y_e[0], 10.977249200050075, places=7)

        self.assertAlmostEqual(x_e[0], 5.5, places=7)

        # in the example below, interval is chosen such that there will be
        # two batches: one containing 11 data points and one with just a single point

        x_e, y_e, x_u_e, u_e = utils.plot_avg(
            x_values=self.values_x,
            y_values=self.values_y,
            interval=11,
            rms=True,
            return_values=True,
            ylabel="rms",
            return_x_u=True,
            plot=False,
        )

        self.assertAlmostEqual(y_e[0], 10.87950533634854, places=7)
        self.assertAlmostEqual(y_e[1], 12.0, places=7)

        self.assertAlmostEqual(x_e[0], 5.0, places=7)
        self.assertAlmostEqual(x_e[1], 11.0, places=7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
