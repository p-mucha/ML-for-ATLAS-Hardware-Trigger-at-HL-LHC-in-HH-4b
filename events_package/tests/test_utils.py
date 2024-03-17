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

        self.example_2d_array = np.array([[1, 2, 3, 4, 5], [0, 7, 8, 7, 8]])
        self.example_2d_list = [[8, 5, 7, 2, 1], [5, 7, 3, 7, 1]]

        self.example_2d_array2 = np.array([[1, 2, 4, 5], [0, 7, 7, 8]])
        self.example_2d_list2 = [[8, 5, 2, 1], [5, 7, 7, 1]]

    def test_fold_list_2d(self):
        """
        fold_list_2d is supposed to return difference between elements symmetric
        with respect to middle. If there is odd number of columns, then first column of output is the
        middle column.
        """
        arr1 = np.array([[3.0, 2.0, 4.0], [8.0, 0.0, 8.0]])
        arr2 = utils.fold_list_2d(self.example_2d_array)
        np.testing.assert_almost_equal(
            arr1,
            arr2,
            decimal=4,
        )

        arr3 = np.array([[7.0, -3.0, -7.0], [3.0, 0.0, -4.0]])
        arr4 = utils.fold_list_2d(self.example_2d_list)

        np.testing.assert_almost_equal(
            arr3,
            arr4,
            decimal=4,
        )

        arr5 = np.array([[2.0, 4.0], [0.0, 8.0]])
        arr6 = utils.fold_list_2d(self.example_2d_array2)
        np.testing.assert_almost_equal(
            arr5,
            arr6,
            decimal=4,
        )

        arr7 = np.array([[-3.0, -7.0], [0.0, -4.0]])
        arr8 = utils.fold_list_2d(self.example_2d_list2)

        np.testing.assert_almost_equal(
            arr7,
            arr8,
            decimal=4,
        )

    def test_plot_avg(self):

        # the code below should split data into batches and from each calculate RMS
        # for interval=5, there should be 3 batches: two batch containing 2 points
        # last two batches containing 5 points each

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

        self.assertAlmostEqual(y_e[0], 11.40175425099138, places=7)
        self.assertAlmostEqual(y_e[1], 9.715966241192895, places=7)
        self.assertAlmostEqual(y_e[2], 11.949895397031725, places=7)

        self.assertAlmostEqual(x_e[0], 0.5, places=7)
        self.assertAlmostEqual(x_e[1], 4.0, places=7)
        self.assertAlmostEqual(x_e[2], 9.0, places=7)

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
        # two batches: one containing 1 data point and second with 11 data points

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

        self.assertAlmostEqual(y_e[0], 8.0, places=7)
        self.assertAlmostEqual(y_e[1], 11.208762805785643, places=7)

        self.assertAlmostEqual(x_e[0], 0.0, places=7)
        self.assertAlmostEqual(x_e[1], 6.0, places=7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
