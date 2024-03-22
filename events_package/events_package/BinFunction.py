import numpy as np

import events_package.utils as utils


# Define a class for creating bin functions
class BinFunction:
    def __init__(self, arr, bins=64, cl=99.8):
        self.bin_edges = self._compute_bin_edges(arr, bins, cl)

    def _compute_bin_edges(self, arr, bins, cl):
        # Compute histogram
        ran1, ran2 = utils.calculate_confidence_range(arr, cl=cl)
        _, bin_edges = np.histogram(arr, bins=bins, range=(ran1, ran2))

        return bin_edges

    def bin_function(self, x):
        # Determine bin index for x
        if x <= self.bin_edges[0]:
            return 0
        elif x >= self.bin_edges[-1]:
            return len(self.bin_edges) - 2
        else:
            bin_index = np.searchsorted(self.bin_edges, x, side="right") - 1
            return bin_index
