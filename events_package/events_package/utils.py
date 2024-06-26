import math
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import scienceplots
from scipy.optimize import curve_fit
from xgboost import Booster


def calculate_confidence_interval(errors_list, cl):
    lim1 = np.percentile(errors_list, 50 + cl / 2)
    lim2 = np.percentile(errors_list, 50 - cl / 2)

    return (lim1 - lim2) / 2


def calculate_confidence_range(errors_list, cl):
    if np.min(errors_list) == 0:
        lim = np.percentile(errors_list, cl)
        return (0, lim)

    elif np.max(errors_list) == 0:
        lim = np.percentile(errors_list, 100 - cl)
        return (lim, 0)

    else:
        lim1 = np.percentile(errors_list, 50 + cl / 2)
        lim2 = np.percentile(errors_list, 50 - cl / 2)
        ran = (lim1 - lim2) / 2
        return (-ran, ran)


def plot_predictions(Y_test, Y_pred, binnum=None, figsize=None):
    if binnum == None:
        binnum = 150
    if figsize == None:
        figsize = (7, 5)

    with plt.style.context(["science", "notebook", "grid"]):
        plt.figure(figsize=figsize, dpi=150)
        common_bins = np.linspace(
            min(np.min(Y_pred), np.min(Y_test)),
            max(np.max(Y_pred), np.max(Y_test)),
            binnum,
        )

        # histograms with common bin edges
        plt.hist(Y_test, bins=common_bins, density=False, histtype="step", label="True")

        plt.hist(
            Y_pred,
            bins=common_bins,
            density=False,
            histtype="step",
            label="Predicted",
            color="orange",
        )

        plt.xlabel("z [mm]")
        plt.ylabel("Occurances")
        plt.legend()
        plt.title("Distributions of True and Predicted z")
        plt.show()


def plot_errors(
    Y_test,
    Y_pred,
    log_yscale=False,
    conficence_level=None,
    xmax=None,
    xcut=None,
    binnum=None,
):
    with plt.style.context(["science", "notebook", "grid"]):
        plt.figure(figsize=(7, 5), dpi=150)
        prediction_errors = Y_test - Y_pred.flatten()

        rms = np.sqrt(np.mean(np.square(prediction_errors)))

        # sigma level
        if conficence_level is not None:
            cl = conficence_level
            confidence_interval = calculate_confidence_interval(prediction_errors, cl)
            cl = str(cl)

        else:  # 90 by default
            confidence_interval = calculate_confidence_interval(prediction_errors, 90)
            cl = "90"

        if xcut is not None:
            x_cut = xcut
            prediction_errors = np.array(
                [error for error in prediction_errors if np.abs(error) <= x_cut]
            )

        if binnum is not None:
            bin_num = binnum
        else:
            bin_num = 100

        # plot fewer bins if the scale is log
        if log_yscale == True:
            n, bins, patches = plt.hist(
                prediction_errors,
                bins=bin_num,
                density=False,
                histtype="step",
                label="Errors",
            )
        else:
            n, bins, patches = plt.hist(
                prediction_errors,
                bins=bin_num,
                density=False,
                histtype="step",
                label="Errors",
                color="k",
            )

        # find fwhm
        max_bin_index = np.argmax(n)
        max_bin_center = (bins[max_bin_index] + bins[max_bin_index + 1]) / 2
        half_max_value = max(n) / 2
        left_idx = np.where(n[:max_bin_index] < half_max_value)[0][-1]
        right_idx = np.where(n[max_bin_index:] < half_max_value)[0][0] + max_bin_index
        fwhm = bins[right_idx] - bins[left_idx]

        if log_yscale == True:
            plt.yscale("log")

            # Plot RMS, FWHM, and cl% range as text
            plt.text(
                0.6 * np.max(bins),
                0.2 * np.max(n),
                f"RMS: {rms:.2f}",
                fontsize=16,
                color="red",
            )
            plt.text(
                0.6 * np.max(bins),
                0.05 * np.max(n),
                f"FWHM: {fwhm:.2f}",
                fontsize=16,
                color="blue",
            )
            plt.text(
                0.6 * np.max(bins),
                0.005 * np.max(n),
                cl + f"%: {confidence_interval:.1f}",
                fontsize=16,
                color="green",
            )

        elif xmax is not None:
            low_lim = max(-xmax, np.min(prediction_errors))
            plt.text(
                0.96 * low_lim,
                0.94 * np.max(n),
                f"RMS: {rms:.3f}",
                fontsize=14,
                color="black",
                bbox=dict(
                    facecolor="white", edgecolor="black", boxstyle="square,pad=0.5"
                ),
            )

        else:
            # Plot RMS, FWHM, and cl% range as text
            plt.text(
                0.6 * 15, 0.7 * np.max(n), f"RMS: {rms:.3f}", fontsize=16, color="red"
            )
            plt.text(
                0.6 * 15,
                0.6 * np.max(n),
                f"FWHM: {fwhm:.3f}",
                fontsize=16,
                color="blue",
            )
            plt.text(
                0.6 * 15,
                0.5 * np.max(n),
                cl + f"%: {confidence_interval:.3f}",
                fontsize=16,
                color="green",
            )
            plt.xlim(-20, 20)

        plt.title("Distribution of Model Prediction Errors")
        plt.xlabel("True - Predicted z [mm]")
        plt.ylabel("Occurances")
        plt.show()


def plot_corelation(Y_test, Y_pred, density=True, log_density=True, plot_line=True):
    with plt.style.context(["science", "notebook", "grid"]):
        plt.figure(figsize=(9, 6), dpi=150)

        plt.xlabel("True z [mm]")
        plt.ylabel("Predicted z [mm]")
        correlation_coefficient = np.corrcoef(
            Y_pred.reshape(
                Y_pred.shape[0],
            ),
            Y_test,
        )[0, 1]

        if density == False:
            plt.plot(Y_test, Y_pred, ".")

        elif log_density == False:
            plt.hist2d(Y_test, Y_pred, bins=100, cmap="viridis")
            plt.colorbar(label="Occurances")

        else:
            plt.hist2d(Y_test, Y_pred, bins=100, cmap="viridis", norm=LogNorm())
            plt.colorbar(label="Occurances")

        # Add text with box background
        text_location_x = 0.925 * np.min(Y_test)
        text_location_y = 0.67 * np.max(Y_pred)
        text = f"r = {correlation_coefficient:.3f}"

        # Create a rectangle patch
        bbox_props = dict(
            boxstyle="square,pad=0.3", fc="white", ec="black", lw=1
        )  # Adjusted lw parameter
        plt.text(
            text_location_x,
            text_location_y,
            text,
            fontsize=14,
            color="black",
            bbox=bbox_props,
        )

        if plot_line == True:
            plt.plot(
                np.sort(Y_test)[[0, -1]],
                np.sort(Y_test)[[0, -1]],
                "--",
                color="black",
                label="y = x",
            )
            plt.legend(loc="upper left")

        plt.title("2d Histogram of Predicted vs True z")
        plt.show()


def count_nodes(model):
    if isinstance(model, Booster):
        trees = model.trees_to_dataframe()
    else:
        trees = model.get_booster().trees_to_dataframe()

    return sum(trees["Split"].notna())


def fold_list(input_list):
    n = len(input_list)

    if n % 2 == 1:
        output_array = np.zeros(int((n / 2) + 1))
        mid = int((n - 1) / 2)
        middle_element = input_list[mid]

        part1 = np.array(input_list[:mid])[::-1]
        part2 = np.array(input_list[mid + 1 :])

        assert len(part1) == len(part2)
        assert len(output_array) == len(part1) + 1

        output_array[0] = middle_element
        output_array[1:] = part2 - part1

    else:
        output_array = np.zeros(int(n / 2))

        part1 = np.array(input_list[: int(n / 2)])[::-1]
        part2 = np.array(input_list[int(n / 2) :])

        assert len(part1) == len(part2)
        assert len(output_array) == len(part1)

        output_array[:] = part2 - part1

    return output_array


def fold_list_2d(input_array):
    # handle list input
    if isinstance(input_array, list):
        input_array = np.array(input_array)

    n = input_array.shape[1]
    height = input_array.shape[0]

    if n % 2 == 1:
        output_array = np.zeros((height, int((n / 2) + 1)))
        mid = int((n - 1) / 2)
        middle_col = input_array[:, mid]

        part1 = np.array(input_array[:, :mid])[:, ::-1]
        part2 = np.array(input_array[:, mid + 1 :])

        assert part1.shape == part2.shape
        assert output_array.shape[1] == part1.shape[1] + 1

        output_array[:, 0] = middle_col
        output_array[:, 1:] = part2 - part1

    else:
        output_array = np.zeros((height, int(n / 2)))

        part1 = np.array(input_array[:, : int(n / 2)])[:, ::-1]
        part2 = np.array(input_array[:, int(n / 2) :])

        assert part1.shape == part2.shape
        assert output_array.shape[1] == part1.shape[1]

        output_array[:] = part2 - part1

    return output_array


def plot_avg(
    x_values,
    y_values,
    interval=1000,
    xlabel=None,
    ylabel=None,
    title=None,
    abs=True,
    rms=False,
    xlim=None,
    ylim=None,
    return_values=True,
    return_x_u=False,
    plot=True,
):
    indices = np.argsort(x_values)
    x_sorted = x_values[indices]

    if abs:
        y_sorted = np.abs(y_values)[indices]
    else:
        y_sorted = y_values[indices]

    df = pd.DataFrame({"x": x_sorted[::-1], "y": y_sorted[::-1]})

    y_avg = []
    x_points = []
    x_range_low = []
    x_range_high = []
    vertical_uncertainty = []

    for batch in range(0, len(df), interval):
        x_batch = df["x"].iloc[batch : batch + interval]
        y_batch = df["y"].iloc[batch : batch + interval]

        if rms is False:
            y_avg.append(y_batch.mean())
            vertical_uncertainty.append(y_batch.std() / np.sqrt(len(y_batch)))

        else:
            # y_avg.append(np.sqrt((np.mean(y_batch**2))))
            # uncertainty of RMS can be calculated:
            # first we have some mean of squares, this mean (m) will
            # have its uncertainty of mean (u), which is std / sqrt(N)
            # then RMS is actually sqrt(m), so its uncertainty is propagated as
            # u / (2 * sqrt(m))

            m = (y_batch**2).mean()
            u = (y_batch**2).std() / np.sqrt(len(y_batch))

            new_u = u / (2 * np.sqrt(m))

            y_avg.append(np.sqrt(m))
            vertical_uncertainty.append(new_u)

        x_points.append(x_batch.mean())

        x_range_low.append(x_batch.min())
        x_range_high.append(x_batch.max())

    x_points = np.array(x_points)
    x_range_low = np.array(x_range_low)
    x_range_high = np.array(x_range_high)
    vertical_uncertainty = np.array(vertical_uncertainty)

    if plot:
        plt.figure(figsize=(10, 6))
        plt.errorbar(
            x_points,
            y_avg,
            xerr=[x_points - x_range_low, x_range_high - x_points],
            yerr=vertical_uncertainty,
            fmt=".",
            label=f"({interval}-interval)",
            color="black",
        )

        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

        plt.title(title)
        plt.xlabel(xlabel)

        if ylabel is not None:
            plt.ylabel(ylabel)
        else:
            plt.ylabel("Absolute Errors (in z) [mm]")

        plt.legend()
        plt.grid()
        plt.show()

    if return_values:
        if return_x_u:
            return (
                x_points[::-1],
                y_avg[::-1],
                [(x_points - x_range_low)[::-1], (x_range_high - x_points)[::-1]],
                vertical_uncertainty[::-1],
            )
        else:
            return x_points[::-1], y_avg[::-1], vertical_uncertainty[::-1]


#########################################################################
def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-0.5 * ((x - mean) / stddev) ** 2)


def decimal_places_to_significant_figure(number):
    if number != 0:
        abs_number = abs(number)
        log_number = math.log10(abs_number)
        integer_part = math.floor(log_number)
        if integer_part < 0:
            decimal_places = abs(integer_part)
        else:
            decimal_places = -integer_part - 1

    else:
        decimal_places = -1  # handle 0 case
    return decimal_places


def round_up_correctly(value, uncertainty, scientific=True):
    num = decimal_places_to_significant_figure(uncertainty)
    if num < 0:
        num = num + 1

    if scientific is not True:
        return round(value, num), round(uncertainty, num)

    else:
        value, uncertainty = round(value, num), round(uncertainty, num)

        if value == 0:
            num = 1
        else:
            num = int(num + round(np.log10(np.abs(value)), 0))
        return format(value, f".{num}e"), format(uncertainty, f".{num}e")


def get_smallest_ix(slupki):
    minh = 10e100
    mix = -1
    for ix, (h, c, w) in enumerate(slupki):
        if h < minh:
            minh = h
            mix = ix
    return mix


def merge_with_lesser_neighbor(ix, slupki):
    if len(slupki) == 1:
        return slupki
    elif ix == 0 or (len(slupki) - 1 > ix > 0 and slupki[ix + 1] <= slupki[ix - 1]):
        r = slupki[ix + 1]
        slupki[ix][0] += r[0]
        slupki[ix][1] = (r[1] * r[2] + slupki[ix][1] * slupki[ix][2]) / (
            slupki[ix][2] + r[2]
        )  # center of mass
        slupki[ix][2] += r[2]
        slupki.pop(ix + 1)
    elif ix == len(slupki) - 1 or (
        len(slupki) - 1 > ix > 0 and slupki[ix + 1] > slupki[ix - 1]
    ):
        l = slupki[ix - 1]
        slupki[ix][0] += l[0]
        slupki[ix][1] = (l[1] * l[2] + slupki[ix][1] * slupki[ix][2]) / (
            slupki[ix][2] + l[2]
        )
        slupki[ix][2] += l[2]
        slupki.pop(ix - 1)
    return slupki


def connect_bins(heights, centers, widths):
    slupki = [[h, c, w] for h, c, w in zip(heights, centers, widths)]

    while slupki[get_smallest_ix(slupki)][0] < 5 and len(slupki) > 1:
        slupki = merge_with_lesser_neighbor(get_smallest_ix(slupki), slupki)

    return np.array(slupki).T


def fit_gaussian_to_data(
    data,
    initial_guesses,
    binnum,
    plot=False,
    maxfev=None,
    x_label=None,
    y_label=None,
    title=None,
    combine_bins=True,
    binnum_correction=False,
    xlim=None,
    ylim=None,
    return_params=True,
    plot_rms=True,
):
    rms = np.sqrt(np.mean(data**2))

    if binnum_correction:
        # Adjust number of bins based on the data range
        bin_num = int(3 * (np.max(data) - np.min(data)) / 5)
        if bin_num > binnum:
            binnum = bin_num

    hist_values, bin_edges = np.histogram(data, bins=binnum)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    bin_heights = hist_values

    bin_widths = np.diff(bin_edges)

    if combine_bins == True:
        # merge bins so that each has at least 5 occurances
        bin_heights, bin_centers, bin_widths = connect_bins(
            bin_heights, bin_centers, bin_widths
        )

    vertical_uncertainties = np.sqrt(bin_heights)

    maxfev_value = 800
    if maxfev is not None:
        maxfev_value = maxfev

    if len(bin_centers) < 5:
        print("bin centers: ", bin_centers)
        print("bin heights: ", bin_heights)
    params, pcov = curve_fit(
        gaussian,
        bin_centers,
        bin_heights,
        p0=initial_guesses,
        sigma=vertical_uncertainties,
        maxfev=maxfev_value,
    )

    amp, mean, stddev = params
    stddev = np.abs(stddev)

    perr = np.sqrt(np.diag(pcov))

    x = np.linspace(np.min(data), np.max(data), 100000)
    fitted_curve = gaussian(x, amp, mean, stddev)

    if plot is not False:
        plt.figure()
        m, u_m = round_up_correctly(mean, perr[1])
        s, u_s = round_up_correctly(stddev, perr[2])
        plt.plot(
            x,
            fitted_curve,
            "r-",
            label="Fit (mean="
            + str(m)
            + "±"
            + str(u_m)
            + "; "
            + "stddev="
            + str(s)
            + "±"
            + str(u_s)
            + ")",
        )

        plt.bar(
            bin_centers,
            bin_heights,
            width=bin_widths,
            align="center",
            color="blue",
            edgecolor="black",
        )

        plt.errorbar(
            bin_centers,
            bin_heights,
            yerr=vertical_uncertainties,
            fmt="none",
            ecolor="r",
            elinewidth=1,
            capsize=2,
        )

        if plot_rms:
            # Display RMS as text on the plot
            plt.text(
                0.1,
                0.85,
                f"RMS: {rms:.3f}",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
                bbox=dict(facecolor="white", alpha=0.7),
            )

        plt.xlabel(x_label)

        if y_label is not None:
            plt.ylabel(y_label)
        else:
            plt.ylabel("Occurances")

        if title is not None:
            plt.title(title)
        else:
            plt.title("Histogram of data")

        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.grid()
        plt.legend()
        plt.show()

    if return_params:
        return params, perr


def R_emb1(eta):
    if np.abs(eta) < 0.8:
        R_l1 = 1558.859292 - 4.990838 * np.abs(eta) - 21.144279 * (np.abs(eta)) ** 2
    else:
        R_l1 = 1522.775373 + 27.970192 * np.abs(eta) - 21.104108 * (np.abs(eta)) ** 2

    return R_l1


def R_emb2(eta):
    return 1698.990944 - 49.431767 * np.abs(eta) - 24.504976 * (np.abs(eta)) ** 2


def eta_to_theta(eta):
    return 2 * np.arctan(np.exp(-eta))


def par_to_det_emb1(z, eta_particle, eta_cell):
    theta_particle = eta_to_theta(eta_particle)
    R = R_emb1(eta_cell)
    theta_d = np.arctan(R / (z + (R / np.tan(theta_particle))))
    if np.tan(theta_d / 2) >= 0:
        eta_d = -np.log(np.abs(np.tan(theta_d / 2)))

    else:
        eta_d = np.log(np.abs(np.tan(theta_d / 2)))

    return eta_d


def par_to_det_emb2(z, eta_particle, eta_cell):
    theta_particle = eta_to_theta(eta_particle)
    R = R_emb2(eta_cell)
    theta_d = np.arctan(R / (z + (R / np.tan(theta_particle))))
    if np.tan(theta_d / 2) >= 0:
        eta_d = -np.log(np.abs(np.tan(theta_d / 2)))

    else:
        eta_d = np.log(np.abs(np.tan(theta_d / 2)))

    return eta_d


def calc_z(eta_l1, eta_l2, eta_cell1, eta_cell2):
    R_l1 = R_emb1(eta_cell1)

    R_l2 = R_emb2(eta_cell2)

    theta_l1 = eta_to_theta(eta_l1)
    theta_l2 = eta_to_theta(eta_l2)

    z_l1 = R_l1 / np.tan(theta_l1)
    z_l2 = R_l2 / np.tan(theta_l2)

    z_vtx = (z_l1 * R_l2 - z_l2 * R_l1) / (R_l2 - R_l1)

    return z_vtx


def par_to_det_emb1_new(z, eta_particle, eta_cell):
    eta_new = eta_cell

    for ix in range(3):
        eta_new = par_to_det_emb1(z=z, eta_particle=eta_particle, eta_cell=eta_new)

    return eta_new


def par_to_det_emb2_new(z, eta_particle, eta_cell):
    eta_new = eta_cell

    for ix in range(3):
        eta_new = par_to_det_emb2(z=z, eta_particle=eta_particle, eta_cell=eta_new)

    return eta_new


def save_table_df(dataframe, filename):
    current_directory = os.getcwd()

    # all tables should be saved to 'tables'
    tables_directory = os.path.join(current_directory, "..", "tables")

    # ensure that the "tables" directory exists, if not, create it
    if not os.path.exists(tables_directory):
        os.makedirs(tables_directory)

    # full path
    csv_path = os.path.join(tables_directory, filename)

    # save to csv in 'tables' directory
    dataframe.to_csv(csv_path, index=False)


def QuadLinearEncoding(arr):
    def quant_scheme(x):
        if x < 0:
            return 0

        if x < 8000:
            return int(x / 31.25)

        elif x < 40000:
            return 192 + int(x / 125)

        elif x < 168000:
            return 432 + int(x / 500)

        elif x < 678000:
            return 686 + int(x / 2000)

        else:
            return 1023

    quantised = np.vectorize(quant_scheme)(arr).astype(np.int16)

    assert np.all(quantised.shape == arr.shape)

    return quantised


def create_mapping(unique_elements_array):
    mapping_dict = {value: idx for idx, value in enumerate(unique_elements_array)}

    return mapping_dict


def closest_value(input_value, array):
    closest = None
    min_difference = float("inf")  # Initialize with positive infinity

    for value in array:
        difference = abs(input_value - value)
        if difference < min_difference:
            min_difference = difference
            closest = value

    return closest


def convert_values_to_float16(obj):
    """Rounds and converts to float16."""
    if isinstance(obj, dict):
        return {key: np.float16(round(value, 5)) for key, value in obj.items()}
    else:
        return np.float16(round(obj, 5))


def import_bin_functions():
    """Import mappings.pkl from quantisations directory."""
    current_directory = os.getcwd()
    quantisations_directory = os.path.join(current_directory, "..", "quantisations")

    with open(quantisations_directory + "\\bin_functions.pkl", "rb") as f:
        bin_functions = pickle.load(f)

    return bin_functions


def import_mappings():
    """Import mappings.pkl from quantisations directory."""
    current_directory = os.getcwd()
    quantisations_directory = os.path.join(current_directory, "..", "quantisations")

    with open(quantisations_directory + "\\mappings.pkl", "rb") as f:
        mappings = pickle.load(f)

    return mappings
