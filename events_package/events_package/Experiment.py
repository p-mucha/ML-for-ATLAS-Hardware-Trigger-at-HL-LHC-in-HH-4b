import logging
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import sys
import xgboost as xgb
from xgboost import XGBRegressor

# Configure logger to output to stdout
logging.basicConfig(
    stream=sys.stdout, level=logging.INFO, format="%(levelname)s: %(message)s"
)

import events_package.utils as utils
from events_package.config import Config

package_ver = "5.0"
# 5.0 - Changed Experiment class such that it uses configuration class, it does no longer need metaclass


class LabeledDataFrame:
    def __init__(self, dataframe, config):
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("Input must be a Pandas DataFrame")

        self.df = dataframe
        self.column_ranges = config.column_ranges

    def __getitem__(self, key):
        if isinstance(key, str):  # If key is a single column name
            if key in self.column_ranges:
                start, end = self.column_ranges[key]
                return self.df.loc[:, start:end]
            else:
                return self.df[key]
        elif isinstance(key, list):  # If key is a list of column names
            return self.df[key]
        else:
            raise TypeError(
                "Unsupported key type. Use a single column name or a list of column names."
            )

    def custom_select(self, key):
        if key in self.column_ranges:
            start, end = self.column_ranges[key]
            return self.df.loc[:, start:end]
        else:
            raise KeyError(f"Key '{key}' not found in column_ranges.")

    def __getattr__(self, attr):
        # delegate any other attribute access to the underlying DataFrame
        return getattr(self.df, attr)

    def __len__(self):
        return len(self.df)


class Experiment:
    __version__ = package_ver

    def __init__(self, dataframe, config):
        if not isinstance(config, Config):
            raise ValueError("configuration should be an object of the Config class")

        self.holder = {}
        self.dataset = dataframe
        self.config = config
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.model = None

        self.is_shuffled = False
        self.is_split = False

        self.layers = self.config.layers
        self.dimensions = self.config.dimensions
        self.noise_thresholds = self.config.noise_thresholds
        self.eta_granularity = self.config.eta_granularity
        self.phi_granularity = self.config.phi_granularity

        for col in self.config.columns:
            self.set_property(col, lambda self, col=col: self.dataset[col])

        for name, (start, end) in self.config.column_ranges.items():
            self.set_property(
                name, lambda self, s=start, e=end: self.dataset.loc[:, s:e]
            )

        self.generate_plot_methods()

    def __getattr__(self, name):
        return self.holder.get(name, lambda _: None)(self)

    def set_property(self, name, value):
        self.holder[name] = value

    def get_column_value(self, column):
        return self.dataset[column]

    @property
    def length(self):
        return len(self.dataset)

    def tot_layers_et(self, index=None):
        if index is not None:
            return np.sum(
                [
                    np.sum(getattr(self, layer + "_cells").loc[index].values)
                    for layer in self.layers
                ]
            )

        else:
            tot_energies = np.zeros(self.length)
            for layer in self.layers:
                tot_energies = tot_energies + np.sum(
                    getattr(self, layer + "_cells").values, axis=1
                )
            return tot_energies

    @property
    def ldf(self):
        return LabeledDataFrame(self.dataset, config=self.config)

    @property
    def training_ldf(self):
        return LabeledDataFrame(self.training_dataset, config=self.config)

    @property
    def testing_ldf(self):
        return LabeledDataFrame(self.testing_dataset, config=self.config)

    def start_end_layer(self, layer):
        """returns starting and ending column of given layer, eg for psb: psb_(0,0), psb_(6,8)
        remember column_ranges contains ranges of layer_cells, so _cells must be added
        """
        start_col = self.config.column_ranges[layer + "_cells"][0]
        end_col = self.config.column_ranges[layer + "_cells"][1]
        return start_col, end_col

    def layer_cells(self, layer):
        """returns cell values from given layer"""
        start_col, end_col = self.start_end_layer(layer)
        return self.dataset.loc[:, start_col:end_col]

    def translate_layers(self, layer):
        """returns names of all columns in given layer, eg for psb: psb_(0,0), psb_(0,1), ... , psb_(6,8)"""
        if layer not in self.layers:
            raise ValueError(
                "Incorrect layer name. Please remember layer does not include _cells"
            )
        else:
            start_col, end_col = self.start_end_layer(layer)

        return self.dataset.loc[:, start_col:end_col].columns.values

    def __str__(self):
        return f"Instance of Experiment class with {self.length} events"

    def __copy__(self):
        """Create a new instance of the class with copied data"""
        new_instance = Experiment(self.dataset.copy())
        return new_instance

    def set_noise_thresholds(self, thresholds_dict):
        """Changes self.noise_thresholds to a new value. Input should be a dictionary for example:
        {'psb': 100, 'emb1': 50, ...}"""
        self.noise_thresholds = thresholds_dict

    def add_tot_layers_et(self):
        """adds a column to self.dataset, containing total energies in all layers"""
        if "tot_layers_et" in self.dataset.columns.values:
            print("Total energies have already been added")
        else:
            df1 = pd.DataFrame({"tot_layers_et": self.tot_layers_et()})
            self.dataset = pd.concat([self.dataset, df1], axis=1)

    def remove_duplicates(self, columns=["z", "et", "eta"], shuffle=True):
        """
        Removes events for which given values occur more than once, by default checks for z, eta and et.
        Of all events with the same all values at columns, selects only the one with highest energy in layers.
        The default will remove true duplicates as well as events with more than one entry.
        """
        self.add_tot_layers_et()

        def remove_string_from_list(original_list, string_to_remove):
            return [item for item in original_list if item != string_to_remove]

        for col in columns:
            if col in self.layers:
                columns = remove_string_from_list(columns, col)
                columns = columns + self.translate_layers(col)

        # groupby by columns and pick from each group event with highest total energy in layers
        res_ix = self.dataset.groupby(columns)["tot_layers_et"].idxmax()

        self.dataset = self.dataset.loc[res_ix].reset_index(drop=True)
        if shuffle:
            # shuffle to randomise order
            self.shuffle_dataset(repeats=11)

    def remove_duplicates_old(self, columns=["z", "eta", "et"]):
        """
        Removes events for which given values occur more than once, by default checks for z, eta and et.
        The default will remove true duplicates as well as events with more than one entry.
        """

        def remove_string_from_list(original_list, string_to_remove):
            return [item for item in original_list if item != string_to_remove]

        for col in columns:
            if col in self.layers:
                columns = remove_string_from_list(columns, col)
                columns = columns + self.translate_layers(col)

        self.dataset = self.dataset[
            ~self.dataset.duplicated(subset=columns, keep=False)
        ]

    def remove_events(self, mask):
        """Only keeps events where mask == True"""
        self.dataset = self.dataset[mask].reset_index(drop=True)

    def et_cut(self, et_threshold=0, data="all"):
        """Delete events with abs(true et) lower than et_threshold. Data can be all, training or testing."""
        if data == "all":
            mask = self.dataset["et"] >= et_threshold
            self.dataset = self.dataset = self.dataset.loc[mask]

        elif data == "training":
            mask = self.testing_dataset["et"] >= et_threshold
            self.training_dataset = self.training_dataset.loc[mask]

        elif data == "testing":
            mask = self.testing_dataset["et"] >= et_threshold
            self.testing_dataset = self.testing_dataset.loc[mask]

        else:
            raise ValueError("Invalid value for 'data' parameter")

    def z_cut(self, z_threshold=130, data="all"):
        """Delete events with abs(true z) lower than z_threshold. Data can be all, training or testing."""
        if data == "all":
            mask = np.abs(self.dataset["z"]) <= z_threshold
            self.dataset = self.dataset.loc[mask]

        elif data == "training":
            mask = np.abs(self.testing_dataset["z"]) <= z_threshold
            self.training_dataset = self.training_dataset.loc[mask]

        elif data == "testing":
            mask = np.abs(self.testing_dataset["z"]) <= z_threshold
            self.testing_dataset = self.testing_dataset.loc[mask]

        else:
            raise ValueError("Invalid value for 'data' parameter")

    def denoisify(self):
        """
        Loop over layers, replacing cells which have energy smaller than threshold for this layer, with 0.
        """
        for layer, noise_threshold in self.noise_thresholds.items():
            start_col, end_col = self.start_end_layer(layer)
            self.dataset.loc[:, start_col:end_col] = np.where(
                self.dataset.loc[:, start_col:end_col] < noise_threshold,
                0,
                self.dataset.loc[:, start_col:end_col],
            )

    def shuffle_dataset(self, repeats=1, rndm_st=21):
        """Randomly reshuffle dataset."""
        if self.is_split:
            raise ValueError(
                "Dataset already split into training and testing datasets, further shuffling will mix up events"
            )

        for i in range(repeats):
            random_state = np.random.RandomState(seed=rndm_st)
            shuffled_index = random_state.permutation(self.dataset.index)
            self.dataset = self.dataset.loc[shuffled_index].reset_index(drop=True)

        self.is_shuffled = True

    def standard_procedure(self, repeats=11):
        self.remove_duplicates()
        logging.info("Removed duplicates")

        self.denoisify()
        logging.info("Denoisified the dataset")

        self.shuffle_dataset(repeats=repeats)
        logging.info("Shuffled dataset")

        logging.info(f"Number of events after removing duplicates: {self.length}")

        mask = self.tot_layers_et() > 0
        self.remove_events(mask=mask)
        logging.info("Removed events with 0 energy in layers after denoisifying")
        logging.info(
            f"Number of events after removing 0 energy (in calorimeters) events: {self.length}"
        )

    def train_test_split(self, get_X, get_Y=None, test_size=0.2, shuffle=False):
        """
        Splits dataset into training and testing dataset, creating training_dataset and testing_dataset attributes of the object.
        Split is controlled by the test_size input parameter.
        get_X input needs to be a callable, which takes dataset as an input and returns list of lists of model inputs.
        """

        # if not shuffled before, do it now
        if shuffle == True or self.is_shuffled == False:
            self.shuffle_dataset()

        ix = int(test_size * self.length)
        self.training_dataset = self.dataset.loc[ix:]
        self.testing_dataset = self.dataset.loc[:ix]

        self.X_train = get_X(self.training_ldf)
        self.X_test = get_X(self.testing_ldf)

        if get_Y == None:
            Y = self.dataset["z"]
            self.y_train = Y.loc[ix:]
            self.y_test = Y.loc[:ix]
        else:
            self.y_train = get_Y(self.training_ldf)
            self.y_test = get_Y(self.testing_ldf)

        self.is_split = True

    def train_xgboost_model(self, params):
        if self.X_train is None:
            raise ValueError(
                "X_train is not set. Please set X_train before proceeding."
            )

        self.model = XGBRegressor(**params, random_state=21)
        self.model.fit(self.X_train, self.y_train)

        self.y_pred = self.model.predict(self.X_test)

        mse = mean_squared_error(self.y_test, self.y_pred)
        logging.info(f"Trained XGBoost model; mean squared error: {mse}")
        return mse

    def test_model(self):
        """Tests current self.model on self.X_test"""
        dtest = xgb.DMatrix(self.X_test, label=self.y_test)
        self.y_pred = self.model.predict(dtest)
        mse = mean_squared_error(self.y_test, self.y_pred)
        print(f"y_pred length: {len(self.y_pred)}, model tested, MSE: {mse}")

    def xgboost_hyperparameter_scan(
        self, param_grid, num_rounds_grid, nodes_info=False
    ):
        column_names = ["MSE", "Num_rounds"]
        for param in param_grid[0]:
            column_names.append(param)

        if nodes_info:
            column_names.append("Nodes NO")

        column_values = []
        for num_rounds in num_rounds_grid:
            for params in param_grid:
                mse = self.train_xgboost_model(params, num_rounds)
                values_toappend = [mse, num_rounds] + [
                    params[param] for param in param_grid[0]
                ]

                if nodes_info:
                    values_toappend.append(utils.count_nodes(self.model))

                column_values.append(values_toappend)

        return pd.DataFrame(column_values, columns=column_names)

    def eta_ticks(self, eta_center, layer):
        """Eta coordinates of cells in layer, is used mostly for plotting."""
        if layer not in self.layers:
            raise ValueError(
                "Layer should be element of self.layers, it should end with _cells. For example: emb1_cells."
            )

        eta_shape = self.dimensions[layer][1]
        delta_eta = self.eta_granularity[layer]

        eta_ticks = np.linspace(
            eta_center - (eta_shape // 2) * delta_eta,
            eta_center + (eta_shape // 2) * delta_eta,
            eta_shape,
        )

        return eta_ticks

    def phi_ticks(self, phi_center, layer):
        """Phi coordinates of cells in layer, is used mostly for plotting."""
        if layer not in self.layers:
            raise ValueError(
                "Layer should be element of self.layers, it should end with _cells. For example: emb1_cells."
            )

        phi_shape = self.dimensions[layer][0]
        delta_phi = self.phi_granularity[layer]

        phi_ticks = np.linspace(
            phi_center - (phi_shape // 2) * delta_phi,
            phi_center + (phi_shape // 2) * delta_phi,
            phi_shape,
        )

        return phi_ticks

    def plot_layer(
        self,
        index,
        layer,
        figsize,
    ):
        if layer not in self.layers:
            raise ValueError("Layer should be psb, emb1, emb2, emb3 or hab1.")

        # dimensions of the sta x phi window, eg can be (7,9) for psb
        reshape_dims = self.dimensions[layer]

        # get cell energies and reshape
        data = getattr(self, layer + "_cells").values[index]
        data = data.reshape(reshape_dims)

        # Handle eta ticks
        eta_center = getattr(self, layer + "_eta")[index]
        eta_ticks = self.eta_ticks(eta_center=eta_center, layer=layer)
        eta_ticks = np.around(eta_ticks, decimals=3)

        # Handle phi ticks
        phi_center = getattr(self, layer + "_phi")[index]
        phi_ticks = self.phi_ticks(phi_center=phi_center, layer=layer)
        phi_ticks = np.around(phi_ticks, decimals=3)

        event_no = self.dataset["event_no"][index]

        plt.figure(figsize=figsize)
        plt.rcdefaults()
        plt.xlabel("eta")
        plt.ylabel("phi")
        plt.title(
            "Layer: "
            + layer
            + f"; Event NO: {int(event_no)}; Z = {self.z[index]:.2f} [mm]"
        )
        plt.imshow(data, cmap="viridis", interpolation=None, origin="upper")
        colorbar = plt.colorbar()
        colorbar.set_label("Energy [MeV]")  # Adding caption to colorbar

        # Modify the plt.xticks line to handle every second eta tick for 'emb1'
        if layer == "emb1":
            plt.xticks(np.arange(0, len(eta_ticks), step=3), eta_ticks[::3])
        else:
            plt.xticks(np.arange(len(eta_ticks)), eta_ticks)

        plt.yticks(np.arange(len(phi_ticks)), phi_ticks)
        plt.show()

    def generate_plot_methods(self):
        """Generate methods for plotting layers, for example plot_emb1, plot_emb2, etc."""
        for layer in self.layers:
            method_name = f"plot_{layer}"

            if layer == "emb1":
                figsize = (20, 3)
            else:
                figsize = (12, 8)

            setattr(
                self,
                method_name,
                lambda index=0, layer=layer, figsize=figsize: self.plot_layer(
                    index, layer, figsize=figsize
                ),
            )

    def plot_layers(self, index):
        """plots multiple layers as a single plot"""
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))

        for i, layer in enumerate(self.layers):
            if layer == "emb1":
                every_second_tick = True
            else:
                every_second_tick = False

            reshape_dims = self.dimensions[layer]

            # get cell energies and reshape
            data = getattr(self, layer + "_cells").values[index]
            data = data.reshape(reshape_dims)

            # Handle eta ticks
            eta_center = getattr(self, layer + "_eta")[index]
            eta_ticks = self.eta_ticks(eta_center=eta_center, layer=layer)
            eta_ticks = np.around(eta_ticks, decimals=3)

            if every_second_tick:
                eta_ticks = [
                    tick if idx % 2 == 0 else "" for idx, tick in enumerate(eta_ticks)
                ]

            # Handle phi ticks
            phi_center = getattr(self, layer + "_phi")[index]
            phi_ticks = self.phi_ticks(phi_center=phi_center, layer=layer)
            phi_ticks = np.around(phi_ticks, decimals=3)

            event_no = self.dataset["event_no"][index]

            # Calculate subplot position
            row = i // 3
            col = i % 3

            ax = axes[row, col]
            ax.set_xlabel("eta")
            ax.set_ylabel("phi")
            ax.set_title(
                "Layer " + layer + f"; Event NO: {event_no}; Z = {self.z[index]:.2f}"
            )
            img = ax.imshow(data, cmap="viridis", interpolation=None, origin="upper")
            ax.set_xticks(np.arange(len(eta_ticks)))
            ax.set_xticklabels(eta_ticks)
            ax.set_yticks(np.arange(len(phi_ticks)))
            ax.set_yticklabels(phi_ticks)
            ax.grid(False)
            fig.colorbar(img, ax=ax)

        # Set the empty subplot as invisible
        axes[1, 2].axis("off")

        plt.tight_layout()
        plt.show()

    def calculate_z(self, index=0):
        eta_emb1 = self.dataset["emb1_eta"].values[index]
        eta_emb2 = self.dataset["emb2_eta"].values[index]

        R_emb1 = utils.R_emb1(eta_emb1)
        R_emb2 = utils.R_emb2(eta_emb2)

        theta_emb1 = utils.eta_to_theta(eta_emb1)
        theta_emb2 = utils.eta_to_theta(eta_emb2)

        z_l1 = R_emb1 / np.tan(theta_emb1)
        z_l2 = R_emb2 / np.tan(theta_emb2)

        z_vtx = (z_l1 * R_emb2 - z_l2 * R_emb1) / (R_emb2 - R_emb1)

        return z_vtx

    def calculate_barycenter(
        self,
        layer,
        index=0,
        linear_weights=False,
        w_0=None,
    ):
        if layer not in self.layers:
            raise ValueError("Layer should be psb, emb1, emb2, emb3 or hab1.")

        phi_shape, eta_shape = self.dimensions[layer]

        data = getattr(self, layer + "_cells").values[index]
        data = data.reshape(phi_shape, eta_shape)

        w_0 = w_0 or 3.6

        eta_center = getattr(self, layer + "_eta")[index]
        phi_center = getattr(self, layer + "_phi")[index]

        eta_ticks = self.eta_ticks(eta_center=eta_center, layer=layer)

        phi_ticks = self.phi_ticks(phi_center=phi_center, layer=layer)

        mask = data > -1
        total_et = np.sum(data[mask])

        weights = (
            data[mask]
            if linear_weights
            else np.maximum(w_0 + np.log(data[mask] / total_et), 0)
        )
        phi_indices, eta_indices = np.nonzero(mask)
        eta_weights = eta_ticks[eta_indices] * weights
        phi_weights = phi_ticks[phi_indices] * weights

        weights_sum = np.sum(weights)
        eta_barycenter = np.sum(eta_weights) / weights_sum
        phi_barycenter = np.sum(phi_weights) / weights_sum

        return eta_barycenter, phi_barycenter

    def barycenter_emb1(self, linear_weights=False, w_0=None, index=0):
        return self.calculate_barycenter(
            layer="emb1",
            index=index,
            linear_weights=linear_weights,
            w_0=w_0,
        )

    def barycenter_emb2(self, linear_weights=False, w_0=None, index=0):
        return self.calculate_barycenter(
            layer="emb2",
            index=index,
            linear_weights=linear_weights,
            w_0=w_0,
        )

    def calculate_z_barycenter(self, linear_weights=False, w_0=None, index=0):
        eta_l1 = self.barycenter_emb1(linear_weights, w_0, index=index)[0]
        eta_l2 = self.barycenter_emb2(linear_weights, w_0, index=index)[0]

        R_l1 = utils.R_emb1(eta_l1)
        R_l2 = utils.R_emb2(eta_l2)

        theta_l1 = utils.eta_to_theta(eta_l1)
        theta_l2 = utils.eta_to_theta(eta_l2)

        z_l1 = R_l1 / np.tan(theta_l1)
        z_l2 = R_l2 / np.tan(theta_l2)

        z_vtx = (z_l1 * R_l2 - z_l2 * R_l1) / (R_l2 - R_l1)

        return z_vtx

    def add_barycenters(self, linear_weights=None, w_0=None):
        if "bary_emb1_eta" in self.dataset.columns.values:
            print("Barycenters have already been added, skipping.")
            return
        bary_emb1_eta = []
        bary_emb2_eta = []

        all_etas_emb1 = np.array(
            [self.eta_ticks(e, layer="emb1") for e in self.emb1_eta]
        )

        phi_dim_emb1 = self.dimensions["emb1"][0]
        all_etas_emb1 = np.tile(all_etas_emb1, (1, phi_dim_emb1))

        all_etas_emb2 = np.array(
            [self.eta_ticks(e, layer="emb2") for e in self.emb2_eta]
        )

        phi_dim_emb2 = self.dimensions["emb2"][0]
        all_etas_emb2 = np.tile(all_etas_emb2, (1, phi_dim_emb2))

        bary_emb1_eta = np.sum(self.emb1_cells.values * all_etas_emb1, axis=1) / np.sum(
            self.emb1_cells.values, axis=1
        )
        bary_emb2_eta = np.sum(self.emb2_cells.values * all_etas_emb2, axis=1) / np.sum(
            self.emb2_cells.values, axis=1
        )

        df1 = pd.DataFrame(
            {"bary_emb1_eta": bary_emb1_eta, "bary_emb2_eta": bary_emb2_eta}
        )

        self.dataset = pd.concat([self.dataset, df1], axis=1)

        del df1
        del all_etas_emb1, all_etas_emb2, bary_emb1_eta, bary_emb2_eta

    def add_physics_object_type(self, typ):
        if "physics_object_type" in self.dataset.columns.values:
            print("Physics object type has already been added to dataframe, skipping.")
            return

        len = self.length
        df1 = pd.DataFrame({"physics_object_type": [typ] * len})
        self.dataset = pd.concat([self.dataset, df1], axis=1)
