import numpy as np


def get_Y_1(dataframe):
    """Creates targets for model training."""
    return dataframe["z"].values


def get_X_1(dataframe):
    """
    Creates inputs for model training.
    Inputs include center etas, differences between center etas from different layers, and all energy information.
    All energy info means that all cells are passed as inputs in consistent ordering, without making any normalisations or changes.
    This type of model has all the information necessary to make predictions, but the inputs are not optimised in any way.
    Each single input is an array of 312 values (for single particle events).
    """
    X1 = np.hstack(
        (
            dataframe.psb_eta.values.reshape(-1, 1),
            dataframe.emb1_eta.values.reshape(-1, 1),
            dataframe.emb2_eta.values.reshape(-1, 1),
            dataframe.emb3_eta.values.reshape(-1, 1),
            dataframe.hab1_eta.values.reshape(-1, 1),
            #########################################
            dataframe.psb_eta.values.reshape(-1, 1)
            - dataframe.emb1_eta.values.reshape(-1, 1),
            dataframe.emb1_eta.values.reshape(-1, 1)
            - dataframe.emb2_eta.values.reshape(-1, 1),
            dataframe.emb2_eta.values.reshape(-1, 1)
            - dataframe.emb3_eta.values.reshape(-1, 1),
            dataframe.emb3_eta.values.reshape(-1, 1)
            - dataframe.hab1_eta.values.reshape(-1, 1),
            ###########################################
            dataframe["psb_cells"].values,
            dataframe["emb1_cells"].values,
            dataframe["emb2_cells"].values,
            dataframe["emb3_cells"].values,
            dataframe["hab1_cells"].values,
        )
    )
    return X1


def get_X_2(dataframe):
    """
    Creates inputs for model training.
    Inputs include center etas, differences between center etas from different layers, and some energy information.
    Compared to get_X_2, this time a sum over phi axis is taken from calorimeter cell windows.
    The reasoning behind this is that the regression problem of finding z is independent of phi.
    This procedure greately decreases number of inputs, without any accuracy loss.
    Each single input is an array of 62 values (for single particle events).
    """
    s = dataframe.shape[0]  # number of rows (events)

    psb_info = np.sum(dataframe["psb_cells"].values.reshape(s, 7, 9), axis=1)
    emb1_info = np.sum(dataframe["emb1_cells"].values.reshape(s, 3, 17), axis=1)
    emb2_info = np.sum(dataframe["emb2_cells"].values.reshape(s, 7, 9), axis=1)
    emb3_info = np.sum(dataframe["emb3_cells"].values.reshape(s, 7, 9), axis=1)
    hab1_info = np.sum(dataframe["hab1_cells"].values.reshape(s, 7, 9), axis=1)

    X1 = np.hstack(
        (
            dataframe.psb_eta.values.reshape(-1, 1),
            dataframe.emb1_eta.values.reshape(-1, 1),
            dataframe.emb2_eta.values.reshape(-1, 1),
            dataframe.emb3_eta.values.reshape(-1, 1),
            dataframe.hab1_eta.values.reshape(-1, 1),
            #########################################
            dataframe.psb_eta.values.reshape(-1, 1)
            - dataframe.emb1_eta.values.reshape(-1, 1),
            dataframe.emb1_eta.values.reshape(-1, 1)
            - dataframe.emb2_eta.values.reshape(-1, 1),
            dataframe.emb2_eta.values.reshape(-1, 1)
            - dataframe.emb3_eta.values.reshape(-1, 1),
            dataframe.emb3_eta.values.reshape(-1, 1)
            - dataframe.hab1_eta.values.reshape(-1, 1),
            ###########################################
            psb_info,
            emb1_info,
            emb2_info,
            emb3_info,
            hab1_info,
        )
    )
    return X1
