import numpy as np

import events_package.utils as utils


def get_Y_1(dataframe):
    """Creates targets for model training."""
    return dataframe["z"].values


def get_Y_2(dataframe):
    """Creates targets for model training."""
    return dataframe["z"].values.astype("float16")


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


def get_X_3(dataframe):
    """
    Creates inputs for model training.
    Inputs include center etas, differences between etas from different layers, energy information.
    Similar to get_X_2, but the energies are normalised to total energy in all layers.
    """
    s = dataframe.shape[0]  # number of rows (events)

    psb_info = np.sum(dataframe["psb_cells"].values.reshape(s, 7, 9), axis=1)
    emb1_info = np.sum(dataframe["emb1_cells"].values.reshape(s, 3, 17), axis=1)
    emb2_info = np.sum(dataframe["emb2_cells"].values.reshape(s, 7, 9), axis=1)
    emb3_info = np.sum(dataframe["emb3_cells"].values.reshape(s, 7, 9), axis=1)
    hab1_info = np.sum(dataframe["hab1_cells"].values.reshape(s, 7, 9), axis=1)

    # calculate total energy in each layer
    psb_tot = np.sum(psb_info, axis=1)
    emb1_tot = np.sum(emb1_info, axis=1)
    emb2_tot = np.sum(emb2_info, axis=1)
    emb3_tot = np.sum(emb3_info, axis=1)
    hab1_tot = np.sum(hab1_info, axis=1)

    # total energy deposited in all layers
    en_tot = psb_tot + emb1_tot + emb2_tot + emb3_tot + hab1_tot

    psb_info = psb_info / en_tot.reshape(-1, 1)
    emb1_info = emb1_info / en_tot.reshape(-1, 1)
    emb2_info = emb2_info / en_tot.reshape(-1, 1)
    emb3_info = emb3_info / en_tot.reshape(-1, 1)
    hab1_info = hab1_info / en_tot.reshape(-1, 1)

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


def get_X_4(dataframe):
    """
    Creates inputs for model training.
    Inputs include center etas, differences between etas from different layers, energy information.
    Compared to get_X_3, this one uses differences between normalised cells,
    instead of just cells.
    The energy inputs are created in the following way:
    - first a sum over phi is taken just as in get_X_2
    - normalisation is done with respect to total deposited energy just like get_X_3
    - next, utils.fold_list_2d is used to take differences between cells symmetric around
    the middle cell, plus the middle cell itself.

    So if energy inputs after summing over phi were for example (0, 1, 2, 4, 6),
    then the input will be: (2, 4-1, 6-0) = (2, 3, 6).
    """
    s = dataframe.shape[0]  # number of rows (events)

    psb_info = np.sum(dataframe["psb_cells"].values.reshape(s, 7, 9), axis=1)
    emb1_info = np.sum(dataframe["emb1_cells"].values.reshape(s, 3, 17), axis=1)
    emb2_info = np.sum(dataframe["emb2_cells"].values.reshape(s, 7, 9), axis=1)
    emb3_info = np.sum(dataframe["emb3_cells"].values.reshape(s, 7, 9), axis=1)
    hab1_info = np.sum(dataframe["hab1_cells"].values.reshape(s, 7, 9), axis=1)

    # calculate total energy in each layer
    psb_tot = np.sum(psb_info, axis=1)
    emb1_tot = np.sum(emb1_info, axis=1)
    emb2_tot = np.sum(emb2_info, axis=1)
    emb3_tot = np.sum(emb3_info, axis=1)
    hab1_tot = np.sum(hab1_info, axis=1)

    # total energy deposited in all layers
    en_tot = psb_tot + emb1_tot + emb2_tot + emb3_tot + hab1_tot

    psb_info = utils.fold_list_2d(psb_info) / en_tot.reshape(-1, 1)
    emb1_info = utils.fold_list_2d(emb1_info) / en_tot.reshape(-1, 1)
    emb2_info = utils.fold_list_2d(emb2_info) / en_tot.reshape(-1, 1)
    emb3_info = utils.fold_list_2d(emb3_info) / en_tot.reshape(-1, 1)
    hab1_info = utils.fold_list_2d(hab1_info) / en_tot.reshape(-1, 1)

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


def get_X_5(dataframe):
    """
    Creates inputs for model training.
    Inputs include center etas, differences between etas from different layers, energy information.
    The idea is similar to get_X_4, to use differences between normalised energies symmetric about
    the middle cell.

    But this time, only certain differences are used, as it has been noticed that some inputs
    from get_X_4 do not contribute much. Therefore number of features in reduced further.
    """
    s = dataframe.shape[0]  # number of rows (events)

    psb_info = np.sum(dataframe["psb_cells"].values.reshape(s, 7, 9), axis=1)
    emb1_info = np.sum(dataframe["emb1_cells"].values.reshape(s, 3, 17), axis=1)
    emb2_info = np.sum(dataframe["emb2_cells"].values.reshape(s, 7, 9), axis=1)
    emb3_info = np.sum(dataframe["emb3_cells"].values.reshape(s, 7, 9), axis=1)
    hab1_info = np.sum(dataframe["hab1_cells"].values.reshape(s, 7, 9), axis=1)

    psb_tot = np.sum(psb_info, axis=1)
    emb1_tot = np.sum(emb1_info, axis=1)
    emb2_tot = np.sum(emb2_info, axis=1)
    emb3_tot = np.sum(emb3_info, axis=1)
    hab1_tot = np.sum(hab1_info, axis=1)

    en_tot = psb_tot + emb1_tot + emb2_tot + emb3_tot + hab1_tot

    psb_info = utils.fold_list_2d(psb_info) / en_tot[:, np.newaxis]
    emb1_info = utils.fold_list_2d(emb1_info) / en_tot[:, np.newaxis]
    emb2_info = utils.fold_list_2d(emb2_info) / en_tot[:, np.newaxis]
    emb3_info = utils.fold_list_2d(emb3_info) / en_tot[:, np.newaxis]
    hab1_info = utils.fold_list_2d(hab1_info) / en_tot[:, np.newaxis]

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
            np.delete(psb_info, [2, 3, 4], axis=1),
            np.delete(emb1_info, [5, 6, 7, 8], axis=1),
            np.delete(emb2_info, [2, 3, 4], axis=1),
            np.delete(emb3_info, [2, 3, 4], axis=1),
            np.delete(hab1_info, [2, 3, 4], axis=1),
        )
    )
    return X1


def get_Y_1(dataframe):
    return dataframe["z"].values
