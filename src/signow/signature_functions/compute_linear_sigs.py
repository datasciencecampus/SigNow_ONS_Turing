"""Functions used to work with signatures.

Responsible for calling the `esig` library to calculate signatures.

Includes functions for data preparation and working with the signatures once
calculated.

The main entry point is `compute_sigs_dates`.

Notes
-----
    This module is a modified version of code created by Lingyi Yang from The Alan Turing Institute.
    The source can be found on the public GitHub repository that reproduces the outputs
    for the academic paper: Nowcasting with Signature Methods (Cohen et al., 2023).
    The source repository is available here: https://github.com/lingyiyang/nowcasting_with_signatures.

"""
import itertools
import warnings

import esig
import numpy as np
import pandas as pd


def all_sig_keys(dim: int, level: int) -> list:
    """Create a list of names for signatures, with the first entry corresponding to the
    innermost integration dimension.

    Parameters
    ----------
    dim : int
        the dimension of the features
    level : int
        truncation level of signature terms

    Returns
    -------
    int
        A list of names for signature terms
    """
    keys = []
    for i in range(level + 1):
        prod = list(itertools.product(np.arange(1, dim + 1), repeat=i))
        prod = [str(x) for x in prod]
        keys.append(prod)

    keys = [i for s in keys for i in s]

    return keys


def find_single_sig_index(dim: int, index: int, level: int) -> list:
    """Find the indices for signature of 1 particular variable.

    This variable is in the dimension given by index on original path.

    Parameters
    ----------
    dim : int
        the dimension of the features
    index : int
        the position of the variable of interest
    level : int
        truncation level of the signature

    Returns
    -------
    list
        A list of indices corresponding to signatures of the variable of
        interest (including the constant 1 term at the 0th order), e.g.
        if index=1, those corresponding to signatures (), (1), (1,1), ...
    """

    ind = [0]
    base_ind = 0
    for i in range(0, level):
        base_ind += dim ** (i)
        increment = (index) * base_ind
        ind.append(base_ind + increment)
    return ind


def compute_linear_sigs(dim: int, level: int) -> list:
    """Wrapper function to compute the indices of all linear signature terms.

    Note it is vital for time index to be at 0 as this is assumed here.

    Parameters
    ----------
    dim : int
        the dimension of the features
    level : int
        truncation level of the signature

    Returns
    -------
    list
        A list of signature indices corresponding to "linear" signatures
    """

    ind1 = find_innermost_sigs(dim, level)
    ind2 = find_outermost_sigs(dim, level)
    ind3 = find_middle_sigs(dim, level)

    # convert to set to remove duplicates caused by inner and outermost sig functions
    ind = list(set(ind1 + ind2 + ind3))
    ind.sort()

    return ind


def find_innermost_sigs(dim: int, level: int) -> list:
    """Find the indices corresponding to signature terms where only where the innermost
    integral is with respect a variable which is not time
    (index of the time variable is assumed to be 0).

    Parameters
    ----------
    dim : int
        the dimension of the features
    level : int
        truncation level of the

    Returns
    -------
    list
        A list of indices corresponding to signatures of the form S(x...)
    """

    ind = [0]

    base_ind = 0
    for i in range(level):
        for j in range(dim):
            ind.append(base_ind + dim**i + j * dim**i)

        base_ind += dim**i
    return ind


def find_outermost_sigs(dim: int, level: int) -> list:
    """Find the indices corresponding to signature terms where only the outermost
    integral is with respect a variable which is not time
    (index of the time variable is assumed to be 0).

    Parameters
    ----------
    dim : int
        the dimension of the features
    level : int
        truncation level of the signature

    Returns
    -------
    list
        A list of indices corresponding to signatures of the form S(...x)
    """

    ind = [0]

    base_ind = 0
    for i in range(level):
        for j in range(dim):
            ind.append(base_ind + dim**i + j)

        base_ind += dim**i
    return ind


def find_middle_sigs(dim: int, level: int) -> list:
    """Find the indices of the linear signature terms where the integral with respect to
    variable (not time) is not the innermost or the outermost one
    (where index of the time variable is assumed to be 0).

    Parameters
    ----------
    dim : int
        the dimension of the features
    level : int
        truncation level of the signature

    Returns
    -------
    list
        A list of indices corresponding to ``linear'' signatures of the
        form S(...x...) (which are not ``innermost'' or ``outermost''
        signatures).
    """

    ind = []
    base = 1 + dim + dim**2

    for i in range(3, level + 1):
        for j in range(1, i - 1):
            for k in range(1, dim):
                ind.append(base + k * dim**j)

        base += dim**i

    return ind


def find_linear_sig_features(
    dim: int, var_level: int, t_level: int = None, keep_sigs: str = "innermost"
) -> list:
    """Filters the linear signatures so that the terms in purely time (t) can be of
    a different level to other variables. This assumes that the list of signatures
    has been filtered down to linear signatures only (from all the signatures) and
    forms an additional filter.

    Parameters
    ----------
    dim : int
        the dimension of the features
    var_level : int
        truncation level of signature terms of features (not time)
    t_level : int
        a potentially different truncation level for signatures of time
    keep_sigs : str
        whether to keep all linear signatures or just the innermost -
        an option between 'innermost' or 'all_linear'

    Returns
    -------
    list
        A list of indices corresponding to the relevant signatures.
    """
    if keep_sigs == "innermost":
        if not t_level:
            t_level = var_level

        ind = [0]

        base = 0
        for i in range(1, t_level + 1):
            base += (dim - 1) * (i - 1) + 1
            ind.append(base)

        base = 2 - dim
        for i in range(1, var_level + 1):
            base += (dim - 1) * (i) + 1
            for j in range(dim - 1):
                ind.append(base + j)

        ind = list(set(ind))
        ind.sort()

    elif keep_sigs == "all_linear":
        ind = list(
            range(
                int(1 + dim * var_level + 0.5 * (dim - 1) * var_level * (var_level - 1))
            )
        )
        if t_level < var_level:
            remove_ind = [
                1 + np.sum(dim + (dim - 1) * np.arange(i + 1))
                for i in range(t_level - 1, var_level - 1)
            ]
            ind = list(set(ind).difference(remove_ind))

        elif t_level > var_level:
            add_ind = [
                1 + np.sum(dim + (dim - 1) * np.arange(i + 1))
                for i in range(var_level - 1, t_level - 1)
            ]
            ind = list(set(ind).union(add_ind))

    else:
        warnings.warn("Linear signature set not specified")
        ind = []

    return ind


def multiplier_for_t_terms(sigs, multiplier: int, dim: int, t_level: int):
    """Multiply signatures (corresponding to time only) by a specified multiplier.

    Parameters
    ----------
    sigs :
        the precomputed signature terms (where it is assumed that time
        is at index 0 for the original data)
    multiplier : int
        specified multiplier which is typically `starting value'
        of target variable
    dim : int
        the dimension of the feature set
    t_level : int
        the truncation level of t_level

    Returns
    -------
        Modified signatures with the time terms multiplied by the
        specified multiplier.
    """

    # Create a logical mask for time terms and use it to set a scaling for those times
    # Note that this is applied to a complete signature dataframe (i.e. level=t_level)
    # which will filter out higher truncation orders of t signatures later
    t_terms_index = find_single_sig_index(dim=dim, index=0, level=t_level)
    t_terms = [i in t_terms_index for i in np.arange(len(sigs))]

    # Apply multiplier for the linear case
    multipliers = [(multiplier - 1) * a + 1 for a in t_terms]
    sigs = [a * b for (a, b) in zip(sigs, multipliers)]

    return sigs


def rectilinear_interpolation(df: pd.DataFrame) -> pd.DataFrame:
    """Find the rectilinear interpolated dataframe, assumng the time column is t.

    Parameters
    ----------
    df : pd.Dataframe
        input dataframe

    Returns
    -------
    pd.DataFrame
        Output dataframe filled by rectilinear interpolation
    """

    df_filled = df.ffill()
    df2 = pd.concat(
        [df_filled["t"].iloc[1:], df_filled.drop("t", axis=1).shift().iloc[1:]], axis=1
    )
    df_filled = (
        pd.concat([df2, df_filled], ignore_index=True)
        .sort_values("t")
        .reset_index(drop=True)
    )

    return df_filled


def _interpolation(df: pd.DataFrame, fill_method: str = "ffill") -> pd.DataFrame:
    """Interpolate over missing values given a specific method.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe
    fill_method : str
        method of interpolation

    Returns
    -------
    pd.DataFrame
        interpolated dataframe
    """
    if fill_method == "rectilinear":
        df_filled = rectilinear_interpolation(df)
    else:
        df_filled = df.interpolate(method=fill_method)

    return df_filled


def _basepoint(df: pd.DataFrame, basepoint: bool = True) -> pd.DataFrame:
    """Add basepoint to the data.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe
    basepoint : bool
        add basepoint to data. Defaults to True.

    Returns
    -------
    pd.DataFrame
        dataframe with or without basepoint added
    """
    if basepoint:
        base_data = pd.DataFrame(
            np.zeros(len(df.columns)).reshape([1, -1]), columns=df.columns
        )
        df = pd.concat([base_data, df.loc[:]]).reset_index(drop=True)

    return df


def _compute_sigs(df: pd.DataFrame, level: int = 3) -> np.array:
    """Computes the signatures of a path to the required truncation level.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe
    level : int
        level of truncation

    Returns
    -------
    np.array
        computed signatures
    """
    sigs = esig.stream2sig(np.array(df), level)
    return sigs


def compute_subframe(df: pd.DataFrame, ind: int, configs: dict) -> pd.DataFrame:
    """From the dataframe of the observed data, select a sub-dateframe based on the current
    index for the rolling/expanding window.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe of the observed data
    ind : int
        the current index
    configs : dict
        the configurations for the model (see compute_sigs_dates for more info)

    Returns
    -------
    pd.DataFrame
        A subset of the input dataframe, across the relevant horizon
    """

    if configs["max_length"] and configs["max_length"] != np.infty:
        if configs["window_type"] == "days":
            start_ind = max(
                [
                    df.index[ind - 1] - pd.Timedelta(configs["max_length"], "D"),
                    df.index[0],
                ]
            )
        elif configs["window_type"] == "ind":
            start_ind = df.index[max(0, ind - int(configs["max_length"]))]
    else:
        start_ind = df.index[0]

    end_ind = df.index[ind - 1]

    df2 = df.loc[start_ind:end_ind, :].copy()

    return df2


def select_signatures(
    df_sigs: pd.DataFrame, dim: int, t_level: int, configs
) -> pd.DataFrame:
    """From the dataframe of signatures, select those relevant to the specified config.

    Parameters
    ----------
    df_sigs : pd.DataFrame
        dataframe of all signatures
    dim : int
        dimension of features
    t_level : int
        truncation level of the time (t) terms
    configs : dict
        the configurations for the model (see compute_sigs_dates for more info)

    Returns
    -------
    pd.DataFrame
        A subset of the input dataframe
    """

    if configs["keep_sigs"] != "all":
        # filter for sigs linear in the observed values
        ind = compute_linear_sigs(dim, configs["level"])
        df_sigs = df_sigs[df_sigs.columns[ind]]

        ind = find_linear_sig_features(
            dim=dim,
            var_level=configs["level"],
            t_level=t_level,
            keep_sigs=configs["keep_sigs"],
        )
        df_sigs = df_sigs[df_sigs.columns[ind]]

    else:
        t_terms_index = find_single_sig_index(dim=dim, index=0, level=t_level)
        t_terms_index2 = find_single_sig_index(dim=dim, index=0, level=configs["level"])
        ind = np.arange(len(df_sigs.columns))
        if t_level < configs["level"]:
            remove_ind = set(t_terms_index2).difference(set(t_terms_index))
            ind = list(set(range(len(df_sigs.columns))).difference(remove_ind))
        elif t_level > configs["level"]:
            add_ind = set(t_terms_index).difference(set(t_terms_index2))
            ind = list(
                set(range(np.sum(dim ** np.arange(configs["level"])))).union(add_ind)
            )

        df_sigs = df_sigs[df_sigs.columns[ind]]

    return df_sigs


def compute_sigs_dates(df: pd.DataFrame, configs: dict, df_target: pd.DataFrame = None):
    """Computes path signatures from a timeseries/path.

    Assumes that time is the zeroth index of the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe of the observation data
    configs : dict
        A dictionary of configurations
        This may include the following keys:

            - max_length: the maximum length of the lookback window
                            (expanding window if data is less than this
                            length)
            - level: truncation level of the signatures of variables
                        not including time
            - t_level: the level of truncation of the time parameter if
                        different to other variables
            - window_type: if we are choosing window based on number of
                            observations (ind) or calendar date (days)
            - keep_sigs: whether to keep only the innermost signatures,
                            all linear signatures or all signatures
            - prefix: a prefix for the new columns of signature terms
            - fill_method: method to fill the dataframe
            - basepoint: Boolean whether to add a basepoint to remove
                            translation invariance of the signature
            - use_multiplier: Boolean whether to multiply the time terms by
                                the (known) value of the target at the start
                                of the sliding window
            - target: variable name of target (or similar/lagged variable
                        used as a multiplication factor to the signature
                        terms of time only
    df_target : pd.DataFrame
        dataframe of the outcome we want to predict
        (needed for multiplying the t-terms in the signature if
                    `use_multipliers' is True in the config)

    Returns
    -------
    pd.DataFrame
        A dataframe with the relevant signature features
    """

    all_sigs = []

    if not configs["t_level"]:
        t_level = configs["level"]
    else:
        t_level = configs["t_level"]

    dim = len(df.columns)

    for ind in range(1, len(df) + 1):
        df2 = compute_subframe(df, ind, configs)
        interpolated_df = _interpolation(df=df2, fill_method=configs["fill_method"])
        basepoint_df = _basepoint(df=interpolated_df, basepoint=configs["basepoint"])

        max_level = max([configs["level"], t_level])
        sigs = _compute_sigs(df=basepoint_df, level=max_level)

        if configs["use_multiplier"]:
            start_ind = df2.index[0]
            multiplier = df_target.loc[start_ind, "y"]
            sigs = multiplier_for_t_terms(
                sigs=sigs, multiplier=multiplier, dim=dim, t_level=t_level
            )

        all_sigs.append(sigs)

    all_sig_names = all_sig_keys(dim, max([configs["level"], t_level]))
    if "prefix" in configs:
        all_sig_names = [str(configs["prefix"]) + n for n in all_sig_names]

    df_sigs = pd.DataFrame(all_sigs, columns=all_sig_names)
    df_sigs = select_signatures(df_sigs, dim, t_level, configs)

    return df_sigs.set_index(df.index.values)
