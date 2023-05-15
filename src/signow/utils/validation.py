"""Validation module input data for signature nowcasting."""
from logging import getLogger

import pandas as pd
import pandas.api.types as pdtypes

LOGGER = getLogger(__name__)


def _raise_verr(msg):
    LOGGER.error(msg)
    raise ValueError(msg)


def _raise_terr(msg):
    LOGGER.error(msg)
    raise TypeError(msg)


def reformat(df: pd.DataFrame) -> pd.DataFrame:
    """Reformat long data format to wide format."""
    str_cols = [i for i in df.columns if pdtypes.is_string_dtype(df[i])]
    ncols = [i for i in df.columns if pdtypes.is_numeric_dtype(df[i])]
    if len(str_cols) == 1 and len(ncols) == 1:
        df = df.pivot(columns=str_cols[0], values=ncols[0])
    elif len(ncols) != len(df.columns):
        msg = "Do not recognise format of indicator dataframe."
        LOGGER.error(msg)
        raise ValueError(msg)
    return df


def _validate_index(index: pd.DatetimeIndex, name: str) -> None:
    """Validate Datetime index."""
    if not isinstance(index, pd.DatetimeIndex):
        _raise_terr(f"Index of series {name} is not a pandas.DatetimeIndex.")

    if not pd.api.types.is_datetime64_dtype(index):
        _raise_terr(f"Indices of series {name} are not datetime64.")

    if index.has_duplicates:
        _raise_verr("DatetimeIndex contain duplicate entries.")


def _validate_dataframe(df: pd.DataFrame, name: str):
    """Validate values of a Pandas DataFrame."""
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if not isinstance(df, pd.DataFrame):
        _raise_terr(f"{name} is {type(df)} expected Pandas DataFrame.")

    if not all(pdtypes.is_numeric_dtype(df[c]) for c in df.columns):
        cols_to_check = [c for c in df.columns if c]
        cols = (c for c in cols_to_check if not pdtypes.is_numeric_dtype(df[c]))
        _raise_verr(f"{','.join(cols)} is {type(df)} expected Numeric type.")
    return df


def check_order(data: pd.DataFrame) -> pd.DataFrame:
    """Check that the datetime index is in the correct order."""
    if data.index.is_monotonic_increasing:
        return data
    return data.sort_index()


def get_y_freq(y: pd.DataFrame, input_freq=None) -> str:
    """Get and validate target y frequency."""
    y_freq = input_freq
    if len(y) < 3 and not isinstance(input_freq, str):
        _raise_verr("Cannot infer frequency for y. Need to use input y_freq")
    else:
        y_freq = pd.infer_freq(y)

    if len(y) >= 3:
        y_freq = pd.infer_freq(y.index, warn=True)

    if not isinstance(input_freq, str):
        y_freq = pd.infer_freq(y.index, warn=True)

    return y_freq


def get_X_freq(data: pd.DataFrame, input_freq) -> dict:
    """Infer frequency of X."""
    freq_dict = {
        col: pd.infer_freq(data[col].dropna().index, warn=True) for col in data.columns
    }
    return freq_dict


def _get_freq(freq: str) -> str:
    """Map pandas offsets aliases to frequencies"""
    if freq in ["M", "MS"]:
        return "M"
    if freq.__contains__("Q"):
        return "Q"
    return freq


def _validate_y_mapped_freq(y_mapped, y_freq):
    """Validate mapped y series."""
    if not y_freq:
        _raise_verr("y_freq needs to be set with y_mapped")

    freq = _get_freq(y_freq)
    if not y_mapped.dropna().resample(freq).nunique().eq(1).all().iloc[0]:
        _raise_verr(
            f"""y_mapped contains multiple values within each
                    period for freq {y_freq}."""
        )


def check_freq_compatibility(y: pd.DataFrame, X: pd.DataFrame):
    """Check that y frequency is lower or equal to higher frequency."""
    if len(X) < len(y):
        _raise_verr("X need to be at a lower frequency than y")


def validate_y(y: pd.DataFrame, y_freq: str):
    """Validate target y."""
    _validate_index(y.index, "y")
    _validate_dataframe(y, "y")
    y = y.rename(columns={y.columns[0]: "y"})
    y = check_order(y)
    return y, get_y_freq(y, y_freq)


def validate_X(X: pd.DataFrame, X_freq: dict) -> (pd.DataFrame, dict):
    """Validate and reformat X dataframe."""
    X = reformat(X)
    _validate_index(X.index, "X")
    _validate_dataframe(X, "X")
    X = check_order(X)
    return X, get_X_freq(X, X_freq)


def validate_X_y(X, y, X_freq, y_freq):
    """Validate X and y."""
    y, y_freq = validate_y(y, y_freq)
    X, X_freq = validate_X(X, X_freq)
    return X, y, X_freq, y_freq


def validate_regressor_type(regressor):
    """Validates and checks type of regressor entered.
    Reverts to regressor type Linear if type not recognised.

    Args:
        regressor (str): A passed in argument that is used to choose regressor

    Raises:
        Warning: The is it is reverting to Linear regressor if it is not recognised
    """

    if regressor not in ["elasticnet", "lasso", "ridge", "linear"]:
        msg = f"Do not recognise regressor type {regressor}, reverting to default regressor: Linear"
        LOGGER.warning(msg)


def validate_bool(user_param, param_arg):
    """Validates the boolean paramater entered. Fails if entered value isn't boolean.

    Args:
        user_param (bool): A passed in argument that wil be tested to confirm it is an bool
        param_arg (str): The key being passed in

    Raises:
        ValueError: Checks it is an bool object
    """
    if type(user_param) != bool:
        msg = f"Invalid {param_arg} argument, please enter a boolean type value"
        LOGGER.error(msg)
        raise ValueError(msg)


def validate_int(user_param, param_arg):
    """Validates the integer paramater entered. Fails if entered value isn't int type and is less than 1.

    Args:
        user_param (int): A passed in argument that will be tested to confirm it is an int
        param_arg (str): The key being passed in

    Raises:
        ValueError: Checks it is an int object
        ValueError: Checks it is greater than zero
    """
    if type(user_param) != int:
        msg = f"Invalid {param_arg} argument, please enter a int type value"
        LOGGER.error(msg)
        raise ValueError(msg)
    else:
        if user_param <= 0:
            msg = (
                f"Invalid {param_arg} argument, {param_arg} can not be negative or zero"
            )
            LOGGER.error(msg)
            raise ValueError(msg)


def validate_keepsigs_type(keep_sigs):
    """Validates and checks type of keep sigs paramater entered.

    Args:
        keep_sigs (str): type of keep sigs

    Raises:
        ValueError: checks whether it is a recognised keep_sigs argument
    """
    if keep_sigs not in ["all", "innermost", "all_linear"]:
        msg = f"Do not recognise keep_sigs type {keep_sigs}"
        LOGGER.error(msg)
        raise ValueError(msg)


def validate_windowtype(windowtype):
    """Validates and checks window type entered.
    Also checks max_length paramater if window_type is days or ind.

    Args:
        windowtype (str/None): specifies the windowtype for the signature

    Raises:
        ValueError: Does not recognise the windowtype passed in
        TypeError: checks whether the max_length is specified
        TypeError: checks whether max_length is an int
        ValueError: checks whether max_length is not negative or zero
    """
    if windowtype["window_type"] not in ["days", "ind", None]:
        msg = (
            f"Do not recognise {windowtype['window_type']} as valid windowtype argument"
        )
        LOGGER.error(msg)
        raise ValueError(msg)
    if windowtype["window_type"] in ["days", "ind"]:
        if "max_length" not in windowtype.keys():
            msg = f"max_length argument not specified in signature paramaters. Please specify max_length as an int."
            LOGGER.error(msg)
            raise TypeError(msg)
        if type(windowtype["max_length"]) != int:
            msg = f"max_length argument is not in a recognisable format. Please specify length of your window as an int."
            LOGGER.error(msg)
            raise TypeError(msg)
        if windowtype["max_length"] <= 0:
            msg = f"max_length can not be negative or zero."
            LOGGER.error(msg)
            raise ValueError(msg)


def validate_model_params(model_params):
    """Validates user params

    Args:
        model_params (dict): all of the user defined paramaters for the signature
        and regressor

    Returns:
        model_params: dictionary of the params
    """
    # checks only those parameters entered
    # Defaults adding in sig estimator
    if "regressor" in model_params:
        validate_regressor_type(model_params["regressor"])
    if "apply_pca" in model_params:
        validate_bool(model_params["apply_pca"], "apply_pca")
    if "standardize" in model_params:
        validate_bool(model_params["standardize"], "standardize")

    if "basepoint" in model_params["sig_params"]:
        validate_bool(model_params["sig_params"]["basepoint"], "basepoint")
    if "use_multiplier" in model_params["sig_params"]:
        validate_bool(model_params["sig_params"]["use_multiplier"], "use_multiplier")
    if "keep_sigs" in model_params["sig_params"]:
        validate_keepsigs_type(model_params["sig_params"]["keep_sigs"])
    if "level" in model_params["sig_params"]:
        validate_int(model_params["sig_params"]["level"], "level")
    if "t_level" in model_params["sig_params"]:
        validate_int(model_params["sig_params"]["t_level"], "t_level")
    if "window_type" in model_params["sig_params"]:
        validate_windowtype(model_params["sig_params"])

    return model_params
