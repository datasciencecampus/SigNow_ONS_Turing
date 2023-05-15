"""Functions to randomly generate test data."""
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import ArmaProcess


def random_ar_data(length: int) -> np.ndarray:
    """Generates random ARMA process data of a specified length.

    Used to demonstrate and test functionality without using real data.

    Parameters
    ----------
    length: int
        length of data to generate

    Returns
    -------
    np.ndarray
        ARMA processed data
    """
    # set params for AR and MA
    arparams = np.array([0.75, -0.25])
    maparams = np.array([0.65, 0.35])
    # Add zero lag to ar and negate
    ar = np.r_[1, -arparams]
    # Add zero lag
    ma = np.r_[1, maparams]
    # generate ARMA process
    arma_process = ArmaProcess(ar, ma)
    y = arma_process.generate_sample(length)

    return y


def generate_dummy_indicators(
    indexed_df: pd.DataFrame, indicator_name: list, length: int
) -> pd.DataFrame:
    """Generates a set of indicators based on
    specified amount of indicators.

    Parameters
    ----------
    indexed_df : pd.DataFrame
        An indexed empty dataframe.
    indicator_name : list
        List of indicators
    length : int
        Number of data points to generate.

    Returns
    -------
    pd.DataFrame
        Indicators in long format.
    """
    df = indexed_df.copy()
    df_concat = pd.DataFrame(columns=["indicator_name", "value"])
    for name in indicator_name:
        # Set the random walk path for this indicator
        df["value"] = random_ar_data(length=length)
        # assign the indictaor its name
        df["indicator_name"] = name
        # concat this indicator with the others
        df_concat = pd.concat([df_concat, df])
        # reset the df
        df = indexed_df.copy()

    df_concat.index.name = "ref_date"
    return df_concat[["value", "indicator_name"]]


def generate_dummy_target(
    index: pd.DatetimeIndex, indicators: pd.DataFrame, indicator_name: list, length: int
) -> pd.DataFrame:
    """Generates a target dataframe that is
    a higher frequency than indicators.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Date index for the generated data.
    indicators : pd.DataFrame
        Indicators DataFrame.
    indicator_name : list
        List of indicator names.
    length : int
        Length of the indicators DataFrame.

    Returns
    -------
    pd.DataFrame
        Target DataFrame
    """
    target_df = indicators.pivot(columns="indicator_name", values="value")
    target_df["noise"] = np.random.randn(length)

    # fmt: off
    _operations = [
        1, 2, 2, 3, 4, 5, 2, 3, 4, 5,
        6, 7, 8, 4, 3, 2, 4, 5, 7, 8,
        2, 3, 4, 5, 2, 3
    ]
    # fmt: on

    # normalise the operations
    _operations = [
        float(i) / sum(_operations[0 : len(indicator_name)])
        for i in _operations[0 : len(indicator_name)]
    ]

    target_df["value"] = 0
    for i, indicator in enumerate(indicator_name):
        target_df["value"] += _operations[i] * target_df[indicator]

    target_df["value"] += target_df["noise"] / 10
    target_df = target_df.loc[index, "value"]
    target_df.index.name = "ref_date"
    target_df.index = pd.to_datetime(target_df.index, dayfirst=True)

    return target_df.to_frame()


def create_data(
    start_date: str,
    end_date: str,
    num_indicators: int,
    wide_indic_df: bool = True,
    SEED=12345,
):
    """Main function call that generates
    the indicator and target dataframe.

    Parameters
    ----------
    start_date : str
        Start date for the generated data.
    end_date : str
        End date for the generated data.
    num_indicators : int
        Number of indicators required.
    wide_indic_df : bool, optional
        convert indicators to wide, by default True
    SEED : int, optional
        Seed for the random generator, by default 12345

    Returns
    -------
    indicators_df : pd.DataFrame
        Indicator DataFrame
    target_df : pd.DataFrame
        Target DataFrame
    """
    np.random.seed(SEED)

    month_index = pd.date_range(start=start_date, end=end_date, freq="MS")
    quarter_index = pd.date_range(
        start=start_date, end=end_date, freq=pd.offsets.MonthBegin(3)
    )

    base_df = pd.DataFrame(index=month_index)

    _indicator_names = "abcdefghijklmnopqrstuvwxyz"
    _names = list(_indicator_names[0:num_indicators])

    indicators_df = generate_dummy_indicators(
        indexed_df=base_df, indicator_name=_names, length=len(base_df)
    )

    target_df = generate_dummy_target(
        index=quarter_index,
        indicators=indicators_df,
        indicator_name=_names,
        length=len(base_df),
    )

    if wide_indic_df:
        indicators_df = indicators_df.pivot(columns="indicator_name", values="value")

    return indicators_df, target_df


if __name__ == "__main__":
    indicators, target = create_data(
        start_date="2000-01-01",
        end_date="2014-06-01",
        num_indicators=3,
        wide_indic_df=False,
    )

    print(">>>>>> DataFrames:")
    print(indicators)
    print(target)
