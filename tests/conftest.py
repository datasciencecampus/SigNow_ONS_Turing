"""Functions available across all tests."""
from typing import List, Tuple

import pandas as pd


def create_dataframe(data: List[Tuple[str]], **kwargs) -> pd.DataFrame:
    """Create pandas df from tuple data with a header.

    Parameters
    ----------
    data : List[Tuple[str]]
        List of tuples to be converted into a DataFrame.

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame of tuple data.
    """
    return pd.DataFrame.from_records(data[1:], columns=data[0], **kwargs)
