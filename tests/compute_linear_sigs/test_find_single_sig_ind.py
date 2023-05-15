"""Unit tests for `find_single_sig_index` function.

This function is used to find the indices for the signature of 1 variable.
Function can be located in `compute_linear_sigs`.

Parameters passed to `compute_linear_sigs` are:

    dim : int
        Dimension of the features (count of the total variables).
    index : int
        Position of the variable of interest (from 0 to dim-1).
    level : int
        Truncation level of the signature.

Parameters dim, index created within the structure of the code based on
on the dataset and the computed signatures.

Parameter level is user defined based on how the signatures are created.

In the pipeline this is mainly used as an multiplier for the t terms

"""
import pandas as pd

from signow.signature_functions.compute_linear_sigs import find_single_sig_index


class TestFindSingleSigIndex:
    """A class that tests function find_single_sig_index"""

    def test_find_single_sig_index_small(self) -> None:
        # Given
        test_data = pd.DataFrame(data={f"col{num}": [] for num in range(0, 3)})
        dim = len(test_data.columns)
        index = 1
        level = 2
        expected = [0, 2, 8]

        # When
        index = find_single_sig_index(dim, index, level)

        # Then
        assert index == expected

    def test_find_single_sig_index_large(self) -> None:
        # Given
        test_data = pd.DataFrame(data={f"col{num}": [] for num in range(0, 16)})
        dim = len(test_data.columns)
        index = 2
        level = 2
        expected = [0, 3, 51]

        # When
        index = find_single_sig_index(dim, index, level)

        # Then
        assert index == expected
