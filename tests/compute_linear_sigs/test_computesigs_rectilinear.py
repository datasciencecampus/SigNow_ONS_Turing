""" Unit tests for compute_sigs, rectilinear_interpolation.

When compute_sigs is called it is passed a frame of data, truncation level of the
signature, the fill_method (to deal with the ragged edge nature of the dataset or
missing values within the dataset) and a bool corresponding to whether to add a
basepoint to the data.

"""
import esig
import hypothesis.strategies as st
import numpy as np
import pandas as pd
from hypothesis import given

from signow.signature_functions.compute_linear_sigs import (
    _compute_sigs,
    _basepoint,
    _interpolation,
)


class TestBasepoint:
    """Tests for the _basepoint function."""

    def test_basepoint_true(self):
        """This function tests the function when thebasepoint setting is
        true.
        When basepoint is set to false this behaviour is not called
        Basepoint will only ever be True or False as defined by user

        """
        # Given
        # creates data with NaNs
        test_df = pd.DataFrame(data={"col1": [1, 1, 2, 2, 3, 3]})
        add_base = True

        # When
        actual_baseadded_df = _basepoint(df=test_df, basepoint=add_base)

        # Then
        expect_baseadded_df = pd.DataFrame(data={"col1": [0.0, 1, 1, 2, 2, 3, 3]})

        pd.testing.assert_frame_equal(
            left=actual_baseadded_df, right=expect_baseadded_df
        )

    def test_basepoint_false(self):
        """This function tests the basepoint setting is set to false
        When basepoint is set to false this behaviour returns the same
        dataframe

        """
        # Given
        # creates data with NaNs
        test_df = pd.DataFrame(data={"col1": [1, 1, 2, 2, 3, 3]})
        add_base = False

        # When
        actual_baseadded_df = _basepoint(df=test_df, basepoint=add_base)

        # Then
        expect_baseadded_df = pd.DataFrame(data={"col1": [1, 1, 2, 2, 3, 3]})

        pd.testing.assert_frame_equal(
            left=actual_baseadded_df, right=expect_baseadded_df
        )

    @given(
        level=st.integers(min_value=2, max_value=3),
        basepoint_feature=st.integers(min_value=0, max_value=1),
    )
    def test_generatedsignature(self, level, basepoint_feature):
        """This function tests the generated signature with and without the basepoint
        feature
        #NOTE: Have restricted level to 2 or 3 to stop hypothesis error messages that
        # were failing tests due to their time to execute
        Args:
            nnans (int): number of random nan values
            basepoint_feature (int): to operate basepoint feature
        """
        # Given
        # creates data with NaNs
        test_df = pd.DataFrame(
            data={"col1": [1.0, 1, 2, 2, 3, 3], "t": [0.0, 10, 20, 30, 40, 50]}
        )
        if basepoint_feature:
            add_base = True
            test_df = _basepoint(df=test_df, basepoint=add_base)

        # When
        actual_sigs = _compute_sigs(df=test_df, level=level)

        # Then
        generated_sigs = esig.stream2sig(np.array(test_df), level)
        assert all(actual_sigs == generated_sigs)


class TestInterpolation:
    """Tests for the _interpolation function."""

    def test_ffill(self):
        """This function tests the ffill method of interpolation."""
        # Given
        test_df = pd.DataFrame(data={"col1": [1, np.nan, np.nan, 2, np.nan, 3]})

        # When
        actual_interp_df = _interpolation(df=test_df, fill_method="ffill")

        # Then
        expect_df = pd.DataFrame(data={"col1": [1.0, 1, 1, 2, 2, 3]})

        pd.testing.assert_frame_equal(left=actual_interp_df, right=expect_df)

    def test_bfill(self):
        """This function tests the bfill method of interpolation"""
        # Given
        test_df = pd.DataFrame(data={"col1": [1, np.nan, np.nan, 2, np.nan, 3]})

        # When
        actual_interp_df = _interpolation(df=test_df, fill_method="bfill")

        # Then
        expect_df = pd.DataFrame(data={"col1": [1.0, 2, 2, 2, 3, 3]})

        pd.testing.assert_frame_equal(left=actual_interp_df, right=expect_df)

    def test_bfill_startnan(self):
        """This function tests the bfill method of interpolation with nans at start
        and end of the series
        """
        # Given
        test_df = pd.DataFrame(data={"col1": [np.nan, np.nan, 1, 2, 3, 3, np.nan]})

        # When
        actual_interp_df = _interpolation(df=test_df, fill_method="bfill")

        # Then
        expect_df = pd.DataFrame(data={"col1": [1.0, 1, 1, 2, 3, 3, np.nan]})

        pd.testing.assert_frame_equal(left=actual_interp_df, right=expect_df)

    def test_rectilinear(self):
        """This function tests the rectiliniear method of interpolation"""
        # Given
        test_df = pd.DataFrame(
            data={
                "col1": [1, np.nan, np.nan, 2, np.nan, 3],
                "t": [0, 10, 20, 30, 40, 50],
            }
        )

        # When
        actual_interp_df = _interpolation(df=test_df, fill_method="rectilinear")

        # Then
        expect_df = pd.DataFrame(
            data={
                "col1": [1.0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3],
                "t": [0, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50],
            }
        )

        pd.testing.assert_frame_equal(
            left=actual_interp_df, right=expect_df[["t", "col1"]]
        )

    def test_rectilinear_startnan(self):
        """This function tests the rectiliniear method of interpolation with nans at
        start and end of the series
        """
        # Given
        test_df = pd.DataFrame(
            data={
                "col1": [np.nan, np.nan, 1, 2, 3, np.nan],
                "t": [0, 10, 20, 30, 40, 50],
            }
        )

        # When
        actual_interp_df = _interpolation(df=test_df, fill_method="rectilinear")

        # Then
        expect_df = pd.DataFrame(
            data={
                "col1": [np.nan, np.nan, np.nan, np.nan, 1, 1, 2, 2, 3, 3, 3],
                "t": [0, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50],
            }
        )

        pd.testing.assert_frame_equal(
            left=actual_interp_df, right=expect_df[["t", "col1"]]
        )
