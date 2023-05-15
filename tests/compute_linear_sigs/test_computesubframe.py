"""Unit tests for compute_subframe.

When `compute_subframe` is called it gets passed the entire dataset, an integer
that relates to a row of the passed dataset and the configs for the
model. It has three key behaviours - depending on the desired windowtype

The parameters passed to `compute_subframe` are:

    df:
        The entire dataset.
    level:
        Row - this is defined in the main loop in compute_sig_dates.
    configs:
        Dictionary of parameters.

"""
import hypothesis.strategies as st
import numpy as np
import pandas as pd
from hypothesis import given, assume

from signow.signature_functions.compute_linear_sigs import compute_subframe


class TestComputeSubFrame:
    def _create_dtindext_df(self, rows):
        """Create a DataFrame of zeros of length n_rows indexed by dates."""
        vals = {"col1": np.zeros(rows, dtype=int)}
        ind = pd.date_range("2000-01-01", periods=rows, freq="MS")
        return pd.DataFrame(data=vals, index=ind)

    @given(
        dtrows=st.integers(min_value=15, max_value=200),
        maxlength=st.integers(min_value=1, max_value=30),
        row_ind=st.integers(min_value=1, max_value=20),
    )
    def test_windowtype_ind(self, dtrows, maxlength, row_ind):
        """Tests the subframe functionality when specified by index.

        Parameters
        ----------
            dtrows : int
                Number of rows for test_df.
            maxlength : int
                Max length of the frame by index.
            row_ind : int
                Index of the current row (horizon).

        """
        assume(dtrows > maxlength)
        assume(dtrows > row_ind)
        assume(row_ind > maxlength)

        # Given
        test_df = self._create_dtindext_df(rows=dtrows)
        configs = {"window_type": "ind", "max_length": maxlength}

        # When
        actual_subframe = compute_subframe(test_df, row_ind, configs)

        # Then
        expect_startdate = test_df.index[row_ind - maxlength]
        expect_enddate = test_df.index[row_ind - 1]

        actual_startdate = actual_subframe.index[0]
        actual_enddate = actual_subframe.index[-1]

        assert actual_startdate == expect_startdate
        assert actual_enddate == expect_enddate

    @given(row_ind=st.integers(min_value=1, max_value=14))
    def test_windowtype_days_365(self, row_ind):
        """Tests the subframe functionality when the windowtype
        for the signatures is specified in days.

        This tests for when that is set to 365.

        Parameters
        ----------
            row_ind : int
                The current row (horizon).

        """
        # Given
        test_df = self._create_dtindext_df(rows=15)
        configs = {"window_type": "days", "max_length": 365}

        # When
        actual_subframe = compute_subframe(df=test_df, ind=row_ind, configs=configs)

        # Then
        amount_of_months = configs["max_length"] / (365 / 12)
        if row_ind <= amount_of_months:
            expect_startdate = test_df.index[0]
            expect_enddate = test_df.index[row_ind - 1]

        else:
            expect_startdate = test_df.index[row_ind] - pd.DateOffset(
                months=amount_of_months
            )
            expect_enddate = test_df.index[row_ind - 1]

        actual_startdate = actual_subframe.index[0]
        actual_enddate = actual_subframe.index[-1]

        assert actual_startdate == expect_startdate
        assert actual_enddate == expect_enddate

    @given(row_ind=st.integers(min_value=1, max_value=14))
    def test_windowtype_days_150(self, row_ind):
        """Tests the subframe functionality when the windowtype
        for the signatures is specified in days.

        This tests for when that is set to 150.

        Parameters
        ----------
            row_ind : int
                The current row (horizon).

        """
        # Given
        test_df = self._create_dtindext_df(rows=15)
        configs = {"window_type": "days", "max_length": 150}

        # When
        actual_subframe = compute_subframe(df=test_df, ind=row_ind, configs=configs)

        # Then
        amount_of_months = round(configs["max_length"] / (365 / 12))
        if row_ind <= amount_of_months:
            expect_startdate = test_df.index[0]
            expect_enddate = test_df.index[row_ind - 1]

        else:
            expect_startdate = test_df.index[row_ind] - pd.DateOffset(
                months=amount_of_months
            )
            expect_enddate = test_df.index[row_ind - 1]

        actual_startdate = actual_subframe.index[0]
        actual_enddate = actual_subframe.index[-1]

        assert actual_startdate == expect_startdate
        assert actual_enddate == expect_enddate

    @given(row_ind=st.integers(min_value=1, max_value=15))
    def test_windowtype_days_730(self, row_ind):
        """Tests the subframe functionality when the windowtype
        for the signatures is specified in days.

        This tests for when that is set to 730 (i.e. 2 years).

        Parameters
        ----------
            row_ind : int
                The current row (horizon).
        """
        # Given
        test_df = self._create_dtindext_df(rows=30)
        configs = {"window_type": "days", "max_length": 730}

        # When
        actual_subframe = compute_subframe(df=test_df, ind=row_ind, configs=configs)

        # Then
        amount_of_months = round(configs["max_length"] / (365 / 12))
        if row_ind <= amount_of_months:
            expect_startdate = test_df.index[0]
            expect_enddate = test_df.index[row_ind - 1]

        else:
            expect_startdate = test_df.index[row_ind] - pd.DateOffset(
                months=amount_of_months
            )
            expect_enddate = test_df.index[row_ind - 1]

        actual_startdate = actual_subframe.index[0]
        actual_enddate = actual_subframe.index[-1]

        assert actual_startdate == expect_startdate
        assert actual_enddate == expect_enddate

    @given(row_ind=st.integers(min_value=1, max_value=89))
    def test_windowtype_days_1825(self, row_ind):
        """Tests the subframe functionality when the windowtype
        for the signatures is specified in days.

        This tests for when that is set to 1825 (i.e. 5 years).

        Parameters
        ----------
            row_ind : int
                The current row (horizon).

        """
        # Given
        test_df = self._create_dtindext_df(rows=90)
        configs = {"window_type": "days", "max_length": 1825}

        # When
        actual_subframe = compute_subframe(df=test_df, ind=row_ind, configs=configs)

        # Then
        amount_of_months = round(configs["max_length"] / (365 / 12))
        if row_ind <= amount_of_months:
            expect_startdate = test_df.index[0]
            expect_enddate = test_df.index[row_ind - 1]

        else:
            expect_startdate = test_df.index[row_ind] - pd.DateOffset(
                months=amount_of_months
            )
            expect_enddate = test_df.index[row_ind - 1]

        actual_startdate = actual_subframe.index[0]
        actual_enddate = actual_subframe.index[-1]

        assert actual_startdate == expect_startdate
        assert actual_enddate == expect_enddate

    @given(
        dtrows=st.integers(min_value=15, max_value=200),
        row_ind=st.integers(min_value=1, max_value=20),
    )
    def test_windowtype_false(self, dtrows, row_ind):
        """Tests the subframe functionality when the windowtype
        for the signatures is False.

        When it is specified to false it returns the subframe from the
        start of the passed dataset row_ind given.

        Parameters
        ----------
            dtrows : int
                Number of rows to use for the test data.
            row_ind : int
                Value to pass as the current row index.

        """
        assume(dtrows > row_ind)
        # Given
        test_data = self._create_dtindext_df(dtrows)
        configs = {"max_length": False}

        # When
        actual_subframe = compute_subframe(test_data, row_ind, configs)

        # Then
        expect_startdate = test_data.index[0]
        expect_enddate = test_data.index[row_ind - 1]

        actual_startdate = actual_subframe.index[0]
        actual_enddate = actual_subframe.index[-1]

        assert actual_startdate == expect_startdate
        assert actual_enddate == expect_enddate
