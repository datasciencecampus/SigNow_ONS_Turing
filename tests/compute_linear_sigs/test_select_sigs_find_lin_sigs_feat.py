"""Unit tests for select_signatures and find_linear_sig_features."""
import hypothesis.strategies as st
import numpy as np
import pandas as pd
from hypothesis import given, assume

from signow.signature_functions.compute_linear_sigs import (
    select_signatures,
    find_linear_sig_features,
    all_sig_keys,
)


class TestFindLinearSigPropertyBased:
    @given(
        level_variables=st.integers(min_value=0, max_value=5),
        level_time=st.integers(min_value=0, max_value=5),
    )
    def test_find_lin_sig_feat_inner_same_corr_len(self, level_variables, level_time):
        """Tests the correct length of returned linear index
        when both the truncation level for the variables and
        time are the same, and when the keep sigs setting is
        set to innermost

        Parameters
        ----------
            level_variables : int
                Truncation level of variables.
            level_time : int
                Truncation level of time param.

        """
        assume(level_variables == level_time)

        # Given
        dim = 5
        keep_sigs = "innermost"

        # When
        actual_lin_sig_inner_ind = find_linear_sig_features(
            dim=dim, var_level=level_variables, t_level=level_time, keep_sigs=keep_sigs
        )

        # Then
        n_indices = len(actual_lin_sig_inner_ind)

        n_expected = (dim * level_variables) + 1

        assert n_indices == n_expected

    @given(level_variables=st.integers(min_value=1, max_value=5))
    def test_find_lin_sig_feat_inner_tlvl0_corr_len(self, level_variables):
        """Tests the correct length of returned linear index
        when truncation level for time is set to zero, and when
        the keep sigs setting is set to innermost. The behaviour
        in find_linear_sig_features when truncation level for
        time is 0 is to make it equal to level for variables.

        Parameters
        ----------
            level_variables : int
                Truncation level of variables.

        """
        # Given
        dim = 5
        level_time = 0
        keep_sigs = "innermost"

        # When
        actual_lin_sig_inner_ind = find_linear_sig_features(
            dim=dim, var_level=level_variables, t_level=level_time, keep_sigs=keep_sigs
        )

        # Then
        n_indices = len(actual_lin_sig_inner_ind)

        n_expected = (dim * level_variables) + 1

        assert n_indices == n_expected


class TestFindLinearSigOutputBased:
    def test_find_lin_sig_feat_none(self):
        """Tests the expected output when `keep_sigs` is set to None.

        This will return an empty list.

        """

        # Given
        dim = 5
        level_variables = 2
        level_time = 2
        keep_sigs = None

        # When
        actual_lin_sig_inner_ind = find_linear_sig_features(
            dim=dim, var_level=level_variables, t_level=level_time, keep_sigs=keep_sigs
        )

        # Then
        expected_ind = []

        assert actual_lin_sig_inner_ind == expected_ind

    def test_find_lin_sig_feat_alllinear_same(self):
        """Tests the expected output when `keep_sigs`
        is set to all_linear and the the truncation levels
        for variables and time are the same.

        This will return an empty list.

        """

        # Given
        dim = 5
        level_variables = 2
        level_time = 2
        keep_sigs = "all_linear"

        # When
        actual_lin_sig_inner_ind = find_linear_sig_features(
            dim=dim, var_level=level_variables, t_level=level_time, keep_sigs=keep_sigs
        )

        # Then
        expected_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

        assert actual_lin_sig_inner_ind == expected_ind

    def test_find_lin_sig_feat_alllinear_vlvldiff(self):
        """Tests the expected output when `keep_sigs`
        is set to all_linear and the the truncation levels
        for variables is larger than for time.

        """

        # Given
        dim = 5
        level_variables = 3
        level_time = 2
        keep_sigs = "all_linear"

        # When
        actual_lin_sig_inner_ind = find_linear_sig_features(
            dim=dim, var_level=level_variables, t_level=level_time, keep_sigs=keep_sigs
        )

        # Then
        expected_ind = [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
        ]

        assert actual_lin_sig_inner_ind == expected_ind

    def test_find_lin_sig_feat_alllinear_tlvldiff(self):
        """Tests the expected output when `keep_sigs`
        is set to all_linear and the the truncation levels
        for time is larger than the level for variables.

        """

        # Given
        dim = 5
        level_variables = 2
        level_time = 3
        keep_sigs = "all_linear"

        # When
        actual_lin_sig_inner_ind = find_linear_sig_features(
            dim=dim, var_level=level_variables, t_level=level_time, keep_sigs=keep_sigs
        )

        # Then
        expected_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

        assert actual_lin_sig_inner_ind == expected_ind

    def test_find_lin_sig_feat_innermost_same(self):
        """Tests the expected output when `keep_sigs`
        is set to innermost and the the truncation levels
        for variables and time are the same.

        """

        # Given
        dim = 5
        level_variables = 2
        level_time = 2
        keep_sigs = "innermost"

        # When
        actual_lin_sig_inner_ind = find_linear_sig_features(
            dim=dim, var_level=level_variables, t_level=level_time, keep_sigs=keep_sigs
        )

        # Then
        expected_ind = [0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14]

        assert actual_lin_sig_inner_ind == expected_ind

    def test_find_lin_sig_feat_innermost_vlvldiff(self):
        """Tests the expected output when `keep_sigs`
        is set to innermost and the the truncation levels
        for variables is larger than for time.

        """

        # Given
        dim = 5
        level_variables = 3
        level_time = 2
        keep_sigs = "innermost"

        # When
        actual_lin_sig_inner_ind = find_linear_sig_features(
            dim=dim, var_level=level_variables, t_level=level_time, keep_sigs=keep_sigs
        )

        # Then
        expected_ind = [0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 24, 25, 26, 27]

        assert actual_lin_sig_inner_ind == expected_ind

    def test_find_lin_sig_feat_innermost_tlvldiff(self):
        """Tests the expected output when keep_sigs
        is set to innermost and the the truncation levels
        for time is larger than the level for variables.

        """

        # Given
        dim = 5
        level_variables = 2
        level_time = 3
        keep_sigs = "innermost"

        # When
        actual_lin_sig_inner_ind = find_linear_sig_features(
            dim=dim, var_level=level_variables, t_level=level_time, keep_sigs=keep_sigs
        )

        # Then
        expected_ind = [0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15]

        assert actual_lin_sig_inner_ind == expected_ind


class TestSelectSigsOutputBased:
    def _generate_sigarray(
        self, col_length: int, row_length: int, keys: list
    ) -> pd.DataFrame:
        """Generates an an array of floats given a specified length

        Parameters
        ----------
            col_length : int
                Truncation level of variables.
            row_length : int
                Truncation level of time param.
            keys : list
                List of sig string titles.

        Returns
        -------
            pd.DataFrame
                Mock signature.
        """
        sig_paths = np.random.rand(col_length, row_length)
        generated_sig = pd.DataFrame(data=sig_paths, columns=keys)
        return generated_sig

    def test_select_sigs_lvlsame_all(self):
        """Tests the expected output of select_signatures
        when `keep_sigs` is set to 'all'.

        There are three behaviours select signatures performs when
        keeps_sigs == all depending on what truncation levels are set
        for the variables or time.

        This test tests the output when truncation levels are the
        same. This returns the same sigs.

        """
        # Given
        dim = 5
        level_time = 2
        level_variables = 2
        keys = all_sig_keys(dim=dim, level=level_variables)
        df_sigs = self._generate_sigarray(col_length=31, row_length=31, keys=keys)
        configs = {"keep_sigs": "all", "level": level_variables}

        # When
        actual_sigs = select_signatures(
            df_sigs=df_sigs, dim=dim, t_level=level_time, configs=configs
        )

        # Then
        expected_sigs = df_sigs.copy()

        assert all(actual_sigs == expected_sigs)

    def test_select_sigs_vlvldiff_all(self):
        """Tests the expected output of select_signatures
        when `keep_sigs` is set to 'all'.

        There are three behaviours select signatures performs when
        keeps_sigs == all depending on what truncation levels are
        set for the variables or time.

        This test tests the output when the truncation level
        for variables is higher than for time. This removes the
        time term.

        """

        # Given
        dim = 5
        level_time = 2
        level_variables = 3
        keys = all_sig_keys(dim=dim, level=level_variables)
        df_sigs = self._generate_sigarray(col_length=31, row_length=156, keys=keys)
        configs = {"keep_sigs": "all", "level": level_variables}

        # When
        actual_sigs = select_signatures(
            df_sigs=df_sigs, dim=dim, t_level=level_time, configs=configs
        )

        # Then
        filt = "(1, 1, 1)"
        expected_sigs = df_sigs.iloc[:, df_sigs.columns != filt]

        assert all(actual_sigs == expected_sigs)
