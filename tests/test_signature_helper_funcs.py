"""Tests for the signature_helper_functions module."""

from unittest import mock

import pytest
import pandas as pd

from conftest import create_dataframe
from signow.signature_functions.signature_helper_funcs import reduce_dim, apply_pca


@pytest.fixture
def _wide_indicators():
    """Wide format test input data.

    Randomly created using the data_generator module.
    """
    # fmt: off
    return create_dataframe([
        ('ref_date',   'a',                 'b',                   'c'),
        ('2000-02-01', None,                0.247951991379312,     -0.2773310967135312),
        ('2000-03-01', -1.4578115401073473, -0.4234475037179659,   -1.789680275229621),
        ('2000-04-01', -1.847418386996561,  -1.3526448447689943,   -0.48928024653680646),
        ('2000-05-01', 0.32154084969705643, 0.8967771423537796,    0.34303406993578855),
        ('2000-06-01', -1.8943011701844885, 0.19212490358890144,   0.9781103644451892),
        ('2000-07-01', 0.4721245577795181,  0.49014220553866905,   -0.4680315288368888),
        ('2000-08-01', -1.4066376509826117, 0.3911625605211986,    -0.7663773723346066),
        ('2000-09-01', 0.0405415443867394,  0.06610240866059103,   2.2825356609419547),
        ('2000-10-01', 1.3817513118967004,  1.0998237512077924,    1.2228265305921413),
        ('2000-11-01', 0.7394449097104836,  -0.49675510960366775,  -0.5804682425475034),
        ('2000-12-01', None,                1.2929896574781514,    -2.2520331446933364),
    ],
        index='ref_date'
    )
    # fmt: on


@pytest.fixture
def _factor_structure():
    """Test data for the factor groupings."""
    return {"alpha_group": ["a", "b"], "beta_group": ["c"]}


class TestReduceDim:
    """Tests for the reduce_dim function"""

    def test_no_factor_structure_backfill(self, _wide_indicators):
        """Test the function when:

        factor_structure = None AND
        fill_method = "backfill"
        """
        expected = create_dataframe(
            [
                ("ref_date", "global_0", "global_1"),
                ("2000-02-01", 0.62872571, -0.01620905),
                ("2000-03-01", 1.50942487, -1.00387557),
                ("2000-04-01", 2.39475744, 0.30124286),
                ("2000-05-01", -1.1505705, 0.1476323),
                ("2000-06-01", 0.73315152, 1.0104155),
                ("2000-07-01", -0.72517532, -0.38035679),
                ("2000-08-01", 0.54588039, -0.44141947),
                ("2000-09-01", -0.52949159, 1.91423741),
                ("2000-10-01", -2.12445708, 0.69441093),
                ("2000-11-01", 0.05578171, -0.2091114),
                ("2000-12-01", -1.33802715, -2.01696672),
            ],
            index="ref_date",
        )
        # Call the function and obtain the result
        result, _ = reduce_dim(_wide_indicators, 2, fill_method="backfill")

        # Assert the outputs value against the expected
        pd.testing.assert_frame_equal(result, expected)

    def test_no_factor_structure(self, _wide_indicators):
        """Test the function when:
        factor_structure = None AND
        fill_method = "fill-em"
        """
        # Define the expected output
        expected = create_dataframe(
            [
                ("ref_date", "global_0", "global_1"),
                ("2000-02-01", 0.05285191, -0.08144462),
                ("2000-03-01", 1.42341593, -1.20363547),
                ("2000-04-01", 2.48047923, -0.13810531),
                ("2000-05-01", -0.99145727, 0.36289275),
                ("2000-06-01", 0.94389655, 1.03015525),
                ("2000-07-01", -0.64590902, -0.29282974),
                ("2000-08-01", 0.55644135, -0.39441373),
                ("2000-09-01", -0.12474247, 1.92918892),
                ("2000-10-01", -1.84777453, 0.98630964),
                ("2000-11-01", 0.1562307, -0.39948146),
                ("2000-12-01", -2.00343238, -1.79863623),
            ],
            index="ref_date",
        )
        # Call the function and obtain the result
        result, _ = reduce_dim(_wide_indicators, 2, fill_method="fill-em")

        # Assert the outputs value against the expected
        pd.testing.assert_frame_equal(result, expected)

    def test_factor_structure(self, _wide_indicators, _factor_structure):
        """Test the function when factor_structure is defined and
        fill_method = "backfill"
        """
        expected = create_dataframe(
            [
                ("ref_date", "alpha_group_0", "beta_group_0"),
                ("2000-02-01", -0.61765586, -0.0911089),
                ("2000-03-01", -1.26524433, -1.29983092),
                ("2000-04-01", -2.39878912, -0.26050602),
                ("2000-05-01", 1.09192439, 0.40470851),
                ("2000-06-01", -0.93735956, 0.91228356),
                ("2000-07-01", 0.79142798, -0.24352331),
                ("2000-08-01", -0.4483553, -0.48197168),
                ("2000-09-01", 0.11955866, 1.95482559),
                ("2000-10-01", 1.93352154, 1.10786922),
                ("2000-11-01", 0.00234985, -0.33338664),
                ("2000-12-01", 1.72862176, -1.66935941),
            ],
            index="ref_date",
        )

        # Call the function and obtain the result
        result, _ = reduce_dim(
            _wide_indicators,
            1,
            factor_structure=_factor_structure,
            fill_method="backfill",
        )

        # Assert the outputs value against the expected
        pd.testing.assert_frame_equal(result, expected)


class TestApplyPCA:
    """Tests for the apply_pca function."""

    @pytest.fixture()
    def _input_model_params(self):
        return {"k": 2, "not_k": 0, "pca_fill_method": "backfill"}

    @mock.patch("signow.signature_functions.signature_helper_funcs.reduce_dim")
    def test_df_grouping(
        self, mock_reduce_dim, _wide_indicators, _factor_structure, _input_model_params
    ):
        """Test the function when df_grouping is not None."""
        input_grouping = create_dataframe(
            [
                ("factor_group", "factor_name"),
                ("alpha_group", "a"),
                ("alpha_group", "b"),
                ("beta_group", "c"),
            ]
        )

        mock_reduce_dim.return_value = (1, 2)

        apply_pca(_wide_indicators, _input_model_params, input_grouping)

        mock_reduce_dim.assert_called_once_with(
            _wide_indicators,
            2,
            factor_structure=_factor_structure,
            fill_method="backfill",
            pca_mu_sigma_eigenvals=None,
        )

    @mock.patch("signow.signature_functions.signature_helper_funcs.reduce_dim")
    def test_no_factor_structure(
        self, mock_reduce_dim, _wide_indicators, _input_model_params
    ):
        mock_reduce_dim.return_value = (1, 2)

        apply_pca(_wide_indicators, _input_model_params, None, None)

        mock_reduce_dim.assert_called_once_with(
            _wide_indicators,
            2,
            factor_structure=None,
            fill_method="backfill",
            pca_mu_sigma_eigenvals=None,
        )
