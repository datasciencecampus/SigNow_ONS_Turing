"""Tests for the signature transformer module."""
import numpy as np
import pandas as pd
from statsmodels.multivariate.pca import PCA

from signow.data_generator import create_data
from signow.signature_functions.signature_helper_funcs import standardize_df
from signow.signature_transformers import PCATransformer
from signow.timeseries_data import TSData


class TestPCATransformer:
    def _generate_data(self):
        """generates target and indicator data.
        Note: This generates the same target and indicator
        data that was used when testing on the pipeline.
        Arguments into function create_data are hard coded and
        shouldn't be changed, otherwise tests will fail.

        Returns:
            indicators, target: pd.DataFrames
        """
        indicators, target = create_data(
            start_date="1999-12-01",
            end_date="2013-12-01",
            num_indicators=3,
            wide_indic_df=False,
        )

        target = target[:-3]
        indicators = indicators[indicators.index < "2013-07-01"]

        return indicators, target

    def _return_params(self):
        """returns parameters for the model.

        Returns:
            sn_params: dictionary containing standard set of parameters
        """

        sig_params = {
            "window_type": "ind",
            "max_length": 5,
            "fill_method": "ffill",
            "level": 2,
            "t_level": 2,
            "basepoint": True,
            "use_multiplier": True,
            "keep_sigs": "all",
        }
        model_params = {"alpha": 0.1, "l1_ratio": 0.5, "fit_intercept": False}
        other_params = {
            "end_train": "2013-03-31",
            "regressor": "elasticnet",
            "apply_pca": True,
            "standardize": False,
        }
        sn_params = {
            "regress_params": {**model_params},
            "sig_params": {**sig_params},
            "pca_params": {"pca_fill_method": "backfill", "k": 2},
            **other_params,
        }

        return sn_params

    def test_pca_transformer(self):
        # Given
        indicator_df, target_df = self._generate_data()

        data = TSData(indicator_df, target_df)
        x_train, x_test, y_train, y_test = data.split()

        pca = PCATransformer()

        # When
        actual_components, y = pca.fit_transform(x_train, y_train)

        # Then
        assert len(actual_components.columns) == 2
        assert "global_1" in actual_components.columns
        assert len(actual_components) == len(x_train)
        pd.testing.assert_frame_equal(y, y_train)

    def test_pca_factor_transformer(self):
        # Given
        indicator_df, target_df = self._generate_data()

        data = TSData(indicator_df, target_df)
        x_train, x_test, y_train, y_test = data.split()

        factors = pd.DataFrame(
            {"factor_group": ["alpha", "beta", "alpha"], "factor_name": ["a", "b", "c"]}
        )
        pca = PCATransformer(factors=factors, k=(2, 1))

        # When
        actual_components, y = pca.fit_transform(x_train, y_train)

        # Then
        assert len(actual_components.columns) == 3
        assert "alpha_0" in actual_components.columns
        assert "alpha_1" in actual_components.columns
        assert "beta_0" in actual_components.columns
        assert len(actual_components) == len(x_train)
        pd.testing.assert_frame_equal(y, y_train)

    def test_pca_transformer_train_test(self):
        # Given
        indicator_df, target_df = self._generate_data()

        data = TSData(indicator_df, target_df)
        x_train, x_test, y_train, y_test = data.split()

        pca = PCATransformer()

        # When
        actual_components, y = pca.fit_transform(x_train, y_train)

        train_components = pca.transform(x_train)
        test_components = pca.transform(x_test)

        # Then
        pd.testing.assert_frame_equal(actual_components, train_components)
        assert len(test_components) == len(x_test)

    def test_pca_factor_transformer_train_test(self):
        # Given
        indicator_df, target_df = self._generate_data()

        data = TSData(indicator_df, target_df)
        x_train, x_test, y_train, y_test = data.split()

        factors = pd.DataFrame(
            {"factor_group": ["alpha", "beta", "alpha"], "factor_name": ["a", "b", "c"]}
        )
        pca = PCATransformer(factors=factors, k=(2, 1))

        # When

        actual_components, y = pca.fit_transform(x_train, y_train)

        train_components = pca.transform(x_train)
        test_components = pca.transform(x_test)

        # Then
        pd.testing.assert_frame_equal(actual_components, train_components)
        assert len(test_components) == len(x_test)

    def test_check_pca_application_params_no_standardisation(self):
        # Given
        indicator_df, target_df = self._generate_data()

        data = TSData(indicator_df, target_df)
        x_train, x_test, y_train, y_test = data.split()

        # When
        res_pca = PCA(
            x_train,
            ncomp=2,
            method="eig",
            normalize=False,
            missing=None,
            standardize=False,
            demean=False,
        )

        pca_arr = np.dot(x_train, res_pca.eigenvecs)
        df_pca = pd.DataFrame(
            data=pca_arr, columns=["comp_0", "comp_1"], index=x_train.index
        )

        # Then
        pd.testing.assert_frame_equal(res_pca.factors.asfreq("MS"), df_pca)

    def test_check_pca_application_params_standardisation(self):
        # Given
        indicator_df, target_df = self._generate_data()

        data = TSData(indicator_df, target_df)
        x_train, x_test, y_train, y_test = data.split()

        # When
        res_pca = PCA(x_train, ncomp=2, method="eig", normalize=False, missing=None)

        x_train_standardised = standardize_df(x_train, res_pca._mu, res_pca._sigma)
        pca_arr = np.dot(x_train_standardised, res_pca.eigenvecs)
        df_pca = pd.DataFrame(
            data=pca_arr, columns=["comp_0", "comp_1"], index=x_train.index
        )

        # Then
        pd.testing.assert_frame_equal(res_pca.factors.asfreq("MS"), df_pca)
