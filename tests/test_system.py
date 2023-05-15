""" System tests for SigNow against known outputs of the pipeline."""
import pytest

from signow.data_generator import create_data
from signow.signature_nowcasting import SigNowcaster


class TestSigNow:
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
        """returns parameters for the model

        Returns:
            sn_params: dictionary containing standard set of parameters
        """

        sig_params = {
            "window_type": "ind",
            "max_length": 365,
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

    def test_elasticnet_PCA(self):
        """Tests SigNow when the regressor is
        set to elasticnet and PCA is True.
        """
        # Given
        sn_params = self._return_params()

        indicator_df, target_df = self._generate_data()

        # When
        sn_ = SigNowcaster(X=indicator_df, y=target_df, **sn_params)
        actual_sn_result = sn_.static_nowcast(sn_.data.X_ref())

        # Then
        expected_sn_result = 0.8051748858631005

        assert actual_sn_result["y"][0] == pytest.approx(expected_sn_result)

    def test_ridge_PCA_standardizer(self):
        """Tests SigNow when the regressor is
        set to ridge, PCA and standardizer are True.
        """
        # Given
        sn_params = self._return_params()
        sn_params["regressor"] = "ridge"
        sn_params["standardize"] = True
        sn_params["regress_params"].pop("l1_ratio")

        indicator_df, target_df = self._generate_data()

        # When
        sn_ = SigNowcaster(X=indicator_df, y=target_df, **sn_params)
        actual_sn_result = sn_.static_nowcast(sn_.data.X_ref())

        # Then
        expected_sn_result = 0.754236337226096

        assert actual_sn_result["y"][0] == pytest.approx(expected_sn_result)

    def test_lasso(self):
        """Tests SigNow when the regressor is
        set to lasso.
        """
        # Given
        sn_params = self._return_params()
        sn_params["regressor"] = "lasso"
        sn_params["apply_pca"] = False
        sn_params["regress_params"].pop("l1_ratio")

        indicator_df, target_df = self._generate_data()

        # When
        sn_ = SigNowcaster(X=indicator_df, y=target_df, **sn_params)
        actual_sn_result = sn_.static_nowcast(sn_.data.X_ref())

        # Then
        expected_sn_result = -0.9694009360520486

        assert actual_sn_result["y"][0] == pytest.approx(expected_sn_result)
