# -*- coding: utf-8 -*-
import pandas as pd

from signow.data_generator import create_data
from signow.signature_nowcasting import SigNowcaster


class TestSignatureNowcasting:
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

    def test_recursive_nowcast(self):
        # Given
        sn_params = self._return_params()

        indicator_df, target_df = self._generate_data()

        # When

        sn_ = SigNowcaster(
            X=indicator_df, y=target_df, start_test="2012-01-01", **sn_params
        )

        x_test = sn_.data.X_test()

        actual_sn_result = sn_.recursive_nowcast(x_test)

        # Then
        assert isinstance(actual_sn_result, pd.DataFrame)
        assert len(actual_sn_result) == 6
        assert actual_sn_result.index.min() < actual_sn_result.index.max()
