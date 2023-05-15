"""Signature Nowcasting Module."""

import numpy as np
import pandas as pd

from signow.signature_estimators import SigEstimator
from signow.timeseries_data import TSData
from signow.utils.validation import validate_model_params


class SigNowcaster:
    """Class responsible for handling data and methods for nowcasting."""

    def __init__(self, X, y, **params):
        """Initialise `SigNowcaster` with data and estimator configuration.

        Parameters
        ----------
        X : pd.DataFrame
            Data used to train, test and predict target y.
        y : pd.DataFrame
            Target data used to train custom signature estimator.
        **params :
            keyword arguments for data and estimator classes.

        """
        data_params, est_params = self._parse_params(**params)
        self.data_params = data_params
        self.data = TSData(X, y, **data_params)
        self.est_params = est_params
        self.estimator = SigEstimator(est_params)
        self.train_rmse_ = None
        self.test_rmse_ = None
        self.train_predict_ = None
        self.train_residuals_ = None
        self.test_predict_ = None
        self.test_residuals_ = None
        self.predict_ = None
        self.model_ = None

    @staticmethod
    def split_params(sub_attrs, **params):
        """Extract a subset of parameters from the full set.

        Parameters
        ----------
        sub_attrs : List
            Set of keys for the parameters we want to extract.

        Returns
        -------
        sub_params
            The kw parameters that have been extracted.
        params
            The original params with sub_params removed.

        """
        shared_keys = sub_attrs.keys() & params.keys()
        sub_params = {k: params.pop(k) for k in shared_keys}
        return sub_params, params

    def _parse_params(self, **params):
        """Split keyword parameters into data and estimator subsets.

        Returns
        -------
        data_params
            Data parameters
        params
            Estimator parameters
        """
        ts_data_attrs = TSData().get_params()
        data_params, params = self.split_params(ts_data_attrs, **params)

        if params:
            params = validate_model_params(params)

        return data_params, params

    def _clear_post_fit_attr(self):
        """Clear the post fit attributes before new fit of model.

        This is done by setting all attributes with the suffix '_' to None.
        """
        post_fit_attrs = (attr for attr in self.__dict__ if attr.endswith("_"))
        for attr in post_fit_attrs:
            self.__setattr__(attr, None)

    def _fit(self, **params):
        """Update existing data and estimator parameters (with parameters provided as function
        arguments) and refit model.
        """
        data_params, est_params = self._parse_params(**params)
        self.data_params.update(data_params)
        self.data.update(**self.data_params)
        self.est_params.update(est_params)

        self.model_ = self.estimator.fit(
            self.data.X, self.data.y, start_test=self.data.start_test, **self.est_params
        )

    def fit(self, **params):
        """Fit model using existing and updated data and estimator parameters."""
        self._fit(**params)
        return self

    def predict(self, X=None):
        """Predict y for X given a model that is already fitted.

        Parameters
        ----------
        X : pd.DataFrame optional
            Training dataframe, by default None

        Returns
        -------
        pd.DataFrame
            Dataframe of predictions
        """
        if not isinstance(X, pd.DataFrame):
            X = self.data.X_ref()
        return self.model_.predict(X)

    def static_nowcast(self, X=None, **params):
        """Fit and predict model using existing and updated configurations.

        The model is not updated.
        """
        self._clear_post_fit_attr()
        self.fit(**params)

        x_train = self.data.X_train()
        x_test = self.data.X_test()

        self.train_predict_ = self.model_.predict(x_train)
        self.test_predict_ = self.model_.predict(x_test)

        self.calc_fit_statistics()

        self.predict_ = self.predict(X)
        return self.predict_

    def recursive_nowcast(self, X=None, **params):
        """Fit and predict model using existing and updated configurations.

        The model updates for each period in the test window.
        """
        self._clear_post_fit_attr()

        pred = []

        train_predictions = None

        start, end = self.data.start_test, self.data.end_test
        if start == end:
            self._fit(**params)
        else:
            dates = pd.date_range(start, end, freq=self.data.y_freq)

            for start_test in dates:
                current_date_params = {
                    key: vars(self.data)[key] for key in {"start_train", "start_ref"}
                }
                start_test_update = {"start_test": start_test, "end_test": start_test}

                new_date_params = {**current_date_params, **start_test_update}
                new_params = {**params, **new_date_params}

                self._fit(**new_params)

                if train_predictions is None:
                    x_train = self.data.X_train()
                    train_predictions = self.model_.predict(x_train)

                pred.append(self.model_.predict(self.data.X_test()))

            current_date_params = {
                key: vars(self.data)[key]
                for key in {"start_train", "start_ref", "end_test"}
            }
            start_test_update = {"start_test": start, "end_test": end}
            new_date_params = {**current_date_params, **start_test_update}
            self.data.update(**new_date_params)
        if pred:
            self.train_predict_ = train_predictions

        self.calc_fit_statistics()

        self.predict_ = self.predict(X)
        return self.predict_

    def _calc_rmse(self, res):
        """Calculate the root mean squared error of residuals.

        Parameters
        ----------
        res : List
            List of residuals.

        Returns
        -------
        float
            Root mean squared error of the residuals (RMSE).
        """
        return np.sqrt(np.sum(res**2) / len(res)).values[0]

    def calc_fit_statistics(self):
        """Calculate residuals and RMSE for train and test datasets."""
        self.train_residuals_ = self.data.y_train() - self.train_predict_
        self.train_rmse_ = self._calc_rmse(self.train_residuals_)

        if isinstance(self.test_predict_, pd.DataFrame):
            self.test_residuals_ = self.data.y_test() - self.test_predict_
            self.test_rmse_ = self._calc_rmse(self.test_residuals_)

    def __call__(self):
        """Fit and predict model using existing and updated configurations.

        The model is not updated.
        """
        self.static_nowcast()
