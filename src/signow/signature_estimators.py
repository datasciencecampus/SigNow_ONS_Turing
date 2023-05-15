"""Signature Estimator Class."""
from dataclasses import dataclass, field

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from signow.signature_transformers import SigTransformerPipe
from signow.timeseries_data import TSData


@dataclass
class SigEstimator(BaseEstimator):
    """Estimator combining signature transformers and Linear regression models.

    Parameters
    ----------
    apply_pca : Boolean, optional
        Option to apply PCA based dimensionality reduction to the data.
        The default is False.
    factors : pd.DataFrame, optional
        Factor structure to be passed to PCA. The default is None.
    standardize : Boolean, optional
        Option to apply scikit learn standard scaling to data
        before regression. The default is False.
    regressor : String, optional
        Option to use sci-kit learn regression.
        The options are "elasticnet"/"lasso"/"ridge". If none of these are selected
        then Linear Regression will be applied.
    pca_params : Dict, optional
        Parameters to pass to PCA transformer. The default is {}.
    sig_params : TYPE, optional
        Parameters to pass to signature transformer. The default is {}.
    regress_params : TYPE, optional
        Parameters to pass to scikit learn regression model.
        The default is {}.
    """

    apply_pca: bool = False
    factors: pd.DataFrame = None
    standardize: bool = False
    regressor: str = "elasticnet"
    pca_params: dict = field(default_factory=dict)
    sig_params: dict = field(default_factory=dict)
    regress_params: dict = field(default_factory=dict)
    start_test: str = None

    def _select_estimator(self):
        """Use regressor string to return sklearn regression class."""
        if self.regressor == "elasticnet":
            return ElasticNet(**self.regress_params)
        elif self.regressor == "lasso":
            return Lasso(**self.regress_params)
        elif self.regressor == "ridge":
            return Ridge(**self.regress_params)
        else:
            return LinearRegression(**self.regress_params)

    def _estimator_pipeline(self):
        """Create sklearn pipeline instance with regression and optional scaler steps."""
        steps = []
        if self.standardize:
            steps.append(("scaler", StandardScaler()))
        steps.append(("regression", self._select_estimator()))
        return Pipeline(steps)

    def _transformer_pipeline(self):
        """Initialises the transformer pipeline with parameters."""
        param_keys = SigTransformerPipe().get_params().keys()
        return SigTransformerPipe(**{key: self.__dict__[key] for key in param_keys})

    def _fit(self, X, y, **params):
        """Fits signature estimator using X, y and params."""
        self.set_params(**params)

        # Create transformer and estimator instances
        self.transform_pipe_ = self._transformer_pipeline()
        self.estimator_ = self._estimator_pipeline()

        # Two of the transformers use y data, the full y has to be passed to fit
        # PCA implementation can't be 'fit', runs across full dataset
        data = TSData(X=X, y=y, start_test=self.start_test)
        x_train = data.X_train()
        self.x_mapped_ = data.X_mapped

        tr_X = self.transform_pipe_.fit_transform(x_train, y=y)

        tr_X = tr_X[tr_X.index.isin(x_train.index)]

        self.y_freq_ = data.y_freq

        y_mapped = data.y_mapped_train()

        if not self.sig_params["basepoint"]:
            y_mapped = y_mapped[1:]
            tr_X = tr_X[1:]

        self.estimator_ = self.estimator_.fit(tr_X, y_mapped)

    def fit(self, X, y, **params):
        """Fits custom signature estimator.

        Parameters
        ----------
        X : pd.DataFrame
            Data used to train custom signature estimator.
        y : pd.DataFrame
            Target data used to train custom signature estimator.
        **params : Dict
            Parameters needed to build custom signature estimator.

        Returns
        -------
        self
        """
        self._fit(X, y, **params)
        return self

    def predict(self, X):
        """Predict y given X using custom signature estimator. Keeps training
        X if needed for transformation pipeline.

        Parameters
        ----------
        X : pd.DataFrame
            New data used to predict target.

        Returns
        -------
        y : List
            Predictions for y from custom signature estimator.
        """
        min_date = X.index.min()

        pre_X = self.x_mapped_[self.x_mapped_.index < min_date]

        X = pre_X.append(X)

        tr_X = self.transform_pipe_.transform(X)
        tr_X = tr_X[min_date <= tr_X.index]

        pred = self.estimator_.predict(tr_X)
        y = pd.DataFrame(pred, tr_X.index, columns=["y"])
        return y.asfreq(self.y_freq_)

    def __call__(self, X, y, **params):
        return self.fit(X, y, **params)
