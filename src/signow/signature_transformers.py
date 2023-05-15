"""Transformers that change aspects of the data and Signature."""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

import signow.signature_functions.compute_linear_sigs as sig_funcs
import signow.signature_functions.signature_helper_funcs as sig_helpers
from signow.timeseries_data import TSData


class PCATransformer(BaseEstimator, TransformerMixin):
    """Apply statsmodels PCA, given an optional factor structure."""

    def __init__(self, factors=None, pca_fill_method="backfill", k=2):
        """Initialise PCA transformer with optional keyword or default parameters.

        Parameters
        ----------
        factors : pd.DataFrame, optional
            Factor structure to use in PCA method.
        pca_fill_method : str, optional
            Method used to fill in missing data before PCA analysis.
            The default is "backfill".
        k : integer/list, optional
            Integer/List of integers specifying number of principal components
            to find. The default is 2.
        """
        self.factors = factors
        self.pca_fill_method = pca_fill_method
        self.k = k
        self.y_ = None
        self.mu_sig_eigenvecs_ = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None, **params):
        """Set PCA Transformer parameters.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing data to be transformed.
        y : pd.DataFrame, optional
            Target passed in but unused. The default is None.
        **params : Dict
            Any parameters that are existing attributes can be updated.

        Returns
        -------
            self - the class instance
        """
        if params:
            self.set_params(**params)

        self.y_ = y

        pca_params = self.get_params(deep=False)

        X, mu_sig_eigenvecs = sig_helpers.apply_pca(X, pca_params, self.factors)

        self.mu_sig_eigenvecs_ = mu_sig_eigenvecs
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform X using statsmodels PCA functions. Y is passed but unused.

        Parameters
        ----------
        X : pd.DataFrame
            Data to be transformed using PCA.

        Returns
        -------
        X_pca : pd.DataFrame
            Transformed data.
        """
        pca_params = self.get_params(deep=False)
        if isinstance(X, tuple):
            X, y = X
        X, _ = sig_helpers.apply_pca(
            X, pca_params, self.factors, self.mu_sig_eigenvecs_
        )
        return X

    def fit_transform(self, X: pd.DataFrame, y: pd.DataFrame = None, **params):
        """Fit and Transform X via statsmodel PCA functions.

        Parameters
        ----------
        X : pd.DataFrame
            Data to be transformed using PCA.
        y : pd.DataFrame, optional
            Target dataframe, passed but unused. The default is None.
        **params : Dictlike
            Any parameters that are existing attributes can be set during
            fitting.

        Returns
        -------
        X : pd.DataFrame
            Transformed DataFrame.
        y : pd.DataFrame
            Target DataFrame unchanged.
        """
        if isinstance(X, tuple):
            X, y = X
        X = self.fit(X, y=y, **params).transform(X)
        return X, y


class TimeColTransformer(BaseEstimator, TransformerMixin):
    """Transform data by adding a time column in days named 't'.

    Needs to be inserted in zeroth column position to work with
    signature functions.
    """

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
        """Standard 'fit' method. Included here but blank (no fitting occurs).

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame of observation data.
        y : pd.DataFrame, optional
            Target DataFrame, by default None.

        Returns
        -------
        pd.DataFrame
            Observation data with 't' column.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """If a column named 't' is not in the input data add it and fill
        it with days.

        Parameters
        ----------
        X : pd.DataFrame
            Observation DataFrame.

        Returns
        -------
        pd.DataFrame
            Observation data with 't' column added.
        """
        if isinstance(X, tuple):
            X, y = X
        X = X.copy()
        if "t" not in X.columns:
            X.insert(0, "t", (X.index - X.index[0]).days.values)
        return X

    def fit_transform(self, X: pd.DataFrame, y: pd.DataFrame = None) -> pd.DataFrame:
        """Apply fit and transform methods.

        'fit' here has no effect.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame of observation data.
        y : pd.DataFrame, optional
            Target DataFrame, by default None.

        Returns
        -------
        X : pd.DataFrame
            Observation data with 't' column.
        y : pd.DataFrame
            Unchanged target data.
        """
        if isinstance(X, tuple):
            X, y = X
        X = self.fit(X, y=y).transform(X)
        return X, y


class LaggedYTransformer(BaseEstimator, TransformerMixin):
    """Add the lagged y timeseries to dataframe."""

    def __init__(self):
        self.y_ = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
        """Standard fit method.

        Here we just assign the target dataframe to a class attribute 'y_'.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame of observation data.
        y : pd.DataFrame, optional
            Target DataFrame, by default None.

        Returns
        -------
        LaggedYTransformer
            Class instance.
        """
        self.y_ = y
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add the lagged Y timeseries using the TSData class.

        Parameters
        ----------
        X : pd.DataFrame
            Observation data.

        Returns
        -------
        pd.DataFrame
            Transformed observation data.
        """
        y = self.y_
        if isinstance(X, tuple):
            X, y = X
        if not isinstance(y, pd.DataFrame):
            raise ValueError("Expecting y to be a DataFrame")

        x_min_idx = X.index.min()
        x_max_idx = X.index.max()

        y_mapped = TSData(X, y).y_map_lag()
        X_with_lagged_y = pd.concat((X, y_mapped), axis=1)

        X_with_lagged_y = X_with_lagged_y[
            (X_with_lagged_y.index >= x_min_idx) & (X_with_lagged_y.index <= x_max_idx)
        ].copy()

        X_with_lagged_y["y"] = X_with_lagged_y["y"].ffill()

        return X_with_lagged_y

    def fit_transform(self, X: pd.DataFrame, y: pd.DataFrame = None, **params):
        """Applies the fit and transform methods.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame of observation data.
        y : pd.DataFrame, optional
            Target DataFrame, by default None.

        Returns
        -------
        X : pd.DataFrame
            Fit and transformed observation data.
        y : pd.DataFrame
            Target DataFrame.
        """
        if isinstance(X, tuple):
            X, y = X
        return self.fit(X, y=y).transform(X), y


class SignatureTransformer(BaseEstimator, TransformerMixin):
    """Transformer to calculate signatures from data."""

    def __init__(
        self,
        window_type="days",
        max_length=365,
        fill_method="ffill",
        level=2,
        t_level=2,
        basepoint=True,
        use_multiplier=False,
        keep_sigs="all",
    ):
        """
        Initialise Signature transformer with optional/default parameter
        settings.

        Parameters
        ----------
        window_type : str, optional
            Option controlling window calculating the subframe.
            The default is "days".
        max_length : str, optional
            Maximum length of window controlling subframe. The default is 365.
        fill_method : str, optional
            Method used to fill in missing values. The default is "ffill".
        level : int, optional
            The truncation level of non-time signature_terms. The default is 2.
        t_level : int, optional
            The truncation level of time signatures. The default is 2.
        basepoint : boolean, optional
            Setting to choose wheter to use basepoint to calculate the
            signatures. The default is True.
        use_multiplier : boolean, optional
            Whether to use a multiplier when computing the signatures. The
            multipliers are taken from y. The default is False.
        keep_sigs : str, optional
            Setting to select between keeping all signature or innermost
            signature terms. The default is "all" but "innermost" can also be specified.
        """
        self.window_type = window_type
        self.max_length = max_length
        self.fill_method = fill_method
        self.level = level
        self.t_level = t_level
        self.basepoint = basepoint
        self.use_multiplier = use_multiplier
        self.keep_sigs = keep_sigs
        self.y_ = None

    def _get_signatures(self, X: pd.DataFrame, y: pd.DataFrame = None) -> pd.DataFrame:
        """Call compute_sig_dates to get signature dataframe.
        Parameters
        ----------
        X : pd.DataFrame
            DataFrame of observation data.
        y : pd.DataFrame, optional
            Target DataFrame, by default None.

        Returns
        -------
        pd.DataFrame
            Signature DataFrame.
        """
        params = self.get_params(deep=False)
        sigs = sig_funcs.compute_sigs_dates(X, params, y)
        return sigs

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None, **params):
        """Fit X using signature functions. Y is passed but unused.

        Parameters
        ----------
        X : pd.DataFrame
            Data to be transformed using esig package.
        y : pd.DataFrame, optional
            Target dataframe, passed but unused. The default is None.
        **params : Dictlike
            Any parameters that are existing attributes can be set during
            fitting.

        Returns
        -------
        X : pd.DataFrame
            Transformed DataFrame.
        y : pd.DataFrame
            Target DataFrame unchanged.
        """
        self.set_params(**params)
        self.y_ = y
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the _get_signatures method.

        Parameters
        ----------
        X : pd.DataFrame
            Observation DataFrame.

        Returns
        -------
        pd.DataFrame
            Observation DataFrame with _get_signatures applied.
        """
        y = self.y_
        if isinstance(X, tuple):
            X, y = X
        y_mapped = TSData(X, y).y_mapped
        return self._get_signatures(X, y=y_mapped)

    def fit_transform(self, X: pd.DataFrame, y: pd.DataFrame = None, **params):
        """Fit and Transform `X` using esig package.

        `y` is passed but unused.

        Parameters
        ----------
        X : pd.DataFrame
            Data to be transformed using PCA.
        y : pd.DataFrame, optional
            Target dataframe, passed but unused. The default is None.
        **params : Dictlike
            Any parameters that are existing attributes can be set during
            fitting.

        Returns
        -------
        X : pd.DataFrame
            Transformed DataFrame.
        y : pd.DataFrame
            Target DataFrame unchanged.
        """
        if isinstance(X, tuple):
            X, y = X
        X = self.fit(X, y=y, **params).transform(X)
        return X, y


class SigTransformerPipe(BaseEstimator, TransformerMixin):
    def __init__(self, apply_pca=False, factors=None, pca_params=None, sig_params=None):
        """Create transformer pipeline that can be used in sig estimator.

         Parameters
        ----------
        X : pd.DataFrame
            Data to be transformed using PCA.
        y : pd.DataFrame, optional
            Target dataframe, passed but unused. The default is None.
        pca_params : Dictlike
            Any parameters that are passed to the PCA transformer.
        sig_params : Dictlike
            Any parameters that are passed to the signature transformer.
        Returns
        -------
        X : pd.DataFrame
            Transformed DataFrame.
        y : pd.DataFrame
            Target DataFrame unchanged.
        """
        if pca_params is None:
            pca_params = {}
        if sig_params is None:
            sig_params = {}

        self.transform_pipe_ = None
        self.apply_pca = apply_pca
        self.factors = factors
        self.pca_params = pca_params
        self.sig_params = sig_params
        self.y_ = None

    def _create_pipe(self) -> Pipeline:
        """Build sklearn transformation pipeline.

        Returns
        -------
        sklearn.pipeline.Pipeline
            sklearn transformation pipeline.
        """
        transformers = []
        if self.apply_pca:
            transformers.append(
                ("pca", PCATransformer(factors=self.factors, **self.pca_params))
            )
        transformers.append(("add_time_col", TimeColTransformer()))
        transformers.append(("sig", SignatureTransformer(**self.sig_params)))
        transformers.append(("add_ylag", LaggedYTransformer()))
        return Pipeline(transformers)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None, **params):
        """
        Fit X using optional PCA and signature functions.

        Parameters
        ----------
        X : pd.DataFrame
            Data to be transformed using esig package.
        y : pd.DataFrame, optional
            Target dataframe, passed but unused. The default is None.
        **params : Dictlike
            Any parameters that are existing attributes can be set during
            fitting.

        Returns
        -------
        X : pd.DataFrame
            Transformed DataFrame.
        y : pd.DataFrame
            Target DataFrame unchanged.
        """
        if params:
            self.set_params(params)
        self.transform_pipe_ = self._create_pipe().fit(X, y=y)
        self.y_ = y
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform `X` using optional PCA and signature functions.
        Transform X using optional PCA and signautre functions.
        `y` is passed but unused.

        Parameters
        ----------
        X : pd.DataFrame
            Data to be transformed using PCA.

        Returns
        -------
        X_pca : pd.DataFrame
            Transformed data.
        """
        X = self.transform_pipe_.transform(X)
        return X

    def fit_transform(
        self, X: pd.DataFrame, y: pd.DataFrame = None, **params
    ) -> pd.DataFrame:
        """Fit and Transform X using esig package.

        Y is passed but unused.

        Parameters
        ----------
        X : pd.DataFrame
            Data to be transformed using PCA.
        y : pd.DataFrame, optional
            Target dataframe, passed but unused. The default is None.
        **params : Dictlike
            Any parameters that are existing attributes can be set during
            fitting.

        Returns
        -------
        X : pd.DataFrame
            Transformed DataFrame.
        y : pd.DataFrame
            Target DataFrame unchanged.
        """
        X = self.fit(X, y=y, **params).transform(X)
        return X

    def __call__(self, X, y, **params):
        return self.fit_transform(X, y=y, **params)
