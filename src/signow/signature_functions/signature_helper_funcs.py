"""Dimensionality reduction functionality applied prior to signatures.

This module wraps statsmodels PCA in a way that allows fitting and applying to
unseen data. Statsmodels PCA is used for consistency with related projects/outputs.

Notes
-----
    This module is a modified version of code created by Lingyi Yang from The Alan Turing Institute.
    This code can be found on the public GitHub repository that reproduces the outputs
    from the academic paper: Nowcasting with Signature Methods (Cohen et al., 2023).
    The source repository can be accessed here: https://github.com/lingyiyang/nowcasting_with_signatures.

"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.multivariate.pca import PCA


@dataclass
class FittedParams:
    mu: float
    sigma: float
    eigenvectors: pd.DataFrame


def standardize_df(df, mu, sigma):
    """Standardise data.

    Standardise data to match statsmodels application of PCA.
    Statsmodels PCA, by default, demeans and standardises variance.

    Apply before multiplying eigenvectors.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to standardize.
    mu : np.array
        Values to subtract from each column.
    sigma : np.array
        Values to divide each column by.

    Returns
    -------
    pd.DataFrame
        A dataframe that has been standardized.

    """
    data = (df - mu) / sigma
    return data


def reduce_dim(
    df, k, factor_structure=None, fill_method="backfill", pca_mu_sigma_eigenvals=None
):
    """Reduce dimension of the input variables by using PCA (optionally specifying a structure).

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to reduce dimension of.
    k
        An integer or a list of integers with the number of principal
        components to find.
    factor_structure
        An optional input to the function which specifies
        structure to compute principal components from.
    fill_method
        Method by which to fill in the missing data
        (only "backfill"/"fill-em" are implemented).

    Returns
    -------
        pd.DataFrame
            A dataframe of the principal components.

    """

    fill = None
    if fill_method == "backfill":
        # forward fill data first before filling in missing data at start
        df = df.ffill()
        df = df.bfill()
    else:
        fill = "fill-em"

    fitted_params = None

    if not factor_structure:
        if pca_mu_sigma_eigenvals is not None:
            mu = pca_mu_sigma_eigenvals.mu
            sigma = pca_mu_sigma_eigenvals.sigma
            eigenvecs = pca_mu_sigma_eigenvals.eigenvectors

            df = standardize_df(df, mu, sigma)
            pca_arr = np.dot(df, eigenvecs)
            cols = [f"global_{i}" for i in range(k)]
            df_pca = pd.DataFrame(data=pca_arr, columns=cols, index=df.index)

        else:
            res_pca = PCA(df, ncomp=k, method="eig", normalize=False, missing=fill)

            df_pca = res_pca.factors
            df_pca.columns = [f"global_{i}" for i in range(k)]

            fitted_params = FittedParams(
                mu=res_pca._mu, sigma=res_pca._sigma, eigenvectors=res_pca.eigenvecs
            )

    else:
        all_pca = []
        # find k principal components of each subgroup (k can be scalar or vector)
        param_dict = {}

        if not isinstance(k, (list, tuple, np.ndarray)):
            k = np.repeat(k, len(factor_structure))
        for a_k, var_subset in zip(k, factor_structure):
            if pca_mu_sigma_eigenvals is not None:
                df_factor = df[factor_structure[var_subset]]

                mu_sigma_eig = pca_mu_sigma_eigenvals[var_subset]

                mu = mu_sigma_eig.mu
                sigma = mu_sigma_eig.sigma
                eigenvectors = mu_sigma_eig.eigenvectors

                df_factor = standardize_df(df_factor, mu, sigma)

                pca_arr = np.dot(df_factor, eigenvectors)
                col = [f"{var_subset}_{i}" for i in range(a_k)]
                df_pca = pd.DataFrame(data=pca_arr, columns=col, index=df.index)

                all_pca.append(df_pca)

            else:
                res_pca = PCA(
                    df[factor_structure[var_subset]],
                    ncomp=a_k,
                    method="eig",
                    normalize=False,
                    missing=fill,
                )
                df_pca = res_pca.factors
                df_pca.columns = [f"{var_subset}_{i}" for i in range(a_k)]
                all_pca.append(df_pca)

                fitted_params = FittedParams(
                    mu=res_pca._mu, sigma=res_pca._sigma, eigenvectors=res_pca.eigenvecs
                )
                param_dict[var_subset] = fitted_params

        df_pca = pd.concat(all_pca, axis=1)
        fitted_params = param_dict

    if pca_mu_sigma_eigenvals is not None:
        return_params = pca_mu_sigma_eigenvals
    else:
        return_params = fitted_params

    return df_pca, return_params


def apply_pca(df, model_params, df_grouping, pca_mu_sigma_eig=None):
    """Main function to apply Principal Component Analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe of data available at current time/horizon
    model_params : dict
        Dictionary containing all necessary parameters for fitting the model
    df_grouping : pd.DataFrame
        Dataframe containing factor groupings.
    pca_mu_sigma_eig : FittedParams, optional
        Previously fitted PCA params to apply.
        Setting this will skip statsmodels PCA and apply the transformation
        directly.

    Returns
    -------
    pd.DataFrame
        Dataframe after dimension reduction.
    FittedParams
        Fitted pca params i.e. mu, sigma, eigenvectors.

    """
    if isinstance(df_grouping, pd.DataFrame):
        factor_structure = {}
        for group in df_grouping["factor_group"].unique():
            new_group = {
                str(group): list(
                    df_grouping[df_grouping["factor_group"] == group].factor_name
                )
            }
            factor_structure.update(new_group)
    else:
        factor_structure = None

    df, fitted_params = reduce_dim(
        df,
        model_params["k"],
        factor_structure=factor_structure,
        fill_method=model_params["pca_fill_method"],
        pca_mu_sigma_eigenvals=pca_mu_sigma_eig,
    )
    return df, fitted_params
