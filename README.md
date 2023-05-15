# SigNow

A tool for  generating Signature nowcasts with economic data.

<img src ="signow_logo.png" alt="SigNow" width=400>
________________________________________________________________

SigNow is a tool that generates nowcasts using the Signature Method. Nowcasting in economics is the prediction of the very recent past, or the present, of an economic indicator. The Signature Method is a collection of feature extraction techniques for multivariate timeseries [Morrill et al., 2021](https://arxiv.org/pdf/2006.00873.pdf). The Signature Method has many useful properties for nowcasting problems; for instance, it can be used flexibly with the irregular sampling patterns often inherent in economic data and it can capture the correlation of a single/multiple data series when determining the Signature path.

To understand how SigNow is used within a nowcasting setting, you can find more detailed documentation in the signow_walkthrough notebook in the notebooks folder of this repository.

## Requirements
_________________________________________________________________

You can find a list of the direct dependencies, with versions, in the pyproject.toml file.

During development the project ran on `Python 3.7` with the following versions for the main dependencies:

| Library | Version |
| ------- | ------- |
| `esig`            | 0.9.8.3 |
| `matplotlib`      | 3.5.3 |
| `numpy`           | 1.21.6 |
| `pandas`          | 1.3.5 |
| `scikit-learn`    | 1.0.2 |
| `statsmodels`     | 0.13.5 |

## Installation
_________________________________________________________________

You can install the development version of _SigNow_ via [pip](https://pip.pypa.io/) using the following command:

```bash
$ pip install git+https://github.com/datasciencecampus/SigNow_ONS_Turing
```

A guide for the installation of `esig` can be found in the `esig` [documentation](https://esig.readthedocs.io/en/latest/installing.html).

## Usage
_________________________________________________________________

This package is designed to be used within an interactive console
session or Jupyter notebook

```python
from signow.signature_nowcasting import SigNowcaster

#setting the parameters for the Signature
sig_params={
    "window_type": "days",
    "max_length": 365,
    "fill_method": "ffill",
    "level": 2,
    "t_level": 2,
    "basepoint": False,
    "use_multiplier": False,
    "keep_sigs": "all"}
#setting parameters for the regression
regress_params = {
    "alpha": 0.1,
    "l1_ratio": 0.5,
    "fit_intercept": True}
pca_params = {
    "pca_fill_method":"backfill",
    "k":2}

params = {"regress_params": {**regress_params},
         "sig_params": {**sig_params},
         "pca_params": {**pca_params}}

# Initiate Nowcast and Validate Data
nowc = SigNowcaster(X=my_indicators, y=my_target,
                    start_train='1990-01-01',
                    start_test='2010-01-01',
                    start_ref='2023-04-01',
                    regressor="elasticnet",
                    apply_pca=False,
                    **params)

# Fit Model
nowc.fit()

# Run Static Nowcast over test period
static_nowc = nowc.static_nowcast()

# Run Recursive Nowcast over test period
recursive_nowc = nowc.recursive_nowcast()

```

## Limitations

The purpose of SigNow is to produce Signature nowcasts in more traditional economic nowcasting settings, such as those where we have economic data in either the monthly and quarterly frequency.

SigNow should be able to handle most cases of mixed frequency data, however we recommend that you check the alignment of the data to ensure that it suits your purpose. Users should consult signow/timeseries_data.py for insight into how this alignment is currently achieved.

Signatures can handle much higher frequencies and for those users intending to use different types of data of a higher frequency for their own nowcast problem, they are recommended to build this using the [esig](https://esig.readthedocs.io/en/latest/) package more directly. In addition, users who are implementing this in a production setting are advised to call the esig package directly.

## License
_________________________________________________________________

Distributed under the terms of the [MIT license](https://opensource.org/licenses/MIT), SigNow is free and open source software.

## Issues
_________________________________________________________________

This package will not be receiving ongoing support and therefore should you encounter any issues/bugs, which we have endevoured to minimise, please feel free to contact us or make these changes yourself.

## Unit Testing

All unit tests can be found in the tests folder of this repository. The system tests were performed primarily on a pipeline that designed to nowcast Quarterly Household income growth in the UK. These system tests included over 40 tests which SigNow passed. Three of these can be found in the tests/test_system.py file.

Individual function level unit tests can be found in the tests folder and are named appropriately to help the user identify the part of SigNow being tested.

## Credits
_________________________________________________________________

This project was generated from a collaboration with [The Alan Turing Institute](https://www.turing.ac.uk/) and [The Data Science Campus](https://datasciencecampus.ons.gov.uk/) at [The Office for National Statistics](https://www.ons.gov.uk/).

Developers of this project include Craig Scott (ONS), Emma Small (ONS) and Lingyi Yang (The Alan Turing Institute). Lingyi Yang created the code in the signow/signature_functions folder which is used in the generation of the signature terms. The implementation of this repository is a modified version of the code used in the academic paper: Nowcasting with Signatures Methods (Cohen et al., 2023). The code that generates the outputs for this paper can be found [here](https://github.com/lingyiyang/nowcasting_with_signatures).

Special thanks to Philip Lee (ONS) and Dylan Purches (ONS) for their valuable feedback and code review throughout this project.

Some unit tests are run with [hypothesis](https://hypothesis.readthedocs.io/en/latest/).

SigNow is essentially a wrapper for [esig](https://esig.readthedocs.io/en/latest/), the main package that generates and handles the Signature of a path.
