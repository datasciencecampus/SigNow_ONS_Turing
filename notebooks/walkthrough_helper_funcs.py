"""Helper functions for the walk through."""
import matplotlib.pyplot as plt


def display_target(target):
    """Generates a plot of the target.

    Parameters
    ----------
    target : pd.DataFrame
        Target data.

    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(target["value"], label="Target")

    plt.xlabel("Time Period (Years)")
    plt.ylabel("Generated Target Values")
    plt.title("Simulated Target Data")

    plt.legend()
    plt.show()


def display_indicators(indicators):
    """Generates a plot of the indicators.

    Parameters
    ----------
    indicators : pd.DataFrame
        Indicator data.

    """
    fig, ax = plt.subplots(figsize=(10, 5))
    for indic in set(indicators.indicator_name):
        ax.plot(indicators[indicators.indicator_name == indic]["value"], label=indic)

    plt.xlabel("Time Period (Years)")
    plt.ylabel("Generated Data Values")
    plt.title("Simulated Indicator Data")
    plt.ylim(bottom=-8, top=8)
    plt.legend()
    plt.show()


def print_period_dates(sn_):
    """Prints the different periods of the data.

    Parameters
    ----------
    sn_ : SigNowcaster object
        Instance of SigNowcaster.

    """
    print("Start of the training period: ", sn_.data.start_train)
    print("End of the training period: ", sn_.data.end_train)
    print("Start of the test period: ", sn_.data.start_test)
    print("End of the test period: ", sn_.data.end_test)
    print("Start of the reference period: ", sn_.data.start_ref)
    print("End of the reference period: ", sn_.data.end_ref)


def display_period_data(sn_):
    """Displays the indicator data for the different periods.

    Parameters
    ----------
    sn_ : SigNowcaster object
        Instance of SigNowcaster.

    """
    _train_X = sn_.data.X_train()
    _test_X = sn_.data.X_test()
    _ref_X = sn_.data.X_ref()

    fig, ax = plt.subplots(figsize=(10, 5))
    for col in _train_X.columns[0:-1]:
        ax.plot(_train_X[col], color="lightgrey")
    ax.plot(_train_X[_train_X.columns[-1]], label="Training", color="lightgrey")

    for col in _test_X.columns[0:-1]:
        ax.plot(_test_X[col], color="grey")
    ax.plot(_test_X[_test_X.columns[-1]], label="Test", color="grey")

    for col in _ref_X.columns[0:-1]:
        ax.plot(_ref_X[col], color="black")
    ax.plot(_ref_X[_ref_X.columns[-1]], label="Reference", color="black")

    plt.xlabel("Time Period (Years)")
    plt.ylabel("Generated Target Values")
    plt.title("Simulated Indicator Data by Period")

    plt.legend()
    plt.show()


def display_t_predictions(y_train, y_test, y_ref, target, ref_realisation):
    """Plots the predicted data against the realisations.

    Parameters
    ----------
    y_train : pd.DataFrame
        Training section of predicted target timeseries.
    y_test : pd.DataFrame
        Test section of predicted target timeseries.
    y_ref : pd.DataFrame
        Reference period section of predicted target timeseries.
    target : pd.DataFrame
        Full timeseries of target.
    ref_realisation : pd.DataFrame
        Real points of target to compare with predictions.

    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        target,
        color="grey",
        linestyle="solid",
        marker="o",
        markerfacecolor="grey",
        markersize=4,
        label="Realisations",
    )
    ax.plot(
        ref_realisation,
        color="grey",
        linestyle="solid",
        marker="x",
        markerfacecolor="grey",
        markersize=4,
        label="Reference Period Realisation",
    )
    ax.plot(
        y_train,
        color="springgreen",
        linestyle="solid",
        marker="o",
        markerfacecolor="springgreen",
        markersize=4,
        label="Training Predictions",
    )
    ax.plot(
        y_test,
        color="lightcoral",
        linestyle="solid",
        marker="o",
        markerfacecolor="lightcoral",
        markersize=4,
        label="Test Predictions",
    )
    ax.plot(
        y_ref,
        color="black",
        marker="o",
        markerfacecolor="black",
        markersize=4,
        label="Reference Period Prediction",
    )

    plt.xlabel("Time Period (Years)")
    plt.ylabel("Target Value")
    plt.title("Predicted Target Data against the Realisations")
    plt.legend()
    plt.ylim(bottom=-6, top=6)
    plt.show()


def display_barh_coef(coef, sig_terms_train):
    """plot horizontal bar graph.

    coef : pd.Series
        Coefficients of elastic net model.
    sig_terms_train : pd.DataFrame
        Transformed X (signature terms) for the training period.

    """
    sig_terms_train = sig_terms_train.rename(columns={"y": "y_lagged"})
    # Plot these in a bar graph
    plt.barh(y=sig_terms_train.columns, width=coef)
    plt.ylabel("Signature Terms")
    plt.xlabel("Coefficients")
    plt.title("Coefficients for the Signature Terms")
