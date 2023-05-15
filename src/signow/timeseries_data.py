"""Module to handle TimeSeries data."""
import pandas as pd
from pandas.tseries.offsets import DateOffset

from signow.utils.validation import validate_X_y


class TSData:
    """Mediates access to timeseries data.

    Takes date indexed data `X` and `y`, aligns data to a common index,
    and provides access to train, test, and reference period data.

    All data is mapped to the highest frequency index detected in `X`, `y`.

    """

    def __init__(self, X=None, y=None, **params):
        """Initialise time series data class.

        Parameters
        ----------
        X : pd.DataFrame , optional
            X. The default is None.
        y : pd.DataFrame, optional
            Target y data. The default is None.

        """
        self.y = None
        self.X = None
        self.y_mapped = None
        self.X_mapped = None
        self.y_freq = None
        self.X_freq = None
        self.start_train = None
        self.end_train = None
        self.start_test = None
        self.end_test = None
        self.start_ref = None
        self.end_ref = None
        self.hf_Xkey = None
        self.hfreq = None

        self._validate(X=X, y=y, **params)

    def _validate(self, X=None, y=None, **params):
        """Validate and set all attributes in TSData."""
        if not isinstance(y, (pd.DataFrame, pd.Series)):
            return

        X, y, X_freq, y_freq = validate_X_y(
            X, y, params.get("X_freq"), params.get("y_freq")
        )
        self.X = X
        self.y = y
        self.X_freq = X_freq
        self.y_freq = y_freq
        self._set_dates(**params)
        self._set_hfreq(X, X_freq, y_freq)

        X_mapped, y_mapped = self._get_mapped(X, y)

        self.X_mapped = X_mapped
        self.y_mapped = y_mapped

    def get_params(self):
        """Return the names of parameters."""
        return self.__dict__

    def _date_start(self, date, forward=False):
        """Get the start date of train, test, reference period.
        Depends on frequency of y."""
        if not date:
            return
        period = pd.to_datetime(date).to_period(self.y_freq[0])
        if forward:
            period = period + 1
        return period.start_time

    def _date_end(self, date, backward=False):
        """Get the end date of either train, test, reference period.
        Depends on frequency of y."""
        if not date:
            return
        period = pd.to_datetime(date).to_period(self.y_freq[0])
        if backward:
            period = period - 1
        return period.end_time

    def _set_dates(self, **params):
        """Set all date attributes based on user input.

        Sets defaults based on data if not explicitly set by user.

        Notes
        -----
        `end_train` and `end_ref` are derived from the other `params`.

        `end_test` can be explicitly set, or derived.

        """
        self.start_ref = self._date_start(params.get("start_ref"))
        if not self.start_ref:
            self.start_ref = self._date_start(self.y.index.max(), forward=True)

        self.start_train = self._date_start(params.get("start_train"))
        if not self.start_train:
            self.start_train = self._date_start(self.y.index.min())

        self.start_test = self._date_start(params.get("start_test"))
        if not self.start_test:
            self.start_test = self.start_ref

        self.end_ref = self._date_end(self.start_ref)

        self.end_test = self._date_end(params.get("end_test"))
        if not self.end_test:
            self.end_test = self.end_ref

        if self.end_test < self.start_test:
            self.end_test = self.start_test

        self.end_train = self._date_end(self.start_test, backward=True)

    def _start_end_period(self, freq):
        if len(freq) == 1:
            return ""
        elif freq[1] != "S":
            return ""
        return "S"

    def _mod_start_end_period(self, y_freq, hfreq):
        if self._start_end_period(y_freq) == self._start_end_period(hfreq):
            return hfreq
        elif self._start_end_period(y_freq) == "S":
            end = "" if len(hfreq) == 1 else hfreq[1:]
            return f"{hfreq[0]}S{end}"
        end = "" if len(hfreq) == 2 else hfreq[2:]
        return f"{hfreq[0]}{end}"

    def _set_hfreq(self, X, X_freq, y_freq):
        """Get the highest freq from X."""
        hfkey = X.count().idxmax()
        hfreq = X_freq[hfkey]

        if hfreq[0] == y_freq[0]:
            hfreq = y_freq

        elif hfreq[0] in ["Q", "M"]:
            hfreq = self._mod_start_end_period(self.y_freq, hfreq)

        elif hfreq[0] == "W":
            hfreq = "D"

        if hfreq[0] == "W":
            hfreq = "D"

        self.hf_Xkey = hfkey
        self.hfreq = hfreq

    def _map_to_hf(self, index, col, freq):
        """Map column to highest frequency."""
        col = col.dropna()
        col.index = col.index.to_period(freq[0]).start_time
        col = col.reindex(index)
        col = col.groupby(col.index.to_period(freq[0])).transform(max)
        return col.backfill()

    def _get_index(self, index, y_index, freq, y_freq):
        """Get high frequency index used for mapping."""
        start = y_index.to_period(y_freq[0]).start_time.min()
        end = index.max() if index.max() > self.end_ref else self.end_ref
        start_freq = freq[0] if freq[0] not in ["Q", "M"] else f"{freq[0]}S"
        return pd.date_range(start, end, freq=start_freq)

    def _apply_offset(self, df):
        """Apply offset to index that has been aligned to the start of the period."""
        if self.hfreq[0] in ["W", "D"]:
            return df

        if self.y_freq[0] == self.hfreq[0]:
            end = df.index.to_period(self.y_freq[0]).end_time.max()
            df.index = pd.date_range(df.index.min(), end, freq=self.y_freq)

            return df
        return df

    def _get_mapped(self, X, y):
        """Map X and y to the highest frequency."""
        hfreq, y_freq = self.hfreq, self.y_freq
        x_min_idx = X.index.to_period(hfreq[0]).start_time.min()
        X_index = self._get_index(X.index, y.index, hfreq, self.y_freq)
        X_mapped = pd.concat(
            [self._map_to_hf(X_index, X[k], i) for k, i in self.X_freq.items()],
            axis=1,
        )
        X_mapped = self._apply_offset(X_mapped)
        yX_index = X_index[X_index < self.end_ref]
        y_mapped = self._map_to_hf(yX_index, y["y"], y_freq)
        y_mapped = self._apply_offset(y_mapped)

        X_mapped = X_mapped[X_mapped.index >= x_min_idx].copy()

        y_mapped = y_mapped.to_frame()
        y_mapped = y_mapped[y_mapped.index >= x_min_idx].copy()

        return X_mapped, y_mapped

    def _y_with_next_period(self, y, y_freq):
        """Returns y with the last value carried forward 1 period.

        Supports quarterly, monthly, weekly or daily frequency.

        Parameters
        ----------
        y : pd.DataFrame
            Single column data frame, target variable.
            Must be date indexed.
        y_freq: str
            Frequency of the data in `y`, must start with one of:
            `{"Q", "M", "W", "D"}`.

        Returns
        -------
        pd.DataFrame
            Single column data frame, with an extra period.

        """
        new = y[-1:].copy()
        freq = {
            "Q": {"months": 3},
            "M": {"months": 1},
            "W": {"days": 7},
            "D": {"days": 1},
        }
        new_date = y.index.max() + DateOffset(**freq[y_freq[0]])
        new.rename(index={new.index[0]: new_date}, inplace=True)

        y_extended = y.append(new)

        return y_extended

    def y_map_lag(self):
        """Lagged `y`, mapped to `X` index.

        `y` is mapped to the highest frequency index from `X` and `y`.

        Values are shifted by one place.

        The index covers the X index, excluding the end of the reference period.

        Periods are offset to match the periods of the `y` index if necessary.

        Returns
        -------
        pd.DataFrame:
            Single column dataframe, date indexed.

        """
        y = self._y_with_next_period(self.y, self.y_freq[0])
        lagged_y = y.shift(1).fillna(method="bfill")

        X_index = self._get_index(self.X.index, y.index, self.hfreq, self.y_freq)
        yX_index = X_index[X_index < self.end_ref]
        lagged_y = self._map_to_hf(yX_index, lagged_y["y"], self.y_freq)
        lagged_y = self._apply_offset(lagged_y)

        return lagged_y[lagged_y.index < self.end_ref]

    def update(self, **params):
        """Update any of the user defined inputs.

        Notes
        -----
            Pass all `params` that are not set to default, this method re-runs
            variable setting based on what is in `params` rather than what is already set.

        """
        if any(params.get(k) for k in ("X", "y")):
            self._validate(**params)
        else:
            self._set_dates(**params)

    def _get_sub_data(self, name, sub, sub_end=None):
        """Filter data into portion given start and end dates specified by
        keywords name and sub."""
        data = self.__dict__.get(name)
        sub_end = sub_end if sub_end else sub
        start = data.index >= self.__dict__.get(f"start_{sub}")
        end = data.index <= self.__dict__.get(f"end_{sub_end}")
        return data[start & end]

    def y_train(self):
        """Return part of target data used for training."""
        return self._get_sub_data("y", "train")

    def y_test(self):
        """Return part of target data used for testing."""
        return self._get_sub_data("y", "test")

    def y_mapped_train(self):
        """Return part of target data used for training."""
        return self._get_sub_data("y_mapped", "train")

    def y_mapped_test(self):
        """Return part of target data used for testing."""
        return self._get_sub_data("y_mapped", "test")

    def X_train(self):
        """Return part of date used for training."""
        return self._get_sub_data("X_mapped", "train")

    def X_test(self):
        """Return part of data used for testing."""
        return self._get_sub_data("X_mapped", "test")

    def X_ref(self):
        """Return data within current reference period."""
        return self._get_sub_data("X_mapped", "ref")

    def split(self):
        """Return X and y data split into training and testing sets."""
        return (
            self.X_train(),
            self.X_test(),
            self.y_mapped_train(),
            self.y_mapped_test(),
        )
