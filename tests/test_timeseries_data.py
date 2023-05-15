"""Tests to cover timeseries data."""
import numpy as np
import pandas as pd

from signow.timeseries_data import TSData

# Standard timeseries setup
start = pd.to_datetime("2012-06-01")
end = pd.to_datetime("2013-04-01")
ref = pd.to_datetime("2013-06-01")


def gen_default_y(freq="QS-DEC"):
    """Generate standard y input."""
    y_index = pd.date_range(start, end, freq=freq)
    y = pd.DataFrame(index=y_index, data=list(range(len(y_index))), columns=["y"])
    return y_index, y.astype(float)


def gen_default_X(nstart=0, nend=7):
    """Generate standard X input. Filter by start and end to get different
    input data combinations."""
    X_index = {
        "QS": pd.date_range(start, ref, freq="QS"),
        "Q": pd.date_range(start, ref, freq="Q"),
        "M": pd.date_range(start, ref, freq="M"),
        "W": pd.date_range(start, ref, freq="W-MON"),
        "W2": pd.date_range(start, ref, freq="W-SUN"),
        "D": pd.date_range(start, ref, freq="D"),
    }
    X_index = {k: X_index[k] for k in list(X_index.keys())[nstart:nend]}
    X = pd.concat(
        [
            pd.DataFrame(index=i, data=np.random.random(len(i)), columns=[f"X{k}"])
            for k, i in X_index.items()
        ],
        axis=1,
    )
    return X_index, X


class TestTSData:
    """Tests for the TSData class."""

    def test_init_data(self):
        "Test TSData attributes are set correctly with correct data."

        _, y = gen_default_y()
        _, X = gen_default_X()

        data = TSData(X, y)

        pd.testing.assert_frame_equal(data.X, X)
        pd.testing.assert_frame_equal(data.y, y)
        assert data.y_freq == "QS-DEC"
        assert data.X_freq == {
            "XQS": "QS-OCT",
            "XQ": "Q-DEC",
            "XM": "M",
            "XW": "W-MON",
            "XW2": "W-SUN",
            "XD": "D",
        }

    def test_y_mapped(self):
        """Test y_mapped is produced correctly given default input y, X."""
        freq = {1: "Q", 2: "Q", 3: "M", 4: "D", 5: "D", 6: "D"}
        hfreq = {1: "QS-DEC", 2: "QS-DEC", 3: "MS", 4: "D", 5: "D", 6: "D"}

        for key in range(1, 7):
            y_index, y = gen_default_y()
            X_index, X = gen_default_X(nstart=0, nend=key)

            data = TSData(X, y)

            assert data.hfreq == hfreq[key]

            # test that not all values are empty
            assert not data.y_mapped.isna().all()[0]

            # test index matches lowest frequency produced in X
            assert pd.infer_freq(data.y_mapped.index)[0] == freq[key][0]

            # test there is only 1 value assigned to all dates within original period
            assert data.y_mapped.dropna().resample(freq[key]).nunique().eq(1).all()[0]

    def test_X_mapped(self):
        """Test X_mapped is produced correctly given default input y, X.

        If the lowest freqency is weekly then it is mapped to daily."""

        exp_index = {
            1: {"freq": "QS-DEC", "start": "2012-09-01", "end": "2013-06-01"},
            2: {"freq": "QS-DEC", "start": "2012-06-01", "end": "2013-06-01"},
            3: {"freq": "MS", "start": "2012-06-01", "end": "2013-06-01"},
            4: {"freq": "D", "start": "2012-06-04", "end": "2013-06-30"},
            5: {"freq": "D", "start": "2012-06-03", "end": "2013-06-30"},
            6: {"freq": "D", "start": "2012-06-01", "end": "2013-06-30"},
        }

        for key in range(1, 7):
            _, y = gen_default_y()
            X_index, X = gen_default_X(nstart=0, nend=key)

            data = TSData(X, y)

            start_index = pd.to_datetime(exp_index[key]["start"])
            end_index = pd.to_datetime(exp_index[key]["end"])

            # test X_mapped has same index as lowest frequency
            assert pd.infer_freq(data.X_mapped.index) == exp_index[key]["freq"]
            assert data.X_mapped.index.min() == start_index
            assert data.X_mapped.index.max() == end_index

            # test there is only 1 value assigned to all dates within
            # original period
            for k, i in X_index.items():
                freq = k[0] if k[0] != "W" else "D"
                Xcol = data.X_mapped[f"X{k}"].dropna().resample(freq)
                assert Xcol.nunique().eq(1).all()

    def test_default_dates(self):
        """Tests that the default dates are correct given X, y.
        """
        _, y = gen_default_y()
        X_index, X = gen_default_X(3)

        data = TSData(X, y)

        assert data.start_train == pd.to_datetime("2012-04-01 00:00:00")
        assert data.end_train == pd.to_datetime("2013-03-31 23:59:59.999999999")
        assert data.start_test == pd.to_datetime("2013-04-01 00:00:00")
        assert data.end_test == pd.to_datetime("2013-06-30 23:59:59.999999999")
        assert data.start_ref == pd.to_datetime("2013-04-01 00:00:00")
        assert data.end_ref == pd.to_datetime("2013-06-30 23:59:59.999999999")

    def test_default_dates_monthly(self):
        """Tests that the default dates are correct given X and a monthly y series.
        """
        _, y = gen_default_y(freq="M")
        X_index, X = gen_default_X(3)

        data = TSData(X, y)

        assert data.start_train == pd.to_datetime("2012-06-01 00:00:00")
        assert data.end_train == pd.to_datetime("2013-03-31 23:59:59.999999999")
        assert data.start_test == pd.to_datetime("2013-04-01 00:00:00")
        assert data.end_test == pd.to_datetime("2013-04-30 23:59:59.999999999")
        assert data.start_ref == pd.to_datetime("2013-04-01 00:00:00")
        assert data.end_ref == pd.to_datetime("2013-04-30 23:59:59.999999999")

    def test_dates_manual(self):
        """Tests user generated input dates.
        """
        _, y = gen_default_y()
        X_index, X = gen_default_X()

        data = TSData(X, y, start_train="2012-08-13", start_test="2013-01-02")

        assert data.start_train == pd.to_datetime("2012-07-01 00:00:00")
        assert data.end_train == pd.to_datetime("2012-12-31 23:59:59.999999999")
        assert data.start_test == pd.to_datetime("2013-01-01 00:00:00")
        assert data.end_test == pd.to_datetime("2013-06-30 23:59:59.999999999")
        assert data.start_ref == pd.to_datetime("2013-04-01 00:00:00")
        assert data.end_ref == pd.to_datetime("2013-06-30 23:59:59.999999999")

    def test_y_map_lag(self):
        """Tests y_map_lag function against different combinations of indicators.
        """
        freq = {1: "Q", 2: "Q", 3: "M", 4: "D", 5: "D", 6: "D"}
        freq_filter = {
            1: [1, -1],
            2: [1, -1],
            3: [3, -3],
            4: [91, -91],
            5: [91, -91],
            6: [91, -91],
        }

        for key in range(1, 7):
            _, y = gen_default_y()
            X_index, X = gen_default_X(nstart=0, nend=key)

            data = TSData(X, y)
            y_map_lag = data.y_map_lag()

            # test that all values are not NaN
            assert not data.y_mapped.isna().all()[0]

            # test index matches lowest frequency produced in X
            assert pd.infer_freq(data.y_mapped.index)[0] == freq[key][0]

            # test there is only 1 value assigned to all dates within original period
            assert data.y_mapped.dropna().resample(freq[key]).nunique().eq(1).all()[0]

            # test correctly shifted using range to test that difference
            # is only one
            fil = freq_filter[key]
            assert (data.y_mapped["y"] - y_map_lag)[fil[0] : fil[1]].eq(1).all()

            # test that final value of y_map_lag is the same as the final
            # value of y
            assert (y_map_lag.dropna().iloc[-1] == data.y_mapped.dropna().iloc[-1])[0]

    def test_y_map_lag_y(self):
        """Tests y_map_lag with different y frequencies.
        """
        freq = {"M": [-30, 30], "W": [-7, 7], "W-MON": [-7, 7], "D": [-1, 1]}
        for k, i in freq.items():
            _, y = gen_default_y(freq=k)
            X_index, X = gen_default_X(nstart=1, nend=7)

            data = TSData(X, y)
            y_map_lag = data.y_map_lag()

            # test that all values are not NaN
            assert not data.y_mapped.isna().all()[0]

            # test index matches lowest frequency produced in X
            assert pd.infer_freq(data.y_mapped.index)[0] == "D"

            # test correctly shifted using range to test that difference
            # is only one
            fil = freq[k]
            assert (data.y_mapped - y_map_lag)[fil[0] : fil[1]].eq(1).all()[0]

            # test that final value of y_map_lag is the same as the final
            # value of y
            assert (y_map_lag.dropna().iloc[-1] == data.y_mapped.dropna().iloc[-1])[0]

    def test_split(self):
        """Tests data splits into correct train and test.
        """
        for key in range(1, 7):
            _, y = gen_default_y()
            X_index, X = gen_default_X(nstart=0, nend=key)
            data = TSData(X, y)

            X_index = data.X_mapped.index
            ym_index = data.y_mapped.index
            y_index = data.y.index

            X_train_index = (X_index >= data.start_train) & (X_index <= data.end_train)
            X_test_index = (X_index >= data.start_test) & (X_index <= data.end_test)
            y_train_index = (y_index >= data.start_train) & (y_index <= data.end_train)
            y_test_index = (y_index >= data.start_test) & (y_index <= data.end_test)
            ym_train_index = (ym_index >= data.start_train) & (
                ym_index <= data.end_train
            )
            ym_test_index = (ym_index >= data.start_test) & (ym_index <= data.end_test)

            pd.testing.assert_frame_equal(data.X_train(), data.X_mapped[X_train_index])
            pd.testing.assert_frame_equal(data.X_test(), data.X_mapped[X_test_index])
            pd.testing.assert_frame_equal(data.y_train(), data.y[y_train_index])
            pd.testing.assert_frame_equal(data.y_test(), data.y[y_test_index])
            pd.testing.assert_frame_equal(
                data.y_mapped_train(), data.y_mapped[ym_train_index]
            )
            pd.testing.assert_frame_equal(
                data.y_mapped_test(), data.y_mapped[ym_test_index]
            )

            xtrain, xtest, ytrain, ytest = data.split()
            pd.testing.assert_frame_equal(xtrain, data.X_mapped[X_train_index])
            pd.testing.assert_frame_equal(xtest, data.X_mapped[X_test_index])
            pd.testing.assert_frame_equal(ytrain, data.y_mapped[ym_train_index])
            pd.testing.assert_frame_equal(ytest, data.y_mapped[ym_test_index])
