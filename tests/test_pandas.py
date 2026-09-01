"""pandas objects in, labelled pandas objects out -- and nothing in the core.

Two claims are tested here. The first is that labels survive: a density
evaluated at a labelled index comes back with that index, because silently
dropping it is how a misaligned join happens later. The second is that the
numbers are the same ones the array path produces, bit for bit -- the pandas
layer converts, it does not compute.

`tests/test_no_pandas.py` covers the other half: that an install without pandas
never imports it and never pays for it.
"""

from __future__ import annotations

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from levy import api  # noqa: E402

XS = np.linspace(-8.0, 8.0, 33)
PARAMS = {"alpha": 1.5, "beta": -0.3, "mu": 0.2, "sigma": 1.1}


@pytest.fixture
def series():
    return pd.Series(XS, index=pd.RangeIndex(100, 133, name="obs"), name="returns")


@pytest.fixture
def frame():
    return pd.DataFrame({"a": XS, "b": XS[::-1]},
                        index=pd.RangeIndex(100, 133, name="obs"))


# --------------------------------------------------------------------------
# labels survive
# --------------------------------------------------------------------------

@pytest.mark.parametrize("function", ["pdf", "cdf", "logpdf"])
def test_series_in_series_out(function, series):
    result = getattr(api, function)(series, **PARAMS)
    assert isinstance(result, pd.Series)
    pd.testing.assert_index_equal(result.index, series.index)
    assert result.name == series.name


@pytest.mark.parametrize("function", ["pdf", "cdf", "logpdf"])
def test_frame_in_frame_out(function, frame):
    result = getattr(api, function)(frame, **PARAMS)
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_index_equal(result.index, frame.index)
    pd.testing.assert_index_equal(result.columns, frame.columns)


def test_a_datetime_index_survives():
    index = pd.date_range("2020-01-01", periods=len(XS), freq="D")
    result = api.pdf(pd.Series(XS, index=index), **PARAMS)
    pd.testing.assert_index_equal(result.index, index)


def test_an_unsorted_index_is_not_reordered():
    index = pd.Index([3, 1, 2, 0][: 4], name="k")
    values = pd.Series([1.0, 2.0, 3.0, 4.0], index=index)
    result = api.pdf(values, **PARAMS)
    pd.testing.assert_index_equal(result.index, index)
    assert result.iloc[0] == api.pdf(np.array([1.0]), **PARAMS)[0]


def test_an_unnamed_series_stays_unnamed(series):
    result = api.pdf(series.rename(None), **PARAMS)
    assert result.name is None


# --------------------------------------------------------------------------
# the numbers do not move
# --------------------------------------------------------------------------

@pytest.mark.parametrize("function", ["pdf", "cdf", "logpdf"])
def test_values_are_bit_identical_to_the_array_path(function, series):
    from_pandas = getattr(api, function)(series, **PARAMS)
    from_array = getattr(api, function)(XS, **PARAMS)
    assert np.array_equal(from_pandas.to_numpy(), from_array)


def test_frame_values_are_bit_identical_column_by_column(frame):
    result = api.pdf(frame, **PARAMS)
    for column in frame.columns:
        expected = api.pdf(frame[column].to_numpy(), **PARAMS)
        assert np.array_equal(result[column].to_numpy(), expected)


def test_fit_on_a_series_matches_fit_on_its_values():
    sample = api.rvs(alpha=1.5, beta=0.0, size=500, random_state=3)
    labelled = pd.Series(sample, index=pd.date_range("2021-01-01", periods=500))

    from_series = api.fit(labelled)
    from_array = api.fit(sample)

    assert from_series.params == from_array.params
    assert from_series.negative_log_likelihood == from_array.negative_log_likelihood


def test_fit_on_a_one_column_frame_matches_the_series():
    sample = api.rvs(alpha=1.5, beta=0.0, size=300, random_state=4)
    frame = pd.DataFrame({"x": sample})
    assert api.fit(frame).params == api.fit(sample).params


# --------------------------------------------------------------------------
# results as pandas
# --------------------------------------------------------------------------

def test_to_series_is_named_by_parameter():
    sample = api.rvs(alpha=1.5, beta=0.0, size=300, random_state=5)
    reported = api.fit(sample).to_series()

    assert isinstance(reported, pd.Series)
    assert list(reported.index) == ["alpha", "beta", "mu", "sigma"]
    np.testing.assert_allclose(
        reported.to_numpy(), api.fit(sample).params.as_tuple(), rtol=0, atol=0)


@pytest.mark.parametrize(("par", "names"), [
    ("0", ["alpha", "beta", "mu", "sigma"]),
    ("1", ["alpha", "beta", "mu", "sigma"]),
    ("M", ["alpha", "beta", "gamma", "lambda"]),
    ("A", ["alpha", "beta", "gamma", "lambda"]),
    ("B", ["alpha", "beta", "gamma", "lambda"]),
])
def test_to_series_uses_the_parametrizations_own_names(par, names):
    sample = api.rvs(alpha=1.5, beta=0.0, size=200, random_state=6)
    assert list(api.fit(sample, par=par).to_series().index) == names


def test_to_series_can_report_in_another_parametrization():
    sample = api.rvs(alpha=1.5, beta=0.0, size=200, random_state=7)
    result = api.fit(sample)
    np.testing.assert_allclose(
        result.to_series("B").to_numpy(), result.params.to_par("B"), rtol=0, atol=0)


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------

def test_fitting_a_wide_frame_is_refused_rather_than_pooled(frame):
    with pytest.raises(ValueError, match="single-column DataFrame"):
        api.fit(frame)


def test_the_refusal_says_what_to_do_instead(frame):
    with pytest.raises(ValueError, match=r"df\[column\]"):
        api.fit(frame)


def test_validation_still_runs_on_pandas_input(series):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        api.pdf(series, alpha=0.3, beta=0.0)


# --------------------------------------------------------------------------
# the core never sees pandas
# --------------------------------------------------------------------------

def test_the_numerical_core_receives_a_plain_array(monkeypatch, series):
    seen = []
    import levy.api

    original = levy.api._levy

    def recording(x, *args, **kwargs):
        seen.append(type(x))
        return original(x, *args, **kwargs)

    monkeypatch.setattr(levy.api, "_levy", recording)
    api.pdf(series, **PARAMS)

    assert seen == [np.ndarray], f"the core was handed {seen}"


def test_deprecated_levy_does_not_grow_pandas_support(series):
    # The 1.x function is frozen. pandas support is a 2.0 feature of levy.api;
    # adding it to levy.levy would change what a 1.x caller gets back.
    import warnings

    import levy

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = levy.levy(series, 1.5, -0.3, 0.2, 1.1)
    assert isinstance(result, np.ndarray)
