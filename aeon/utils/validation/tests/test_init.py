"""Test functions in utils.validation.__init__."""

import pytest

from aeon.utils.validation import check_window_length


@pytest.mark.parametrize(
    "window_length, n_timepoints, expected",
    [
        (0.2, 33, 7),
        (43, 23, 43),
        (33, 1, 33),
        (33, None, 33),
        (None, 19, None),
        (None, None, None),
        (67, 0.3, 67),  # bad arg
    ],
)
def test_check_window_length(window_length, n_timepoints, expected):
    """Test check_window_length."""
    assert check_window_length(window_length, n_timepoints) == expected


@pytest.mark.parametrize(
    "window_length, n_timepoints",
    [
        ("string", 34),
        ("string", "string"),
        (6.2, 33),
        (-5, 34),
        (-0.5, 15),
        (6.1, 0.3),
        (0.3, 0.1),
        (-2.4, 10),
        (0.2, None),
    ],
)
def test_window_length_bad_arg(window_length, n_timepoints):
    """Test check_window_length with bad arguments."""
    with pytest.raises(ValueError):
        check_window_length(window_length, n_timepoints)
