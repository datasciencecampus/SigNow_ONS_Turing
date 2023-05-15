""" Unit tests for the find_sigs functions.

In the pipeline these all operate together when the user defined
`model_param['keep_sigs']` is set to anything other than 'all'.
Other possible arguments for `model_param['keep_sigs']` are `'all_linear'` and `'innermost'`.

`compute_linear_sigs` is a wrapper function to compute the indices of all linear
signature terms.

It takes accepts as arguments:

    dim
        Dimension of the features.
    level
        Level of truncation of the signatures.

These same arguments are then passed onto `find_innermost_sigs`, `find_outermost_sigs` and
`find_middle_sigs`.

Find innermost returns indices corresponding to signatures of the form (x..).

Find outermost returns indices corresponding to signatures of the form (..x).

Find middle returns indices corresponding to signatures of the form (.x.).

"""
import hypothesis.strategies as st
from hypothesis import given

from signow.signature_functions.compute_linear_sigs import (
    find_innermost_sigs,
    find_outermost_sigs,
    find_middle_sigs,
    compute_linear_sigs,
)


class TestFindSigsPropertyBased:
    """A class that tests the group of functions that finds signature terms.

    These tests are property based.

    """
    @given(
        dim=st.integers(min_value=0, max_value=5),
        level=st.integers(min_value=0, max_value=5),
    )
    def test_find_innermost_sigs_expected_num(self, dim, level):
        # When
        innermost_indices = find_innermost_sigs(dim, level)

        # Then
        n_indices = len(innermost_indices)
        n_indices_expected = (dim * level) + 1

        assert n_indices == n_indices_expected

    @given(
        dim=st.integers(min_value=0, max_value=5),
        level=st.integers(min_value=0, max_value=5),
    )
    def test_find_outermost_sigs_expected_num(self, dim, level):
        # When
        outermost_indices = find_innermost_sigs(dim, level)

        # Then
        n_indices = len(outermost_indices)
        n_indices_expected = (dim * level) + 1

        assert n_indices == n_indices_expected

    @given(
        dim=st.integers(min_value=1, max_value=5),
        level=st.integers(min_value=2, max_value=5),
    )
    def test_find_middle_sigs_expected_num(self, dim, level):
        # NOTE: middle indices only returns values when the level is > 3 and when
        # dimensions > 1

        # When
        middle_indices = find_middle_sigs(dim, level)

        # Then
        n_indices = len(middle_indices)

        if level < 3:
            assert n_indices == 0
        elif dim == 1:
            assert n_indices == 0
        else:
            assert n_indices < dim * level

    @given(
        dim=st.integers(min_value=1, max_value=5),
        level=st.integers(min_value=2, max_value=5),
    )
    def test_compute_linear_sigs_expected_num(self, dim, level):
        # When
        innermost_indices = find_innermost_sigs(dim, level)
        outermost_indices = find_outermost_sigs(dim, level)
        middle_indices = find_middle_sigs(dim, level)

        # Then
        n_indices_combinded = len(
            set(innermost_indices + outermost_indices + middle_indices)
        )
        ttl_indices_combined = len(
            innermost_indices + outermost_indices + middle_indices
        )

        assert n_indices_combinded <= ttl_indices_combined

    @given(
        dim=st.integers(min_value=1, max_value=5),
        level=st.integers(min_value=1, max_value=5),
    )
    def test_find_innermost_sigs_expected_diffs(self, dim, level):
        # When
        innermost_indices = find_innermost_sigs(dim, level)

        # Then
        differences = set(
            [j - i for i, j in zip(innermost_indices[:-1], innermost_indices[1:])]
        )
        differences_expected = set([dim**i for i in range(0, level)])

        assert differences == differences_expected

    @given(
        dim=st.integers(min_value=1, max_value=5),
        level=st.integers(min_value=1, max_value=5),
    )
    def test_find_outermost_sigs_expected_num_diffs(self, dim, level):
        # When
        outer_indices = find_outermost_sigs(dim, level)

        # Then
        num_differences = len(
            set([j - i for i, j in zip(outer_indices[:-1], outer_indices[1:])])
        )

        if dim == 1:
            num_differences_expected = dim
        elif level <= 2:
            num_differences_expected = 1
        else:
            num_differences_expected = level - 1

        assert num_differences == num_differences_expected

    @given(
        dim=st.integers(min_value=2, max_value=10),
        level=st.integers(min_value=4, max_value=8),
    )
    def test_find_middle_sigs_expected_diffs_modulus(self, dim, level):
        # NOTE: middle indices only returns values when the level is > 3 and when
        # dimensions > 1

        # When
        middle_indices = find_middle_sigs(dim, level)

        # Then
        differences_mod = set(
            [(j - i) % dim for i, j in zip(middle_indices[:-1], middle_indices[1:])]
        )

        if not len(differences_mod):
            # fail the test
            assert 1 == 0
        else:
            assert differences_mod.pop() == 0


class TestFindSigsOutputBased:
    """A class that tests the group of functions that finds signature terms.

    These tests are output based.

    """

    def test_find_innermost_sigs_output(self):
        # Given
        dim = 5
        level = 2

        # When
        innermost_indices = find_innermost_sigs(dim, level)

        # Then
        expected_indices = [0, 1, 2, 3, 4, 5, 6, 11, 16, 21, 26]

        assert innermost_indices == expected_indices

    def test_find_outermost_sigs_output(self):
        # Given
        dim = 5
        level = 2

        # When
        outermost_indices = find_outermost_sigs(dim, level)

        # Then
        expected_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        assert outermost_indices == expected_indices

    def test_find_middle_sigs_output(self):
        # Given
        dim = 5
        level = 3

        # When
        middle_indices = find_middle_sigs(dim, level)

        # Then
        expected_indices = [36, 41, 46, 51]

        assert middle_indices == expected_indices

    def test_compute_linear_sigs_output(self):
        # Given
        dim = 5
        level = 2

        # When
        combined_indices = compute_linear_sigs(dim, level)

        # Then
        expected_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 21, 26]

        assert combined_indices == expected_indices
