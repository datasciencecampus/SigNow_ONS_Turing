"""Unit tests for all_sig_keys function.

This function is used to create a list of names for signatures, with the first
entry corresponding to the innermost integration dimension

The function returns a list of strings that correspond to the names of signature terms.
"""

from signow.signature_functions.compute_linear_sigs import all_sig_keys


class TestAllSigKeys:
    """A class that tests function all_sig_keys"""

    def test_method(self) -> None:
        # Given
        dim = 3
        level = 2
        expected = [
            "()",
            "(1,)",
            "(2,)",
            "(3,)",
            "(1, 1)",
            "(1, 2)",
            "(1, 3)",
            "(2, 1)",
            "(2, 2)",
            "(2, 3)",
            "(3, 1)",
            "(3, 2)",
            "(3, 3)",
        ]

        # When
        keys = all_sig_keys(dim, level)

        # Then
        assert keys == expected
