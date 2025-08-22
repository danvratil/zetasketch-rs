"""
Error handling and edge case tests for zetasketch Python bindings.

These tests focus on error conditions, invalid inputs, and boundary cases
to ensure robust error handling in the Python bindings.
"""

import pytest
from zetasketch_py import HyperLogLogPlusPlus


class TestConstructorErrorHandling:
    """Test error handling in constructor."""

    @pytest.mark.parametrize("invalid_type", [None, "str", 42, [], {}])
    def test_invalid_type_parameter(self, invalid_type: object) -> None:
        """Test constructor with invalid type parameter."""

        with pytest.raises(TypeError):
            HyperLogLogPlusPlus(invalid_type) # type: ignore[arg-type]

    @pytest.mark.parametrize("precision", [-1, 0, 3, 25, 3.14, "15", None])
    def test_invalid_precision_values(self, precision: object) -> None:
        """Test constructor with invalid precision values."""

        with pytest.raises((ValueError, TypeError)):
            HyperLogLogPlusPlus(int, normal_precision=precision) # type: ignore[arg-type]

    @pytest.mark.parametrize("precision", [-1, 3.14, "20"])
    def test_invalid_sparse_precision_values(self, precision: object) -> None:
        """Test constructor with invalid sparse_precision values."""

        with pytest.raises((ValueError, TypeError)):
            HyperLogLogPlusPlus(int, sparse_precision=precision)  # type: ignore[arg-type]

    def test_too_many_positional_arguments(self) -> None:
        """Test constructor with too many positional arguments."""
        with pytest.raises(TypeError):
            HyperLogLogPlusPlus(int, 15, 20, 25)  # type: ignore[call-arg]

    def test_unknown_keyword_arguments(self) -> None:
        """Test constructor with unknown keyword arguments."""
        with pytest.raises(TypeError):
            HyperLogLogPlusPlus(int, unknown_param=42) # type: ignore[call-arg]


class TestSerializationErrorHandling:
    """Test error handling in serialization/deserialization."""

    @pytest.mark.parametrize(
        "invalid_data",
        [
            pytest.param(b"", id="empty bytes"),
            pytest.param(b"not_protobuf_data", id="invalid protobuf"),
            pytest.param(b"\x00" * 100, id="all zeros"),
            pytest.param(b"\xff" * 100, id="all ones"),
            pytest.param(b"random garbage data", id="random text"),
            pytest.param(bytes(range(256)), id="all byte values"),
        ],
    )
    def test_from_bytes_with_invalid_data(self, invalid_data: bytes) -> None:
        """Test from_bytes with various invalid data."""
        with pytest.raises(Exception):  # Should raise some exception
            HyperLogLogPlusPlus.from_bytes(invalid_data)

    @pytest.mark.parametrize(
        "invalid_input",
        [
            pytest.param("string", id="string instead of bytes"),
            pytest.param(123, id="integer"),
            pytest.param([1, 2, 3], id="list"),
            pytest.param(None, id="None"),
            pytest.param(bytearray(b"test"), id="bytearray"),
        ],
    )
    def test_from_bytes_with_wrong_types(self, invalid_input: object) -> None:
        """Test from_bytes with wrong input types."""

        with pytest.raises((TypeError, ValueError)):
            HyperLogLogPlusPlus.from_bytes(invalid_input) # type: ignore[arg-type]

    def test_from_bytes_without_arguments(self) -> None:
        """Test from_bytes called without arguments."""
        with pytest.raises(TypeError):
            HyperLogLogPlusPlus.from_bytes() # type: ignore[call-arg]

    def test_from_bytes_with_too_many_arguments(self) -> None:
        """Test from_bytes called with too many arguments."""
        with pytest.raises(TypeError):
            HyperLogLogPlusPlus.from_bytes(b"data1", b"data2") # type: ignore[call-arg]


class TestMergeErrorHandling:
    """Test error handling in merge operations."""

    @pytest.mark.parametrize(
        "invalid_input", ["string", 123, 3.14, [], {}, None, set()]
    )
    def test_merge_with_invalid_types(self, invalid_input: object) -> None:
        """Test merge with various invalid types."""
        hll = HyperLogLogPlusPlus(int)
        hll.add(42)

        with pytest.raises((TypeError, RuntimeError)):
            hll.merge(invalid_input) # type: ignore[arg-type]

    def test_merge_sketches_with_incompatible_precision(self) -> None:
        """Test merging sketches with incompatible precisions."""
        # Create sketches with different precisions
        hll1 = HyperLogLogPlusPlus(int, normal_precision=12, sparse_precision=15)
        hll2 = HyperLogLogPlusPlus(int, normal_precision=10, sparse_precision=17)

        hll1.add(1)
        hll2.add(2)

        with pytest.raises(RuntimeError, match="Incompatible precision"):
            hll1.merge(hll2)

        serialized = hll2.to_bytes()
        with pytest.raises(RuntimeError, match="Incompatible precision"):
            hll1.merge(serialized)

    def test_merge_incompatible_sketch(self) -> None:
        """Test that merging sketches of different types raises RuntimeError."""

        hll_int = HyperLogLogPlusPlus(int)
        hll_str = HyperLogLogPlusPlus(str)
        hll_int.add(42)
        hll_str.add("hello")

        with pytest.raises(
            RuntimeError, match="Aggregator of type {LONG} is incompatible.*"
        ):
            hll_int.merge(hll_str) # type: ignore[arg-type]

        # Also test merging serialized incompatible sketch
        serialized_str = hll_str.to_bytes()
        with pytest.raises(
            RuntimeError, match="Aggregator of type {LONG} is incompatible.*"
        ):
            hll_int.merge(serialized_str)

    def test_merge_without_arguments(self) -> None:
        """Test merge called without arguments."""
        hll = HyperLogLogPlusPlus(int)

        with pytest.raises(TypeError):
            hll.merge() # type: ignore[call-arg]

    def test_merge_with_too_many_arguments(self) -> None:
        """Test merge called with too many arguments."""
        hll1 = HyperLogLogPlusPlus(int)
        hll2 = HyperLogLogPlusPlus(int)

        with pytest.raises(TypeError):
            hll1.merge(hll2, hll2) # type: ignore[call-arg]


class TestAddErrorHandling:
    """Test error handling in add operations."""

    def test_add_without_arguments(self) -> None:
        """Test add called without arguments."""
        hll = HyperLogLogPlusPlus(int)

        with pytest.raises(TypeError):
            hll.add() # type: ignore[call-arg]

    def test_add_with_too_many_arguments(self) -> None:
        """Test add called with too many arguments."""
        hll = HyperLogLogPlusPlus(int)

        with pytest.raises(TypeError):
            hll.add(1, 2) # type: ignore[call-arg]

    def test_add_with_complex_invalid_types(self) -> None:
        """Test add with complex invalid types."""
        hll = HyperLogLogPlusPlus(int)

        invalid_values = [
            {"key": "value"},  # Dict
            [1, 2, 3],  # List
            (1, 2, 3),  # Tuple
            {1, 2, 3},  # Set
            lambda x: x,  # Function
            open(__file__),  # File object (will be cleaned up)
        ]

        for invalid_value in invalid_values:
            try:
                with pytest.raises((TypeError, RuntimeError)):
                    hll.add(invalid_value) # type: ignore[arg-type]
            except Exception:
                # Some complex types might not even be convertible
                pass
            finally:
                # Clean up file handle if it was opened
                if hasattr(invalid_value, "close"):
                    invalid_value.close()


class TestPropertyAccessErrors:
    """Test error handling for property access."""

    def test_precision_properties_are_readonly(self) -> None:
        """Test that precision properties cannot be modified."""
        hll = HyperLogLogPlusPlus(int)

        with pytest.raises(AttributeError):
            hll.normal_precision = 10

        with pytest.raises(AttributeError):
            hll.sparse_precision = 15

    def test_nonexistent_properties(self) -> None:
        """Test access to nonexistent properties."""
        hll = HyperLogLogPlusPlus(int)

        with pytest.raises(AttributeError):
            _ = hll.nonexistent_property # type: ignore[attr-defined]

        with pytest.raises(AttributeError):
            hll.another_fake_property = 42 # type: ignore[attr-defined]


class TestEdgeCases:
    """Test various edge cases and boundary conditions."""

    def test_extremely_large_strings(self) -> None:
        """Test adding extremely large strings."""
        hll = HyperLogLogPlusPlus(str)

        # Very large string (10MB)
        large_string = "x" * (10 * 1024 * 1024)

        # Should handle large strings without crashing
        hll.add(large_string)
        assert hll.num_values() == 1
        assert hll.result() == 1

    def test_extremely_large_bytes(self) -> None:
        """Test adding extremely large bytes."""
        hll = HyperLogLogPlusPlus(bytes)

        # Large bytes object (1MB)
        large_bytes = b"x" * (1024 * 1024)

        # Should handle large bytes without crashing
        hll.add(large_bytes)
        assert hll.num_values() == 1
        assert hll.result() == 1

    def test_unicode_edge_cases(self) -> None:
        """Test various Unicode edge cases."""
        hll = HyperLogLogPlusPlus(str)

        unicode_strings = [
            "\u0000",  # Null character
            "\uffff",  # Max BMP character
            "\U0010ffff",  # Max Unicode character
            "🏳️‍🌈",  # Complex emoji with ZWJ
            "A\u0300\u0301",  # Combining characters
            "\r\n\t",  # Various whitespace
            "𝕌𝕟𝕚𝔠𝕠𝔡𝔢",  # Mathematical script characters
        ]

        for unicode_str in unicode_strings:
            hll.add(unicode_str)

        assert hll.num_values() == len(unicode_strings)

    def test_integer_boundary_values(self) -> None:
        """Test integer boundary values."""
        hll = HyperLogLogPlusPlus(int)

        boundary_values = [
            0,  # Zero
            1,  # One
            -1,  # Negative one
            2**31 - 1,  # Max 32-bit signed int
            -(2**31),  # Min 32-bit signed int
            2**32 - 1,  # Max 32-bit unsigned int
            2**63 - 1,  # Max 64-bit signed int
            -(2**63),  # Min 64-bit signed int (might fail if not supported)
        ]

        for value in boundary_values:
            try:
                hll.add(value)
            except OverflowError:
                # Some very large values might cause overflow
                pass

    def test_empty_values(self) -> None:
        """Test adding empty values."""
        # Empty string
        hll_str = HyperLogLogPlusPlus(str)
        hll_str.add("")
        assert hll_str.num_values() == 1

        # Empty bytes
        hll_bytes = HyperLogLogPlusPlus(bytes)
        hll_bytes.add(b"")
        assert hll_bytes.num_values() == 1

    def test_many_operations_on_same_sketch(self) -> None:
        """Test performing many operations on the same sketch."""
        hll = HyperLogLogPlusPlus(int)

        # Add many values
        for i in range(1000):
            hll.add(i)

        # Multiple serialization/deserialization cycles
        for _ in range(10):
            data = hll.to_bytes()
            hll_copy: HyperLogLogPlusPlus[int] = HyperLogLogPlusPlus.from_bytes(data)

            # Verify consistency
            assert hll.result() == hll_copy.result()
            assert hll.num_values() == hll_copy.num_values()

        # Multiple merge operations
        for i in range(5):
            other = HyperLogLogPlusPlus(int)
            for j in range(100):
                other.add(1000 + i * 100 + j)
            hll.merge(other)

    def test_sketch_after_many_merges(self) -> None:
        """Test sketch behavior after many merge operations."""
        main_hll = HyperLogLogPlusPlus(int)

        # Create and merge many small sketches
        for i in range(20):
            temp_hll = HyperLogLogPlusPlus(int)
            for j in range(50):
                temp_hll.add(i * 50 + j)
            main_hll.merge(temp_hll)

        # Should have approximately 1000 unique values
        result = main_hll.result()
        assert 800 <= result <= 1200  # Allow for estimation error
