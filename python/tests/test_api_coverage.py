"""
API coverage tests for Python bindings.
"""

import pytest
from zetasketch_py import HyperLogLogPlusPlus


class TestConstructorAPI:
    """Test all constructor parameter combinations and type validation."""

    def test_all_supported_types(self) -> None:
        """Test that all documented types work in constructor."""
        # All three supported types should work
        hll_int = HyperLogLogPlusPlus(int)
        hll_str = HyperLogLogPlusPlus(str)
        hll_bytes = HyperLogLogPlusPlus(bytes)

        assert isinstance(hll_int, HyperLogLogPlusPlus)
        assert isinstance(hll_str, HyperLogLogPlusPlus)
        assert isinstance(hll_bytes, HyperLogLogPlusPlus)

    @pytest.mark.parametrize("unsupported_type", [float, list, dict, tuple, set, bool])
    def test_unsupported_types_raise_error(self, unsupported_type: type) -> None:
        """Test that unsupported types raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported type"):
            HyperLogLogPlusPlus(unsupported_type)

    def test_constructor_parameter_combinations(self) -> None:
        """Test all valid constructor parameter combinations."""
        # Default parameters
        hll1 = HyperLogLogPlusPlus(int)
        assert hll1.normal_precision == 15
        assert hll1.sparse_precision == 20

        # Custom normal precision only
        hll2 = HyperLogLogPlusPlus(int, normal_precision=12)
        assert hll2.normal_precision == 12
        assert hll2.sparse_precision == 20  # Uses default sparse precision

        # Both precisions specified
        hll3 = HyperLogLogPlusPlus(int, normal_precision=12, sparse_precision=18)
        assert hll3.normal_precision == 12
        assert hll3.sparse_precision == 18

        # Sparse precision disabled
        hll4 = HyperLogLogPlusPlus(int, sparse_precision=None)
        assert hll4.normal_precision == 15
        assert hll4.sparse_precision == 0

        # Named parameters in different order
        hll5 = HyperLogLogPlusPlus(int, sparse_precision=19, normal_precision=11)
        assert hll5.normal_precision == 11
        assert hll5.sparse_precision == 19

    def test_precision_property_access(self) -> None:
        """Test that precision properties are read-only and correct type."""
        hll = HyperLogLogPlusPlus(int, normal_precision=13, sparse_precision=18)

        # Should be integers
        assert isinstance(hll.normal_precision, int)
        assert isinstance(hll.sparse_precision, int)

        # Should have correct values
        assert hll.normal_precision == 13
        assert hll.sparse_precision == 18

        # Properties should be read-only (attempting to set should fail)
        with pytest.raises(AttributeError):
            hll.normal_precision = 10
        with pytest.raises(AttributeError):
            hll.sparse_precision = 15


class TestAddMethodAPI:
    """Test the add() method with different types and edge cases."""

    def test_add_int_values(self) -> None:
        """Test adding various integer values."""
        hll = HyperLogLogPlusPlus(int)

        # Regular integers
        hll.add(42)
        hll.add(0)
        hll.add(-1)
        hll.add(1234567890)

        # Large integers
        hll.add(2**63 - 1)  # Max i64
        hll.add(-(2**63))  # Min i64

        assert hll.num_values() == 6
        assert hll.result() > 0

    def test_add_str_values(self) -> None:
        """Test adding various string values."""
        hll = HyperLogLogPlusPlus(str)

        # Different string types
        hll.add("hello")
        hll.add("")  # Empty string
        hll.add("a" * 1000)  # Long string
        hll.add("café")  # Unicode
        hll.add("🎯🔥")  # Emojis
        hll.add("Hello\nWorld\t!")  # Control characters

        assert hll.num_values() == 6
        assert hll.result() > 0

    def test_add_bytes_values(self) -> None:
        """Test adding various bytes values."""
        hll = HyperLogLogPlusPlus(bytes)

        # Different bytes types
        hll.add(b"hello")
        hll.add(b"")  # Empty bytes
        hll.add(b"\x00\x01\x02\x03")  # Binary data
        hll.add("café".encode("utf-8"))  # Encoded string
        hll.add(bytes(range(256)))  # All byte values

        assert hll.num_values() == 5
        assert hll.result() > 0

    def test_add_wrong_type_raises_error(self) -> None:
        """Test that adding wrong types raises appropriate errors."""
        hll_int = HyperLogLogPlusPlus(int)
        hll_str = HyperLogLogPlusPlus(str)
        hll_bytes = HyperLogLogPlusPlus(bytes)

        # Int HLL should reject non-ints
        with pytest.raises(TypeError, match="Unable to add type STRING"):
            hll_int.add("string") # type: ignore[arg-type]
        with pytest.raises(TypeError, match="Unable to add type BYTES"):
            hll_int.add(b"bytes")   # type: ignore[arg-type]
        with pytest.raises(TypeError, match="Unsupported type"):
            hll_int.add(3.14)    # type: ignore[arg-type]

        # String HLL should reject integers (but accepts bytes)
        with pytest.raises(TypeError, match="Unable to add type LONG"):
            hll_str.add(123)    # type: ignore[arg-type]
        with pytest.raises(TypeError, match="Unsupported type"):
            hll_str.add(3.14)   # type: ignore[arg-type]

        # Bytes should work with string HLL, but not really exposed through type system
        hll_str.add(b"bytes")  # type: ignore[arg-type]

        # Bytes HLL should reject integers (but accepts strings)
        with pytest.raises(TypeError, match="Unable to add type LONG"):
            hll_bytes.add(123)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="Unsupported type"):
            hll_bytes.add(3.14)  # type: ignore[arg-type]

        # String should work with bytes HLL, but not really exposed through type system
        hll_bytes.add("string")  # type: ignore[arg-type]

    def test_add_returns_none(self) -> None:
        """Test that add() returns None."""
        hll = HyperLogLogPlusPlus(int)
        result = hll.add(42) # type: ignore[func-returns-value]
        assert result is None


class TestMergeMethodAPI:
    """Test the merge() method with different input types."""

    def test_merge_hll_objects(self) -> None:
        """Test merging HyperLogLogPlusPlus objects."""
        hll1 = HyperLogLogPlusPlus(int)
        hll2 = HyperLogLogPlusPlus(int)

        # Add different values to each
        for i in range(100):
            hll1.add(i)
            hll2.add(i + 50)  # 50 overlap

        original_result = hll1.result()
        hll1.merge(hll2)

        # Result should be different after merge
        assert hll1.result() != original_result
        # Should have roughly 150 unique values
        assert 120 <= hll1.result() <= 180

    def test_merge_bytes_data(self) -> None:
        """Test merging serialized bytes data."""
        hll1 = HyperLogLogPlusPlus(int)
        hll2 = HyperLogLogPlusPlus(int)

        # Populate sketches
        for i in range(100):
            hll1.add(i)
            hll2.add(i + 50)  # 50 overlap

        # Serialize hll2 and merge into hll1
        serialized = hll2.to_bytes()
        original_result = hll1.result()
        hll1.merge(serialized)

        # Result should change after merge
        assert hll1.result() != original_result

    def test_merge_wrong_type_raises_error(self) -> None:
        """Test that merging wrong types raises TypeError."""
        hll = HyperLogLogPlusPlus(int)

        with pytest.raises(TypeError, match="Unsupported type.*merge"):
            hll.merge("string") # type: ignore[arg-type]
        with pytest.raises(TypeError, match="Unsupported type.*merge"):
            hll.merge(123)  # type: ignore[arg-type]
        with pytest.raises(
            RuntimeError, match="ZetaSketch error.*Proto deserialization"
        ):
            hll.merge([1, 2, 3]) # type: ignore[arg-type]

    def test_merge_returns_none(self) -> None:
        """Test that merge() returns None."""
        hll1 = HyperLogLogPlusPlus(int)
        hll2 = HyperLogLogPlusPlus(int)
        hll1.add(1)
        hll2.add(2)

        result = hll1.merge(hll2) # type: ignore[func-returns-value]
        assert result is None


class TestResultAndValueCountAPI:
    """Test result() and num_values() methods."""

    def test_result_return_type(self) -> None:
        """Test that result() returns int."""
        hll = HyperLogLogPlusPlus(int)
        hll.add(42)

        result = hll.result()
        assert isinstance(result, int)
        assert result > 0

    def test_num_values_return_type(self) -> None:
        """Test that num_values() returns int."""
        hll = HyperLogLogPlusPlus(int)
        hll.add(42)

        count = hll.num_values()
        assert isinstance(count, int)
        assert count == 1

    def test_empty_sketch_behavior(self) -> None:
        """Test behavior with empty sketch."""
        hll = HyperLogLogPlusPlus(int)

        assert hll.result() == 0
        assert hll.num_values() == 0

    def test_duplicate_values_handling(self) -> None:
        """Test that duplicates increase num_values but not result."""
        hll = HyperLogLogPlusPlus(int)

        # Add same value multiple times
        for _ in range(10):
            hll.add(42)

        assert hll.num_values() == 10
        assert hll.result() == 1  # Only one unique value


class TestSerializationAPI:
    """Test serialization and deserialization methods."""

    def test_to_bytes_return_type(self) -> None:
        """Test that to_bytes() returns bytes."""
        hll = HyperLogLogPlusPlus(int)
        hll.add(42)

        serialized = hll.to_bytes()
        assert isinstance(serialized, bytes)
        assert len(serialized) > 0

    def test_from_bytes_class_method(self) -> None:
        """Test from_bytes() class method."""
        hll1 = HyperLogLogPlusPlus(int)
        for i in range(100):
            hll1.add(i)

        # Serialize and deserialize
        data = hll1.to_bytes()
        hll2: HyperLogLogPlusPlus[int] = HyperLogLogPlusPlus.from_bytes(data)

        # Should be equivalent
        assert isinstance(hll2, HyperLogLogPlusPlus)
        assert hll1.result() == hll2.result()
        assert hll1.num_values() == hll2.num_values()
        assert hll1.normal_precision == hll2.normal_precision
        assert hll1.sparse_precision == hll2.sparse_precision

    def test_serialization_roundtrip(self) -> None:
        """Test multiple serialization round trips."""
        original = HyperLogLogPlusPlus(str, normal_precision=12, sparse_precision=17)

        for i in range(50):
            original.add(f"value_{i}")

        # Multiple round trips
        data1 = original.to_bytes()
        hll1: HyperLogLogPlusPlus[str] = HyperLogLogPlusPlus.from_bytes(data1)
        data2 = hll1.to_bytes()
        hll2: HyperLogLogPlusPlus[str] = HyperLogLogPlusPlus.from_bytes(data2)

        # All should be equivalent
        assert original.result() == hll1.result() == hll2.result()
        assert original.num_values() == hll1.num_values() == hll2.num_values()
        assert data1 == data2  # Serialized data should be identical


class TestStringRepresentation:
    """Test __repr__ method."""

    def test_repr_format(self) -> None:
        """Test that __repr__ returns expected format."""
        hll = HyperLogLogPlusPlus(int)
        hll.add(42)

        repr_str = repr(hll)

        assert isinstance(repr_str, str)
        assert "HyperLogLogPlusPlus" in repr_str
        assert "cardinality=" in repr_str
        assert "num_values=" in repr_str
        assert "1" in repr_str  # Should show cardinality and num_values as 1

    def test_repr_with_empty_sketch(self) -> None:
        """Test __repr__ with empty sketch."""
        hll = HyperLogLogPlusPlus(int)
        repr_str = repr(hll)

        assert "cardinality=0" in repr_str
        assert "num_values=0" in repr_str
