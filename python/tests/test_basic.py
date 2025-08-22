import pytest
from zetasketch_py import HyperLogLogPlusPlus


def test_new_hll_with_invalid_type_throws() -> None:
    """Test that creating a HyperLogLog++ sketch with an invalid type raises an error."""
    with pytest.raises(ValueError):
        HyperLogLogPlusPlus(float)  # type: ignore[type-var]


def test_new_hll_with_default_precisions() -> None:
    """Test that creating a HyperLogLog++ sketch with default precisions works."""
    hll = HyperLogLogPlusPlus(int)
    assert hll.normal_precision == 15
    assert hll.sparse_precision == 20


def test_new_hll_with_custom_precisions() -> None:
    """Test that creating a HyperLogLog++ sketch with custom precisions works."""
    hll = HyperLogLogPlusPlus(int, normal_precision=12, sparse_precision=17)
    assert hll.normal_precision == 12
    assert hll.sparse_precision == 17


def test_new_hll_without_sparse_precision() -> None:
    """Test that creating a HyperLogLog++ sketch without sparse precision uses default."""
    hll = HyperLogLogPlusPlus(int, sparse_precision=None)
    assert hll.normal_precision == 15
    assert hll.sparse_precision == 0


def test_basic_functionality() -> None:
    """Test basic HyperLogLog++ functionality."""
    # Create a sketch for integers
    hll = HyperLogLogPlusPlus(int)

    # Add some values
    for i in range(1000):
        hll.add(i)

    # Check basic properties
    assert hll.num_values() == 1000
    result = hll.result()
    assert 950 <= result <= 1050  # Should be approximately 1000


def test_builder_configuration() -> None:
    """Test builder pattern with custom precision."""
    hll = HyperLogLogPlusPlus(str, normal_precision=12, sparse_precision=20)

    # Add some string values
    for i in range(100):
        hll.add(f"value_{i}")

    assert hll.num_values() == 100
    result = hll.result()
    assert 90 <= result <= 110  # Should be approximately 100


def test_bytes_operations() -> None:
    """Test serialization and deserialization."""
    # Create and populate a sketch
    hll1 = HyperLogLogPlusPlus(int)
    for i in range(500):
        hll1.add(i)

    # Serialize to bytes
    serialized = hll1.to_bytes()
    assert isinstance(serialized, bytes)
    assert len(serialized) > 0

    # Deserialize from bytes
    hll2: HyperLogLogPlusPlus[int] = HyperLogLogPlusPlus.from_bytes(serialized)

    # Both should have same properties
    assert hll1.num_values() == hll2.num_values()
    assert hll1.result() == hll2.result()


def test_merge_hll_objects() -> None:
    """Test merging sketches."""
    # Create two sketches
    hll1 = HyperLogLogPlusPlus(int)
    hll2 = HyperLogLogPlusPlus(int)

    # Add different values to each
    for i in range(500):
        hll1.add(i)
    for i in range(250, 750):  # 250 overlap
        hll2.add(i)

    # Merge hll2 into hll1
    hll1.merge(hll2)

    # Should have approximately 750 unique values (0-749)
    result = hll1.result()
    assert 700 <= result <= 800


def test_merge_bytes() -> None:
    """Test merging serialized sketch."""
    # Create two sketches
    hll1 = HyperLogLogPlusPlus(int)
    hll2 = HyperLogLogPlusPlus(int)

    # Add different values to each
    for i in range(500):
        hll1.add(i)
    for i in range(250, 750):  # 250 overlap
        hll2.add(i)

    # Serialize hll2
    serialized = hll2.to_bytes()

    # Merge serialized hll2 into hll1
    hll1.merge(serialized)

    # Should have approximately 750 unique values (0-749)
    result = hll1.result()
    assert 700 <= result <= 800


def test_string_operations() -> None:
    """Test string-specific operations."""
    hll = HyperLogLogPlusPlus(str)

    strings = [f"test_string_{i}" for i in range(200)]
    for s in strings:
        hll.add(s)

    assert hll.num_values() == 200
    result = hll.result()
    assert 180 <= result <= 220


def test_bytes_input() -> None:
    """Test bytes input operations."""
    hll = HyperLogLogPlusPlus(bytes)

    for i in range(150):
        hll.add(f"bytes_data_{i}".encode("utf-8"))

    assert hll.num_values() == 150
    result = hll.result()
    assert 130 <= result <= 170


def test_repr() -> None:
    """Test string representations."""
    hll = HyperLogLogPlusPlus(int)
    hll.add(42)
    repr_str = repr(hll)
    assert "HyperLogLogPlusPlus" in repr_str
    assert "cardinality=" in repr_str
    assert "num_values=" in repr_str


def test_error_handling() -> None:
    """Test error handling."""
    # Test invalid serialized data
    with pytest.raises(Exception):  # Should raise some Python exception
        HyperLogLogPlusPlus.from_bytes(b"invalid_data")
