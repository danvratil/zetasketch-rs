"""
Python-specific integration tests for zetasketch Python bindings.

These tests focus on Python-specific behaviors, memory management,
and integration with Python standard library features.
"""

import gc
import pytest
import sys
import threading
import weakref
from zetasketch_py import HyperLogLogPlusPlus


class TestObjectLifecycle:
    """Test object creation, destruction, and memory management."""

    def test_object_creation_destruction(self) -> None:
        """Test that objects can be created and destroyed cleanly."""
        # Create many objects to test memory management
        objects = []
        for i in range(100):
            hll = HyperLogLogPlusPlus(int)
            hll.add(i)
            objects.append(hll)

        # Delete all objects
        del objects
        gc.collect()  # Force garbage collection

        # Should not crash
        assert True

    def test_weak_references(self) -> None:
        """Test that objects cannot be weakly referenced (expected for PyO3 objects)."""
        hll = HyperLogLogPlusPlus(int)

        # PyO3 objects typically don't support weak references
        with pytest.raises(TypeError, match="cannot create weak reference"):
            weakref.ref(hll)


class TestPythonTypeIntegration:
    """Test integration with Python's type system and standard library."""

    def test_isinstance_checks(self) -> None:
        """Test isinstance checks work correctly."""
        hll = HyperLogLogPlusPlus(int)

        assert isinstance(hll, HyperLogLogPlusPlus)
        assert not isinstance(hll, int)
        assert not isinstance(hll, str)
        assert not isinstance(hll, list)

    def test_type_and_class_attributes(self) -> None:
        """Test type and class attributes."""
        hll = HyperLogLogPlusPlus(int)

        assert type(hll) is HyperLogLogPlusPlus
        assert hll.__class__ == HyperLogLogPlusPlus
        assert hasattr(hll, "add")
        assert hasattr(hll, "merge")
        assert hasattr(hll, "result")
        assert hasattr(hll, "num_values")
        assert hasattr(hll, "to_bytes")
        assert hasattr(hll, "normal_precision")
        assert hasattr(hll, "sparse_precision")

    def test_dir_output(self) -> None:
        """Test that dir() shows expected methods and attributes."""
        hll = HyperLogLogPlusPlus(int)
        dir_output = dir(hll)

        expected_methods = [
            "add",
            "merge",
            "result",
            "num_values",
            "to_bytes",
            "from_bytes",
        ]
        expected_attributes = ["normal_precision", "sparse_precision"]

        for method in expected_methods:
            assert method in dir_output

        for attr in expected_attributes:
            assert attr in dir_output

    def test_hash_and_equality(self) -> None:
        """Test hashing and equality behavior."""
        hll1 = HyperLogLogPlusPlus(int)
        hll2 = HyperLogLogPlusPlus(int)

        hll1.add(42)
        hll2.add(42)

        # Objects should not be equal even with same data (separate instances)
        assert hll1 is not hll2

        # Test that objects are hashable (or not, depending on implementation)
        try:
            hash(hll1)
            # If hashable, different instances should have different hashes
            # (unless specifically designed otherwise)
        except TypeError:
            # If not hashable, that's also valid behavior
            pass

    def test_string_conversions(self) -> None:
        """Test string conversion methods."""
        hll = HyperLogLogPlusPlus(int)
        hll.add(42)

        # Test repr
        repr_str = repr(hll)
        assert isinstance(repr_str, str)
        assert len(repr_str) > 0

        # Test str (might be same as repr)
        str_str = str(hll)
        assert isinstance(str_str, str)
        assert len(str_str) > 0


class TestSerializationCompatibility:
    """Test serialization compatibility with Python standards."""

    def test_bytes_serialization_consistency(self) -> None:
        """Test that serialization is consistent across calls."""
        hll = HyperLogLogPlusPlus(int)
        for i in range(100):
            hll.add(i)

        # Serialize multiple times
        data1 = hll.to_bytes()
        data2 = hll.to_bytes()
        data3 = hll.to_bytes()

        # Should be identical
        assert data1 == data2 == data3
        assert isinstance(data1, bytes)


class TestConcurrencyBehavior:
    """Test behavior in concurrent/threaded scenarios."""

    def test_thread_safety_basic(self) -> None:
        """Basic thread safety test."""
        hll = HyperLogLogPlusPlus(int)
        results = []
        errors = []

        def worker(thread_id: int, start_val: int, end_val: int) -> None:
            try:
                for i in range(start_val, end_val):
                    hll.add(i)
                results.append(f"Thread {thread_id} completed")
            except Exception as e:
                errors.append(f"Thread {thread_id} error: {e}")

        # Create multiple threads
        threads = []
        for i in range(4):
            thread = threading.Thread(target=worker, args=(i, i * 250, (i + 1) * 250))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Check for errors
        if errors:
            pytest.fail(f"Thread errors occurred: {errors}")

        # Verify some reasonable result (exact result may vary due to concurrency)
        final_result = hll.result()
        assert final_result > 0
        assert hll.num_values() == 1000  # All additions should be counted

    def test_separate_objects_thread_safety(self) -> None:
        """Test that separate objects work fine in different threads."""
        results = {}
        errors = []

        def worker(thread_id: int) -> None:
            try:
                # Each thread gets its own HLL object
                local_hll = HyperLogLogPlusPlus(int)
                for i in range(100):
                    local_hll.add(thread_id * 100 + i)
                results[thread_id] = local_hll.result()
            except Exception as e:
                errors.append(f"Thread {thread_id} error: {e}")

        # Create threads with separate objects
        threads = []
        for i in range(4):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify results
        if errors:
            pytest.fail(f"Thread errors: {errors}")

        assert len(results) == 4
        for _, result in results.items():
            assert 90 <= result <= 110  # Each thread should have ~100 unique values


class TestPythonVersionCompatibility:
    """Test compatibility across Python versions."""

    def test_python_version_features(self) -> None:
        """Test features work across Python versions."""
        hll = HyperLogLogPlusPlus(int)

        # Test with different integer types available in different Python versions
        if sys.version_info >= (3, 8):
            # Python 3.8+ features
            hll.add(42)

        # Features available in all supported versions
        hll.add(int(42))
        hll.add(int(-42))

        assert hll.num_values() >= 1


class TestIntegrationWithPythonEcosystem:
    """Test integration with common Python libraries and patterns."""

    def test_with_collections(self) -> None:
        """Test using HLL with Python collections."""
        from collections import defaultdict

        # Using HLL in a defaultdict
        hll_dict: defaultdict[str, HyperLogLogPlusPlus[int]] = defaultdict(lambda: HyperLogLogPlusPlus(int))

        # Add data to different HLLs
        for category in ["A", "B", "C"]:
            for i in range(100):
                hll_dict[category].add(i)

        # Verify all categories have data
        for category in ["A", "B", "C"]:
            assert hll_dict[category].result() > 80

    def test_with_itertools(self) -> None:
        """Test using HLL with itertools."""
        import itertools

        hll = HyperLogLogPlusPlus(int)

        # Use itertools to generate data
        for i in itertools.islice(itertools.count(0), 1000):
            hll.add(i)

        assert hll.result() > 900
        assert hll.num_values() == 1000

    def test_with_json_serialization(self) -> None:
        """Test serializing HLL data with JSON."""
        import json
        import base64

        hll = HyperLogLogPlusPlus(int)
        for i in range(100):
            hll.add(i)

        # Serialize HLL data to JSON-compatible format
        hll_bytes = hll.to_bytes()
        encoded = base64.b64encode(hll_bytes).decode("ascii")

        json_data = {
            "hll_data": encoded,
            "metadata": {
                "normal_precision": hll.normal_precision,
                "sparse_precision": hll.sparse_precision,
                "num_values": hll.num_values(),
                "result": hll.result(),
            },
        }

        # Should be JSON serializable
        json_str = json.dumps(json_data)
        reconstructed = json.loads(json_str)

        # Reconstruct HLL from JSON data
        decoded_bytes = base64.b64decode(reconstructed["hll_data"])
        reconstructed_hll: HyperLogLogPlusPlus[int] = HyperLogLogPlusPlus.from_bytes(decoded_bytes)

        # Verify data integrity
        assert reconstructed_hll.result() == hll.result()
        assert reconstructed_hll.num_values() == hll.num_values()
