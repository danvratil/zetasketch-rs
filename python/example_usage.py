#!/usr/bin/env python3
"""
Example usage of zetasketch Python bindings.

This demonstrates how to use the Python bindings for zetasketch-rs,
a Rust implementation of Google's ZetaSketch HyperLogLog++ algorithm.
"""

from zetasketch_py import HyperLogLogPlusPlus


def main() -> None:
    print("🎯 ZetaSketch Python Bindings Example")
    print("=" * 40)

    # Create a HyperLogLog++ sketch for integers
    print("\n1. Creating HLL++ sketch for integers...")
    hll = HyperLogLogPlusPlus(int)

    # Add some values
    print("2. Adding 10,000 unique integers...")
    for i in range(10000):
        hll.add(i)

    print(f"   Added values: {hll.num_values()}")
    print(f"   Estimated cardinality: {hll.result()}")
    print(f"   Accuracy: {abs(hll.result() - 10000) / 10000 * 100:.2f}% error")

    # Test serialization
    print("\n3. Testing serialization...")
    serialized = hll.to_bytes()
    print(f"   Serialized size: {len(serialized)} bytes")

    # Deserialize
    hll2: HyperLogLogPlusPlus[int] = HyperLogLogPlusPlus.from_bytes(serialized)
    print(f"   Deserialized cardinality: {hll2.result()}")

    # Test merging
    print("\n4. Testing sketch merging...")
    hll3 = HyperLogLogPlusPlus(int)

    # Add overlapping values (5000-14999)
    for i in range(5000, 15000):
        hll3.add(i)

    print(f"   Second sketch cardinality: {hll3.result()}")

    # Merge the sketches
    hll.merge(hll3)
    print(f"   Merged cardinality: {hll.result()}")
    print("   Expected ~15,000 (union of 0-9999 and 5000-14999)")

    # Test string sketches
    print("\n5. Testing string sketches...")
    hll_str = HyperLogLogPlusPlus(str)

    words = ["apple", "banana", "cherry", "date", "elderberry"] * 1000
    for word in words:
        hll_str.add(word)

    print(f"   Added {len(words)} string values")
    print(f"   Estimated unique strings: {hll_str.result()}")
    print("   Actual unique strings: 5")

    print("\n✅ Example completed successfully!")


if __name__ == "__main__":
    main()
