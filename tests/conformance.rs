// SPDX-FileCopyrightText: 2025 Daniel Vrátil <me@dvratil.cz>
//
// SPDX-License-Identifier: MIT

use assert2::assert;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use zetasketch_java::Zetasketch as JavaZetasketch;
use zetasketch_rs::{Aggregator, HyperLogLogPlusPlus};

#[test]
fn test_compatibility_with_java_implementation() {
    let java_zs = JavaZetasketch::new().expect("Failed to create Zetasketch JVM");
    let java_builder = java_zs.builder().expect("Failed to create Java builder");
    let java_hll = java_builder
        .build_for_longs()
        .expect("Failed to create Java HLL for Longs");

    let mut rust_hll = HyperLogLogPlusPlus::builder()
        .build_for_u64()
        .expect("Failed to create Rust HLL for u64");

    let mut rng = StdRng::seed_from_u64(42);
    let num_values = rng.random_range(10_000..100_000);
    for _i in 0..num_values {
        let value = rng.random_range(0..10_000);

        java_hll
            .add(value)
            .expect("Failed to add value to Java HLL");
        rust_hll
            .add_u64(value)
            .expect("Failed to add value to Rust HLL");

        if _i % 1000 == 0 {
            assert_eq!(
                java_hll
                    .result()
                    .expect("Failed to get result from Java HLL"),
                rust_hll
                    .result()
                    .expect("Failed to get result from Rust HLL")
            );
        }
    }

    assert!(
        java_hll
            .num_values()
            .expect("Failed to get number of values from Java HLL")
            == rust_hll.num_values()
    );
    assert!(
        java_hll
            .result()
            .expect("Failed to get result from Java HLL")
            == rust_hll
                .result()
                .expect("Failed to get result from Rust HLL")
    );

    let java_bytes = java_hll
        .serialize_to_byte_array()
        .expect("Failed to serialize Java HLL to byte array");
    let rust_bytes = rust_hll
        .serialize_to_bytes()
        .expect("Failed to serialize Rust HLL to bytes");

    assert!(java_bytes == rust_bytes);

    let rust_hll = HyperLogLogPlusPlus::from_bytes(&java_bytes)
        .expect("Failed to deserialize Rust HLL from byte array");
    assert!(
        java_hll
            .result()
            .expect("Failed to get result from Java HLL")
            == rust_hll
                .result()
                .expect("Failed to get result from Rust HLL")
    );

    let java_hll = java_zs
        .hll_for_bytes::<u64>(&rust_bytes)
        .expect("Failed to create Java HLL from Rust bytes");
    assert!(
        java_hll
            .result()
            .expect("Failed to get result from Java HLL")
            == rust_hll
                .result()
                .expect("Failed to get result from Rust HLL")
    );

    // TODO: test merging from Java to Rust and vice versa
    // TODO: test merging from Rust to Java and vice versa
}
