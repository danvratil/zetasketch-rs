# Zetasketch-Java

This Rust crate provides limited Rust bindings for the original Zetasketch Java
library using the amazing [`j4rs`](https://crates.io/crates/j4rs) crate.

This crate is not published separately on crates.io, as it is only intended for
use in our conformance tests to compare the behavior of the native Rust
implementation of Zetasketch with the behavior of the original Java
implementation.

The original Java library and `fastutil` are downloaded from Maven Central the
first time `Zetasketch::new` starts a JVM. That avoids starting a JVM from a
Cargo `build.rs`, which j4rs 0.25 cannot do until its `jassets` directory
already exists.

