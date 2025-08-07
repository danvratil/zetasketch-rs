// SPDX-FileCopyrightText: 2025 Daniel Vrátil <me@dvratil.cz>
//
// SPDX-License-Identifier: MIT
//
// Based on the original Zetasketch implementation by Google:
// https://github.com/google/zetasketch
// Published under the Apache License 2.0

//! This crate provides fully compatible re-implementation of the
//! [ZetaSketch Java library](https://github.com/google/zetasketch) from Google in Rust.
//!
//! The ZetaSketch library contains implementation of the HyperLogLog++ algorithm as
//! used by various Google Cloud products, such as BigQuery and BigTable. You can use
//! this crate to decode and encode the HyperLogLog++ sketches in a way that is fully
//! compatible with the implementation in BigQuery and BigTable.
//!
//! This crate strives to be a 100% compatible re-implementation of the original Java
//! library and any deviation from the behavior of the Java library is considered to
//! be a bug.
//!
//! ## Usage
//!
//! To decode an existing sketch, you can use [`HyperLogLogPlusPlus::from_bytes`].
//! To create a branch new sketch, you can use the [`HyperLogLogPlusPlusBuilder`].
//! See documentation for each of the classes for more details.

mod fingerprint2011;

mod aggregator;
mod error;
mod hll;
mod hyperloglogplusplus;
pub mod protos;
pub(crate) mod utils;

pub use aggregator::Aggregator;
pub use error::SketchError;
pub use hyperloglogplusplus::{HyperLogLogPlusPlus, HyperLogLogPlusPlusBuilder};
