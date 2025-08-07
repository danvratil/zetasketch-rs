// SPDX-FileCopyrightText: 2025 Daniel Vrátil <me@dvratil.cz>
//
// SPDX-License-Identifier: MIT
//
// Based on the original Zetasketch implementation by Google:
// https://github.com/google/zetasketch
// Published under the Apache License 2.0

use crate::{error::SketchError, protos::AggregatorStateProto};

/// A common interface for one-pass, distributed, online aggregation algorithms
/// in the Rust version of the aggregation library.
pub trait Aggregator<R, A: Aggregator<R, A>> {
    /// Merges the state from `other` aggregator to this one.
    ///
    /// In general, the supplied aggregator must be of the same type and initialized with the same
    /// parameters as this one. Implementations which are more accommodating will document this.
    ///
    /// Clients should use the most direct merge operation possible as it allows aggregators to
    /// implement performance optimizations. In general, assume that [`Aggregator::merge_bytes`]
    /// is faster than [`Aggregator::merge_proto`] or that [`Aggregator::merge_proto`] is faster
    /// than [`Aggregator::merge_aggregator`].
    fn merge_aggregator(&mut self, other: A) -> Result<(), SketchError>;

    /// Merge the state from `proto` into this one.
    ///
    /// In general, the supplied aggregator state must be of the same type and initialized with
    /// the same parameters as this one. Implementations which are more accommodating will document
    /// this.
    ///
    /// See [`Aggregator::merge_aggregator`] on details regarding merging performance.
    fn merge_proto(&mut self, proto: AggregatorStateProto) -> Result<(), SketchError>;

    /// Merges the stage from `data` into this one.
    ///
    /// In general, the supplied aggregator state must be of the same type and initialized with
    /// the same parameters as this one. Implementations which are more accommodating will document
    /// this.
    ///
    /// See [`Aggregator::merge_aggregator`] on details regarding merging performance.
    fn merge_bytes(&mut self, data: &[u8]) -> Result<(), SketchError>;

    /// Returns the total number of input values that this aggregator has seen.
    fn num_values(&self) -> u64;

    /// Returns the aggregated result of this aggregator.
    fn result(&self) -> Result<R, SketchError>;

    /// Returns the internal state of the aggregator as a serialized string.
    ///
    /// The returned value can be deserialized into an [`AggregatorStateProto`]
    /// or passed to [`Aggregator::merge_bytes`].
    ///
    /// For some aggregators, this may be faster than calling the semantically equivalent
    /// [`Aggregator::serialize_to_proto`] as it permits individual aggregators to implement
    /// performance improvements that do not use the default proto serializer.
    fn serialize_to_bytes(self) -> Result<Vec<u8>, SketchError>;

    /// Returns the internal state of the aggregator as a protocol bugger.
    ///
    /// The returned value can be passed in to [`Aggregator::merge_proto`].
    ///
    /// See [`Aggregator::serialize_to_bytes`] on details regarding serialization performance.
    fn serialize_to_proto(self) -> Result<AggregatorStateProto, SketchError>;
}
