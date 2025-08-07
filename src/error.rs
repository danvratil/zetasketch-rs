// SPDX-FileCopyrightText: 2025 Daniel Vrátil <me@dvratil.cz>
//
// SPDX-License-Identifier: MIT
//
// Based on the original Zetasketch implementation by Google:
// https://github.com/google/zetasketch
// Published under the Apache License 2.0

use thiserror::Error;

/// Error type for the ZetaSketch library.
#[derive(Error, Debug)]
pub enum SketchError {
    /// An operation was attempted on an aggregator with a different type than the one expected.
    #[error("Incompatible precision: {0}")]
    IncompatiblePrecision(String),
    /// An operation was attempted on an aggregator with an invalid state.
    #[error("Invalid HLL++ state: {0}")]
    InvalidState(String),
    /// An error occurred while serializing the aggregator state to a protocol buffer.
    #[error("Proto serialization error: {0}")]
    ProtoSerialization(protobuf::Error),
    /// An error occurred while deserializing the aggregator state from a protocol buffer.
    #[error("Proto deserialization error: {0}")]
    ProtoDeserialization(protobuf::Error),
    /// An error occurred while reading or writing to a file.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// An operation was attempted with an invalid argument.
    #[error("Invalid argument: {0}")]
    IllegalArgument(String),
    /// An error occurred while performing a generic operation.
    #[error("Generic error: {0}")]
    Generic(String),
}
