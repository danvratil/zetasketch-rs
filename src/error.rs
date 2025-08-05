// SPDX-FileCopyrightText: 2025 Daniel Vrátil <me@dvratil.cz>
//
// SPDX-License-Identifier: MIT
//
// Based on the original Zetasketch implementation by Google:
// https://github.com/google/zetasketch
// Published under the Apache License 2.0

use thiserror::Error;

#[derive(Error, Debug)]
pub enum SketchError {
    #[error("Incompatible precision: {0}")]
    IncompatiblePrecision(String),
    #[error("Invalid HLL++ state: {0}")]
    InvalidState(String),
    #[error("Proto serialization error: {0}")]
    ProtoSerialization(protobuf::Error),
    #[error("Proto deserialization error: {0}")]
    ProtoDeserialization(protobuf::Error),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid argument: {0}")]
    IllegalArgument(String),
    // TODO: Add more specific error types as needed.
    #[error("Generic error: {0}")]
    Generic(String),
}
