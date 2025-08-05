// SPDX-FileCopyrightText: 2025 Daniel Vrátil <me@dvratil.cz>
//
// SPDX-License-Identifier: MIT
//
// Based on the original Zetasketch implementation by Google:
// https://github.com/google/zetasketch
// Published under the Apache License 2.0


pub mod buffer_traits;
pub mod byte_slice;
pub mod difference_iter;
pub mod merged_int_iter;
pub mod var_int;

pub use buffer_traits::{
    FixedVarIntWriter, GrowableWriteBuffer, GrowingVarIntWriter, SimpleVarIntReader, VarIntReader,
    WriteBuffer,
};
pub use byte_slice::ByteSlice;
pub use difference_iter::{DifferenceDecoder, DifferenceEncoder};
pub use merged_int_iter::MergedIntIterator;
