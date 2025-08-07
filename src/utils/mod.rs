// SPDX-FileCopyrightText: 2025 Daniel Vrátil <me@dvratil.cz>
//
// SPDX-License-Identifier: MIT
//
// Based on the original Zetasketch implementation by Google:
// https://github.com/google/zetasketch
// Published under the Apache License 2.0

pub mod buffer_traits;
mod difference_iter;
mod merged_int_iter;
mod var_int;

pub use difference_iter::{DifferenceDecoder, DifferenceEncoder};
pub use merged_int_iter::MergedIntIterator;
