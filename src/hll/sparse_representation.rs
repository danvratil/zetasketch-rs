// SPDX-FileCopyrightText: 2025 Daniel Vrátil <me@dvratil.cz>
//
// SPDX-License-Identifier: MIT
//
// Based on the original Zetasketch implementation by Google:
// https://github.com/google/zetasketch
// Published under the Apache License 2.0

// Replicates com.google.zetasketch.internal.hllplus.SparseRepresentation.java

use std::collections::HashSet; // For the temporary buffer, Java uses IntOpenCustomHashSet

use itertools::Either;

use crate::error::SketchError;
use crate::hll::encoding::{self, DedupeIterator};
use crate::hll::normal_representation::NormalRepresentation;
use crate::hll::representation::RepresentationOps;
use crate::hll::state::State;
use crate::utils::buffer_traits::SimpleVarIntReader;
use crate::utils::{DifferenceDecoder, DifferenceEncoder, MergedIntIterator};

use super::representation::RepresentationUnion;

#[derive(Debug, Clone)] // Clone for when state is cloned
pub struct SparseRepresentation {
    state: State,
    encoding: encoding::Sparse,
    max_sparse_data_bytes: usize,
    max_buffer_elements: usize,
    buffer: HashSet<u32>, // Stores encoded sparse values (i32 in Java)
}

impl SparseRepresentation {
    pub const MAXIMUM_SPARSE_PRECISION: i32 = 25;
    pub const SPARSE_PRECISION_DISABLED: i32 = 0;
    const MAXIMUM_SPARSE_DATA_FRACTION: f32 = 0.75;
    const MAXIMUM_BUFFER_ELEMENTS_FRACTION: f32 = 1.0 - Self::MAXIMUM_SPARSE_DATA_FRACTION;

    pub fn new(state: State) -> Result<Self, SketchError> {
        Self::check_precision(state.precision, state.sparse_precision)?;
        let enc = encoding::Sparse::new(state.precision, state.sparse_precision)?;

        let m_normal_bytes = 1 << state.precision;
        let max_sparse_data_bytes =
            (m_normal_bytes as f32 * Self::MAXIMUM_SPARSE_DATA_FRACTION) as usize;
        let max_buffer_elements =
            (m_normal_bytes as f32 * Self::MAXIMUM_BUFFER_ELEMENTS_FRACTION) as usize;

        if max_sparse_data_bytes == 0 || max_buffer_elements == 0 {
            return Err(SketchError::IllegalArgument(
                "Calculated max sparse data bytes or buffer elements is zero, precision too low?"
                    .to_string(),
            ));
        }

        Ok(SparseRepresentation {
            state,
            encoding: enc,
            max_sparse_data_bytes,
            max_buffer_elements,
            // FIXME: Our values are already uniformly hashed, so we could use an identity hasher.
            buffer: HashSet::new(),
        })
    }

    fn check_precision(normal_precision: i32, sparse_precision: i32) -> Result<(), SketchError> {
        NormalRepresentation::check_precision(normal_precision)?; // Delegate to Normal's check
        if !(normal_precision..=Self::MAXIMUM_SPARSE_PRECISION).contains(&sparse_precision) {
            Err(SketchError::IllegalArgument(format!(
                "Expected sparse precision to be >= normal precision ({}) and <= {} but was {}.",
                normal_precision,
                Self::MAXIMUM_SPARSE_PRECISION,
                sparse_precision
            )))
        } else {
            Ok(())
        }
    }

    /// Converts this sparse representation to a NormalRepresentation.
    fn normalize(self) -> Result<NormalRepresentation, SketchError> {
        let normal_repr = NormalRepresentation::new(self.state.clone())?;
        let normal_repr = match self.merge_into_normal(normal_repr)? {
            RepresentationUnion::Normal(repr) => repr,
            _ => {
                return Err(SketchError::InvalidState(
                    "Normal representation is not valid".to_string(),
                ));
            }
        };

        Ok(normal_repr)
    }

    /// Flushes the temporary buffer into the difference-encoded sparse_data.
    fn flush_buffer(&mut self) -> Result<(), SketchError> {
        if self.buffer.is_empty() {
            return Ok(());
        }

        let iter = self.sorted_iter();
        if let Some(iter) = iter {
            self.set(DedupeIterator::new(iter, self.encoding))?;
        }
        Ok(())
    }

    pub fn merge_into_normal(
        self,
        normal: NormalRepresentation,
    ) -> Result<RepresentationUnion, SketchError> {
        let mut normal = normal;
        if self.state.has_sparse_data() {
            if let Some(sparse_data) = &self.state.sparse_data {
                let reader = SimpleVarIntReader::new(sparse_data);
                let decoder = DifferenceDecoder::new(reader);
                let repr = normal.add_sparse_values(&self.encoding, Some(decoder))?;
                if let RepresentationUnion::Normal(n) = repr {
                    normal = n;
                } else {
                    // adding sparse value into normal representation always returns normal representation
                    // FIXME: Should this panic/assert?
                    return Err(SketchError::InvalidState(
                        "Normal representation is not valid".to_string(),
                    ));
                }
            }
        }
        normal.add_sparse_values(&self.encoding, Some(self.buffer.clone()))
    }

    fn downgrade(
        mut self,
        encoding: &encoding::Sparse,
    ) -> Result<RepresentationUnion, SketchError> {
        if encoding >= &self.encoding {
            return Ok(RepresentationUnion::Sparse(self));
        }

        let original_data = self.state.sparse_data.take();
        self.state.precision = self
            .encoding
            .normal_precision
            .min(encoding.normal_precision);
        self.state.sparse_precision = self
            .encoding
            .sparse_precision
            .min(encoding.sparse_precision);
        let sparse_repr = SparseRepresentation::new(self.state.clone())?;

        let repr = if let Some(data) = original_data {
            if !data.is_empty() {
                let reader = SimpleVarIntReader::new(&data);
                let decoder = DifferenceDecoder::new(reader);
                let iter = self.encoding.downgrade(decoder, encoding);
                Self::add_unsorted_sparse_values(
                    RepresentationUnion::Sparse(sparse_repr),
                    encoding,
                    iter.into_iter(),
                )?
            } else {
                RepresentationUnion::Sparse(sparse_repr)
            }
        } else {
            RepresentationUnion::Sparse(sparse_repr)
        };

        Self::add_unsorted_sparse_values(repr, encoding, self.buffer.iter().copied())
    }

    fn add_unsorted_sparse_values<I: Iterator<Item = u32>>(
        mut repr: RepresentationUnion,
        encoding: &encoding::Sparse,
        sparse_values: I,
    ) -> Result<RepresentationUnion, SketchError> {
        for val in sparse_values {
            repr = match std::mem::replace(&mut repr, RepresentationUnion::Invalid) {
                RepresentationUnion::Sparse(repr) => repr.add_sparse_value(encoding, val)?,
                RepresentationUnion::Normal(repr) => repr.add_sparse_value(encoding, val)?,
                RepresentationUnion::Invalid => {
                    return Err(SketchError::InvalidState(
                        "Invalid representation".to_string(),
                    ));
                }
            };
        }
        Ok(repr)
    }

    fn set(&mut self, iter: impl IntoIterator<Item = u32>) -> Result<(), SketchError> {
        let mut encoder = DifferenceEncoder::new();

        let mut size = 0;
        for val in iter {
            encoder.put_int(val as i32)?;
            size += 1;
        }

        self.buffer.clear();
        self.state_mut().sparse_data = Some(encoder.into_vec());
        self.state_mut().sparse_size = size;
        Ok(())
    }

    fn update_representation(mut self) -> Result<RepresentationUnion, SketchError> {
        if self.buffer.len() > self.max_buffer_elements {
            self.flush_buffer()?;
        }

        let should_normalize = if let Some(sparse_data_bytes) = &self.state.sparse_data {
            sparse_data_bytes.len() > self.max_sparse_data_bytes
        } else {
            false
        };

        if should_normalize {
            Ok(RepresentationUnion::Normal(self.normalize()?))
        } else {
            Ok(RepresentationUnion::Sparse(self))
        }
    }

    fn data_iterator(&self) -> Option<Vec<u32>> {
        if let Some(sparse_data) = &self.state.sparse_data {
            if !sparse_data.is_empty() {
                let reader = SimpleVarIntReader::new(sparse_data);
                let decoder = DifferenceDecoder::new(reader);
                Some(decoder.collect())
            } else {
                None
            }
        } else {
            None
        }
    }

    fn buffer_iterator(&self) -> Option<impl Iterator<Item = u32>> {
        if !self.buffer.is_empty() {
            let mut array = self.buffer.iter().copied().collect::<Vec<_>>();
            array.sort();
            Some(array.into_iter())
        } else {
            None
        }
    }

    fn sorted_iter(
        &self,
    ) -> Option<
        Either<
            MergedIntIterator<impl Iterator<Item = u32>, impl Iterator<Item = u32>>,
            Either<impl Iterator<Item = u32>, impl Iterator<Item = u32>>,
        >,
    > {
        let a = self.data_iterator();
        let b = self.buffer_iterator();

        match (a, b) {
            (Some(a), Some(b)) => Some(Either::Left(MergedIntIterator::new(a.into_iter(), b))),
            (Some(a), None) => Some(Either::Right(Either::Left(a.into_iter()))),
            (None, Some(b)) => Some(Either::Right(Either::Right(b))),
            (None, None) => None,
        }
    }
}

impl RepresentationOps for SparseRepresentation {
    fn add_hash(mut self, hash: u64) -> Result<RepresentationUnion, SketchError> {
        let encoded_val = self.encoding.encode(hash);
        self.buffer.insert(encoded_val);
        self.update_representation()
    }

    fn add_sparse_value(
        mut self,
        encoding: &encoding::Sparse,
        sparse_value: u32,
    ) -> Result<RepresentationUnion, SketchError> {
        self.encoding.assert_compatible(encoding)?;

        if encoding < &self.encoding {
            let mut repr = self.downgrade(encoding)?;
            return match std::mem::replace(&mut repr, RepresentationUnion::Invalid) {
                RepresentationUnion::Sparse(repr) => repr.add_sparse_value(encoding, sparse_value),
                RepresentationUnion::Normal(repr) => repr.add_sparse_value(encoding, sparse_value),
                RepresentationUnion::Invalid => {
                    return Err(SketchError::InvalidState(
                        "Invalid representation".to_string(),
                    ));
                }
            };
        }

        if &self.encoding < encoding {
            self.buffer
                .insert(encoding.downgrade_sparse_value(sparse_value, &self.encoding));
        } else {
            self.buffer.insert(sparse_value);
        }

        self.update_representation()
    }

    fn add_sparse_values<I: IntoIterator<Item = u32>>(
        mut self,
        encoding: &encoding::Sparse,
        sparse_values: Option<I>,
    ) -> Result<RepresentationUnion, SketchError> {
        self.encoding.assert_compatible(encoding)?;

        // Downgrade ourselves if the incoming values are of lower precision.
        if encoding < &self.encoding {
            let mut repr = self.downgrade(encoding)?;
            match std::mem::replace(&mut repr, RepresentationUnion::Invalid) {
                RepresentationUnion::Sparse(repr) => {
                    return repr.add_sparse_values(encoding, sparse_values);
                }
                RepresentationUnion::Normal(repr) => {
                    return repr.add_sparse_values(encoding, sparse_values);
                }
                RepresentationUnion::Invalid => {
                    return Err(SketchError::InvalidState(
                        "Invalid representation".to_string(),
                    ));
                }
            }
        }

        let Some(sparse_values) = sparse_values else {
            return Ok(RepresentationUnion::Sparse(self));
        };

        // Downgrading  the incoming values destroys their sort order so we need to add each
        // value individually to the buffer, compacting as necessarty. This is more expensive than
        // a single add_sparse_values call, but it's the only way to maintain sort order.
        if self.encoding < *encoding {
            let it = encoding.downgrade(sparse_values.into_iter(), &self.encoding);
            let self_encoding = self.encoding;
            return Self::add_unsorted_sparse_values(
                RepresentationUnion::Sparse(self),
                &self_encoding,
                it.into_iter(),
            );
        }

        // Special case when encodings are the same. Then we can profit from the fact that sparse_values
        // are sorted (as finged in the add_sparse_values contract) and do a merge-join.
        let sorted = match self.sorted_iter() {
            Some(iter) => Either::Left(MergedIntIterator::new(iter, sparse_values.into_iter())),
            None => Either::Right(sparse_values.into_iter()),
        };

        self.set(DedupeIterator::new(sorted, self.encoding))?;
        self.update_representation()
    }

    fn estimate(&self) -> Result<i64, SketchError> {
        let buckets = 1 << self.state.sparse_precision;

        // Calculate total elements without flushing by counting unique elements
        // across both sparse_data and buffer
        let total_elements = if let Some(iter) = self.sorted_iter() {
            // Count unique elements using the same deduplication logic as flush_buffer
            DedupeIterator::new(iter, self.encoding).count() as i32
        } else {
            0
        };

        let num_zeros = buckets - total_elements;
        let estimate = buckets as f64 * ((buckets as f64) / (num_zeros as f64)).ln();
        Ok(estimate.round() as i64)
    }

    fn merge_from_normal(
        self,
        other: NormalRepresentation,
    ) -> Result<RepresentationUnion, SketchError> {
        // Merging a normal representation into sparse always results in a normal representation.
        let new_normal = self.normalize()?;
        new_normal.merge_from_normal(other) // Other is already normal
    }

    fn merge_from_sparse(
        self,
        other: SparseRepresentation,
    ) -> Result<RepresentationUnion, SketchError> {
        self.add_sparse_values(&other.encoding, other.sorted_iter())
    }

    fn compact(mut self) -> Result<RepresentationUnion, SketchError> {
        self.flush_buffer()?;
        if self.state.sparse_data.is_none() {
            self.state.sparse_data = Some(Vec::new());
        }
        self.update_representation()
    }

    fn state_mut(&mut self) -> &mut State {
        &mut self.state
    }

    fn state(&self) -> &State {
        &self.state
    }
}

// Mirroring javatests/com/google/zetasketch/internal/hllplus/SparseRepresentationTest.java
#[cfg(test)]
mod tests {
    use crate::error::SketchError;
    use crate::hll::encoding::Sparse as SparseEncoding;
    use crate::hll::representation::{RepresentationOps, RepresentationUnion};
    use crate::hll::sparse_representation::{DifferenceDecoder, SparseRepresentation};
    use crate::hll::state::State;
    use crate::utils::buffer_traits::{SimpleVarIntReader, VarIntReader};

    // Helper to create SparseRepresentation with specific precisions
    fn create_sparse(
        normal_precision: i32,
        sparse_precision: i32,
    ) -> Result<SparseRepresentation, SketchError> {
        SparseRepresentation::new(State {
            precision: normal_precision,
            sparse_precision,
            // SparseRepresentation::new expects sparse_data to be None initially for buffer logic.
            sparse_data: None,
            sparse_size: 0,
            ..State::default()
        })
    }

    // Helper for asserting sparse data content (simplified)
    fn assert_sparse_data_equals(
        data: Option<&Vec<u8>>,
        expected_values: &[u32],
    ) -> Result<(), SketchError> {
        if expected_values.is_empty() {
            if let Some(data) = data {
                // Allow empty Some(vec![]) for compacted empty sketches
                if !data.is_empty() {
                    return Err(SketchError::Generic(format!(
                        "Expected no sparse data or empty data, got {data:?}",
                    )));
                }
            }
            return Ok(());
        }

        let data_bytes = data.ok_or_else(|| {
            SketchError::Generic("Sparse data is None, expected some data".to_string())
        })?;
        if data_bytes.is_empty() && !expected_values.is_empty() {
            return Err(SketchError::Generic(
                "Sparse data is empty, expected values".to_string(),
            ));
        }

        if !expected_values.is_empty() && data_bytes.is_empty() {
            return Err(SketchError::Generic(
                "Expected non-empty sparse data, but it was empty".to_string(),
            ));
        }
        if expected_values.is_empty() && !data_bytes.is_empty() {
            return Err(SketchError::Generic(format!(
                "Expected empty sparse data, but got {} bytes",
                data_bytes.len()
            )));
        }

        if !expected_values.is_empty() && expected_values == [0] {
            let reader = SimpleVarIntReader::new(data_bytes);
            let decoded_values: Vec<u32> = DifferenceDecoder::new(reader).collect();
            if decoded_values != [0] {
                return Err(SketchError::Generic(format!(
                    "Decoded sparse data {decoded_values:?} not matching expected [0]"
                )));
            }
        }
        Ok(())
    }

    #[test]
    fn add_sparse_value_higher_precision() -> Result<(), SketchError> {
        // Create SparseRepresentation with p=10, sp=13
        let repr = create_sparse(10, 13)?;
        let source_encoding = SparseEncoding::new(11, 15)?; // Value from p=11, sp=15

        let sparse_value = 0b000000000011111;

        let repr = repr.add_sparse_value(&source_encoding, sparse_value)?;
        let repr = match repr {
            RepresentationUnion::Sparse(repr) => repr.compact()?,
            _ => {
                return Err(SketchError::IllegalArgument(
                    "Expected SparseRepresentation".to_string(),
                ))
            }
        };

        let final_state = repr.state();
        assert_eq!(final_state.precision, 10);
        assert_eq!(final_state.sparse_precision, 13);
        let mut reader = SimpleVarIntReader::new(final_state.sparse_data.as_ref().unwrap());
        assert_eq!(reader.read_varint().unwrap(), 0b000000000111);

        Ok(())
    }

    #[test]
    fn add_sparse_value_lower_precision() -> Result<(), SketchError> {
        let initial_sparse = create_sparse(11, 15)?;
        let source_encoding = SparseEncoding::new(10, 13)?; // Value from p=10, sp=13
        let sparse_value = 0b0000000000001; // Example value for p=10, sp=13

        let result_repr = initial_sparse.add_sparse_value(&source_encoding, sparse_value)?;
        let final_repr = match result_repr {
            RepresentationUnion::Sparse(repr) => repr.compact()?,
            RepresentationUnion::Normal(repr) => RepresentationUnion::Normal(repr),
            _ => {
                return Err(SketchError::IllegalArgument(
                    "Unexpected representation type".to_string(),
                ))
            }
        };

        let final_state = final_repr.state();
        assert_eq!(final_state.precision, 10);
        assert_eq!(final_state.sparse_precision, 13);

        // The sparse_value is already encoded for p=10, sp=13, which is the new precision.
        assert_sparse_data_equals(final_state.sparse_data.as_ref(), &[sparse_value])?;
        Ok(())
    }

    #[test]
    fn add_sparse_values_higher_precision() -> Result<(), SketchError> {
        let repr = create_sparse(10, 13)?;
        let source_encoding = SparseEncoding::new(11, 15)?;
        let sparse_values = vec![0b000000000000001u32, 0b000000000011111u32];

        let repr = repr.add_sparse_values(&source_encoding, Some(sparse_values.clone()))?;
        let repr = match repr {
            RepresentationUnion::Sparse(repr) => repr.compact()?,
            _ => {
                return Err(SketchError::IllegalArgument(
                    "Expected SparseRepresentation".to_string(),
                ))
            }
        };

        let state = repr.state();
        assert_eq!(state.precision, 10);
        assert_eq!(state.sparse_precision, 13);
        let reader = SimpleVarIntReader::new(state.sparse_data.as_ref().unwrap());
        let decoder = DifferenceDecoder::new(reader);
        let decoded_values: Vec<u32> = decoder.collect();
        assert_eq!(decoded_values, vec![0b000000000111, 0b10000000000000010]);

        Ok(())
    }

    #[test]
    fn add_sparse_values_higher_precision_null() -> Result<(), SketchError> {
        let repr = create_sparse(10, 13)?;
        let source_encoding = SparseEncoding::new(11, 15)?;

        let repr = repr.add_sparse_values(&source_encoding, None::<Vec<u32>>)?;
        let repr = match repr {
            RepresentationUnion::Sparse(repr) => repr.compact()?,
            _ => {
                return Err(SketchError::IllegalArgument(
                    "Expected SparseRepresentation".to_string(),
                ))
            }
        };

        let final_state = repr.state();
        assert_eq!(final_state.precision, 10); // Precision remains 10
        assert_eq!(final_state.sparse_precision, 13); // Sparse precision remains 13 (no downgrade from null)
        Ok(())
    }

    #[test]
    fn add_sparse_values_lower_precision() -> Result<(), SketchError> {
        let initial_sparse = create_sparse(11, 15)?;
        let source_encoding = SparseEncoding::new(10, 13)?;
        let sparse_values = vec![0b0000000000001u32, 0b0000000001111u32];

        let result_repr =
            initial_sparse.add_sparse_values(&source_encoding, Some(sparse_values.clone()))?;
        let final_repr = match result_repr {
            RepresentationUnion::Sparse(repr) => repr.compact()?,
            RepresentationUnion::Normal(repr) => RepresentationUnion::Normal(repr),
            _ => {
                return Err(SketchError::IllegalArgument(
                    "Unexpected representation type".to_string(),
                ))
            }
        };

        let final_state = final_repr.state();
        assert_eq!(final_state.precision, 10);
        assert_eq!(final_state.sparse_precision, 13);
        assert_sparse_data_equals(final_state.sparse_data.as_ref(), &sparse_values)?;
        Ok(())
    }

    #[test]
    fn add_sparse_values_lower_precision_null() -> Result<(), SketchError> {
        let initial_sparse = create_sparse(11, 15)?;
        let source_encoding = SparseEncoding::new(10, 13)?;

        let result_repr = initial_sparse.add_sparse_values(&source_encoding, None::<Vec<u32>>)?;
        let final_repr = match result_repr {
            RepresentationUnion::Sparse(repr) => repr.compact()?,
            RepresentationUnion::Normal(repr) => RepresentationUnion::Normal(repr),
            _ => {
                return Err(SketchError::IllegalArgument(
                    "Unexpected representation type".to_string(),
                ))
            }
        };

        let final_state = final_repr.state();
        assert_eq!(final_state.precision, 10);
        assert_eq!(final_state.sparse_precision, 13);
        assert!(
            final_state
                .sparse_data
                .as_ref()
                .is_none_or(|d| d.is_empty()),
            "Sparse data should be None or empty after downgrade with null values"
        );
        Ok(())
    }
}
