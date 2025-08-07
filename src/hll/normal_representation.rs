// SPDX-FileCopyrightText: 2025 Daniel Vrátil <me@dvratil.cz>
//
// SPDX-License-Identifier: MIT
//
// Based on the original Zetasketch implementation by Google:
// https://github.com/google/zetasketch
// Published under the Apache License 2.0

// Replicates com.google.zetasketch.internal.hllplus.NormalRepresentation.java

use crate::error::SketchError;
use crate::hll::data;
use crate::hll::encoding;
use crate::hll::representation::RepresentationOps;
use crate::hll::state::State;

use super::representation::RepresentationUnion;
use super::sparse_representation::SparseRepresentation;

#[derive(Debug, Clone)] // Clone for when state is cloned
pub struct NormalRepresentation {
    state: State,
    encoding: encoding::Normal,
}

impl NormalRepresentation {
    pub const MINIMUM_PRECISION: i32 = 10;
    pub const MAXIMUM_PRECISION: i32 = 24;

    pub fn new(mut state: State) -> Result<Self, SketchError> {
        Self::check_precision(state.precision)?;
        let encoding = encoding::Normal::new(state.precision)?;

        if let Some(data_bytes) = &state.data {
            if data_bytes.len() != (1 << state.precision) {
                return Err(SketchError::InvalidState(format!(
                    "Expected normal data to consist of exactly {} bytes but got {}",
                    1 << state.precision,
                    data_bytes.len()
                )));
            }
        }

        state.sparse_data = None;
        state.sparse_size = 0;
        Ok(NormalRepresentation { state, encoding })
    }

    pub fn check_precision(precision: i32) -> Result<(), SketchError> {
        if !(Self::MINIMUM_PRECISION..=Self::MAXIMUM_PRECISION).contains(&precision) {
            Err(SketchError::IllegalArgument(format!(
                "Expected normal precision to be >= {} and <= {} but was {}",
                Self::MINIMUM_PRECISION,
                Self::MAXIMUM_PRECISION,
                precision
            )))
        } else {
            Ok(())
        }
    }

    fn ensure_data_mut(state: &mut State) -> &mut Vec<u8> {
        if state.data.is_none() {
            state.data = Some(vec![0; 1 << state.precision]);
        }
        state.data.as_mut().unwrap() // Should be safe due to above check
    }

    fn ensure_data(state: &State) -> Result<&[u8], SketchError> {
        state
            .data
            .as_deref()
            .ok_or_else(|| SketchError::InvalidState("Normal data not initialized".to_string()))
    }

    fn get_writeable_data_mut(state: &mut State) -> &mut [u8] {
        if state.data.is_none() {
            state.data = Some(vec![0; 1 << state.precision]);
        }
        state.data.as_mut().unwrap().as_mut_slice() // Should be safe due to above check
    }

    fn maybe_downgrade(
        &mut self,
        encoding: &encoding::Normal,
        sparse_precision: i32,
    ) -> Result<(), SketchError> {
        if self.state.precision <= encoding.precision
            && self.state.sparse_precision <= sparse_precision
        {
            return Ok(());
        }

        if self.state.precision > encoding.precision {
            let source_data_opt = self.state.data.take(); // Take ownership
            self.state.data = Some(vec![0; 1 << encoding.precision]); // Allocate new data for target precision
            self.state.precision = encoding.precision; // Update precision first

            // Need to create a new target_encoding based on the new self.state.precision
            let new_target_encoding = encoding::Normal::new(self.state.precision)?;

            Self::merge_normal_data_maybe_downgrading(
                &mut self.state,
                &new_target_encoding,
                source_data_opt,
                &self.encoding,
            )?;
        }

        self.state.sparse_precision = self.state.sparse_precision.min(sparse_precision);
        // Update self.encoding to match new self.state.precision if it changed
        if self.encoding.precision != self.state.precision {
            self.encoding = encoding::Normal::new(self.state.precision)?;
        }
        Ok(())
    }

    fn merge_normal_data_maybe_downgrading(
        state: &mut State,
        target_encoding: &encoding::Normal,
        source_data_opt: Option<Vec<u8>>,
        source_encoding: &encoding::Normal,
    ) -> Result<(), SketchError> {
        let Some(source_data) = source_data_opt else {
            return Ok(());
        };

        if target_encoding.precision == source_encoding.precision {
            let data_slice = Self::ensure_data_mut(state);
            // Assuming data_slice.put_max translates to iterating and taking max.
            // For Vec<u8>, this means element-wise max.
            if data_slice.len() == source_data.len() {
                for i in 0..data_slice.len() {
                    data_slice[i] = data_slice[i].max(source_data[i]);
                }
            } else {
                return Err(SketchError::InvalidState("Mismatched data lengths in merge_normal_data_maybe_downgrading for same precision".to_string()));
            }
            return Ok(());
        }

        // Merging from higher precision to lower.
        // The target_array is ensured/allocated by maybe_downgrade before this call for target_encoding.precision
        let target_array = Self::get_writeable_data_mut(state);

        for (old_index, old_rho_w) in source_data.iter().enumerate() {
            let new_index = source_encoding
                .downgrade_index(old_index as u32, target_encoding.precision)
                as usize;
            let new_rho_w = source_encoding.downgrade_rho_w(
                old_index as u32,
                *old_rho_w,
                target_encoding.precision,
            );

            if new_index < target_array.len() {
                // Check bounds
                if target_array[new_index] < new_rho_w {
                    target_array[new_index] = new_rho_w;
                }
            } else {
                return Err(SketchError::InvalidState(format!(
                    "Downgraded index {} out of bounds for target array length {}",
                    new_index,
                    target_array.len()
                )));
            }
        }
        Ok(())
    }

    /// Adds a sparse value to a backing array, downgrading the values if necessary if the target
    /// encoding has a lower precision than the source.
    /// Internal helper, assumes data is already writeable and correctly sized.
    fn add_sparse_value_maybe_downgrading(
        data: &mut [u8], // Writeable data slice
        target_normal_encoding: &encoding::Normal,
        sparse_value: u32,
        source_sparse_encoding: &encoding::Sparse,
    ) -> Result<(), SketchError> {
        let idx: usize;
        let rho_w: u8;

        if target_normal_encoding.precision < source_sparse_encoding.normal_precision {
            idx = source_sparse_encoding
                .decode_and_downgrade_normal_index(sparse_value, target_normal_encoding.precision)
                as usize;
            rho_w = source_sparse_encoding
                .decode_and_downgrade_normal_rho_w(sparse_value, target_normal_encoding.precision);
        } else {
            idx = source_sparse_encoding.decode_normal_index(sparse_value) as usize;
            rho_w = source_sparse_encoding.decode_normal_rho_w(sparse_value);
        }

        if idx < data.len() {
            // Check bounds
            if data[idx] < rho_w {
                data[idx] = rho_w;
            }
        } else {
            return Err(SketchError::InvalidState(format!(
                "Decoded index {} out of bounds for data length {} (sparse_value={}, source_precision={}, target_precision={})",
                idx,
                data.len(),
                sparse_value,
                source_sparse_encoding.sparse_precision,
                target_normal_encoding.precision
            )));
        }
        Ok(())
    }
}

impl RepresentationOps for NormalRepresentation {
    fn add_hash(mut self, hash: u64) -> Result<RepresentationUnion, SketchError> {
        let idx = self.encoding.index(hash) as usize;
        let rho_w = self.encoding.rho_w(hash);

        let data_slice = Self::get_writeable_data_mut(&mut self.state);

        if idx < data_slice.len() {
            // Bounds check
            if data_slice[idx] < rho_w {
                data_slice[idx] = rho_w;
            }
        } else {
            return Err(SketchError::InvalidState(format!(
                "Index {} out of bounds for data length {}",
                idx,
                data_slice.len()
            )));
        }

        Ok(RepresentationUnion::Normal(self)) // Normal representation does not change type by adding a hash
    }

    fn add_sparse_value(
        mut self,
        source_sparse_encoding: &crate::hll::encoding::Sparse,
        sparse_value: u32,
    ) -> Result<RepresentationUnion, SketchError> {
        let state_clone = self.state.clone();
        let mut temp_self = std::mem::replace(&mut self, NormalRepresentation::new(state_clone)?);
        temp_self.maybe_downgrade(
            &source_sparse_encoding.normal(),
            source_sparse_encoding.sparse_precision,
        )?;
        self = temp_self;

        let data_slice = Self::get_writeable_data_mut(&mut self.state);
        Self::add_sparse_value_maybe_downgrading(
            data_slice,
            &self.encoding,
            sparse_value,
            source_sparse_encoding,
        )?;

        Ok(RepresentationUnion::Normal(self)) // Does not change to Sparse, but precision might have changed
    }

    fn add_sparse_values<I: IntoIterator<Item = u32>>(
        mut self,
        source_sparse_encoding: &crate::hll::encoding::Sparse,
        sparse_values: Option<I>,
    ) -> Result<RepresentationUnion, SketchError> {
        let state_clone = self.state.clone();
        let mut temp_self = std::mem::replace(&mut self, NormalRepresentation::new(state_clone)?);
        temp_self.maybe_downgrade(
            &source_sparse_encoding.normal(),
            source_sparse_encoding.sparse_precision,
        )?;
        self = temp_self;

        let Some(sparse_values) = sparse_values else {
            return Ok(RepresentationUnion::Normal(self));
        };

        let data_slice = Self::get_writeable_data_mut(&mut self.state);
        for sparse_value in sparse_values {
            Self::add_sparse_value_maybe_downgrading(
                data_slice,
                &self.encoding,
                sparse_value,
                source_sparse_encoding,
            )?;
        }
        Ok(RepresentationUnion::Normal(self))
    }

    fn estimate(&self) -> Result<i64, SketchError> {
        let data_bytes = match Self::ensure_data(&self.state) {
            Ok(d) => d,
            Err(_) => return Ok(0), // No data, estimate is 0
        };

        let mut num_zeros = 0;
        let mut sum = 0.0f64;

        for &v_byte in data_bytes {
            let v = v_byte as i32;
            if v == 0 {
                num_zeros += 1;
            }
            // sum += 2.0_f64.powi(-v);
            // Optimization from Java: sum += 1.0 / (1L << v);
            // Ensure v is within reasonable bounds for bit shift.
            assert!(
                0 <= v
                    && v <= 65 - self.state.precision
                    && self.state.precision >= Self::MINIMUM_PRECISION
            );
            sum += 1.0 / ((1u64 << v) as f64);
        }

        let m = (1 << self.state.precision) as f64;

        if num_zeros > 0 {
            let linear_count_threshold =
                data::linear_counting_threshold(self.state.precision) as f64;
            let h = m * (m / num_zeros as f64).ln();
            if h <= linear_count_threshold {
                return Ok(h.round() as i64);
            }
        }

        let raw_estimate = data::alpha(self.state.precision) * m * m / sum;
        let bias_correction = data::estimate_bias(raw_estimate, self.state.precision);

        Ok((raw_estimate - bias_correction).round() as i64)
    }

    fn merge_from_normal(
        mut self,
        mut other: NormalRepresentation,
    ) -> Result<RepresentationUnion, SketchError> {
        let state_clone = self.state.clone();
        let mut temp_self = std::mem::replace(&mut self, NormalRepresentation::new(state_clone)?);
        temp_self.maybe_downgrade(&other.encoding, other.state.sparse_precision)?;
        self = temp_self;

        Self::merge_normal_data_maybe_downgrading(
            &mut self.state,
            &self.encoding,
            other.state.data.take(),
            &other.encoding,
        )?;

        Ok(RepresentationUnion::Normal(self))
    }

    fn merge_from_sparse(
        self,
        other: SparseRepresentation,
    ) -> Result<RepresentationUnion, SketchError> {
        other.merge_into_normal(self)
    }

    fn compact(self) -> Result<RepresentationUnion, SketchError> {
        Ok(RepresentationUnion::Normal(self)) // Normal representation is already compact
    }

    fn state_mut(&mut self) -> &mut State {
        &mut self.state
    }

    fn state(&self) -> &State {
        &self.state
    }
}

#[cfg(test)]
mod tests {
    use crate::error::SketchError;
    use crate::hll::encoding::{Normal as NormalEncoding, Sparse as SparseEncoding};
    use crate::hll::normal_representation::NormalRepresentation;
    use crate::hll::representation::{RepresentationOps, RepresentationUnion}; // For add_hash, merge etc.
    use crate::hll::state::State;

    // Helper to create NormalRepresentation with specific precisions
    fn create_normal(
        normal_precision: i32,
        sparse_precision: i32,
    ) -> Result<NormalRepresentation, SketchError> {
        NormalRepresentation::new(State {
            precision: normal_precision,
            sparse_precision,
            // NormalRepresentation::new expects data to be None or correctly sized.
            // For these tests, we often start empty and let operations initialize data.
            data: None,
            ..Default::default()
        })
    }

    #[test]
    fn add_sparse_value_downgrades_sparse_precision() -> Result<(), SketchError> {
        let repr = create_normal(10, 15)?;
        let sparse_encoding = SparseEncoding::new(10, 13)?;

        // add_sparse_value is on RepresentationOps, NormalRepresentation implements it.
        // It might return a new representation if types changed, but for N->N it's None.
        // The method on NormalRepresentation itself mutates self.
        let repr = repr.add_sparse_value(&sparse_encoding, 0b0000000000001)?;

        assert_eq!(repr.state().sparse_precision, 13);
        Ok(())
    }

    #[test]
    fn add_sparse_value_higher_precision() -> Result<(), SketchError> {
        let repr = create_normal(10, 15)?;
        let source_sparse_encoding = SparseEncoding::new(11, 13)?; // p=11, sp=13

        let sparse_value = 0b0000000000001; // Encoded with p=11, sp=13 related logic
        let repr = repr.add_sparse_value(&source_sparse_encoding, sparse_value)?;

        // Verification:
        // repr (p=10) should have its data updated by downgrading sparse_value from (p=11).
        // Original Java test:
        // byte[] expected = new byte[1 << 10];
        // Encoding.Normal normalEncoding = new Encoding.Normal(10);
        // int newIndex = sparseEncoding.decodeAndDowngradeNormalIndex(sparseValue, normalEncoding);
        // byte newRhoW = sparseEncoding.decodeAndDowngradeNormalRhoW(sparseValue, normalEncoding);
        // expected[newIndex] = newRhoW;
        // assertThat(repr.state.data.toByteArray()).isEqualTo(expected);

        let mut expected_data = vec![0u8; 1 << 10];
        let target_normal_encoding = NormalEncoding::new(10)?;

        let new_index = source_sparse_encoding
            .decode_and_downgrade_normal_index(sparse_value, target_normal_encoding.precision)
            as usize;
        let new_rho_w = source_sparse_encoding
            .decode_and_downgrade_normal_rho_w(sparse_value, target_normal_encoding.precision);

        if new_index < expected_data.len() {
            expected_data[new_index] = new_rho_w;
        } else {
            panic!("Index out of bounds during test setup");
        }

        assert_eq!(
            repr.state()
                .data
                .as_deref()
                .expect("Data should be initialized"),
            expected_data.as_slice()
        );
        assert_eq!(repr.state().precision, 10);
        assert_eq!(repr.state().sparse_precision, 13); // Downgraded from 15 to match source_sparse_encoding's sp=13
        Ok(())
    }

    #[test]
    fn add_sparse_value_lower_precision() -> Result<(), SketchError> {
        let repr = create_normal(11, 15)?; // self is p=11
        let source_sparse_encoding = SparseEncoding::new(10, 13)?; // incoming is p=10

        let sparse_value = 0b0000000000001; // Encoded with p=10 related logic
        let repr = repr.add_sparse_value(&source_sparse_encoding, sparse_value)?;

        // repr should downgrade its own precision to 10.
        assert_eq!(repr.state().precision, 10);
        assert_eq!(repr.state().sparse_precision, 13);

        let mut expected_data = vec![0u8; 1 << 10];
        // The sparse_value was encoded for p=10. When added to repr (now p=10),
        // it should be decoded directly, not downgraded further.
        let normal_encoding = NormalEncoding::new(10)?; // repr's new encoding

        let new_index = source_sparse_encoding
            .decode_and_downgrade_normal_index(sparse_value, normal_encoding.precision)
            as usize;
        let new_rho_w = source_sparse_encoding
            .decode_and_downgrade_normal_rho_w(sparse_value, normal_encoding.precision);

        if new_index < expected_data.len() {
            expected_data[new_index] = new_rho_w;
        } else {
            panic!("Index out of bounds during test setup");
        }
        assert_eq!(
            repr.state()
                .data
                .as_deref()
                .expect("Data should be initialized"),
            expected_data.as_slice()
        );
        Ok(())
    }

    #[test]
    fn add_sparse_values_downgrades_sparse_precision() -> Result<(), SketchError> {
        let repr = create_normal(10, 15)?;
        let sparse_encoding = SparseEncoding::new(10, 13)?;
        let empty_iter: &mut dyn Iterator<Item = u32> = &mut std::iter::empty();

        let repr = repr.add_sparse_values(&sparse_encoding, Some(empty_iter))?;
        assert_eq!(repr.state().sparse_precision, 13);
        Ok(())
    }

    #[test]
    fn add_sparse_values_higher_precision() -> Result<(), SketchError> {
        let repr = create_normal(10, 15)?; // self p=10
        let source_sparse_encoding = SparseEncoding::new(11, 13)?; // incoming p=11

        let sparse_values = vec![0b0000000000001u32, 0b00000000011111u32];
        let iter: &mut dyn Iterator<Item = u32> = &mut sparse_values.clone().into_iter();

        let repr = repr.add_sparse_values(&source_sparse_encoding, Some(iter))?;

        let mut expected_data = vec![0u8; 1 << 10];
        let target_normal_encoding = NormalEncoding::new(10)?;
        for &sparse_value in &sparse_values {
            let new_index = source_sparse_encoding
                .decode_and_downgrade_normal_index(sparse_value, target_normal_encoding.precision)
                as usize;
            let new_rho_w = source_sparse_encoding
                .decode_and_downgrade_normal_rho_w(sparse_value, target_normal_encoding.precision);
            if new_index < expected_data.len() {
                // In HLL, we take the max if multiple values map to the same index
                expected_data[new_index] = expected_data[new_index].max(new_rho_w);
            } else {
                panic!("Index out of bounds during test setup");
            }
        }
        assert_eq!(
            repr.state()
                .data
                .as_deref()
                .expect("Data should be initialized"),
            expected_data.as_slice()
        );
        assert_eq!(repr.state().precision, 10);
        assert_eq!(repr.state().sparse_precision, 13);
        Ok(())
    }

    #[test]
    fn add_sparse_values_lower_precision() -> Result<(), SketchError> {
        let repr = create_normal(11, 15)?; // self p=11
        let source_sparse_encoding = SparseEncoding::new(10, 13)?; // incoming p=10

        let sparse_values = vec![0b0000000000001u32, 0b0000000001001u32];
        let iter: &mut dyn Iterator<Item = u32> = &mut sparse_values.clone().into_iter();

        let repr = repr.add_sparse_values(&source_sparse_encoding, Some(iter))?;

        assert_eq!(repr.state().precision, 10); // Self downgraded
        assert_eq!(repr.state().sparse_precision, 13);

        let mut expected_data = vec![0u8; 1 << 10];
        // Values are for p=10, repr is now p=10. Direct decode.
        for &sparse_value in &sparse_values {
            let new_index = source_sparse_encoding.decode_normal_index(sparse_value) as usize;
            let new_rho_w = source_sparse_encoding.decode_normal_rho_w(sparse_value);
            if new_index < expected_data.len() {
                expected_data[new_index] = expected_data[new_index].max(new_rho_w);
            } else {
                panic!("Index out of bounds during test setup");
            }
        }
        assert_eq!(
            repr.state()
                .data
                .as_deref()
                .expect("Data should be initialized"),
            expected_data.as_slice()
        );
        Ok(())
    }

    #[test]
    fn add_sparse_values_higher_precision_null() -> Result<(), SketchError> {
        let repr = create_normal(10, 15)?;
        let sparse_encoding = SparseEncoding::new(11, 13)?; // Higher normal precision

        let repr = repr.add_sparse_values(&sparse_encoding, None::<std::iter::Empty<u32>>)?; // Null iterator

        assert_eq!(repr.state().precision, 10); // Self precision remains 10
        assert_eq!(repr.state().sparse_precision, 13); // Self sparse precision downgrades to 13
        assert!(repr.state().data.is_none()); // Data should not be initialized by adding no values
        Ok(())
    }

    #[test]
    fn add_sparse_values_lower_precision_null() -> Result<(), SketchError> {
        let repr = create_normal(11, 15)?; // Self p=11
        let sparse_encoding = SparseEncoding::new(10, 13)?; // Lower normal precision

        let repr = repr.add_sparse_values(&sparse_encoding, None::<std::iter::Empty<u32>>)?; // Null iterator

        assert_eq!(repr.state().precision, 10); // Self precision downgrades to 10
        assert_eq!(repr.state().sparse_precision, 13); // Self sparse precision downgrades to 13

        // Java: assertThat(repr.state.data.toByteArray()).isEqualTo(new byte[1 << 10]);
        // This implies data gets initialized to zeros upon precision downgrade, even if no values added.
        // Our maybe_downgrade allocates new data: self.state.data = Some(vec![0; 1 << encoding.precision]);
        assert_eq!(
            repr.state()
                .data
                .as_deref()
                .expect("Data should be initialized after downgrade"),
            vec![0u8; 1 << 10].as_slice()
        );
        Ok(())
    }

    #[test]
    fn merge_downgrades_sparse_precision() -> Result<(), SketchError> {
        let a = create_normal(10, 14)?; // sparse_precision = 14
        let b = create_normal(10, 15)?; // sparse_precision = 15

        // Merging a into b. b should downgrade its sparse_precision to 14.
        // Java: b.merge(a); implies a.merge_from_normal(&b) or b.merge_from_normal(&a)
        // If it's `b.merge(a)` in Java, it is `b.merge_from_normal(&a)` in Rust trait.
        // So, b is `self`.
        // `b.merge(a)` means `a` is merged into `b`. `b` is `self`.
        // `self.merge_from_normal(&other)`
        // Here, let `b` be `self`. `b.merge_from_normal(&a)`.
        // `b` (self) has sp=15. `a` (other) has sp=14.
        // `b` should downgrade its `sparse_precision` to `min(15, 14) = 14`.

        // Let target be `b`, source be `a`.
        let target_b = b; // sp = 15
        let source_a = a; // sp = 14

        let target_b = target_b.merge_from_normal(source_a)?;
        assert_eq!(target_b.state().sparse_precision, 14);
        Ok(())
    }

    fn add_hash_to_normal(
        repr: NormalRepresentation,
        p: i32,
        index: u64,
        rho_w_val: u8,
    ) -> Result<RepresentationUnion, SketchError> {
        // rho_w_val is the actual rho_w (e.g. 1, 2, 3...).
        // The hash encoding for rhoW is rho_w - 1 if > 0, or a special pattern for 0.
        // However, HLL++ paper uses rhoW directly as number of leading zeros + 1.
        // Encoding.rhoW(hash) directly gives this value.
        // The example values like (0b001L << 51) /* rhoW = 2 + 1 = 3 */ suggest rhoW is number of leading zeros +1.
        // Let's assume rho_w_val is the final HLL register value (1-based count of leading zeros).

        // Constructing a hash that yields a specific index and rho_w is complex.
        // Encoding.index(hash) = (hash >>> (64 - P))
        // Encoding.rhoW(hash):
        //  long value = hash << P;
        //  if (value == 0) return (byte) (64 - P + 1);
        //  return (byte) (Long.numberOfLeadingZeros(value) + 1);
        //
        // Let's use a simplified hash generation for testing specific index/rho_w values.
        // index = hash >> (64-P)
        // rho_w is based on (hash << P)
        // To get specific index `idx` and rho_w `r`:
        // hash_prefix = idx << (64-P)
        // hash_suffix_for_rho_w:
        //   if r = 64-P+1: suffix_bits = 0
        //   if r < 64-P+1: need `r-1` leading zeros in the suffix (P bits long).
        //                   value = 1L << (64 - (r-1) -1) = 1L << (64-r) (this is for Long.numberOfLeadingZeros)
        //                   This value refers to the `hash << P` part.
        //                   So, (hash << P) should have `r-1` leading zeros.
        //                   The number of leading zeros of `X` is `r-1`.
        //                   `X` has `64-P` bits that matter for rho_w calculation after shifting out the index.
        //                   Actually, it's `64-P` bits from `value = hash << P`. The number of leading zeros is taken on these `64-P` bits.
        //                   No, `Long.numberOfLeadingZeros` is on the full 64-bit `value`.
        //
        // Let value_for_rho = hash << P. We want number_of_leading_zeros(value_for_rho) = rho_w_val -1.
        // (Unless value_for_rho is 0, then rho_w = 64-P+1).
        //
        // If rho_w_val = 1 (meaning 0 leading zeros for `value_for_rho`):
        //   value_for_rho starts with a 1. E.g., 1000...000 (64 bits)
        //   (hash << P) = 1L << 63
        //   hash = (1L << 63) >> P
        //
        // If rho_w_val = 2 (meaning 1 leading zero for `value_for_rho`):
        //   value_for_rho starts with 01. E.g., 0100...000
        //   (hash << P) = 1L << 62
        //   hash = (1L << 62) >> P
        //
        // So, if rho_w_val <= 64-P:
        //   (hash << P) = 1L << (64 - (rho_w_val - 1) -1) = 1L << (64 - rho_w_val)
        //   let suffix_mask = (1u64 << (64-p)) -1; // Mask for the lower 64-P bits
        //   let suffix_val_for_rho = (1u64 << ( (64-p) - (rho_w_val -1) -1) ) ;
        // This is getting too complex. For tests, it's easier to use the encoding struct if possible,
        // or find hashes that produce the desired outcome.
        // The Java tests directly construct hashes: (index_bits << X) | (rho_w_bits << Y)
        // (0b0000000000L << 54) means index is 0, using 10 bits for index (P=10). 64-10 = 54.
        // (0b001L << 51) means rhoW related bits.
        // Let's assume rhoW = 3. (0b001 pattern shifted).
        // The Java HLL hash:
        // Index: top P bits.
        // RhoRemainder: next 6 bits (for precision 10-16) - not used by basic HLL.
        // RhoW: remaining 64 - P bits. Number of leading zeros + 1.

        // To get index `idx` (P bits) and rho_w `r` from the remaining 64-P bits:
        // `hash_idx_part = idx << (64 - p)`
        // For rho_w `r`:
        //   `value_after_shifting_idx = hash & ((1u64 << (64-p)) -1)`
        //   We need `(value_after_shifting_idx << p)` to have `r-1` leading zeros.
        //   This is equivalent to `value_after_shifting_idx` having `r-1` leading zeros within its `64-p` bits.
        //   Let `k = 64 - p`. We need `r-1` leading zeros in a k-bit number.
        //   This means the first `r-1` bits are 0, and the `r`-th bit is 1.
        //   `suffix_for_rho = 1u64 << (k - (r-1) - 1) = 1u64 << (k-r)`
        //   This applies if `r <= k`.
        //   If `r = k+1` (all zeros in suffix), then `suffix_for_rho = 0`.
        //   The rho_w calculation is `Long.numberOfLeadingZeros(hash << p) + 1`.
        //   Let `val_for_rho = hash << p`.
        //   If we want `numberOfLeadingZeros(val_for_rho) == r-1`:
        //      `val_for_rho` should be `1u64 << (63 - (r-1))` = `1u64 << (64-r)`.
        //   So, `hash << p = 1u64 << (64-r)`.
        //   `hash_suffix_part = (1u64 << (64-r)) >> p`. (This is the part of hash other than index bits).
        //
        //   `final_hash = (index << (64-p)) | hash_suffix_part`
        //   Let's test this:
        //   `final_hash << p = ( (index << (64-p)) | ((1u64 << (64-r)) >> p) ) << p`
        //                  `= (index << 64) /* effectively 0 */ | (1u64 << (64-r))`
        //                  `= 1u64 << (64-r)`.
        //   `numberOfLeadingZeros(1u64 << (64-r))` is `64 - (64-r) -1 = r-1` if `64-r < 63`.
        //   This is correct for `r > 1`. If `r=1`, `numberOfLeadingZeros(1u64 << 63)` is 0.
        //
        //   This works for `1 <= rho_w_val <= 64-p`.
        //   If `rho_w_val = 64-p+1` (all zeros in suffix):
        //      `hash << p = 0`. So `hash_suffix_part = 0`.

        let hash_suffix_part = if rho_w_val <= (64 - p) as u8 {
            // e.g. p=10, max rho_w from non-zero suffix is 54.
            (1u64 << (64 - rho_w_val as i32)) >> p
        } else if rho_w_val == (64 - p + 1) as u8 {
            // all zeros in suffix
            0
        } else {
            return Err(SketchError::IllegalArgument(format!(
                "rho_w_val {rho_w_val} is too large for p={p}"
            )));
        };

        let final_hash = (index << (64 - p)) | hash_suffix_part;
        let repr = repr.add_hash(final_hash)?;
        Ok(repr)
    }

    #[test]
    fn merge_normal_with_higher_precision() -> Result<(), SketchError> {
        let p_target = 10;
        let p_source = 11;
        let sp = 15;

        let target = create_normal(p_target, sp)?;
        // Target (p=10): idx=0, rhoW=3; idx=1, rhoW=1
        let target = add_hash_to_normal(target, p_target, 0, 3)?;
        let target = match target {
            RepresentationUnion::Normal(repr) => repr,
            _ => {
                return Err(SketchError::IllegalArgument(
                    "target is not a NormalRepresentation".to_string(),
                ))
            }
        };
        let target = add_hash_to_normal(target, p_target, 1, 1)?;
        let target = match target {
            RepresentationUnion::Normal(repr) => repr,
            _ => {
                return Err(SketchError::IllegalArgument(
                    "target is not a NormalRepresentation".to_string(),
                ))
            }
        };

        let source = create_normal(p_source, sp)?;
        // Source (p=11): idx=0, rhoW=1; idx=2, rhoW=3
        // idx_source=0 (0b00000000000) -> idx_target=0 (0b0000000000)
        // idx_source=2 (0b00000000010) -> idx_target=1 (0b0000000001)
        let source = add_hash_to_normal(source, p_source, 0, 1)?; // Downgrades to idx=0, rhoW around 1 + (11-10)=2
        let source = match source {
            RepresentationUnion::Normal(repr) => repr,
            _ => {
                return Err(SketchError::IllegalArgument(
                    "source is not a NormalRepresentation".to_string(),
                ))
            }
        };

        let source = add_hash_to_normal(source, p_source, 2, 3)?; // Downgrades to idx=1, rhoW around 3 + (11-10)=4
        let source = match source {
            RepresentationUnion::Normal(repr) => repr,
            _ => {
                return Err(SketchError::IllegalArgument(
                    "source is not a NormalRepresentation".to_string(),
                ))
            }
        };

        let target = target.merge_from_normal(source)?;

        let mut expected = vec![0u8; 1 << p_target];
        expected[0] = 3u8; // preserved
        expected[1] = 4u8; // target had 3, 4 after downgrade

        assert_eq!(target.state().data.as_deref().unwrap(), expected.as_slice());
        Ok(())
    }

    #[test]
    fn merge_normal_with_lower_precision() -> Result<(), SketchError> {
        let p_source = 10;
        let p_target_orig = 11; // Target starts with higher precision
        let sp = 15;

        let source = create_normal(p_source, sp)?;
        // Source (p=10): idx=0, rhoW=3; idx=1, rhoW=1
        let source = add_hash_to_normal(source, p_source, 0, 3)?;
        let source = match source {
            RepresentationUnion::Normal(repr) => repr,
            _ => {
                return Err(SketchError::IllegalArgument(
                    "source is not a NormalRepresentation".to_string(),
                ))
            }
        };
        let source = add_hash_to_normal(source, p_source, 1, 1)?;
        let source = match source {
            RepresentationUnion::Normal(repr) => repr,
            _ => {
                return Err(SketchError::IllegalArgument(
                    "source is not a NormalRepresentation".to_string(),
                ))
            }
        };

        let target = create_normal(p_target_orig, sp)?;
        // Target (p=11): idx=0, rhoW=1; idx=2, rhoW=3
        let target = add_hash_to_normal(target, p_target_orig, 0, 1)?;
        let target = match target {
            RepresentationUnion::Normal(repr) => repr,
            _ => {
                return Err(SketchError::IllegalArgument(
                    "target is not a NormalRepresentation".to_string(),
                ))
            }
        };
        let target = add_hash_to_normal(target, p_target_orig, 2, 3)?;
        let target = match target {
            RepresentationUnion::Normal(repr) => repr,
            _ => {
                return Err(SketchError::IllegalArgument(
                    "target is not a NormalRepresentation".to_string(),
                ))
            }
        };
        // Target merges source. Target should downgrade to p_source=10.
        let target = target.merge_from_normal(source)?;

        assert_eq!(target.state().precision, p_source); // Target downgraded to 10.

        // Expected data in target (now p=10)
        // Original target values (p=11) before its downgrade:
        //   idx=0 (0b00000000000), rhoW=1  -> downgrades to idx=0 (p=10), rhoW=1+(11-10)=2
        //   idx=2 (0b00000000010), rhoW=3  -> downgrades to idx=1 (p=10), rhoW=3+(11-10)=4
        // These become the initial values for the target's new p=10 data array.
        // Downgraded target data: data[0]=2, data[1]=4

        // Source values (p=10):
        //   idx=0, rhoW=3
        //   idx=1, rhoW=1

        // Merging:
        // expected[0] = max(downgraded_target_val_at_0, source_val_at_0) = max(2, 3) = 3
        // expected[1] = max(downgraded_target_val_at_1, source_val_at_1) = max(4, 1) = 4

        let mut expected = vec![0u8; 1 << p_source];
        expected[0] = 3;
        expected[1] = 4;

        assert_eq!(target.state().data.as_deref().unwrap(), expected.as_slice());
        Ok(())
    }
}
