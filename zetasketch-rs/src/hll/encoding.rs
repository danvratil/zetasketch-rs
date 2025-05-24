// Replicates com.google.zetasketch.internal.hllplus.Encoding.java

use std::cmp::Ordering;

use crate::error::SketchError;

/// Computes the number of leading zeros + 1 in the lower `bits` of `value`.
/// `value` here is the part of the hash *after* the index bits have been shifted out,
/// and `bits` is the number of bits in this remaining part (e.g., 64 - precision).
pub fn compute_rho_w(value_suffix: u64, bits: i32) -> u8 {
    if bits == 0 {
        // Should not happen if precision < 64
        return 1;
    }
    // Mask to keep only the relevant `bits` from the LSB side.
    // If bits is, say, 59 (for precision 5), we want to look at those 59 bits.
    // Java: long w = value << (64 - bits);
    // This means we are interested in the leading zeros of `value_suffix` *within its lowest `bits`*.
    // Example: precision=5, bits = 59. Hash suffix (lowest 59 bits) is `0010...`.
    // We need to count leading zeros in this 59-bit segment.
    // If `value_suffix` already has its higher bits zeroed out (i.e., it IS the suffix),
    // then `value_suffix.leading_zeros()` counts from MSB of u64.
    // We need to adjust for the fact that we're looking at a segment of `bits` length.

    // If the relevant part `w` is all zeros, rhoW is `bits + 1`.
    // The Java code `long w = value << (64 - bits);` effectively takes the `bits` LSBs of original hash
    // (after index part) and shifts them to become MSBs of `w`.
    // Then `Long.numberOfLeadingZeros(w)` is called.

    if value_suffix == 0 {
        // All relevant bits are zero
        return (bits + 1) as u8;
    }

    // Consider only the lowest `bits` of `value_suffix`.
    // We can find leading zeros of `value_suffix` and subtract (64 - bits) to get leading zeros within the window.
    let total_leading_zeros = value_suffix.leading_zeros() as i32;
    let leading_zeros_in_window = total_leading_zeros - (64 - bits);

    (leading_zeros_in_window + 1) as u8
}

/// Computes a downgraded rho_w (or rho_w') in a target precision
/// from an existing index and rho_w in a higher precision.
fn downgrade_rho_w(index: u32, rho_w: u8, source_p: i32, target_p: i32) -> u8 {
    if source_p == target_p {
        return rho_w;
    }
    assert!(target_p < source_p);

    let suffix = index << (32 - source_p + target_p);
    if suffix == 0 {
        rho_w + source_p as u8 - target_p as u8
    } else {
        (1 + suffix.leading_zeros()) as u8
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Normal {
    pub precision: i32,
}

impl Normal {
    pub fn new(precision: i32) -> Result<Self, SketchError> {
        if !(1..=63).contains(&precision) {
            return Err(SketchError::IllegalArgument(format!(
                "Normal precision must be between 1 and 63, got {}",
                precision
            )));
        }
        Ok(Self { precision })
    }

    /// Extracts the HLL++ index from the hash.
    pub fn index(&self, hash: u64) -> u32 {
        (hash >> (64 - self.precision)) as u32
    }

    /// Computes rho_w (run of zeros + 1) for the hash.
    pub fn rho_w(&self, hash: u64) -> u8 {
        // The part of the hash after the index bits.
        let num_suffix_bits = 64 - self.precision;
        let suffix_mask = if num_suffix_bits >= 64 {
            !0
        } else {
            (1u64 << num_suffix_bits) - 1
        };
        let value_suffix = hash & suffix_mask;
        compute_rho_w(value_suffix, num_suffix_bits)
    }

    /// Downgrades an index for the given target encoding.
    pub fn downgrade_index(&self, index: u32, target_precision: i32) -> u32 {
        assert!(target_precision <= self.precision);
        index >> (self.precision - target_precision)
    }

    /// Downgrades a rho_w for the given target encoding.
    pub fn downgrade_rho_w(&self, index: u32, rho_w: u8, target_precision: i32) -> u8 {
        if rho_w == 0 {
            // Unset register
            return 0;
        }
        downgrade_rho_w(index, rho_w, self.precision, target_precision)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Sparse {
    pub normal_precision: i32,
    pub sparse_precision: i32,
    rho_encoded_flag: u32,  // Precomputed flag for efficiency
    normal_encoder: Normal, // For delegation
}

impl PartialEq for Sparse {
    fn eq(&self, other: &Self) -> bool {
        return self.normal_precision == other.normal_precision
            && self.sparse_precision == other.sparse_precision;
    }
}

impl PartialOrd for Sparse {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.normal_precision < other.normal_precision
            || self.sparse_precision < other.sparse_precision {
                return Some(Ordering::Less);
        }

        return None
    }
}

const RHOW_BITS: i32 = 6;
const RHOW_MASK: u32 = (1 << RHOW_BITS) - 1;

impl Sparse {
    pub fn new(normal_precision: i32, sparse_precision: i32) -> Result<Self, SketchError> {
        if !((1..=24).contains(&normal_precision)) {
            return Err(SketchError::IllegalArgument(format!(
                "Sparse mode: normal precision must be 1-24, got {}",
                normal_precision
            )));
        }
        if !((1..=30).contains(&sparse_precision)) {
            return Err(SketchError::IllegalArgument(format!(
                "Sparse mode: sparse precision must be 1-30, got {}",
                sparse_precision
            )));
        }
        if sparse_precision < normal_precision {
            return Err(SketchError::IllegalArgument(format!(
                "Sparse precision ({}) must be >= normal precision ({})",
                sparse_precision, normal_precision
            )));
        }

        let rho_encoded_flag = 1u32 << (sparse_precision.max(normal_precision + RHOW_BITS));
        let normal_encoder = Normal::new(normal_precision)?;

        Ok(Self {
            normal_precision,
            sparse_precision,
            rho_encoded_flag,
            normal_encoder,
        })
    }

    // FIXME: Redo to return Result<(), SketchError> with IncompativeEncoding error
    pub fn assert_compatible(&self, other: &Self) {
        if ((self.normal_precision <= other.normal_precision)
            && (self.sparse_precision <= other.sparse_precision))
            || ((self.normal_precision >= other.normal_precision)
                && (self.sparse_precision >= other.sparse_precision)) {
            return;
        }
        assert!(false, "Precisions (p={}, sp={}) and (p={}, sp={}) are not compatible",
            self.normal_precision, self.sparse_precision,
            other.normal_precision, other.sparse_precision
        );
    }

    #[cfg(test)]
    pub fn is_less_than(&self, other: &Self) -> bool {
        return self.normal_precision < other.normal_precision
            || self.sparse_precision < other.sparse_precision;
    }

    pub fn encode(&self, hash: u64) -> u32 {
        let sparse_index = (hash >> (64 - self.sparse_precision)) as u32;

        let num_sparse_suffix_bits = 64 - self.sparse_precision;
        let sparse_suffix_mask = if num_sparse_suffix_bits >= 64 {
            !0
        } else {
            (1u64 << num_sparse_suffix_bits) - 1
        };
        let sparse_value_suffix = hash & sparse_suffix_mask;
        let sparse_rho_w = compute_rho_w(sparse_value_suffix, num_sparse_suffix_bits);

        self.encode_parts(sparse_index, sparse_rho_w)
    }

    pub fn encode_parts(&self, sparse_index: u32, sparse_rho_w: u8) -> u32 {
        assert!(sparse_index < (1u32 << self.sparse_precision));
        assert!(sparse_rho_w < (1u8 << RHOW_BITS)); // rho_w' fits in RHOW_BITS

        let diff_precision = self.sparse_precision - self.normal_precision;
        let reconstruction_mask = (1u32 << diff_precision) - 1;

        if (sparse_index & reconstruction_mask) != 0 {
            // Normal rhoW can be reconstructed, store only sparse_index
            sparse_index
        } else {
            // Normal rhoW cannot be reconstructed, store flag | normal_index | sparse_rho_w
            let normal_index = sparse_index >> diff_precision;
            self.rho_encoded_flag | (normal_index << RHOW_BITS) | (sparse_rho_w as u32)
        }
    }

    pub fn decode_sparse_index(&self, sparse_value: u32) -> u32 {
        if (sparse_value & self.rho_encoded_flag) == 0 {
            sparse_value // Not rho-encoded, value is the sparse index
        } else {
            // Rho-encoded: flag | normal_idx | rho_w'
            // normal_idx needs to be shifted left to become sparse_idx
            ((sparse_value ^ self.rho_encoded_flag) >> RHOW_BITS)
                << (self.sparse_precision - self.normal_precision)
        }
    }

    pub fn decode_sparse_rho_w_if_present(&self, sparse_value: u32) -> u8 {
        if (sparse_value & self.rho_encoded_flag) == 0 {
            0 // Not present, normal_rho_w will be computed from sparse_index
        } else {
            (sparse_value & RHOW_MASK) as u8
        }
    }

    pub fn decode_normal_index(&self, sparse_value: u32) -> u32 {
        if (sparse_value & self.rho_encoded_flag) == 0 {
            // Not rho-encoded, normal index is high bits of sparse_value (sparse_index)
            sparse_value >> (self.sparse_precision - self.normal_precision)
        } else {
            // Rho-encoded: flag | normal_idx | rho_w'
            (sparse_value ^ self.rho_encoded_flag) >> RHOW_BITS
        }
    }

    pub fn decode_normal_rho_w(&self, sparse_value: u32) -> u8 {
        if (sparse_value & self.rho_encoded_flag) == 0 {
            // Not rho-encoded. Normal rhoW is computed from last (sp-p) bits of sparse_index.
            let num_relevant_bits = self.sparse_precision - self.normal_precision;
            let suffix_mask = (1u32 << num_relevant_bits) - 1;
            let relevant_suffix_of_sparse_index = sparse_value & suffix_mask;

            compute_rho_w(relevant_suffix_of_sparse_index as u64, num_relevant_bits)
        } else {
            // Rho-encoded. Normal rhoW is sparse_rho_w' + (sp - p)
            ((sparse_value & RHOW_MASK) as i32 + (self.sparse_precision - self.normal_precision))
                as u8
        }
    }

    pub fn decode_and_downgrade_normal_index(
        &self,
        sparse_value: u32,
        target_normal_precision: i32,
    ) -> u32 {
        let normal_idx = self.decode_normal_index(sparse_value);
        normal_idx >> (self.normal_precision - target_normal_precision)
    }

    pub fn decode_and_downgrade_normal_rho_w(
        &self,
        sparse_value: u32,
        target_normal_precision: i32,
    ) -> u8 {
        let normal_idx = self.decode_normal_index(sparse_value);
        let normal_rho_w = self.decode_normal_rho_w(sparse_value);
        downgrade_rho_w(
            normal_idx,
            normal_rho_w,
            self.normal_precision,
            target_normal_precision,
        )
    }

    pub fn downgrade_sparse_value(&self, sparse_value: u32, target: &Sparse) -> u32 {
        let old_sparse_index = self.decode_sparse_index(sparse_value);
        let old_sparse_rho_w = self.decode_sparse_rho_w_if_present(sparse_value);

        let new_sparse_index = old_sparse_index >> (self.sparse_precision - target.sparse_precision);
        let new_sparse_rho_w = downgrade_rho_w(
            old_sparse_index,
            old_sparse_rho_w,
            self.sparse_precision,
            target.sparse_precision,
        );

       target.encode_parts(new_sparse_index, new_sparse_rho_w)
    }

    pub fn normal(&self) -> Normal {
        return self.normal_encoder;
    }

    /// Takes a sorted vector of sparse values and returns a vector with deduplicated indices,
    /// returning only the one with the largest rho(w'). For example, a list of sparse
    /// values with p=4 and sp=7 such as:
    ///
    /// ```text
    /// 0 000 0010100
    /// 0 000 1010100
    /// 0 000 1010101
    /// 1 1010 001100
    /// 1 1010 010000
    /// 1 1110 000000
    /// ```
    ///
    /// Will be deduplicated to:
    ///
    /// ```text
    /// 0 000 0010100
    /// 0 000 1010100
    /// 0 000 1010101
    /// 1 1010 010000
    /// 1 1110 000000
    /// ```
    pub fn dedupe(&self, sorted_values: Vec<u32>) -> Vec<u32> {
        let mut result = Vec::new();
        let mut iter = sorted_values.into_iter().peekable();

        while let Some(mut current) = iter.next() {
            // Value is not rho-encoded so we don't need to do any special decoding as the value is
            // the sparse index. Simply skip exact duplicates.
            if (current & self.rho_encoded_flag) == 0 {
                result.push(current);
                while let Some(&next) = iter.peek() {
                    if next != current {
                        break;
                    }
                    iter.next();
                }
                continue;
            }

            // Keep consuming values until we encounter one with a different index or run out of
            // values. We return the largest value (which will be the one with the largest rhoW).
            let sparse_index = self.decode_sparse_index(current);
            let mut max_sparse_value = current;

            while let Some(&next) = iter.peek() {
                if sparse_index != self.decode_sparse_index(next) {
                    break;
                }
                current = iter.next().expect("peek() returned Some but next() returned None");
                max_sparse_value = current;
            }

            result.push(max_sparse_value);
        }

        result
    }

    pub fn downgrade<I: Iterator<Item = u32>>(&self, iter: I, target: &Sparse) -> Vec<u32> {
        iter.map(|val| self.downgrade_sparse_value(val, target)).collect()
    }

    // TODO: `downgrade(sparse_value, target_sparse_encoding)` to re-encode for different sparse precision
}

#[cfg(test)]
mod tests {
    use super::*;
    mod normal {
        use super::*;

        #[test]
        fn test_index() {
            let encoding = Normal::new(5).unwrap();
            assert_eq!(encoding.index(0b101110001 << 55), 0b10111);
        }

        #[test]
        fn test_rho_w() {
            let encoding = Normal::new(5).unwrap();
            // Number of leading zero bits after the index (= the first 5 bits) is 3, rhoW is 3 + 1 = 4.
            assert_eq!(4, encoding.rho_w(0b101110001 << 55));
        }

        #[test]
        fn test_downgrade_index() {
            let source = Normal::new(5).unwrap();
            let target = Normal::new(3).unwrap();
            assert_eq!(source.downgrade_index(0b10111, target.precision), 0b101);
        }

        #[test]
        fn test_downgrade_rho_w_none() {
            let source = Normal::new(5).unwrap();
            let target = Normal::new(3).unwrap();
            // 0 indicates no value, should be kept as 0.
            assert_eq!(source.downgrade_rho_w(0b10001, 0, target.precision), 0);
        }

        #[test]
        fn test_downgrade_rho_w_non_zero() {
            let source = Normal::new(5).unwrap();
            let target = Normal::new(3).unwrap();
            // Number of leading zero bits after the new index (= the first 3 bits) is 1, rhoW is 1 + 1 = 2.
            assert_eq!(source.downgrade_rho_w(0b10001, 4, target.precision), 2);
        }

        #[test]
        fn test_downgrade_rho_w_zero() {
            let source = Normal::new(5).unwrap();
            let target = Normal::new(3).unwrap();
            // Number of leading zero bits after the new index is known (since all zeros) so new rhoW is
            // the old rhoW + 5 - 3 = 6.
            assert_eq!(source.downgrade_rho_w(0b10000, 4, target.precision), 6);
        }
    }

    mod sparse {
        use super::*;

        #[test]
        fn test_assert_compatible_matching_precisions() {
            let a = Sparse::new(6, 11).unwrap();
            let b = Sparse::new(6, 11).unwrap();
            a.assert_compatible(&b);
        }

        #[test]
        fn test_assert_compatible_downgrade_normal_precision() {
            let a = Sparse::new(6, 11).unwrap();
            let b = Sparse::new(7, 11).unwrap();
            a.assert_compatible(&b);
            b.assert_compatible(&a);
        }

        #[test]
        fn test_assert_compatible_downgrade_sparse_precision() {
            let a = Sparse::new(6, 11).unwrap();
            let b = Sparse::new(6, 12).unwrap();
            a.assert_compatible(&b);
            b.assert_compatible(&a);
        }

        #[test]
        #[should_panic]
        fn test_assert_compatible_incompatible_downgrade_a_to_b() {
            let a = Sparse::new(7, 11).unwrap();
            let b = Sparse::new(6, 12).unwrap();

            a.assert_compatible(&b);
        }

        #[test]
        #[should_panic]
        fn test_assert_compatible_incompatible_downgrade_b_to_a() {
            let a = Sparse::new(6, 11).unwrap();
            let b = Sparse::new(7, 10).unwrap();

            b.assert_compatible(&a);
        }

        #[test]
        fn test_decode_normal_index_when_not_rho_encoded() {
            let encoding = Sparse::new(4, 7).unwrap();
            // No leading flag, so normal index is just the highest 4 bits
            assert_eq!(encoding.decode_normal_index(0b1010100), 0b1010);
        }

        #[test]
        fn test_decode_normal_index_when_rho_encoded() {
            let encoding = Sparse::new(4, 7).unwrap();
            // Leading flag, next 4 bits are the normal index
            assert_eq!(encoding.decode_normal_index(0b11010001100), 0b1010);
        }

        #[test]
        fn test_decode_and_downgrade_normal_index_when_not_rho_encoded() {
            let source = Sparse::new(4, 7).unwrap();
            let target = Normal::new(3).unwrap();
            // No leading flag, so normal index is just the highest 3 bits
            assert_eq!(
                source.decode_and_downgrade_normal_index(0b1010100, target.precision),
                0b101
            );
        }

        #[test]
        fn test_decode_and_downgrade_normal_index_when_rho_encoded() {
            let source = Sparse::new(4, 7).unwrap();
            let target = Normal::new(3).unwrap();
            // Leading flag, next 3 bits are the normal index
            assert_eq!(
                source.decode_and_downgrade_normal_index(0b11010001100, target.precision),
                0b101
            );
        }

        #[test]
        fn test_decode_normal_rho_w_when_not_rho_encoded() {
            let encoding = Sparse::new(4, 7).unwrap();
            // No leading flag, normal rhoW determined by the last sp-p = 3 bits
            assert_eq!(1, encoding.decode_normal_rho_w(0b1010100));
        }

        #[test]
        fn test_decode_normal_rho_w_when_rho_encoded() {
            let encoding = Sparse::new(4, 7).unwrap();
            // Leading flag, normal rhoW' is the value of the last 6 bits + sp-p (3)
            assert_eq!(0b1100 + 3, encoding.decode_normal_rho_w(0b11010001100));
        }

        #[test]
        fn test_decode_and_downgrade_normal_rho_w_when_not_rho_encoded() {
            let source = Sparse::new(4, 7).unwrap();
            let target = Normal::new(3).unwrap();
            // No leading flag, normal rhoW determined by the last sp-p' = 2 bits
            assert_eq!(
                2,
                source.decode_and_downgrade_normal_rho_w(0b1010100, target.precision)
            );
        }

        #[test]
        fn test_decode_and_downgrade_normal_rho_w_when_rho_encoded() {
            let source = Sparse::new(4, 7).unwrap();
            let target = Normal::new(3).unwrap();
            // Leading flag, normal rhoW' is the value of the last 6 bits + sp-p' (3)
            assert_eq!(
                0b1100 + 4,
                source.decode_and_downgrade_normal_rho_w(0b11010001100, target.precision)
            );
        }

        #[test]
        fn test_decode_sparse_index_when_not_rho_encoded() {
            let encoding = Sparse::new(4, 7).unwrap();
            assert_eq!(encoding.decode_sparse_index(0b1010100), 0b1010100);
        }

        #[test]
        fn test_decode_sparse_index_when_rho_encoded() {
            let encoding = Sparse::new(4, 7).unwrap();
            // Leading flag, sparse index is the next 4 bits (normal index) plus 3 zeros
            assert_eq!(encoding.decode_sparse_index(0b11010001100), 0b1010000);
        }

        #[test]
        fn test_decode_sparse_rho_w_if_present_when_not_rho_encoded() {
            let encoding = Sparse::new(4, 7).unwrap();
            // No leading flag, sparse rhoW' is unknown
            assert_eq!(0, encoding.decode_sparse_rho_w_if_present(0b1010100));
        }

        #[test]
        fn test_decode_sparse_rho_w_if_present_when_rho_encoded() {
            let encoding = Sparse::new(4, 7).unwrap();
            // Leading flag, sparse rhoW' is the last sp-p = 6 bits
            assert_eq!(
                0b1100,
                encoding.decode_sparse_rho_w_if_present(0b11010001100)
            );
        }

        #[test]
        fn test_dedupe() {
            let input = vec![
                0b00000010100,
                0b00001010100,
                0b00001010101,
                0b11010001100,
                0b11010010000,
                0b11110000000,
            ];

            let encoding = Sparse::new(4, 7).unwrap();
            assert_eq!(
                encoding.dedupe(input),
                vec![
                    0b00000010100,
                    0b00001010100,
                    0b00001010101,
                    0b11010010000,
                    0b11110000000
                ]
            );
        }

        #[test]
        fn test_dedupe_exact_duplicates() {
            let input = vec![
                0b00000010100,
                0b00000010100,
                0b00000010100,
                0b11010001100,
                0b11010001100,
                0b11010001100,
            ];
            let encoding = Sparse::new(4, 7).unwrap();
            assert_eq!(encoding.dedupe(input), vec![0b00000010100, 0b11010001100,]);
        }

        #[test]
        fn test_encode_without_rho_w() {
            let encoding = Sparse::new(4, 7).unwrap();
            assert_eq!(encoding.encode(0b101100101 << 55), 0b1011001);
        }

        #[test]
        fn test_encode_without_rho_w_at_maximum_sparse_precision() {
            let encoding = Sparse::new(4, 30).unwrap();
            assert_eq!(encoding.encode(0b101100101 << 55), 0b101100101 << 21);
        }

        #[test]
        fn test_encode_with_rho_w_at_maximum_normal_precision() {
            let encoding = Sparse::new(24, 26).unwrap();
            assert_eq!(encoding.encode(0b101 << 61), 1 << 30 | (0b101 << 27) | 39);
        }

        #[test]
        fn test_encode_with_rho_w_at_minimum_normal_precision() {
            let encoding = Sparse::new(1, 5).unwrap();
            assert_eq!(encoding.encode(0b1 << 63), (1 << 7) | (0b1 << 6) | 60);
        }

        #[test]
        fn test_encode_with_rho_w_when_length_dominated_by_normal_index() {
            let encoding = Sparse::new(4, 7).unwrap();
            assert_eq!(
                encoding.encode(0b101100001 << 55),
                (1 << 10) | (0b1011 << 6) | 2
            );
        }

        #[test]
        fn test_encode_with_rho_w_when_length_dominated_by_sparse_index() {
            let encoding = Sparse::new(2, 9).unwrap();
            assert_eq!(
                encoding.encode(0b110000000001 << 52),
                (1 << 9) | (0b11 << 6) | 3
            );
        }

        #[test]
        fn test_encode_with_rho_w_when_sparse_precision_is_equal_to_normal_precision() {
            let encoding = Sparse::new(3, 3).unwrap();
            assert_eq!(encoding.encode(0b10111 << 59), (1 << 9) | (0b101 << 6) | 1);
        }

        #[test]
        fn test_downgrade_rho_w_to_non_rho_w() {
            let source = Sparse::new(3, 5).unwrap();
            let target = Sparse::new(2, 5).unwrap();

            assert_eq!(
                source.downgrade_sparse_value(
                    (1 << 9) /* flag */ | (0b111 << 6) /* normal index */ | 2, /* rhoW' */
                    &target
                ),
                0b11100
            );
        }

        #[test]
        fn test_downgrade_non_rho_w_to_rho_w() {
            let source = Sparse::new(3, 5).unwrap();
            let target = Sparse::new(3, 4).unwrap();

            assert_eq!(
                source.downgrade_sparse_value(0b11101, &target),
                (1 << 9) | (0b111 << 6) | 1
            );
        }

        #[test]
        fn test_downgrade_iterator() {
            let source = Sparse::new(11, 15).unwrap();
            let target = Sparse::new(10, 13).unwrap();

            let iter = vec![0b000000000000001, 0b000000000011111];

            // Preserves ordering of input values rather than sorting values.
            assert_eq!(
                source.downgrade(iter.into_iter(), &target),
                vec![0b10000000000000010, 0b000000000111]
            );
        }

        #[test]
        fn test_normal() {
            let sparse = Sparse::new(3, 5).unwrap();
            assert_eq!(sparse.normal().precision, 3);
        }

        #[test]
        fn test_is_less_than() {
            assert!(Sparse::new(3, 5).unwrap().is_less_than(&Sparse::new(4, 5).unwrap()));
            assert!(Sparse::new(4, 5).unwrap().is_less_than(&Sparse::new(4, 6).unwrap()));
            assert!(Sparse::new(3, 5).unwrap().is_less_than(&Sparse::new(4, 6).unwrap()));
            assert!(!Sparse::new(3, 5).unwrap().is_less_than(&Sparse::new(3, 5).unwrap()));

            // Currently doesn't verify that precisions are incompatible.
            assert!(Sparse::new(2, 6).unwrap().is_less_than(&Sparse::new(3, 5).unwrap()));
        }

        #[test]
        fn test_equals_to() {
            assert_eq!(Sparse::new(4, 5).unwrap(), Sparse::new(4, 5).unwrap());

            assert_ne!(Sparse::new(4, 5).unwrap(), Sparse::new(3, 5).unwrap());
            assert_ne!(Sparse::new(4, 5).unwrap(), Sparse::new(4, 6).unwrap());
            assert_ne!(Sparse::new(4, 5).unwrap(), Sparse::new(4, 6).unwrap());
            assert_ne!(Sparse::new(4, 5).unwrap(), Sparse::new(4, 6).unwrap());
        }
    }
}
