// SPDX-FileCopyrightText: 2025 Daniel Vrátil <me@dvratil.cz>
//
// SPDX-License-Identifier: MIT
//
// Based on the original Zetasketch implementation by Google:
// https://github.com/google/zetasketch
// Published under the Apache License 2.0

use super::buffer_traits::{GrowingVarIntWriter, VarIntReader, WriteBuffer};
use crate::error::SketchError;

pub struct DifferenceDecoder<R: VarIntReader> {
    reader: R,
    last: u32,
}

impl<R: VarIntReader> DifferenceDecoder<R> {
    pub fn new(reader: R) -> Self {
        Self { reader, last: 0 }
    }
}

impl<R: VarIntReader> Iterator for DifferenceDecoder<R> {
    type Item = u32;
    fn next(&mut self) -> Option<Self::Item> {
        if self.reader.has_remaining() {
            match self.reader.read_varint() {
                Ok(diff) => {
                    self.last = self.last.wrapping_add(diff as u32);
                    Some(self.last)
                }
                Err(_) => None, // Error reading, end iteration
            }
        } else {
            None
        }
    }
}

pub struct DifferenceEncoder {
    writer: GrowingVarIntWriter,
    last: Option<i32>,
}

impl DifferenceEncoder {
    pub fn new() -> Self {
        Self {
            writer: GrowingVarIntWriter::new(),
            last: None,
        }
    }

    pub fn put_int(&mut self, val: i32) -> Result<(), SketchError> {
        if val < 0 {
            return Err(SketchError::IllegalArgument(
                "Only positive integers are supported".to_string(),
            ));
        }
        if self.last.is_some_and(|last| val < last) {
            return Err(SketchError::IllegalArgument(format!(
                "{} put after {:?} but values are required to be in increasing order",
                val, self.last
            )));
        }
        self.writer.write_varint(val - self.last.unwrap_or(0))?;
        self.last = Some(val);
        Ok(())
    }

    pub fn into_vec(self) -> Vec<u8> {
        self.writer.into_vec()
    }
}

impl Default for DifferenceEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::var_int::VarInt;

    use super::*;

    fn varints(values: &[i32]) -> Vec<u8> {
        let mut buf = Vec::new();
        for &val in values {
            let size = VarInt::var_int_size(val);
            let pos = buf.len();
            buf.resize(buf.len() + size, 0);
            VarInt::set_var_int(val, &mut buf[pos..]);
        }
        buf
    }

    mod difference_decoder {
        use crate::utils::buffer_traits::SimpleVarIntReader;

        use super::*;

        #[test]
        fn has_next_returns_false_when_empty() {
            let mut iter = DifferenceDecoder::new(SimpleVarIntReader::new(&[]));
            assert!(iter.next().is_none());
        }

        #[test]
        fn next_decodes_integers() {
            let values = varints(&[42, 170 - 42, 2903 - 170, 20160531 - 2903]);
            let mut iter = DifferenceDecoder::new(SimpleVarIntReader::new(&values));
            assert_eq!(iter.next(), Some(42));
            assert_eq!(iter.next(), Some(170));
            assert_eq!(iter.next(), Some(2903));
            assert_eq!(iter.next(), Some(20160531));
            assert_eq!(iter.next(), None);
        }

        #[test]
        fn next_throws_when_empty() {
            let mut iter = DifferenceDecoder::new(SimpleVarIntReader::new(&[]));
            assert!(iter.next().is_none());
        }
    }

    mod difference_encoder {
        use super::*;

        #[test]
        fn put_int_correctly_writes_equal_elements() {
            let mut encoder = DifferenceEncoder::new();
            encoder.put_int(42).unwrap();
            encoder.put_int(42).unwrap();

            assert_eq!(encoder.into_vec(), varints(&[42, 42 - 42]));
        }

        #[test]
        fn put_int_correctly_writes_multiple() {
            let mut encoder = DifferenceEncoder::new();
            encoder.put_int(42).unwrap();
            encoder.put_int(170).unwrap();
            encoder.put_int(2903).unwrap();

            assert_eq!(encoder.into_vec(), varints(&[42, 170 - 42, 2903 - 170]));
        }

        #[test]
        fn put_int_correctly_writes_single() {
            let mut encoder = DifferenceEncoder::new();
            encoder.put_int(42).unwrap();

            assert_eq!(encoder.into_vec(), varints(&[42]));
        }

        #[test]
        fn put_int_correctly_writes_zero() {
            let mut encoder = DifferenceEncoder::new();
            encoder.put_int(0).unwrap();

            assert_eq!(encoder.into_vec(), varints(&[0]));
        }

        #[test]
        fn put_int_throws_when_negative() {
            let mut encoder = DifferenceEncoder::new();
            assert!(matches!(
                encoder.put_int(-1),
                Err(SketchError::IllegalArgument(_))
            ));
        }

        #[test]
        fn put_int_throws_when_not_sorted() {
            let mut encoder = DifferenceEncoder::new();
            encoder.put_int(42).unwrap();
            assert!(matches!(
                encoder.put_int(12),
                Err(SketchError::IllegalArgument(_))
            ));
        }
    }
}
