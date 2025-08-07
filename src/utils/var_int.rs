// SPDX-FileCopyrightText: 2025 Daniel Vrátil <me@dvratil.cz>
//
// SPDX-License-Identifier: MIT
//
// Based on the original Zetasketch implementation by Google:
// https://github.com/google/zetasketch
// Published under the Apache License 2.0

pub struct VarInt;

impl VarInt {
    pub const fn var_int_size(i: i32) -> usize {
        let mut result = 0;
        let mut i = i;
        loop {
            result += 1;
            i = ((i as u32) >> 7) as i32;
            if i == 0 {
                return result;
            }
        }
    }

    /// Returns a tuple of the decoded value and the amount of bytes read.
    pub fn get_var_int(src: &[u8]) -> (i32, usize) {
        let mut result: i32 = 0;
        let mut shift = 0;
        let mut offset = 0;

        loop {
            if shift >= 32 {
                panic!("varint too long");
            }

            let b = src[offset];
            offset += 1;
            result |= ((b & 0x7F) as i32) << shift;
            shift += 7;

            if (b & 0x80) == 0 {
                break;
            }
        }

        (result, offset)
    }

    /// Returns the amount of bytes written
    pub fn set_var_int(v: i32, sink: &mut [u8]) -> usize {
        let mut offset = 0_usize;
        let mut v = v;
        loop {
            let bits = v & 0x7F;
            v = ((v as u32) >> 7) as i32;
            let b = (bits + (if v != 0 { 0x80 } else { 0 })) as u8;
            sink[offset] = b;
            offset += 1;

            if v == 0 {
                break;
            }
        }

        offset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_var_int_size() {
        assert_eq!(VarInt::var_int_size(0), 1);
        assert_eq!(VarInt::var_int_size(i32::MAX), 5);
    }

    #[test]
    fn test_get_var_int() {
        assert_eq!(VarInt::get_var_int(&[0x00]), (0, 1));
        assert_eq!(
            VarInt::get_var_int(&[0xff, 0xff, 0xff, 0xff, 0x07]),
            (i32::MAX, 5)
        );
    }
}
