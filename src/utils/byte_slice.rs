// SPDX-FileCopyrightText: 2025 Daniel Vrátil <me@dvratil.cz>
//
// SPDX-License-Identifier: MIT
//
// Based on the original Zetasketch implementation by Google:
// https://github.com/google/zetasketch
// Published under the Apache License 2.0


use std::borrow::Cow;

use crate::utils::var_int::VarInt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteSliceError {
    OutOfBounds,
}

pub struct ByteSlice<'a> {
    array: Cow<'a, [u8]>,
    array_offset: usize,
    limit: usize,
    position: usize,
}

impl<'a> ByteSlice<'a> {
    pub fn with_capacity(capacity: usize) -> Self {
        let v = vec![0; capacity];
        Self {
            array: Cow::Owned(v),
            array_offset: 0,
            position: 0,
            limit: capacity,
        }
    }

    pub fn copy_on_write(array: &'a [u8]) -> Self {
        Self {
            array: Cow::Borrowed(array),
            array_offset: 0,
            position: 0,
            limit: array.len(),
        }
    }

    pub fn copy_on_write_with_offset_length(array: &'a [u8], offset: usize, length: usize) -> Self {
        Self {
            array: Cow::Borrowed(array),
            array_offset: 0,
            limit: offset + length,
            position: offset,
        }
    }

    pub fn copy_on_write_from_byte_slice(source: &'a ByteSlice<'a>) -> Self {
        Self {
            array: Cow::Borrowed(&source.array),
            array_offset: source.array_offset,
            limit: source.limit,
            position: source.position,
        }
    }

    pub fn array(&self) -> &[u8] {
        &self.array
    }

    pub fn array_offset(&self) -> usize {
        self.array_offset
    }

    pub fn byte_buffer(&self) -> &[u8] {
        &self.array[self.array_offset..self.limit]
    }

    pub fn capacity(&self) -> usize {
        let capacity = self.array.len() - self.array_offset;
        assert!(self.position <= capacity);
        assert!(self.limit <= capacity);
        capacity
    }

    pub fn clear(&mut self) {
        self.limit = self.array.len() - self.array_offset;
        self.position = 0;
    }

    pub fn flip(&mut self) {
        self.limit = self.position;
        self.position = 0;
    }

    pub fn get_next_var_int(&mut self) -> i32 {
        let (result, read) = VarInt::get_var_int(&self.array[self.position..]);
        self.position += read;
        result
    }

    pub fn get_var_int(&mut self, index: usize) -> i32 {
        let (result, _) = VarInt::get_var_int(&self.array[index..]);
        result
    }

    pub fn has_remaining(&self) -> bool {
        self.position < self.limit
    }

    pub fn is_copy_on_write(&self) -> bool {
        matches!(self.array, Cow::Borrowed(_))
    }

    pub fn limit(&self) -> usize {
        assert!(self.position <= self.limit);
        assert!(self.limit <= self.array.len() - self.array_offset);
        self.limit
    }

    pub fn set_limit(&mut self, new_limit: usize) -> Result<(), ByteSliceError> {
        if new_limit > self.capacity() {
            return Err(ByteSliceError::OutOfBounds);
        }
        self.limit = new_limit;
        if self.position > new_limit {
            self.position = new_limit;
        }
        Ok(())
    }

    pub fn position(&self) -> usize {
        assert!(self.position <= self.limit);
        assert!(self.position <= self.array.len() - self.array_offset);
        self.position
    }

    pub fn set_position(&mut self, new_position: usize) -> Result<(), ByteSliceError> {
        if new_position > self.limit {
            return Err(ByteSliceError::OutOfBounds);
        }
        self.position = new_position;
        Ok(())
    }

    pub fn put_max(&mut self, index: usize, b: u8) {
        if self.array[index] < b {
            self.array.to_mut()[index] = b;
        }
    }

    pub fn put_max_with_byte_slice(&mut self, index: usize, src: &ByteSlice) {
        let remaining = src.remaining();
        let src_offset = src.array_offset + src.position;
        let mut_arr = self.array.to_mut();
        for i in 0..remaining {
            let b = src.array[src_offset + i];
            if mut_arr[index + i] < b {
                mut_arr[index + i] = b;
            }
        }
    }

    pub fn put_next_var_int(&mut self, value: i32) -> usize {
        let read = self.put_var_int(self.position, value);
        self.position += read;
        read
    }

    /// Returns the amount of bytes written
    pub fn put_var_int(&mut self, index: usize, value: i32) -> usize {
        assert_eq!(self.array_offset, 0);
        assert!(self.array.len() >= index + VarInt::var_int_size(value));

        VarInt::set_var_int(value, &mut self.array.to_mut()[index..])
    }

    pub fn remaining(&self) -> usize {
        assert!(self.position <= self.limit);
        self.limit - self.position
    }

    pub fn to_byte_array(&self) -> &[u8] {
        &self.array[self.array_offset + self.position..self.array_offset + self.limit]
    }

    pub fn to_vec(&self) -> Vec<u8> {
        self.to_byte_array().to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocate() {
        let slice = ByteSlice::with_capacity(5);
        assert_eq!(slice.array(), &[0; 5]);
        assert!(matches!(slice.array, Cow::Owned(_)));
        assert_eq!(slice.array_offset(), 0);
        assert_eq!(slice.capacity(), 5);
        assert_eq!(slice.position(), 0);
        assert_eq!(slice.limit(), 5);
    }

    #[test]
    fn test_allocate_empty() {
        let slice = ByteSlice::with_capacity(0);
        assert_eq!(slice.array(), &[]);
        assert!(matches!(slice.array, Cow::Owned(_)));
        assert_eq!(slice.array_offset(), 0);
        assert_eq!(slice.capacity(), 0);
        assert_eq!(slice.position(), 0);
        assert_eq!(slice.limit(), 0);
    }

    #[test]
    fn test_clear() {
        let data = [1, 2, 3, 4, 5];
        let mut slice = ByteSlice::copy_on_write_with_offset_length(&data, 1, 3);

        slice.clear();
        assert!(matches!(slice.array, Cow::Borrowed(_)));
        assert_eq!(slice.array(), &data);
        assert_eq!(slice.array_offset(), 0);
        assert_eq!(slice.position(), 0);
        assert_eq!(slice.limit(), 5);
    }

    #[test]
    fn test_cow_from_array() {
        let data = [1, 2, 3, 4, 5];
        let slice = ByteSlice::copy_on_write(&data);

        assert!(matches!(slice.array, Cow::Borrowed(_)));
        assert_eq!(slice.array(), &data);
        assert_eq!(slice.array_offset(), 0);
        assert_eq!(slice.capacity(), 5);
        assert_eq!(slice.position(), 0);
        assert_eq!(slice.limit(), 5);
    }

    #[test]
    fn test_cow_from_array_with_bounds() {
        let data = [1, 2, 3, 4, 5];
        let slice = ByteSlice::copy_on_write_with_offset_length(&data, 1, 3);

        assert!(matches!(slice.array, Cow::Borrowed(_)));
        assert_eq!(slice.array(), &data);
        assert_eq!(slice.array_offset(), 0);
        assert_eq!(slice.capacity(), 5);
        assert_eq!(slice.position(), 1);
        assert_eq!(slice.limit(), 4);
    }

    #[test]
    fn test_flip() {
        let mut slice = ByteSlice::with_capacity(5);
        slice.flip();

        assert_eq!(slice.position(), 0);
        assert_eq!(slice.limit(), 0);

        slice.set_limit(5).expect("Failed to set limit");
        slice.set_position(3).expect("Failed to set position");
        slice.flip();

        assert_eq!(slice.position(), 0);
        assert_eq!(slice.limit(), 3);
    }

    #[test]
    fn test_get_var_int() {
        let mut v: [u8; 5] = [0; 5];
        VarInt::set_var_int(45678, &mut v[2..]);

        let mut slice = ByteSlice::copy_on_write(&v);
        assert_eq!(slice.get_var_int(2), 45678);
        assert_eq!(slice.position(), 0);
    }

    #[test]
    fn test_get_next_var_int() {
        let mut v =
            [0; VarInt::var_int_size(123) + VarInt::var_int_size(456) + VarInt::var_int_size(7890)];
        let mut len = VarInt::set_var_int(123, v.as_mut_slice());
        len += VarInt::set_var_int(456, &mut (&mut v)[len..]);
        VarInt::set_var_int(7890, &mut (&mut v)[len..]);

        let mut slice = ByteSlice::copy_on_write(&v);
        assert_eq!(slice.get_next_var_int(), 123);
        assert_eq!(slice.position(), VarInt::var_int_size(123));

        assert_eq!(slice.get_next_var_int(), 456);
        assert_eq!(
            slice.position(),
            VarInt::var_int_size(123) + VarInt::var_int_size(456)
        );

        assert_eq!(slice.get_next_var_int(), 7890);
        assert_eq!(
            slice.position(),
            VarInt::var_int_size(123) + VarInt::var_int_size(456) + VarInt::var_int_size(7890)
        );
        assert_eq!(slice.limit(), slice.position());
    }

    #[test]
    fn test_has_remaining() {
        let mut slice = ByteSlice::copy_on_write(&[1, 2, 3, 4, 5]);
        assert!(slice.has_remaining());

        slice.set_position(3).expect("Failed to set position");
        assert!(slice.has_remaining());

        slice.set_limit(4).expect("Failed to set limit");
        assert!(slice.has_remaining());

        slice.set_position(4).expect("Failed to set position");
        assert!(!slice.has_remaining());
    }

    #[test]
    fn test_limit() {
        let mut slice = ByteSlice::copy_on_write(&[1, 2, 3, 4, 5]);
        slice.set_limit(4).expect("Failed to set limit");
        assert_eq!(slice.limit(), 4);
        assert_eq!(slice.position(), 0);

        slice.set_position(4).expect("Failed to set position");
        slice.set_limit(3).expect("Failed to set limit");
        assert_eq!(slice.limit(), 3);
        assert_eq!(slice.position(), 3);
    }

    #[test]
    fn test_position() {
        let mut slice = ByteSlice::copy_on_write(&[1, 2, 3, 4, 5]);
        slice.set_position(4).expect("Failed to set position");
        assert_eq!(slice.position(), 4);
        assert_eq!(slice.limit(), 5);

        slice.set_position(5).expect("Failed to set position");
        assert_eq!(slice.position(), 5);
        assert_eq!(slice.limit(), 5);
    }

    #[test]
    fn test_put_max_byte() {
        let data = [1, 2];
        let mut slice = ByteSlice::copy_on_write(&data);

        slice.put_max(0, 4);

        assert_eq!(data, [1, 2]);
        assert_eq!(slice.to_byte_array(), &[4, 2]);
    }

    #[test]
    fn test_put_max_byte_slice() {
        let mut slice = ByteSlice::with_capacity(4);

        slice.put_max_with_byte_slice(0, &ByteSlice::copy_on_write(&[1, 2, 3]));
        assert_eq!(slice.to_byte_array(), &[1, 2, 3, 0]);

        slice.put_max_with_byte_slice(1, &ByteSlice::copy_on_write(&[3, 2, 1]));
        assert_eq!(slice.to_byte_array(), &[1, 3, 3, 1]);
    }

    #[test]
    fn test_put_max_byte_slice_copies_on_write() {
        let data = [1, 2];
        let mut slice = ByteSlice::copy_on_write(&data);

        slice.put_max_with_byte_slice(0, &ByteSlice::copy_on_write(&[4, 2]));
        assert_eq!(data, [1, 2]);
        assert_eq!(slice.to_byte_array(), &[4, 2]);
    }

    #[test]
    fn test_put_var_int() {
        let mut slice = ByteSlice::with_capacity(10);

        let mut expected = [0; 10];
        VarInt::set_var_int(1234, &mut expected.as_mut_slice()[1..]);

        slice.put_var_int(1, 1234);
        assert_eq!(slice.array(), expected);
    }

    #[test]
    fn test_put_var_int_copies_on_write() {
        let data = [0; 5];
        let mut slice = ByteSlice::copy_on_write(&data);

        let mut expected = [0; 5];
        VarInt::set_var_int(1234, &mut expected[1..]);

        slice.put_var_int(1, 1234);
        assert_eq!(data, [0; 5]);
        assert_eq!(slice.array(), expected);
    }

    #[test]
    fn test_put_next_var_int() {
        let mut slice = ByteSlice::with_capacity(10);
        slice.set_position(1).expect("Failed to set position");

        let mut expected = [0; 10];
        VarInt::set_var_int(1234, &mut expected[1..]);

        slice.put_next_var_int(1234);
        assert_eq!(slice.position(), 1 + VarInt::var_int_size(1234));
        assert_eq!(slice.array(), expected);
    }

    #[test]
    fn test_put_next_var_int_copies_on_write() {
        let data = [0; 5];
        let mut slice = ByteSlice::copy_on_write(&data);

        slice.put_next_var_int(1234);
        assert_eq!(data, [0; 5]);
        assert_eq!(
            VarInt::get_var_int(slice.array()),
            (1234, VarInt::var_int_size(1234))
        );
    }

    #[test]
    fn test_remaining() {
        let mut slice = ByteSlice::with_capacity(5);
        assert_eq!(slice.remaining(), 5);

        slice.set_position(1).expect("Failed to set position");
        assert_eq!(slice.remaining(), 4);

        slice.set_limit(3).expect("Failed to set limit");
        assert_eq!(slice.remaining(), 2);

        slice.set_position(3).expect("Failed to set position");
        assert_eq!(slice.remaining(), 0);
    }

    #[test]
    fn test_to_byte_array() {
        let mut slice = ByteSlice::copy_on_write(&[1, 2, 3, 4, 5]);
        assert_eq!(slice.to_byte_array(), &[1, 2, 3, 4, 5]);
        assert_eq!(slice.position(), 0);
        assert_eq!(slice.limit(), 5);

        slice.set_position(2).expect("Failed to set position");
        assert_eq!(slice.to_byte_array(), &[3, 4, 5]);

        slice.set_limit(4).expect("Failed to set limit");
        assert_eq!(slice.to_byte_array(), &[3, 4]);
    }
}
