use super::buffer_traits::{VarIntReader, WriteBuffer, GrowingVarIntWriter};
use crate::error::SketchError;

// TODO: DifferenceEncoder and DifferenceDecoder are needed for sparseData handling.
// These will operate on Vec<u8> or &[u8].
// For now, we'll use placeholders for their iterators.

// Placeholder for DifferenceDecoder functionality
pub struct DifferenceDecoder<R: VarIntReader> {
    reader: R,
    last: u32,
}

impl<R: VarIntReader> DifferenceDecoder<R> {
    pub fn new(reader: R) -> Self {
        // In a real implementation, this would parse the start of the data.
        Self { reader, last: 0 }
    }
}

impl<R: VarIntReader> Iterator for DifferenceDecoder<R> {
    type Item = u32;
    fn next(&mut self) -> Option<Self::Item> {
        if self.reader.has_remaining() {
            match self.reader.read_varint() {
                Ok(diff) => {
                    self.last += diff as u32;
                    Some(self.last)
                }
                Err(_) => None, // Error reading, end iteration
            }
        } else {
            None
        }
    }
}

// Placeholder for DifferenceEncoder functionality
pub struct DifferenceEncoder {
    writer: GrowingVarIntWriter,
    last: i32,
}

impl DifferenceEncoder {
    pub fn new() -> Self {
        Self { 
            writer: GrowingVarIntWriter::new(),
            last: 0 
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            writer: GrowingVarIntWriter::with_capacity(capacity),
            last: 0,
        }
    }

    pub fn put_int(&mut self, val: i32) -> Result<(), SketchError> {
        assert!(val >= 0, "Only positive integers are supported");
        assert!(
            val >= self.last,
            "{} put after {} but values are required to be in increasing order",
            val,
            self.last
        );
        self.writer.write_varint(val - self.last)?;
        self.last = val;
        Ok(())
    }

    pub fn into_vec(self) -> Vec<u8> {
        self.writer.into_vec()
    }

    pub fn as_slice(&self) -> &[u8] {
        self.writer.as_slice()
    }
}

impl Default for DifferenceEncoder {
    fn default() -> Self {
        Self::new()
    }
}
