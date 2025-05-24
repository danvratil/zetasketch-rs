use crate::error::SketchError;
use crate::utils::var_int::VarInt;

/// Trait for reading VarInts from a buffer with position tracking
pub trait VarIntReader {
    /// Read the next VarInt from the buffer, advancing the position
    fn read_varint(&mut self) -> Result<i32, SketchError>;
    
    /// Check if there are more bytes to read
    fn has_remaining(&self) -> bool;
    
    /// Get the number of remaining bytes
    fn remaining(&self) -> usize;
}

/// Trait for writing data to a buffer
pub trait WriteBuffer {
    /// Write a VarInt to the buffer
    fn write_varint(&mut self, value: i32) -> Result<(), SketchError>;
    
    /// Write the maximum of the current value and the new value at the given index
    fn write_max(&mut self, index: usize, value: u8) -> Result<(), SketchError>;
    
    /// Get the current capacity of the buffer
    fn capacity(&self) -> usize;
    
    /// Get the current length/size of the buffer
    fn len(&self) -> usize;
    
    /// Check if the buffer is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Trait for buffers that can grow automatically
pub trait GrowableWriteBuffer: WriteBuffer {
    /// Ensure the buffer has at least the specified capacity
    fn ensure_capacity(&mut self, needed: usize);
}

/// Simple VarInt reader that wraps a byte slice with position tracking
pub struct SimpleVarIntReader<'a> {
    data: &'a [u8],
    position: usize,
}

impl<'a> SimpleVarIntReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, position: 0 }
    }
    
    pub fn position(&self) -> usize {
        self.position
    }
}

impl<'a> VarIntReader for SimpleVarIntReader<'a> {
    fn read_varint(&mut self) -> Result<i32, SketchError> {
        if self.position >= self.data.len() {
            return Err(SketchError::InvalidState("No more data to read".to_string()));
        }
        
        let (value, bytes_read) = VarInt::get_var_int(&self.data[self.position..]);
        self.position += bytes_read;
        Ok(value)
    }
    
    fn has_remaining(&self) -> bool {
        self.position < self.data.len()
    }
    
    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.position)
    }
}

/// Simple write buffer that wraps a Vec<u8> with growing capability
pub struct GrowingVarIntWriter {
    data: Vec<u8>,
}

impl GrowingVarIntWriter {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }
    
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
        }
    }
    
    pub fn into_vec(self) -> Vec<u8> {
        self.data
    }
    
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }
    
    pub fn clear(&mut self) {
        self.data.clear();
    }
}

impl Default for GrowingVarIntWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl WriteBuffer for GrowingVarIntWriter {
    fn write_varint(&mut self, value: i32) -> Result<(), SketchError> {
        let size_needed = VarInt::var_int_size(value);
        let start_pos = self.data.len();
        self.data.resize(start_pos + size_needed, 0);
        
        let bytes_written = VarInt::set_var_int(value, &mut self.data[start_pos..]);
        debug_assert_eq!(bytes_written, size_needed);
        
        Ok(())
    }
    
    fn write_max(&mut self, index: usize, value: u8) -> Result<(), SketchError> {
        if index >= self.data.len() {
            return Err(SketchError::InvalidState(format!(
                "Index {} out of bounds for buffer length {}",
                index,
                self.data.len()
            )));
        }
        
        if self.data[index] < value {
            self.data[index] = value;
        }
        Ok(())
    }
    
    fn capacity(&self) -> usize {
        self.data.capacity()
    }
    
    fn len(&self) -> usize {
        self.data.len()
    }
}

impl GrowableWriteBuffer for GrowingVarIntWriter {
    fn ensure_capacity(&mut self, needed: usize) {
        if self.data.capacity() < needed {
            self.data.reserve(needed - self.data.capacity());
        }
    }
}

/// Fixed-size write buffer that wraps a mutable slice
pub struct FixedVarIntWriter<'a> {
    data: &'a mut [u8],
}

impl<'a> FixedVarIntWriter<'a> {
    pub fn new(data: &'a mut [u8]) -> Self {
        Self { data }
    }
}

impl<'a> WriteBuffer for FixedVarIntWriter<'a> {
    fn write_varint(&mut self, _value: i32) -> Result<(), SketchError> {
        // Fixed buffers typically don't support VarInt writing in our use case
        // This is mainly for normal representation data which uses write_max
        Err(SketchError::InvalidState(
            "VarInt writing not supported for fixed buffers".to_string(),
        ))
    }
    
    fn write_max(&mut self, index: usize, value: u8) -> Result<(), SketchError> {
        if index >= self.data.len() {
            return Err(SketchError::InvalidState(format!(
                "Index {} out of bounds for buffer length {}",
                index,
                self.data.len()
            )));
        }
        
        if self.data[index] < value {
            self.data[index] = value;
        }
        Ok(())
    }
    
    fn capacity(&self) -> usize {
        self.data.len()
    }
    
    fn len(&self) -> usize {
        self.data.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_varint_reader() {
        let data = vec![0x08, 0x96, 0x01]; // VarInt encoding of 8 and 150
        let mut reader = SimpleVarIntReader::new(&data);
        
        assert!(reader.has_remaining());
        assert_eq!(reader.remaining(), 3);
        
        let value1 = reader.read_varint().expect("Failed to read first varint");
        assert_eq!(value1, 8);
        assert_eq!(reader.position(), 1);
        
        let value2 = reader.read_varint().expect("Failed to read second varint");
        assert_eq!(value2, 150);
        assert_eq!(reader.position(), 3);
        
        assert!(!reader.has_remaining());
        assert_eq!(reader.remaining(), 0);
        
        // Should fail to read beyond end
        assert!(reader.read_varint().is_err());
    }
    
    #[test]
    fn test_growing_varint_writer() {
        let mut writer = GrowingVarIntWriter::new();
        
        assert_eq!(writer.len(), 0);
        assert!(writer.is_empty());
        
        writer.write_varint(8).expect("Failed to write varint");
        writer.write_varint(150).expect("Failed to write varint");
        
        let data = writer.into_vec();
        assert_eq!(data, vec![0x08, 0x96, 0x01]);
    }
    
    #[test]
    fn test_fixed_varint_writer() {
        let mut data = vec![1, 2, 3, 4, 5];
        let mut writer = FixedVarIntWriter::new(&mut data);
        
        assert_eq!(writer.len(), 5);
        writer.write_max(2, 10).expect("Failed to write max");
        drop(writer);
        assert_eq!(data[2], 10);
        
        let mut writer = FixedVarIntWriter::new(&mut data);
        writer.write_max(2, 5).expect("Failed to write max");
        drop(writer);
        assert_eq!(data[2], 10); // Should remain 10 since 10 > 5
        
        let mut writer = FixedVarIntWriter::new(&mut data);
        // Should fail for out of bounds
        assert!(writer.write_max(10, 1).is_err());
    }
} 