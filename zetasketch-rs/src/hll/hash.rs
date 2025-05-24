use crate::fingerprint2011::fingerprint;

pub struct Hash;

impl Hash {
    pub fn of_i32(value: i32) -> u64 {
        fingerprint(&value.to_le_bytes()) as u64
    }

    pub fn of_u32(value: u32) -> u64 {
        fingerprint(&value.to_le_bytes()) as u64
    }

    pub fn of_i64(value: i64) -> u64 {
        fingerprint(&value.to_le_bytes()) as u64
    }

    pub fn of_u64(value: u64) -> u64 {
        fingerprint(&value.to_le_bytes()) as u64
    }

    pub fn of_bytes(value: &[u8]) -> u64 {
        fingerprint(value) as u64
    }

    pub fn of_string(value: &str) -> u64 {
        fingerprint(value.as_bytes()) as u64
    }
}
