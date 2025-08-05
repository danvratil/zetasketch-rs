// SPDX-FileCopyrightText: 2025 Daniel Vrátil <me@dvratil.cz>
//
// SPDX-License-Identifier: MIT

//! Implementation of Geoff Pike's fingerprint2011 hash.
//!
//! This code was taken from the implementation of fingerprint2011 hash in Google's
//! [Guava library](https://github.com/google/guava) and translated from Java to Rust.
//! Apparently the Java code itself was translated from original implementation in C++,
//! but I couldn't find that version online.
//!
//! I did only minimal modifications to the code (other than changing Java to Rust syntax),
//! mostly just replacing of `offset` and `length` parameters in favor of using Rust slices.
//!
//! The test cases are taken from the tests in the original Java code in order to ensure
//! that the Rust version behaves the same way.
//!
//! The original code is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

use byteorder::{ByteOrder, LittleEndian};

static K0: i64 = 0xa5b85c5e198ed849u64 as i64;
static K1: i64 = 0x8d58ac26afe12e47u64 as i64;
static K2: i64 = 0xc47b6e9e3a970ed3u64 as i64;
static K3: i64 = 0xc6a4a7935bd1e995u64 as i64;

fn load_64(bytes: &[u8]) -> i64 {
    LittleEndian::read_i64(&bytes[0..8])
}

fn load_64_safely(bytes: &[u8]) -> i64 {
    let mut result: i64 = 0;

    let limit = bytes.len().min(8);
    for (i, b) in bytes.iter().enumerate().take(limit) {
        result |= ((*b as i64) & 0xFF) << (i * 8)
    }

    result
}

pub fn fingerprint(bytes: &[u8]) -> i64 {
    let result = if bytes.len() <= 32 {
        murmur_hash64_with_seed(bytes, K0 ^ K1 ^ K2)
    } else if bytes.len() <= 64 {
        hash_length_33_to_64(bytes)
    } else {
        full_fingerprint(bytes)
    };

    let u = if bytes.len() >= 8 { load_64(bytes) } else { K0 };
    let v = if bytes.len() >= 9 {
        load_64(&bytes[bytes.len() - 8..])
    } else {
        K0
    };
    let result = hash_128_to_64(result.wrapping_add(v), u);
    if result == 0 || result == 1 {
        result + !1
    } else {
        result
    }
}

fn shift_mix(val: i64) -> i64 {
    val ^ (((val as u64) >> 47) as i64)
}

fn hash_128_to_64(high: i64, low: i64) -> i64 {
    let mut a = (low ^ high).wrapping_mul(K3);
    a ^= ((a as u64) >> 47) as i64;
    let mut b = (high ^ a).wrapping_mul(K3);
    b ^= ((b as u64) >> 47) as i64;
    b = b.wrapping_mul(K3);
    b
}

fn weak_hash_length_32_with_seed(bytes: &[u8], seed_a: i64, seed_b: i64) -> [i64; 2] {
    let part1 = load_64(bytes);
    let part2 = load_64(&bytes[8..]);
    let part3 = load_64(&bytes[16..]);
    let part4 = load_64(&bytes[24..]);

    let mut seed_a = seed_a.wrapping_add(part1);
    let mut seed_b = seed_b
        .wrapping_add(seed_a)
        .wrapping_add(part4)
        .rotate_right(51);
    let c = seed_a;
    seed_a = seed_a.wrapping_add(part2);
    seed_a = seed_a.wrapping_add(part3);
    seed_b = seed_b.wrapping_add(seed_a.rotate_right(23));
    [seed_a.wrapping_add(part4), seed_b.wrapping_add(c)]
}

fn full_fingerprint(bytes: &[u8]) -> i64 {
    let mut x = load_64(bytes);
    let mut y = load_64(&bytes[bytes.len() - 16..]) ^ K1;
    let mut z = load_64(&bytes[bytes.len() - 56..]) ^ K0;
    let mut v = weak_hash_length_32_with_seed(&bytes[bytes.len() - 64..], bytes.len() as i64, y);
    let mut w = weak_hash_length_32_with_seed(
        &bytes[bytes.len() - 32..],
        (bytes.len() as i64).wrapping_mul(K1),
        K0,
    );
    z = z.wrapping_add(shift_mix(v[1]).wrapping_mul(K1));
    x = z.wrapping_add(x).rotate_right(39).wrapping_mul(K1);
    y = y.rotate_right(33).wrapping_mul(K1);

    // Decrease length to the nearest multiple of 64, and operate on 64-byte chunks.
    let mut offset = 0;
    let mut length = (bytes.len() - 1) & !63;
    loop {
        x = x
            .wrapping_add(y)
            .wrapping_add(v[0])
            .wrapping_add(load_64(&bytes[offset + 16..]))
            .rotate_right(37)
            .wrapping_mul(K1);
        y = y
            .wrapping_add(v[1])
            .wrapping_add(load_64(&bytes[offset + 48..]))
            .rotate_right(42)
            .wrapping_mul(K1);
        x ^= w[1];
        y ^= v[0];
        z = (z ^ w[0]).rotate_right(33);
        v = weak_hash_length_32_with_seed(
            &bytes[offset..],
            v[1].wrapping_mul(K1),
            x.wrapping_add(w[0]),
        );
        w = weak_hash_length_32_with_seed(&bytes[offset + 32..], z.wrapping_add(w[1]), y);
        std::mem::swap(&mut z, &mut x);
        offset += 64;
        length -= 64;

        if length == 0 {
            break;
        }
    }
    hash_128_to_64(
        hash_128_to_64(v[0], w[0])
            .wrapping_add(shift_mix(y).wrapping_mul(K1))
            .wrapping_add(z),
        hash_128_to_64(v[1], w[1]).wrapping_add(x),
    )
}

fn hash_length_33_to_64(bytes: &[u8]) -> i64 {
    let mut z = load_64(&bytes[24..]);
    let mut a = load_64(bytes).wrapping_add(
        ((bytes.len() as i64) + load_64(&bytes[bytes.len() - 16..])).wrapping_mul(K0),
    );
    let mut b = a.wrapping_add(z).rotate_right(52);
    let mut c = a.rotate_right(37);
    a = a.wrapping_add(load_64(&bytes[8..]));
    c = c.wrapping_add(a.rotate_right(7));
    a = a.wrapping_add(load_64(&bytes[16..]));
    let vf = a.wrapping_add(z);
    let vs = b.wrapping_add(a.rotate_right(31)).wrapping_add(c);
    a = load_64(&bytes[16..]).wrapping_add(load_64(&bytes[bytes.len() - 32..]));
    z = load_64(&bytes[bytes.len() - 8..]);
    b = a.wrapping_add(z).rotate_right(52);
    c = a.rotate_right(37);
    a = a.wrapping_add(load_64(&bytes[bytes.len() - 24..]));
    c = c.wrapping_add(a.rotate_right(7));
    a = a.wrapping_add(load_64(&bytes[bytes.len() - 16..]));
    let wf = a.wrapping_add(z);
    let ws = b.wrapping_add(a.rotate_right(31)).wrapping_add(c);
    let r = shift_mix(
        (vf.wrapping_add(ws))
            .wrapping_mul(K2)
            .wrapping_add((wf.wrapping_add(vs)).wrapping_mul(K0)),
    );
    shift_mix(r.wrapping_mul(K0).wrapping_add(vs)).wrapping_mul(K2)
}

fn murmur_hash64_with_seed(bytes: &[u8], seed: i64) -> i64 {
    let mul = K3;
    let top_bit = 0x7;
    let length = bytes.len();

    let length_aligned = length & !top_bit;
    let length_reminder = length & top_bit;
    let mut hash = seed ^ ((length as i64).wrapping_mul(mul));

    for i in (0..length_aligned).step_by(8) {
        let loaded = load_64(&bytes[i..]);
        let data = shift_mix(loaded.wrapping_mul(mul)).wrapping_mul(mul);
        hash ^= data;
        hash = hash.wrapping_mul(mul);
    }

    if length_reminder != 0 {
        let data = load_64_safely(&bytes[length_aligned..length_aligned + length_reminder]);

        hash ^= data;
        hash = hash.wrapping_mul(mul);
    }

    hash = shift_mix(hash).wrapping_mul(mul);
    hash = shift_mix(hash);
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_really_simple_fingerprints() {
        assert_eq!(8473225671271759044, fingerprint("test".as_bytes()));
        // 32 characters long
        assert_eq!(
            7345148637025587076,
            fingerprint("test".repeat(8).as_bytes())
        );
        // 256 characters long
        assert_eq!(
            4904844928629814570,
            fingerprint("test".repeat(64).as_bytes())
        );
    }

    #[test]
    fn test_string_consistency() {
        for s in ["", "some", "test", "strings", "to", "try"] {
            assert_eq!(fingerprint(s.as_bytes()), fingerprint(s.as_bytes()));
        }
    }

    #[test]
    fn test_murmur_hash64() {
        let bytes = b"test";
        assert_eq!(1618900948208871284, murmur_hash64_with_seed(bytes, 1));

        let bytes = b"test test test";
        assert_eq!(
            12313169684067793560u64 as i64,
            murmur_hash64_with_seed(bytes, 1)
        );
    }

    #[test]
    fn test_non_chars() {
        let bytes: [u8; 8] = [0x1, 0x1, 0x0, 0x1, 0x0, 0x0, 0x0, 0x0];
        let hash_code = fingerprint(&bytes);

        assert_eq!(-7885230138691441470, hash_code);
    }

    fn remix(h: i64) -> i64 {
        let h = h ^ ((h as u64) >> 41) as i64;
        h.wrapping_mul(949921979)
    }

    fn get_char(h: i64) -> u8 {
        (('a' as i64) + ((h & 0xfffff) % 26)) as u8
    }

    #[test]
    fn test_multiple_lengths() {
        const ITERATIONS: usize = 800;
        let mut buf = [0u8; ITERATIONS * 4];

        let mut buf_len = 0;
        let mut h = 0;

        for i in 0..ITERATIONS {
            h ^= fingerprint(&buf[..i]);
            h = remix(h);
            buf[buf_len] = get_char(h);
            buf_len += 1;

            h ^= fingerprint(&buf[..i * i % buf_len]);
            h = remix(h);
            buf[buf_len] = get_char(h);
            buf_len += 1;

            h ^= fingerprint(&buf[..i * i * i % buf_len]);
            h = remix(h);
            buf[buf_len] = get_char(h);
            buf_len += 1;

            h ^= fingerprint(&buf[..buf_len]);
            h = remix(h);
            buf[buf_len] = get_char(h);
            buf_len += 1;

            let x0: i32 = buf[buf_len - 1] as i32;
            let x1: i32 = buf[buf_len - 2] as i32;
            let x2: i32 = buf[buf_len - 3] as i32;
            let x3: i32 = buf[buf_len / 2] as i32;
            buf[(((x0 << 16) + (x1 << 8) + x2) % buf_len as i32) as usize] ^= x3 as u8;
            buf[(((x1 << 16) + (x2 << 8) + x3) % buf_len as i32) as usize] ^=
                ((i as i32) % 256) as u8;
        }

        assert_eq!(0xeaa3b1c985261632u64 as i64, h);
    }
}
