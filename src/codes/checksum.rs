//! Checksums and check digits: cheap ways to notice that data changed.
//!
//! None of these corrects anything, and none of them resists an adversary.
//! What they do is turn a class of likely accidents into a mismatch, and the
//! useful question about each is which class. A single parity bit catches any
//! odd number of flipped bits and nothing else. A Fletcher or Adler sum
//! catches reordering, which a plain sum does not, because the second
//! accumulator weights each byte by its position. A CRC of width `w` catches
//! every burst of `w` bits or fewer, every odd number of bit errors when the
//! polynomial has `x + 1` as a factor, and all but `2^-w` of everything else.
//! The decimal check digits catch every single-digit error and, except for
//! Luhn, every transposition of adjacent digits.
//!
//! For an adversary, none of this is relevant: all of it is linear or nearly
//! so, and a forger can adjust the data to hit any checksum they like.

/// Even parity: `true` when an odd number of bits are set, so that appending
/// it makes the total even.
///
/// Detects any odd number of bit errors and no even number, which is the
/// whole of what a single bit can promise.
#[must_use]
pub fn parity(bits: &[bool]) -> bool {
    bits.iter().filter(|&&b| b).count() % 2 == 1
}

/// Parity of the set bits of a word.
#[must_use]
pub fn parity_u64(x: u64) -> bool {
    x.count_ones() % 2 == 1
}

/// The Fletcher-16 checksum: a running byte sum and a running sum of that
/// sum, both modulo 255, packed into sixteen bits.
///
/// The second accumulator is what makes it more than a sum: it weights each
/// byte by how many bytes follow it, so swapping two bytes changes the
/// result, which a plain sum cannot notice. Modulo 255 rather than 256
/// because a modulus with a factor of two lets the high bits of a byte fall
/// out of the low accumulator entirely.
#[must_use]
pub fn checksum_fletcher16(data: &[u8]) -> u16 {
    let (mut lo, mut hi) = (0u16, 0u16);
    for &b in data {
        lo = (lo + u16::from(b)) % 255;
        hi = (hi + lo) % 255;
    }
    (hi << 8) | lo
}

/// The Fletcher-32 checksum, over sixteen-bit words modulo 65535.
///
/// Odd-length input is padded with a zero byte, which is the usual
/// convention and the reason Fletcher-32 cannot distinguish `"ab"` from
/// `"ab\0"`.
#[must_use]
pub fn checksum_fletcher32(data: &[u8]) -> u32 {
    let (mut lo, mut hi) = (0u32, 0u32);
    for pair in data.chunks(2) {
        let w = u32::from(pair[0]) | (u32::from(*pair.get(1).unwrap_or(&0)) << 8);
        lo = (lo + w) % 65535;
        hi = (hi + lo) % 65535;
    }
    (hi << 16) | lo
}

/// Adler-32, as used by zlib: Fletcher's idea with a prime modulus.
///
/// The accumulators start at one and zero and run modulo 65521, the largest
/// prime below `2^16`. The prime modulus spreads the values more evenly than
/// Fletcher's 65535, and the leading one makes the checksum of an empty
/// input distinguishable from the checksum of a run of zero bytes.
#[must_use]
pub fn adler32(data: &[u8]) -> u32 {
    const MOD: u32 = 65521;
    let (mut a, mut b) = (1u32, 0u32);
    for &x in data {
        a = (a + u32::from(x)) % MOD;
        b = (b + a) % MOD;
    }
    (b << 16) | a
}

/// Reverse the low `width` bits of `x`.
fn reflect(x: u64, width: u32) -> u64 {
    let mut out = 0u64;
    for i in 0..width {
        if x & (1 << i) != 0 {
            out |= 1 << (width - 1 - i);
        }
    }
    out
}

/// A cyclic redundancy check, in the parametric form every named CRC is an
/// instance of.
///
/// The message is treated as a polynomial over `GF(2)`, shifted left by
/// `width` and divided by `poly`; the remainder is the check value. Because
/// the code is linear, the difference between a message and a corrupted one
/// has its own remainder, so a corruption goes unnoticed exactly when its
/// error pattern is itself a multiple of `poly` -- which no burst shorter
/// than `width + 1` can be, since `poly` has degree `width`.
///
/// `init` seeds the register, so a run of leading zero bytes changes the
/// result; `xor_out` is applied at the end; `reflect` reverses the bits of
/// each input byte and of the final register, which is what the
/// bit-at-a-time hardware of a serial line does naturally. The named CRCs in
/// wide use all reflect input and output together or neither, so one flag
/// covers them.
///
/// # Panics
/// Panics unless `width` is between 8 and 64.
#[must_use]
pub fn crc(data: &[u8], poly: u64, width: u32, init: u64, xor_out: u64, reflect_io: bool) -> u64 {
    assert!((8..=64).contains(&width), "CRC width must be between 8 and 64");
    let mask = if width == 64 { u64::MAX } else { (1u64 << width) - 1 };
    let top = 1u64 << (width - 1);
    let mut reg = init & mask;
    for &byte in data {
        let b = if reflect_io { reflect(u64::from(byte), 8) } else { u64::from(byte) };
        reg ^= b << (width - 8);
        for _ in 0..8 {
            reg = if reg & top != 0 { ((reg << 1) ^ poly) & mask } else { (reg << 1) & mask };
        }
    }
    if reflect_io {
        reg = reflect(reg, width);
    }
    (reg ^ xor_out) & mask
}

/// CRC-32 as used by Ethernet, zip, PNG and gzip.
///
/// Polynomial `0x04C11DB7`, register seeded to all ones, reflected in and
/// out, complemented at the end. The check value of `"123456789"` is
/// `0xCBF43926`.
#[must_use]
pub fn crc32_ieee(data: &[u8]) -> u32 {
    crc(data, 0x04C1_1DB7, 32, 0xFFFF_FFFF, 0xFFFF_FFFF, true) as u32
}

/// CRC-16/CCITT-FALSE: polynomial `0x1021`, seeded to all ones, unreflected,
/// no final xor. The check value of `"123456789"` is `0x29B1`.
///
/// The name records a long-standing confusion: the true CCITT parameters
/// seed the register to zero, and this variant -- which is the one actually
/// deployed, in XMODEM's successors and in many microcontroller libraries --
/// does not.
#[must_use]
pub fn crc16_ccitt(data: &[u8]) -> u16 {
    crc(data, 0x1021, 16, 0xFFFF, 0x0000, false) as u16
}

/// CRC-8/SMBUS: polynomial `0x07`, zero seed, unreflected. The check value
/// of `"123456789"` is `0xF4`.
#[must_use]
pub fn crc8(data: &[u8]) -> u8 {
    crc(data, 0x07, 8, 0x00, 0x00, false) as u8
}

/// The 256-entry lookup table for a reflected 32-bit CRC.
///
/// `poly` is the *reversed* polynomial -- `0xEDB88320` for CRC-32 -- because
/// a reflected CRC shifts right, and the table holds the remainder of each
/// possible byte. Processing a byte becomes one table lookup instead of
/// eight conditional shifts; the table is the loop unrolled once and cached.
#[must_use]
pub fn crc_table(poly: u32) -> [u32; 256] {
    let mut table = [0u32; 256];
    for (i, entry) in table.iter_mut().enumerate() {
        let mut c = i as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { (c >> 1) ^ poly } else { c >> 1 };
        }
        *entry = c;
    }
    table
}

/// CRC-32 driven by a precomputed table rather than bit by bit.
///
/// The same value as [`crc32_ieee`], computed eight bits at a time. Pass the
/// table from [`crc_table`] with the reversed polynomial.
#[must_use]
pub fn crc32_with_table(data: &[u8], table: &[u32; 256]) -> u32 {
    let mut c = 0xFFFF_FFFFu32;
    for &b in data {
        c = table[((c ^ u32::from(b)) & 0xFF) as usize] ^ (c >> 8);
    }
    c ^ 0xFFFF_FFFF
}

// ---------------------------------------------------------------------------
// Decimal check digits
// ---------------------------------------------------------------------------

/// The Luhn checksum test, as used on payment card numbers.
///
/// Doubling every second digit from the right and casting out nines catches
/// every single-digit error and every transposition of adjacent digits
/// except `09` against `90`, which it maps to the same sum. That one blind
/// spot is why Verhoeff and Damm exist.
///
/// The check digit is the last element of `digits`.
///
/// # Panics
/// Panics if any entry is above nine.
#[must_use]
pub fn luhn_check(digits: &[u8]) -> bool {
    assert!(digits.iter().all(|&d| d <= 9), "decimal digits only");
    if digits.is_empty() {
        return false;
    }
    luhn_sum(digits).is_multiple_of(10)
}

fn luhn_sum(digits: &[u8]) -> u32 {
    digits
        .iter()
        .rev()
        .enumerate()
        .map(|(i, &d)| {
            let mut v = u32::from(d);
            if i % 2 == 1 {
                v *= 2;
                if v > 9 {
                    v -= 9;
                }
            }
            v
        })
        .sum()
}

/// The Luhn check digit that completes `payload`.
///
/// # Panics
/// Panics if any entry is above nine.
#[must_use]
pub fn luhn_generate(payload: &[u8]) -> u8 {
    assert!(payload.iter().all(|&d| d <= 9), "decimal digits only");
    let mut with_slot = payload.to_vec();
    with_slot.push(0);
    ((10 - luhn_sum(&with_slot) % 10) % 10) as u8
}

/// ISBN-10, whose check digit is a weighted sum modulo eleven.
///
/// Weights ten down to one, and the modulus is prime, which is what lets it
/// catch every transposition -- swapping two digits changes the sum by a
/// non-zero multiple of their difference, and a prime modulus has no zero
/// divisors to hide that. The price is that the check digit sometimes has to
/// be ten, written `X`; pass it as the value `10`.
///
/// # Panics
/// Panics unless there are ten entries, each at most nine, except the last
/// which may be ten.
#[must_use]
pub fn isbn10_check(digits: &[u8]) -> bool {
    assert_eq!(digits.len(), 10, "an ISBN-10 has ten digits");
    assert!(digits[..9].iter().all(|&d| d <= 9), "only the check digit may be X");
    assert!(digits[9] <= 10, "the check digit is 0 to 9 or X");
    let sum: u32 =
        digits.iter().enumerate().map(|(i, &d)| (10 - i as u32) * u32::from(d)).sum();
    sum.is_multiple_of(11)
}

/// ISBN-13, the same numbering embedded in the EAN-13 scheme: alternating
/// weights of one and three modulo ten.
///
/// The modulus is composite, so unlike ISBN-10 it misses transpositions of
/// adjacent digits differing by five -- but it never needs an `X`, which is
/// what the change bought.
///
/// # Panics
/// Panics unless there are thirteen digits, each at most nine.
#[must_use]
pub fn isbn13_check(digits: &[u8]) -> bool {
    assert_eq!(digits.len(), 13, "an ISBN-13 has thirteen digits");
    assert!(digits.iter().all(|&d| d <= 9), "decimal digits only");
    let sum: u32 = digits
        .iter()
        .enumerate()
        .map(|(i, &d)| if i % 2 == 0 { u32::from(d) } else { 3 * u32::from(d) })
        .sum();
    sum.is_multiple_of(10)
}

/// The multiplication table of the dihedral group of order ten.
const VERHOEFF_D: [[u8; 10]; 10] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
];

/// The permutation applied at each position, of order eight.
const VERHOEFF_P: [[u8; 10]; 8] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
];

/// The inverse in the dihedral group.
const VERHOEFF_INV: [u8; 10] = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9];

/// The Verhoeff check, which catches every single-digit error and every
/// transposition of adjacent digits.
///
/// It works by giving up on arithmetic modulo ten and using the dihedral
/// group of order ten instead, which is not commutative -- so swapping two
/// digits genuinely changes the product, with no cases left over. A
/// position-dependent permutation of order eight is applied first, which is
/// what extends the guarantee past the two digits nearest the check digit.
///
/// The check digit is the last element of `digits`.
///
/// # Panics
/// Panics if any entry is above nine.
#[must_use]
pub fn verhoeff_check(digits: &[u8]) -> bool {
    assert!(digits.iter().all(|&d| d <= 9), "decimal digits only");
    let mut c = 0usize;
    for (i, &d) in digits.iter().rev().enumerate() {
        c = VERHOEFF_D[c][VERHOEFF_P[i % 8][d as usize] as usize] as usize;
    }
    c == 0
}

/// The Verhoeff check digit that completes `payload`.
///
/// # Panics
/// Panics if any entry is above nine.
#[must_use]
pub fn verhoeff_generate(payload: &[u8]) -> u8 {
    assert!(payload.iter().all(|&d| d <= 9), "decimal digits only");
    let mut c = 0usize;
    // The check digit occupies position zero, so the payload starts at one.
    for (i, &d) in payload.iter().rev().enumerate() {
        c = VERHOEFF_D[c][VERHOEFF_P[(i + 1) % 8][d as usize] as usize] as usize;
    }
    VERHOEFF_INV[c]
}

/// A totally anti-symmetric quasigroup of order ten.
const DAMM: [[u8; 10]; 10] = [
    [0, 3, 1, 7, 5, 9, 8, 6, 4, 2],
    [7, 0, 9, 2, 1, 5, 4, 8, 6, 3],
    [4, 2, 0, 6, 8, 7, 1, 3, 5, 9],
    [1, 7, 5, 0, 9, 8, 3, 4, 2, 6],
    [6, 1, 2, 3, 0, 4, 5, 9, 7, 8],
    [3, 6, 7, 4, 2, 0, 9, 5, 8, 1],
    [5, 8, 6, 9, 7, 2, 0, 1, 3, 4],
    [8, 9, 4, 5, 3, 6, 2, 0, 1, 7],
    [9, 4, 3, 8, 6, 1, 7, 2, 0, 5],
    [2, 5, 8, 1, 4, 3, 6, 7, 9, 0],
];

/// The Damm check, with the same guarantees as Verhoeff and none of its
/// tables.
///
/// One quasigroup operation folded across the digits, with no permutation
/// and no inverse: the check digit is simply the interim value, because the
/// table's diagonal is zero. Total anti-symmetry -- that `(a * b) * c` and
/// `(a * c) * b` differ whenever `b` and `c` do -- is exactly the property
/// that catches transpositions, and it is built into the table rather than
/// arranged around it.
///
/// The check digit is the last element of `digits`.
///
/// # Panics
/// Panics if any entry is above nine.
#[must_use]
pub fn damm_check(digits: &[u8]) -> bool {
    assert!(digits.iter().all(|&d| d <= 9), "decimal digits only");
    let mut interim = 0usize;
    for &d in digits {
        interim = DAMM[interim][d as usize] as usize;
    }
    interim == 0
}

/// The Damm check digit that completes `payload`.
///
/// # Panics
/// Panics if any entry is above nine.
#[must_use]
pub fn damm_generate(payload: &[u8]) -> u8 {
    assert!(payload.iter().all(|&d| d <= 9), "decimal digits only");
    let mut interim = 0usize;
    for &d in payload {
        interim = DAMM[interim][d as usize] as usize;
    }
    interim as u8
}

// ---------------------------------------------------------------------------
// Hamming distance
// ---------------------------------------------------------------------------

/// The number of bit positions in which two words differ.
///
/// The distance a code needs to survive: a code whose words are all at least
/// `d` apart detects `d - 1` errors and corrects `(d - 1) / 2`, because a
/// received word within that radius of a codeword is within that radius of
/// no other.
#[must_use]
pub fn hamming_distance_bits(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// The bitwise Hamming distance between two byte strings, or `None` if they
/// are different lengths.
#[must_use]
pub fn hamming_distance_bytes(a: &[u8], b: &[u8]) -> Option<u32> {
    if a.len() != b.len() {
        return None;
    }
    Some(a.iter().zip(b).map(|(&x, &y)| u32::from((x ^ y).count_ones() as u8)).sum())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monte_carlo::Rng;

    fn pick(rng: &mut Rng, n: usize) -> usize {
        ((u128::from(rng.next_u64()) * n as u128) >> 64) as usize
    }

    const CHECK: &[u8] = b"123456789";

    /// Every named CRC against its published check value: the result of
    /// running it over the nine ASCII digits, which is how the CRC catalogue
    /// identifies a parameter set.
    #[test]
    fn crcs_match_their_published_check_values() {
        // (name, poly, width, init, xorout, reflect, check)
        let cases: [(&str, u64, u32, u64, u64, bool, u64); 8] = [
            ("CRC-8/SMBUS", 0x07, 8, 0x00, 0x00, false, 0xF4),
            ("CRC-8/MAXIM-DOW", 0x31, 8, 0x00, 0x00, true, 0xA1),
            ("CRC-16/ARC", 0x8005, 16, 0x0000, 0x0000, true, 0xBB3D),
            ("CRC-16/CCITT-FALSE", 0x1021, 16, 0xFFFF, 0x0000, false, 0x29B1),
            ("CRC-16/XMODEM", 0x1021, 16, 0x0000, 0x0000, false, 0x31C3),
            ("CRC-32/ISO-HDLC", 0x04C1_1DB7, 32, 0xFFFF_FFFF, 0xFFFF_FFFF, true, 0xCBF4_3926),
            ("CRC-32/BZIP2", 0x04C1_1DB7, 32, 0xFFFF_FFFF, 0xFFFF_FFFF, false, 0xFC89_1918),
            (
                "CRC-64/XZ",
                0x42F0_E1EB_A9EA_3693,
                64,
                0xFFFF_FFFF_FFFF_FFFF,
                0xFFFF_FFFF_FFFF_FFFF,
                true,
                0x995D_C9BB_DF19_39FA,
            ),
        ];
        for (name, poly, width, init, xorout, reflect_io, want) in cases {
            assert_eq!(crc(CHECK, poly, width, init, xorout, reflect_io), want, "{name}");
        }
        assert_eq!(crc32_ieee(CHECK), 0xCBF4_3926);
        assert_eq!(crc16_ccitt(CHECK), 0x29B1);
        assert_eq!(crc8(CHECK), 0xF4);

        // The table-driven form must agree with the bit-at-a-time form on
        // every input, which is the only reason to trust the table.
        let table = crc_table(0xEDB8_8320);
        let mut rng = Rng::new(0x_C2C0);
        for _ in 0..500 {
            let n = pick(&mut rng, 64);
            let data: Vec<u8> = (0..n).map(|_| (rng.next_u64() & 0xFF) as u8).collect();
            assert_eq!(crc32_with_table(&data, &table), crc32_ieee(&data));
        }
    }

    /// The guarantee a CRC of width `w` is chosen for: no burst of `w` bits
    /// or fewer can go unnoticed.
    ///
    /// A burst is an error pattern whose set bits all lie within a window of
    /// that many positions. Such a pattern is a polynomial of degree below
    /// `w` times a power of `x`, and the generator has degree `w` with a
    /// non-zero constant term, so it cannot divide one -- which is exactly
    /// what makes the corrupted message's remainder differ.
    #[test]
    fn a_crc_detects_every_burst_no_longer_than_its_width() {
        let mut rng = Rng::new(0x_B025);
        // A burst is contiguous in the order the CRC consumes bits, which is
        // most-significant first within each byte -- unless the CRC reflects
        // its input, in which case it is least-significant first. Numbering
        // bits the other way scatters a window across up to twice its span
        // and the guarantee stops applying, so each case carries its own.
        for (width, reflected, f) in [
            (8u32, false, (|d: &[u8]| u64::from(crc8(d))) as fn(&[u8]) -> u64),
            (16, false, |d: &[u8]| u64::from(crc16_ccitt(d))),
            (16, true, |d: &[u8]| crc(d, 0x8005, 16, 0, 0, true)),
            (32, true, |d: &[u8]| u64::from(crc32_ieee(d))),
            (32, false, |d: &[u8]| crc(d, 0x04C1_1DB7, 32, 0xFFFF_FFFF, 0xFFFF_FFFF, false)),
        ] {
            for _ in 0..400 {
                let bytes = 8 + pick(&mut rng, 24);
                let data: Vec<u8> = (0..bytes).map(|_| (rng.next_u64() & 0xFF) as u8).collect();
                let clean = f(&data);
                let total_bits = bytes * 8;
                let start = pick(&mut rng, total_bits - width as usize);
                // A pattern confined to `width` bits, with the first bit set
                // so the burst really starts where it says it does.
                let mut pattern = rng.next_u64() & ((1u64 << (width - 1)) - 1);
                pattern = (pattern << 1) | 1;
                let mut bad = data.clone();
                for k in 0..width as usize {
                    if pattern & (1 << k) != 0 {
                        let bit = start + k;
                        let within = if reflected { bit % 8 } else { 7 - bit % 8 };
                        bad[bit / 8] ^= 1 << within;
                    }
                }
                assert_ne!(
                    f(&bad),
                    clean,
                    "a {width}-bit burst went undetected (reflected: {reflected})"
                );
            }
        }
    }

    /// A CRC seeded to zero with no final xor is linear over `GF(2)`: the
    /// check value of the bitwise difference of two messages is the
    /// difference of their check values.
    ///
    /// This is not decoration. It is why the burst guarantee above is a
    /// statement about error patterns at all: an undetected corruption is
    /// exactly an error pattern whose own check value is zero, independent of
    /// what was sent.
    #[test]
    fn a_zero_seeded_crc_is_linear() {
        let mut rng = Rng::new(0x_11EA);
        for _ in 0..400 {
            let n = 1 + pick(&mut rng, 32);
            let a: Vec<u8> = (0..n).map(|_| (rng.next_u64() & 0xFF) as u8).collect();
            let b: Vec<u8> = (0..n).map(|_| (rng.next_u64() & 0xFF) as u8).collect();
            let x: Vec<u8> = a.iter().zip(&b).map(|(&p, &q)| p ^ q).collect();
            for (poly, width, reflect_io) in
                [(0x07u64, 8u32, false), (0x1021, 16, false), (0x8005, 16, true), (0x04C1_1DB7, 32, true)]
            {
                let ca = crc(&a, poly, width, 0, 0, reflect_io);
                let cb = crc(&b, poly, width, 0, 0, reflect_io);
                let cx = crc(&x, poly, width, 0, 0, reflect_io);
                assert_eq!(cx, ca ^ cb, "not linear for polynomial {poly:#x}");
            }
        }
    }

    /// A generator with an even number of terms is divisible by `x + 1`, and
    /// a CRC built on one detects every odd number of bit errors.
    ///
    /// The reason is that an error pattern with an odd number of set bits
    /// evaluates to one at `x = 1`, so `x + 1` does not divide it, so the
    /// generator does not either. CRC-16/CCITT and CRC-8/SMBUS both qualify;
    /// CRC-32 does not, and this test says so rather than claiming a
    /// guarantee it does not have.
    #[test]
    fn an_even_term_generator_detects_every_odd_error_count() {
        // Counting terms, the implicit leading one included.
        assert_eq!(0x1021u64.count_ones() + 1, 4, "CRC-16/CCITT has an even term count");
        assert_eq!(0x07u64.count_ones() + 1, 4, "CRC-8/SMBUS has an even term count");
        assert_eq!(0x04C1_1DB7u64.count_ones() + 1, 15, "CRC-32 has an odd term count");

        let mut rng = Rng::new(0x_0DD1);
        for (name, f) in [
            ("CRC-16/CCITT", (|d: &[u8]| u64::from(crc16_ccitt(d))) as fn(&[u8]) -> u64),
            ("CRC-8/SMBUS", |d: &[u8]| u64::from(crc8(d))),
        ] {
            for _ in 0..1500 {
                let bytes = 4 + pick(&mut rng, 28);
                let data: Vec<u8> = (0..bytes).map(|_| (rng.next_u64() & 0xFF) as u8).collect();
                let clean = f(&data);
                let flips = 1 + 2 * pick(&mut rng, 5);
                let mut bad = data.clone();
                let mut chosen = std::collections::BTreeSet::new();
                while chosen.len() < flips {
                    chosen.insert(pick(&mut rng, bytes * 8));
                }
                for bit in chosen {
                    bad[bit / 8] ^= 1 << (bit % 8);
                }
                assert_ne!(f(&bad), clean, "{name} missed {flips} bit errors");
            }
        }
    }

    /// Fletcher and Adler notice reordering, which is the whole reason to
    /// carry a second accumulator; a plain byte sum cannot.
    #[test]
    fn position_weighted_sums_notice_reordering() {
        // Published check values.
        assert_eq!(checksum_fletcher16(b"abcde"), 0xC8F0);
        assert_eq!(checksum_fletcher32(b"abcde"), 0xF04F_C729);
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
        assert_eq!(adler32(b""), 1, "the leading one distinguishes empty from zeros");
        assert_ne!(adler32(b""), adler32(&[0u8]));

        let mut rng = Rng::new(0x_F1E7);
        let mut swaps = 0;
        let mut fletcher_blind = 0;
        for _ in 0..2000 {
            let n = 2 + pick(&mut rng, 30);
            let mut data: Vec<u8> = (0..n).map(|_| (rng.next_u64() & 0xFF) as u8).collect();
            let (i, j) = (pick(&mut rng, n), pick(&mut rng, n));
            if i == j || data[i] == data[j] {
                continue;
            }
            let plain: u32 = data.iter().map(|&b| u32::from(b)).sum();
            let before = (checksum_fletcher16(&data), adler32(&data));
            data.swap(i, j);
            let after = (checksum_fletcher16(&data), adler32(&data));
            swaps += 1;
            // The plain sum is blind to the swap by construction.
            assert_eq!(plain, data.iter().map(|&b| u32::from(b)).sum::<u32>());
            // Both second accumulators weight byte m by the number of bytes
            // after it, so a swap moves them by (d_i - d_j)(i - j) and the
            // low accumulator not at all. Whether the swap is caught is
            // therefore exactly whether that product survives the modulus.
            let shift = (data[j] as i64 - data[i] as i64) * (j as i64 - i as i64);
            assert_eq!(
                before.0 != after.0,
                shift.rem_euclid(255) != 0,
                "Fletcher-16's blind spot is not where the modulus puts it"
            );
            // Adler-32's modulus is 65521, and the shift is bounded by 255
            // times the length, so it can never reach a multiple. That is
            // what the prime buys, and it is why Adler-32 never misses one.
            assert!(shift.abs() < 65521);
            assert_ne!(before.1, after.1, "Adler-32 missed a transposition");
            if shift.rem_euclid(255) == 0 {
                fletcher_blind += 1;
            }
        }
        assert!(swaps > 1000, "only {swaps} genuine transpositions were drawn");
        assert!(fletcher_blind > 0, "Fletcher-16's blind spot was never exercised");
    }

    /// Parity detects an odd number of flips and nothing else -- checked
    /// exhaustively over every error pattern on ten bits.
    #[test]
    fn parity_detects_exactly_the_odd_error_counts() {
        let bits: Vec<bool> = (0..10).map(|i| i % 3 == 0).collect();
        let p = parity(&bits);
        for pattern in 0u32..1024 {
            let flipped: Vec<bool> =
                bits.iter().enumerate().map(|(i, &b)| b ^ (pattern & (1 << i) != 0)).collect();
            let detected = parity(&flipped) != p;
            assert_eq!(detected, pattern.count_ones() % 2 == 1, "pattern {pattern:#b}");
        }
        assert!(!parity(&[]));
        for x in [0u64, 1, 3, 7, 0xFF, u64::MAX] {
            assert_eq!(parity_u64(x), x.count_ones() % 2 == 1);
        }
    }

    /// Luhn catches every single-digit error and every adjacent
    /// transposition except the one it is known to miss.
    ///
    /// Doubling and casting out nines sends 0 to 0 and 9 to 9, so a `09`
    /// against a `90` contributes the same either way. The test asserts the
    /// blind spot exists rather than working around it, because a change
    /// that closed it would change the algorithm.
    #[test]
    fn luhn_catches_all_but_its_one_known_blind_spot() {
        // The textbook valid number.
        assert!(luhn_check(&[7, 9, 9, 2, 7, 3, 9, 8, 7, 1, 3]));
        assert_eq!(luhn_generate(&[7, 9, 9, 2, 7, 3, 9, 8, 7, 1]), 3);
        assert!(!luhn_check(&[7, 9, 9, 2, 7, 3, 9, 8, 7, 1, 4]));

        let mut rng = Rng::new(0x_1A4A);
        let mut blind = 0;
        let mut caught = 0;
        for _ in 0..600 {
            let n = 4 + pick(&mut rng, 12);
            let payload: Vec<u8> = (0..n).map(|_| pick(&mut rng, 10) as u8).collect();
            let mut full = payload.clone();
            full.push(luhn_generate(&payload));
            assert!(luhn_check(&full), "generated check digit does not validate");

            // Every single-digit error, at every position.
            for i in 0..full.len() {
                for d in 0..10u8 {
                    if d == full[i] {
                        continue;
                    }
                    let mut bad = full.clone();
                    bad[i] = d;
                    assert!(!luhn_check(&bad), "a single-digit error went undetected");
                }
            }
            // Every adjacent transposition.
            for i in 0..full.len() - 1 {
                if full[i] == full[i + 1] {
                    continue;
                }
                let mut bad = full.clone();
                bad.swap(i, i + 1);
                let pair = (full[i].min(full[i + 1]), full[i].max(full[i + 1]));
                if luhn_check(&bad) {
                    assert_eq!(pair, (0, 9), "an unexpected transposition went undetected");
                    blind += 1;
                } else {
                    caught += 1;
                }
            }
        }
        assert!(caught > 1000, "only {caught} transpositions were tested");
        assert!(blind > 0, "the 09-against-90 blind spot was never exercised");
    }

    /// Verhoeff and Damm have no blind spot: every single-digit error and
    /// every adjacent transposition is caught, which is what the dihedral
    /// group and the anti-symmetric quasigroup buy over arithmetic modulo
    /// ten.
    #[test]
    fn verhoeff_and_damm_catch_every_single_error_and_transposition() {
        assert!(verhoeff_check(&[2, 3, 6, 3]));
        assert_eq!(verhoeff_generate(&[2, 3, 6]), 3);
        assert!(damm_check(&[5, 7, 2, 4]));
        assert_eq!(damm_generate(&[5, 7, 2]), 4);

        let mut rng = Rng::new(0x_5E1F);
        let mut transpositions = 0;
        for _ in 0..400 {
            let n = 3 + pick(&mut rng, 12);
            let payload: Vec<u8> = (0..n).map(|_| pick(&mut rng, 10) as u8).collect();
            for (name, generate, check) in [
                (
                    "Verhoeff",
                    verhoeff_generate as fn(&[u8]) -> u8,
                    verhoeff_check as fn(&[u8]) -> bool,
                ),
                ("Damm", damm_generate, damm_check),
            ] {
                let mut full = payload.clone();
                full.push(generate(&payload));
                assert!(check(&full), "{name} rejects its own check digit");
                for i in 0..full.len() {
                    for d in 0..10u8 {
                        if d == full[i] {
                            continue;
                        }
                        let mut bad = full.clone();
                        bad[i] = d;
                        assert!(!check(&bad), "{name} missed a single-digit error");
                    }
                }
                for i in 0..full.len() - 1 {
                    if full[i] == full[i + 1] {
                        continue;
                    }
                    let mut bad = full.clone();
                    bad.swap(i, i + 1);
                    assert!(!check(&bad), "{name} missed a transposition");
                    transpositions += 1;
                }
            }
        }
        assert!(transpositions > 2000, "only {transpositions} transpositions were tested");
    }

    /// The two ISBN schemes, and the difference a prime modulus makes.
    ///
    /// ISBN-10 works modulo eleven and catches every transposition, at the
    /// cost of a check digit that is sometimes ten. ISBN-13 works modulo ten
    /// and never needs an `X`, and in exchange misses transpositions of
    /// adjacent digits differing by five -- which this test finds rather than
    /// assumes.
    #[test]
    fn isbn_checks_and_the_price_of_a_composite_modulus() {
        assert!(isbn10_check(&[0, 3, 0, 6, 4, 0, 6, 1, 5, 2]));
        assert!(isbn10_check(&[0, 8, 0, 4, 4, 2, 9, 5, 7, 10]), "a check digit of X");
        assert!(!isbn10_check(&[0, 3, 0, 6, 4, 0, 6, 1, 5, 3]));
        assert!(isbn13_check(&[9, 7, 8, 0, 3, 0, 6, 4, 0, 6, 1, 5, 7]));
        assert!(!isbn13_check(&[9, 7, 8, 0, 3, 0, 6, 4, 0, 6, 1, 5, 8]));

        let mut rng = Rng::new(0x_15B4);
        let mut ten_missed = 0;
        let mut thirteen_missed_by_five = 0;
        let mut thirteen_missed_otherwise = 0;
        for _ in 0..2000 {
            // A valid ISBN-10: choose nine digits and solve for the tenth.
            let body: Vec<u8> = (0..9).map(|_| pick(&mut rng, 10) as u8).collect();
            let weighted: u32 =
                body.iter().enumerate().map(|(i, &d)| (10 - i as u32) * u32::from(d)).sum();
            let mut ten = body.clone();
            ten.push(((11 - weighted % 11) % 11) as u8);
            assert!(isbn10_check(&ten));
            for i in 0..9 {
                if ten[i] == ten[i + 1] || ten[i + 1] > 9 {
                    continue;
                }
                let mut bad = ten.clone();
                bad.swap(i, i + 1);
                if isbn10_check(&bad) {
                    ten_missed += 1;
                }
            }

            // A valid ISBN-13 the same way.
            let body: Vec<u8> = (0..12).map(|_| pick(&mut rng, 10) as u8).collect();
            let weighted: u32 = body
                .iter()
                .enumerate()
                .map(|(i, &d)| if i % 2 == 0 { u32::from(d) } else { 3 * u32::from(d) })
                .sum();
            let mut thirteen = body.clone();
            thirteen.push(((10 - weighted % 10) % 10) as u8);
            assert!(isbn13_check(&thirteen));
            for i in 0..12 {
                if thirteen[i] == thirteen[i + 1] {
                    continue;
                }
                let mut bad = thirteen.clone();
                bad.swap(i, i + 1);
                if isbn13_check(&bad) {
                    let gap = thirteen[i].abs_diff(thirteen[i + 1]);
                    if gap == 5 {
                        thirteen_missed_by_five += 1;
                    } else {
                        thirteen_missed_otherwise += 1;
                    }
                }
            }
        }
        assert_eq!(ten_missed, 0, "ISBN-10 missed {ten_missed} transpositions");
        assert!(thirteen_missed_by_five > 0, "the ISBN-13 blind spot was never exercised");
        assert_eq!(
            thirteen_missed_otherwise, 0,
            "ISBN-13 missed a transposition of digits not differing by five"
        );
    }

    /// Hamming distance is a metric, and the byte form agrees with the bit
    /// form.
    #[test]
    fn hamming_distance_is_a_metric() {
        let mut rng = Rng::new(0x_4A33);
        for _ in 0..3000 {
            let (a, b, c) = (rng.next_u64(), rng.next_u64(), rng.next_u64());
            assert_eq!(hamming_distance_bits(a, a), 0);
            assert_eq!(hamming_distance_bits(a, b) == 0, a == b);
            assert_eq!(hamming_distance_bits(a, b), hamming_distance_bits(b, a));
            assert!(
                hamming_distance_bits(a, c)
                    <= hamming_distance_bits(a, b) + hamming_distance_bits(b, c),
                "the triangle inequality fails"
            );
            // Translation invariance, which is what makes a linear code's
            // minimum distance equal to its minimum non-zero weight.
            let t = rng.next_u64();
            assert_eq!(hamming_distance_bits(a ^ t, b ^ t), hamming_distance_bits(a, b));
        }
        assert_eq!(hamming_distance_bytes(b"karolin", b"kathrin"), Some(9));
        assert_eq!(hamming_distance_bytes(b"abc", b"abcd"), None);
        for _ in 0..500 {
            let x = rng.next_u64();
            let y = rng.next_u64();
            assert_eq!(
                hamming_distance_bytes(&x.to_le_bytes(), &y.to_le_bytes()),
                Some(hamming_distance_bits(x, y))
            );
        }
    }
}
