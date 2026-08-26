//! Lossless compression, and the string machinery it is built on.
//!
//! Every method here is one of two ideas. *Entropy coding* -- Huffman,
//! Shannon-Fano, arithmetic -- assumes the symbols are drawn independently
//! from a known distribution and spends about `-log2 p` bits on a symbol of
//! probability `p`. It cannot beat the entropy, and Shannon's theorem says
//! nothing can. *Modelling* -- run lengths, LZ77, LZW, the Burrows-Wheeler
//! transform -- changes what the symbols are, so that a stream with obvious
//! structure and high byte entropy becomes one with low entropy that an
//! entropy coder can then finish off. Real compressors are a modelling stage
//! followed by an entropy stage, and the two halves are here separately.
//!
//! The suffix array and its longest-common-prefix array sit underneath: they
//! are what makes the Burrows-Wheeler transform computable in near-linear
//! time, and they answer questions about repetition in their own right.

use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// Bit-level input and output
// ---------------------------------------------------------------------------

/// Packs bits into bytes, most significant bit first.
#[derive(Debug, Default, Clone)]
pub struct BitWriter {
    bytes: Vec<u8>,
    partial: u8,
    filled: u32,
}

impl BitWriter {
    /// An empty writer.
    #[must_use]
    pub fn new() -> Self {
        BitWriter::default()
    }

    /// Appends one bit.
    pub fn push(&mut self, bit: bool) {
        self.partial = (self.partial << 1) | u8::from(bit);
        self.filled += 1;
        if self.filled == 8 {
            self.bytes.push(self.partial);
            self.partial = 0;
            self.filled = 0;
        }
    }

    /// Appends the low `len` bits of `code`, most significant first.
    pub fn push_bits(&mut self, code: u64, len: u8) {
        for i in (0..len).rev() {
            self.push(code >> i & 1 == 1);
        }
    }

    /// How many bits have been written.
    #[must_use]
    pub fn bit_len(&self) -> usize {
        self.bytes.len() * 8 + self.filled as usize
    }

    /// The bytes, with the last one padded with zeros.
    #[must_use]
    pub fn finish(mut self) -> Vec<u8> {
        if self.filled > 0 {
            self.bytes.push(self.partial << (8 - self.filled));
        }
        self.bytes
    }
}

/// Reads bits from bytes, most significant bit first.
#[derive(Debug, Clone)]
pub struct BitReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> BitReader<'a> {
    /// A reader over the given bytes.
    #[must_use]
    pub fn new(bytes: &'a [u8]) -> Self {
        BitReader { bytes, pos: 0 }
    }

    /// The next bit, or `false` once the input runs out.
    ///
    /// Running off the end is not an error: an arithmetic decoder needs to
    /// keep shifting after the last real bit, and zeros are the right thing
    /// to feed it.
    pub fn next_bit(&mut self) -> bool {
        let byte = self.pos / 8;
        let out = byte < self.bytes.len() && self.bytes[byte] >> (7 - self.pos % 8) & 1 == 1;
        self.pos += 1;
        out
    }
}

// ---------------------------------------------------------------------------
// Huffman and Shannon-Fano
// ---------------------------------------------------------------------------

/// Optimal prefix code lengths and codewords for the given symbol
/// frequencies, one entry per symbol.
///
/// Returns `(codeword, length)` pairs; a symbol of zero frequency gets
/// `(0, 0)` and must not be encoded. The codes are canonical, so a decoder
/// needs only the lengths.
///
/// Huffman's construction repeatedly merges the two least frequent symbols.
/// It is optimal, and the proof is short: in some optimal code the two rarest
/// symbols are siblings at the greatest depth, so merging them and solving
/// the smaller problem loses nothing. Optimal means no prefix code has a
/// smaller expected length -- not that it reaches the entropy, which it
/// cannot when the probabilities are not powers of two.
///
/// # Panics
/// Panics on an empty frequency table.
#[must_use]
pub fn huffman_build(freqs: &[u64]) -> Vec<(u64, u8)> {
    assert!(!freqs.is_empty(), "a code needs at least one symbol");
    let present: Vec<usize> = (0..freqs.len()).filter(|&i| freqs[i] > 0).collect();
    let mut lengths = vec![0u8; freqs.len()];
    match present.len() {
        0 => return vec![(0, 0); freqs.len()],
        1 => {
            // A single symbol still needs a bit, or the stream has no length.
            lengths[present[0]] = 1;
            return canonical_from_lengths(&lengths);
        }
        _ => {}
    }
    // Nodes: leaves first, then merges. `parent` is enough to recover depth.
    #[derive(Clone, Copy)]
    struct Node {
        weight: u64,
        left: usize,
        right: usize,
    }
    let mut nodes: Vec<Node> =
        present.iter().map(|&i| Node { weight: freqs[i], left: usize::MAX, right: usize::MAX }).collect();
    // A set ordered by (weight, insertion order) is a priority queue with a
    // deterministic tie-break, which keeps the output reproducible.
    let mut live: std::collections::BTreeSet<(u64, usize)> =
        (0..nodes.len()).map(|i| (nodes[i].weight, i)).collect();
    while live.len() > 1 {
        let a = *live.iter().next().expect("non-empty");
        live.remove(&a);
        let b = *live.iter().next().expect("non-empty");
        live.remove(&b);
        let idx = nodes.len();
        nodes.push(Node { weight: a.0 + b.0, left: a.1, right: b.1 });
        live.insert((a.0 + b.0, idx));
    }
    let root = live.iter().next().expect("one node remains").1;
    // Walk down, recording depth. Iterative so a degenerate tree of depth
    // 255 cannot overflow the stack.
    let mut stack = vec![(root, 0u8)];
    while let Some((n, depth)) = stack.pop() {
        if nodes[n].left == usize::MAX {
            lengths[present[n]] = depth.max(1);
        } else {
            stack.push((nodes[n].left, depth + 1));
            stack.push((nodes[n].right, depth + 1));
        }
    }
    canonical_from_lengths(&lengths)
}

/// Canonical codewords for the given code lengths.
///
/// Symbols are ordered by length and then by index, and codewords are
/// assigned in increasing numeric order, doubling at each length increase.
/// Any two prefix codes with the same length multiset compress identically,
/// so a decoder can be handed the lengths alone -- which is why every real
/// format transmits lengths rather than a tree.
///
/// # Panics
/// Panics if the lengths do not satisfy Kraft's inequality, since no prefix
/// code has them.
#[must_use]
pub fn canonical_huffman(lengths: &[u8]) -> Vec<u64> {
    canonical_from_lengths(lengths).into_iter().map(|(c, _)| c).collect()
}

fn canonical_from_lengths(lengths: &[u8]) -> Vec<(u64, u8)> {
    let kraft: f64 = lengths.iter().filter(|&&l| l > 0).map(|&l| 2.0f64.powi(-i32::from(l))).sum();
    assert!(kraft <= 1.0 + 1e-9, "the lengths violate Kraft's inequality");
    let mut order: Vec<usize> = (0..lengths.len()).filter(|&i| lengths[i] > 0).collect();
    order.sort_by_key(|&i| (lengths[i], i));
    let mut out = vec![(0u64, 0u8); lengths.len()];
    let mut code = 0u64;
    let mut prev = 0u8;
    for &i in &order {
        let l = lengths[i];
        code <<= u32::from(l - prev);
        prev = l;
        out[i] = (code, l);
        code += 1;
    }
    out
}

/// The Kraft sum of a set of code lengths: `sum 2^-l`.
///
/// At most one for any prefix code, and exactly one when the code wastes
/// nothing -- which Huffman's always does, since a tree with an only child
/// could shorten that child by a bit.
#[must_use]
pub fn kraft_sum(lengths: &[u8]) -> f64 {
    lengths.iter().filter(|&&l| l > 0).map(|&l| 2.0f64.powi(-i32::from(l))).sum()
}

/// Huffman-codes a byte string, returning the packed bits, the code table,
/// and the number of bits that matter.
#[must_use]
pub fn huffman_encode(data: &[u8]) -> (Vec<u8>, Vec<(u64, u8)>, usize) {
    let mut freqs = vec![0u64; 256];
    for &b in data {
        freqs[b as usize] += 1;
    }
    let table = huffman_build(&freqs);
    let mut w = BitWriter::new();
    for &b in data {
        let (code, len) = table[b as usize];
        w.push_bits(code, len);
    }
    let bits = w.bit_len();
    (w.finish(), table, bits)
}

/// Decodes `n` symbols from a Huffman-coded bit string.
///
/// # Panics
/// Panics if the bits do not spell out `n` valid codewords.
#[must_use]
pub fn huffman_decode(bits: &[u8], table: &[(u64, u8)], n: usize) -> Vec<u8> {
    // Codeword to symbol, keyed by length so a walk can stop as soon as it
    // matches -- which a prefix code guarantees is unambiguous.
    let mut lookup: BTreeMap<(u8, u64), usize> = BTreeMap::new();
    for (sym, &(code, len)) in table.iter().enumerate() {
        if len > 0 {
            lookup.insert((len, code), sym);
        }
    }
    let mut r = BitReader::new(bits);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let mut code = 0u64;
        let mut len = 0u8;
        loop {
            code = (code << 1) | u64::from(r.next_bit());
            len += 1;
            assert!(len <= 64, "the bits do not spell a codeword");
            if let Some(&sym) = lookup.get(&(len, code)) {
                out.push(sym as u8);
                break;
            }
        }
    }
    out
}

/// Shannon-Fano coding: split the frequency-sorted symbols into two halves of
/// as nearly equal weight as possible, and recurse.
///
/// The older construction, and never better than Huffman: it decides the top
/// of the tree first and cannot revise, while Huffman builds from the leaves
/// and so is optimal. The gap is usually small and occasionally a whole bit
/// per symbol.
///
/// # Panics
/// Panics on an empty frequency table.
#[must_use]
pub fn shannon_fano(freqs: &[u64]) -> Vec<(u64, u8)> {
    assert!(!freqs.is_empty(), "a code needs at least one symbol");
    let mut present: Vec<usize> = (0..freqs.len()).filter(|&i| freqs[i] > 0).collect();
    let mut lengths = vec![0u8; freqs.len()];
    if present.len() == 1 {
        lengths[present[0]] = 1;
        return canonical_from_lengths(&lengths);
    }
    if present.is_empty() {
        return vec![(0, 0); freqs.len()];
    }
    present.sort_by_key(|&i| (std::cmp::Reverse(freqs[i]), i));
    // Each split adds a bit to everything below it.
    let mut work = vec![(0usize, present.len())];
    while let Some((lo, hi)) = work.pop() {
        if hi - lo < 2 {
            continue;
        }
        let total: u64 = present[lo..hi].iter().map(|&i| freqs[i]).sum();
        // The split point where the running sum first reaches half.
        let mut running = 0u64;
        let mut split = lo + 1;
        for j in lo..hi - 1 {
            running += freqs[present[j]];
            if 2 * running >= total {
                split = j + 1;
                break;
            }
            split = j + 2;
        }
        for j in lo..hi {
            lengths[present[j]] += 1;
        }
        work.push((lo, split));
        work.push((split, hi));
    }
    canonical_from_lengths(&lengths)
}

/// The average code length of a prefix code against the given frequencies, in
/// bits per symbol.
#[must_use]
pub fn average_code_length(table: &[(u64, u8)], freqs: &[u64]) -> f64 {
    let total: u64 = freqs.iter().sum();
    if total == 0 {
        return 0.0;
    }
    let sum: u64 =
        freqs.iter().enumerate().map(|(i, &f)| f * u64::from(table[i].1)).sum();
    sum as f64 / total as f64
}

// ---------------------------------------------------------------------------
// Arithmetic coding
// ---------------------------------------------------------------------------

const AC_BITS: u32 = 32;
const AC_TOP: u64 = 1 << AC_BITS;
const AC_HALF: u64 = AC_TOP / 2;
const AC_QUARTER: u64 = AC_TOP / 4;
const AC_THREE_QUARTER: u64 = 3 * AC_QUARTER;

/// Cumulative frequencies, and the total.
fn cumulative(model: &[u64]) -> (Vec<u64>, u64) {
    let mut cum = Vec::with_capacity(model.len() + 1);
    let mut sum = 0u64;
    cum.push(0);
    for &c in model {
        sum += c;
        cum.push(sum);
    }
    (cum, sum)
}

/// Arithmetic coding against a fixed model of symbol frequencies.
///
/// Where a prefix code must spend a whole number of bits on every symbol,
/// arithmetic coding narrows a single interval by a factor of each symbol's
/// probability and writes out one number identifying it. The cost of a
/// message is therefore `-log2` of its probability to within two bits *in
/// total*, not per symbol, which is what makes it beat Huffman whenever some
/// symbol is much more likely than a half.
///
/// # Panics
/// Panics unless the model has one non-negative count per symbol value, the
/// total is between one and 65536, and every byte that occurs has a positive
/// count.
#[must_use]
pub fn arithmetic_encode(data: &[u8], model: &[u64]) -> Vec<u8> {
    assert_eq!(model.len(), 256, "the model needs one count per byte value");
    let (cum, total) = cumulative(model);
    assert!(total > 0 && total <= 1 << 16, "the model's total must lie in 1..=65536");
    assert!(data.iter().all(|&b| model[b as usize] > 0), "a byte has zero probability");
    let mut low = 0u64;
    let mut high = AC_TOP - 1;
    let mut pending = 0u64;
    let mut w = BitWriter::new();
    let emit = |w: &mut BitWriter, bit: bool, pending: &mut u64| {
        w.push(bit);
        for _ in 0..*pending {
            w.push(!bit);
        }
        *pending = 0;
    };
    for &b in data {
        let range = high - low + 1;
        let s = b as usize;
        high = low + range * cum[s + 1] / total - 1;
        low += range * cum[s] / total;
        loop {
            if high < AC_HALF {
                emit(&mut w, false, &mut pending);
            } else if low >= AC_HALF {
                emit(&mut w, true, &mut pending);
                low -= AC_HALF;
                high -= AC_HALF;
            } else if low >= AC_QUARTER && high < AC_THREE_QUARTER {
                // The interval straddles the midpoint but sits inside the
                // middle half: the next bit is not yet decided, so remember
                // that one bit of the opposite kind will follow whichever it
                // turns out to be.
                pending += 1;
                low -= AC_QUARTER;
                high -= AC_QUARTER;
            } else {
                break;
            }
            low <<= 1;
            high = (high << 1) | 1;
        }
    }
    pending += 1;
    if low < AC_QUARTER {
        emit(&mut w, false, &mut pending);
    } else {
        emit(&mut w, true, &mut pending);
    }
    w.finish()
}

/// Decodes `n` symbols from an arithmetic-coded stream.
///
/// # Panics
/// Panics under the same conditions as [`arithmetic_encode`].
#[must_use]
pub fn arithmetic_decode(bits: &[u8], model: &[u64], n: usize) -> Vec<u8> {
    assert_eq!(model.len(), 256, "the model needs one count per byte value");
    let (cum, total) = cumulative(model);
    assert!(total > 0 && total <= 1 << 16, "the model's total must lie in 1..=65536");
    let mut r = BitReader::new(bits);
    let mut value = 0u64;
    for _ in 0..AC_BITS {
        value = (value << 1) | u64::from(r.next_bit());
    }
    let mut low = 0u64;
    let mut high = AC_TOP - 1;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let range = high - low + 1;
        // Where the value sits in the current interval, scaled to the model.
        let scaled = ((value - low + 1) * total - 1) / range;
        let s = cum.partition_point(|&c| c <= scaled) - 1;
        out.push(s as u8);
        high = low + range * cum[s + 1] / total - 1;
        low += range * cum[s] / total;
        loop {
            if high < AC_HALF {
                // nothing to strip
            } else if low >= AC_HALF {
                low -= AC_HALF;
                high -= AC_HALF;
                value -= AC_HALF;
            } else if low >= AC_QUARTER && high < AC_THREE_QUARTER {
                low -= AC_QUARTER;
                high -= AC_QUARTER;
                value -= AC_QUARTER;
            } else {
                break;
            }
            low <<= 1;
            high = (high << 1) | 1;
            value = (value << 1) | u64::from(r.next_bit());
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Dictionary methods
// ---------------------------------------------------------------------------

/// One LZ77 token: a back reference and the literal that follows it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lz77Token {
    /// How far back the match starts, or zero for none.
    pub offset: usize,
    /// How long the match is.
    pub length: usize,
    /// The byte that broke the match, or that stands alone.
    pub next: u8,
}

/// LZ77: replace repeats with references to earlier text.
///
/// The window bounds how far back a reference may point and the lookahead how
/// long a match may be. A match is allowed to run past its own start -- an
/// offset of one with length twenty is a run of twenty identical bytes -- and
/// the decompressor copying one byte at a time handles that for free, which
/// is why run-length encoding falls out of LZ77 rather than needing to be
/// added to it.
///
/// # Panics
/// Panics if the window or lookahead is zero.
#[must_use]
pub fn lz77_compress(data: &[u8], window: usize, lookahead: usize) -> Vec<Lz77Token> {
    assert!(window > 0 && lookahead > 0, "the window and lookahead must be positive");
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < data.len() {
        let start = i.saturating_sub(window);
        let mut best = (0usize, 0usize);
        for j in start..i {
            let mut l = 0usize;
            while l < lookahead && i + l < data.len() && data[j + l] == data[i + l] {
                l += 1;
            }
            if l > best.1 {
                best = (i - j, l);
            }
        }
        // The literal that follows the match, or the byte itself when there
        // was none. A match running to the very end has no follower, so it is
        // shortened by one to leave a literal.
        let (offset, mut length) = best;
        if i + length >= data.len() && length > 0 {
            length -= 1;
        }
        let next = data[i + length];
        out.push(Lz77Token { offset: if length == 0 { 0 } else { offset }, length, next });
        i += length + 1;
    }
    out
}

/// Rebuilds the original from LZ77 tokens.
///
/// # Panics
/// Panics if a token points further back than the output so far.
#[must_use]
pub fn lz77_decompress(tokens: &[Lz77Token]) -> Vec<u8> {
    let mut out = Vec::new();
    for t in tokens {
        if t.length > 0 {
            assert!(t.offset <= out.len(), "a back reference points before the start");
            let start = out.len() - t.offset;
            for k in 0..t.length {
                out.push(out[start + k]);
            }
        }
        out.push(t.next);
    }
    out
}

/// LZW: build a dictionary of every phrase seen plus one byte, and emit
/// dictionary indices.
///
/// The decoder rebuilds the same dictionary from the same output, so nothing
/// has to be transmitted with the data -- which is what made it practical for
/// modems and printers with no memory to spare.
#[must_use]
pub fn lzw_compress(data: &[u8]) -> Vec<u16> {
    let mut dict: BTreeMap<Vec<u8>, u16> =
        (0..256u16).map(|i| (vec![i as u8], i)).collect();
    let mut next_code = 256u16;
    let mut out = Vec::new();
    let mut current: Vec<u8> = Vec::new();
    for &b in data {
        let mut extended = current.clone();
        extended.push(b);
        if dict.contains_key(&extended) {
            current = extended;
        } else {
            out.push(dict[&current]);
            if next_code < u16::MAX {
                dict.insert(extended, next_code);
                next_code += 1;
            }
            current = vec![b];
        }
    }
    if !current.is_empty() {
        out.push(dict[&current]);
    }
    out
}

/// Rebuilds the original from LZW codes.
///
/// # Panics
/// Panics on a code the dictionary cannot yet contain.
#[must_use]
pub fn lzw_decompress(codes: &[u16]) -> Vec<u8> {
    if codes.is_empty() {
        return Vec::new();
    }
    let mut dict: Vec<Vec<u8>> = (0..256).map(|i| vec![i as u8]).collect();
    let mut out = dict[codes[0] as usize].clone();
    let mut previous = out.clone();
    for &code in &codes[1..] {
        let entry = if (code as usize) < dict.len() {
            dict[code as usize].clone()
        } else {
            // The encoder can emit a code it has only just defined, when a
            // phrase is immediately followed by itself. The decoder is one
            // step behind and reconstructs it from what it has.
            assert_eq!(code as usize, dict.len(), "a code the dictionary cannot hold");
            let mut e = previous.clone();
            e.push(previous[0]);
            e
        };
        out.extend_from_slice(&entry);
        if dict.len() < u16::MAX as usize {
            let mut new = previous.clone();
            new.push(entry[0]);
            dict.push(new);
        }
        previous = entry;
    }
    out
}

/// Run-length encoding in the PackBits scheme.
///
/// A control byte below 128 means "the next `n + 1` bytes are literal"; one
/// at or above means "repeat the next byte `257 - n` times". Incompressible
/// data grows by one byte in every 128, which is the price of never needing
/// an escape character.
#[must_use]
pub fn rle_compress(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < data.len() {
        // How long the run starting here is.
        let mut run = 1usize;
        while i + run < data.len() && data[i + run] == data[i] && run < 128 {
            run += 1;
        }
        if run >= 2 {
            out.push((257 - run) as u8);
            out.push(data[i]);
            i += run;
        } else {
            // Gather literals until a run of three or more begins, since a
            // run of two barely pays for its own control byte.
            let start = i;
            while i < data.len() && i - start < 128 {
                let mut ahead = 1usize;
                while i + ahead < data.len() && data[i + ahead] == data[i] {
                    ahead += 1;
                    if ahead >= 3 {
                        break;
                    }
                }
                if ahead >= 3 {
                    break;
                }
                i += 1;
            }
            out.push((i - start - 1) as u8);
            out.extend_from_slice(&data[start..i]);
        }
    }
    out
}

/// Rebuilds the original from PackBits run-length encoding.
///
/// # Panics
/// Panics if the stream is truncated part way through a run or literal.
#[must_use]
pub fn rle_decompress(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < data.len() {
        let n = data[i];
        i += 1;
        if n < 128 {
            let count = n as usize + 1;
            assert!(i + count <= data.len(), "the literal run is truncated");
            out.extend_from_slice(&data[i..i + count]);
            i += count;
        } else {
            assert!(i < data.len(), "the repeat has no byte to repeat");
            let count = 257 - n as usize;
            out.extend(std::iter::repeat_n(data[i], count));
            i += 1;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Suffix arrays and the Burrows-Wheeler transform
// ---------------------------------------------------------------------------

/// The suffix array: the starting positions of the suffixes, in the order
/// those suffixes sort.
///
/// Built by prefix doubling. After round `k` the suffixes are sorted by their
/// first `2^k` characters, and the next round sorts by pairs of the ranks
/// already computed -- so each round doubles the prefix length examined and
/// `log n` rounds settle it. Not the linear-time construction, but the
/// simplest one whose correctness is visible.
#[must_use]
pub fn suffix_array(data: &[u8]) -> Vec<usize> {
    let n = data.len();
    if n == 0 {
        return Vec::new();
    }
    let mut sa: Vec<usize> = (0..n).collect();
    let mut rank: Vec<i64> = data.iter().map(|&b| i64::from(b)).collect();
    let mut tmp = vec![0i64; n];
    let mut k = 1usize;
    while k < n {
        let key = |i: usize, rank: &Vec<i64>| -> (i64, i64) {
            (rank[i], if i + k < n { rank[i + k] } else { -1 })
        };
        sa.sort_by_key(|&i| key(i, &rank));
        tmp[sa[0]] = 0;
        for w in 1..n {
            let prev = key(sa[w - 1], &rank);
            let cur = key(sa[w], &rank);
            tmp[sa[w]] = tmp[sa[w - 1]] + i64::from(cur != prev);
        }
        rank.copy_from_slice(&tmp);
        if rank[sa[n - 1]] == (n - 1) as i64 {
            break;
        }
        k *= 2;
    }
    sa
}

/// The longest common prefix of each adjacent pair in the suffix array, by
/// Kasai's algorithm.
///
/// `lcp[i]` is the overlap between the suffixes at `sa[i - 1]` and `sa[i]`,
/// with `lcp[0]` zero. Kasai's insight is that walking the suffixes in
/// *text* order lets the previous answer be reused: dropping the first
/// character of a suffix shortens its overlap with its neighbour by at most
/// one, so the total work is linear rather than quadratic.
///
/// # Panics
/// Panics unless the suffix array matches the data's length.
#[must_use]
pub fn lcp_array(data: &[u8], sa: &[usize]) -> Vec<usize> {
    let n = data.len();
    assert_eq!(sa.len(), n, "the suffix array must match the data");
    if n == 0 {
        return Vec::new();
    }
    let mut inv = vec![0usize; n];
    for (i, &s) in sa.iter().enumerate() {
        inv[s] = i;
    }
    let mut lcp = vec![0usize; n];
    let mut h = 0usize;
    for i in 0..n {
        if inv[i] > 0 {
            let j = sa[inv[i] - 1];
            while i + h < n && j + h < n && data[i + h] == data[j + h] {
                h += 1;
            }
            lcp[inv[i]] = h;
            h = h.saturating_sub(1);
        } else {
            h = 0;
        }
    }
    lcp
}

/// The longest substring that occurs at least twice, as `(start, length)`.
///
/// The largest entry of the longest-common-prefix array, because two
/// occurrences of the same substring are two suffixes sharing that prefix,
/// and suffixes sharing a long prefix are adjacent in the suffix array.
/// Length zero when nothing repeats.
#[must_use]
pub fn longest_repeated_substring(data: &[u8]) -> (usize, usize) {
    let sa = suffix_array(data);
    let lcp = lcp_array(data, &sa);
    let mut best = (0usize, 0usize);
    for i in 1..lcp.len() {
        if lcp[i] > best.1 {
            best = (sa[i], lcp[i]);
        }
    }
    best
}

/// The Burrows-Wheeler transform: the last column of the sorted rotations,
/// and which row the original occupies.
///
/// The transform is reversible and sorts nothing about the data itself -- it
/// is a permutation of the bytes. What it does is bring together the bytes
/// that precede similar contexts, so English text comes out in long runs of
/// the same letter, and a run-length or move-to-front stage that could do
/// nothing with the original then has plenty to work with.
#[must_use]
pub fn bwt(data: &[u8]) -> (Vec<u8>, usize) {
    let n = data.len();
    if n == 0 {
        return (Vec::new(), 0);
    }
    // Prefix doubling on the *cyclic* string sorts rotations rather than
    // suffixes, which is what the transform is defined on.
    let mut sa: Vec<usize> = (0..n).collect();
    let mut rank: Vec<i64> = data.iter().map(|&b| i64::from(b)).collect();
    let mut tmp = vec![0i64; n];
    let mut k = 1usize;
    while k < n {
        let key = |i: usize, rank: &Vec<i64>| -> (i64, i64) { (rank[i], rank[(i + k) % n]) };
        sa.sort_by_key(|&i| (key(i, &rank), i));
        tmp[sa[0]] = 0;
        for w in 1..n {
            let prev = key(sa[w - 1], &rank);
            let cur = key(sa[w], &rank);
            tmp[sa[w]] = tmp[sa[w - 1]] + i64::from(cur != prev);
        }
        rank.copy_from_slice(&tmp);
        k *= 2;
    }
    let last: Vec<u8> = sa.iter().map(|&i| data[(i + n - 1) % n]).collect();
    let idx = sa.iter().position(|&i| i == 0).expect("the original rotation is present");
    (last, idx)
}

/// Inverts the Burrows-Wheeler transform.
///
/// The last column plus the row index is enough, because sorting the last
/// column gives the first, and the `i`-th occurrence of a byte in the last
/// column is the `i`-th in the first -- rotations sharing a first byte stay
/// in the same relative order. That correspondence is the whole inverse.
///
/// # Panics
/// Panics if the index is outside the data.
#[must_use]
pub fn ibwt(data: &[u8], idx: usize) -> Vec<u8> {
    let n = data.len();
    if n == 0 {
        return Vec::new();
    }
    assert!(idx < n, "the row index is outside the data");
    let mut count = [0usize; 256];
    for &c in data {
        count[c as usize] += 1;
    }
    let mut start = [0usize; 256];
    let mut sum = 0usize;
    for c in 0..256 {
        start[c] = sum;
        sum += count[c];
    }
    // The mapping from a row to the row whose rotation starts one earlier.
    let mut lf = vec![0usize; n];
    let mut occ = [0usize; 256];
    for (i, &c) in data.iter().enumerate() {
        lf[i] = start[c as usize] + occ[c as usize];
        occ[c as usize] += 1;
    }
    let mut out = Vec::with_capacity(n);
    let mut p = idx;
    for _ in 0..n {
        out.push(data[p]);
        p = lf[p];
    }
    out.reverse();
    out
}

/// Move-to-front coding: emit each byte's position in a list, then move it to
/// the front.
///
/// It turns locality into small numbers. A stretch using only a few distinct
/// bytes -- which is what the Burrows-Wheeler transform produces -- becomes a
/// stretch of values near zero, and a stretch of one repeated byte becomes a
/// run of zeros, which an entropy coder or a run-length stage can then
/// exploit.
#[must_use]
pub fn mtf_encode(data: &[u8]) -> Vec<u8> {
    let mut list: Vec<u8> = (0..=255).collect();
    data.iter()
        .map(|&b| {
            let pos = list.iter().position(|&x| x == b).expect("every byte is in the list");
            let v = list.remove(pos);
            list.insert(0, v);
            pos as u8
        })
        .collect()
}

/// Inverts move-to-front coding.
#[must_use]
pub fn mtf_decode(data: &[u8]) -> Vec<u8> {
    let mut list: Vec<u8> = (0..=255).collect();
    data.iter()
        .map(|&p| {
            let v = list.remove(p as usize);
            list.insert(0, v);
            v
        })
        .collect()
}

/// Differences between consecutive bytes, modulo 256, with the first byte
/// kept as it is.
///
/// Worth doing when the data is a slowly varying signal: a smooth ramp has
/// high byte entropy and near-zero difference entropy.
#[must_use]
pub fn delta_encode(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut prev = 0u8;
    for &b in data {
        out.push(b.wrapping_sub(prev));
        prev = b;
    }
    out
}

/// Inverts delta coding.
#[must_use]
pub fn delta_decode(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut prev = 0u8;
    for &d in data {
        prev = prev.wrapping_add(d);
        out.push(prev);
    }
    out
}

// ---------------------------------------------------------------------------
// Measures
// ---------------------------------------------------------------------------

/// The Shannon entropy of the byte histogram, in bits per byte.
///
/// The floor for any coder that treats the bytes as independent draws.
/// Between zero, for a constant stream, and eight, for a uniform one. It is
/// not a floor for compression in general: a stream of a million alternating
/// bytes has an entropy of one bit per byte and compresses to nothing, since
/// the bytes are not independent.
#[must_use]
pub fn entropy_bytes(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut counts = [0u64; 256];
    for &b in data {
        counts[b as usize] += 1;
    }
    let n = data.len() as f64;
    counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / n;
            -p * p.log2()
        })
        .sum()
}

/// The size in bytes that the byte entropy allows, which no memoryless coder
/// can beat.
#[must_use]
pub fn compression_bound(data: &[u8]) -> f64 {
    entropy_bytes(data) * data.len() as f64 / 8.0
}

/// The size the module's own best pipeline achieves, as a stand-in for the
/// incompressible content of the data.
///
/// Kolmogorov complexity is not computable, and this is not an approximation
/// to it in any rigorous sense -- it is an upper bound that happens to behave
/// sensibly, which is what the practical literature uses it for. The pipeline
/// is Burrows-Wheeler, then move-to-front, then run lengths, then Huffman:
/// each stage exposes structure the next can spend.
#[must_use]
pub fn kolmogorov_estimate_by_compressors(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let (t, _) = bwt(data);
    let piped = rle_compress(&mtf_encode(&t));
    let (_, _, bits) = huffman_encode(&piped);
    // The direct route, for data the pipeline does not suit.
    let (_, _, plain) = huffman_encode(data);
    (bits.min(plain) as f64 / 8.0).min(data.len() as f64)
}

/// The normalized compression distance between two byte strings.
///
/// `(C(ab) - min(C(a), C(b))) / max(C(a), C(b))`: if knowing `a` makes `b`
/// cheap to describe, they are close. Near zero for identical inputs and near
/// one for unrelated ones, and it needs no notion of what the data means,
/// which is why it gets used on genomes and on music alike.
#[must_use]
pub fn normalized_compression_distance(a: &[u8], b: &[u8]) -> f64 {
    let ca = kolmogorov_estimate_by_compressors(a);
    let cb = kolmogorov_estimate_by_compressors(b);
    let mut joined = a.to_vec();
    joined.extend_from_slice(b);
    let cab = kolmogorov_estimate_by_compressors(&joined);
    let denom = ca.max(cb);
    if denom <= 0.0 {
        return 0.0;
    }
    ((cab - ca.min(cb)) / denom).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monte_carlo::Rng;

    fn pick(rng: &mut Rng, n: usize) -> usize {
        ((u128::from(rng.next_u64()) * n as u128) >> 64) as usize
    }

    fn histogram(data: &[u8]) -> Vec<u64> {
        let mut f = vec![0u64; 256];
        for &b in data {
            f[b as usize] += 1;
        }
        f
    }

    /// A spread of inputs with different structure, since a compressor that
    /// works on one kind often fails on another.
    fn corpus(rng: &mut Rng) -> Vec<(&'static str, Vec<u8>)> {
        let mut out: Vec<(&'static str, Vec<u8>)> = vec![
            ("empty", Vec::new()),
            ("one byte", vec![42]),
            ("constant", vec![7u8; 500]),
            ("two alternating", (0..500).map(|i| if i % 2 == 0 { 1 } else { 2 }).collect()),
            ("ramp", (0..500).map(|i| (i % 256) as u8).collect()),
            (
                "english-ish",
                b"the quick brown fox jumps over the lazy dog. \
                  the quick brown fox jumps over the lazy dog. \
                  she sells sea shells by the sea shore, and the shells she sells are sea shells."
                    .to_vec(),
            ),
            ("runs", (0..80).flat_map(|i| vec![(i % 7) as u8; 1 + i % 11]).collect()),
        ];
        out.push(("uniform random", (0..600).map(|_| (rng.next_u64() & 0xFF) as u8).collect()));
        // A skewed source: one byte dominates, which is where arithmetic
        // coding pulls away from Huffman.
        out.push((
            "skewed",
            (0..800)
                .map(|_| if rng.next_f64() < 0.9 { 0u8 } else { (1 + pick(rng, 5)) as u8 })
                .collect(),
        ));
        out
    }

    /// Huffman's code is a prefix code that wastes nothing, sits within a bit
    /// of the entropy, and is optimal -- checked against an exhaustive search
    /// over every length assignment a prefix code could have.
    #[test]
    fn huffman_is_a_tight_prefix_code_and_optimal() {
        let mut rng = Rng::new(0x_4FF7);
        for _ in 0..200 {
            let alphabet = 1 + pick(&mut rng, 6);
            let freqs: Vec<u64> = (0..alphabet).map(|_| 1 + pick(&mut rng, 40) as u64).collect();
            let table = huffman_build(&freqs);
            let lengths: Vec<u8> = table.iter().map(|&(_, l)| l).collect();

            // Kraft holds with equality once there is more than one symbol:
            // an incomplete tree could shorten a codeword, so an optimal
            // code never leaves slack. A lone symbol is the exception --
            // it still needs a bit, and half the space goes unused, because
            // a zero-length codeword would leave the message with no length.
            let kraft = kraft_sum(&lengths);
            if alphabet == 1 {
                assert!((kraft - 0.5).abs() < 1e-9, "a lone symbol should take one bit");
            } else {
                assert!((kraft - 1.0).abs() < 1e-9, "the code leaves slack");
            }
            // Prefix-free: no codeword begins another.
            for i in 0..alphabet {
                for j in 0..alphabet {
                    if i == j || lengths[i] == 0 || lengths[j] == 0 || lengths[i] > lengths[j] {
                        continue;
                    }
                    let shift = lengths[j] - lengths[i];
                    assert_ne!(
                        table[j].0 >> shift,
                        table[i].0,
                        "codeword {i} is a prefix of {j}"
                    );
                }
            }

            let total: u64 = freqs.iter().sum();
            let entropy: f64 = freqs
                .iter()
                .filter(|&&f| f > 0)
                .map(|&f| {
                    let p = f as f64 / total as f64;
                    -p * p.log2()
                })
                .sum();
            let avg = average_code_length(&table, &freqs);
            // Shannon's bound, both halves of it.
            assert!(avg >= entropy - 1e-9, "the code beat the entropy: {avg} against {entropy}");
            assert!(avg < entropy + 1.0 + 1e-9, "the code is more than a bit over the entropy");

            // Optimality, by exhaustion: no length assignment satisfying
            // Kraft does better. This is the statement Huffman's algorithm
            // exists to make, and nothing weaker distinguishes it from any
            // other prefix code.
            if (2..=5).contains(&alphabet) {
                let max_len = 6u8;
                let mut best = f64::INFINITY;
                let mut assignment = vec![1u8; alphabet];
                loop {
                    if kraft_sum(&assignment) <= 1.0 + 1e-12 {
                        let cost: u64 = freqs
                            .iter()
                            .zip(&assignment)
                            .map(|(&f, &l)| f * u64::from(l))
                            .sum();
                        best = best.min(cost as f64 / total as f64);
                    }
                    let mut k = 0;
                    while k < alphabet {
                        assignment[k] += 1;
                        if assignment[k] <= max_len {
                            break;
                        }
                        assignment[k] = 1;
                        k += 1;
                    }
                    if k == alphabet {
                        break;
                    }
                }
                assert!(
                    (avg - best).abs() < 1e-9,
                    "Huffman gave {avg} where {best} was available"
                );
            }
        }
        // A single symbol still needs one bit, or a message has no length.
        let one = huffman_build(&[5]);
        assert_eq!(one[0].1, 1);
    }

    /// Canonical codes are determined by their lengths alone, which is why a
    /// decoder can be handed lengths rather than a tree.
    #[test]
    fn canonical_codes_are_determined_by_their_lengths() {
        let mut rng = Rng::new(0x_C4A0);
        for _ in 0..300 {
            let alphabet = 2 + pick(&mut rng, 10);
            let freqs: Vec<u64> = (0..alphabet).map(|_| 1 + pick(&mut rng, 60) as u64).collect();
            let table = huffman_build(&freqs);
            let lengths: Vec<u8> = table.iter().map(|&(_, l)| l).collect();
            let codes = canonical_huffman(&lengths);
            assert_eq!(codes, table.iter().map(|&(c, _)| c).collect::<Vec<_>>());
            // Sorted by length then index, the codewords increase.
            let mut order: Vec<usize> = (0..alphabet).filter(|&i| lengths[i] > 0).collect();
            order.sort_by_key(|&i| (lengths[i], i));
            for w in order.windows(2) {
                let (a, b) = (w[0], w[1]);
                let shifted = codes[a] << (lengths[b] - lengths[a]);
                assert!(codes[b] > shifted || (lengths[a] == lengths[b] && codes[b] > codes[a]));
            }
        }
        // Lengths that no prefix code could have are refused.
        assert!(std::panic::catch_unwind(|| canonical_huffman(&[1, 1, 1])).is_err());
    }

    /// Huffman and Shannon-Fano both round-trip, and Shannon-Fano is never
    /// the better of the two -- which is the reason Huffman replaced it.
    #[test]
    fn huffman_roundtrips_and_never_loses_to_shannon_fano() {
        let mut rng = Rng::new(0x_5F40);
        let mut strictly_better = 0;
        for (name, data) in corpus(&mut rng) {
            if data.is_empty() {
                continue;
            }
            let (bytes, table, bits) = huffman_encode(&data);
            assert_eq!(bytes.len(), bits.div_ceil(8), "{name}: the packing is the wrong size");
            assert_eq!(huffman_decode(&bytes, &table, data.len()), data, "{name}: roundtrip");

            let freqs = histogram(&data);
            let sf = shannon_fano(&freqs);
            let sf_lengths: Vec<u8> = sf.iter().map(|&(_, l)| l).collect();
            assert!(kraft_sum(&sf_lengths) <= 1.0 + 1e-9, "{name}: Shannon-Fano breaks Kraft");
            let h = average_code_length(&table, &freqs);
            let s = average_code_length(&sf, &freqs);
            assert!(h <= s + 1e-9, "{name}: Shannon-Fano beat Huffman, {s} against {h}");
            if s > h + 1e-9 {
                strictly_better += 1;
            }
            // Shannon-Fano is a real code too, so it must also decode.
            let mut w = BitWriter::new();
            for &b in &data {
                w.push_bits(sf[b as usize].0, sf[b as usize].1);
            }
            assert_eq!(huffman_decode(&w.finish(), &sf, data.len()), data, "{name}: Shannon-Fano");
        }
        assert!(strictly_better > 0, "the two constructions never actually differed");
    }

    /// Arithmetic coding round-trips, lands within a couple of bytes of the
    /// message's own information content, and beats Huffman where a whole bit
    /// per symbol is too coarse a unit.
    #[test]
    fn arithmetic_coding_is_exact_and_beats_huffman_on_skew() {
        let mut rng = Rng::new(0x_A21C);
        let mut beat_huffman = 0;
        for (name, data) in corpus(&mut rng) {
            if data.is_empty() {
                continue;
            }
            let model = histogram(&data);
            let coded = arithmetic_encode(&data, &model);
            assert_eq!(arithmetic_decode(&coded, &model, data.len()), data, "{name}: roundtrip");

            // The ideal cost is minus the log probability of the message
            // under its own model; arithmetic coding pays that plus at most
            // a couple of bits for the whole message.
            let total: u64 = model.iter().sum();
            let ideal_bits: f64 = data
                .iter()
                .map(|&b| -(model[b as usize] as f64 / total as f64).log2())
                .sum();
            let actual_bits = coded.len() as f64 * 8.0;
            assert!(
                actual_bits >= ideal_bits - 1e-6,
                "{name}: beat the entropy, {actual_bits} against {ideal_bits}"
            );
            assert!(
                actual_bits <= ideal_bits + 16.0,
                "{name}: {actual_bits} bits against an ideal {ideal_bits}"
            );
            let (_, _, huff_bits) = huffman_encode(&data);
            if (huff_bits as f64) > actual_bits + 8.0 {
                beat_huffman += 1;
            }
        }
        assert!(beat_huffman > 0, "arithmetic coding never pulled ahead of Huffman");
        // A byte the model gives no probability to cannot be coded.
        let mut model = vec![0u64; 256];
        model[0] = 1;
        assert!(std::panic::catch_unwind(move || arithmetic_encode(&[1u8], &model)).is_err());
    }

    /// The dictionary and run-length methods all invert exactly, and each
    /// shrinks the kind of data it is built for.
    #[test]
    fn dictionary_and_run_length_methods_invert_exactly() {
        let mut rng = Rng::new(0x_D1C7);
        for (name, data) in corpus(&mut rng) {
            let tokens = lz77_compress(&data, 64, 32);
            assert_eq!(lz77_decompress(&tokens), data, "{name}: LZ77 roundtrip");
            let codes = lzw_compress(&data);
            assert_eq!(lzw_decompress(&codes), data, "{name}: LZW roundtrip");
            let packed = rle_compress(&data);
            assert_eq!(rle_decompress(&packed), data, "{name}: run-length roundtrip");
            // PackBits never grows by more than one byte in 128, plus one.
            assert!(
                packed.len() <= data.len() + data.len().div_ceil(128) + 1,
                "{name}: run-length grew from {} to {}",
                data.len(),
                packed.len()
            );
            assert_eq!(mtf_decode(&mtf_encode(&data)), data, "{name}: move-to-front roundtrip");
            assert_eq!(delta_decode(&delta_encode(&data)), data, "{name}: delta roundtrip");
        }
        // Each method earns its place on the data it suits.
        let runs = vec![9u8; 4000];
        assert!(rle_compress(&runs).len() < 100, "run lengths should crush a constant stream");
        let repeated: Vec<u8> = std::iter::repeat_n(b"abracadabra".as_slice(), 200)
            .flatten()
            .copied()
            .collect();
        assert!(
            lzw_compress(&repeated).len() * 4 < repeated.len(),
            "LZW should exploit a repeated phrase"
        );
        assert!(lz77_compress(&repeated, 512, 255).len() * 8 < repeated.len());
        // A ramp is high-entropy per byte and trivial as differences.
        let ramp: Vec<u8> = (0..2000).map(|i| (i % 256) as u8).collect();
        assert!(entropy_bytes(&ramp) > 7.9);
        assert!(entropy_bytes(&delta_encode(&ramp)) < 0.2, "the differences should be constant");
        // The KwKwK case, where the encoder emits a code it has just made.
        let tricky = b"aaaaaaaaaaaaaaaaaaaa".to_vec();
        assert_eq!(lzw_decompress(&lzw_compress(&tricky)), tricky);
    }

    /// The suffix array really is the sorted suffixes, checked against a
    /// naive sort, and the longest-common-prefix array against a naive
    /// comparison.
    #[test]
    fn suffix_and_lcp_arrays_match_the_naive_construction() {
        let mut rng = Rng::new(0x_5FF1);
        for _ in 0..200 {
            let n = pick(&mut rng, 60);
            // A small alphabet, so that ties are common and the doubling has
            // something to resolve.
            let data: Vec<u8> = (0..n).map(|_| b'a' + (pick(&mut rng, 3) as u8)).collect();
            let sa = suffix_array(&data);
            let mut want: Vec<usize> = (0..n).collect();
            want.sort_by_key(|&i| &data[i..]);
            assert_eq!(sa, want, "the suffix array is not the sorted suffixes");
            // Sorted, by the definition rather than by construction.
            for w in sa.windows(2) {
                assert!(data[w[0]..] < data[w[1]..], "the suffix array is out of order");
            }
            let lcp = lcp_array(&data, &sa);
            assert_eq!(lcp.len(), n);
            for i in 1..n {
                let (a, b) = (&data[sa[i - 1]..], &data[sa[i]..]);
                let want = a.iter().zip(b).take_while(|(x, y)| x == y).count();
                assert_eq!(lcp[i], want, "the overlap at {i} is wrong");
            }
            // The longest repeat, against brute force.
            let (start, len) = longest_repeated_substring(&data);
            let mut brute = 0usize;
            for i in 0..n {
                for j in i + 1..n {
                    let l = data[i..].iter().zip(&data[j..]).take_while(|(x, y)| x == y).count();
                    brute = brute.max(l);
                }
            }
            assert_eq!(len, brute, "the longest repeat is the wrong length");
            if len > 0 {
                let piece = &data[start..start + len];
                let occurrences =
                    (0..=n - len).filter(|&i| &data[i..i + len] == piece).count();
                assert!(occurrences >= 2, "the reported repeat occurs once");
            }
        }
    }

    /// The Burrows-Wheeler transform inverts exactly, permutes rather than
    /// changes the bytes, and groups them into runs -- which is the whole
    /// reason to apply it.
    #[test]
    fn the_burrows_wheeler_transform_inverts_and_groups_runs() {
        let mut rng = Rng::new(0x_B77A);
        let runs = |d: &[u8]| d.windows(2).filter(|w| w[0] != w[1]).count() + usize::from(!d.is_empty());
        for (name, data) in corpus(&mut rng) {
            let (t, idx) = bwt(&data);
            assert_eq!(t.len(), data.len());
            assert_eq!(ibwt(&t, idx), data, "{name}: the transform did not invert");
            // It is a permutation of the bytes, so the histogram is the same.
            assert_eq!(histogram(&t), histogram(&data), "{name}: bytes were not preserved");
            // The full pipeline, which is the roadmap's stated property.
            let piped = rle_compress(&mtf_encode(&t));
            let back = ibwt(&mtf_decode(&rle_decompress(&piped)), idx);
            assert_eq!(back, data, "{name}: the pipeline did not roundtrip");
        }
        // On text with repeated context the transform really does group the
        // bytes: the run count falls sharply, which is what the later stages
        // then live on.
        let text: Vec<u8> = std::iter::repeat_n(
            b"the rain in spain falls mainly on the plain. ".as_slice(),
            40,
        )
        .flatten()
        .copied()
        .collect();
        let (t, _) = bwt(&text);
        assert!(
            runs(&t) * 2 < runs(&text),
            "the transform left {} runs against the original's {}",
            runs(&t),
            runs(&text)
        );
        // And the pipeline beats coding the text directly.
        let (_, _, direct) = huffman_encode(&text);
        let (_, _, piped) = huffman_encode(&rle_compress(&mtf_encode(&t)));
        assert!(piped < direct, "the pipeline made it bigger: {piped} against {direct}");
        // A periodic string is the case where rotations tie, so it is worth
        // its own check.
        for s in [b"abab".as_slice(), b"aaaa", b"abcabcabc", b"a"] {
            let (t, i) = bwt(s);
            assert_eq!(ibwt(&t, i), s, "a periodic string did not invert");
        }
    }

    /// The measures behave the way their definitions require.
    #[test]
    fn entropy_and_compression_distance_behave() {
        assert_eq!(entropy_bytes(&[]), 0.0);
        assert_eq!(entropy_bytes(&[3u8; 100]), 0.0, "a constant stream carries nothing");
        let uniform: Vec<u8> = (0..=255).cycle().take(256 * 40).collect();
        assert!((entropy_bytes(&uniform) - 8.0).abs() < 1e-9, "a flat histogram is eight bits");
        let half: Vec<u8> = (0..=127).cycle().take(128 * 40).collect();
        assert!((entropy_bytes(&half) - 7.0).abs() < 1e-9);
        assert!((compression_bound(&uniform) - uniform.len() as f64).abs() < 1e-6);

        // No memoryless coder beats the entropy bound.
        let mut rng = Rng::new(0x_E177);
        for (name, data) in corpus(&mut rng) {
            if data.is_empty() {
                continue;
            }
            let (_, _, bits) = huffman_encode(&data);
            assert!(
                bits as f64 / 8.0 >= compression_bound(&data) - 1e-6,
                "{name}: Huffman beat the entropy bound"
            );
        }

        // The compression distance: near zero for a string against itself,
        // larger for unrelated ones, and never negative.
        let a: Vec<u8> =
            std::iter::repeat_n(b"the same sentence over and over. ".as_slice(), 30)
                .flatten()
                .copied()
                .collect();
        let b: Vec<u8> = (0..900).map(|_| (rng.next_u64() & 0xFF) as u8).collect();
        let self_distance = normalized_compression_distance(&a, &a);
        let cross = normalized_compression_distance(&a, &b);
        assert!(self_distance < 0.2, "a string is not close to itself: {self_distance}");
        assert!(cross > self_distance, "unrelated data is not further: {cross}");
        assert!(cross <= 1.2, "the distance ran away: {cross}");
        assert!(normalized_compression_distance(&[], &[]) >= 0.0);
        // The estimate never claims to beat storing the bytes.
        for (name, data) in corpus(&mut rng) {
            let est = kolmogorov_estimate_by_compressors(&data);
            assert!(est >= 0.0 && est <= data.len() as f64, "{name}: implausible estimate {est}");
        }
    }
}
