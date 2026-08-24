//! Binary linear block codes.
//!
//! A linear code of length `n` and dimension `k` is a `k`-dimensional
//! subspace of `GF(2)^n`. Everything follows from that one sentence. The
//! subspace is described either by a basis -- the rows of a generator matrix
//! `G` -- or by the equations that cut it out -- the rows of a parity check
//! matrix `H`, with `C = { x : H x' = 0 }`. Encoding is a matrix product.
//! Decoding is the observation that `H (c + e)' = H e'`, so the syndrome
//! depends only on the error and not on what was sent: correcting is
//! choosing the lightest error pattern with the observed syndrome.
//!
//! Linearity is also what makes the minimum distance computable at all. The
//! distance between two codewords is the weight of their difference, which is
//! another codeword, so the minimum distance over all `2^k (2^k - 1) / 2`
//! pairs is just the minimum weight over the `2^k - 1` non-zero codewords.
//!
//! The `_small` routines enumerate the whole code and are exponential in `k`
//! by construction; they are for the classical codes, which are small.

use crate::exact::BigInt;
use crate::monte_carlo::Rng;
use std::collections::BTreeMap;

/// A matrix over `GF(2)`, one bit per entry, packed sixty-four to a word.
///
/// Packing is not only for space: a row operation becomes a handful of word
/// XORs rather than a loop over bits, so elimination on a code-sized matrix
/// costs what a floating-point elimination on a matrix sixty-four times
/// smaller would.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gf2Matrix {
    /// Row count.
    pub rows: usize,
    /// Column count.
    pub cols: usize,
    /// Row-major bit storage, `words_per_row()` words per row.
    pub data: Vec<u64>,
}

impl Gf2Matrix {
    /// Words needed to hold one row.
    #[must_use]
    pub fn words_per_row(&self) -> usize {
        self.cols.div_ceil(64)
    }

    /// An all-zero matrix.
    #[must_use]
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Gf2Matrix { rows, cols, data: vec![0; rows * cols.div_ceil(64)] }
    }

    /// The `n` by `n` identity.
    #[must_use]
    pub fn identity(n: usize) -> Self {
        let mut m = Gf2Matrix::zeros(n, n);
        for i in 0..n {
            m.set(i, i, true);
        }
        m
    }

    /// A matrix from rows of booleans.
    ///
    /// # Panics
    /// Panics if the rows are not all the same length.
    #[must_use]
    pub fn from_rows(rows: &[Vec<bool>]) -> Self {
        let cols = rows.first().map_or(0, Vec::len);
        assert!(rows.iter().all(|r| r.len() == cols), "rows must be the same length");
        let mut m = Gf2Matrix::zeros(rows.len(), cols);
        for (i, row) in rows.iter().enumerate() {
            for (j, &b) in row.iter().enumerate() {
                m.set(i, j, b);
            }
        }
        m
    }

    /// The entry at `(r, c)`.
    ///
    /// # Panics
    /// Panics if the index is out of range.
    #[must_use]
    pub fn get(&self, r: usize, c: usize) -> bool {
        assert!(r < self.rows && c < self.cols, "index ({r}, {c}) is out of range");
        self.data[r * self.words_per_row() + c / 64] & (1u64 << (c % 64)) != 0
    }

    /// Sets the entry at `(r, c)`.
    ///
    /// # Panics
    /// Panics if the index is out of range.
    pub fn set(&mut self, r: usize, c: usize, value: bool) {
        assert!(r < self.rows && c < self.cols, "index ({r}, {c}) is out of range");
        let w = self.words_per_row();
        let idx = r * w + c / 64;
        let bit = 1u64 << (c % 64);
        if value {
            self.data[idx] |= bit;
        } else {
            self.data[idx] &= !bit;
        }
    }

    /// Row `r` as a vector of booleans.
    ///
    /// # Panics
    /// Panics if `r` is out of range.
    #[must_use]
    pub fn row(&self, r: usize) -> Vec<bool> {
        (0..self.cols).map(|c| self.get(r, c)).collect()
    }

    /// Every row as a vector of booleans.
    #[must_use]
    pub fn to_rows(&self) -> Vec<Vec<bool>> {
        (0..self.rows).map(|r| self.row(r)).collect()
    }

    /// Adds row `src` into row `dst`, in place. Addition over `GF(2)` is XOR.
    fn add_row(&mut self, dst: usize, src: usize) {
        let w = self.words_per_row();
        for k in 0..w {
            self.data[dst * w + k] ^= self.data[src * w + k];
        }
    }

    fn swap_rows(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }
        let w = self.words_per_row();
        for k in 0..w {
            self.data.swap(a * w + k, b * w + k);
        }
    }

    /// The reduced row echelon form, and the pivot column of each non-zero
    /// row in order.
    ///
    /// Over `GF(2)` there is no scaling step: the only non-zero scalar is
    /// one. Elimination is therefore exactly "find a row with a one in this
    /// column, move it up, and XOR it into every other row that has one".
    #[must_use]
    pub fn rref(&self) -> (Gf2Matrix, Vec<usize>) {
        let mut m = self.clone();
        let mut pivots = Vec::new();
        let mut r = 0;
        for c in 0..m.cols {
            if r == m.rows {
                break;
            }
            let Some(p) = (r..m.rows).find(|&i| m.get(i, c)) else { continue };
            m.swap_rows(r, p);
            for i in 0..m.rows {
                if i != r && m.get(i, c) {
                    m.add_row(i, r);
                }
            }
            pivots.push(c);
            r += 1;
        }
        (m, pivots)
    }

    /// The rank: the number of independent rows.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.rref().1.len()
    }

    /// The transpose.
    #[must_use]
    pub fn transpose(&self) -> Gf2Matrix {
        let mut t = Gf2Matrix::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                if self.get(r, c) {
                    t.set(c, r, true);
                }
            }
        }
        t
    }

    /// The matrix product over `GF(2)`.
    ///
    /// # Panics
    /// Panics unless the shapes agree.
    #[must_use]
    pub fn mul(&self, other: &Gf2Matrix) -> Gf2Matrix {
        assert_eq!(self.cols, other.rows, "shapes do not agree");
        let mut out = Gf2Matrix::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                if self.get(i, k) {
                    // Adding a whole row at a time keeps the inner loop on
                    // words rather than bits.
                    let w = out.words_per_row();
                    let ow = other.words_per_row();
                    for t in 0..w.min(ow) {
                        out.data[i * w + t] ^= other.data[k * ow + t];
                    }
                }
            }
        }
        out
    }

    /// The product with a column vector: `M x'`.
    ///
    /// # Panics
    /// Panics unless `x` has one entry per column.
    #[must_use]
    pub fn mul_vec(&self, x: &[bool]) -> Vec<bool> {
        assert_eq!(x.len(), self.cols, "one entry per column is required");
        (0..self.rows)
            .map(|r| (0..self.cols).filter(|&c| x[c] && self.get(r, c)).count() % 2 == 1)
            .collect()
    }

    /// The product with a row vector on the left: `x M`.
    ///
    /// # Panics
    /// Panics unless `x` has one entry per row.
    #[must_use]
    pub fn vec_mul(&self, x: &[bool]) -> Vec<bool> {
        assert_eq!(x.len(), self.rows, "one entry per row is required");
        let w = self.words_per_row();
        let mut acc = vec![0u64; w];
        for (r, &on) in x.iter().enumerate() {
            if on {
                for k in 0..w {
                    acc[k] ^= self.data[r * w + k];
                }
            }
        }
        (0..self.cols).map(|c| acc[c / 64] & (1u64 << (c % 64)) != 0).collect()
    }

    /// A solution `x` of `M x' = b'`, or `None` if there is none.
    ///
    /// Any solution: the system is under-determined whenever the kernel is
    /// non-trivial, and the free variables are left at zero.
    ///
    /// # Panics
    /// Panics unless `b` has one entry per row.
    #[must_use]
    pub fn solve(&self, b: &[bool]) -> Option<Vec<bool>> {
        assert_eq!(b.len(), self.rows, "one right-hand entry per row is required");
        // Augment and eliminate.
        let mut aug = Gf2Matrix::zeros(self.rows, self.cols + 1);
        for r in 0..self.rows {
            for c in 0..self.cols {
                if self.get(r, c) {
                    aug.set(r, c, true);
                }
            }
            if b[r] {
                aug.set(r, self.cols, true);
            }
        }
        let (e, pivots) = aug.rref();
        // A pivot in the augmented column is the equation 0 = 1.
        if pivots.last() == Some(&self.cols) {
            return None;
        }
        let mut x = vec![false; self.cols];
        for (row, &col) in pivots.iter().enumerate() {
            x[col] = e.get(row, self.cols);
        }
        Some(x)
    }

    /// A basis for the kernel `{ x : M x' = 0 }`.
    ///
    /// One basis vector per free column: set that free variable to one, the
    /// others to zero, and read the pivot variables off the echelon form.
    /// The count is `cols - rank`, which is the rank-nullity theorem and is
    /// what the tests check it against.
    #[must_use]
    pub fn kernel_basis(&self) -> Vec<Vec<bool>> {
        let (e, pivots) = self.rref();
        let free: Vec<usize> = (0..self.cols).filter(|c| !pivots.contains(c)).collect();
        free.iter()
            .map(|&f| {
                let mut v = vec![false; self.cols];
                v[f] = true;
                for (row, &p) in pivots.iter().enumerate() {
                    if e.get(row, f) {
                        v[p] = true;
                    }
                }
                v
            })
            .collect()
    }
}

/// The number of ones in a bit vector: its Hamming weight.
#[must_use]
pub fn weight(v: &[bool]) -> usize {
    v.iter().filter(|&&b| b).count()
}

/// The bitwise difference of two equal-length vectors.
///
/// # Panics
/// Panics unless the lengths agree.
#[must_use]
pub fn xor(a: &[bool], b: &[bool]) -> Vec<bool> {
    assert_eq!(a.len(), b.len(), "lengths must agree");
    a.iter().zip(b).map(|(&x, &y)| x != y).collect()
}

/// A binary linear code, held by both of its descriptions.
///
/// `g` is `k` by `n` and its rows are a basis of the code; `h` is `n - k` by
/// `n` and its rows are a basis of the dual, so `G H'` is zero and a word is
/// a codeword exactly when its syndrome vanishes.
#[derive(Debug, Clone)]
pub struct LinearCode {
    /// Generator matrix, `k` by `n`.
    pub g: Gf2Matrix,
    /// Parity check matrix, `n - k` by `n`.
    pub h: Gf2Matrix,
    /// Block length.
    pub n: usize,
    /// Dimension.
    pub k: usize,
    /// Minimum distance.
    pub d: usize,
}

impl LinearCode {
    /// The code generated by the rows of `g`, with the parity check matrix
    /// and minimum distance derived.
    ///
    /// Dependent rows are dropped, so `k` is the rank rather than the row
    /// count. The parity check matrix is a basis of the kernel of `g`, which
    /// is the dual code by definition.
    ///
    /// # Panics
    /// Panics if the generator has no columns, or if the dimension exceeds
    /// twenty, since the distance is found by enumerating the code.
    #[must_use]
    pub fn from_generator(g: &Gf2Matrix) -> Self {
        assert!(g.cols > 0, "a code needs a positive length");
        let (e, pivots) = g.rref();
        let k = pivots.len();
        assert!(k <= 20, "the distance search enumerates 2^k codewords");
        let basis: Vec<Vec<bool>> = (0..k).map(|r| e.row(r)).collect();
        let g = Gf2Matrix::from_rows(&basis);
        let kernel = g.kernel_basis();
        let h = if kernel.is_empty() {
            Gf2Matrix::zeros(0, g.cols)
        } else {
            Gf2Matrix::from_rows(&kernel)
        };
        let n = g.cols;
        let mut code = LinearCode { g, h, n, k, d: 0 };
        code.d = code.minimum_distance_small();
        code
    }

    /// Every codeword, in order of the message it encodes.
    ///
    /// # Panics
    /// Panics if the dimension exceeds twenty.
    #[must_use]
    pub fn codewords(&self) -> Vec<Vec<bool>> {
        assert!(self.k <= 20, "enumerating a code of dimension {} is not small", self.k);
        (0..1u64 << self.k)
            .map(|m| {
                let msg: Vec<bool> = (0..self.k).map(|i| m & (1 << i) != 0).collect();
                self.encode(&msg)
            })
            .collect()
    }

    /// The message times the generator.
    ///
    /// # Panics
    /// Panics unless `msg` has one bit per dimension.
    #[must_use]
    pub fn encode(&self, msg: &[bool]) -> Vec<bool> {
        assert_eq!(msg.len(), self.k, "one message bit per dimension is required");
        self.g.vec_mul(msg)
    }

    /// The syndrome `H x'`, which is zero exactly on codewords.
    ///
    /// # Panics
    /// Panics unless `recv` has one bit per position.
    #[must_use]
    pub fn syndrome(&self, recv: &[bool]) -> Vec<bool> {
        assert_eq!(recv.len(), self.n, "one bit per position is required");
        self.h.mul_vec(recv)
    }

    /// Whether the word is in the code.
    ///
    /// # Panics
    /// Panics unless `x` has one bit per position.
    #[must_use]
    pub fn contains(&self, x: &[bool]) -> bool {
        self.syndrome(x).iter().all(|&b| !b)
    }

    /// The minimum distance, by enumeration.
    ///
    /// Linearity turns a search over pairs into a search over words: the
    /// distance between two codewords is the weight of their difference,
    /// which is itself a codeword. Zero for the zero code, which has no
    /// non-zero word to measure.
    ///
    /// # Panics
    /// Panics if the dimension exceeds twenty.
    #[must_use]
    pub fn minimum_distance_small(&self) -> usize {
        self.codewords()
            .into_iter()
            .map(|c| weight(&c))
            .filter(|&w| w > 0)
            .min()
            .unwrap_or(0)
    }

    /// The weight enumerator: how many codewords have each weight, indexed
    /// from zero to `n`.
    ///
    /// The coefficients of a linear code's weight enumerator determine its
    /// undetected error probability on a symmetric channel exactly, and by
    /// MacWilliams's identity they determine the dual code's enumerator too.
    /// Counts are exact integers because a code of dimension sixty would
    /// overflow anything narrower.
    ///
    /// # Panics
    /// Panics if the dimension exceeds twenty.
    #[must_use]
    pub fn weight_enumerator(&self) -> Vec<BigInt> {
        let mut out = vec![BigInt::zero(); self.n + 1];
        for c in self.codewords() {
            let w = weight(&c);
            out[w] = out[w].add(&BigInt::one());
        }
        out
    }

    /// The dual code, whose generator is this one's parity check matrix.
    ///
    /// # Panics
    /// Panics if the dual's dimension exceeds twenty.
    #[must_use]
    pub fn dual(&self) -> LinearCode {
        LinearCode::from_generator(&self.h)
    }

    /// Whether the code equals its own dual, which needs `n = 2k` and every
    /// pair of generator rows orthogonal.
    #[must_use]
    pub fn is_self_dual(&self) -> bool {
        if self.n != 2 * self.k {
            return false;
        }
        let prod = self.g.mul(&self.g.transpose());
        prod.data.iter().all(|&w| w == 0)
    }

    /// The lightest error pattern with the given syndrome, found by
    /// searching error weights upward.
    ///
    /// The coset leader. Every syndrome is achieved by some pattern, since
    /// `H` has full row rank, so the search always terminates; how quickly
    /// depends on the leader's weight, which for a code correcting `t` errors
    /// is at most `t` on any word within `t` of a codeword.
    fn coset_leader(&self, syndrome: &[bool]) -> Vec<bool> {
        let r = self.h.rows;
        assert!(r <= 64, "syndrome decoding here packs the syndrome into a word");
        let target: u64 = (0..r).filter(|&i| syndrome[i]).map(|i| 1u64 << i).sum();
        let make = |bits: &[usize]| -> Vec<bool> {
            let mut e = vec![false; self.n];
            for &i in bits {
                e[i] = true;
            }
            e
        };
        if target == 0 {
            return vec![false; self.n];
        }
        // Each column of H, packed. An error pattern's syndrome is the XOR of
        // the columns it selects, which is the whole of syndrome decoding.
        let col: Vec<u64> =
            (0..self.n).map(|c| (0..r).filter(|&i| self.h.get(i, c)).map(|i| 1u64 << i).sum()).collect();
        for a in 0..self.n {
            if col[a] == target {
                return make(&[a]);
            }
        }
        for a in 0..self.n {
            for b in a + 1..self.n {
                if col[a] ^ col[b] == target {
                    return make(&[a, b]);
                }
            }
        }
        for a in 0..self.n {
            for b in a + 1..self.n {
                let ab = col[a] ^ col[b];
                for c in b + 1..self.n {
                    if ab ^ col[c] == target {
                        return make(&[a, b, c]);
                    }
                }
            }
        }
        for w in 4..=self.n {
            for combo in crate::discrete::combinatorics::combinations_iter(self.n, w) {
                if combo.iter().fold(0u64, |acc, &i| acc ^ col[i]) == target {
                    return make(&combo);
                }
            }
        }
        unreachable!("a full-rank parity check matrix reaches every syndrome")
    }

    /// Syndrome decoding: subtract the lightest error pattern consistent with
    /// what was received.
    ///
    /// Returns the corrected word and how many bits were changed. Correct
    /// whenever the true error weighs at most `(d - 1) / 2`; beyond that the
    /// lightest consistent pattern is some other coset member and the result
    /// is a different codeword, which is not a failure of the method but the
    /// definition of exceeding the correction radius.
    ///
    /// # Panics
    /// Panics unless `recv` has one bit per position.
    #[must_use]
    pub fn decode_syndrome(&self, recv: &[bool]) -> (Vec<bool>, usize) {
        let s = self.syndrome(recv);
        let e = self.coset_leader(&s);
        (xor(recv, &e), weight(&e))
    }

    /// The full syndrome table: every syndrome mapped to its coset leader.
    ///
    /// This is the standard array with only its first column kept, which is
    /// all decoding needs. Built by walking error patterns in weight order,
    /// so the first pattern to reach a syndrome is a lightest one.
    ///
    /// # Panics
    /// Panics if the redundancy `n - k` exceeds twenty, since the table has
    /// one entry per syndrome.
    #[must_use]
    pub fn syndrome_table_small(&self) -> BTreeMap<Vec<bool>, Vec<bool>> {
        let r = self.n - self.k;
        assert!(r <= 20, "the syndrome table has 2^{r} entries");
        let mut table: BTreeMap<Vec<bool>, Vec<bool>> = BTreeMap::new();
        table.insert(vec![false; r], vec![false; self.n]);
        for w in 1..=self.n {
            if table.len() == 1usize << r {
                break;
            }
            for combo in crate::discrete::combinatorics::combinations_iter(self.n, w) {
                let mut e = vec![false; self.n];
                for &i in &combo {
                    e[i] = true;
                }
                table.entry(self.syndrome(&e)).or_insert(e);
            }
        }
        table
    }

    /// Decoding through an explicit standard array.
    ///
    /// The same answer [`decode_syndrome`](Self::decode_syndrome) gives, by a
    /// different route: build the whole table first, then look up. Slower per
    /// word and faster per thousand words, and useful as the reference the
    /// incremental search is checked against.
    ///
    /// # Panics
    /// Panics unless `recv` has one bit per position, or if the redundancy
    /// exceeds twenty.
    #[must_use]
    pub fn standard_array_decode_small(&self, recv: &[bool]) -> (Vec<bool>, usize) {
        let table = self.syndrome_table_small();
        let s = self.syndrome(recv);
        let e = table.get(&s).expect("the table covers every syndrome").clone();
        (xor(recv, &e), weight(&e))
    }

    // -- Named families -----------------------------------------------------

    /// The Hamming code of redundancy `r`: length `2^r - 1`, dimension
    /// `2^r - 1 - r`, distance three.
    ///
    /// The parity check matrix has every non-zero `r`-bit column exactly
    /// once, which is the whole construction. A single error in position `j`
    /// then produces the syndrome that *is* column `j`, so the syndrome names
    /// the error outright. It is perfect: the spheres of radius one around
    /// the codewords tile the space with nothing left over, since
    /// `2^k (1 + n) = 2^k 2^r = 2^n`.
    ///
    /// The distance is three by construction rather than by search: no one
    /// or two distinct non-zero columns can sum to zero, and columns one,
    /// two and three do. Rediscovering that by enumerating `2^26` words is
    /// the only thing that would stop the family at `r = 4`.
    ///
    /// # Panics
    /// Panics unless `r` is between two and eight.
    #[must_use]
    pub fn hamming(r: usize) -> Self {
        assert!((2..=8).contains(&r), "r must be between two and eight");
        let n = (1usize << r) - 1;
        let mut h = Gf2Matrix::zeros(r, n);
        for c in 0..n {
            for b in 0..r {
                if (c + 1) & (1 << b) != 0 {
                    h.set(b, c, true);
                }
            }
        }
        let g = Gf2Matrix::from_rows(&h.kernel_basis());
        LinearCode { g, h, n, k: n - r, d: 3 }
    }

    /// The extended Hamming code: a Hamming code with an overall parity bit,
    /// giving length `2^r`, the same dimension, and distance four.
    ///
    /// The extra bit raises the distance from three to four, which does not
    /// improve correction -- still one error -- but makes two errors always
    /// detectable rather than sometimes mistaken for one. That is the
    /// single-error-correcting, double-error-detecting code memory uses.
    ///
    /// # Panics
    /// Panics unless `r` is between two and eight.
    #[must_use]
    pub fn extended_hamming(r: usize) -> Self {
        let base = LinearCode::hamming(r);
        let n = base.n + 1;
        // The check matrix gains a zero column and an all-ones row: the old
        // checks ignore the new bit, and the new check is the overall parity.
        let mut h = Gf2Matrix::zeros(r + 1, n);
        for i in 0..r {
            for c in 0..base.n {
                if base.h.get(i, c) {
                    h.set(i, c, true);
                }
            }
        }
        for c in 0..n {
            h.set(r, c, true);
        }
        let g = extend_with_parity(&base.g);
        LinearCode { g, h, n, k: base.k, d: 4 }
    }

    /// The repetition code: one bit sent `n` times, distance `n`.
    ///
    /// # Panics
    /// Panics if `n` is zero.
    #[must_use]
    pub fn repetition(n: usize) -> Self {
        assert!(n > 0, "a code needs a positive length");
        LinearCode::from_generator(&Gf2Matrix::from_rows(&[vec![true; n]]))
    }

    /// The single parity check code: `n - 1` message bits and their parity,
    /// distance two.
    ///
    /// The dual of the repetition code of the same length, which is why the
    /// two appear together.
    ///
    /// # Panics
    /// Panics unless `n` is at least two.
    #[must_use]
    pub fn parity_check(n: usize) -> Self {
        assert!(n >= 2, "a parity check code needs at least two positions");
        let rows: Vec<Vec<bool>> = (0..n - 1)
            .map(|i| (0..n).map(|j| j == i || j == n - 1).collect())
            .collect();
        LinearCode::from_generator(&Gf2Matrix::from_rows(&rows))
    }

    /// The binary Golay code, `[23, 12, 7]`.
    ///
    /// Cyclic, generated by `1 + x + x^5 + x^6 + x^7 + x^9 + x^11`, one of
    /// the two irreducible factors of `x^23 - 1` over `GF(2)` besides
    /// `x - 1`. It is perfect: spheres of radius three around its 4096
    /// codewords tile `GF(2)^23` exactly, since
    /// `4096 * (1 + 23 + 253 + 1771) = 2^23`. Only two non-trivial perfect
    /// binary codes exist -- this and the Hamming family -- so the
    /// arithmetic working out is not a coincidence that could have gone
    /// another way.
    #[must_use]
    pub fn golay23() -> Self {
        // Coefficients of the generator polynomial, constant term first.
        const G: [usize; 7] = [0, 1, 5, 6, 7, 9, 11];
        let rows: Vec<Vec<bool>> = (0..12)
            .map(|shift| {
                let mut row = vec![false; 23];
                for &e in &G {
                    row[e + shift] = true;
                }
                row
            })
            .collect();
        LinearCode::from_generator(&Gf2Matrix::from_rows(&rows))
    }

    /// The extended binary Golay code, `[24, 12, 8]`.
    ///
    /// Self-dual, and the distance rises to eight, so every weight is a
    /// multiple of four. It corrects three errors and detects four.
    #[must_use]
    pub fn golay24() -> Self {
        LinearCode::from_generator(&extend_with_parity(&LinearCode::golay23().g))
    }

    /// The Reed-Muller code `RM(r, m)`: length `2^m`, distance `2^(m - r)`.
    ///
    /// The codewords are the truth tables of every Boolean polynomial in `m`
    /// variables of degree at most `r`, so the generator rows are the
    /// products of up to `r` coordinate functions evaluated at all `2^m`
    /// points. `RM(0, m)` is the repetition code and `RM(m - 1, m)` is the
    /// single parity check code, which is the cleanest statement of what the
    /// family interpolates between.
    ///
    /// # Panics
    /// Panics unless `r <= m` and the dimension stays at or below twenty.
    #[must_use]
    pub fn reed_muller(r: usize, m: usize) -> Self {
        assert!(r <= m, "the degree cannot exceed the number of variables");
        let n = 1usize << m;
        let mut rows: Vec<Vec<bool>> = Vec::new();
        // One row per monomial of degree at most r: the subsets of variables.
        for degree in 0..=r {
            for subset in crate::discrete::combinatorics::combinations_iter(m, degree) {
                let row: Vec<bool> = (0..n)
                    .map(|point| subset.iter().all(|&v| point & (1 << v) != 0))
                    .collect();
                rows.push(row);
            }
        }
        LinearCode::from_generator(&Gf2Matrix::from_rows(&rows))
    }
}

/// A generator matrix with an overall parity column appended.
fn extend_with_parity(g: &Gf2Matrix) -> Gf2Matrix {
    let rows: Vec<Vec<bool>> = (0..g.rows)
        .map(|r| {
            let mut row = g.row(r);
            let p = weight(&row) % 2 == 1;
            row.push(p);
            row
        })
        .collect();
    Gf2Matrix::from_rows(&rows)
}

// ---------------------------------------------------------------------------
// Hamming(7, 4) in the classical bit layout
// ---------------------------------------------------------------------------

/// Which positions each of the three parity bits covers, in the numbering
/// where position `i` is checked by parity bit `b` exactly when bit `b` of
/// `i + 1` is set.
const H74_COVER: [u8; 3] = [0b101_0101, 0b110_0110, 0b111_1000];

/// Hamming(7, 4) encoding: four data bits in, seven out.
///
/// The classical layout, with the parity bits at the powers of two: position
/// one, two and four, counting from one at the least significant bit of the
/// result. Parity bit `b` covers exactly the positions whose index has bit
/// `b` set, so the three parity checks of a corrupted word spell out the
/// binary numeral of the corrupted position.
///
/// # Panics
/// Panics if `nibble` has anything above its low four bits.
#[must_use]
pub fn hamming_74_encode(nibble: u8) -> u8 {
    assert!(nibble < 16, "four data bits only");
    // Data bits go to positions 3, 5, 6 and 7; the rest are parity.
    let mut word = 0u8;
    for (i, pos) in [3u8, 5, 6, 7].iter().enumerate() {
        if nibble & (1 << i) != 0 {
            word |= 1 << (pos - 1);
        }
    }
    for (b, cover) in H74_COVER.iter().enumerate() {
        let parity = (word & cover).count_ones() % 2;
        if parity == 1 {
            word |= 1 << ((1 << b) - 1);
        }
    }
    word
}

/// Hamming(7, 4) decoding: correct any single error and return the four data
/// bits, with a flag saying whether a correction was made.
///
/// # Panics
/// Panics if `byte` has its top bit set, which is outside the seven-bit code.
#[must_use]
pub fn hamming_74_decode(byte: u8) -> (u8, bool) {
    assert!(byte < 128, "a seven-bit codeword only");
    let mut syndrome = 0u8;
    for (b, cover) in H74_COVER.iter().enumerate() {
        if (byte & cover).count_ones() % 2 == 1 {
            syndrome |= 1 << b;
        }
    }
    let corrected = syndrome != 0;
    // The syndrome read as a binary numeral is the position, counting from
    // one, of the flipped bit.
    let word = if corrected { byte ^ (1 << (syndrome - 1)) } else { byte };
    let mut nibble = 0u8;
    for (i, pos) in [3u8, 5, 6, 7].iter().enumerate() {
        if word & (1 << (pos - 1)) != 0 {
            nibble |= 1 << i;
        }
    }
    (nibble, corrected)
}

// ---------------------------------------------------------------------------
// Bounds
// ---------------------------------------------------------------------------

/// The Singleton bound: `d <= n - k + 1`.
///
/// Deleting `d - 1` positions must leave the codewords distinct, since they
/// differ in at least `d`, so the code embeds in `GF(2)^(n - d + 1)` and
/// `k <= n - d + 1`. Returns the largest distance the parameters allow.
///
/// # Panics
/// Panics unless `k <= n`.
#[must_use]
pub fn singleton_bound(n: usize, k: usize) -> usize {
    assert!(k <= n, "the dimension cannot exceed the length");
    n - k + 1
}

/// The Hamming, or sphere-packing, bound on how many codewords a binary code
/// of length `n` and distance `d` can have.
///
/// Spheres of radius `t = (d - 1) / 2` around distinct codewords are
/// disjoint, so their total volume fits inside `2^n`. A code meeting it with
/// equality is *perfect* -- the spheres tile the space -- which the Hamming
/// and Golay codes do and almost nothing else does.
#[must_use]
pub fn hamming_bound(n: usize, d: usize) -> f64 {
    let t = (d.saturating_sub(1)) / 2;
    let volume: f64 = (0..=t)
        .map(|i| crate::discrete::combinatorics::binomial_u64(n as u64, i as u64).map_or(f64::INFINITY, |x| x as f64))
        .sum();
    (2.0f64).powi(n as i32) / volume
}

/// The Gilbert-Varshamov bound: a code of length `n` and distance `d` with at
/// least this many codewords exists.
///
/// A lower bound, and a constructive one: keep adding any word at distance
/// `d` or more from everything chosen so far, and you can only be stuck once
/// the balls of radius `d - 1` cover the space. Where the Hamming bound says
/// what is impossible, this says what is unavoidable, and the best known
/// binary codes sit between them.
#[must_use]
pub fn gilbert_varshamov(n: usize, d: usize) -> f64 {
    let volume: f64 = (0..d)
        .map(|i| crate::discrete::combinatorics::binomial_u64(n as u64, i as u64).map_or(f64::INFINITY, |x| x as f64))
        .sum();
    (2.0f64).powi(n as i32) / volume
}

/// The Plotkin bound, for codes whose distance is more than half their
/// length.
///
/// When `2d > n` the average distance between codewords cannot reach `d`
/// unless there are very few of them, and the count is capped at
/// `2 * floor(d / (2d - n))`. Outside that regime the bound says nothing and
/// this returns infinity.
#[must_use]
pub fn plotkin_bound(n: usize, d: usize) -> f64 {
    if 2 * d > n {
        2.0 * ((d as f64) / (2 * d - n) as f64).floor()
    } else {
        f64::INFINITY
    }
}

// ---------------------------------------------------------------------------
// Low-density parity check codes
// ---------------------------------------------------------------------------

/// A regular low-density parity check matrix by Gallager's construction:
/// `wc` ones in every column and `wr` in every row.
///
/// The first band of rows partitions the columns into consecutive runs of
/// `wr`; each later band is a column permutation of that one. The result is
/// sparse by construction, which is the whole point -- belief propagation
/// costs one message per one in the matrix, and its accuracy depends on the
/// Tanner graph having few short cycles, which a sparse random matrix
/// mostly does.
///
/// # Panics
/// Panics unless `wr` divides `n` and `wc` is between one and `n / wr`.
#[must_use]
pub fn ldpc_regular(n: usize, wc: usize, wr: usize, rng: &mut Rng) -> Gf2Matrix {
    assert!(wr > 0 && n.is_multiple_of(wr), "the row weight must divide the length");
    let band = n / wr;
    assert!(wc >= 1 && wc <= band, "the column weight must fit the bands");
    let mut h = Gf2Matrix::zeros(wc * band, n);
    for b in 0..wc {
        let perm: Vec<usize> = if b == 0 {
            (0..n).collect()
        } else {
            crate::discrete::combinatorics::random_permutation(n, rng)
        };
        for i in 0..band {
            for j in 0..wr {
                h.set(b * band + i, perm[i * wr + j], true);
            }
        }
    }
    h
}

/// Belief propagation decoding of an LDPC code, in the log-likelihood domain.
///
/// `llr[i]` is the log of the ratio of the probability that bit `i` is zero
/// to the probability that it is one, so a positive value leans towards zero.
/// Each round every check tells each of its bits what the other bits imply,
/// and every bit tells each of its checks what the other checks imply; the
/// exclusions are what keep a message from being fed its own output back.
///
/// Returns the hard decisions and whether every parity check is satisfied.
/// A `true` is strong evidence of a correct decode but not proof: the
/// algorithm can settle on a different codeword.
///
/// # Panics
/// Panics unless `llr` has one entry per column.
#[must_use]
pub fn ldpc_decode_bp(h: &Gf2Matrix, llr: &[f64], iters: usize) -> (Vec<bool>, bool) {
    assert_eq!(llr.len(), h.cols, "one log-likelihood per position is required");
    let (m, n) = (h.rows, h.cols);
    let edges: Vec<Vec<usize>> = (0..m)
        .map(|r| (0..n).filter(|&c| h.get(r, c)).collect())
        .collect();
    // Messages from check to bit, one per edge.
    let mut to_bit: Vec<Vec<f64>> = edges.iter().map(|e| vec![0.0; e.len()]).collect();
    let mut hard = vec![false; n];
    for _ in 0..=iters {
        // Total belief at each bit, then the message it sends back excludes
        // the check it is going to.
        let mut total = llr.to_vec();
        for (r, row) in edges.iter().enumerate() {
            for (idx, &c) in row.iter().enumerate() {
                total[c] += to_bit[r][idx];
            }
        }
        hard = total.iter().map(|&x| x < 0.0).collect();
        if satisfies(h, &hard) {
            return (hard, true);
        }
        // Check to bit: the tanh rule, which is the product form of the
        // parity of several independent bits.
        for (r, row) in edges.iter().enumerate() {
            let to_check: Vec<f64> = row
                .iter()
                .enumerate()
                .map(|(idx, &c)| total[c] - to_bit[r][idx])
                .collect();
            for idx in 0..row.len() {
                let mut prod = 1.0f64;
                for (other, &v) in to_check.iter().enumerate() {
                    if other != idx {
                        prod *= (v / 2.0).clamp(-30.0, 30.0).tanh();
                    }
                }
                // Keep the argument of atanh strictly inside the interval, or
                // a saturated product returns infinity and poisons the rest.
                to_bit[r][idx] = 2.0 * prod.clamp(-1.0 + 1e-12, 1.0 - 1e-12).atanh();
            }
        }
    }
    let ok = satisfies(h, &hard);
    (hard, ok)
}

/// Whether every parity check is satisfied.
fn satisfies(h: &Gf2Matrix, x: &[bool]) -> bool {
    h.mul_vec(x).iter().all(|&b| !b)
}

/// Gallager's bit-flipping decoder: repeatedly flip whichever bits sit in the
/// most unsatisfied checks.
///
/// Hard decisions only, so it throws away the channel's confidence and pays
/// for it -- roughly two decibels against belief propagation on the same
/// code. What it buys is that a round is a handful of parity computations
/// with no transcendental functions anywhere.
///
/// # Panics
/// Panics unless `recv` has one entry per column.
#[must_use]
pub fn ldpc_decode_bitflip(h: &Gf2Matrix, recv: &[bool], iters: usize) -> Vec<bool> {
    assert_eq!(recv.len(), h.cols, "one bit per position is required");
    let (m, n) = (h.rows, h.cols);
    let mut x = recv.to_vec();
    // Flipping is not monotone: a round can leave more checks unsatisfied
    // than it found, and with many bits tied for the worst it can oscillate
    // between two states forever. Keeping the best iterate seen turns that
    // from a failure into a plateau.
    let mut best_x = x.clone();
    let mut best_unsatisfied = weight(&h.mul_vec(&x));
    for _ in 0..iters {
        let s = h.mul_vec(&x);
        let unsatisfied = weight(&s);
        if unsatisfied == 0 {
            return x;
        }
        if unsatisfied < best_unsatisfied {
            best_unsatisfied = unsatisfied;
            best_x = x.clone();
        }
        let mut votes = vec![0usize; n];
        for r in 0..m {
            if s[r] {
                for c in 0..n {
                    if h.get(r, c) {
                        votes[c] += 1;
                    }
                }
            }
        }
        let best = votes.iter().copied().max().unwrap_or(0);
        if best == 0 {
            break;
        }
        for c in 0..n {
            if votes[c] == best {
                x[c] = !x[c];
            }
        }
    }
    if weight(&h.mul_vec(&x)) <= best_unsatisfied {
        x
    } else {
        best_x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pick(rng: &mut Rng, n: usize) -> usize {
        ((u128::from(rng.next_u64()) * n as u128) >> 64) as usize
    }

    fn random_matrix(rows: usize, cols: usize, rng: &mut Rng) -> Gf2Matrix {
        let mut m = Gf2Matrix::zeros(rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                if rng.next_u64() & 1 == 1 {
                    m.set(r, c, true);
                }
            }
        }
        m
    }

    fn random_vec(n: usize, rng: &mut Rng) -> Vec<bool> {
        (0..n).map(|_| rng.next_u64() & 1 == 1).collect()
    }

    /// The linear algebra the whole module rests on: rank-nullity, the
    /// kernel really being the kernel, the product agreeing with the
    /// definition, and solve returning a solution exactly when one exists.
    #[test]
    fn gf2_linear_algebra_holds() {
        let mut rng = Rng::new(0x_6F20);
        for _ in 0..300 {
            let rows = 1 + pick(&mut rng, 12);
            let cols = 1 + pick(&mut rng, 12);
            let m = random_matrix(rows, cols, &mut rng);

            // The product against the definition, entry by entry.
            let other = random_matrix(cols, 1 + pick(&mut rng, 10), &mut rng);
            let p = m.mul(&other);
            for i in 0..p.rows {
                for j in 0..p.cols {
                    let want = (0..cols).filter(|&t| m.get(i, t) && other.get(t, j)).count() % 2 == 1;
                    assert_eq!(p.get(i, j), want, "the product is wrong at ({i}, {j})");
                }
            }
            // Transposition is an involution and reverses products.
            assert_eq!(m.transpose().transpose(), m);
            assert_eq!(p.transpose(), other.transpose().mul(&m.transpose()));

            // Rank-nullity, and the kernel vectors really lying in it.
            let rank = m.rank();
            let kernel = m.kernel_basis();
            assert_eq!(kernel.len(), cols - rank, "rank-nullity fails");
            for v in &kernel {
                assert!(m.mul_vec(v).iter().all(|&b| !b), "a kernel vector is not in the kernel");
            }
            // The basis is independent: its own rank is its size.
            if !kernel.is_empty() {
                assert_eq!(Gf2Matrix::from_rows(&kernel).rank(), kernel.len());
            }
            // Reduced echelon form is idempotent and preserves rank.
            let (e, pivots) = m.rref();
            assert_eq!(pivots.len(), rank);
            assert_eq!(e.rref().0, e, "rref is not idempotent");
            // A pivot column has a single one, in its own row.
            for (row, &c) in pivots.iter().enumerate() {
                for i in 0..rows {
                    assert_eq!(e.get(i, c), i == row, "pivot column {c} is not cleared");
                }
            }

            // Solving: a right-hand side taken from the column space always
            // has a solution, and one outside it never does.
            let x = random_vec(cols, &mut rng);
            let b = m.mul_vec(&x);
            let found = m.solve(&b).expect("a right-hand side from the column space is solvable");
            assert_eq!(m.mul_vec(&found), b, "solve returned a non-solution");
            let random_b = random_vec(rows, &mut rng);
            match m.solve(&random_b) {
                Some(y) => assert_eq!(m.mul_vec(&y), random_b),
                None => {
                    // Unsolvable means the augmented system has higher rank,
                    // which is the Rouche-Capelli criterion.
                    let mut aug = Gf2Matrix::zeros(rows, cols + 1);
                    for r in 0..rows {
                        for c in 0..cols {
                            if m.get(r, c) {
                                aug.set(r, c, true);
                            }
                        }
                        if random_b[r] {
                            aug.set(r, cols, true);
                        }
                    }
                    assert_eq!(aug.rank(), rank + 1, "declared unsolvable but the ranks agree");
                }
            }
        }
        assert_eq!(Gf2Matrix::identity(5).rank(), 5);
        assert!(Gf2Matrix::identity(5).kernel_basis().is_empty());
    }

    /// Hamming(7, 4) corrects every single error, checked over every one of
    /// the 16 by 8 possibilities rather than sampled.
    #[test]
    fn hamming_74_corrects_every_single_error_exhaustively() {
        for nibble in 0..16u8 {
            let word = hamming_74_encode(nibble);
            assert!(word < 128);
            assert_eq!(hamming_74_decode(word), (nibble, false), "a clean word was 'corrected'");
            for bit in 0..7 {
                let (got, corrected) = hamming_74_decode(word ^ (1 << bit));
                assert_eq!(got, nibble, "a single error in bit {bit} was not corrected");
                assert!(corrected, "the correction went unreported");
            }
        }
        // The sixteen codewords are pairwise at distance three or more, which
        // is what makes the above possible at all.
        let words: Vec<u8> = (0..16u8).map(hamming_74_encode).collect();
        for i in 0..16 {
            for j in i + 1..16 {
                assert!(
                    (words[i] ^ words[j]).count_ones() >= 3,
                    "codewords {i} and {j} are too close"
                );
            }
        }
        // And two errors are detected as a wrong correction, not silence:
        // every double error yields a non-zero syndrome.
        let mut misleads = 0;
        for nibble in 0..16u8 {
            let word = hamming_74_encode(nibble);
            for a in 0..7 {
                for b in a + 1..7 {
                    let (got, corrected) = hamming_74_decode(word ^ (1 << a) ^ (1 << b));
                    assert!(corrected, "a double error looked clean");
                    assert_ne!(got, nibble, "a double error was somehow corrected");
                    misleads += 1;
                }
            }
        }
        assert_eq!(misleads, 16 * 21);
    }

    /// A named code and the parameters it is named for.
    fn named() -> Vec<(&'static str, LinearCode, usize, usize, usize)> {
        vec![
            ("Hamming(2)", LinearCode::hamming(2), 3, 1, 3),
            ("Hamming(3)", LinearCode::hamming(3), 7, 4, 3),
            ("Hamming(4)", LinearCode::hamming(4), 15, 11, 3),
            ("extended Hamming(3)", LinearCode::extended_hamming(3), 8, 4, 4),
            ("extended Hamming(4)", LinearCode::extended_hamming(4), 16, 11, 4),
            ("repetition(5)", LinearCode::repetition(5), 5, 1, 5),
            ("repetition(8)", LinearCode::repetition(8), 8, 1, 8),
            ("parity check(6)", LinearCode::parity_check(6), 6, 5, 2),
            ("Golay(23)", LinearCode::golay23(), 23, 12, 7),
            ("Golay(24)", LinearCode::golay24(), 24, 12, 8),
            ("RM(1, 3)", LinearCode::reed_muller(1, 3), 8, 4, 4),
            ("RM(1, 4)", LinearCode::reed_muller(1, 4), 16, 5, 8),
            ("RM(2, 4)", LinearCode::reed_muller(2, 4), 16, 11, 4),
            ("RM(2, 5)", LinearCode::reed_muller(2, 5), 32, 16, 8),
        ]
    }

    /// Every named code has the length, dimension and distance it is named
    /// for, and its two matrices describe the same subspace.
    #[test]
    fn named_codes_have_their_stated_parameters() {
        for (name, c, n, k, d) in named() {
            assert_eq!((c.n, c.k, c.d), (n, k, d), "{name} has the wrong parameters");
            assert_eq!(c.g.rows, k);
            assert_eq!(c.h.rows, n - k, "{name}: the check matrix has the wrong height");
            // G H' = 0: every generator row is orthogonal to every check row,
            // which is what makes the two descriptions agree.
            assert!(c.g.mul(&c.h.transpose()).data.iter().all(|&w| w == 0), "{name}: G H' is not zero");
            // And the syndrome vanishes exactly on the code.
            for word in c.codewords() {
                assert!(c.contains(&word), "{name}: a codeword has a non-zero syndrome");
            }
            assert_eq!(c.h.rank(), n - k, "{name}: the check matrix is not full rank");
        }
        // Reed-Muller interpolates between repetition and single parity.
        for m in 2..=4usize {
            let low = LinearCode::reed_muller(0, m);
            assert_eq!((low.n, low.k, low.d), (1 << m, 1, 1 << m));
            let high = LinearCode::reed_muller(m - 1, m);
            assert_eq!((high.n, high.k, high.d), (1 << m, (1 << m) - 1, 2));
            // RM(r, m) has dimension the sum of binomials up to r.
            for r in 0..=m {
                let c = LinearCode::reed_muller(r, m);
                let want: usize = (0..=r)
                    .map(|i| crate::discrete::combinatorics::binomial_u64(m as u64, i as u64).unwrap() as usize)
                    .sum();
                assert_eq!(c.k, want, "RM({r}, {m}) has the wrong dimension");
                assert_eq!(c.d, 1 << (m - r), "RM({r}, {m}) has the wrong distance");
            }
        }
    }

    /// The Hamming and Golay codes are perfect: spheres of the correction
    /// radius around their codewords tile the space with nothing left over.
    #[test]
    fn the_perfect_codes_tile_the_space() {
        let volume = |n: usize, t: usize| -> u128 {
            (0..=t)
                .map(|i| u128::from(crate::discrete::combinatorics::binomial_u64(n as u64, i as u64).unwrap()))
                .sum()
        };
        for (name, _c, n, k, d) in named() {
            let t = (d - 1) / 2;
            let packed = (1u128 << k) * volume(n, t);
            let space = 1u128 << n;
            assert!(packed <= space, "{name} packs more than the space holds");
            let perfect = packed == space;
            let expected = name.starts_with("Hamming")
                || name == "Golay(23)"
                || name.starts_with("repetition") && n % 2 == 1;
            assert_eq!(perfect, expected, "{name}: perfection is not where it should be");
        }
        // Being perfect means every syndrome has a coset leader of weight at
        // most t, which is the same statement counted the other way.
        for c in [LinearCode::hamming(4), LinearCode::golay23()] {
            let t = (c.d - 1) / 2;
            let table = c.syndrome_table_small();
            assert_eq!(table.len(), 1usize << (c.n - c.k), "the table is not full");
            assert!(
                table.values().all(|e| weight(e) <= t),
                "a coset leader is heavier than the correction radius"
            );
        }
        // The extended Golay code is not perfect, and its covering radius is
        // one past its correction radius: some coset needs weight four.
        let g24 = LinearCode::golay24();
        let heaviest = g24.syndrome_table_small().values().map(|e| weight(e)).max().unwrap();
        assert_eq!(heaviest, 4, "the extended Golay covering radius is four");
    }

    /// Syndrome decoding corrects every error the distance promises, and the
    /// incremental search agrees with the explicit standard array.
    #[test]
    fn syndrome_decoding_corrects_up_to_the_radius() {
        let mut rng = Rng::new(0x_5943);
        for (name, c, n, k, d) in named() {
            if n > 24 {
                continue;
            }
            let t = (d - 1) / 2;
            let mut cross_checked = false;
            for _ in 0..12 {
                let msg = random_vec(k, &mut rng);
                let sent = c.encode(&msg);
                assert!(c.contains(&sent));
                for w in 0..=t {
                    let mut positions = std::collections::BTreeSet::new();
                    while positions.len() < w {
                        positions.insert(pick(&mut rng, n));
                    }
                    let mut recv = sent.clone();
                    for &i in &positions {
                        recv[i] = !recv[i];
                    }
                    let (fixed, corrected) = c.decode_syndrome(&recv);
                    assert_eq!(fixed, sent, "{name} failed on {w} errors");
                    assert_eq!(corrected, w, "{name} reported the wrong error count");
                    if !cross_checked {
                        // The two decoders find the same coset leader by
                        // different routes: one searches, one tabulates.
                        // Building the whole table is expensive, so this runs
                        // once per code rather than once per injected error.
                        assert_eq!(
                            c.standard_array_decode_small(&recv),
                            (fixed.clone(), corrected),
                            "{name}: the two decoders disagree"
                        );
                        cross_checked = true;
                    }
                }
            }
            // Beyond the radius on a perfect code, decoding must land on some
            // other codeword: there is nowhere else for it to land.
            if (1u128 << k)
                * (0..=t)
                    .map(|i| u128::from(crate::discrete::combinatorics::binomial_u64(n as u64, i as u64).unwrap()))
                    .sum::<u128>()
                == 1u128 << n
                && t < n
            {
                let sent = c.encode(&random_vec(k, &mut rng));
                let mut positions = std::collections::BTreeSet::new();
                while positions.len() < t + 1 {
                    positions.insert(pick(&mut rng, n));
                }
                let mut recv = sent.clone();
                for &i in &positions {
                    recv[i] = !recv[i];
                }
                let (fixed, _) = c.decode_syndrome(&recv);
                assert!(c.contains(&fixed), "{name} decoded to a non-codeword");
                assert_ne!(fixed, sent, "{name} corrected past its radius on a perfect code");
            }
        }
    }

    /// Duality, and MacWilliams's identity connecting a code's weight
    /// enumerator to its dual's.
    ///
    /// The identity is the strongest single statement available about a
    /// linear code's structure: the dual's weight distribution is determined
    /// by the code's, through the Krawtchouk transform, with no reference to
    /// either code's actual words. Here both sides are computed -- one by
    /// enumerating the dual, one by transforming the primal -- and required
    /// to agree exactly.
    #[test]
    fn duality_and_macwilliams() {
        let krawtchouk = |k: i64, x: i64, n: i64| -> i128 {
            (0..=k)
                .map(|i| {
                    let a = crate::discrete::combinatorics::binomial_u64(x as u64, i as u64)
                        .map_or(0i128, i128::from);
                    let b = crate::discrete::combinatorics::binomial_u64((n - x) as u64, (k - i) as u64)
                        .map_or(0i128, i128::from);
                    if i % 2 == 0 { a * b } else { -(a * b) }
                })
                .sum()
        };
        for (name, c, n, k, _) in named() {
            if n > 24 || n - k > 20 {
                continue;
            }
            let dual = c.dual();
            assert_eq!(dual.n, n, "{name}: the dual has a different length");
            assert_eq!(dual.k, n - k, "{name}: the dual has the wrong dimension");
            // Duality is an involution.
            let back = dual.dual();
            assert_eq!(back.k, k);
            let mut mine: Vec<Vec<bool>> = c.codewords();
            let mut theirs: Vec<Vec<bool>> = back.codewords();
            mine.sort();
            theirs.sort();
            assert_eq!(mine, theirs, "{name}: the double dual is a different code");

            // MacWilliams.
            let a: Vec<i128> = c
                .weight_enumerator()
                .iter()
                .map(|x| x.to_string_radix(10).parse::<i128>().expect("fits"))
                .collect();
            let b: Vec<i128> = dual
                .weight_enumerator()
                .iter()
                .map(|x| x.to_string_radix(10).parse::<i128>().expect("fits"))
                .collect();
            let size = 1i128 << k;
            for j in 0..=n {
                let transformed: i128 = (0..=n)
                    .map(|i| a[i] * krawtchouk(j as i64, i as i64, n as i64))
                    .sum::<i128>()
                    / size;
                assert_eq!(transformed, b[j], "{name}: MacWilliams fails at weight {j}");
            }
        }
        // The extended Golay code is its own dual, and the repetition code's
        // dual is the single parity check code of the same length.
        assert!(LinearCode::golay24().is_self_dual());
        assert!(!LinearCode::golay23().is_self_dual());
        for n in 2..=8usize {
            let dual = LinearCode::repetition(n).dual();
            let parity = LinearCode::parity_check(n);
            assert_eq!((dual.n, dual.k, dual.d), (parity.n, parity.k, parity.d));
            let mut a = dual.codewords();
            let mut b = parity.codewords();
            a.sort();
            b.sort();
            assert_eq!(a, b, "the dual of repetition is not the parity check code");
        }
    }

    /// The weight enumerator counts the code, and reproduces the published
    /// distribution of the extended Golay code.
    #[test]
    fn weight_enumerators_count_the_code() {
        for (name, c, n, k, d) in named() {
            let a = c.weight_enumerator();
            assert_eq!(a.len(), n + 1);
            let total: BigInt = a.iter().fold(BigInt::zero(), |acc, x| acc.add(x));
            assert_eq!(total.to_string_radix(10), (1u64 << k).to_string(), "{name}: wrong total");
            assert_eq!(a[0].to_string_radix(10), "1", "{name}: the zero word is not unique");
            for (w, count) in a.iter().enumerate().take(d).skip(1) {
                assert_eq!(count.to_string_radix(10), "0", "{name}: a word of weight {w} exists");
            }
            assert_ne!(a[d].to_string_radix(10), "0", "{name}: nothing achieves the distance");
        }
        // The extended Golay code's distribution is the classical one, and
        // every weight in it is a multiple of four.
        let a: Vec<String> =
            LinearCode::golay24().weight_enumerator().iter().map(|x| x.to_string_radix(10)).collect();
        for (w, count) in a.iter().enumerate() {
            let want = match w {
                0 | 24 => "1",
                8 | 16 => "759",
                12 => "2576",
                _ => "0",
            };
            assert_eq!(count, want, "the Golay(24) distribution is wrong at weight {w}");
        }
    }

    /// The bounds, against the codes that meet them.
    #[test]
    fn bounds_bracket_the_named_codes() {
        for (name, _c, n, k, d) in named() {
            assert!(d <= singleton_bound(n, k), "{name} beats Singleton");
            let size = (1u64 << k) as f64;
            assert!(size <= hamming_bound(n, d) + 1e-6, "{name} beats the sphere-packing bound");
            // The Gilbert-Varshamov bound is a promise that something exists,
            // so it can never exceed what the sphere packing allows.
            assert!(gilbert_varshamov(n, d) <= hamming_bound(n, d) + 1e-6);
            if 2 * d > n {
                assert!(size <= plotkin_bound(n, d) + 1e-6, "{name} beats Plotkin");
            }
        }
        // The repetition code of length n meets Singleton and Plotkin at
        // once: two codewords, distance n.
        for n in 2..=10usize {
            assert_eq!(LinearCode::repetition(n).d, singleton_bound(n, 1));
            assert!((plotkin_bound(n, n) - 2.0).abs() < 1e-12);
        }
        // Hamming and Golay meet the sphere-packing bound exactly, which is
        // what perfection means.
        for (name, c, n, _, d) in named() {
            let meets = ((1u64 << c.k) as f64 - hamming_bound(n, d)).abs() < 1e-6;
            if name.starts_with("Hamming") || name == "Golay(23)" {
                assert!(meets, "{name} should be perfect");
            }
        }
        assert!(plotkin_bound(10, 3).is_infinite(), "Plotkin says nothing when 2d <= n");
    }

    /// The LDPC construction is regular, and belief propagation beats bit
    /// flipping at the same noise -- which is the whole reason to carry soft
    /// information through the decoder.
    #[test]
    fn ldpc_is_regular_and_belief_propagation_beats_bit_flipping() {
        let mut rng = Rng::new(0x_1DBC);
        let (n, wc, wr) = (504usize, 3usize, 6usize);
        let h = ldpc_regular(n, wc, wr, &mut rng);
        assert_eq!(h.rows, n * wc / wr);
        for c in 0..n {
            let ones = (0..h.rows).filter(|&r| h.get(r, c)).count();
            assert_eq!(ones, wc, "column {c} has the wrong weight");
        }
        for r in 0..h.rows {
            let ones = (0..n).filter(|&c| h.get(r, c)).count();
            assert_eq!(ones, wr, "row {r} has the wrong weight");
        }
        // The all-zero word is a codeword of every linear code, and the code
        // here is the kernel of H, so it is what gets transmitted. Linearity
        // makes that no loss of generality: the error probability of a
        // symmetric channel does not depend on what was sent.
        let zero = vec![false; n];
        assert!(h.mul_vec(&zero).iter().all(|&b| !b));
        // With no noise at all, both decoders must return what was sent.
        assert_eq!(ldpc_decode_bp(&h, &vec![4.0; n], 30), (zero.clone(), true));
        assert_eq!(ldpc_decode_bitflip(&h, &zero, 30), zero);

        // A binary symmetric channel, at two crossover probabilities.
        let trials = 20;
        let mut summary = Vec::new();
        for p in [0.02f64, 0.05] {
            let mut raw = 0usize;
            let mut bp_errors = 0usize;
            let mut flip_errors = 0usize;
            let mut converged = 0usize;
            for _ in 0..trials {
                let recv: Vec<bool> = (0..n).map(|_| rng.next_f64() < p).collect();
                raw += weight(&recv);
                // The log-likelihood a symmetric channel of that crossover
                // implies: one magnitude for every bit, signed by what came
                // out of the channel.
                let mag = ((1.0 - p) / p).ln();
                let llr: Vec<f64> = recv.iter().map(|&b| if b { -mag } else { mag }).collect();
                let (bp, ok) = ldpc_decode_bp(&h, &llr, 60);
                bp_errors += weight(&bp);
                converged += usize::from(ok);
                flip_errors += weight(&ldpc_decode_bitflip(&h, &recv, 60));
            }
            summary.push((p, raw, bp_errors, flip_errors, converged));
        }
        // Both decoders have a threshold: a crossover probability below which
        // they clean the block up and above which they do not. Belief
        // propagation's is the higher, and the two probabilities here sit on
        // either side of bit flipping's, so the gap shows as a difference in
        // kind rather than a difference of a few per cent.
        for &(p, raw, bp_errors, flip_errors, converged) in &summary {
            assert!(raw > 100, "the channel at {p} was too quiet to compare decoders");
            assert!(bp_errors * 20 < raw, "belief propagation at {p} left {bp_errors} of {raw}");
            assert!(
                converged * 2 > trials,
                "only {converged} of {trials} blocks converged at {p}"
            );
            assert!(
                bp_errors < flip_errors,
                "at {p}, belief propagation left {bp_errors} and bit flipping {flip_errors}"
            );
        }
        // Below bit flipping's threshold it is a real decoder in its own
        // right, removing most of the errors on hard decisions alone.
        let (_, raw_low, _, flip_low, _) = summary[0];
        assert!(
            flip_low * 4 < raw_low,
            "bit flipping left {flip_low} of {raw_low} at the low crossover"
        );
        // Above it, it stalls, while belief propagation carries on.
        let (_, raw_high, bp_high, flip_high, _) = summary[1];
        assert!(
            flip_high * 4 > raw_high,
            "bit flipping was expected to stall at the high crossover, not clear it"
        );
        assert!(
            flip_high > 5 * bp_high.max(1),
            "the gap at the high crossover is only {flip_high} against {bp_high}"
        );
    }
}
