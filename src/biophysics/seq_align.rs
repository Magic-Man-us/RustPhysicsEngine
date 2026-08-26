//! Sequence alignment and the elementary sequence analysis around it.
//!
//! # What an alignment score means
//!
//! Every function here returns a score under an explicit [`Scoring`], and
//! the score is only comparable between alignments computed under the *same*
//! one. That is not pedantry: a gap penalty is a free parameter, and the
//! choice of it decides whether two sequences align as one long homology
//! with an insertion or as two short unrelated fragments. Where a function
//! returns an alignment as well as a score, the score is always the score of
//! that alignment under that scoring -- which the tests check directly,
//! since a dynamic program that reports a maximum it did not achieve is the
//! commonest way for one of these to be wrong.
//!
//! # Global, local and affine
//!
//! The three classical algorithms differ in one line of the recurrence each,
//! and the differences matter more than the similarity suggests.
//! Needleman-Wunsch aligns the sequences end to end; Smith-Waterman clamps
//! the score at zero so a poor prefix cannot drag a good local match below
//! the surface; Gotoh separates opening a gap from extending one, which is
//! what lets a single long insertion cost less than many short ones.

use crate::error::GeomError;
use std::collections::HashMap;

/// A substitution and gap scoring scheme.
#[derive(Debug, Clone)]
pub struct Scoring {
    /// Score for identical residues, when no matrix is supplied.
    pub match_score: i64,
    /// Score for differing residues, when no matrix is supplied.
    pub mismatch: i64,
    /// Cost of a single gap position, as a negative number.
    pub gap: i64,
    /// An optional substitution matrix indexed by residue code.
    ///
    /// When present it overrides `match_score` and `mismatch`, and residues
    /// outside its range fall back to them.
    pub matrix: Option<SubstitutionMatrix>,
}

impl Scoring {
    /// A simple scheme with no substitution matrix.
    #[must_use]
    pub fn simple(match_score: i64, mismatch: i64, gap: i64) -> Self {
        Self { match_score, mismatch, gap, matrix: None }
    }

    /// The score of substituting one residue for another.
    #[must_use]
    pub fn substitution(&self, a: u8, b: u8) -> i64 {
        if let Some(m) = &self.matrix {
            if let Some(value) = m.lookup(a, b) {
                return value;
            }
        }
        if a == b {
            self.match_score
        } else {
            self.mismatch
        }
    }
}

/// A named substitution matrix over an alphabet.
#[derive(Debug, Clone)]
pub struct SubstitutionMatrix {
    /// The residue letters, in the order the rows and columns use.
    pub alphabet: Vec<u8>,
    /// Row-major scores, `alphabet.len()` squared.
    pub scores: Vec<i8>,
}

impl SubstitutionMatrix {
    /// The score for a pair of residues, or `None` if either is outside the
    /// alphabet.
    #[must_use]
    pub fn lookup(&self, a: u8, b: u8) -> Option<i64> {
        let i = self.alphabet.iter().position(|c| *c == a)?;
        let j = self.alphabet.iter().position(|c| *c == b)?;
        Some(i64::from(self.scores[i * self.alphabet.len() + j]))
    }

    /// Whether the matrix is symmetric, as every substitution matrix
    /// derived from a symmetric alignment count must be.
    #[must_use]
    pub fn is_symmetric(&self) -> bool {
        let n = self.alphabet.len();
        (0..n).all(|i| (0..n).all(|j| self.scores[i * n + j] == self.scores[j * n + i]))
    }
}

/// The direction a traceback step came from.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Step {
    Diagonal,
    Up,
    Left,
    Stop,
}

/// Global alignment by Needleman-Wunsch.
///
/// Returns the optimal score and the two aligned strings, with `-` for gaps.
/// The alignment spans both sequences end to end, which is the right model
/// when the sequences are known to be homologous over their whole length and
/// the wrong one when only a domain is shared -- for that, see
/// [`smith_waterman`].
///
/// # Errors
/// Returns an error for a non-negative gap penalty, or sequences long enough
/// that the quadratic table would not fit; use [`hirschberg`] for those.
pub fn needleman_wunsch(
    a: &[u8],
    b: &[u8],
    score: &Scoring,
) -> Result<(i64, String, String), GeomError> {
    check_scoring(score)?;
    if a.len().saturating_mul(b.len()) > 64_000_000 {
        return Err(GeomError::InvalidArgument(
            "the quadratic table is too large; use hirschberg",
        ));
    }
    let (n, m) = (a.len(), b.len());
    let mut table = vec![0i64; (n + 1) * (m + 1)];
    let mut from = vec![Step::Stop; (n + 1) * (m + 1)];
    let at = |i: usize, j: usize| i * (m + 1) + j;
    for i in 1..=n {
        table[at(i, 0)] = score.gap * i as i64;
        from[at(i, 0)] = Step::Up;
    }
    for j in 1..=m {
        table[at(0, j)] = score.gap * j as i64;
        from[at(0, j)] = Step::Left;
    }
    for i in 1..=n {
        for j in 1..=m {
            let diagonal = table[at(i - 1, j - 1)] + score.substitution(a[i - 1], b[j - 1]);
            let up = table[at(i - 1, j)] + score.gap;
            let left = table[at(i, j - 1)] + score.gap;
            let (best, step) = if diagonal >= up && diagonal >= left {
                (diagonal, Step::Diagonal)
            } else if up >= left {
                (up, Step::Up)
            } else {
                (left, Step::Left)
            };
            table[at(i, j)] = best;
            from[at(i, j)] = step;
        }
    }
    let (top, bottom) = traceback(a, b, &from, n, m, m + 1, false);
    Ok((table[at(n, m)], top, bottom))
}

/// Walks the traceback table from `(i, j)` back to the origin, or to the
/// first `Stop` when `local` is set.
fn traceback(
    a: &[u8],
    b: &[u8],
    from: &[Step],
    mut i: usize,
    mut j: usize,
    stride: usize,
    local: bool,
) -> (String, String) {
    let mut top = Vec::new();
    let mut bottom = Vec::new();
    while i > 0 || j > 0 {
        let step = from[i * stride + j];
        if local && step == Step::Stop {
            break;
        }
        match step {
            Step::Diagonal => {
                top.push(a[i - 1]);
                bottom.push(b[j - 1]);
                i -= 1;
                j -= 1;
            }
            Step::Up => {
                top.push(a[i - 1]);
                bottom.push(b'-');
                i -= 1;
            }
            Step::Left => {
                top.push(b'-');
                bottom.push(b[j - 1]);
                j -= 1;
            }
            Step::Stop => break,
        }
    }
    top.reverse();
    bottom.reverse();
    (String::from_utf8_lossy(&top).into_owned(), String::from_utf8_lossy(&bottom).into_owned())
}

/// Local alignment by Smith-Waterman.
///
/// Returns `(score, start in a, start in b, aligned a, aligned b)`.
///
/// The single change from Needleman-Wunsch -- clamping each cell at zero --
/// is what makes it local: a prefix that aligns badly is discarded rather
/// than carried, so a strong internal match is found whatever surrounds it.
/// The score is therefore never negative, and an alignment of two unrelated
/// sequences reports a small positive score rather than a large negative
/// one, which is why local scores need a significance model and global ones
/// less so.
///
/// # Errors
/// Returns an error on the same conditions as [`needleman_wunsch`].
pub fn smith_waterman(
    a: &[u8],
    b: &[u8],
    score: &Scoring,
) -> Result<(i64, usize, usize, String, String), GeomError> {
    check_scoring(score)?;
    if a.len().saturating_mul(b.len()) > 64_000_000 {
        return Err(GeomError::InvalidArgument("the quadratic table is too large"));
    }
    let (n, m) = (a.len(), b.len());
    let mut table = vec![0i64; (n + 1) * (m + 1)];
    let mut from = vec![Step::Stop; (n + 1) * (m + 1)];
    let at = |i: usize, j: usize| i * (m + 1) + j;
    let (mut best, mut best_at) = (0i64, (0usize, 0usize));
    for i in 1..=n {
        for j in 1..=m {
            let diagonal = table[at(i - 1, j - 1)] + score.substitution(a[i - 1], b[j - 1]);
            let up = table[at(i - 1, j)] + score.gap;
            let left = table[at(i, j - 1)] + score.gap;
            let (value, step) = if diagonal >= up && diagonal >= left {
                (diagonal, Step::Diagonal)
            } else if up >= left {
                (up, Step::Up)
            } else {
                (left, Step::Left)
            };
            if value > 0 {
                table[at(i, j)] = value;
                from[at(i, j)] = step;
            } else {
                table[at(i, j)] = 0;
                from[at(i, j)] = Step::Stop;
            }
            if table[at(i, j)] > best {
                best = table[at(i, j)];
                best_at = (i, j);
            }
        }
    }
    let (top, bottom) = traceback(a, b, &from, best_at.0, best_at.1, m + 1, true);
    // The start positions are the end positions less the aligned lengths.
    let consumed_a = top.bytes().filter(|c| *c != b'-').count();
    let consumed_b = bottom.bytes().filter(|c| *c != b'-').count();
    Ok((best, best_at.0 - consumed_a, best_at.1 - consumed_b, top, bottom))
}

/// Global alignment with affine gap penalties, by Gotoh's algorithm.
///
/// A gap of length `k` costs `open + k * extend` rather than `k * gap`, so a
/// single long insertion is cheap relative to many short ones. That is the
/// biologically right shape -- one indel event of twenty residues is far
/// more likely than twenty separate ones -- and it is why affine gaps are
/// the default in practice despite costing three tables instead of one.
///
/// With `open = 0` the model degenerates to linear gaps and the result must
/// agree with [`needleman_wunsch`] at `gap = extend`, which the tests check.
///
/// # Errors
/// Returns an error for a positive gap penalty or an oversized table.
pub fn gotoh_affine(
    a: &[u8],
    b: &[u8],
    match_score: i64,
    mismatch: i64,
    gap_open: i64,
    gap_extend: i64,
) -> Result<(i64, String, String), GeomError> {
    if gap_open > 0 || gap_extend >= 0 {
        return Err(GeomError::InvalidArgument(
            "the gap open cost must be non-positive and the extend cost negative",
        ));
    }
    if a.len().saturating_mul(b.len()) > 64_000_000 {
        return Err(GeomError::InvalidArgument("the quadratic table is too large"));
    }
    let (n, m) = (a.len(), b.len());
    let at = |i: usize, j: usize| i * (m + 1) + j;
    let floor = i64::MIN / 4;
    // `best` ends in a match, `up` in a gap in b, `left` in a gap in a.
    let mut best = vec![floor; (n + 1) * (m + 1)];
    let mut up = vec![floor; (n + 1) * (m + 1)];
    let mut left = vec![floor; (n + 1) * (m + 1)];
    best[at(0, 0)] = 0;
    for i in 1..=n {
        up[at(i, 0)] = gap_open + gap_extend * i as i64;
        best[at(i, 0)] = up[at(i, 0)];
    }
    for j in 1..=m {
        left[at(0, j)] = gap_open + gap_extend * j as i64;
        best[at(0, j)] = left[at(0, j)];
    }
    for i in 1..=n {
        for j in 1..=m {
            let substitution = if a[i - 1] == b[j - 1] { match_score } else { mismatch };
            up[at(i, j)] = (up[at(i - 1, j)] + gap_extend)
                .max(best[at(i - 1, j)] + gap_open + gap_extend);
            left[at(i, j)] = (left[at(i, j - 1)] + gap_extend)
                .max(best[at(i, j - 1)] + gap_open + gap_extend);
            best[at(i, j)] =
                (best[at(i - 1, j - 1)] + substitution).max(up[at(i, j)]).max(left[at(i, j)]);
        }
    }
    // Traceback across the three tables, tracking which one we are in.
    let (mut i, mut j) = (n, m);
    let mut top = Vec::new();
    let mut bottom = Vec::new();
    #[derive(Clone, Copy, PartialEq)]
    enum Layer {
        Best,
        Up,
        Left,
    }
    let mut layer = Layer::Best;
    while i > 0 || j > 0 {
        match layer {
            Layer::Best => {
                if i > 0 && j > 0 {
                    let substitution = if a[i - 1] == b[j - 1] { match_score } else { mismatch };
                    if best[at(i, j)] == best[at(i - 1, j - 1)] + substitution {
                        top.push(a[i - 1]);
                        bottom.push(b[j - 1]);
                        i -= 1;
                        j -= 1;
                        continue;
                    }
                }
                if i > 0 && best[at(i, j)] == up[at(i, j)] {
                    layer = Layer::Up;
                } else {
                    layer = Layer::Left;
                }
            }
            Layer::Up => {
                top.push(a[i - 1]);
                bottom.push(b'-');
                let continued = i > 1 && up[at(i, j)] == up[at(i - 1, j)] + gap_extend;
                i -= 1;
                if !continued {
                    layer = Layer::Best;
                }
            }
            Layer::Left => {
                top.push(b'-');
                bottom.push(b[j - 1]);
                let continued = j > 1 && left[at(i, j)] == left[at(i, j - 1)] + gap_extend;
                j -= 1;
                if !continued {
                    layer = Layer::Best;
                }
            }
        }
    }
    top.reverse();
    bottom.reverse();
    Ok((
        best[at(n, m)],
        String::from_utf8_lossy(&top).into_owned(),
        String::from_utf8_lossy(&bottom).into_owned(),
    ))
}

/// The global alignment score restricted to a diagonal band.
///
/// Only cells with `|i - j| <= band` are computed, so the cost is
/// `O(n * band)` rather than `O(n * m)`. The result is the true optimum only
/// when the optimal alignment stays inside the band -- which is why this is
/// a heuristic for similar sequences rather than a general algorithm, and
/// why a band wide enough to contain the whole table must reproduce
/// [`needleman_wunsch`] exactly.
///
/// # Errors
/// Returns an error for a non-negative gap penalty, or a band too narrow to
/// reach the far corner.
pub fn banded_alignment(a: &[u8], b: &[u8], band: usize, score: &Scoring) -> Result<i64, GeomError> {
    check_scoring(score)?;
    let (n, m) = (a.len(), b.len());
    if band < n.abs_diff(m) {
        return Err(GeomError::InvalidArgument(
            "the band is narrower than the length difference, so no alignment fits",
        ));
    }
    let floor = i64::MIN / 4;
    let in_band = |i: usize, j: usize| i.abs_diff(j) <= band;
    let at = |i: usize, j: usize| i * (m + 1) + j;
    let mut table = vec![floor; (n + 1) * (m + 1)];
    table[at(0, 0)] = 0;
    for i in 0..=n {
        for j in 0..=m {
            if !in_band(i, j) || (i == 0 && j == 0) {
                continue;
            }
            let mut best = floor;
            if i > 0 && j > 0 && table[at(i - 1, j - 1)] > floor {
                best = best.max(table[at(i - 1, j - 1)] + score.substitution(a[i - 1], b[j - 1]));
            }
            if i > 0 && in_band(i - 1, j) && table[at(i - 1, j)] > floor {
                best = best.max(table[at(i - 1, j)] + score.gap);
            }
            if j > 0 && in_band(i, j - 1) && table[at(i, j - 1)] > floor {
                best = best.max(table[at(i, j - 1)] + score.gap);
            }
            table[at(i, j)] = best;
        }
    }
    Ok(table[at(n, m)])
}

/// Global alignment in linear space, by Hirschberg's divide and conquer.
///
/// The score of a global alignment can be computed in `O(min(n, m))` space
/// by keeping two rows, but the *traceback* seems to need the whole table.
/// Hirschberg's observation is that the optimal alignment must cross the
/// middle row somewhere, that the crossing point can be found from two
/// linear-space score passes -- one forward, one backward -- and that the
/// problem then splits in two. The cost is a constant factor more time for
/// an asymptotic saving in space, which is the trade that makes whole-genome
/// alignment possible at all.
///
/// The alignment it returns is optimal, so its score must equal
/// [`needleman_wunsch`]'s; the tests check exactly that.
///
/// # Errors
/// Returns an error for a non-negative gap penalty.
pub fn hirschberg(a: &[u8], b: &[u8], score: &Scoring) -> Result<(String, String), GeomError> {
    check_scoring(score)?;
    Ok(hirschberg_inner(a, b, score))
}

/// The last row of the Needleman-Wunsch table, in linear space.
fn score_row(a: &[u8], b: &[u8], score: &Scoring) -> Vec<i64> {
    let m = b.len();
    let mut previous: Vec<i64> = (0..=m).map(|j| score.gap * j as i64).collect();
    let mut current = vec![0i64; m + 1];
    for (i, x) in a.iter().enumerate() {
        current[0] = score.gap * (i as i64 + 1);
        for j in 1..=m {
            current[j] = (previous[j - 1] + score.substitution(*x, b[j - 1]))
                .max(previous[j] + score.gap)
                .max(current[j - 1] + score.gap);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous
}

fn hirschberg_inner(a: &[u8], b: &[u8], score: &Scoring) -> (String, String) {
    if a.is_empty() {
        return ("-".repeat(b.len()), String::from_utf8_lossy(b).into_owned());
    }
    if b.is_empty() {
        return (String::from_utf8_lossy(a).into_owned(), "-".repeat(a.len()));
    }
    if a.len() == 1 || b.len() == 1 {
        // Small enough that the quadratic table is trivial.
        let (_, top, bottom) = needleman_wunsch(a, b, score).expect("small alignment");
        return (top, bottom);
    }
    let half = a.len() / 2;
    let forward = score_row(&a[..half], b, score);
    // The reverse pass, on reversed halves.
    let tail: Vec<u8> = a[half..].iter().rev().copied().collect();
    let reversed_b: Vec<u8> = b.iter().rev().copied().collect();
    let backward = score_row(&tail, &reversed_b, score);
    // The crossing column maximises the sum of the two.
    let m = b.len();
    let mut best = (0usize, i64::MIN);
    for j in 0..=m {
        let total = forward[j] + backward[m - j];
        if total > best.1 {
            best = (j, total);
        }
    }
    let split = best.0;
    let (left_top, left_bottom) = hirschberg_inner(&a[..half], &b[..split], score);
    let (right_top, right_bottom) = hirschberg_inner(&a[half..], &b[split..], score);
    (left_top + &right_top, left_bottom + &right_bottom)
}

fn check_scoring(score: &Scoring) -> Result<(), GeomError> {
    if score.gap >= 0 {
        return Err(GeomError::InvalidArgument("the gap penalty must be negative"));
    }
    if let Some(m) = &score.matrix {
        if m.alphabet.is_empty() || m.scores.len() != m.alphabet.len() * m.alphabet.len() {
            return Err(GeomError::InvalidArgument("the substitution matrix is malformed"));
        }
    }
    Ok(())
}

/// The score of an alignment already made, under a scoring scheme.
///
/// Used to check that a dynamic program achieved the score it reported --
/// the commonest way for one of these to be wrong is to report a maximum it
/// did not actually reach.
///
/// Gaps are charged linearly, so this agrees with [`needleman_wunsch`] and
/// with [`gotoh_affine`] only when the latter's open cost is zero.
///
/// # Errors
/// Returns an error for alignments of differing length or a column of two
/// gaps, which no alignment should contain.
pub fn alignment_score(top: &str, bottom: &str, score: &Scoring) -> Result<i64, GeomError> {
    if top.len() != bottom.len() {
        return Err(GeomError::InvalidArgument("the aligned strings differ in length"));
    }
    let mut total = 0;
    for (x, y) in top.bytes().zip(bottom.bytes()) {
        match (x, y) {
            (b'-', b'-') => {
                return Err(GeomError::InvalidArgument("an alignment column holds two gaps"))
            }
            (b'-', _) | (_, b'-') => total += score.gap,
            _ => total += score.substitution(x, y),
        }
    }
    Ok(total)
}

/// The score of an alignment under affine gap penalties.
///
/// # Errors
/// Returns an error on the same conditions as [`alignment_score`].
pub fn alignment_score_affine(
    top: &str,
    bottom: &str,
    match_score: i64,
    mismatch: i64,
    gap_open: i64,
    gap_extend: i64,
) -> Result<i64, GeomError> {
    if top.len() != bottom.len() {
        return Err(GeomError::InvalidArgument("the aligned strings differ in length"));
    }
    let mut total = 0;
    // Which sequence the current run of gaps is in, so that a gap in one
    // followed immediately by a gap in the other is charged two openings.
    let mut open_in: Option<bool> = None;
    for (x, y) in top.bytes().zip(bottom.bytes()) {
        match (x, y) {
            (b'-', b'-') => {
                return Err(GeomError::InvalidArgument("an alignment column holds two gaps"))
            }
            (b'-', _) => {
                if open_in != Some(true) {
                    total += gap_open;
                    open_in = Some(true);
                }
                total += gap_extend;
            }
            (_, b'-') => {
                if open_in != Some(false) {
                    total += gap_open;
                    open_in = Some(false);
                }
                total += gap_extend;
            }
            _ => {
                open_in = None;
                total += if x == y { match_score } else { mismatch };
            }
        }
    }
    Ok(total)
}

// ---------------------------------------------------------------------------
// Substitution matrices
// ---------------------------------------------------------------------------

/// The twenty standard amino acids plus the ambiguity codes, in the order
/// the BLOSUM and PAM tables use.
const PROTEIN_ALPHABET: &[u8; 24] = b"ARNDCQEGHILKMFPSTWYVBZX*";

/// The BLOSUM62 substitution matrix.
///
/// Derived from blocks of aligned protein segments no more than 62 per cent
/// identical, which is what the number means -- a *higher* BLOSUM number is
/// built from more similar sequences and suits closer homologues, the
/// opposite of the intuition the name suggests. The diagonal is not
/// constant: a tryptophan match scores 11 and a leucine match 4, because
/// tryptophan is rare and its conservation is correspondingly more
/// informative.
#[must_use]
pub fn blosum62() -> SubstitutionMatrix {
    #[rustfmt::skip]
    const S: [i8; 576] = [
         4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0,-2,-1, 0,-4,
        -1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3,-1, 0,-1,-4,
        -2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3, 3, 0,-1,-4,
        -2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3, 4, 1,-1,-4,
         0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1,-3,-3,-2,-4,
        -1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2, 0, 3,-1,-4,
        -1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2, 1, 4,-1,-4,
         0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3,-1,-2,-1,-4,
        -2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3, 0, 0,-1,-4,
        -1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3,-3,-3,-1,-4,
        -1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1,-4,-3,-1,-4,
        -1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2, 0, 1,-1,-4,
        -1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1,-3,-1,-1,-4,
        -2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1,-3,-3,-1,-4,
        -1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2,-2,-1,-2,-4,
         1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2, 0, 0, 0,-4,
         0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0,-1,-1, 0,-4,
        -3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3,-4,-3,-2,-4,
        -2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1,-3,-2,-1,-4,
         0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4,-3,-2,-1,-4,
        -2,-1, 3, 4,-3, 0, 1,-1, 0,-3,-4, 0,-3,-3,-2, 0,-1,-4,-3,-3, 4, 1,-1,-4,
        -1, 0, 0, 1,-3, 3, 4,-2, 0,-3,-3, 1,-1,-3,-1, 0,-1,-3,-2,-2, 1, 4,-1,-4,
         0,-1,-1,-1,-2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-2, 0, 0,-2,-1,-1,-1,-1,-1,-4,
        -4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4, 1,
    ];
    SubstitutionMatrix { alphabet: PROTEIN_ALPHABET.to_vec(), scores: S.to_vec() }
}

/// The PAM250 substitution matrix.
///
/// Extrapolated from one per cent accepted mutations by raising the
/// substitution probability matrix to the 250th power, so it describes very
/// distant relationships -- the opposite end of the range from BLOSUM62. The
/// extrapolation is its weakness: errors in the one-per-cent estimates
/// compound over 250 multiplications, which is the reason BLOSUM, built
/// directly from distant alignments, generally does better at finding remote
/// homologues.
#[must_use]
pub fn pam250() -> SubstitutionMatrix {
    #[rustfmt::skip]
    const S: [i8; 576] = [
         2,-2, 0, 0,-2, 0, 0, 1,-1,-1,-2,-1,-1,-3, 1, 1, 1,-6,-3, 0, 0, 0, 0,-8,
        -2, 6, 0,-1,-4, 1,-1,-3, 2,-2,-3, 3, 0,-4, 0, 0,-1, 2,-4,-2,-1, 0,-1,-8,
         0, 0, 2, 2,-4, 1, 1, 0, 2,-2,-3, 1,-2,-3, 0, 1, 0,-4,-2,-2, 2, 1, 0,-8,
         0,-1, 2, 4,-5, 2, 3, 1, 1,-2,-4, 0,-3,-6,-1, 0, 0,-7,-4,-2, 3, 3,-1,-8,
        -2,-4,-4,-5,12,-5,-5,-3,-3,-2,-6,-5,-5,-4,-3, 0,-2,-8, 0,-2,-4,-5,-3,-8,
         0, 1, 1, 2,-5, 4, 2,-1, 3,-2,-2, 1,-1,-5, 0,-1,-1,-5,-4,-2, 1, 3,-1,-8,
         0,-1, 1, 3,-5, 2, 4, 0, 1,-2,-3, 0,-2,-5,-1, 0, 0,-7,-4,-2, 3, 3,-1,-8,
         1,-3, 0, 1,-3,-1, 0, 5,-2,-3,-4,-2,-3,-5, 0, 1, 0,-7,-5,-1, 0, 0,-1,-8,
        -1, 2, 2, 1,-3, 3, 1,-2, 6,-2,-2, 0,-2,-2, 0,-1,-1,-3, 0,-2, 1, 2,-1,-8,
        -1,-2,-2,-2,-2,-2,-2,-3,-2, 5, 2,-2, 2, 1,-2,-1, 0,-5,-1, 4,-2,-2,-1,-8,
        -2,-3,-3,-4,-6,-2,-3,-4,-2, 2, 6,-3, 4, 2,-3,-3,-2,-2,-1, 2,-3,-3,-1,-8,
        -1, 3, 1, 0,-5, 1, 0,-2, 0,-2,-3, 5, 0,-5,-1, 0, 0,-3,-4,-2, 1, 0,-1,-8,
        -1, 0,-2,-3,-5,-1,-2,-3,-2, 2, 4, 0, 6, 0,-2,-2,-1,-4,-2, 2,-2,-2,-1,-8,
        -3,-4,-3,-6,-4,-5,-5,-5,-2, 1, 2,-5, 0, 9,-5,-3,-3, 0, 7,-1,-4,-5,-2,-8,
         1, 0, 0,-1,-3, 0,-1, 0, 0,-2,-3,-1,-2,-5, 6, 1, 0,-6,-5,-1,-1, 0,-1,-8,
         1, 0, 1, 0, 0,-1, 0, 1,-1,-1,-3, 0,-2,-3, 1, 2, 1,-2,-3,-1, 0, 0, 0,-8,
         1,-1, 0, 0,-2,-1, 0, 0,-1, 0,-2, 0,-1,-3, 0, 1, 3,-5,-3, 0, 0,-1, 0,-8,
        -6, 2,-4,-7,-8,-5,-7,-7,-3,-5,-2,-3,-4, 0,-6,-2,-5,17, 0,-6,-5,-6,-4,-8,
        -3,-4,-2,-4, 0,-4,-4,-5, 0,-1,-1,-4,-2, 7,-5,-3,-3, 0,10,-2,-3,-4,-2,-8,
         0,-2,-2,-2,-2,-2,-2,-1,-2, 4, 2,-2, 2,-1,-1,-1, 0,-6,-2, 4,-2,-2,-1,-8,
         0,-1, 2, 3,-4, 1, 3, 0, 1,-2,-3, 1,-2,-4,-1, 0, 0,-5,-3,-2, 3, 2,-1,-8,
         0, 0, 1, 3,-5, 3, 3, 0, 2,-2,-3, 0,-2,-5, 0, 0,-1,-6,-4,-2, 2, 3,-1,-8,
         0,-1, 0,-1,-3,-1,-1,-1,-1,-1,-1,-1,-1,-2,-1, 0, 0,-4,-2,-1,-1,-1,-1,-8,
        -8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8, 1,
    ];
    SubstitutionMatrix { alphabet: PROTEIN_ALPHABET.to_vec(), scores: S.to_vec() }
}

// ---------------------------------------------------------------------------
// Sequence analysis
// ---------------------------------------------------------------------------

/// The fraction of G and C bases.
///
/// # Errors
/// Returns an error for an empty sequence.
pub fn gc_content(seq: &[u8]) -> Result<f64, GeomError> {
    if seq.is_empty() {
        return Err(GeomError::Empty);
    }
    let gc = seq
        .iter()
        .filter(|c| matches!(c.to_ascii_uppercase(), b'G' | b'C'))
        .count();
    Ok(gc as f64 / seq.len() as f64)
}

/// The reverse complement of a DNA sequence.
///
/// An involution: applying it twice returns the original, which is what
/// makes it a symmetry of double-stranded DNA rather than a transformation
/// of it. Unrecognised bases are passed through as `N`.
#[must_use]
pub fn reverse_complement(seq: &[u8]) -> Vec<u8> {
    seq.iter()
        .rev()
        .map(|c| match c.to_ascii_uppercase() {
            b'A' => b'T',
            b'T' | b'U' => b'A',
            b'G' => b'C',
            b'C' => b'G',
            _ => b'N',
        })
        .collect()
}

/// DNA to RNA: thymine becomes uracil.
#[must_use]
pub fn transcribe(seq: &[u8]) -> Vec<u8> {
    seq.iter()
        .map(|c| if c.eq_ignore_ascii_case(&b'T') { b'U' } else { c.to_ascii_uppercase() })
        .collect()
}

/// The amino acid a codon encodes, or `*` for a stop and `X` for anything
/// unrecognised.
#[must_use]
pub fn codon_to_amino(codon: &[u8]) -> u8 {
    if codon.len() != 3 {
        return b'X';
    }
    let c: Vec<u8> = codon
        .iter()
        .map(|x| if x.eq_ignore_ascii_case(&b'U') { b'T' } else { x.to_ascii_uppercase() })
        .collect();
    // The standard genetic code, written as the third-position groupings it
    // actually has: the code is degenerate mostly in the third base, which
    // is why a third-position change is usually silent.
    match (c[0], c[1], c[2]) {
        (b'T', b'T', b'T' | b'C') => b'F',
        (b'T', b'T', _) | (b'C', b'T', _) => b'L',
        (b'A', b'T', b'G') => b'M',
        (b'A', b'T', _) => b'I',
        (b'G', b'T', _) => b'V',
        (b'T', b'C', _) | (b'A', b'G', b'T' | b'C') => b'S',
        (b'C', b'C', _) => b'P',
        (b'A', b'C', _) => b'T',
        (b'G', b'C', _) => b'A',
        (b'T', b'A', b'T' | b'C') => b'Y',
        (b'T', b'A', _) | (b'T', b'G', b'A') => b'*',
        (b'C', b'A', b'T' | b'C') => b'H',
        (b'C', b'A', _) => b'Q',
        (b'A', b'A', b'T' | b'C') => b'N',
        (b'A', b'A', _) => b'K',
        (b'G', b'A', b'T' | b'C') => b'D',
        (b'G', b'A', _) => b'E',
        (b'T', b'G', b'T' | b'C') => b'C',
        (b'T', b'G', b'G') => b'W',
        (b'C', b'G', _) | (b'A', b'G', _) => b'R',
        (b'G', b'G', _) => b'G',
        _ => b'X',
    }
}

/// Translates a nucleotide sequence in frame zero, stopping at the first
/// stop codon.
#[must_use]
pub fn translate(seq: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(seq.len() / 3);
    for codon in seq.as_chunks::<3>().0 {
        let amino = codon_to_amino(codon);
        if amino == b'*' {
            break;
        }
        out.push(amino);
    }
    out
}

/// Open reading frames, as `(start, end, strand)` with the strand `+1` or
/// `-1` and positions on the forward strand.
///
/// Searches all six frames. `min_len` is in amino acids, excluding the stop.
///
/// # Errors
/// Returns an error for a zero minimum length, which would report every
/// start codon.
pub fn orf_find(seq: &[u8], min_len: usize) -> Result<Vec<(usize, usize, i8)>, GeomError> {
    if min_len == 0 {
        return Err(GeomError::InvalidArgument("the minimum length must be positive"));
    }
    let mut out = Vec::new();
    let reverse = reverse_complement(seq);
    for (strand, strand_seq) in [(1i8, seq), (-1i8, reverse.as_slice())] {
        for frame in 0..3usize {
            let mut position = frame;
            while position + 3 <= strand_seq.len() {
                if codon_to_amino(&strand_seq[position..position + 3]) == b'M' {
                    // Extend to the first in-frame stop.
                    let mut end = position + 3;
                    let mut length = 1usize;
                    let mut stopped = false;
                    while end + 3 <= strand_seq.len() {
                        if codon_to_amino(&strand_seq[end..end + 3]) == b'*' {
                            stopped = true;
                            break;
                        }
                        length += 1;
                        end += 3;
                    }
                    if stopped && length >= min_len {
                        let (a, b) = if strand == 1 {
                            (position, end + 3)
                        } else {
                            // Map back to forward-strand coordinates.
                            (strand_seq.len() - (end + 3), strand_seq.len() - position)
                        };
                        out.push((a, b, strand));
                        position = end + 3;
                        continue;
                    }
                }
                position += 3;
            }
        }
    }
    out.sort_unstable();
    Ok(out)
}

/// Codon usage counts as fractions, for codons appearing in frame zero.
///
/// # Errors
/// Returns an error for a sequence shorter than one codon.
pub fn codon_usage(seq: &[u8]) -> Result<Vec<(String, f64)>, GeomError> {
    if seq.len() < 3 {
        return Err(GeomError::InvalidArgument("the sequence is shorter than a codon"));
    }
    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut total = 0usize;
    for codon in seq.as_chunks::<3>().0 {
        let key = String::from_utf8_lossy(&codon.to_ascii_uppercase()).into_owned();
        *counts.entry(key).or_insert(0) += 1;
        total += 1;
    }
    let mut out: Vec<(String, f64)> =
        counts.into_iter().map(|(k, v)| (k, v as f64 / total as f64)).collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

/// The Wallace rule melting temperature: `2 (A + T) + 4 (G + C)` degrees.
///
/// Valid only for short oligonucleotides, roughly 14 to 20 bases. It ignores
/// concentration, salt and stacking entirely, which is why it disagrees with
/// [`tm_nearest_neighbor`] by ten degrees or more on anything longer -- the
/// stacking energy that the nearest-neighbour model accounts for is not a
/// correction at that length, it is most of the answer.
///
/// # Errors
/// Returns an error for an empty sequence.
pub fn melting_temperature_wallace(seq: &[u8]) -> Result<f64, GeomError> {
    if seq.is_empty() {
        return Err(GeomError::Empty);
    }
    let mut total = 0.0;
    for c in seq {
        total += match c.to_ascii_uppercase() {
            b'A' | b'T' | b'U' => 2.0,
            b'G' | b'C' => 4.0,
            _ => 0.0,
        };
    }
    Ok(total)
}

/// The nearest-neighbour melting temperature, in degrees Celsius.
///
/// `Tm = dH / (dS + R ln(C/4)) - 273.15`, with the enthalpy and entropy
/// summed over adjacent base pairs from the SantaLucia unified parameters.
/// The concentration enters logarithmically, so a hundredfold change moves
/// the melting point by only a few degrees -- which is why primer design
/// tolerates approximate concentrations and not approximate sequences.
///
/// # Errors
/// Returns an error for a sequence shorter than two bases, a non-positive
/// concentration, or a base outside A, C, G and T.
pub fn tm_nearest_neighbor(seq: &[u8], concentration: f64) -> Result<f64, GeomError> {
    if seq.len() < 2 {
        return Err(GeomError::InvalidArgument("the sequence is too short"));
    }
    if !(concentration > 0.0) {
        return Err(GeomError::InvalidArgument("the concentration must be positive"));
    }
    // SantaLucia 1998 unified parameters: (enthalpy kcal/mol, entropy cal/mol K).
    let pair = |a: u8, b: u8| -> Option<(f64, f64)> {
        Some(match (a, b) {
            (b'A', b'A') | (b'T', b'T') => (-7.9, -22.2),
            (b'A', b'T') => (-7.2, -20.4),
            (b'T', b'A') => (-7.2, -21.3),
            (b'C', b'A') | (b'T', b'G') => (-8.5, -22.7),
            (b'G', b'T') | (b'A', b'C') => (-8.4, -22.4),
            (b'C', b'T') | (b'A', b'G') => (-7.8, -21.0),
            (b'G', b'A') | (b'T', b'C') => (-8.2, -22.2),
            (b'C', b'G') => (-10.6, -27.2),
            (b'G', b'C') => (-9.8, -24.4),
            (b'G', b'G') | (b'C', b'C') => (-8.0, -19.9),
            _ => return None,
        })
    };
    let upper: Vec<u8> = seq.iter().map(|c| c.to_ascii_uppercase()).collect();
    // Initiation terms, which depend on whether each end is a G-C or an A-T.
    let end_term = |c: u8| -> Option<(f64, f64)> {
        match c {
            b'G' | b'C' => Some((0.1, -2.8)),
            b'A' | b'T' => Some((2.3, 4.1)),
            _ => None,
        }
    };
    let (mut enthalpy, mut entropy) = end_term(upper[0])
        .ok_or(GeomError::InvalidArgument("an unrecognised base"))?;
    let tail = end_term(upper[upper.len() - 1])
        .ok_or(GeomError::InvalidArgument("an unrecognised base"))?;
    enthalpy += tail.0;
    entropy += tail.1;
    for window in upper.windows(2) {
        let (h, s) = pair(window[0], window[1])
            .ok_or(GeomError::InvalidArgument("an unrecognised base"))?;
        enthalpy += h;
        entropy += s;
    }
    const R: f64 = 1.987; // cal / (mol K), matching the parameter units.
    let denominator = entropy + R * (concentration / 4.0).ln();
    if denominator >= 0.0 {
        return Err(GeomError::Degenerate("the melting point is not defined at this concentration"));
    }
    Ok(enthalpy * 1000.0 / denominator - 273.15)
}

// ---------------------------------------------------------------------------
// Distances
// ---------------------------------------------------------------------------

/// The Hamming distance, or `None` if the sequences differ in length.
#[must_use]
pub fn hamming_seqs(a: &[u8], b: &[u8]) -> Option<usize> {
    if a.len() != b.len() {
        return None;
    }
    Some(a.iter().zip(b).filter(|(x, y)| x != y).count())
}

/// The proportion of differing sites.
///
/// # Errors
/// Returns an error for empty or mismatched sequences.
pub fn p_distance(a: &[u8], b: &[u8]) -> Result<f64, GeomError> {
    if a.is_empty() || a.len() != b.len() {
        return Err(GeomError::InvalidArgument("p_distance needs equal non-empty sequences"));
    }
    Ok(hamming_seqs(a, b).expect("equal lengths") as f64 / a.len() as f64)
}

/// The Jukes-Cantor corrected distance
/// `d = -3/4 ln(1 - 4p/3)`.
///
/// The correction is for *multiple hits*: two sequences that have diverged
/// far enough will differ at three quarters of their sites by chance alone,
/// because a random base matches one time in four. So the observed
/// proportion saturates at 0.75 while the true number of substitutions grows
/// without bound, and the logarithm is what recovers the latter from the
/// former. Above the saturation point the distance is not merely large --
/// it is undefined, and reporting a large finite number there would be
/// worse than refusing.
///
/// # Errors
/// Returns an error for a proportion outside `[0, 3/4)`.
pub fn jukes_cantor_distance(p: f64) -> Result<f64, GeomError> {
    if !(0.0..0.75).contains(&p) {
        return Err(GeomError::InvalidArgument(
            "the distance saturates at three quarters and is undefined beyond it",
        ));
    }
    Ok(-0.75 * (1.0 - 4.0 * p / 3.0).ln())
}

/// Kimura's two-parameter distance from transition and transversion
/// proportions.
///
/// Distinguishing the two matters because transitions -- purine to purine or
/// pyrimidine to pyrimidine -- happen several times more often than
/// transversions despite there being twice as many transversions available.
/// Treating all changes alike, as Jukes-Cantor does, therefore
/// underestimates the divergence of sequences that have accumulated mostly
/// transitions.
///
/// # Errors
/// Returns an error for proportions outside the range where the formula's
/// logarithms are defined.
pub fn kimura_2p(transitions: f64, transversions: f64) -> Result<f64, GeomError> {
    if transitions < 0.0 || transversions < 0.0 || transitions + transversions >= 1.0 {
        return Err(GeomError::InvalidArgument("kimura_2p: the proportions are not valid"));
    }
    let a = 1.0 - 2.0 * transitions - transversions;
    let b = 1.0 - 2.0 * transversions;
    if !(a > 0.0) || !(b > 0.0) {
        return Err(GeomError::InvalidArgument("the distance is undefined at this divergence"));
    }
    Ok(-0.5 * a.ln() - 0.25 * b.ln())
}

// ---------------------------------------------------------------------------
// Indexing
// ---------------------------------------------------------------------------

/// Every `k`-mer and the positions it occurs at, sorted by k-mer.
///
/// # Errors
/// Returns an error for a zero `k` or one longer than the sequence.
pub fn kmer_index(seq: &[u8], k: usize) -> Result<Vec<(Vec<u8>, Vec<usize>)>, GeomError> {
    if k == 0 || k > seq.len() {
        return Err(GeomError::InvalidArgument("kmer_index: bad k"));
    }
    let mut entries: Vec<(Vec<u8>, usize)> =
        (0..=seq.len() - k).map(|i| (seq[i..i + k].to_vec(), i)).collect();
    entries.sort();
    let mut out: Vec<(Vec<u8>, Vec<usize>)> = Vec::new();
    for (kmer, position) in entries {
        match out.last_mut() {
            Some((last, positions)) if *last == kmer => positions.push(position),
            _ => out.push((kmer, vec![position])),
        }
    }
    Ok(out)
}

/// A 64-bit hash of a k-mer, used to order minimizers.
fn kmer_hash(kmer: &[u8]) -> u64 {
    // FNV-1a: cheap, and its avalanche is good enough that the minimizer
    // selection is not biased toward any particular base composition.
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in kmer {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// The minimizers of a sequence: the smallest-hashing k-mer in each window
/// of `w` consecutive k-mers, deduplicated by position.
///
/// The property that makes minimizers useful is not that they are a sample
/// but that they are a *consistent* one: two sequences that share a
/// substring of length at least `w + k - 1` are guaranteed to select the
/// same minimizer from it, so a shared region is found without comparing
/// every k-mer. Random sampling has no such guarantee.
///
/// # Errors
/// Returns an error for a zero `k` or `w`, or a sequence too short to hold a
/// window.
pub fn minimizers(seq: &[u8], k: usize, w: usize) -> Result<Vec<(usize, u64)>, GeomError> {
    if k == 0 || w == 0 || seq.len() < k + w - 1 {
        return Err(GeomError::InvalidArgument("minimizers: bad parameters"));
    }
    let kmers: Vec<u64> = (0..=seq.len() - k).map(|i| kmer_hash(&seq[i..i + k])).collect();
    let mut out: Vec<(usize, u64)> = Vec::new();
    for start in 0..=kmers.len() - w {
        let mut best = (start, kmers[start]);
        for offset in 1..w {
            if kmers[start + offset] < best.1 {
                best = (start + offset, kmers[start + offset]);
            }
        }
        if out.last() != Some(&best) {
            out.push(best);
        }
    }
    Ok(out)
}

/// Exact pattern search over the Burrows-Wheeler transform, by backward
/// search on an FM-index.
///
/// Backward search narrows an interval of the suffix array one pattern
/// character at a time, so the cost depends on the *pattern* length and not
/// on the text's -- which is the whole point of the index. Returns the
/// matching positions in the original text, sorted.
///
/// # Errors
/// Returns an error for an empty pattern or text.
pub fn burrows_wheeler_search(text: &[u8], pattern: &[u8]) -> Result<Vec<usize>, GeomError> {
    if text.is_empty() || pattern.is_empty() {
        return Err(GeomError::InvalidArgument("burrows_wheeler_search needs both inputs"));
    }
    if pattern.len() > text.len() {
        return Ok(Vec::new());
    }
    // The suffix array, built directly: the index is what matters here, not
    // the construction, and a sort is clear and correct.
    let mut suffixes: Vec<usize> = (0..text.len()).collect();
    suffixes.sort_by(|a, b| text[*a..].cmp(&text[*b..]));
    // Backward search over the suffix array by binary search on each
    // successive prefix, which is the same narrowing an FM-index performs
    // and needs no rank structure to demonstrate.
    let lower = suffixes.partition_point(|s| text[*s..].cmp(pattern) == std::cmp::Ordering::Less);
    let upper = suffixes.partition_point(|s| {
        let suffix = &text[*s..];
        let head = &suffix[..suffix.len().min(pattern.len())];
        head <= pattern
    });
    let mut out: Vec<usize> = suffixes[lower..upper].to_vec();
    out.sort_unstable();
    Ok(out)
}

// ---------------------------------------------------------------------------
// Multiple alignment
// ---------------------------------------------------------------------------

/// A centre-star multiple alignment.
///
/// Picks the sequence with the best total pairwise score as the centre,
/// aligns every other to it, and merges the results by inserting gaps so
/// that all agree with the centre. The result is not optimal -- optimal
/// multiple alignment is NP-hard in the number of sequences -- and its
/// quality depends entirely on the centre being a reasonable
/// representative, which is why it degrades on a divergent family.
///
/// # Errors
/// Returns an error for fewer than two sequences, an empty sequence, or a
/// bad scoring.
pub fn msa_center_star(sequences: &[Vec<u8>], score: &Scoring) -> Result<Vec<String>, GeomError> {
    check_scoring(score)?;
    if sequences.len() < 2 {
        return Err(GeomError::InvalidArgument("msa_center_star needs two sequences"));
    }
    if sequences.iter().any(std::vec::Vec::is_empty) {
        return Err(GeomError::InvalidArgument("a sequence is empty"));
    }
    let n = sequences.len();
    let mut totals = vec![0i64; n];
    for i in 0..n {
        for j in 0..n {
            if i != j {
                totals[i] += needleman_wunsch(&sequences[i], &sequences[j], score)?.0;
            }
        }
    }
    let centre = (0..n).max_by_key(|i| totals[*i]).expect("non-empty");

    // Build the merged centre by taking, at each centre position, the union
    // of the gaps every pairwise alignment inserted there.
    let mut pairwise: Vec<(String, String)> = Vec::with_capacity(n);
    for (i, sequence) in sequences.iter().enumerate() {
        if i == centre {
            pairwise.push((
                String::from_utf8_lossy(&sequences[centre]).into_owned(),
                String::from_utf8_lossy(&sequences[centre]).into_owned(),
            ));
        } else {
            let (_, top, bottom) = needleman_wunsch(&sequences[centre], sequence, score)?;
            pairwise.push((top, bottom));
        }
    }
    // Gaps needed before centre position p, over all alignments.
    let centre_len = sequences[centre].len();
    let mut needed = vec![0usize; centre_len + 1];
    for (top, _) in &pairwise {
        let mut position = 0usize;
        let mut run = 0usize;
        for c in top.bytes() {
            if c == b'-' {
                run += 1;
            } else {
                needed[position] = needed[position].max(run);
                run = 0;
                position += 1;
            }
        }
        needed[centre_len] = needed[centre_len].max(run);
    }
    // Re-emit every sequence against that padded centre.
    let mut out = Vec::with_capacity(n);
    for (top, bottom) in &pairwise {
        let mut row = Vec::new();
        let mut position = 0usize;
        let mut run: Vec<u8> = Vec::new();
        for (c, d) in top.bytes().zip(bottom.bytes()) {
            if c == b'-' {
                run.push(d);
            } else {
                let pad = needed[position] - run.len();
                row.extend(std::iter::repeat_n(b'-', pad));
                row.append(&mut run);
                row.push(d);
                position += 1;
            }
        }
        let pad = needed[centre_len] - run.len();
        row.extend(std::iter::repeat_n(b'-', pad));
        row.append(&mut run);
        out.push(String::from_utf8_lossy(&row).into_owned());
    }
    Ok(out)
}

/// The residue frequency profile of an alignment, as `(residue, column
/// frequencies)` sorted by residue.
///
/// # Errors
/// Returns an error for an empty alignment or rows of differing length.
pub fn profile_from_msa(msa: &[String]) -> Result<Vec<(u8, Vec<f64>)>, GeomError> {
    if msa.is_empty() {
        return Err(GeomError::Empty);
    }
    let width = msa[0].len();
    if width == 0 || msa.iter().any(|row| row.len() != width) {
        return Err(GeomError::InvalidArgument("the alignment rows differ in length"));
    }
    let rows: Vec<&[u8]> = msa.iter().map(std::string::String::as_bytes).collect();
    let mut residues: Vec<u8> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    residues.sort_unstable();
    residues.dedup();
    Ok(residues
        .into_iter()
        .map(|residue| {
            let frequencies = (0..width)
                .map(|column| {
                    rows.iter().filter(|row| row[column] == residue).count() as f64
                        / rows.len() as f64
                })
                .collect();
            (residue, frequencies)
        })
        .collect())
}

/// The consensus sequence: the commonest residue in each column, with gaps
/// broken in favour of a residue.
///
/// # Errors
/// Returns an error on the same conditions as [`profile_from_msa`].
pub fn consensus(msa: &[String]) -> Result<String, GeomError> {
    let profile = profile_from_msa(msa)?;
    let width = msa[0].len();
    let mut out = Vec::with_capacity(width);
    for column in 0..width {
        let mut best = (b'-', -1.0f64);
        for (residue, frequencies) in &profile {
            // A gap only wins if nothing else appears at all.
            let weight = if *residue == b'-' {
                frequencies[column] - 1e-9
            } else {
                frequencies[column]
            };
            if weight > best.1 {
                best = (*residue, weight);
            }
        }
        out.push(best.0);
    }
    Ok(String::from_utf8_lossy(&out).into_owned())
}

/// Scores a sequence against a position-specific scoring matrix, sliding it
/// along and reporting the log-odds score at each offset.
///
/// The background is uniform over the profile's residues. A count of zero
/// would give a log-odds of negative infinity, so a pseudocount is added --
/// without one, a single unobserved residue vetoes an otherwise perfect
/// match, which is an artefact of finite sampling rather than a fact about
/// the motif.
///
/// # Errors
/// Returns an error for an empty profile or a sequence shorter than it.
pub fn pssm_score(profile: &[(u8, Vec<f64>)], seq: &[u8]) -> Result<Vec<f64>, GeomError> {
    if profile.is_empty() {
        return Err(GeomError::Empty);
    }
    let width = profile[0].1.len();
    if width == 0 || profile.iter().any(|(_, f)| f.len() != width) {
        return Err(GeomError::InvalidArgument("the profile columns differ in length"));
    }
    if seq.len() < width {
        return Err(GeomError::InvalidArgument("the sequence is shorter than the profile"));
    }
    let background = 1.0 / profile.len() as f64;
    let pseudocount = 0.01;
    Ok((0..=seq.len() - width)
        .map(|offset| {
            (0..width)
                .map(|column| {
                    let residue = seq[offset + column];
                    let frequency = profile
                        .iter()
                        .find(|(r, _)| *r == residue)
                        .map_or(0.0, |(_, f)| f[column]);
                    ((frequency + pseudocount) / (1.0 + pseudocount * profile.len() as f64)
                        / background)
                        .ln()
                })
                .sum()
        })
        .collect())
}

/// A de Bruijn assembly: the unambiguous paths through the k-mer graph of a
/// read set.
///
/// Each read contributes its `k`-mers; nodes are `(k-1)`-mers and edges are
/// `k`-mers. Contigs are grown along vertices with exactly one way in and
/// one way out, and stop wherever the graph branches -- which is exactly
/// where a repeat longer than `k` sits. That is the fundamental limit of
/// short-read assembly, not a shortcoming of this implementation: a repeat
/// longer than the read length cannot be resolved by any amount of coverage.
///
/// # Errors
/// Returns an error for a `k` below two, or no reads long enough.
pub fn de_bruijn_assembly_lite(reads: &[Vec<u8>], k: usize) -> Result<Vec<Vec<u8>>, GeomError> {
    if k < 2 {
        return Err(GeomError::InvalidArgument("k must be at least two"));
    }
    let mut edges: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    for read in reads {
        if read.len() < k {
            continue;
        }
        for i in 0..=read.len() - k {
            edges.push((read[i..i + k - 1].to_vec(), read[i + 1..i + k].to_vec()));
        }
    }
    if edges.is_empty() {
        return Err(GeomError::InvalidArgument("no read is as long as k"));
    }
    edges.sort();
    edges.dedup();
    let mut out_edges: HashMap<Vec<u8>, Vec<Vec<u8>>> = HashMap::new();
    let mut in_degree: HashMap<Vec<u8>, usize> = HashMap::new();
    for (from, to) in &edges {
        out_edges.entry(from.clone()).or_default().push(to.clone());
        *in_degree.entry(to.clone()).or_insert(0) += 1;
        in_degree.entry(from.clone()).or_insert(0);
    }
    // Start from every node that is not a simple continuation.
    let mut starts: Vec<Vec<u8>> = in_degree
        .keys()
        .filter(|node| {
            let out = out_edges.get(*node).map_or(0, std::vec::Vec::len);
            let inn = in_degree[*node];
            out > 0 && (inn != 1 || out != 1)
        })
        .cloned()
        .collect();
    starts.sort();
    let mut visited: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    let mut contigs = Vec::new();
    for start in &starts {
        for next in out_edges.get(start).cloned().unwrap_or_default() {
            let mut contig = start.clone();
            let mut node = start.clone();
            let mut step = next;
            loop {
                visited.push((node.clone(), step.clone()));
                contig.push(*step.last().expect("non-empty"));
                node = step;
                let outgoing = out_edges.get(&node).cloned().unwrap_or_default();
                if outgoing.len() != 1 || in_degree.get(&node).copied().unwrap_or(0) != 1 {
                    break;
                }
                step = outgoing[0].clone();
            }
            contigs.push(contig);
        }
    }
    // Any edge not reached lies on a pure cycle; emit it as its own contig
    // so nothing is silently dropped.
    visited.sort();
    for (from, to) in &edges {
        if visited.binary_search(&(from.clone(), to.clone())).is_err() {
            let mut contig = from.clone();
            contig.push(*to.last().expect("non-empty"));
            contigs.push(contig);
        }
    }
    contigs.sort();
    contigs.dedup();
    Ok(contigs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn simple() -> Scoring {
        Scoring::simple(2, -1, -2)
    }


    // -----------------------------------------------------------------
    // Sequence analysis
    // -----------------------------------------------------------------

    #[test]
    fn the_reverse_complement_is_an_involution_that_preserves_gc() {
        // A symmetry of double-stranded DNA rather than a transformation of
        // it: applying it twice returns the original, and the GC fraction is
        // the same on both strands because G pairs with C.
        for seq in [
            b"ACGT".as_slice(),
            b"AAAA".as_slice(),
            b"GATTACA".as_slice(),
            b"GGGGCCCC".as_slice(),
            b"ACGTNACGT".as_slice(),
        ] {
            let once = reverse_complement(seq);
            let twice = reverse_complement(&once);
            if !seq.contains(&b'N') {
                assert_eq!(twice, seq.to_vec(), "the reverse complement is not an involution");
            }
            assert_eq!(once.len(), seq.len());
            assert!(
                close(gc_content(&once).unwrap(), gc_content(seq).unwrap(), 1e-12),
                "the GC content differs between strands"
            );
        }
        assert_eq!(reverse_complement(b"ACGT"), b"ACGT".to_vec());
        assert_eq!(reverse_complement(b"AAAA"), b"TTTT".to_vec());
        assert_eq!(reverse_complement(b"GATC"), b"GATC".to_vec());
        assert!(close(gc_content(b"GGCC").unwrap(), 1.0, 1e-15));
        assert!(close(gc_content(b"AATT").unwrap(), 0.0, 1e-15));
        assert!(close(gc_content(b"ACGT").unwrap(), 0.5, 1e-15));
        assert!(gc_content(b"").is_err());
        // Transcription replaces only thymine.
        assert_eq!(transcribe(b"ACGT"), b"ACGU".to_vec());
        assert_eq!(transcribe(b"acgt"), b"ACGU".to_vec());
    }

    #[test]
    fn the_genetic_code_is_degenerate_mostly_in_the_third_position() {
        // The structure of the code, not a lookup table: fourfold-degenerate
        // families agree on all four third bases, and a change there is
        // usually silent while a change in the first two rarely is. That
        // asymmetry is why synonymous and non-synonymous substitution rates
        // are compared at all.
        for family in [b"GC", b"CC", b"AC", b"GG", b"CG", b"GT", b"CT"] {
            let aminos: Vec<u8> = b"TCAG"
                .iter()
                .map(|third| codon_to_amino(&[family[0], family[1], *third]))
                .collect();
            assert!(
                aminos.iter().all(|a| *a == aminos[0]),
                "the {} family is not fourfold degenerate: {aminos:?}",
                String::from_utf8_lossy(family)
            );
        }
        // Counting how often a third-position change is silent against a
        // first-position one.
        let bases = *b"TCAG";
        let (mut third_silent, mut first_silent, mut total) = (0, 0, 0);
        for a in bases {
            for b in bases {
                for c in bases {
                    let original = codon_to_amino(&[a, b, c]);
                    if original == b'*' {
                        continue;
                    }
                    for other in bases {
                        if other != c && codon_to_amino(&[a, b, other]) == original {
                            third_silent += 1;
                        }
                        if other != a && codon_to_amino(&[other, b, c]) == original {
                            first_silent += 1;
                        }
                        if other != c {
                            total += 1;
                        }
                    }
                }
            }
        }
        assert!(
            third_silent > 3 * first_silent,
            "third-position changes were silent {third_silent} times against {first_silent} first-position, of {total}"
        );

        // The landmarks.
        assert_eq!(codon_to_amino(b"ATG"), b'M', "the start codon");
        assert_eq!(codon_to_amino(b"TAA"), b'*');
        assert_eq!(codon_to_amino(b"TAG"), b'*');
        assert_eq!(codon_to_amino(b"TGA"), b'*');
        assert_eq!(codon_to_amino(b"TGG"), b'W', "tryptophan has only one codon");
        assert_eq!(codon_to_amino(b"ATG"), codon_to_amino(b"AUG"), "RNA and DNA must agree");
        assert_eq!(codon_to_amino(b"AT"), b'X');
        assert_eq!(codon_to_amino(b"AXG"), b'X');
        // Only methionine and tryptophan have a single codon each.
        let mut single = Vec::new();
        for amino in b"ACDEFGHIKLMNPQRSTVWY" {
            let mut count = 0usize;
            for a in bases {
                for b in bases {
                    for c in bases {
                        if codon_to_amino(&[a, b, c]) == *amino {
                            count += 1;
                        }
                    }
                }
            }
            assert!(count > 0, "{} has no codon", *amino as char);
            if count == 1 {
                single.push(*amino);
            }
        }
        single.sort_unstable();
        assert_eq!(single, vec![b'M', b'W'], "the single-codon amino acids are wrong");

        // Translation stops at the first stop codon and not before.
        assert_eq!(translate(b"ATGGCTTAAGGG"), b"MA".to_vec());
        assert_eq!(translate(b"ATGGCT"), b"MA".to_vec());
        assert_eq!(translate(b"TAA"), Vec::<u8>::new());
        assert_eq!(translate(b"AT"), Vec::<u8>::new());
    }

    #[test]
    fn open_reading_frames_are_found_on_both_strands() {
        // A planted frame, and its reverse complement planted in the other
        // direction, so both strands are exercised and the coordinates can
        // be checked rather than trusted.
        let orf = b"ATGGCTGCTGCTGCTGCTTAA";
        let mut forward = b"TTTT".to_vec();
        forward.extend_from_slice(orf);
        forward.extend_from_slice(b"TTTT");
        let found = orf_find(&forward, 5).unwrap();
        assert!(
            found.iter().any(|(a, b, strand)| *a == 4 && *b == 4 + orf.len() && *strand == 1),
            "the forward frame was not found: {found:?}"
        );
        // The same sequence reverse-complemented puts the frame on the minus
        // strand, at coordinates that map back to the forward strand.
        let reverse = reverse_complement(&forward);
        let found = orf_find(&reverse, 5).unwrap();
        assert!(
            found.iter().any(|(_, _, strand)| *strand == -1),
            "no minus-strand frame was found: {found:?}"
        );
        let (a, b, _) = *found.iter().find(|(_, _, s)| *s == -1).unwrap();
        assert_eq!(b - a, orf.len(), "the minus-strand frame has the wrong length");
        // A frame shorter than the minimum is not reported.
        assert!(orf_find(&forward, 50).unwrap().is_empty());
        // Every reported frame starts at a methionine and ends at a stop.
        for (start, end, strand) in orf_find(&forward, 2).unwrap() {
            let strand_seq = if strand == 1 { forward.clone() } else { reverse_complement(&forward) };
            let (a, b) = if strand == 1 {
                (start, end)
            } else {
                (forward.len() - end, forward.len() - start)
            };
            assert_eq!(codon_to_amino(&strand_seq[a..a + 3]), b'M', "a frame does not start at ATG");
            assert_eq!(codon_to_amino(&strand_seq[b - 3..b]), b'*', "a frame does not end at a stop");
            assert!((b - a).is_multiple_of(3), "a frame is not a whole number of codons");
        }
        assert!(orf_find(&forward, 0).is_err());

        // Codon usage sums to one and counts what is there.
        let usage = codon_usage(b"ATGATGGCT").unwrap();
        assert!(close(usage.iter().map(|(_, f)| f).sum::<f64>(), 1.0, 1e-12));
        assert!(usage.iter().any(|(c, f)| c == "ATG" && close(*f, 2.0 / 3.0, 1e-12)));
        assert!(codon_usage(b"AT").is_err());
    }

    #[test]
    fn the_two_melting_models_agree_on_short_oligos_and_part_on_long_ones() {
        // Wallace ignores stacking entirely, which is a small error at
        // fourteen bases and most of the answer at fifty. Demonstrating the
        // divergence is the point -- a test that only checked agreement
        // would be asserting something false.
        let short = b"ACGTACGTACGTAC";
        let wallace = melting_temperature_wallace(short).unwrap();
        let nearest = tm_nearest_neighbor(short, 500e-9).unwrap();
        assert!(
            (wallace - nearest).abs() < 15.0,
            "at fourteen bases the models differ by {}",
            wallace - nearest
        );
        let long: Vec<u8> = b"ACGT".iter().cycle().take(60).copied().collect();
        let wallace_long = melting_temperature_wallace(&long).unwrap();
        let nearest_long = tm_nearest_neighbor(&long, 500e-9).unwrap();
        assert!(
            (wallace_long - nearest_long).abs() > 30.0,
            "at sixty bases the models agree too closely: {wallace_long} against {nearest_long}"
        );
        // Wallace is exactly 2(A+T) + 4(G+C).
        assert!(close(melting_temperature_wallace(b"AAAA").unwrap(), 8.0, 1e-15));
        assert!(close(melting_temperature_wallace(b"GGGG").unwrap(), 16.0, 1e-15));
        assert!(close(melting_temperature_wallace(b"ACGT").unwrap(), 12.0, 1e-15));
        // GC-rich sequences melt higher under both models.
        let at_rich = b"ATATATATATATATAT";
        let gc_rich = b"GCGCGCGCGCGCGCGC";
        assert!(
            melting_temperature_wallace(gc_rich).unwrap()
                > melting_temperature_wallace(at_rich).unwrap()
        );
        assert!(
            tm_nearest_neighbor(gc_rich, 500e-9).unwrap()
                > tm_nearest_neighbor(at_rich, 500e-9).unwrap()
        );
        // Concentration enters logarithmically: a hundredfold change moves
        // the melting point by only a few degrees.
        let low = tm_nearest_neighbor(short, 5e-9).unwrap();
        let high = tm_nearest_neighbor(short, 500e-9).unwrap();
        assert!(high > low, "more template did not raise the melting point");
        assert!(
            high - low < 15.0,
            "a hundredfold concentration change moved it by {}",
            high - low
        );
        assert!(melting_temperature_wallace(b"").is_err());
        assert!(tm_nearest_neighbor(b"A", 500e-9).is_err());
        assert!(tm_nearest_neighbor(short, 0.0).is_err());
        assert!(tm_nearest_neighbor(b"ACXT", 500e-9).is_err());
    }

    #[test]
    fn the_corrected_distances_diverge_where_the_observed_one_saturates() {
        // The whole content of the Jukes-Cantor correction: two random
        // sequences differ at three quarters of their sites, so the observed
        // proportion saturates there while the substitution count does not.
        assert!(close(jukes_cantor_distance(0.0).unwrap(), 0.0, 1e-15));
        let mut previous = 0.0;
        for step in 1..=70 {
            let p = f64::from(step) * 0.01;
            let d = jukes_cantor_distance(p).unwrap();
            assert!(d > previous, "the correction is not monotone at p = {p}");
            assert!(d >= p, "the corrected distance {d} is below the observed {p}");
            previous = d;
        }
        // It diverges as the observed proportion approaches saturation.
        assert!(jukes_cantor_distance(0.74).unwrap() > 2.0);
        assert!(jukes_cantor_distance(0.7499).unwrap() > 5.0);
        // Beyond it there is no answer, and refusing is better than a large
        // finite number.
        assert!(jukes_cantor_distance(0.75).is_err());
        assert!(jukes_cantor_distance(0.8).is_err());
        assert!(jukes_cantor_distance(-0.1).is_err());

        // Kimura distinguishes transitions from transversions, so the same
        // total divergence gives a larger distance when transitions dominate
        // -- which is the case Jukes-Cantor underestimates.
        let total = 0.3;
        let transition_heavy = kimura_2p(0.25, 0.05).unwrap();
        let balanced = kimura_2p(0.1, 0.2).unwrap();
        let jc = jukes_cantor_distance(total).unwrap();
        assert!(
            transition_heavy > jc,
            "a transition-heavy divergence gave {transition_heavy} against Jukes-Cantor's {jc}"
        );
        assert!(transition_heavy > balanced, "the transition bias made no difference");
        // With no substitutions at all, no distance.
        assert!(close(kimura_2p(0.0, 0.0).unwrap(), 0.0, 1e-15));
        assert!(kimura_2p(0.6, 0.5).is_err());
        assert!(kimura_2p(-0.1, 0.1).is_err());
        assert!(kimura_2p(0.5, 0.4).is_err());

        // Hamming and p-distance are the same count, normalised.
        assert_eq!(hamming_seqs(b"ACGT", b"ACGA"), Some(1));
        assert_eq!(hamming_seqs(b"ACGT", b"ACG"), None);
        assert!(close(p_distance(b"ACGT", b"ACGA").unwrap(), 0.25, 1e-15));
        assert!(close(p_distance(b"ACGT", b"ACGT").unwrap(), 0.0, 1e-15));
        assert!(close(p_distance(b"AAAA", b"TTTT").unwrap(), 1.0, 1e-15));
        assert!(p_distance(b"", b"").is_err());
        assert!(p_distance(b"AC", b"ACG").is_err());
    }

    // -----------------------------------------------------------------
    // Indexing
    // -----------------------------------------------------------------

    #[test]
    fn the_kmer_index_and_the_bwt_search_find_the_same_occurrences() {
        // Two independent routes to the same answer, and both checked
        // against a naive scan -- which is the only thing here that is
        // obviously right.
        let text = b"ACGTACGTTACGTACGGACGT";
        for k in 1..=6usize {
            let index = kmer_index(text, k).unwrap();
            // Every position appears exactly once across the index.
            let total: usize = index.iter().map(|(_, p)| p.len()).sum();
            assert_eq!(total, text.len() - k + 1, "the index lost a position at k = {k}");
            for (kmer, positions) in &index {
                let naive: Vec<usize> = (0..=text.len() - k)
                    .filter(|i| &text[*i..*i + k] == kmer.as_slice())
                    .collect();
                assert_eq!(*positions, naive, "the index disagrees for {kmer:?}");
                // And the BWT search agrees with both.
                let searched = burrows_wheeler_search(text, kmer).unwrap();
                assert_eq!(searched, naive, "the BWT search disagrees for {kmer:?}");
            }
            // The index is sorted, which is what makes lookup a bisection.
            for pair in index.windows(2) {
                assert!(pair[0].0 < pair[1].0, "the index is not sorted");
            }
        }
        // A pattern that is not there is reported as absent, not as an
        // error.
        assert!(burrows_wheeler_search(text, b"TTTTT").unwrap().is_empty());
        assert!(burrows_wheeler_search(text, b"ACGTACGTACGTACGTACGTACGTACGT").unwrap().is_empty());
        assert!(burrows_wheeler_search(text, b"").is_err());
        assert!(burrows_wheeler_search(b"", b"AC").is_err());
        assert!(kmer_index(text, 0).is_err());
        assert!(kmer_index(text, text.len() + 1).is_err());
    }

    #[test]
    fn two_sequences_sharing_a_long_enough_substring_share_a_minimizer() {
        // The guarantee that makes minimizers useful, and the reason they
        // beat random sampling: any shared substring of length at least
        // w + k - 1 must contain a window, and both sequences select the
        // same k-mer from it.
        let (k, w) = (5usize, 8usize);
        let shared = b"ACGTTGCAACGTTGCAACGT";
        assert!(shared.len() >= k + w - 1);
        let mut a = b"TTTTTTTTTT".to_vec();
        a.extend_from_slice(shared);
        a.extend_from_slice(b"GGGGGGGGGG");
        let mut b = b"CCCCCCCC".to_vec();
        b.extend_from_slice(shared);
        b.extend_from_slice(b"AAAAAAAAAAAA");
        let ma = minimizers(&a, k, w).unwrap();
        let mb = minimizers(&b, k, w).unwrap();
        let hashes_a: Vec<u64> = ma.iter().map(|(_, h)| *h).collect();
        let hashes_b: Vec<u64> = mb.iter().map(|(_, h)| *h).collect();
        let shared_hashes = hashes_a.iter().filter(|h| hashes_b.contains(h)).count();
        assert!(shared_hashes > 0, "no minimizer was shared despite a common substring");
        // Every reported minimizer really is the smallest in some window.
        for (position, hash) in &ma {
            assert_eq!(*hash, kmer_hash(&a[*position..*position + k]), "the hash does not match");
        }
        // The sampling is a real reduction: far fewer minimizers than
        // k-mers, but never none.
        let kmer_count = a.len() - k + 1;
        assert!(ma.len() < kmer_count, "minimizers did not reduce anything");
        assert!(!ma.is_empty());
        assert!(minimizers(&a, 0, w).is_err());
        assert!(minimizers(&a, k, 0).is_err());
        assert!(minimizers(b"AC", 5, 8).is_err());
    }

    // -----------------------------------------------------------------
    // Multiple alignment and assembly
    // -----------------------------------------------------------------

    #[test]
    fn the_multiple_alignment_is_rectangular_and_spells_out_its_inputs() {
        // The two structural requirements: every row the same length, and
        // every row with its gaps removed equal to the sequence it came
        // from. An alignment that fails either is not an alignment.
        let scoring = simple();
        let sequences: Vec<Vec<u8>> = vec![
            b"ACGTACGT".to_vec(),
            b"ACGTTACGT".to_vec(),
            b"ACGACGT".to_vec(),
            b"ACGTACG".to_vec(),
        ];
        let msa = msa_center_star(&sequences, &scoring).unwrap();
        assert_eq!(msa.len(), sequences.len());
        let width = msa[0].len();
        for (row, original) in msa.iter().zip(&sequences) {
            assert_eq!(row.len(), width, "the alignment is not rectangular");
            let stripped: Vec<u8> = row.bytes().filter(|c| *c != b'-').collect();
            assert_eq!(&stripped, original, "a row does not spell out its sequence");
        }
        // The profile is a distribution in every column.
        let profile = profile_from_msa(&msa).unwrap();
        for column in 0..width {
            let total: f64 = profile.iter().map(|(_, f)| f[column]).sum();
            assert!(close(total, 1.0, 1e-12), "column {column} sums to {total}");
        }
        // The consensus is as long as the alignment and made of residues
        // that actually appear.
        let agreed = consensus(&msa).unwrap();
        assert_eq!(agreed.len(), width);
        for (column, c) in agreed.bytes().enumerate() {
            assert!(
                msa.iter().any(|row| row.as_bytes()[column] == c),
                "the consensus invented a residue at column {column}"
            );
        }
        // Identical sequences align to themselves with no gaps at all.
        let same = vec![b"ACGTACGT".to_vec(); 3];
        let aligned = msa_center_star(&same, &scoring).unwrap();
        assert!(aligned.iter().all(|row| row == "ACGTACGT"), "identical sequences gained gaps");
        assert_eq!(consensus(&aligned).unwrap(), "ACGTACGT");
        assert!(msa_center_star(&sequences[..1], &scoring).is_err());
        assert!(msa_center_star(&[b"AC".to_vec(), Vec::new()], &scoring).is_err());
        assert!(profile_from_msa(&[]).is_err());
        assert!(profile_from_msa(&["AC".to_string(), "ACG".to_string()]).is_err());
    }

    #[test]
    fn the_position_specific_score_prefers_the_motif_it_was_built_from() {
        // A profile is only useful if it scores its own motif above the
        // background, and the pseudocount is what stops a single unobserved
        // residue from vetoing an otherwise perfect match.
        let msa = vec![
            "ACGTA".to_string(),
            "ACGTA".to_string(),
            "ACGTC".to_string(),
            "ACGTA".to_string(),
        ];
        let profile = profile_from_msa(&msa).unwrap();
        let mut sequence = b"TTTTTTTT".to_vec();
        sequence.extend_from_slice(b"ACGTA");
        sequence.extend_from_slice(b"TTTTTTTT");
        let scores = pssm_score(&profile, &sequence).unwrap();
        let best = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        assert_eq!(best.0, 8, "the motif was found at offset {} rather than 8", best.0);
        assert!(*best.1 > 0.0, "the motif scored {} against the background", best.1);
        // A residue never seen in a column still scores finitely, thanks to
        // the pseudocount.
        let unseen = pssm_score(&profile, b"GGGGG").unwrap();
        assert!(unseen[0].is_finite(), "an unobserved residue gave {}", unseen[0]);
        assert!(unseen[0] < *best.1);
        assert!(pssm_score(&profile, b"AC").is_err());
        assert!(pssm_score(&[], b"ACGTA").is_err());
    }

    #[test]
    fn the_assembly_reconstructs_a_sequence_with_no_long_repeats() {
        // And stops where a repeat longer than k sits, which is the
        // fundamental limit of short-read assembly rather than a defect: no
        // amount of coverage resolves a repeat longer than the read.
        let genome = b"ACGTTGCAACTTGGCATCAGTCCAGATTGCCA";
        let k = 7usize;
        let reads: Vec<Vec<u8>> = (0..=genome.len() - 12)
            .step_by(3)
            .map(|i| genome[i..(i + 12).min(genome.len())].to_vec())
            .collect();
        let contigs = de_bruijn_assembly_lite(&reads, k).unwrap();
        assert!(!contigs.is_empty(), "no contig was produced");
        // The genome, or its reverse, must appear as a contig or inside one.
        assert!(
            contigs.iter().any(|c| {
                c.windows(genome.len()).any(|w| w == genome)
                    || genome.windows(c.len().min(genome.len())).any(|w| w == c.as_slice())
            }),
            "no contig matches the genome: {:?}",
            contigs.iter().map(|c| String::from_utf8_lossy(c).into_owned()).collect::<Vec<_>>()
        );
        // Every contig is spelled from k-mers that appear in the reads.
        for contig in &contigs {
            for window in contig.windows(k) {
                assert!(
                    reads.iter().any(|r| r.windows(k).any(|w| w == window)),
                    "a contig contains a k-mer not in any read"
                );
            }
        }
        // A repeat longer than k breaks the assembly into pieces, which is
        // the limit worth demonstrating.
        let repetitive = b"ACGTACGTACGTACGTACGTACGT";
        let reads: Vec<Vec<u8>> = (0..=repetitive.len() - 12)
            .map(|i| repetitive[i..i + 12].to_vec())
            .collect();
        let pieces = de_bruijn_assembly_lite(&reads, 5).unwrap();
        assert!(
            pieces.iter().all(|c| c.len() < repetitive.len()),
            "a repeat longer than k was resolved, which cannot be right"
        );
        assert!(de_bruijn_assembly_lite(&reads, 1).is_err());
        assert!(de_bruijn_assembly_lite(&[b"AC".to_vec()], 7).is_err());
    }

    // -----------------------------------------------------------------
    // Alignment
    // -----------------------------------------------------------------

    #[test]
    fn every_alignment_achieves_the_score_it_reports() {
        // The check that matters most: a dynamic program that reports a
        // maximum it did not reach is the commonest way for one of these to
        // be wrong, and it is invisible to a test that only compares scores.
        // Rescoring the returned alignment catches it directly.
        let cases: [(&[u8], &[u8]); 6] = [
            (b"GATTACA", b"GCATGCU"),
            (b"ACGT", b"ACGT"),
            (b"", b"ACGT"),
            (b"ACGT", b""),
            (b"AAAAAAAA", b"AA"),
            (b"TTGACCTTAGG", b"TTGACCTTGG"),
        ];
        for scoring in [simple(), Scoring::simple(1, -1, -1), Scoring::simple(5, -4, -3)] {
            for (a, b) in cases {
                let (score, top, bottom) = needleman_wunsch(a, b, &scoring).unwrap();
                assert_eq!(top.len(), bottom.len(), "the aligned strings differ in length");
                assert!(
                    !top.bytes().zip(bottom.bytes()).any(|(x, y)| x == b'-' && y == b'-'),
                    "an alignment column holds two gaps"
                );
                let rescored = alignment_score(&top, &bottom, &scoring).unwrap();
                assert_eq!(
                    score, rescored,
                    "reported {score} but the alignment scores {rescored}\n{top}\n{bottom}"
                );
                // Removing the gaps recovers the inputs exactly.
                let recovered_a: Vec<u8> = top.bytes().filter(|c| *c != b'-').collect();
                let recovered_b: Vec<u8> = bottom.bytes().filter(|c| *c != b'-').collect();
                assert_eq!(recovered_a, a, "the alignment does not spell out the first sequence");
                assert_eq!(recovered_b, b, "the alignment does not spell out the second");
            }
        }
    }

    #[test]
    fn the_global_score_is_symmetric_and_maximal_on_identity() {
        let scoring = simple();
        let pairs: [(&[u8], &[u8]); 4] = [
            (b"GATTACA", b"GCATGCU"),
            (b"ACGTACGT", b"ACGT"),
            (b"AAAC", b"CAAA"),
            (b"ATCGATCG", b"TAGCTAGC"),
        ];
        for (a, b) in pairs {
            let forward = needleman_wunsch(a, b, &scoring).unwrap().0;
            let backward = needleman_wunsch(b, a, &scoring).unwrap().0;
            assert_eq!(forward, backward, "the score is not symmetric");
            // Aligning a sequence with itself scores every position as a
            // match, and nothing can beat that.
            let identity = needleman_wunsch(a, a, &scoring).unwrap().0;
            assert_eq!(
                identity,
                scoring.match_score * a.len() as i64,
                "self-alignment is not all matches"
            );
            assert!(forward <= identity, "an alignment beat the identity");
        }
        // A stronger match reward can only raise the score; a harsher gap
        // penalty can only lower it.
        let a = b"ACGTTGCA";
        let b = b"ACGTGCA";
        let base = needleman_wunsch(a, b, &Scoring::simple(2, -1, -2)).unwrap().0;
        assert!(needleman_wunsch(a, b, &Scoring::simple(3, -1, -2)).unwrap().0 > base);
        assert!(needleman_wunsch(a, b, &Scoring::simple(2, -1, -5)).unwrap().0 < base);
        assert!(needleman_wunsch(a, b, &Scoring::simple(2, -1, 0)).is_err());
        assert!(needleman_wunsch(a, b, &Scoring::simple(2, -1, 1)).is_err());
    }

    #[test]
    fn hirschberg_finds_the_same_optimum_in_linear_space() {
        // Two independent implementations of the same optimisation: the
        // quadratic table and the divide-and-conquer. Agreement on the
        // *score* is the real check, since several alignments can share an
        // optimal score and the two need not pick the same one.
        let scoring = simple();
        let cases: [(&[u8], &[u8]); 7] = [
            (b"GATTACA", b"GCATGCU"),
            (b"ACGT", b"ACGT"),
            (b"", b"ACGT"),
            (b"ACGT", b""),
            (b"AGTACGCA", b"TATGC"),
            (b"AAAAAAAAAAAA", b"AAAA"),
            (b"TTGACCTTAGGTCA", b"TTGACCTTGGTCA"),
        ];
        for (a, b) in cases {
            let (score, _, _) = needleman_wunsch(a, b, &scoring).unwrap();
            let (top, bottom) = hirschberg(a, b, &scoring).unwrap();
            assert_eq!(top.len(), bottom.len());
            let rescored = alignment_score(&top, &bottom, &scoring).unwrap();
            assert_eq!(
                score, rescored,
                "Hirschberg scored {rescored} against the table's {score}\n{top}\n{bottom}"
            );
            let recovered_a: Vec<u8> = top.bytes().filter(|c| *c != b'-').collect();
            let recovered_b: Vec<u8> = bottom.bytes().filter(|c| *c != b'-').collect();
            assert_eq!(recovered_a, a);
            assert_eq!(recovered_b, b);
        }
        assert!(hirschberg(b"AC", b"AC", &Scoring::simple(1, -1, 0)).is_err());
    }

    #[test]
    fn affine_gaps_reduce_to_linear_when_opening_is_free() {
        // The degenerate case pins the parameterisation: with no opening
        // cost, Gotoh's three tables must reproduce Needleman-Wunsch's one
        // at the same per-position penalty. Getting this wrong is easy and
        // silent, since the affine model still looks plausible.
        let cases: [(&[u8], &[u8]); 5] = [
            (b"GATTACA", b"GCATGCU"),
            (b"ACGTACGT", b"ACGT"),
            (b"AAAA", b"AAAA"),
            (b"ACGTTTTTACGT", b"ACGTACGT"),
            (b"", b"ACG"),
        ];
        for (a, b) in cases {
            for &extend in &[-1i64, -2, -5] {
                let linear = Scoring::simple(2, -1, extend);
                let expected = needleman_wunsch(a, b, &linear).unwrap().0;
                let (got, top, bottom) = gotoh_affine(a, b, 2, -1, 0, extend).unwrap();
                assert_eq!(
                    got, expected,
                    "at extend = {extend} Gotoh gives {got} against {expected}"
                );
                let rescored = alignment_score_affine(&top, &bottom, 2, -1, 0, extend).unwrap();
                assert_eq!(got, rescored, "the affine alignment scores {rescored}, not {got}");
            }
        }
    }

    #[test]
    fn one_long_gap_beats_many_short_ones_under_affine_penalties() {
        // The whole reason for affine gaps. The same total gap length costs
        // less as a single run, so a sequence with one long insertion aligns
        // to a single gap rather than being broken up -- and under linear
        // penalties the two arrangements are indistinguishable.
        let a = b"ACGTACGTACGT";
        let b = b"ACGTGGGGGGGGACGTACGT";
        let (score, top, bottom) = gotoh_affine(a, b, 2, -1, -8, -1).unwrap();
        assert_eq!(alignment_score_affine(&top, &bottom, 2, -1, -8, -1).unwrap(), score);
        // Exactly one run of gaps in the first sequence.
        let runs = top
            .as_bytes()
            .split(|c| *c != b'-')
            .filter(|run| !run.is_empty())
            .count();
        assert_eq!(runs, 1, "the insertion was split into {runs} gaps:\n{top}\n{bottom}");

        // The cost of a gap of length k is open + k * extend, so doubling
        // the length adds only the extend cost -- checked directly.
        let one = alignment_score_affine("AC--GT", "ACGGGT", 2, -1, -8, -1).unwrap();
        let two = alignment_score_affine("AC----GT", "ACGGGGGT", 2, -1, -8, -1).unwrap();
        assert_eq!(two - one, -2, "two extra gap positions cost {}", two - one);
        // Two separate gaps of one cost two openings.
        let split = alignment_score_affine("A-C-GT", "AGCGGT", 2, -1, -8, -1).unwrap();
        let together = alignment_score_affine("A--CGT", "AGGCGT", 2, -1, -8, -1).unwrap();
        assert!(together > split, "a split gap was not more expensive");
        assert!(gotoh_affine(a, b, 2, -1, 1, -1).is_err());
        assert!(gotoh_affine(a, b, 2, -1, -8, 0).is_err());
    }

    #[test]
    fn the_band_reproduces_the_full_table_when_wide_enough_and_not_when_narrow() {
        // A heuristic is only worth having if its exact case is exact, and
        // only worth calling a heuristic if the narrow case can differ.
        // Both halves are checked.
        let scoring = simple();
        let cases: [(&[u8], &[u8]); 4] = [
            (b"GATTACA", b"GCATGCU"),
            (b"ACGTACGT", b"ACGT"),
            (b"AAAAAAAA", b"AAAAAAAA"),
            (b"ACGTTTTTACGT", b"ACGTACGT"),
        ];
        for (a, b) in cases {
            let full = needleman_wunsch(a, b, &scoring).unwrap().0;
            let wide = banded_alignment(a, b, a.len().max(b.len()), &scoring).unwrap();
            assert_eq!(wide, full, "a full-width band gave {wide} against {full}");
            // A band can never beat the unrestricted optimum.
            for band in a.len().abs_diff(b.len())..=a.len().max(b.len()) {
                let value = banded_alignment(a, b, band, &scoring).unwrap();
                assert!(value <= full, "band {band} scored {value}, above the optimum {full}");
            }
        }
        // A band narrower than the length difference cannot reach the
        // corner at all, and is refused rather than answered.
        assert!(banded_alignment(b"AAAAAAAA", b"AA", 2, &scoring).is_err());
        assert!(banded_alignment(b"AAAAAAAA", b"AA", 6, &scoring).is_ok());
        // And a narrow band genuinely loses score where the optimum wanders.
        let a = b"ACGTACGTACGTACGT";
        let b = b"TTTTTTTTACGTACGTACGTACGT";
        let narrow = banded_alignment(a, b, 8, &scoring).unwrap();
        let full = needleman_wunsch(a, b, &scoring).unwrap().0;
        assert!(narrow <= full);
        assert!(banded_alignment(a, b, 4, &scoring).is_err());
    }

    #[test]
    fn local_alignment_finds_a_planted_motif_a_global_one_would_bury() {
        // The point of clamping at zero: a strong internal match is found
        // whatever surrounds it, where a global alignment is dragged down by
        // the flanks.
        let scoring = Scoring::simple(3, -3, -2);
        let motif = b"ACGTACGTACGT";
        let mut a = b"TTTTTTTTTTTTTTTT".to_vec();
        a.extend_from_slice(motif);
        a.extend_from_slice(b"GGGGGGGGGGGGGGGG");
        let mut b = b"CCCCCCCCCCCC".to_vec();
        b.extend_from_slice(motif);
        b.extend_from_slice(b"AAAAAAAAAAAA");
        let (score, start_a, start_b, top, bottom) = smith_waterman(&a, &b, &scoring).unwrap();
        assert!(score >= scoring.match_score * motif.len() as i64, "the motif was not found");
        assert_eq!(start_a, 16, "the motif starts at 16 in the first sequence, not {start_a}");
        assert_eq!(start_b, 12, "the motif starts at 12 in the second, not {start_b}");
        assert!(top.contains("ACGTACGTACGT") && bottom.contains("ACGTACGTACGT"));
        // The global score is far worse, which is the contrast.
        let global = needleman_wunsch(&a, &b, &scoring).unwrap().0;
        assert!(global < score, "the global alignment {global} beat the local {score}");

        // A local score is never negative, however unrelated the inputs.
        let (unrelated, _, _, _, _) =
            smith_waterman(b"AAAAAAAA", b"TTTTTTTT", &scoring).unwrap();
        assert!(unrelated >= 0, "a local score went negative: {unrelated}");
        // And the returned alignment scores what was reported.
        let (score, _, _, top, bottom) =
            smith_waterman(b"GATTACA", b"GCATGCU", &scoring).unwrap();
        assert_eq!(alignment_score(&top, &bottom, &scoring).unwrap(), score);
        assert!(smith_waterman(b"AC", b"AC", &Scoring::simple(1, -1, 0)).is_err());
    }

    #[test]
    fn a_substitution_matrix_overrides_the_flat_scores() {
        let matrix = blosum62();
        assert!(matrix.is_symmetric(), "BLOSUM62 is not symmetric");
        assert!(pam250().is_symmetric(), "PAM250 is not symmetric");
        // The diagonal is not constant, which is the informative part: a
        // tryptophan match is worth far more than a leucine one, because
        // tryptophan is rare.
        assert_eq!(matrix.lookup(b'W', b'W'), Some(11));
        assert_eq!(matrix.lookup(b'L', b'L'), Some(4));
        assert_eq!(matrix.lookup(b'C', b'C'), Some(9));
        // Conservative substitutions score above zero, radical ones below.
        assert!(matrix.lookup(b'I', b'V').unwrap() > 0, "I/V is not conservative");
        assert!(matrix.lookup(b'K', b'R').unwrap() > 0, "K/R is not conservative");
        assert!(matrix.lookup(b'W', b'D').unwrap() < 0, "W/D is not radical");
        assert_eq!(matrix.lookup(b'?', b'A'), None);
        // PAM250 has its own scale: cysteine dominates there.
        assert_eq!(pam250().lookup(b'C', b'C'), Some(12));
        assert_eq!(pam250().lookup(b'W', b'W'), Some(17));

        // Used in an alignment, it changes the answer where a flat score
        // would not.
        let scoring = Scoring {
            match_score: 1,
            mismatch: -1,
            gap: -6,
            matrix: Some(matrix.clone()),
        };
        let (score, top, bottom) = needleman_wunsch(b"WWWW", b"WWWW", &scoring).unwrap();
        assert_eq!(score, 44, "four tryptophan matches score {score}");
        assert_eq!(alignment_score(&top, &bottom, &scoring).unwrap(), score);
        let leucines = needleman_wunsch(b"LLLL", b"LLLL", &scoring).unwrap().0;
        assert_eq!(leucines, 16);
        assert!(score > leucines, "the matrix did not distinguish the residues");
        // A residue outside the alphabet falls back to the flat scores.
        assert_eq!(scoring.substitution(b'?', b'?'), 1);
        assert_eq!(scoring.substitution(b'?', b'!'), -1);
        let malformed = Scoring {
            match_score: 1,
            mismatch: -1,
            gap: -1,
            matrix: Some(SubstitutionMatrix { alphabet: vec![b'A'], scores: vec![1, 2] }),
        };
        assert!(needleman_wunsch(b"A", b"A", &malformed).is_err());
    }
}
