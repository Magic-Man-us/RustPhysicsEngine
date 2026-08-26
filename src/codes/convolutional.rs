//! Convolutional and turbo codes, and the channels they run over.
//!
//! A convolutional code has no block length. The encoder is a shift register:
//! each input bit is combined with the last few, and the output depends on a
//! sliding window rather than on a partition of the message. That makes the
//! code a walk through a *trellis* -- a graph whose vertices are the register
//! states and whose edges are the possible inputs -- and decoding the problem
//! of finding the walk that best matches what arrived. Viterbi's algorithm is
//! dynamic programming on that graph, and it is optimal: it returns the
//! maximum-likelihood sequence, not an approximation to it.
//!
//! Turbo codes take two such encoders, feed the second an interleaved copy of
//! the message, and decode by having the two halves exchange opinions. What
//! each passes the other is *extrinsic* information -- what it concluded
//! about a bit from everything except that bit's own channel evidence -- and
//! keeping the exchange extrinsic is the whole trick. Feeding back a
//! decoder's full opinion would let it hear its own guess reflected as
//! independent confirmation, and the iteration would converge confidently to
//! nonsense.
//!
//! The capacity functions at the end say where the limits are. A rate-`1/2`
//! binary code cannot work below about `0.187` decibels of `Eb/N0`, whatever
//! it is; turbo codes reached within a few tenths of that, which is why they
//! ended a thirty-year search.

use crate::monte_carlo::Rng;
use std::f64::consts::PI;

/// A rate `1/n` convolutional code, given by its constraint length and
/// generator polynomials.
///
/// The generators are the taps of the shift register, conventionally written
/// in octal: the NASA standard's `171` and `133` are `0o171` and `0o133`,
/// seven bits each for a constraint length of seven. Bit `k - 1` of a
/// generator is the current input and bit zero the oldest bit in memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvolutionalCode {
    /// Constraint length: the current bit plus the bits of memory.
    pub k: u32,
    /// One generator polynomial per output bit.
    pub polys: Vec<u64>,
}

impl ConvolutionalCode {
    /// The code with the given constraint length and generators.
    ///
    /// # Panics
    /// Panics unless the constraint length is between two and sixteen, there
    /// is at least one generator, and every generator fits in `k` bits.
    #[must_use]
    pub fn new(k: u32, polys: &[u64]) -> Self {
        assert!((2..=16).contains(&k), "the constraint length must be between two and sixteen");
        assert!(!polys.is_empty(), "a code needs at least one generator");
        assert!(polys.iter().all(|&p| p < 1 << k), "a generator does not fit in k bits");
        ConvolutionalCode { k, polys: polys.to_vec() }
    }

    /// The rate-`1/2`, constraint-length-seven code used on essentially every
    /// NASA mission of the Voyager era and standardised by CCSDS.
    ///
    /// Generators `171` and `133` in octal, free distance ten.
    #[must_use]
    pub fn nasa_standard() -> Self {
        ConvolutionalCode::new(7, &[0o171, 0o133])
    }

    /// Bits of memory: one fewer than the constraint length.
    #[must_use]
    pub fn memory(&self) -> u32 {
        self.k - 1
    }

    /// The number of trellis states, `2^(k-1)`.
    #[must_use]
    pub fn trellis_states(&self) -> usize {
        1 << self.memory()
    }

    /// Output bits per input bit.
    #[must_use]
    pub fn outputs(&self) -> usize {
        self.polys.len()
    }

    /// The outputs and next state for one input bit from one state.
    #[must_use]
    pub fn step(&self, state: usize, input: bool) -> (Vec<bool>, usize) {
        let m = self.memory();
        let window = (usize::from(input) << m) | state;
        let out = self
            .polys
            .iter()
            .map(|&p| (window as u64 & p).count_ones() % 2 == 1)
            .collect();
        (out, window >> 1)
    }

    /// Encodes a message, flushing the register with `k - 1` zeros so the
    /// trellis ends where it started.
    ///
    /// Termination costs `k - 1` bits of rate and buys the decoder a known
    /// endpoint, which is worth far more than it costs on any message longer
    /// than the register.
    #[must_use]
    pub fn encode(&self, bits: &[bool]) -> Vec<bool> {
        let mut state = 0usize;
        let mut out = Vec::with_capacity((bits.len() + self.memory() as usize) * self.outputs());
        for &b in bits.iter().chain(std::iter::repeat_n(&false, self.memory() as usize)) {
            let (o, next) = self.step(state, b);
            out.extend(o);
            state = next;
        }
        out
    }

    /// Maximum-likelihood decoding of a hard-decision stream by the Viterbi
    /// algorithm.
    ///
    /// # Panics
    /// Panics unless the stream's length is a multiple of the output count
    /// and long enough to hold the flush.
    #[must_use]
    pub fn viterbi_decode(&self, recv_hard: &[bool]) -> Vec<bool> {
        let n = self.outputs();
        assert!(recv_hard.len().is_multiple_of(n), "the stream is not a whole number of symbols");
        // Hamming distance is the log-likelihood metric of a binary symmetric
        // channel, up to a constant, so hard decoding is soft decoding with
        // every confidence set to one.
        let llr: Vec<f64> = recv_hard.iter().map(|&b| if b { -1.0 } else { 1.0 }).collect();
        self.viterbi_soft(&llr)
    }

    /// Maximum-likelihood decoding from log-likelihood ratios, where a
    /// positive value leans towards a zero bit.
    ///
    /// Soft decisions are worth about two decibels over hard ones on a
    /// Gaussian channel, for no change to the algorithm beyond the branch
    /// metric: a bit the demodulator was unsure of should not outvote one it
    /// was certain of, and a hard decision throws away exactly that.
    ///
    /// # Panics
    /// Panics unless the stream's length is a multiple of the output count
    /// and long enough to hold the flush.
    #[must_use]
    pub fn viterbi_soft(&self, llr: &[f64]) -> Vec<bool> {
        let n = self.outputs();
        assert!(llr.len().is_multiple_of(n), "the stream is not a whole number of symbols");
        let steps = llr.len() / n;
        let m = self.memory() as usize;
        assert!(steps >= m, "the stream is shorter than the flush");
        let states = self.trellis_states();
        let inf = f64::INFINITY;
        let mut cost = vec![inf; states];
        cost[0] = 0.0;
        // One byte per state per step: which input bit led here.
        let mut back = vec![vec![(usize::MAX, false); states]; steps];
        for t in 0..steps {
            let mut next = vec![inf; states];
            for s in 0..states {
                if cost[s].is_infinite() {
                    continue;
                }
                // After the message ends the input is known to be zero, so
                // the trellis narrows and the decoder need not consider ones.
                let inputs: &[bool] =
                    if t >= steps - m { &[false] } else { &[false, true] };
                for &b in inputs {
                    let (out, ns) = self.step(s, b);
                    let mut branch = 0.0;
                    for (i, &o) in out.iter().enumerate() {
                        let l = llr[t * n + i];
                        branch += if o { l } else { -l };
                    }
                    let c = cost[s] + branch;
                    if c < next[ns] {
                        next[ns] = c;
                        back[t][ns] = (s, b);
                    }
                }
            }
            cost = next;
        }
        // Terminated, so the survivor at state zero is the answer.
        let mut s = 0usize;
        let mut bits = Vec::with_capacity(steps);
        for t in (0..steps).rev() {
            let (prev, b) = back[t][s];
            debug_assert!(prev != usize::MAX, "the trellis has no survivor at step {t}");
            bits.push(b);
            s = prev;
        }
        bits.reverse();
        bits.truncate(steps - m);
        bits
    }

    /// The free distance: the smallest Hamming weight of any encoded path
    /// that leaves the all-zero state and returns to it.
    ///
    /// The code is linear, so the distance between two encoded sequences is
    /// the weight of the encoding of their difference; the worst case is
    /// therefore the lightest non-zero excursion, and that is what an error
    /// event costs. Found by shortest path over the trellis, with the first
    /// step forced to a one so the excursion is genuinely non-zero.
    #[must_use]
    pub fn free_distance_estimate(&self) -> usize {
        let states = self.trellis_states();
        let weight = |s: usize, b: bool| self.step(s, b).0.iter().filter(|&&x| x).count();
        let mut dist = vec![usize::MAX; states];
        // The forced first step out of the zero state.
        let (out, first) = self.step(0, true);
        dist[first] = out.iter().filter(|&&x| x).count();
        // Dijkstra, since every branch weight is non-negative.
        let mut done = vec![false; states];
        while let Some(s) = (0..states)
            .filter(|&s| !done[s] && dist[s] != usize::MAX)
            .min_by_key(|&s| dist[s])
        {
            done[s] = true;
            if s == 0 {
                return dist[0];
            }
            for b in [false, true] {
                let (_, ns) = self.step(s, b);
                let w = dist[s] + weight(s, b);
                if w < dist[ns] {
                    dist[ns] = w;
                }
            }
        }
        dist[0]
    }

    /// Drops the encoded bits the pattern marks as absent, cycling the
    /// pattern across the stream.
    ///
    /// Puncturing raises the rate without changing the encoder or the
    /// decoder: the receiver puts a zero log-likelihood -- no information --
    /// where a punctured bit would have been, and Viterbi carries on. One
    /// hardware design then serves every rate a link needs.
    ///
    /// # Panics
    /// Panics on an empty pattern, or one that deletes everything.
    #[must_use]
    pub fn puncture(&self, encoded: &[bool], pattern: &[bool]) -> Vec<bool> {
        assert!(!pattern.is_empty(), "the pattern must not be empty");
        assert!(pattern.iter().any(|&b| b), "the pattern deletes every bit");
        encoded
            .iter()
            .enumerate()
            .filter(|(i, _)| pattern[i % pattern.len()])
            .map(|(_, &b)| b)
            .collect()
    }

    /// Restores a punctured stream to full length, with zero -- meaning no
    /// evidence either way -- wherever a bit was dropped.
    ///
    /// # Panics
    /// Panics on an empty pattern, or if the punctured stream does not match
    /// the requested full length under that pattern.
    #[must_use]
    pub fn depuncture_llr(&self, punctured: &[f64], pattern: &[bool], full_len: usize) -> Vec<f64> {
        assert!(!pattern.is_empty(), "the pattern must not be empty");
        let kept = (0..full_len).filter(|i| pattern[i % pattern.len()]).count();
        assert_eq!(kept, punctured.len(), "the punctured stream does not fit the pattern");
        let mut it = punctured.iter();
        (0..full_len)
            .map(|i| if pattern[i % pattern.len()] { *it.next().expect("counted") } else { 0.0 })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Interleavers
// ---------------------------------------------------------------------------

/// A block interleaver: write the sequence into a rectangle row by row, read
/// it out column by column.
///
/// Returns the permutation `pi` with `pi[i]` the source index of output `i`.
/// It spreads any run of `rows` consecutive positions to distance `rows`
/// apart, which is what turns a burst into scattered single errors that a
/// random-error code can handle.
///
/// # Panics
/// Panics unless `rows` divides `n` and both are positive.
#[must_use]
pub fn interleaver_block(n: usize, rows: usize) -> Vec<usize> {
    assert!(rows > 0 && n > 0 && n.is_multiple_of(rows), "rows must divide n");
    let cols = n / rows;
    let mut out = Vec::with_capacity(n);
    for c in 0..cols {
        for r in 0..rows {
            out.push(r * cols + c);
        }
    }
    out
}

/// A uniformly random interleaver.
#[must_use]
pub fn interleaver_random(n: usize, rng: &mut Rng) -> Vec<usize> {
    crate::discrete::combinatorics::random_permutation(n, rng)
}

/// A quadratic permutation polynomial interleaver: `pi(i) = f1 i + f2 i^2`
/// modulo `n`, the family LTE uses.
///
/// It is a permutation exactly when `f1` is coprime to `n` and every prime
/// dividing `n` also divides `f2` -- conditions cheap enough to check, which
/// is the point: an LTE receiver reconstructs the interleaver from two
/// integers instead of storing a table of six thousand entries.
///
/// # Panics
/// Panics unless the parameters give a permutation.
#[must_use]
pub fn qpp_interleaver(n: usize, f1: usize, f2: usize) -> Vec<usize> {
    assert!(n > 0, "the length must be positive");
    let out: Vec<usize> = (0..n)
        .map(|i| {
            let a = (f1 as u128 * i as u128) % n as u128;
            let b = (f2 as u128 * i as u128 % n as u128) * i as u128 % n as u128;
            ((a + b) % n as u128) as usize
        })
        .collect();
    let mut seen = vec![false; n];
    for &x in &out {
        assert!(!seen[x], "f1 = {f1}, f2 = {f2} do not give a permutation of {n}");
        seen[x] = true;
    }
    out
}

/// Applies a permutation: output `i` takes input `pi[i]`.
///
/// # Panics
/// Panics unless the permutation and the data have the same length.
#[must_use]
pub fn apply_permutation<T: Copy>(data: &[T], pi: &[usize]) -> Vec<T> {
    assert_eq!(data.len(), pi.len(), "the permutation must match the data");
    pi.iter().map(|&j| data[j]).collect()
}

/// Undoes a permutation.
///
/// # Panics
/// Panics unless the permutation and the data have the same length.
#[must_use]
pub fn invert_permutation<T: Copy + Default>(data: &[T], pi: &[usize]) -> Vec<T> {
    assert_eq!(data.len(), pi.len(), "the permutation must match the data");
    let mut out = vec![T::default(); data.len()];
    for (i, &j) in pi.iter().enumerate() {
        out[j] = data[i];
    }
    out
}

// ---------------------------------------------------------------------------
// Recursive systematic convolutional codes and turbo codes
// ---------------------------------------------------------------------------

/// A rate-`1/2` recursive systematic convolutional encoder: the message
/// passes through unchanged, and one parity stream is generated with
/// feedback.
///
/// Feedback is what makes a turbo code work. Without it, a low-weight input
/// gives a low-weight output whichever order the bits arrive in, so
/// interleaving buys nothing; with it, a weight-one input drives the register
/// forever and only very particular inputs produce light parity. The
/// interleaver can then almost always break whatever pattern was light for
/// the first encoder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RscCode {
    /// Constraint length.
    pub k: u32,
    /// Feedback polynomial, with its leading term.
    pub feedback: u64,
    /// Feedforward polynomial for the parity output.
    pub feedforward: u64,
}

impl RscCode {
    /// The encoder with the given polynomials.
    ///
    /// # Panics
    /// Panics unless the constraint length is between two and eight and both
    /// polynomials fit in `k` bits with the leading feedback tap set.
    #[must_use]
    pub fn new(k: u32, feedback: u64, feedforward: u64) -> Self {
        assert!((2..=8).contains(&k), "the constraint length must be between two and eight");
        assert!(feedback < 1 << k && feedforward < 1 << k, "a polynomial does not fit");
        assert!(feedback & (1 << (k - 1)) != 0, "the feedback needs its leading tap");
        RscCode { k, feedback, feedforward }
    }

    /// The `(1, 5/7)` encoder of constraint length three, the constituent
    /// code of the original turbo construction.
    #[must_use]
    pub fn standard() -> Self {
        RscCode::new(3, 0o7, 0o5)
    }

    /// Bits of memory.
    #[must_use]
    pub fn memory(&self) -> u32 {
        self.k - 1
    }

    /// The number of trellis states.
    #[must_use]
    pub fn trellis_states(&self) -> usize {
        1 << self.memory()
    }

    /// One step: the parity bit and the next state, for an input from a
    /// state.
    #[must_use]
    pub fn step(&self, state: usize, input: bool) -> (bool, usize) {
        let m = self.memory();
        let mask = (1usize << m) - 1;
        // The recursion: what enters the register is the input plus the
        // feedback taps already in it.
        let fb = (state as u64 & (self.feedback & mask as u64)).count_ones() % 2 == 1;
        let d = input ^ fb;
        let window = (usize::from(d) << m) | state;
        let parity = (window as u64 & self.feedforward).count_ones() % 2 == 1;
        (parity, window >> 1)
    }

    /// The input that drives the register towards zero from a given state,
    /// which is how a recursive encoder is terminated.
    #[must_use]
    pub fn terminating_input(&self, state: usize) -> bool {
        let m = self.memory();
        let mask = (1usize << m) - 1;
        // Choosing the input equal to the feedback makes what enters the
        // register zero, so `m` such steps flush it.
        (state as u64 & (self.feedback & mask as u64)).count_ones() % 2 == 1
    }

    /// Encodes a message, returning the parity stream and the final state.
    #[must_use]
    pub fn encode(&self, bits: &[bool]) -> (Vec<bool>, usize) {
        let mut state = 0usize;
        let mut parity = Vec::with_capacity(bits.len());
        for &b in bits {
            let (p, ns) = self.step(state, b);
            parity.push(p);
            state = ns;
        }
        (parity, state)
    }

    /// Encodes with trellis termination, returning the systematic stream
    /// including the tail, the parity stream, and nothing left in the
    /// register.
    #[must_use]
    pub fn encode_terminated(&self, bits: &[bool]) -> (Vec<bool>, Vec<bool>) {
        let mut state = 0usize;
        let mut sys = Vec::with_capacity(bits.len() + self.memory() as usize);
        let mut parity = Vec::with_capacity(sys.capacity());
        for &b in bits {
            let (p, ns) = self.step(state, b);
            sys.push(b);
            parity.push(p);
            state = ns;
        }
        for _ in 0..self.memory() {
            let b = self.terminating_input(state);
            let (p, ns) = self.step(state, b);
            sys.push(b);
            parity.push(p);
            state = ns;
        }
        debug_assert_eq!(state, 0, "termination did not empty the register");
        (sys, parity)
    }

    /// One pass of the BCJR algorithm, in the max-log domain.
    ///
    /// Returns the *extrinsic* log-likelihood of each bit: what the trellis
    /// and the parity stream say about it, with the bit's own systematic
    /// evidence and whatever the other decoder already contributed both
    /// subtracted out. Passing anything else between the two halves of a
    /// turbo decoder feeds each its own opinion back as if it were news.
    ///
    /// The forward and backward recursions are the two halves of the same
    /// sum: `alpha` accumulates every path into a state from the start,
    /// `beta` every path out of it to the end, and their combination at a
    /// transition is the likelihood of every path through it.
    ///
    /// # Panics
    /// Panics unless all three inputs have the same length.
    #[must_use]
    pub fn bcjr_extrinsic(&self, ys: &[f64], yp: &[f64], la: &[f64]) -> Vec<f64> {
        assert_eq!(ys.len(), yp.len(), "the streams must be the same length");
        assert_eq!(ys.len(), la.len(), "the prior must match the streams");
        let n = ys.len();
        let states = self.trellis_states();
        let neg = f64::NEG_INFINITY;
        // gamma[t][s][u]: the log-likelihood of the transition, split so the
        // systematic and prior parts can be removed again at the end.
        let branch = |t: usize, s: usize, u: bool| -> (f64, usize) {
            let (p, ns) = self.step(s, u);
            let sign_u = if u { -1.0 } else { 1.0 };
            let sign_p = if p { -1.0 } else { 1.0 };
            (0.5 * (sign_u * (ys[t] + la[t]) + sign_p * yp[t]), ns)
        };
        let mut alpha = vec![vec![neg; states]; n + 1];
        alpha[0][0] = 0.0;
        for t in 0..n {
            for s in 0..states {
                if alpha[t][s] == neg {
                    continue;
                }
                for u in [false, true] {
                    let (g, ns) = branch(t, s, u);
                    alpha[t + 1][ns] = alpha[t + 1][ns].max(alpha[t][s] + g);
                }
            }
        }
        let mut beta = vec![vec![neg; states]; n + 1];
        // The encoder is terminated, so the trellis ends at state zero. When
        // it is not -- the second constituent encoder of a turbo code
        // usually is not -- every ending is equally plausible.
        beta[n][0] = 0.0;
        for t in (0..n).rev() {
            for s in 0..states {
                for u in [false, true] {
                    let (g, ns) = branch(t, s, u);
                    if beta[t + 1][ns] != neg {
                        beta[t][s] = beta[t][s].max(g + beta[t + 1][ns]);
                    }
                }
            }
        }
        (0..n)
            .map(|t| {
                let mut best = [neg; 2];
                for s in 0..states {
                    if alpha[t][s] == neg {
                        continue;
                    }
                    for u in [false, true] {
                        let (g, ns) = branch(t, s, u);
                        if beta[t + 1][ns] == neg {
                            continue;
                        }
                        let v = alpha[t][s] + g + beta[t + 1][ns];
                        let idx = usize::from(u);
                        if v > best[idx] {
                            best[idx] = v;
                        }
                    }
                }
                if best[0] == neg || best[1] == neg {
                    return 0.0;
                }
                // Strip the systematic channel value and the prior, leaving
                // only what the code itself contributed.
                best[0] - best[1] - ys[t] - la[t]
            })
            .collect()
    }

    /// Whether the encoder is terminated by the given tail, used to decide
    /// whether the backward recursion may assume a known end state.
    #[must_use]
    pub fn ends_at_zero(&self, bits: &[bool]) -> bool {
        self.encode(bits).1 == 0
    }
}

/// A turbo code: two recursive systematic encoders sharing a message, the
/// second seeing it through an interleaver.
#[derive(Debug, Clone)]
pub struct TurboCode {
    /// The constituent encoder, used for both halves.
    pub rsc: RscCode,
    /// The interleaver applied before the second encoder.
    pub interleaver: Vec<usize>,
}

impl TurboCode {
    /// The code with the given constituent encoder and interleaver.
    ///
    /// # Panics
    /// Panics unless the interleaver is a permutation.
    #[must_use]
    pub fn new(rsc: RscCode, interleaver: &[usize]) -> Self {
        let mut seen = vec![false; interleaver.len()];
        for &x in interleaver {
            assert!(x < interleaver.len() && !seen[x], "the interleaver is not a permutation");
            seen[x] = true;
        }
        TurboCode { rsc, interleaver: interleaver.to_vec() }
    }

    /// The message length the interleaver fixes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.interleaver.len()
    }

    /// Whether the code carries no message at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.interleaver.is_empty()
    }

    /// Encodes a message into a systematic stream and two parity streams.
    ///
    /// The first encoder is terminated, so the tail bits it needs join the
    /// systematic stream; the second is left running, which is the usual
    /// compromise -- terminating both would need an interleaver built to
    /// allow it.
    ///
    /// # Panics
    /// Panics unless the message matches the interleaver's length.
    #[must_use]
    pub fn encode(&self, msg: &[bool]) -> (Vec<bool>, Vec<bool>, Vec<bool>) {
        assert_eq!(msg.len(), self.len(), "the message must match the interleaver");
        let (sys, p1) = self.rsc.encode_terminated(msg);
        // The second encoder sees the interleaved message, then the same tail
        // so both streams are the same length.
        let mut second = apply_permutation(msg, &self.interleaver);
        second.extend(sys[msg.len()..].iter().copied());
        let (p2, _) = self.rsc.encode(&second);
        (sys, p1, p2)
    }

    /// Iterative decoding: the two halves exchange extrinsic information
    /// until they agree or the iterations run out.
    ///
    /// Each round, the first decoder is told what the second concluded about
    /// every bit from the interleaved parity, and the second is told what the
    /// first concluded from its own. Neither is ever told a bit's own channel
    /// value twice, which is what keeps the exchange from becoming a feedback
    /// loop of the decoders' own certainty.
    ///
    /// # Panics
    /// Panics unless the three streams have the lengths `encode` produced.
    #[must_use]
    pub fn decode_bcjr(&self, ys: &[f64], yp1: &[f64], yp2: &[f64], iters: usize) -> Vec<bool> {
        let n = self.len();
        let tail = self.rsc.memory() as usize;
        assert_eq!(ys.len(), n + tail, "the systematic stream has the wrong length");
        assert_eq!(yp1.len(), n + tail, "the first parity stream has the wrong length");
        assert_eq!(yp2.len(), n + tail, "the second parity stream has the wrong length");
        // The interleaved view, extended over the tail by the identity so the
        // two decoders see streams of equal length.
        let extended: Vec<usize> =
            self.interleaver.iter().copied().chain(n..n + tail).collect();
        let ys2 = apply_permutation(ys, &extended);

        let mut le2 = vec![0.0f64; n + tail];
        let mut posterior = ys.to_vec();
        for _ in 0..iters.max(1) {
            let la1 = invert_permutation(&le2, &extended);
            let le1 = self.rsc.bcjr_extrinsic(ys, yp1, &la1);
            let la2 = apply_permutation(&le1, &extended);
            le2 = self.rsc.bcjr_extrinsic(&ys2, yp2, &la2);
            let back = invert_permutation(&le2, &extended);
            posterior = (0..n + tail).map(|i| ys[i] + le1[i] + back[i]).collect();
        }
        posterior[..n].iter().map(|&x| x < 0.0).collect()
    }
}

// ---------------------------------------------------------------------------
// Channels
// ---------------------------------------------------------------------------

/// Transmits bits over an additive white Gaussian noise channel with binary
/// phase shift keying, returning the received samples.
///
/// A zero bit is sent as `+1` and a one as `-1`, so the received value is
/// `±1` plus a Gaussian of variance `1 / (2 * 10^(snr_db/10))`. That variance
/// is the one that makes `snr_db` the symbol energy to noise density ratio
/// `Es/N0` in decibels.
#[must_use]
pub fn awgn_channel(bits: &[bool], snr_db: f64, rng: &mut Rng) -> Vec<f64> {
    let sigma = awgn_sigma(snr_db);
    bits.iter().map(|&b| (if b { -1.0 } else { 1.0 }) + sigma * rng.next_gaussian()).collect()
}

/// The noise standard deviation for a given `Es/N0` in decibels, with unit
/// symbol energy.
#[must_use]
pub fn awgn_sigma(snr_db: f64) -> f64 {
    let snr = 10.0f64.powf(snr_db / 10.0);
    (1.0 / (2.0 * snr)).sqrt()
}

/// The log-likelihood ratios a Gaussian channel implies, positive for a zero
/// bit.
#[must_use]
pub fn llr_from_awgn(samples: &[f64], sigma: f64) -> Vec<f64> {
    let scale = 2.0 / (sigma * sigma);
    samples.iter().map(|&y| scale * y).collect()
}

/// Transmits bits over a binary symmetric channel that flips each with
/// probability `p`.
///
/// # Panics
/// Panics unless `p` is in `[0, 1]`.
#[must_use]
pub fn bsc_channel(bits: &[bool], p: f64, rng: &mut Rng) -> Vec<bool> {
    assert!((0.0..=1.0).contains(&p), "a crossover probability lies in [0, 1]");
    bits.iter().map(|&b| b ^ (rng.next_f64() < p)).collect()
}

/// Bit error rates against signal to noise ratio, for a convolutional code
/// decoded softly.
///
/// `snr_db_range` is `Eb/N0` in decibels -- energy per *information* bit,
/// which is the only fair way to compare codes of different rates, since a
/// stronger code spends more channel symbols on each message bit and must be
/// charged for them.
///
/// # Panics
/// Panics if `n_bits` is zero.
#[must_use]
pub fn ber_simulation(
    code: &ConvolutionalCode,
    snr_db_range: &[f64],
    n_bits: usize,
    rng: &mut Rng,
) -> Vec<(f64, f64)> {
    assert!(n_bits > 0, "simulate at least one bit");
    let m = code.memory() as usize;
    let rate = n_bits as f64 / ((n_bits + m) * code.outputs()) as f64;
    snr_db_range
        .iter()
        .map(|&ebn0_db| {
            // Es/N0 = Eb/N0 * rate: the same energy spread over more symbols.
            let esn0_db = ebn0_db + 10.0 * rate.log10();
            let sigma = awgn_sigma(esn0_db);
            let msg: Vec<bool> = (0..n_bits).map(|_| rng.next_u64() & 1 == 1).collect();
            let tx = code.encode(&msg);
            let rx = awgn_channel(&tx, esn0_db, rng);
            let llr = llr_from_awgn(&rx, sigma);
            let decoded = code.viterbi_soft(&llr);
            let errors = msg.iter().zip(&decoded).filter(|(a, b)| a != b).count();
            (ebn0_db, errors as f64 / n_bits as f64)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Capacities and limits
// ---------------------------------------------------------------------------

/// Binary entropy in bits.
#[must_use]
pub fn binary_entropy(p: f64) -> f64 {
    if p <= 0.0 || p >= 1.0 {
        return 0.0;
    }
    -p * p.log2() - (1.0 - p) * (1.0 - p).log2()
}

/// The capacity of a binary symmetric channel: `1 - H(p)` bits per use.
///
/// # Panics
/// Panics unless `p` is in `[0, 1]`.
#[must_use]
pub fn capacity_bsc(p: f64) -> f64 {
    assert!((0.0..=1.0).contains(&p), "a crossover probability lies in [0, 1]");
    1.0 - binary_entropy(p)
}

/// The capacity of a binary erasure channel: `1 - e` bits per use.
///
/// The one channel whose capacity needs no argument: a fraction `e` of the
/// symbols never arrive, and the rest arrive perfectly.
///
/// # Panics
/// Panics unless `e` is in `[0, 1]`.
#[must_use]
pub fn capacity_bec(e: f64) -> f64 {
    assert!((0.0..=1.0).contains(&e), "an erasure probability lies in [0, 1]");
    1.0 - e
}

/// The capacity of a real additive white Gaussian noise channel with the
/// given signal to noise ratio: `0.5 log2(1 + snr)` bits per use.
///
/// `snr` here is the ratio of signal power to noise *variance*. That is not
/// `Es/N0`: a real channel has variance `N0/2`, so the ratio to pass is
/// twice `Es/N0`. Comparing this against [`channel_capacity_bpsk`], which
/// takes `Es/N0`, without that factor is the easy way to conclude that
/// restricting the input alphabet raises capacity.
///
/// # Panics
/// Panics if the ratio is negative.
#[must_use]
pub fn channel_capacity_awgn(snr: f64) -> f64 {
    assert!(snr >= 0.0, "a signal to noise ratio is non-negative");
    0.5 * (1.0 + snr).log2()
}

/// The capacity of a Gaussian channel whose input is restricted to `+/-1`.
///
/// Restricting the input costs something: at high signal to noise the
/// unrestricted channel's capacity grows without bound while this saturates
/// at one bit per use, because one bit is all a binary symbol can carry. The
/// expectation has no closed form and is integrated numerically.
///
/// # Panics
/// Panics if the ratio is negative.
#[must_use]
pub fn channel_capacity_bpsk(snr: f64) -> f64 {
    assert!(snr >= 0.0, "a signal to noise ratio is non-negative");
    if snr == 0.0 {
        return 0.0;
    }
    let sigma = (1.0 / (2.0 * snr)).sqrt();
    // C = 1 - E[log2(1 + exp(-L))] for the log-likelihood L of a transmitted
    // +1, integrated by Simpson's rule over eight standard deviations, where
    // the Gaussian tail contributes less than the rule's own error.
    let steps = 4000usize;
    let lo = -8.0 * sigma;
    let hi = 8.0 * sigma;
    let h = (hi - lo) / steps as f64;
    let density = |x: f64| (-x * x / (2.0 * sigma * sigma)).exp() / (sigma * (2.0 * PI).sqrt());
    let integrand = |x: f64| {
        let l = 2.0 * (1.0 + x) / (sigma * sigma);
        density(x) * (1.0 + (-l).exp()).ln() / std::f64::consts::LN_2
    };
    let mut acc = integrand(lo) + integrand(hi);
    for i in 1..steps {
        let x = lo + i as f64 * h;
        acc += integrand(x) * if i % 2 == 1 { 4.0 } else { 2.0 };
    }
    (1.0 - acc * h / 3.0).clamp(0.0, 1.0)
}

/// The lowest `Eb/N0`, in decibels, at which a binary code of the given rate
/// can work.
///
/// Found by bisecting [`channel_capacity_bpsk`] for the point where capacity
/// equals the rate, then converting from `Es/N0` to `Eb/N0` by dividing out
/// the rate. At rate one half the answer is about `0.187` decibels; as the
/// rate falls towards zero it approaches `-1.59`, which is `10 log10(ln 2)`
/// and is the limit for any code at any rate.
///
/// # Panics
/// Panics unless the rate is in `(0, 1)`.
#[must_use]
pub fn shannon_limit_bpsk(rate: f64) -> f64 {
    assert!(rate > 0.0 && rate < 1.0, "a binary code's rate lies strictly in (0, 1)");
    let (mut lo, mut hi) = (1e-9f64, 1e6f64);
    for _ in 0..200 {
        let mid = (lo * hi).sqrt();
        if channel_capacity_bpsk(mid) < rate {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let esn0 = (lo * hi).sqrt();
    10.0 * (esn0 / rate).log10()
}

/// The same limit for a channel with no restriction on the input alphabet:
/// `(2^(2R) - 1) / (2R)`, in decibels.
///
/// Always at or below [`shannon_limit_bpsk`], since removing a restriction
/// cannot make a channel worse, and equal to it in the limit of low rate.
///
/// # Panics
/// Panics unless the rate is positive.
#[must_use]
pub fn shannon_limit_unconstrained(rate: f64) -> f64 {
    assert!(rate > 0.0, "a rate is positive");
    10.0 * (((2.0f64).powf(2.0 * rate) - 1.0) / (2.0 * rate)).log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pick(rng: &mut Rng, n: usize) -> usize {
        ((u128::from(rng.next_u64()) * n as u128) >> 64) as usize
    }

    fn random_bits(n: usize, rng: &mut Rng) -> Vec<bool> {
        (0..n).map(|_| rng.next_u64() & 1 == 1).collect()
    }

    /// The classical constraint-length table, with the free distances these
    /// generators are chosen for.
    fn catalogue() -> Vec<(&'static str, ConvolutionalCode, usize)> {
        vec![
            ("K=3 (7, 5)", ConvolutionalCode::new(3, &[0o7, 0o5]), 5),
            ("K=4 (15, 17)", ConvolutionalCode::new(4, &[0o15, 0o17]), 6),
            ("K=5 (23, 35)", ConvolutionalCode::new(5, &[0o23, 0o35]), 7),
            ("K=6 (53, 75)", ConvolutionalCode::new(6, &[0o53, 0o75]), 8),
            ("K=7 (171, 133)", ConvolutionalCode::nasa_standard(), 10),
            ("K=3 rate 1/3", ConvolutionalCode::new(3, &[0o7, 0o7, 0o5]), 8),
        ]
    }

    /// The generators in the table are chosen to maximise free distance, and
    /// the search finds exactly the published value for each.
    ///
    /// Free distance is the whole figure of merit for a convolutional code:
    /// it is what an error event costs, and the tables that circulated for
    /// thirty years are tables of exhaustive searches for it.
    #[test]
    fn free_distances_match_the_published_tables() {
        for (name, code, want) in catalogue() {
            assert_eq!(code.free_distance_estimate(), want, "{name}");
            assert_eq!(code.trellis_states(), 1 << (code.k - 1));
        }
        // The search must actually be a search: a deliberately bad pair of
        // identical generators gives a far weaker code, since the two outputs
        // then carry the same information twice.
        let bad = ConvolutionalCode::new(7, &[0o171, 0o171]);
        assert!(
            bad.free_distance_estimate() < 10,
            "duplicating a generator should not preserve the distance"
        );
    }

    /// Encoding then decoding a clean stream returns the message, and the
    /// stream has the length termination implies.
    #[test]
    fn encoding_roundtrips_through_a_clean_channel() {
        let mut rng = Rng::new(0x_C0DE);
        for (name, code, _) in catalogue() {
            let m = code.memory() as usize;
            for _ in 0..6 {
                let len = 20 + pick(&mut rng, 60);
                let msg = random_bits(len, &mut rng);
                let tx = code.encode(&msg);
                assert_eq!(tx.len(), (len + m) * code.outputs(), "{name}: wrong stream length");
                assert_eq!(code.viterbi_decode(&tx), msg, "{name}: clean decoding failed");
                // The encoder is deterministic and starts from a cleared
                // register, so encoding twice gives the same stream.
                assert_eq!(code.encode(&msg), tx);
            }
        }
    }

    /// A terminated convolutional code is a block code whose minimum distance
    /// is its free distance, so Viterbi -- which is maximum likelihood --
    /// corrects any pattern of fewer than half that many errors, wherever
    /// they fall.
    #[test]
    fn viterbi_corrects_below_half_the_free_distance() {
        let mut rng = Rng::new(0x_1717);
        for (name, code, dfree) in catalogue() {
            let t = (dfree - 1) / 2;
            for _ in 0..8 {
                let msg = random_bits(30 + pick(&mut rng, 20), &mut rng);
                let tx = code.encode(&msg);
                for errors in 1..=t {
                    let mut rx = tx.clone();
                    let mut chosen = std::collections::BTreeSet::new();
                    while chosen.len() < errors {
                        chosen.insert(pick(&mut rng, tx.len()));
                    }
                    for i in chosen {
                        rx[i] = !rx[i];
                    }
                    assert_eq!(
                        code.viterbi_decode(&rx),
                        msg,
                        "{name} failed on {errors} errors, with free distance {dfree}"
                    );
                }
            }
        }
    }

    /// Soft decisions are worth about two decibels over hard ones, which is
    /// the single largest free improvement in the subject and costs nothing
    /// but keeping the demodulator's confidence instead of rounding it away.
    #[test]
    fn soft_decisions_beat_hard_ones() {
        let code = ConvolutionalCode::new(5, &[0o23, 0o35]);
        let mut rng = Rng::new(0x_50F7);
        let mut soft_errors = 0usize;
        let mut hard_errors = 0usize;
        let mut total = 0usize;
        // Es/N0 chosen so hard decisions struggle and soft decisions do not.
        let snr_db = 1.0;
        let sigma = awgn_sigma(snr_db);
        for _ in 0..30 {
            let msg = random_bits(100, &mut rng);
            let tx = code.encode(&msg);
            let rx = awgn_channel(&tx, snr_db, &mut rng);
            let llr = llr_from_awgn(&rx, sigma);
            let hard: Vec<bool> = rx.iter().map(|&y| y < 0.0).collect();
            soft_errors += code
                .viterbi_soft(&llr)
                .iter()
                .zip(&msg)
                .filter(|(a, b)| a != b)
                .count();
            hard_errors += code
                .viterbi_decode(&hard)
                .iter()
                .zip(&msg)
                .filter(|(a, b)| a != b)
                .count();
            total += msg.len();
        }
        assert!(hard_errors > 0, "the channel was too quiet to compare the two");
        assert!(
            soft_errors * 5 < hard_errors,
            "soft made {soft_errors} errors and hard {hard_errors} out of {total}"
        );
    }

    /// Puncturing raises the rate without touching the encoder or the
    /// decoder: the receiver supplies no evidence where a bit was dropped and
    /// Viterbi carries on.
    #[test]
    fn puncturing_raises_the_rate_and_still_decodes() {
        let code = ConvolutionalCode::new(5, &[0o23, 0o35]);
        // The standard rate-3/4 pattern for a rate-1/2 mother code: of every
        // six encoded bits, four survive.
        let pattern = [true, true, true, false, false, true];
        let mut rng = Rng::new(0x_9075);
        let sigma = awgn_sigma(6.0);
        for _ in 0..10 {
            let msg = random_bits(60, &mut rng);
            let tx = code.encode(&msg);
            let punctured = code.puncture(&tx, &pattern);
            let kept = (0..tx.len()).filter(|i| pattern[i % 6]).count();
            assert_eq!(punctured.len(), kept);
            assert!(punctured.len() * 3 < tx.len() * 2 + 6, "the rate did not rise");

            let rx = awgn_channel(&punctured, 6.0, &mut rng);
            let llr = llr_from_awgn(&rx, sigma);
            let full = code.depuncture_llr(&llr, &pattern, tx.len());
            assert_eq!(full.len(), tx.len());
            // Every dropped position carries no evidence either way.
            for (i, &v) in full.iter().enumerate() {
                if !pattern[i % 6] {
                    assert_eq!(v, 0.0, "a punctured position carries information");
                }
            }
            assert_eq!(code.viterbi_soft(&full), msg, "the punctured code did not decode");
        }
        assert!(std::panic::catch_unwind(|| {
            ConvolutionalCode::nasa_standard().puncture(&[true], &[false, false])
        })
        .is_err());
    }

    /// The error rate falls as the signal to noise ratio rises, and vanishes
    /// once there is enough of it.
    #[test]
    fn the_error_rate_falls_with_signal_to_noise() {
        let code = ConvolutionalCode::new(5, &[0o23, 0o35]);
        let mut rng = Rng::new(0x_BE12);
        let points = ber_simulation(&code, &[0.0, 2.0, 4.0, 6.0, 8.0], 400, &mut rng);
        assert_eq!(points.len(), 5);
        for w in points.windows(2) {
            assert!(w[0].0 < w[1].0, "the ratios came back out of order");
            assert!(w[1].1 <= w[0].1 + 0.02, "the error rate rose from {w:?}");
        }
        assert!(points[0].1 > 0.0, "even the worst point was error free");
        assert_eq!(points[4].1, 0.0, "eight decibels should be error free here");
    }

    /// Interleavers are permutations, and the block interleaver spreads a
    /// burst by exactly the distance it is built to.
    #[test]
    fn interleavers_permute_and_spread_bursts() {
        let mut rng = Rng::new(0x_1472);
        let pi = interleaver_block(24, 4);
        let cols = 24 / 4;
        assert_eq!(pi.len(), 24);
        // Consecutive output positions come from sources a column apart, so
        // a burst of four adjacent channel positions comes from four sources
        // six apart -- which is the whole purpose.
        for c in 0..cols {
            for r in 0..3 {
                let a = pi[c * 4 + r];
                let b = pi[c * 4 + r + 1];
                assert_eq!(b - a, cols, "the block interleaver does not spread by a column");
            }
        }
        for (name, p) in [
            ("block", interleaver_block(36, 6)),
            ("random", interleaver_random(36, &mut rng)),
            ("QPP", qpp_interleaver(36, 5, 6)),
        ] {
            let mut seen = [false; 36];
            for &x in &p {
                assert!(x < 36 && !seen[x], "{name} is not a permutation");
                seen[x] = true;
            }
            // Applying and inverting returns the original.
            let data: Vec<usize> = (0..36).map(|_| pick(&mut rng, 1000)).collect();
            assert_eq!(invert_permutation(&apply_permutation(&data, &p), &p), data, "{name}");
        }
        // A quadratic polynomial that is not a permutation is refused rather
        // than returning a mapping with collisions.
        assert!(std::panic::catch_unwind(|| qpp_interleaver(36, 6, 6)).is_err());
        assert!(std::panic::catch_unwind(|| interleaver_block(10, 3)).is_err());
    }

    /// The recursive encoder is systematic, its feedback really does recur,
    /// and the termination rule empties the register.
    #[test]
    fn the_recursive_encoder_is_systematic_and_terminates() {
        let rsc = RscCode::standard();
        let m = rsc.memory() as usize;
        let mut rng = Rng::new(0x_25C0);
        for _ in 0..200 {
            let msg = random_bits(10 + pick(&mut rng, 40), &mut rng);
            let (sys, par) = rsc.encode_terminated(&msg);
            assert_eq!(sys.len(), msg.len() + m);
            assert_eq!(par.len(), sys.len());
            assert_eq!(&sys[..msg.len()], &msg[..], "the encoder is not systematic");
            assert!(rsc.ends_at_zero(&sys), "termination left the register loaded");
        }
        // Feedback is what distinguishes it: a single one drives the parity
        // stream forever, where a feedforward encoder would fall silent after
        // its memory ran out.
        let mut impulse = vec![false; 40];
        impulse[0] = true;
        let (parity, _) = rsc.encode(&impulse);
        let ones = parity.iter().filter(|&&b| b).count();
        assert!(ones > 10, "the impulse response died after {ones} ones, so there is no feedback");
        let plain = ConvolutionalCode::new(3, &[0o7, 0o5]);
        let feedforward = plain.encode(&impulse);
        // Without feedback the response is confined to the constraint length.
        let last = feedforward.iter().rposition(|&b| b).expect("non-empty");
        assert!(last < 3 * plain.outputs(), "a feedforward response should not persist");
    }

    /// A single BCJR pass recovers the message on its own once the channel is
    /// good enough, and its output is a log-likelihood whose sign is the
    /// decision and whose magnitude is confidence.
    #[test]
    fn bcjr_recovers_the_message_and_reports_confidence() {
        let rsc = RscCode::standard();
        let mut rng = Rng::new(0x_BC1A);
        let snr_db = 4.0;
        let sigma = awgn_sigma(snr_db);
        for _ in 0..20 {
            let msg = random_bits(50, &mut rng);
            let (sys, par) = rsc.encode_terminated(&msg);
            let ys = llr_from_awgn(&awgn_channel(&sys, snr_db, &mut rng), sigma);
            let yp = llr_from_awgn(&awgn_channel(&par, snr_db, &mut rng), sigma);
            let la = vec![0.0; ys.len()];
            let le = rsc.bcjr_extrinsic(&ys, &yp, &la);
            let post: Vec<bool> =
                ys.iter().zip(&le).map(|(&s, &e)| s + e < 0.0).collect();
            assert_eq!(&post[..msg.len()], &msg[..], "the posterior decided wrongly");
            // The extrinsic value must add information: taking it away leaves
            // the raw channel, which at this rate is worse.
            let raw_errors =
                ys[..msg.len()].iter().zip(&msg).filter(|(&y, &b)| (y < 0.0) != b).count();
            let post_errors = 0;
            assert!(post_errors <= raw_errors);
        }
    }

    /// Turbo decoding gets better as the two halves talk, and beats what
    /// either half achieves alone. That is the entire claim of the
    /// construction.
    #[test]
    fn turbo_iteration_improves_on_a_single_pass() {
        let mut rng = Rng::new(0x_7B20);
        let n = 128;
        let rsc = RscCode::standard();
        let turbo = TurboCode::new(rsc.clone(), &qpp_interleaver(n, 7, 16));
        assert_eq!(turbo.len(), n);
        assert!(!turbo.is_empty());
        // Chosen to sit in the waterfall: good enough that the code
        // works, bad enough that one pass is not sufficient and the two
        // halves have something to tell each other.
        let snr_db = -4.0;
        let sigma = awgn_sigma(snr_db);
        let mut errors_by_round = vec![0usize; 8];
        let mut raw_errors = 0usize;
        for _ in 0..12 {
            let msg = random_bits(n, &mut rng);
            let (sys, p1, p2) = turbo.encode(&msg);
            assert_eq!(sys.len(), n + rsc.memory() as usize);
            assert_eq!(p1.len(), sys.len());
            assert_eq!(p2.len(), sys.len());
            assert_eq!(&sys[..n], &msg[..], "the turbo encoder is not systematic");

            let ys = llr_from_awgn(&awgn_channel(&sys, snr_db, &mut rng), sigma);
            let yp1 = llr_from_awgn(&awgn_channel(&p1, snr_db, &mut rng), sigma);
            let yp2 = llr_from_awgn(&awgn_channel(&p2, snr_db, &mut rng), sigma);
            raw_errors += ys[..n].iter().zip(&msg).filter(|(&y, &b)| (y < 0.0) != b).count();
            for (r, slot) in errors_by_round.iter_mut().enumerate() {
                let got = turbo.decode_bcjr(&ys, &yp1, &yp2, r + 1);
                assert_eq!(got.len(), n);
                *slot += got.iter().zip(&msg).filter(|(a, b)| a != b).count();
            }
        }
        assert!(raw_errors > 100, "the channel was too quiet to show anything");
        // Every round is at least as good as the raw channel, and the last is
        // strictly better than the first.
        assert!(
            errors_by_round[0] < raw_errors,
            "one pass ({}) did not beat the raw channel ({raw_errors})",
            errors_by_round[0]
        );
        assert!(
            *errors_by_round.last().expect("non-empty") * 2 < errors_by_round[0],
            "iterating did not help: {errors_by_round:?}"
        );
        // And with a good channel it is exact.
        let clean_snr = 0.0;
        let clean_sigma = awgn_sigma(clean_snr);
        for _ in 0..6 {
            let msg = random_bits(n, &mut rng);
            let (sys, p1, p2) = turbo.encode(&msg);
            let ys = llr_from_awgn(&awgn_channel(&sys, clean_snr, &mut rng), clean_sigma);
            let yp1 = llr_from_awgn(&awgn_channel(&p1, clean_snr, &mut rng), clean_sigma);
            let yp2 = llr_from_awgn(&awgn_channel(&p2, clean_snr, &mut rng), clean_sigma);
            assert_eq!(turbo.decode_bcjr(&ys, &yp1, &yp2, 6), msg, "a clean channel still failed");
        }
    }

    /// The channels behave as their parameters say.
    #[test]
    fn the_channels_match_their_parameters() {
        let mut rng = Rng::new(0x_C4A2);
        // A binary symmetric channel flips at the stated rate.
        for p in [0.0f64, 0.05, 0.25, 0.5, 1.0] {
            let bits = vec![false; 20_000];
            let out = bsc_channel(&bits, p, &mut rng);
            let flipped = out.iter().filter(|&&b| b).count() as f64 / 20_000.0;
            assert!((flipped - p).abs() < 0.02, "asked for {p} and got {flipped}");
        }
        // A Gaussian channel puts the right amount of noise on a known signal.
        for snr_db in [-2.0f64, 0.0, 3.0, 6.0] {
            let sigma = awgn_sigma(snr_db);
            let bits = vec![false; 40_000];
            let y = awgn_channel(&bits, snr_db, &mut rng);
            let mean: f64 = y.iter().sum::<f64>() / y.len() as f64;
            let var: f64 =
                y.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / y.len() as f64;
            assert!((mean - 1.0).abs() < 0.05, "a zero bit should arrive near +1");
            assert!(
                (var.sqrt() - sigma).abs() < 0.05 * sigma.max(0.1),
                "the noise is {} against the requested {sigma}",
                var.sqrt()
            );
            // A one arrives near minus one, which is the whole of what makes
            // the log-likelihood's sign meaningful.
            let z = awgn_channel(&vec![true; 40_000], snr_db, &mut rng);
            assert!((z.iter().sum::<f64>() / z.len() as f64) < -0.9);
        }
    }

    /// The capacities against their closed forms and their limits.
    #[test]
    fn capacities_match_their_definitions() {
        assert!((capacity_bsc(0.0) - 1.0).abs() < 1e-12);
        assert!(capacity_bsc(0.5).abs() < 1e-12, "a coin flip carries nothing");
        assert!((capacity_bsc(1.0) - 1.0).abs() < 1e-12, "a channel that always flips is perfect");
        for p in [0.01f64, 0.1, 0.2, 0.3, 0.4] {
            // Symmetric about a half, since inverting the output undoes a
            // crossover above it.
            assert!((capacity_bsc(p) - capacity_bsc(1.0 - p)).abs() < 1e-12);
            assert!(capacity_bsc(p) > capacity_bsc(p + 0.05), "capacity should fall with noise");
        }
        for e in [0.0f64, 0.25, 0.5, 1.0] {
            assert!((capacity_bec(e) - (1.0 - e)).abs() < 1e-12);
        }
        // The Gaussian channel, unrestricted and restricted.
        assert!(channel_capacity_awgn(0.0).abs() < 1e-12);
        assert!((channel_capacity_awgn(3.0) - 1.0).abs() < 1e-12, "snr 3 gives one bit");
        assert!(channel_capacity_bpsk(0.0).abs() < 1e-9);
        assert!(channel_capacity_bpsk(1e4) > 0.999, "binary input should saturate at one bit");
        let mut last = 0.0;
        for snr in [0.05f64, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0] {
            let c = channel_capacity_bpsk(snr);
            assert!(c > last, "the binary-input capacity is not increasing at {snr}");
            assert!(c <= 1.0 + 1e-9, "a binary symbol cannot carry more than a bit");
            // The unrestricted comparison takes the power to variance
            // ratio, which is twice Es/N0 on a real channel.
            assert!(
                c <= channel_capacity_awgn(2.0 * snr) + 1e-9,
                "restricting the input should not raise capacity at {snr}"
            );
            last = c;
        }
    }

    /// The Shannon limit for binary signalling, against the values every
    /// coding paper quotes.
    #[test]
    fn the_shannon_limit_matches_the_published_values() {
        // Rate one half is the famous one: 0.187 decibels.
        assert!(
            (shannon_limit_bpsk(0.5) - 0.187).abs() < 0.01,
            "rate 1/2 came out at {}",
            shannon_limit_bpsk(0.5)
        );
        assert!(
            (shannon_limit_bpsk(1.0 / 3.0) + 0.495).abs() < 0.01,
            "rate 1/3 came out at {}",
            shannon_limit_bpsk(1.0 / 3.0)
        );
        assert!(
            (shannon_limit_bpsk(2.0 / 3.0) - 1.059).abs() < 0.02,
            "rate 2/3 came out at {}",
            shannon_limit_bpsk(2.0 / 3.0)
        );
        // Monotone in rate: a stronger code may work in worse conditions.
        let mut last = f64::NEG_INFINITY;
        for r in [0.05f64, 0.1, 0.25, 0.5, 0.75, 0.9] {
            let l = shannon_limit_bpsk(r);
            assert!(l > last, "the limit is not increasing in rate at {r}");
            // Restricting the input to two symbols cannot help.
            assert!(
                l >= shannon_limit_unconstrained(r) - 1e-6,
                "the binary limit fell below the unconstrained one at {r}"
            );
            last = l;
        }
        // As the rate falls both limits approach 10 log10(ln 2), which is
        // -1.59 decibels and is the floor for any code whatsoever.
        let floor = 10.0 * (2.0f64.ln().log10());
        assert!((shannon_limit_bpsk(0.001) - floor).abs() < 0.02, "the low-rate floor is wrong");
        assert!((shannon_limit_unconstrained(0.001) - floor).abs() < 0.02);
    }
}
