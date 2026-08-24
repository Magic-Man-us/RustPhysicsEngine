//! Infinite impulse response filters: RBJ biquads, second-order-section
//! cascades, and classical designs (Butterworth, Chebyshev I/II,
//! elliptic, Bessel) via analog prototypes, frequency transformation,
//! and the bilinear transform.
//!
//! Frequencies are in Hz against an explicit sample rate. The elliptic
//! prototype follows Orfanidis' lecture notes (the same construction as
//! scipy's `ellipap`); Chebyshev and Butterworth prototypes are the
//! textbook pole formulas.
//!
//! The pre-Part-3 first-order RC filters remain here unchanged.

use crate::error::SolveError;
use crate::fractals::Complex;
use crate::math::constants::PI;
use crate::numerical::polynomial_roots;
use crate::signal_processing::exponential_moving_average;
use crate::special::{elliptic_k, jacobi_elliptic};

const TWO_PI: f64 = 2.0 * PI;

fn cis(theta: f64) -> Complex {
    Complex::new(theta.cos(), theta.sin())
}

const C_ZERO: Complex = Complex { re: 0.0, im: 0.0 };
const C_ONE: Complex = Complex { re: 1.0, im: 0.0 };

fn csqrt(z: Complex) -> Complex {
    let r = z.norm().sqrt();
    let th = z.arg() / 2.0;
    Complex::new(r * th.cos(), r * th.sin())
}

// --- Biquad ------------------------------------------------------------

/// One second-order section in transposed direct form II, with the RBJ
/// cookbook designs as constructors. Coefficients are normalized
/// (a0 = 1); `a1`, `a2` are the denominator terms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Biquad {
    pub b0: f64,
    pub b1: f64,
    pub b2: f64,
    pub a1: f64,
    pub a2: f64,
    z1: f64,
    z2: f64,
}

impl Biquad {
    /// Build from raw normalized coefficients.
    #[must_use]
    pub fn from_coeffs(b0: f64, b1: f64, b2: f64, a1: f64, a2: f64) -> Self {
        Self { b0, b1, b2, a1, a2, z1: 0.0, z2: 0.0 }
    }

    /// Identity (pass-through) section.
    #[must_use]
    pub fn identity() -> Self {
        Self::from_coeffs(1.0, 0.0, 0.0, 0.0, 0.0)
    }

    fn rbj(b0: f64, b1: f64, b2: f64, a0: f64, a1: f64, a2: f64) -> Self {
        Self::from_coeffs(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
    }

    /// RBJ low-pass with resonance q at cutoff fc.
    #[must_use]
    pub fn lowpass(fc: f64, fs: f64, q: f64) -> Self {
        let w = TWO_PI * fc / fs;
        let alpha = w.sin() / (2.0 * q);
        let c = w.cos();
        Self::rbj((1.0 - c) / 2.0, 1.0 - c, (1.0 - c) / 2.0, 1.0 + alpha, -2.0 * c, 1.0 - alpha)
    }

    /// RBJ high-pass.
    #[must_use]
    pub fn highpass(fc: f64, fs: f64, q: f64) -> Self {
        let w = TWO_PI * fc / fs;
        let alpha = w.sin() / (2.0 * q);
        let c = w.cos();
        Self::rbj((1.0 + c) / 2.0, -(1.0 + c), (1.0 + c) / 2.0, 1.0 + alpha, -2.0 * c, 1.0 - alpha)
    }

    /// RBJ band-pass (constant 0 dB peak gain).
    #[must_use]
    pub fn bandpass(fc: f64, fs: f64, q: f64) -> Self {
        let w = TWO_PI * fc / fs;
        let alpha = w.sin() / (2.0 * q);
        let c = w.cos();
        Self::rbj(alpha, 0.0, -alpha, 1.0 + alpha, -2.0 * c, 1.0 - alpha)
    }

    /// RBJ notch.
    #[must_use]
    pub fn notch(fc: f64, fs: f64, q: f64) -> Self {
        let w = TWO_PI * fc / fs;
        let alpha = w.sin() / (2.0 * q);
        let c = w.cos();
        Self::rbj(1.0, -2.0 * c, 1.0, 1.0 + alpha, -2.0 * c, 1.0 - alpha)
    }

    /// RBJ all-pass (unit magnitude, phase rotation around fc).
    #[must_use]
    pub fn allpass(fc: f64, fs: f64, q: f64) -> Self {
        let w = TWO_PI * fc / fs;
        let alpha = w.sin() / (2.0 * q);
        let c = w.cos();
        Self::rbj(1.0 - alpha, -2.0 * c, 1.0 + alpha, 1.0 + alpha, -2.0 * c, 1.0 - alpha)
    }

    /// RBJ peaking EQ: ±gain_db at fc.
    #[must_use]
    pub fn peaking(fc: f64, fs: f64, q: f64, gain_db: f64) -> Self {
        let a = 10.0_f64.powf(gain_db / 40.0);
        let w = TWO_PI * fc / fs;
        let alpha = w.sin() / (2.0 * q);
        let c = w.cos();
        Self::rbj(
            1.0 + alpha * a,
            -2.0 * c,
            1.0 - alpha * a,
            1.0 + alpha / a,
            -2.0 * c,
            1.0 - alpha / a,
        )
    }

    /// RBJ low shelf with shelf slope S (S = 1 is steepest without ripple).
    #[must_use]
    pub fn lowshelf(fc: f64, fs: f64, slope: f64, gain_db: f64) -> Self {
        let a = 10.0_f64.powf(gain_db / 40.0);
        let w = TWO_PI * fc / fs;
        let c = w.cos();
        let alpha = w.sin() / 2.0 * ((a + 1.0 / a) * (1.0 / slope - 1.0) + 2.0).sqrt();
        let beta = 2.0 * a.sqrt() * alpha;
        Self::rbj(
            a * ((a + 1.0) - (a - 1.0) * c + beta),
            2.0 * a * ((a - 1.0) - (a + 1.0) * c),
            a * ((a + 1.0) - (a - 1.0) * c - beta),
            (a + 1.0) + (a - 1.0) * c + beta,
            -2.0 * ((a - 1.0) + (a + 1.0) * c),
            (a + 1.0) + (a - 1.0) * c - beta,
        )
    }

    /// RBJ high shelf.
    #[must_use]
    pub fn highshelf(fc: f64, fs: f64, slope: f64, gain_db: f64) -> Self {
        let a = 10.0_f64.powf(gain_db / 40.0);
        let w = TWO_PI * fc / fs;
        let c = w.cos();
        let alpha = w.sin() / 2.0 * ((a + 1.0 / a) * (1.0 / slope - 1.0) + 2.0).sqrt();
        let beta = 2.0 * a.sqrt() * alpha;
        Self::rbj(
            a * ((a + 1.0) + (a - 1.0) * c + beta),
            -2.0 * a * ((a - 1.0) + (a + 1.0) * c),
            a * ((a + 1.0) + (a - 1.0) * c - beta),
            (a + 1.0) - (a - 1.0) * c + beta,
            2.0 * ((a - 1.0) - (a + 1.0) * c),
            (a + 1.0) - (a - 1.0) * c - beta,
        )
    }

    /// One sample through the transposed direct form II.
    pub fn process(&mut self, x: f64) -> f64 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }

    /// Filter a whole block.
    pub fn process_block(&mut self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&v| self.process(v)).collect()
    }

    /// Clear the state.
    pub fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }

    /// Prime the state so a constant input `v` is already in steady
    /// state (used by [`filtfilt`] for transient-free starts).
    pub fn prime(&mut self, v: f64) {
        let h1 = (self.b0 + self.b1 + self.b2) / (1.0 + self.a1 + self.a2);
        let y = h1 * v;
        self.z2 = self.b2 * v - self.a2 * y;
        self.z1 = self.b1 * v - self.a1 * y + self.z2;
    }

    /// Complex response at frequency f (Hz).
    #[must_use]
    pub fn freq_response(&self, f: f64, fs: f64) -> Complex {
        let z1 = cis(-TWO_PI * f / fs);
        let z2 = z1 * z1;
        let num = Complex::new(self.b0, 0.0)
            + Complex::new(self.b1, 0.0) * z1
            + Complex::new(self.b2, 0.0) * z2;
        let den = C_ONE + Complex::new(self.a1, 0.0) * z1 + Complex::new(self.a2, 0.0) * z2;
        num / den
    }

    /// Coefficients as (\[b0, b1, b2\], \[1, a1, a2\]).
    #[must_use]
    pub fn coeffs(&self) -> ([f64; 3], [f64; 3]) {
        ([self.b0, self.b1, self.b2], [1.0, self.a1, self.a2])
    }

    /// Both poles strictly inside the unit circle (Jury criterion).
    #[must_use]
    pub fn is_stable(&self) -> bool {
        self.a2.abs() < 1.0 && self.a1.abs() < 1.0 + self.a2
    }
}

// --- Second-order-section cascade --------------------------------------

/// A cascade of biquads with an overall gain.
#[derive(Debug, Clone)]
pub struct Sos {
    pub sections: Vec<Biquad>,
    pub gain: f64,
}

impl Sos {
    /// One sample through the whole cascade.
    pub fn process(&mut self, x: f64) -> f64 {
        let mut v = x * self.gain;
        for s in self.sections.iter_mut() {
            v = s.process(v);
        }
        v
    }

    /// Filter a whole block.
    pub fn process_block(&mut self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&v| self.process(v)).collect()
    }

    /// Clear all section states.
    pub fn reset(&mut self) {
        for s in self.sections.iter_mut() {
            s.reset();
        }
    }

    /// Complex response at frequency f (Hz).
    #[must_use]
    pub fn freq_response(&self, f: f64, fs: f64) -> Complex {
        let mut h = Complex::new(self.gain, 0.0);
        for s in &self.sections {
            h = h * s.freq_response(f, fs);
        }
        h
    }

    /// Expand the cascade into (b, a) polynomial coefficients in z⁻¹.
    #[must_use]
    pub fn to_tf(&self) -> (Vec<f64>, Vec<f64>) {
        let mut b = vec![self.gain];
        let mut a = vec![1.0];
        let mul = |p: &[f64], q: &[f64]| -> Vec<f64> {
            let mut out = vec![0.0; p.len() + q.len() - 1];
            for (i, &pv) in p.iter().enumerate() {
                for (j, &qv) in q.iter().enumerate() {
                    out[i + j] += pv * qv;
                }
            }
            out
        };
        for s in &self.sections {
            b = mul(&b, &[s.b0, s.b1, s.b2]);
            a = mul(&a, &[1.0, s.a1, s.a2]);
        }
        (b, a)
    }

    /// Poles of every section.
    #[must_use]
    pub fn poles(&self) -> Vec<Complex> {
        self.sections.iter().flat_map(|s| quad_roots(1.0, s.a1, s.a2)).collect()
    }

    /// Zeros of every section.
    #[must_use]
    pub fn zeros(&self) -> Vec<Complex> {
        self.sections
            .iter()
            .flat_map(|s| {
                if s.b0 != 0.0 {
                    quad_roots(s.b0, s.b1, s.b2)
                } else {
                    quad_roots(1.0, 0.0, 0.0)
                        .into_iter()
                        .take(0)
                        .collect()
                }
            })
            .collect()
    }
}

/// Roots of c0 z² + c1 z + c2 (drops the trailing degenerate root when
/// c2 = 0 leaves a first-order factor).
fn quad_roots(c0: f64, c1: f64, c2: f64) -> Vec<Complex> {
    if c2 == 0.0 {
        if c1 == 0.0 {
            return Vec::new();
        }
        return vec![Complex::new(-c1 / c0, 0.0)];
    }
    let disc = c1 * c1 - 4.0 * c0 * c2;
    if disc >= 0.0 {
        let sq = disc.sqrt();
        vec![
            Complex::new((-c1 + sq) / (2.0 * c0), 0.0),
            Complex::new((-c1 - sq) / (2.0 * c0), 0.0),
        ]
    } else {
        let sq = (-disc).sqrt() / (2.0 * c0);
        let re = -c1 / (2.0 * c0);
        vec![Complex::new(re, sq), Complex::new(re, -sq)]
    }
}

// --- Analog prototypes --------------------------------------------------

struct Zpk {
    zeros: Vec<Complex>,
    poles: Vec<Complex>,
    gain: f64,
}

fn butterworth_proto(order: usize) -> Zpk {
    let poles = (0..order)
        .map(|k| {
            let theta = PI * (2.0 * k as f64 + 1.0) / (2.0 * order as f64) + PI / 2.0;
            cis(theta)
        })
        .collect();
    Zpk { zeros: Vec::new(), poles, gain: 1.0 }
}

fn chebyshev1_proto(order: usize, ripple_db: f64) -> Zpk {
    let eps = (10.0_f64.powf(ripple_db / 10.0) - 1.0).sqrt();
    let mu = (1.0 / eps).asinh() / order as f64;
    let poles: Vec<Complex> = (0..order)
        .map(|k| {
            let theta = PI * (2.0 * k as f64 + 1.0) / (2.0 * order as f64);
            Complex::new(-mu.sinh() * theta.sin(), mu.cosh() * theta.cos())
        })
        .collect();
    let mut gain = poles.iter().fold(C_ONE, |acc, p| acc * (C_ZERO - *p)).re;
    if order.is_multiple_of(2) {
        gain /= (1.0 + eps * eps).sqrt();
    }
    Zpk { zeros: Vec::new(), poles, gain }
}

fn chebyshev2_proto(order: usize, atten_db: f64) -> Zpk {
    let eps = 1.0 / (10.0_f64.powf(atten_db / 10.0) - 1.0).sqrt();
    let mu = (1.0 / eps).asinh() / order as f64;
    let mut zeros = Vec::new();
    let mut poles = Vec::new();
    for k in 0..order {
        let theta = PI * (2.0 * k as f64 + 1.0) / (2.0 * order as f64);
        // Chebyshev-I poles, inverted.
        let p1 = Complex::new(-mu.sinh() * theta.sin(), mu.cosh() * theta.cos());
        poles.push(C_ONE / p1);
        // Zeros on the imaginary axis (skip θ = π/2 where cos = 0).
        if theta.cos().abs() > 1e-12 {
            zeros.push(Complex::new(0.0, 1.0 / theta.cos()));
        }
    }
    let num = zeros.iter().fold(C_ONE, |acc, z| acc * (C_ZERO - *z));
    let den = poles.iter().fold(C_ONE, |acc, p| acc * (C_ZERO - *p));
    Zpk { zeros, poles, gain: (den / num).re }
}

/// Complex inverse Jacobi sn by the descending Landen recursion
/// (Orfanidis eq. 56; the construction used by scipy).
fn arc_jac_sn(w: Complex, m: f64) -> Complex {
    let complement = |kx: f64| ((1.0 - kx) * (1.0 + kx)).sqrt();
    let k = m.sqrt();
    let mut ks = vec![k];
    for _ in 0..30 {
        let k_ = *ks.last().unwrap();
        if k_ == 0.0 {
            break;
        }
        let k_p = complement(k_);
        ks.push((1.0 - k_p) / (1.0 + k_p));
        if *ks.last().unwrap() < 1e-16 {
            ks.push(0.0);
            break;
        }
    }
    let big_k: f64 = ks[1..].iter().map(|&v| 1.0 + v).product::<f64>() * PI / 2.0;
    let mut wn = w;
    for pair in ks.windows(2) {
        let (kn, knext) = (pair[0], pair[1]);
        let inner = csqrt(C_ONE - wn * wn * Complex::new(kn * kn, 0.0));
        wn = wn * Complex::new(2.0, 0.0)
            / ((C_ONE + inner) * Complex::new(1.0 + knext, 0.0));
    }
    // u = (2/π) asin(w_final), complex asin.
    let z = wn;
    let asin_z = {
        // asin z = −i ln(iz + sqrt(1−z²))
        let iz = Complex::new(-z.im, z.re);
        let root = csqrt(C_ONE - z * z);
        let arg = iz + root;
        let ln = Complex::new(arg.norm().ln(), arg.arg());
        Complex::new(ln.im, -ln.re)
    };
    asin_z * Complex::new(2.0 / PI * big_k, 0.0)
}

/// Solve the elliptic degree equation for m given N and m1 (nome series;
/// Orfanidis eq. 49).
fn ellip_deg(n: usize, m1: f64) -> f64 {
    let k1 = elliptic_k(m1);
    let k1p = elliptic_k(1.0 - m1);
    let q1 = (-PI * k1p / k1).exp();
    let q = q1.powf(1.0 / n as f64);
    let mut num = 0.0;
    for i in 0..8usize {
        num += q.powi((i * (i + 1)) as i32);
    }
    let mut den = 1.0;
    for i in 1..9usize {
        den += 2.0 * q.powi((i * i) as i32);
    }
    16.0 * q * (num / den).powi(4)
}

fn elliptic_proto(order: usize, ripple_db: f64, atten_db: f64) -> Zpk {
    if order == 1 {
        let p = -(1.0 / (10.0_f64.powf(0.1 * ripple_db) - 1.0)).sqrt();
        return Zpk { zeros: Vec::new(), poles: vec![Complex::new(p, 0.0)], gain: -p };
    }
    let eps_sq = 10.0_f64.powf(0.1 * ripple_db) - 1.0;
    let eps = eps_sq.sqrt();
    let ck1_sq = eps_sq / (10.0_f64.powf(0.1 * atten_db) - 1.0);
    let m = ellip_deg(order, ck1_sq);
    let capk = elliptic_k(m);
    let n = order;
    let js: Vec<f64> = (0..)
        .map(|i| (1 - n % 2) as f64 + 2.0 * i as f64)
        .take_while(|&j| j < n as f64)
        .collect();
    let mut s_vals = Vec::new();
    let mut c_vals = Vec::new();
    let mut d_vals = Vec::new();
    for &j in &js {
        let (s, c, d) = jacobi_elliptic(j * capk / n as f64, m);
        s_vals.push(s);
        c_vals.push(c);
        d_vals.push(d);
    }
    // Zeros: ±i/(√m·s) for the nonzero sn values.
    let mut zeros = Vec::new();
    for &s in &s_vals {
        if s.abs() > 1e-12 {
            let z = 1.0 / (m.sqrt() * s);
            zeros.push(Complex::new(0.0, z));
            zeros.push(Complex::new(0.0, -z));
        }
    }
    // v0 from the inverse sc with complementary modulus.
    let r = {
        let zc = arc_jac_sn(Complex::new(0.0, 1.0 / eps), ck1_sq);
        zc.im
    };
    let v0 = capk * r / (n as f64 * elliptic_k(ck1_sq));
    let (sv, cv, dv) = jacobi_elliptic(v0, 1.0 - m);
    let mut poles = Vec::new();
    for i in 0..js.len() {
        let denom = 1.0 - (d_vals[i] * sv).powi(2);
        let p = Complex::new(
            -(c_vals[i] * d_vals[i] * sv * cv) / denom,
            -(s_vals[i] * dv) / denom,
        );
        poles.push(p);
        // Conjugate for genuinely complex poles.
        if p.im.abs() > 1e-10 * p.norm() {
            poles.push(p.conjugate());
        }
    }
    let num = zeros.iter().fold(C_ONE, |acc, z| acc * (C_ZERO - *z));
    let den = poles.iter().fold(C_ONE, |acc, p| acc * (C_ZERO - *p));
    let mut gain = (den / num).re;
    if n.is_multiple_of(2) {
        gain /= (1.0 + eps_sq).sqrt();
    }
    Zpk { zeros, poles, gain }
}

fn bessel_proto(order: usize) -> Zpk {
    // Reversed Bessel polynomial θ_n(s) = Σ a_k s^k,
    // a_k = (2n−k)!/(2^{n−k}·k!·(n−k)!).
    let fact = |k: usize| -> f64 { (1..=k).map(|v| v as f64).product::<f64>().max(1.0) };
    let n = order;
    let coeffs: Vec<f64> = (0..=n)
        .rev()
        .map(|k| fact(2 * n - k) / (2.0_f64.powi((n - k) as i32) * fact(k) * fact(n - k)))
        .collect();
    let mut poles = polynomial_roots(&coeffs).expect("Bessel polynomial roots failed");
    // Normalize so |H(j·1)| = 1/√2 (magnitude normalization): find the
    // −3 dB frequency of the unnormalized filter, then scale poles.
    let mag_sq = |w: f64, poles: &[Complex]| -> f64 {
        let mut den = C_ONE;
        for p in poles {
            den = den * (Complex::new(0.0, w) - *p);
        }
        let k: f64 = poles.iter().fold(C_ONE, |acc, p| acc * (C_ZERO - *p)).re;
        (k * k) / den.norm_sq()
    };
    let mut lo = 1e-6_f64;
    let mut hi = 1e6_f64;
    for _ in 0..200 {
        let mid = (lo * hi).sqrt();
        if mag_sq(mid, &poles) > 0.5 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let w3 = (lo * hi).sqrt();
    for p in poles.iter_mut() {
        *p = Complex::new(p.re / w3, p.im / w3);
    }
    let gain = poles.iter().fold(C_ONE, |acc, p| acc * (C_ZERO - *p)).re;
    Zpk { zeros: Vec::new(), poles, gain }
}

// --- Frequency transforms and bilinear ----------------------------------

/// Filter band selectors for the classical designs; frequencies in Hz.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IirKind {
    Lowpass(f64),
    Highpass(f64),
    Bandpass(f64, f64),
    Bandstop(f64, f64),
}

fn prewarp(f: f64, fs: f64) -> f64 {
    2.0 * fs * (PI * f / fs).tan()
}

fn lp_to_lp(p: Zpk, w: f64) -> Zpk {
    let scale = |v: &Complex| Complex::new(v.re * w, v.im * w);
    let deg = p.poles.len() as i32 - p.zeros.len() as i32;
    Zpk {
        zeros: p.zeros.iter().map(&scale).collect(),
        poles: p.poles.iter().map(&scale).collect(),
        gain: p.gain * w.powi(deg),
    }
}

fn lp_to_hp(p: Zpk, w: f64) -> Zpk {
    let inv = |v: &Complex| Complex::new(w, 0.0) / *v;
    let deg = p.poles.len() - p.zeros.len();
    let mut zeros: Vec<Complex> = p.zeros.iter().map(&inv).collect();
    let poles: Vec<Complex> = p.poles.iter().map(&inv).collect();
    let num = p.zeros.iter().fold(C_ONE, |acc, z| acc * (C_ZERO - *z));
    let den = p.poles.iter().fold(C_ONE, |acc, q| acc * (C_ZERO - *q));
    zeros.extend(std::iter::repeat_n(C_ZERO, deg));
    Zpk { zeros, poles, gain: p.gain * (num / den).re }
}

fn lp_to_bp(p: Zpk, w0: f64, bw: f64) -> Zpk {
    let map = |v: &Complex| -> (Complex, Complex) {
        let half = *v * Complex::new(bw / 2.0, 0.0);
        let disc = csqrt(half * half - Complex::new(w0 * w0, 0.0));
        (half + disc, half - disc)
    };
    let deg = p.poles.len() - p.zeros.len();
    let mut zeros = Vec::new();
    for z in &p.zeros {
        let (a, b) = map(z);
        zeros.push(a);
        zeros.push(b);
    }
    let mut poles = Vec::new();
    for q in &p.poles {
        let (a, b) = map(q);
        poles.push(a);
        poles.push(b);
    }
    zeros.extend(std::iter::repeat_n(C_ZERO, deg));
    Zpk { zeros, poles, gain: p.gain * bw.powi(deg as i32) }
}

fn lp_to_bs(p: Zpk, w0: f64, bw: f64) -> Zpk {
    let map = |v: &Complex| -> (Complex, Complex) {
        let half = Complex::new(bw / 2.0, 0.0) / *v;
        let disc = csqrt(half * half - Complex::new(w0 * w0, 0.0));
        (half + disc, half - disc)
    };
    let deg = p.poles.len() - p.zeros.len();
    let num = p.zeros.iter().fold(C_ONE, |acc, z| acc * (C_ZERO - *z));
    let den = p.poles.iter().fold(C_ONE, |acc, q| acc * (C_ZERO - *q));
    let mut zeros = Vec::new();
    for z in &p.zeros {
        let (a, b) = map(z);
        zeros.push(a);
        zeros.push(b);
    }
    let mut poles = Vec::new();
    for q in &p.poles {
        let (a, b) = map(q);
        poles.push(a);
        poles.push(b);
    }
    for _ in 0..deg {
        zeros.push(Complex::new(0.0, w0));
        zeros.push(Complex::new(0.0, -w0));
    }
    Zpk { zeros, poles, gain: p.gain * (num / den).re }
}

/// Bilinear transform of an analog (z, p, k) description to a digital
/// [`Sos`] at sample rate fs. `prewarp` optionally pins one analog
/// frequency (Hz) to its digital location.
#[must_use]
pub fn bilinear_transform(
    s_zeros: &[Complex],
    s_poles: &[Complex],
    gain: f64,
    fs: f64,
    prewarp_hz: Option<f64>,
) -> Sos {
    // Optional prewarp: scale the analog frequencies so `prewarp_hz` maps
    // exactly.
    let (zeros, poles, gain) = if let Some(fp) = prewarp_hz {
        let wd = TWO_PI * fp;
        let wa = 2.0 * fs * (PI * fp / fs).tan();
        let ratio = wa / wd;
        let scale = |v: &Complex| Complex::new(v.re * ratio, v.im * ratio);
        let deg = s_poles.len() as i32 - s_zeros.len() as i32;
        (
            s_zeros.iter().map(&scale).collect::<Vec<_>>(),
            s_poles.iter().map(&scale).collect::<Vec<_>>(),
            gain * ratio.powi(deg),
        )
    } else {
        (s_zeros.to_vec(), s_poles.to_vec(), gain)
    };
    let fs2 = Complex::new(2.0 * fs, 0.0);
    let map = |v: &Complex| (fs2 + *v) / (fs2 - *v);
    let deg = poles.len() - zeros.len();
    let mut zd: Vec<Complex> = zeros.iter().map(&map).collect();
    let pd: Vec<Complex> = poles.iter().map(&map).collect();
    zd.extend(std::iter::repeat_n(Complex::new(-1.0, 0.0), deg));
    let num = zeros.iter().fold(C_ONE, |acc, z| acc * (fs2 - *z));
    let den = poles.iter().fold(C_ONE, |acc, p| acc * (fs2 - *p));
    let k = gain * (num / den).re;
    zpk_to_sos(&zd, &pd, k)
}

/// Group a digital (z, p, k) set into second-order sections. Complex
/// values must come in conjugate pairs.
#[must_use]
pub fn zpk_to_sos(zeros: &[Complex], poles: &[Complex], gain: f64) -> Sos {
    // Split into conjugate pairs + reals, pairing largest-magnitude poles
    // (closest to the unit circle) with the nearest zeros.
    fn pair(vals: &[Complex]) -> (Vec<(Complex, Complex)>, Vec<f64>) {
        let mut pairs = Vec::new();
        let mut reals = Vec::new();
        let mut used = vec![false; vals.len()];
        for i in 0..vals.len() {
            if used[i] {
                continue;
            }
            if vals[i].im.abs() > 1e-9 * (1.0 + vals[i].norm()) {
                // Find its conjugate.
                let mut found = false;
                for j in i + 1..vals.len() {
                    if !used[j]
                        && (vals[j].re - vals[i].re).abs() < 1e-6 * (1.0 + vals[i].norm())
                        && (vals[j].im + vals[i].im).abs() < 1e-6 * (1.0 + vals[i].norm())
                    {
                        used[j] = true;
                        found = true;
                        break;
                    }
                }
                assert!(found, "complex value without conjugate partner");
                used[i] = true;
                pairs.push((vals[i], vals[i].conjugate()));
            } else {
                used[i] = true;
                reals.push(vals[i].re);
            }
        }
        (pairs, reals)
    }
    let (zpairs, mut zreals) = pair(zeros);
    let (ppairs, mut preals) = pair(poles);
    let mut sections: Vec<Biquad> = Vec::new();
    // Complex pole pairs.
    let mut zpair_iter = zpairs.into_iter();
    for (p, pc) in ppairs {
        let a1 = -(p + pc).re;
        let a2 = (p * pc).re;
        if let Some((z, zc)) = zpair_iter.next() {
            sections.push(Biquad::from_coeffs(1.0, -(z + zc).re, (z * zc).re, a1, a2));
        } else {
            let b1 = if let Some(z) = zreals.pop() { -z } else { f64::NAN };
            if b1.is_nan() {
                sections.push(Biquad::from_coeffs(1.0, 0.0, 0.0, a1, a2));
            } else {
                let b2 = if let Some(z2) = zreals.pop() { -z2 } else { 0.0 };
                // (1 + b1 z⁻¹)(1 + b2 z⁻¹)
                sections.push(Biquad::from_coeffs(1.0, b1 + b2, b1 * b2, a1, a2));
            }
        }
    }
    // Leftover complex zero pairs (more zeros than pole pairs): pair with
    // real poles.
    for (z, zc) in zpair_iter {
        let p1 = preals.pop().unwrap_or(0.0);
        let p2 = preals.pop().unwrap_or(0.0);
        sections.push(Biquad::from_coeffs(
            1.0,
            -(z + zc).re,
            (z * zc).re,
            -(p1 + p2),
            p1 * p2,
        ));
    }
    // Remaining real poles, two at a time.
    while let Some(p1) = preals.pop() {
        let p2 = preals.pop();
        let z1 = zreals.pop();
        let z2 = zreals.pop();
        let (a1, a2) = match p2 {
            Some(p2) => (-(p1 + p2), p1 * p2),
            None => (-p1, 0.0),
        };
        let (b1, b2) = match (z1, z2) {
            (Some(z1), Some(z2)) => (-(z1 + z2), z1 * z2),
            (Some(z1), None) => (-z1, 0.0),
            _ => (0.0, 0.0),
        };
        sections.push(Biquad::from_coeffs(1.0, b1, b2, a1, a2));
    }
    // Any leftover real zeros (more zeros than poles): absorb.
    while let Some(z1) = zreals.pop() {
        let z2 = zreals.pop();
        let (b1, b2) = match z2 {
            Some(z2) => (-(z1 + z2), z1 * z2),
            None => (-z1, 0.0),
        };
        sections.push(Biquad::from_coeffs(1.0, b1, b2, 0.0, 0.0));
    }
    if sections.is_empty() {
        sections.push(Biquad::identity());
    }
    Sos { sections, gain }
}

/// Digital (zeros, poles, gain) from transfer-function coefficient
/// arrays in z⁻¹ order (b\[0\] + b\[1\]z⁻¹ + …), using
/// `numerical::polynomial_roots`.
///
/// # Panics
/// Panics if either polynomial is degenerate (all zero).
#[must_use]
pub fn tf_to_zpk(b: &[f64], a: &[f64]) -> (Vec<Complex>, Vec<Complex>, f64) {
    let n = b.len().max(a.len());
    let pad = |v: &[f64]| -> Vec<f64> {
        let mut out = v.to_vec();
        out.resize(n, 0.0);
        out
    };
    let bp = pad(b);
    let ap = pad(a);
    // As polynomials in z (multiply through by z^{n-1}): coefficients are
    // already highest-power-first.
    let strip = |v: &[f64]| -> Vec<Complex> {
        if v.iter().all(|&c| c == 0.0) {
            panic!("degenerate polynomial in tf_to_zpk");
        }
        // Trailing zero coefficients give roots at z = 0. Use the
        // companion-matrix QR eigen solve: robust to multiple roots
        // (e.g. the (1 + z^-1)^n numerators of bilinear designs).
        let last_nonzero = v.iter().rposition(|&c| c != 0.0).unwrap();
        let mut roots = if last_nonzero == 0 {
            Vec::new()
        } else {
            let poly = &v[..=last_nonzero];
            let deg = poly.len() - 1;
            let lead = poly[0];
            let mut comp = crate::linalg::Matrix::zeros(deg, deg);
            for i in 0..deg {
                comp.set(0, i, -poly[i + 1] / lead);
                if i + 1 < deg {
                    comp.set(i + 1, i, 1.0);
                }
            }
            crate::linalg::eigenvalues_general(&comp, 60).expect("root finding failed")
        };
        roots.extend(std::iter::repeat_n(C_ZERO, v.len() - 1 - last_nonzero));
        roots
    };
    (strip(&bp), strip(&ap), b[0] / a[0])
}

// --- Classical designs ---------------------------------------------------

fn design(proto: Zpk, kind: IirKind, fs: f64) -> Sos {
    let analog = match kind {
        IirKind::Lowpass(fc) => lp_to_lp(proto, prewarp(fc, fs)),
        IirKind::Highpass(fc) => lp_to_hp(proto, prewarp(fc, fs)),
        IirKind::Bandpass(lo, hi) => {
            let wl = prewarp(lo, fs);
            let wh = prewarp(hi, fs);
            lp_to_bp(proto, (wl * wh).sqrt(), wh - wl)
        }
        IirKind::Bandstop(lo, hi) => {
            let wl = prewarp(lo, fs);
            let wh = prewarp(hi, fs);
            lp_to_bs(proto, (wl * wh).sqrt(), wh - wl)
        }
    };
    bilinear_transform(&analog.zeros, &analog.poles, analog.gain, fs, None)
}

/// Butterworth digital filter (maximally flat magnitude).
///
/// # Panics
/// Panics if `order == 0` or the band edges are invalid for fs.
#[must_use]
pub fn butterworth(order: usize, kind: IirKind, fs: f64) -> Sos {
    assert!(order > 0, "order must be positive");
    design(butterworth_proto(order), kind, fs)
}

/// Chebyshev type I (equiripple passband, `ripple_db` peak-to-peak).
///
/// # Panics
/// Panics if `order == 0`.
#[must_use]
pub fn chebyshev1(order: usize, ripple_db: f64, kind: IirKind, fs: f64) -> Sos {
    assert!(order > 0, "order must be positive");
    design(chebyshev1_proto(order, ripple_db), kind, fs)
}

/// Chebyshev type II (monotone passband, equiripple stopband at
/// −`atten_db`). The cutoff marks the stopband edge.
///
/// # Panics
/// Panics if `order == 0`.
#[must_use]
pub fn chebyshev2(order: usize, atten_db: f64, kind: IirKind, fs: f64) -> Sos {
    assert!(order > 0, "order must be positive");
    design(chebyshev2_proto(order, atten_db), kind, fs)
}

/// Elliptic (Cauer) filter: `ripple_db` passband ripple and `atten_db`
/// stopband attenuation.
///
/// # Panics
/// Panics if `order == 0`.
#[must_use]
pub fn elliptic(order: usize, ripple_db: f64, atten_db: f64, kind: IirKind, fs: f64) -> Sos {
    assert!(order > 0, "order must be positive");
    design(elliptic_proto(order, ripple_db, atten_db), kind, fs)
}

/// Bessel-Thomson filter (maximally flat group delay), −3 dB at the
/// cutoff.
///
/// # Panics
/// Panics if `order == 0`.
#[must_use]
pub fn bessel(order: usize, kind: IirKind, fs: f64) -> Sos {
    assert!(order > 0, "order must be positive");
    design(bessel_proto(order), kind, fs)
}

/// Minimum Butterworth order meeting a low-pass spec: passband edge,
/// stopband edge (Hz), maximum passband ripple and minimum stopband
/// attenuation (dB).
///
/// # Panics
/// Panics unless `0 < pass < stop < fs/2`.
#[must_use]
pub fn butterworth_order(pass: f64, stop: f64, ripple_db: f64, atten_db: f64, fs: f64) -> usize {
    assert!(pass > 0.0 && pass < stop && stop < fs / 2.0, "need 0 < pass < stop < fs/2");
    let wp = prewarp(pass, fs);
    let ws = prewarp(stop, fs);
    let num = (10.0_f64.powf(atten_db / 10.0) - 1.0) / (10.0_f64.powf(ripple_db / 10.0) - 1.0);
    (num.ln() / (2.0 * (ws / wp).ln())).ceil().max(1.0) as usize
}

// --- Application ---------------------------------------------------------

/// Zero-phase filtering: odd-reflection padding, steady-state priming
/// of every section, forward pass, backward pass (the same edge-
/// transient suppression goal as Gustafsson's method).
#[must_use]
pub fn filtfilt(sos: &Sos, x: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let pad = (3 * 2 * (sos.sections.len() + 1)).min(n - 1).max(1);
    // Odd extension: 2x0 − x[pad..1], x, 2x[n−1] − x[n−2..].
    let mut ext = Vec::with_capacity(n + 2 * pad);
    for i in (1..=pad).rev() {
        ext.push(2.0 * x[0] - x[i.min(n - 1)]);
    }
    ext.extend_from_slice(x);
    for i in 1..=pad {
        ext.push(2.0 * x[n - 1] - x[n - 1 - i.min(n - 1)]);
    }
    let run = |data: &[f64], sos: &Sos| -> Vec<f64> {
        let mut work = sos.clone();
        let first = data[0] / 1.0;
        let mut v0 = first * work.gain;
        for s in work.sections.iter_mut() {
            s.prime(v0);
            let h1 = (s.b0 + s.b1 + s.b2) / (1.0 + s.a1 + s.a2);
            v0 *= h1;
        }
        work.process_block(data)
    };
    let fwd = run(&ext, sos);
    let rev: Vec<f64> = fwd.into_iter().rev().collect();
    let back = run(&rev, sos);
    let full: Vec<f64> = back.into_iter().rev().collect();
    full[pad..pad + n].to_vec()
}

/// Direct-form II transposed filtering with arbitrary-order (b, a)
/// coefficient arrays in z⁻¹ order.
///
/// # Panics
/// Panics if `a` is empty or `a[0] == 0`.
#[must_use]
pub fn iir_apply(b: &[f64], a: &[f64], x: &[f64]) -> Vec<f64> {
    assert!(!a.is_empty() && a[0] != 0.0, "a[0] must be nonzero");
    let bn: Vec<f64> = b.iter().map(|&v| v / a[0]).collect();
    let an: Vec<f64> = a.iter().map(|&v| v / a[0]).collect();
    let order = bn.len().max(an.len());
    let mut state = vec![0.0; order];
    let get = |v: &[f64], i: usize| if i < v.len() { v[i] } else { 0.0 };
    x.iter()
        .map(|&xv| {
            let y = get(&bn, 0) * xv + state[0];
            for i in 0..order - 1 {
                state[i] = get(&bn, i + 1) * xv - get(&an, i + 1) * y + state[i + 1];
            }
            if order > 0 {
                state[order - 1] = 0.0;
            }
            y
        })
        .collect()
}

/// Impulse response of a cascade (n samples).
#[must_use]
pub fn impulse_response(sos: &Sos, n: usize) -> Vec<f64> {
    let mut work = sos.clone();
    work.reset();
    (0..n).map(|i| work.process(if i == 0 { 1.0 } else { 0.0 })).collect()
}

/// Step response of a cascade (n samples).
#[must_use]
pub fn step_response(sos: &Sos, n: usize) -> Vec<f64> {
    let mut work = sos.clone();
    work.reset();
    (0..n).map(|_| work.process(1.0)).collect()
}

/// Group delay in samples over `n_points` frequencies spanning
/// (0, fs/2): τ(ω) = −dφ/dω from the unwrapped phase.
#[must_use]
pub fn group_delay(sos: &Sos, n_points: usize, fs: f64) -> (Vec<f64>, Vec<f64>) {
    let n = n_points.max(3);
    let freqs: Vec<f64> = (0..n)
        .map(|i| fs / 2.0 * (i as f64 + 0.5) / (n as f64 + 1.0))
        .collect();
    let phases: Vec<f64> = freqs.iter().map(|&f| sos.freq_response(f, fs).arg()).collect();
    // Unwrap.
    let mut unwrapped = phases.clone();
    for i in 1..n {
        let mut d = unwrapped[i] - unwrapped[i - 1];
        while d > PI {
            unwrapped[i] -= TWO_PI;
            d = unwrapped[i] - unwrapped[i - 1];
        }
        while d < -PI {
            unwrapped[i] += TWO_PI;
            d = unwrapped[i] - unwrapped[i - 1];
        }
    }
    let delays: Vec<f64> = (0..n)
        .map(|i| {
            let (i0, i1) = if i == 0 {
                (0, 1)
            } else if i == n - 1 {
                (n - 2, n - 1)
            } else {
                (i - 1, i + 1)
            };
            let dw = TWO_PI * (freqs[i1] - freqs[i0]) / fs;
            -(unwrapped[i1] - unwrapped[i0]) / dw
        })
        .collect();
    (freqs, delays)
}

/// One-pole low-pass coefficients (b0, a1) for
/// y\[n\] = b0·x\[n\] + a1·y\[n−1\], with a1 = e^(−2π·fc/fs).
/// The pre-Part-3 `first_order_lowpass` is this filter with
/// α = dt/(RC + dt).
#[must_use]
pub fn one_pole_lowpass(fc: f64, fs: f64) -> (f64, f64) {
    let a1 = (-TWO_PI * fc / fs).exp();
    (1.0 - a1, a1)
}

/// DC-blocking filter: H(z) = (1 − z⁻¹)/(1 − r·z⁻¹), r slightly below 1.
#[must_use]
pub fn dc_blocker(r: f64) -> Biquad {
    Biquad::from_coeffs(1.0, -1.0, 0.0, -r, 0.0)
}

/// Chamberlin state-variable filter producing simultaneous low-pass,
/// high-pass, band-pass, and notch outputs.
pub struct Svf {
    f: f64,
    q1: f64,
    low: f64,
    band: f64,
}

impl Svf {
    /// One sample in, (low, high, band, notch) out.
    pub fn process(&mut self, x: f64) -> (f64, f64, f64, f64) {
        self.low += self.f * self.band;
        let high = x - self.low - self.q1 * self.band;
        self.band += self.f * high;
        let notch = high + self.low;
        (self.low, high, self.band, notch)
    }

    /// Clear the integrator states.
    pub fn reset(&mut self) {
        self.low = 0.0;
        self.band = 0.0;
    }
}

/// Build a Chamberlin SVF at cutoff fc with resonance q.
#[must_use]
pub fn state_variable_filter(fc: f64, fs: f64, q: f64) -> Svf {
    Svf { f: 2.0 * (PI * fc / fs).sin(), q1: 1.0 / q, low: 0.0, band: 0.0 }
}

/// IEC 61672 A-weighting as a digital cascade (bilinear transform of the
/// standard analog poles), normalized to exactly 0 dB at 1 kHz.
#[must_use]
pub fn a_weighting_filter(fs: f64) -> Sos {
    let p1 = -TWO_PI * 20.598997;
    let p2 = -TWO_PI * 107.65265;
    let p3 = -TWO_PI * 737.86223;
    let p4 = -TWO_PI * 12194.217;
    let zeros = vec![C_ZERO; 4];
    let poles: Vec<Complex> = [p1, p1, p2, p3, p4, p4]
        .iter()
        .map(|&p| Complex::new(p, 0.0))
        .collect();
    let mut sos = bilinear_transform(&zeros, &poles, 1.0, fs, None);
    let g = sos.freq_response(1000.0, fs).norm();
    sos.gain /= g;
    sos
}

/// IEC 61672 C-weighting, normalized to 0 dB at 1 kHz.
#[must_use]
pub fn c_weighting_filter(fs: f64) -> Sos {
    let p1 = -TWO_PI * 20.598997;
    let p4 = -TWO_PI * 12194.217;
    let zeros = vec![C_ZERO; 2];
    let poles: Vec<Complex> = [p1, p1, p4, p4].iter().map(|&p| Complex::new(p, 0.0)).collect();
    let mut sos = bilinear_transform(&zeros, &poles, 1.0, fs, None);
    let g = sos.freq_response(1000.0, fs).norm();
    sos.gain /= g;
    sos
}

/// RBJ Q for a given bandwidth in octaves at center fc:
/// 1/Q = 2·sinh(ln2/2 · BW · ω/sin ω).
#[must_use]
pub fn rbj_q_from_bandwidth(bw_octaves: f64, fc: f64, fs: f64) -> f64 {
    let w = TWO_PI * fc / fs;
    let arg = std::f64::consts::LN_2 / 2.0 * bw_octaves * w / w.sin();
    1.0 / (2.0 * arg.sinh())
}

// --- Pre-Part-3 first-order RC filters (unchanged) -----------------------

/// First-order RC low-pass filter: α = dt / (RC + dt)
///
/// # Panics
/// Panics if `dt <= 0` or `rc < 0`.
#[must_use]
pub fn first_order_lowpass(signal: &[f64], dt: f64, rc: f64) -> Vec<f64> {
    assert!(dt > 0.0, "time step dt must be positive");
    assert!(rc >= 0.0, "RC time constant must be non-negative");
    let alpha = dt / (rc + dt);
    exponential_moving_average(signal, alpha)
}

/// First-order RC high-pass filter: α = RC / (RC + dt)
///
/// # Panics
/// Panics if `dt <= 0` or `rc < 0`.
#[must_use]
pub fn first_order_highpass(signal: &[f64], dt: f64, rc: f64) -> Vec<f64> {
    assert!(dt > 0.0, "time step dt must be positive");
    assert!(rc >= 0.0, "RC time constant must be non-negative");
    if signal.is_empty() {
        return Vec::new();
    }
    let alpha = rc / (rc + dt);
    let mut output = Vec::with_capacity(signal.len());
    output.push(signal[0]);
    for i in 1..signal.len() {
        let prev = output[i - 1];
        output.push(alpha * (prev + signal[i] - signal[i - 1]));
    }
    output
}

/// Placeholder to keep the unused error type import when features are
/// trimmed.
#[allow(dead_code)]
fn _uses_solve_error(_e: SolveError) {}

#[cfg(test)]
mod tests {
    use super::*;

    fn mag_db(sos: &Sos, f: f64, fs: f64) -> f64 {
        20.0 * sos.freq_response(f, fs).norm().log10()
    }

    fn assert_stable(sos: &Sos) {
        for s in &sos.sections {
            assert!(s.is_stable(), "unstable section {s:?}");
        }
    }

    #[test]
    fn test_butterworth_halfpower_and_monotone() {
        let fs = 48000.0;
        for &order in &[2usize, 4, 7] {
            let sos = butterworth(order, IirKind::Lowpass(1000.0), fs);
            assert_stable(&sos);
            let g = sos.freq_response(1000.0, fs).norm();
            assert!((g - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-6, "order {order}: {g}");
            assert!((sos.freq_response(1.0, fs).norm() - 1.0).abs() < 1e-6);
            // Monotone decreasing.
            let mut prev = f64::MAX;
            for i in 1..200 {
                let f = 24000.0 * i as f64 / 200.0;
                let m = sos.freq_response(f, fs).norm();
                assert!(m <= prev + 1e-9, "not monotone at {f}");
                prev = m;
            }
        }
    }

    #[test]
    fn test_butterworth_highpass_bandpass_bandstop() {
        let fs = 48000.0;
        let hp = butterworth(4, IirKind::Highpass(1000.0), fs);
        assert_stable(&hp);
        assert!((hp.freq_response(1000.0, fs).norm() - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-6);
        assert!(hp.freq_response(20000.0, fs).norm() > 0.999);
        assert!(hp.freq_response(50.0, fs).norm() < 1e-4);

        let bp = butterworth(3, IirKind::Bandpass(800.0, 1200.0), fs);
        assert_stable(&bp);
        assert!(bp.freq_response(980.0, fs).norm() > 0.95);
        assert!(bp.freq_response(100.0, fs).norm() < 1e-3);
        assert!(bp.freq_response(10000.0, fs).norm() < 1e-2);

        let bs = butterworth(3, IirKind::Bandstop(800.0, 1200.0), fs);
        assert_stable(&bs);
        assert!(bs.freq_response(980.0, fs).norm() < 1e-3);
        assert!((bs.freq_response(50.0, fs).norm() - 1.0).abs() < 1e-3);
        assert!((bs.freq_response(20000.0, fs).norm() - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_chebyshev1_ripple() {
        let fs = 10000.0;
        let rp = 1.0;
        let sos = chebyshev1(5, rp, IirKind::Lowpass(1000.0), fs);
        assert_stable(&sos);
        // Passband: magnitude within [−rp, 0] dB, touching both bounds.
        let mut min_db = 0.0_f64;
        let mut max_db = -100.0_f64;
        for i in 1..500 {
            let f = 1000.0 * i as f64 / 500.0;
            let db = mag_db(&sos, f, fs);
            min_db = min_db.min(db);
            max_db = max_db.max(db);
            assert!(db < 0.01 && db > -rp - 0.05, "at {f}: {db} dB");
        }
        assert!(min_db < -rp + 0.05, "ripple floor {min_db}");
        assert!(max_db > -0.05, "ripple ceiling {max_db}");
    }

    #[test]
    fn test_chebyshev2_stopband() {
        let fs = 10000.0;
        let rs = 40.0;
        let sos = chebyshev2(5, rs, IirKind::Lowpass(1000.0), fs);
        assert_stable(&sos);
        // Stopband (beyond the edge): everything at or below −rs.
        for i in 0..200 {
            let f = 1000.0 + (4000.0 - 1000.0) * i as f64 / 200.0;
            let db = mag_db(&sos, f, fs);
            assert!(db <= -rs + 0.1, "at {f}: {db} dB");
        }
        // DC gain 1.
        assert!((sos.freq_response(1.0, fs).norm() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_elliptic_meets_spec() {
        let fs = 10000.0;
        let (rp, rs) = (1.0, 40.0);
        for &order in &[3usize, 4, 5] {
            let sos = elliptic(order, rp, rs, IirKind::Lowpass(1000.0), fs);
            assert_stable(&sos);
            // Passband within ripple, touching the floor (equiripple).
            let mut floor = 0.0_f64;
            for i in 1..300 {
                let f = 995.0 * i as f64 / 300.0;
                let db = mag_db(&sos, f, fs);
                assert!(db < 0.02 && db > -rp - 0.1, "order {order} at {f}: {db} dB");
                floor = floor.min(db);
            }
            assert!(floor < -rp + 0.1, "order {order}: passband floor {floor}");
            // Stopband: below −rs beyond the theoretical stopband edge
            // from the degree equation (analog ratio 1/k, unwarped back
            // to a digital frequency).
            let eps_sq = 10.0_f64.powf(0.1 * rp) - 1.0;
            let ck1_sq = eps_sq / (10.0_f64.powf(0.1 * rs) - 1.0);
            let m = ellip_deg(order, ck1_sq);
            let wa_stop = prewarp(1000.0, fs) / m.sqrt();
            let f_stop = fs / PI * (wa_stop / (2.0 * fs)).atan();
            for i in 0..=100 {
                let f = f_stop * 1.001 + (4900.0 - f_stop) * i as f64 / 100.0;
                let db = mag_db(&sos, f, fs);
                assert!(db <= -rs + 0.5, "order {order} at {f}: {db} dB (edge {f_stop})");
            }
        }
    }

    #[test]
    fn test_bessel_flat_group_delay() {
        let fs = 48000.0;
        let sos = bessel(4, IirKind::Lowpass(1000.0), fs);
        assert_stable(&sos);
        assert!((sos.freq_response(1000.0, fs).norm() - std::f64::consts::FRAC_1_SQRT_2).abs() < 0.02);
        // Group delay flat within the passband: compare at 100 Hz vs 800 Hz.
        let (freqs, delay) = group_delay(&sos, 512, fs);
        let d_at = |f: f64| {
            let i = freqs.iter().position(|&x| x >= f).unwrap();
            delay[i]
        };
        let d100 = d_at(100.0);
        let d800 = d_at(800.0);
        assert!((d100 - d800).abs() / d100 < 0.06, "delay {d100} vs {d800}");
    }

    #[test]
    fn test_butterworth_order_formula() {
        // Pass 1 kHz, stop 2 kHz, 1 dB / 40 dB at fs = 10 kHz: matches the
        // hand-evaluated analog estimate on prewarped edges.
        let n = butterworth_order(1000.0, 2000.0, 1.0, 40.0, 10000.0);
        let wp = 2.0 * 10000.0 * (PI * 0.1).tan();
        let ws = 2.0 * 10000.0 * (PI * 0.2).tan();
        let expect = (((10.0_f64.powf(4.0) - 1.0) / (10.0_f64.powf(0.1) - 1.0)).ln()
            / (2.0 * (ws / wp).ln()))
        .ceil() as usize;
        assert_eq!(n, expect);
        // A cutoff-at-pass-edge design of that order meets the stop spec.
        let sos = butterworth(n, IirKind::Lowpass(1000.0), 10000.0);
        assert!(mag_db(&sos, 2000.0, 10000.0) <= -40.0);
        // Tighter specs need higher orders.
        assert!(butterworth_order(1000.0, 1200.0, 1.0, 60.0, 10000.0) > n);
    }

    #[test]
    fn test_filtfilt_zero_phase() {
        let fs = 1000.0;
        let sos = butterworth(4, IirKind::Lowpass(100.0), fs);
        let n = 2000;
        let x: Vec<f64> = (0..n).map(|i| (TWO_PI * 20.0 * i as f64 / fs).sin()).collect();
        let y = filtfilt(&sos, &x);
        assert_eq!(y.len(), n);
        let mut dot = 0.0;
        let mut xx = 0.0;
        let mut yy = 0.0;
        for i in 100..n - 100 {
            dot += x[i] * y[i];
            xx += x[i] * x[i];
            yy += y[i] * y[i];
        }
        let corr = dot / (xx * yy).sqrt();
        assert!(corr > 0.99999, "phase shift detected: corr {corr}");
        // Magnitude squared: |H|² at 20 Hz.
        let expected = sos.freq_response(20.0, fs).norm_sq();
        assert!(((yy / xx).sqrt() - expected).abs() < 0.01);
    }

    #[test]
    fn test_iir_apply_matches_sos() {
        let fs = 8000.0;
        let sos = chebyshev1(4, 0.5, IirKind::Lowpass(800.0), fs);
        let (b, a) = sos.to_tf();
        let x: Vec<f64> = (0..200).map(|i| ((i * 7919) % 100) as f64 / 50.0 - 1.0).collect();
        let mut work = sos.clone();
        work.reset();
        let y_sos = work.process_block(&x);
        let y_tf = iir_apply(&b, &a, &x);
        for (u, v) in y_sos.iter().zip(&y_tf) {
            assert!((u - v).abs() < 1e-9, "{u} vs {v}");
        }
    }

    #[test]
    fn test_impulse_and_step_response() {
        let fs = 1000.0;
        let sos = butterworth(3, IirKind::Lowpass(50.0), fs);
        let h = impulse_response(&sos, 400);
        // Stable: decays to nothing.
        let head: f64 = h[..100].iter().map(|v| v.abs()).sum();
        let tail: f64 = h[300..].iter().map(|v| v.abs()).sum();
        assert!(tail < 1e-6 * head);
        // Step response settles at the DC gain (1).
        let s = step_response(&sos, 400);
        assert!((s[399] - 1.0).abs() < 1e-6);
        // Sum of impulse response equals DC gain.
        let sum: f64 = h.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_tf_zpk_sos_roundtrip() {
        let fs = 8000.0;
        let sos = butterworth(4, IirKind::Lowpass(1000.0), fs);
        let (b, a) = sos.to_tf();
        let (z, p, k) = tf_to_zpk(&b, &a);
        assert_eq!(p.len(), 4);
        let sos2 = zpk_to_sos(&z, &p, k);
        for &f in &[100.0, 1000.0, 3000.0] {
            let h1 = sos.freq_response(f, fs);
            let h2 = sos2.freq_response(f, fs);
            assert!((h1.norm() - h2.norm()).abs() < 1e-8);
        }
    }

    #[test]
    fn test_rbj_biquads() {
        let fs = 48000.0;
        // Peaking: +6 dB at fc, ~0 far away.
        let pk = Biquad::peaking(1000.0, fs, 1.0, 6.0);
        assert!((20.0 * pk.freq_response(1000.0, fs).norm().log10() - 6.0).abs() < 1e-9);
        assert!(20.0 * pk.freq_response(20.0, fs).norm().log10() < 0.2);
        // Notch kills fc.
        let nt = Biquad::notch(1000.0, fs, 5.0);
        assert!(nt.freq_response(1000.0, fs).norm() < 1e-10);
        // Allpass: |H| = 1 everywhere.
        let ap = Biquad::allpass(1000.0, fs, 0.7);
        for &f in &[100.0, 1000.0, 10000.0] {
            assert!((ap.freq_response(f, fs).norm() - 1.0).abs() < 1e-12);
        }
        // Shelves reach their gains.
        let ls = Biquad::lowshelf(1000.0, fs, 1.0, 6.0);
        assert!((20.0 * ls.freq_response(10.0, fs).norm().log10() - 6.0).abs() < 0.05);
        assert!(20.0 * ls.freq_response(20000.0, fs).norm().log10() < 0.2);
        let hs = Biquad::highshelf(1000.0, fs, 1.0, -6.0);
        assert!((20.0 * hs.freq_response(20000.0, fs).norm().log10() + 6.0).abs() < 0.1);
        // Lowpass biquad at fc with q = 1/√2 is −3 dB.
        let lp = Biquad::lowpass(1000.0, fs, std::f64::consts::FRAC_1_SQRT_2);
        assert!((lp.freq_response(1000.0, fs).norm() - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-9);
        // process matches freq_response on a tone.
        let mut lp2 = lp;
        let f0 = 500.0;
        let x: Vec<f64> = (0..4000).map(|i| (TWO_PI * f0 * i as f64 / fs).sin()).collect();
        let y = lp2.process_block(&x);
        let amp = y[2000..].iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        assert!((amp - lp.freq_response(f0, fs).norm()).abs() < 0.01);
    }

    #[test]
    fn test_quad_roots_recovers_known_factors() {
        // (z − a)(z − b) = z² − (a+b)z + ab.
        for &(a, b) in &[(0.5_f64, -0.25_f64), (2.0, 1.0), (-0.9, -0.1), (3.0, 3.0)] {
            let roots = quad_roots(1.0, -(a + b), a * b);
            assert_eq!(roots.len(), 2);
            assert!(roots.iter().all(|r| r.im == 0.0), "spurious imaginary part");
            let mut got = [roots[0].re, roots[1].re];
            got.sort_by(|x, y| x.partial_cmp(y).unwrap());
            let mut want = [a, b];
            want.sort_by(|x, y| x.partial_cmp(y).unwrap());
            assert!((got[0] - want[0]).abs() < 1e-12, "{got:?} vs {want:?}");
            assert!((got[1] - want[1]).abs() < 1e-12, "{got:?} vs {want:?}");
        }
        // A non-monic leading coefficient: 2z² − 6z + 4 = 2(z−2)(z−1).
        let r = quad_roots(2.0, -6.0, 4.0);
        assert!((r[0].re - 2.0).abs() < 1e-12 && (r[1].re - 1.0).abs() < 1e-12);
        // Negative discriminant gives a conjugate pair: z² + 1 → ±i.
        let c = quad_roots(1.0, 0.0, 1.0);
        assert_eq!(c.len(), 2);
        assert!(c[0].re.abs() < 1e-15 && (c[0].im - 1.0).abs() < 1e-12);
        assert!((c[1].im + 1.0).abs() < 1e-12, "not a conjugate pair");
        // z² − 2z cos θ + 1 has both roots on the unit circle at ±θ.
        let theta = 0.7_f64;
        let u = quad_roots(1.0, -2.0 * theta.cos(), 1.0);
        for root in &u {
            assert!((root.norm() - 1.0).abs() < 1e-12, "|z| = {}", root.norm());
            assert!((root.arg().abs() - theta).abs() < 1e-12);
        }
        // Vieta on the complex pair: sum = −c1/c0, product = c2/c0.
        let v = quad_roots(1.0, 0.4, 0.9);
        let sum = v[0] + v[1];
        let prod = v[0] * v[1];
        assert!((sum.re + 0.4).abs() < 1e-12 && sum.im.abs() < 1e-15);
        assert!((prod.re - 0.9).abs() < 1e-12 && prod.im.abs() < 1e-15);
        // Degenerate trailing coefficient leaves a first-order factor.
        let first = quad_roots(2.0, -4.0, 0.0);
        assert_eq!(first.len(), 1);
        assert!((first[0].re - 2.0).abs() < 1e-12);
        assert!(quad_roots(1.0, 0.0, 0.0).is_empty(), "no roots to report");
    }

    #[test]
    fn test_biquad_coeffs_describe_the_filter() {
        let fs = 48000.0;
        let q = std::f64::consts::FRAC_1_SQRT_2;
        let lp = Biquad::lowpass(1000.0, fs, q);
        let (b, a) = lp.coeffs();
        // The denominator is reported normalized.
        assert_eq!(a[0], 1.0);
        assert_eq!((a[1], a[2]), (lp.a1, lp.a2));
        assert_eq!(b, [lp.b0, lp.b1, lp.b2]);
        // H(z) at z = 1 (DC) and z = −1 (Nyquist) read straight off the
        // coefficients: a low-pass is unity at DC with a double zero at
        // Nyquist.
        let dc = (b[0] + b[1] + b[2]) / (a[0] + a[1] + a[2]);
        let nyq_num = b[0] - b[1] + b[2];
        assert!((dc - 1.0).abs() < 1e-12, "DC gain {dc}");
        assert!(nyq_num.abs() < 1e-18, "Nyquist numerator {nyq_num}");
        assert!((dc - lp.freq_response(0.0, fs).norm()).abs() < 1e-12);

        // A high-pass mirrors it: zero at DC, unity at Nyquist.
        let hp = Biquad::highpass(1000.0, fs, q);
        let (hb, ha) = hp.coeffs();
        assert!((hb[0] + hb[1] + hb[2]).abs() < 1e-18, "high-pass leaks DC");
        let hnyq = (hb[0] - hb[1] + hb[2]) / (ha[0] - ha[1] + ha[2]);
        assert!((hnyq - 1.0).abs() < 1e-12, "high-pass Nyquist gain {hnyq}");

        // The reported coefficients drive an independent filter
        // implementation to the same output.
        let x: Vec<f64> = (0..500).map(|i| ((i * 7919) % 100) as f64 / 50.0 - 1.0).collect();
        let mut run = lp;
        let direct = run.process_block(&x);
        let via_coeffs = iir_apply(&b, &a, &x);
        for (u, v) in direct.iter().zip(&via_coeffs) {
            assert!((u - v).abs() < 1e-12, "{u} vs {v}");
        }
        // Round-tripping through from_coeffs rebuilds the same section.
        let rebuilt = Biquad::from_coeffs(b[0], b[1], b[2], a[1], a[2]);
        assert_eq!(rebuilt.coeffs(), (b, a));
        for &f in &[10.0, 1000.0, 20000.0] {
            let (d, r) = (lp.freq_response(f, fs), rebuilt.freq_response(f, fs));
            assert!((d.re - r.re).abs() < 1e-15 && (d.im - r.im).abs() < 1e-15);
        }
    }

    #[test]
    fn test_sos_poles_and_zeros_rebuild_the_response() {
        let fs = 48000.0;
        let sos = butterworth(4, IirKind::Lowpass(1000.0), fs);
        let poles = sos.poles();
        let zeros = sos.zeros();
        assert_eq!(poles.len(), 4, "one pole pair per section");
        assert_eq!(zeros.len(), 4);
        // Stability: every pole strictly inside the unit circle.
        for p in &poles {
            assert!(p.norm() < 1.0, "pole outside the unit circle: {p:?}");
        }
        // A bilinear low-pass puts all of its zeros at z = −1 (Nyquist).
        for z in &zeros {
            assert!((z.re + 1.0).abs() < 1e-9 && z.im.abs() < 1e-9, "zero {z:?}");
        }
        // Poles come in conjugate pairs.
        for p in &poles {
            assert!(
                poles.iter().any(|o| (o.re - p.re).abs() < 1e-12 && (o.im + p.im).abs() < 1e-12),
                "unpaired pole {p:?}"
            );
        }
        // Factored form H(z) = k·Π(1 − zᵢz⁻¹)/Π(1 − pᵢz⁻¹) must reproduce
        // the cascade's own frequency response.
        assert!(sos.sections.iter().all(|s| (s.b0 - 1.0).abs() < 1e-12));
        for &f in &[1.0, 250.0, 1000.0, 5000.0, 23000.0] {
            let zinv = cis(-TWO_PI * f / fs);
            let num = zeros.iter().fold(C_ONE, |acc, z| acc * (C_ONE - *z * zinv));
            let den = poles.iter().fold(C_ONE, |acc, p| acc * (C_ONE - *p * zinv));
            let h = Complex::new(sos.gain, 0.0) * num / den;
            let want = sos.freq_response(f, fs);
            assert!(
                (h.re - want.re).abs() < 1e-9 && (h.im - want.im).abs() < 1e-9,
                "at {f} Hz: {h:?} vs {want:?}"
            );
        }
        // Vieta ties the reported roots back to the stored coefficients:
        // per section, Σp = −a1 and Πp = a2 (and likewise b1/b2 for zeros
        // once b0 = 1).
        for s in &sos.sections {
            let sp = quad_roots(1.0, s.a1, s.a2);
            let (sum, prod) = (sp[0] + sp[1], sp[0] * sp[1]);
            assert!((sum.re + s.a1).abs() < 1e-12 && sum.im.abs() < 1e-15);
            assert!((prod.re - s.a2).abs() < 1e-12 && prod.im.abs() < 1e-15);
            let sz = quad_roots(s.b0, s.b1, s.b2);
            let zsum = sz[0] + sz[1];
            assert!((zsum.re + s.b1 / s.b0).abs() < 1e-9);
        }

        // A high-pass (via lp_to_hp) instead puts all of its zeros at z = +1.
        let hp = butterworth(4, IirKind::Highpass(1000.0), fs);
        for z in hp.zeros() {
            assert!((z.re - 1.0).abs() < 1e-9 && z.im.abs() < 1e-9, "high-pass zero {z:?}");
        }
        assert!(hp.poles().iter().all(|p| p.norm() < 1.0));

        // A band-stop (via lp_to_bs) puts its zeros exactly on the unit
        // circle at the (bilinear-warped) geometric-mean center frequency.
        let (lo, hi) = (800.0, 1200.0);
        let bs = butterworth(2, IirKind::Bandstop(lo, hi), fs);
        let w0 = (prewarp(lo, fs) * prewarp(hi, fs)).sqrt();
        let f0 = fs / PI * (w0 / (2.0 * fs)).atan();
        let bz = bs.zeros();
        assert_eq!(bz.len(), 4);
        for z in &bz {
            assert!((z.norm() - 1.0).abs() < 1e-9, "notch zero off the circle: {}", z.norm());
            assert!(
                (z.arg().abs() - TWO_PI * f0 / fs).abs() < 1e-9,
                "notch at {} rad vs {}",
                z.arg().abs(),
                TWO_PI * f0 / fs
            );
        }
        // ...and the response really does vanish there.
        assert!(bs.freq_response(f0, fs).norm() < 1e-12, "band-stop does not notch");
    }

    #[test]
    fn test_bilinear_transform_prewarp_pins_the_cutoff() {
        let fs = 48000.0;
        let fc = 8000.0; // high enough that the bilinear warping matters
        let wa = TWO_PI * fc;
        // One-pole analog low-pass H(s) = ωa/(s + ωa): −3 dB at ωa.
        let poles = [Complex::new(-wa, 0.0)];

        // With prewarping the digital −3 dB point lands exactly on fc.
        let warped = bilinear_transform(&[], &poles, wa, fs, Some(fc));
        assert!((warped.freq_response(0.0, fs).norm() - 1.0).abs() < 1e-12, "DC gain");
        let g = warped.freq_response(fc, fs).norm();
        assert!(
            (g - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12,
            "prewarped cutoff gain {g}"
        );
        // Single real pole, single zero at Nyquist.
        assert_eq!(warped.poles().len(), 1);
        assert!(warped.poles()[0].norm() < 1.0);
        assert!((warped.zeros()[0].re + 1.0).abs() < 1e-12);

        // Without prewarping the cutoff is pulled down to the frequency
        // that maps to ωa: f' = (fs/π)·atan(π·fc/fs).
        let plain = bilinear_transform(&[], &poles, wa, fs, None);
        let f_warped = fs / PI * (PI * fc / fs).atan();
        assert!(f_warped < fc - 500.0, "expected a visible warp, got {f_warped}");
        let gp = plain.freq_response(f_warped, fs).norm();
        assert!(
            (gp - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12,
            "unwarped cutoff gain {gp} at {f_warped} Hz"
        );
        assert!((plain.freq_response(0.0, fs).norm() - 1.0).abs() < 1e-12);
        // The un-prewarped filter is already past −3 dB at fc.
        assert!(plain.freq_response(fc, fs).norm() < std::f64::consts::FRAC_1_SQRT_2);

        // Prewarping at the cutoff is what the RBJ cookbook does, so the
        // pinned one-pole agrees with a matched-cutoff design at fc.
        let rbj = Biquad::lowpass(fc, fs, 0.5);
        assert!(rbj.freq_response(fc, fs).norm() < 1.0);
    }

    #[test]
    fn test_a_weighting_reference_points() {
        let fs = 48000.0;
        let a = a_weighting_filter(fs);
        assert_stable(&a);
        let db = |f: f64| 20.0 * a.freq_response(f, fs).norm().log10();
        assert!(db(1000.0).abs() < 1e-9);
        assert!((db(100.0) - (-19.1)).abs() < 0.3, "100 Hz: {}", db(100.0));
        // Bilinear warping compresses the top octaves at 48 kHz; check
        // the 10 kHz point at a higher rate where warping is small.
        let a96 = a_weighting_filter(96000.0);
        let db96 = 20.0 * a96.freq_response(10000.0, 96000.0).norm().log10();
        assert!((db96 - (-2.5)).abs() < 0.4, "10 kHz: {db96}");
        let c = c_weighting_filter(fs);
        let cdb = |f: f64| 20.0 * c.freq_response(f, fs).norm().log10();
        assert!(cdb(1000.0).abs() < 1e-9);
        assert!((cdb(100.0) - (-0.3)).abs() < 0.3, "C 100 Hz: {}", cdb(100.0));
    }

    #[test]
    fn test_svf_and_dc_blocker() {
        let fs = 48000.0;
        let mut svf = state_variable_filter(1000.0, fs, 1.0);
        // Feed a low tone: LP output should track it, HP kill it.
        let f0 = 50.0;
        let x: Vec<f64> = (0..8000).map(|i| (TWO_PI * f0 * i as f64 / fs).sin()).collect();
        let mut lp_amp = 0.0_f64;
        let mut hp_amp = 0.0_f64;
        for (i, &v) in x.iter().enumerate() {
            let (lo, hi, _, _) = svf.process(v);
            if i > 4000 {
                lp_amp = lp_amp.max(lo.abs());
                hp_amp = hp_amp.max(hi.abs());
            }
        }
        assert!((lp_amp - 1.0).abs() < 0.02, "lp {lp_amp}");
        assert!(hp_amp < 0.01, "hp {hp_amp}");
        svf.reset();

        let mut dc = dc_blocker(0.995);
        let y: Vec<f64> = (0..4000).map(|_| dc.process(1.0)).collect();
        assert!(y[3999].abs() < 1e-3, "dc leak {}", y[3999]);
    }

    #[test]
    fn test_one_pole_and_rbj_q() {
        let (b0, a1) = one_pole_lowpass(1000.0, 48000.0);
        assert!((b0 + a1 - 1.0).abs() < 1e-12);
        assert!(a1 > 0.0 && a1 < 1.0);
        // One-octave bandwidth at low fc: Q ≈ 1/(2 sinh(ln2/2)) ≈ 1.414.
        let q = rbj_q_from_bandwidth(1.0, 100.0, 48000.0);
        assert!((q - 1.0 / (2.0 * (std::f64::consts::LN_2 / 2.0).sinh())).abs() < 1e-3);
    }
}
