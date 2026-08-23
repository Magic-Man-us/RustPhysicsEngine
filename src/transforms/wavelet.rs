//! Discrete and continuous wavelet transforms.
//!
//! Filter banks, boundary handling, and coefficient lengths follow the
//! PyWavelets conventions (dwt output length ⌊(n + L − 1)/2⌋, idwt
//! output length 2·len − L + 2), so round trips are exact for every
//! wavelet and padding mode. The CWT follows Torrence & Compo (1998).

use crate::fractals::Complex;
use crate::math::constants::PI;
use crate::special::gamma;
use crate::transforms::fft::{fft_any, ifft_any};
use crate::transforms::wavelet_tables;

/// Wavelet families: Haar, Daubechies 1–20, symlets 2–20, coiflets 1–5,
/// and the biorthogonal spline family (bior p.q as in PyWavelets).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wavelet {
    Haar,
    Db(u8),
    Sym(u8),
    Coif(u8),
    Bior(u8, u8),
}

/// Signal extension at the boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadMode {
    Zero,
    /// Half-sample symmetric: … x1 x0 | x0 x1 …
    Symmetric,
    Periodic,
    /// Whole-sample symmetric: … x2 x1 | x0 | x1 x2 …
    Reflect,
}

/// Decomposition and reconstruction filters (dec_lo, dec_hi, rec_lo,
/// rec_hi), quadrature-mirror related for the orthogonal families.
///
/// # Panics
/// Panics for an unsupported order (Db/Sym > 20, Coif > 5, or a bior
/// pair outside the standard set).
#[must_use]
pub fn wavelet_filters(w: Wavelet) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    match w {
        Wavelet::Haar => orthogonal_bank(wavelet_tables::db_table(1).unwrap()),
        Wavelet::Db(n) => orthogonal_bank(
            wavelet_tables::db_table(n as usize).unwrap_or_else(|| panic!("db{n} not available (1..=20)")),
        ),
        Wavelet::Sym(n) => orthogonal_bank(
            wavelet_tables::sym_table(n as usize)
                .unwrap_or_else(|| panic!("sym{n} not available (2..=20)")),
        ),
        Wavelet::Coif(n) => orthogonal_bank(
            wavelet_tables::coif_table(n as usize)
                .unwrap_or_else(|| panic!("coif{n} not available (1..=5)")),
        ),
        Wavelet::Bior(p, q) => {
            let (base, dual, m_max) = wavelet_tables::bior_tables(p as usize, q as usize)
                .unwrap_or_else(|| panic!("bior{p}.{q} is not in the standard family"));
            let len = if p == 1 { 2 * q as usize } else { 2 * q as usize + 2 };
            let off = m_max - q as usize;
            let mut dec_lo = vec![0.0; len];
            let mut dec_hi = vec![0.0; len];
            let mut rec_lo = vec![0.0; len];
            let mut rec_hi = vec![0.0; len];
            for i in 0..len {
                rec_lo[i] = base[i + off];
                dec_lo[i] = dual[len - 1 - i];
                rec_hi[i] = sign(i) * dual[len - 1 - i];
                dec_hi[i] = sign(len - 1 - i) * base[i + off];
            }
            (dec_lo, dec_hi, rec_lo, rec_hi)
        }
    }
}

fn sign(i: usize) -> f64 {
    if i.is_multiple_of(2) { 1.0 } else { -1.0 }
}

/// PyWavelets quadrature-mirror bank from a rec_lo table.
fn orthogonal_bank(t: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let l = t.len();
    let rec_lo = t.to_vec();
    let dec_lo: Vec<f64> = (0..l).map(|i| t[l - 1 - i]).collect();
    let rec_hi: Vec<f64> = (0..l).map(|i| sign(i) * t[l - 1 - i]).collect();
    let dec_hi: Vec<f64> = (0..l).map(|i| sign(l - 1 - i) * t[i]).collect();
    (dec_lo, dec_hi, rec_lo, rec_hi)
}

/// Boundary-extended sample x[i] for a conceptual index that may lie
/// outside 0..n.
fn sample(x: &[f64], i: isize, mode: PadMode) -> f64 {
    let n = x.len() as isize;
    if n == 0 {
        return 0.0;
    }
    if (0..n).contains(&i) {
        return x[i as usize];
    }
    match mode {
        PadMode::Zero => 0.0,
        PadMode::Periodic => {
            let m = i.rem_euclid(n);
            x[m as usize]
        }
        PadMode::Symmetric => {
            let period = 2 * n;
            let m = i.rem_euclid(period);
            if m < n {
                x[m as usize]
            } else {
                x[(period - 1 - m) as usize]
            }
        }
        PadMode::Reflect => {
            if n == 1 {
                return x[0];
            }
            let period = 2 * n - 2;
            let m = i.rem_euclid(period);
            if m < n {
                x[m as usize]
            } else {
                x[(period - m) as usize]
            }
        }
    }
}

fn dwt_channel(x: &[f64], filt: &[f64], mode: PadMode) -> Vec<f64> {
    let n = x.len();
    let flen = filt.len();
    let out_len = (n + flen - 1) / 2;
    (0..out_len)
        .map(|i| {
            let mut acc = 0.0;
            for (j, &f) in filt.iter().enumerate() {
                acc += f * sample(x, (2 * i + 1) as isize - j as isize, mode);
            }
            acc
        })
        .collect()
}

/// One-level DWT: (approximation, detail), each of length ⌊(n+L−1)/2⌋.
#[must_use]
pub fn dwt(x: &[f64], w: Wavelet, mode: PadMode) -> (Vec<f64>, Vec<f64>) {
    let (dec_lo, dec_hi, _, _) = wavelet_filters(w);
    (dwt_channel(x, &dec_lo, mode), dwt_channel(x, &dec_hi, mode))
}

/// One-level inverse DWT; output length 2·len − L + 2. `mode` is
/// accepted for API symmetry (reconstruction itself needs no padding).
///
/// # Panics
/// Panics if the approximation and detail lengths differ.
#[must_use]
pub fn idwt(a: &[f64], d: &[f64], w: Wavelet, mode: PadMode) -> Vec<f64> {
    let _ = mode;
    assert_eq!(a.len(), d.len(), "idwt needs equal-length coefficient arrays");
    let (_, _, rec_lo, rec_hi) = wavelet_filters(w);
    let flen = rec_lo.len();
    let l = a.len();
    if l == 0 {
        return Vec::new();
    }
    let out_len = (2 * l).saturating_sub(flen) + 2;
    let mut out = vec![0.0; out_len];
    for (k, slot) in out.iter_mut().enumerate() {
        let mut acc = 0.0;
        for (i, (&av, &dv)) in a.iter().zip(d).enumerate() {
            let j = k as isize - 2 * i as isize + flen as isize - 2;
            if (0..flen as isize).contains(&j) {
                acc += av * rec_lo[j as usize] + dv * rec_hi[j as usize];
            }
        }
        *slot = acc;
    }
    out
}

/// Multilevel decomposition: returns \[a_L, d_L, d_{L−1}, …, d_1\].
#[must_use]
pub fn wavedec(x: &[f64], w: Wavelet, levels: usize, mode: PadMode) -> Vec<Vec<f64>> {
    let mut details: Vec<Vec<f64>> = Vec::new();
    let mut a = x.to_vec();
    for _ in 0..levels {
        if a.is_empty() {
            break;
        }
        let (na, d) = dwt(&a, w, mode);
        details.push(d);
        a = na;
    }
    let mut out = vec![a];
    out.extend(details.into_iter().rev());
    out
}

/// Multilevel reconstruction (inverse of [`wavedec`]).
#[must_use]
pub fn waverec(coeffs: &[Vec<f64>], w: Wavelet, mode: PadMode) -> Vec<f64> {
    if coeffs.is_empty() {
        return Vec::new();
    }
    let mut a = coeffs[0].clone();
    for d in &coeffs[1..] {
        if a.len() == d.len() + 1 {
            a.pop(); // odd-length parent left one extra sample
        }
        a = idwt(&a, d, w, mode);
    }
    a
}

/// One-level separable 2D DWT with symmetric extension (rows along x
/// first, then columns): returns (LL, LH, HL, HH) where the first
/// letter is the x (row-direction) channel. Sub-band dims are
/// ⌊(w+L−1)/2⌋ × ⌊(h+L−1)/2⌋.
///
/// # Panics
/// Panics unless `img.len() == w * h`.
#[must_use]
pub fn dwt_2d(
    img: &[f64],
    w: usize,
    h: usize,
    wavelet: Wavelet,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    assert_eq!(img.len(), w * h, "dwt_2d expects w*h samples");
    let mode = PadMode::Symmetric;
    let (dec_lo, dec_hi, _, _) = wavelet_filters(wavelet);
    let flen = dec_lo.len();
    let cw = (w + flen - 1) / 2;
    let ch = (h + flen - 1) / 2;
    // Rows.
    let mut low = vec![0.0; cw * h];
    let mut high = vec![0.0; cw * h];
    for y in 0..h {
        let row = &img[y * w..(y + 1) * w];
        let l = dwt_channel(row, &dec_lo, mode);
        let hi = dwt_channel(row, &dec_hi, mode);
        low[y * cw..(y + 1) * cw].copy_from_slice(&l);
        high[y * cw..(y + 1) * cw].copy_from_slice(&hi);
    }
    // Columns.
    let mut ll = vec![0.0; cw * ch];
    let mut lh = vec![0.0; cw * ch];
    let mut hl = vec![0.0; cw * ch];
    let mut hh = vec![0.0; cw * ch];
    let mut col = vec![0.0; h];
    for x in 0..cw {
        for (y, v) in col.iter_mut().enumerate() {
            *v = low[y * cw + x];
        }
        let l = dwt_channel(&col, &dec_lo, mode);
        let hi = dwt_channel(&col, &dec_hi, mode);
        for y in 0..ch {
            ll[y * cw + x] = l[y];
            lh[y * cw + x] = hi[y];
        }
        for (y, v) in col.iter_mut().enumerate() {
            *v = high[y * cw + x];
        }
        let l = dwt_channel(&col, &dec_lo, mode);
        let hi = dwt_channel(&col, &dec_hi, mode);
        for y in 0..ch {
            hl[y * cw + x] = l[y];
            hh[y * cw + x] = hi[y];
        }
    }
    (ll, lh, hl, hh)
}

/// Inverse of [`dwt_2d`]; `w` and `h` are the original image dimensions.
#[must_use]
pub fn idwt_2d(
    ll: &[f64],
    lh: &[f64],
    hl: &[f64],
    hh: &[f64],
    w: usize,
    h: usize,
    wavelet: Wavelet,
) -> Vec<f64> {
    let mode = PadMode::Symmetric;
    let (dec_lo, _, _, _) = wavelet_filters(wavelet);
    let flen = dec_lo.len();
    let cw = (w + flen - 1) / 2;
    let ch = (h + flen - 1) / 2;
    assert_eq!(ll.len(), cw * ch, "sub-band dims inconsistent with w/h");
    // Columns first (undo the second pass).
    let mut low = vec![0.0; cw * h];
    let mut high = vec![0.0; cw * h];
    let mut ca = vec![0.0; ch];
    let mut cd = vec![0.0; ch];
    for x in 0..cw {
        for y in 0..ch {
            ca[y] = ll[y * cw + x];
            cd[y] = lh[y * cw + x];
        }
        let mut rec = idwt(&ca, &cd, wavelet, mode);
        rec.truncate(h);
        for (y, v) in rec.iter().enumerate() {
            low[y * cw + x] = *v;
        }
        for y in 0..ch {
            ca[y] = hl[y * cw + x];
            cd[y] = hh[y * cw + x];
        }
        let mut rec = idwt(&ca, &cd, wavelet, mode);
        rec.truncate(h);
        for (y, v) in rec.iter().enumerate() {
            high[y * cw + x] = *v;
        }
    }
    // Rows.
    let mut out = vec![0.0; w * h];
    for y in 0..h {
        let mut rec = idwt(&low[y * cw..(y + 1) * cw], &high[y * cw..(y + 1) * cw], wavelet, mode);
        rec.truncate(w);
        out[y * w..y * w + rec.len()].copy_from_slice(&rec);
    }
    out
}

/// Detail-coefficient thresholding rules for [`wavelet_denoise`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Threshold {
    Hard(f64),
    Soft(f64),
    /// Universal threshold σ√(2 ln n) with σ from the finest detail MAD.
    VisuShrink,
    /// Per-subband adaptive Bayes threshold σ²/σ_x.
    BayesShrink,
}

fn apply_threshold(d: &mut [f64], t: f64, soft: bool) {
    for v in d.iter_mut() {
        if v.abs() <= t {
            *v = 0.0;
        } else if soft {
            *v = v.signum() * (v.abs() - t);
        }
    }
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = v.len() / 2;
    if v.len().is_multiple_of(2) {
        0.5 * (v[mid - 1] + v[mid])
    } else {
        v[mid]
    }
}

/// Wavelet shrinkage denoising: decompose, threshold the detail bands,
/// reconstruct (trimmed to the input length).
#[must_use]
pub fn wavelet_denoise(x: &[f64], w: Wavelet, levels: usize, t: Threshold) -> Vec<f64> {
    let mode = PadMode::Symmetric;
    let mut coeffs = wavedec(x, w, levels, mode);
    if coeffs.len() < 2 {
        return x.to_vec();
    }
    // Noise σ from the finest-scale details (last entry).
    let sigma = {
        let mut mags: Vec<f64> = coeffs.last().unwrap().iter().map(|v| v.abs()).collect();
        median(&mut mags) / 0.6745
    };
    let n = x.len() as f64;
    for d in coeffs.iter_mut().skip(1) {
        match t {
            Threshold::Hard(th) => apply_threshold(d, th, false),
            Threshold::Soft(th) => apply_threshold(d, th, true),
            Threshold::VisuShrink => {
                apply_threshold(d, sigma * (2.0 * n.ln()).sqrt(), true);
            }
            Threshold::BayesShrink => {
                let var_y: f64 = d.iter().map(|v| v * v).sum::<f64>() / d.len().max(1) as f64;
                let sig_x = (var_y - sigma * sigma).max(0.0).sqrt();
                let th = if sig_x > 1e-12 { sigma * sigma / sig_x } else { f64::INFINITY };
                apply_threshold(d, th, true);
            }
        }
    }
    let mut out = waverec(&coeffs, w, mode);
    out.truncate(x.len());
    out
}

/// Keep the largest `keep_fraction` of all coefficients (approximation
/// always kept), zero the rest, and reconstruct.
#[must_use]
pub fn wavelet_compress(x: &[f64], w: Wavelet, levels: usize, keep_fraction: f64) -> Vec<f64> {
    let mode = PadMode::Symmetric;
    let mut coeffs = wavedec(x, w, levels, mode);
    let total: usize = coeffs.iter().map(Vec::len).sum();
    let approx_len = coeffs[0].len();
    let keep_total = ((keep_fraction.clamp(0.0, 1.0) * total as f64).round() as usize).max(approx_len);
    let keep_detail = keep_total - approx_len;
    let mut mags: Vec<f64> = coeffs[1..]
        .iter()
        .flat_map(|d| d.iter().map(|v| v.abs()))
        .collect();
    mags.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let cutoff = if keep_detail == 0 {
        f64::INFINITY
    } else if keep_detail >= mags.len() {
        0.0
    } else {
        mags[keep_detail - 1]
    };
    for d in coeffs.iter_mut().skip(1) {
        for v in d.iter_mut() {
            if v.abs() < cutoff {
                *v = 0.0;
            }
        }
    }
    let mut out = waverec(&coeffs, w, mode);
    out.truncate(x.len());
    out
}

/// Mother wavelets for the CWT (Torrence & Compo definitions):
/// Morlet(ω₀), Mexican hat (DOG order 2), Paul(m), DOG(m).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mother {
    Morlet(f64),
    MexicanHat,
    Paul(u8),
    Dog(u8),
}

/// Fourier-domain mother wavelet ψ̂(sω) (T&C table 1).
fn mother_hat(m: Mother, s_omega: f64) -> Complex {
    match m {
        Mother::Morlet(w0) => {
            if s_omega > 0.0 {
                let v = PI.powf(-0.25) * (-(s_omega - w0).powi(2) / 2.0).exp();
                Complex::new(v, 0.0)
            } else {
                Complex::new(0.0, 0.0)
            }
        }
        Mother::MexicanHat => mother_hat(Mother::Dog(2), s_omega),
        Mother::Paul(m) => {
            if s_omega > 0.0 {
                let m = m as f64;
                let norm = 2.0_f64.powf(m) / (m * gamma(2.0 * m)).sqrt();
                Complex::new(norm * s_omega.powf(m) * (-s_omega).exp(), 0.0)
            } else {
                Complex::new(0.0, 0.0)
            }
        }
        Mother::Dog(m) => {
            let mf = m as f64;
            let norm = -1.0 / gamma(mf + 0.5).sqrt();
            let mag = norm * s_omega.powi(m as i32) * (-s_omega * s_omega / 2.0).exp();
            // −(i)^m factor
            match m % 4 {
                0 => Complex::new(mag, 0.0),
                1 => Complex::new(0.0, mag),
                2 => Complex::new(-mag, 0.0),
                _ => Complex::new(0.0, -mag),
            }
        }
    }
}

/// Continuous wavelet transform. `scales` are in samples; row s of the
/// output holds W(s, t) at every sample. Computed in the Fourier domain
/// (Torrence & Compo eq. 4) with unit-energy normalization √(2πs).
#[must_use]
pub fn cwt(x: &[f64], scales: &[f64], mother: Mother, fs: f64) -> Vec<Vec<Complex>> {
    let _ = fs; // scales are in samples; fs only matters for frequency mapping
    let n = x.len();
    if n == 0 {
        return scales.iter().map(|_| Vec::new()).collect();
    }
    let spec = fft_any(&x.iter().map(|&v| Complex::new(v, 0.0)).collect::<Vec<_>>());
    let omegas: Vec<f64> = (0..n)
        .map(|k| {
            if k <= n / 2 {
                2.0 * PI * k as f64 / n as f64
            } else {
                -2.0 * PI * (n - k) as f64 / n as f64
            }
        })
        .collect();
    scales
        .iter()
        .map(|&s| {
            let norm = (2.0 * PI * s).sqrt();
            let prod: Vec<Complex> = spec
                .iter()
                .zip(&omegas)
                .map(|(v, &w)| {
                    let h = mother_hat(mother, s * w).conjugate();
                    *v * h * Complex::new(norm, 0.0)
                })
                .collect();
            ifft_any(&prod)
        })
        .collect()
}

/// |CWT|² per scale and sample.
#[must_use]
pub fn scalogram(x: &[f64], scales: &[f64], mother: Mother, fs: f64) -> Vec<Vec<f64>> {
    cwt(x, scales, mother, fs)
        .into_iter()
        .map(|row| row.into_iter().map(|c| c.norm_sq()).collect())
        .collect()
}

/// Equivalent Fourier frequency (Hz) of a CWT scale in samples
/// (Torrence & Compo table 1).
#[must_use]
pub fn scale_to_frequency(scale: f64, mother: Mother, fs: f64) -> f64 {
    let lambda = match mother {
        Mother::Morlet(w0) => 4.0 * PI * scale / (w0 + (2.0 + w0 * w0).sqrt()),
        Mother::MexicanHat => 2.0 * PI * scale / (2.5_f64).sqrt(),
        Mother::Paul(m) => 4.0 * PI * scale / (2.0 * m as f64 + 1.0),
        Mother::Dog(m) => 2.0 * PI * scale / (m as f64 + 0.5).sqrt(),
    };
    fs / lambda
}

/// Full wavelet-packet tree at the given depth: 2^levels leaves in
/// natural (frequency-ordered-by-index) order, symmetric extension.
#[must_use]
pub fn wavelet_packet_decompose(x: &[f64], w: Wavelet, levels: usize) -> Vec<Vec<f64>> {
    let mut nodes = vec![x.to_vec()];
    for _ in 0..levels {
        let mut next = Vec::with_capacity(nodes.len() * 2);
        for node in &nodes {
            let (a, d) = dwt(node, w, PadMode::Symmetric);
            next.push(a);
            next.push(d);
        }
        nodes = next;
    }
    nodes
}

/// Lossless integer 5/3 (LeGall) lifting DWT, in place: the first half
/// becomes the approximation, the second half the detail.
///
/// # Panics
/// Panics unless the length is even and ≥ 2.
pub fn lifting_dwt_53(x: &mut [i32]) {
    let n = x.len();
    assert!(n >= 2 && n.is_multiple_of(2), "lifting_dwt_53 needs an even length >= 2");
    let half = n / 2;
    let mut a = vec![0i32; half];
    let mut d = vec![0i32; half];
    // Predict: d[i] = odd − ⌊(left even + right even)/2⌋ (mirror at end).
    for i in 0..half {
        let left = x[2 * i];
        let right = if 2 * i + 2 < n { x[2 * i + 2] } else { x[2 * i] };
        d[i] = x[2 * i + 1] - ((left + right) >> 1);
    }
    // Update: a[i] = even + ⌊(d[i−1] + d[i] + 2)/4⌋ (mirror at start).
    for i in 0..half {
        let dl = if i > 0 { d[i - 1] } else { d[0] };
        a[i] = x[2 * i] + ((dl + d[i] + 2) >> 2);
    }
    x[..half].copy_from_slice(&a);
    x[half..].copy_from_slice(&d);
}

/// Exact inverse of [`lifting_dwt_53`].
///
/// # Panics
/// Panics unless the length is even and ≥ 2.
pub fn lifting_idwt_53(x: &mut [i32]) {
    let n = x.len();
    assert!(n >= 2 && n.is_multiple_of(2), "lifting_idwt_53 needs an even length >= 2");
    let half = n / 2;
    let a = x[..half].to_vec();
    let d = x[half..].to_vec();
    let mut even = vec![0i32; half];
    for i in 0..half {
        let dl = if i > 0 { d[i - 1] } else { d[0] };
        even[i] = a[i] - ((dl + d[i] + 2) >> 2);
    }
    for i in 0..half {
        let left = even[i];
        let right = if i + 1 < half { even[i + 1] } else { even[i] };
        x[2 * i] = even[i];
        x[2 * i + 1] = d[i] + ((left + right) >> 1);
    }
}

/// Multiresolution analysis: the input split into levels+1 additive
/// components (details from coarse to fine, then the approximation
/// first). Component 0 is the level-L approximation signal; component k
/// (k ≥ 1) is the detail at level L+1−k. The components sum to x.
#[must_use]
pub fn multiresolution_analysis(x: &[f64], w: Wavelet, levels: usize) -> Vec<Vec<f64>> {
    let mode = PadMode::Symmetric;
    let coeffs = wavedec(x, w, levels, mode);
    (0..coeffs.len())
        .map(|keep| {
            let isolated: Vec<Vec<f64>> = coeffs
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    if i == keep {
                        c.clone()
                    } else {
                        vec![0.0; c.len()]
                    }
                })
                .collect();
            let mut rec = waverec(&isolated, w, mode);
            rec.truncate(x.len());
            rec
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn sample_signal(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| (i as f64 * 0.23).sin() + 0.5 * (i as f64 * 1.7).cos() + 0.01 * i as f64)
            .collect()
    }

    const ALL_WAVELETS: [Wavelet; 10] = [
        Wavelet::Haar,
        Wavelet::Db(2),
        Wavelet::Db(4),
        Wavelet::Db(10),
        Wavelet::Sym(4),
        Wavelet::Sym(8),
        Wavelet::Coif(1),
        Wavelet::Coif(3),
        Wavelet::Bior(2, 2),
        Wavelet::Bior(3, 5),
    ];

    const ALL_MODES: [PadMode; 4] =
        [PadMode::Zero, PadMode::Symmetric, PadMode::Periodic, PadMode::Reflect];

    #[test]
    fn test_dwt_idwt_roundtrip_every_wavelet_and_mode() {
        let x = sample_signal(64);
        for &w in &ALL_WAVELETS {
            for &mode in &ALL_MODES {
                let (a, d) = dwt(&x, w, mode);
                let mut rec = idwt(&a, &d, w, mode);
                rec.truncate(64);
                for (i, (u, v)) in x.iter().zip(&rec).enumerate() {
                    assert!(approx(*u, *v, 1e-10), "{w:?} {mode:?} sample {i}: {u} vs {v}");
                }
            }
        }
    }

    #[test]
    fn test_wavedec_waverec_roundtrip() {
        for &n in &[61usize, 64] {
            let x = sample_signal(n);
            for &w in &ALL_WAVELETS {
                for &mode in &ALL_MODES {
                    let coeffs = wavedec(&x, w, 3, mode);
                    let mut rec = waverec(&coeffs, w, mode);
                    rec.truncate(n);
                    for (u, v) in x.iter().zip(&rec) {
                        assert!(approx(*u, *v, 1e-9), "{w:?} {mode:?} n={n}: {u} vs {v}");
                    }
                }
            }
        }
    }

    #[test]
    fn test_haar_detail_of_constant_is_zero() {
        let x = vec![3.5; 32];
        let (a, d) = dwt(&x, Wavelet::Haar, PadMode::Symmetric);
        for v in &d {
            assert!(v.abs() < 1e-12);
        }
        // Approximation carries √2 times the constant.
        for v in &a {
            assert!(approx(*v, 3.5 * std::f64::consts::SQRT_2, 1e-12));
        }
    }

    #[test]
    fn test_daubechies_orthogonality_and_moments() {
        for &nv in &[2usize, 4, 6, 10] {
            let (dec_lo, dec_hi, _, _) = wavelet_filters(Wavelet::Db(nv as u8));
            let l = dec_lo.len();
            // Σ h[k] h[k+2m] = δ_m
            for m in 0..l / 2 {
                let mut acc = 0.0;
                for k in 0..l - 2 * m {
                    acc += dec_lo[k] * dec_lo[k + 2 * m];
                }
                let expect = if m == 0 { 1.0 } else { 0.0 };
                assert!(approx(acc, expect, 1e-10), "db{nv} shift {m}: {acc}");
            }
            // Vanishing moments: Σ k^p g[k] = 0 for p < nv.
            for p in 0..nv {
                let acc: f64 = dec_hi
                    .iter()
                    .enumerate()
                    .map(|(k, &g)| (k as f64).powi(p as i32) * g)
                    .sum();
                assert!(acc.abs() < 1e-7, "db{nv} moment {p}: {acc}");
            }
            // Unit sum of scaling filter (·√2).
            let s: f64 = dec_lo.iter().sum();
            assert!(approx(s, std::f64::consts::SQRT_2, 1e-10));
        }
    }

    #[test]
    fn test_parseval_orthogonal() {
        // With periodic extension the coefficient sequence repeats with
        // period n/2, so the first n/2 approximation and detail
        // coefficients form the orthonormal periodized DWT: Parseval
        // holds exactly there for orthogonal wavelets.
        let x = sample_signal(128);
        let energy_x: f64 = x.iter().map(|v| v * v).sum();
        for &w in &[Wavelet::Db(4), Wavelet::Sym(6), Wavelet::Coif(2), Wavelet::Haar] {
            let (a, d) = dwt(&x, w, PadMode::Periodic);
            let energy_c: f64 = a[..64].iter().chain(&d[..64]).map(|v| v * v).sum();
            assert!(
                approx(energy_c, energy_x, 1e-8 * energy_x),
                "{w:?}: {energy_c} vs {energy_x}"
            );
        }
    }

    #[test]
    fn test_dwt_2d_roundtrip() {
        let (w, h) = (16, 12);
        let img: Vec<f64> = (0..w * h).map(|i| ((i * 37) % 11) as f64 - 5.0).collect();
        for &wav in &[Wavelet::Haar, Wavelet::Db(2), Wavelet::Bior(2, 2)] {
            let (ll, lh, hl, hh) = dwt_2d(&img, w, h, wav);
            let rec = idwt_2d(&ll, &lh, &hl, &hh, w, h, wav);
            for (a, b) in img.iter().zip(&rec) {
                assert!(approx(*a, *b, 1e-9), "{wav:?}");
            }
        }
    }

    #[test]
    fn test_denoise_reduces_noise() {
        let n = 256;
        let clean: Vec<f64> = (0..n).map(|i| (2.0 * PI * 4.0 * i as f64 / n as f64).sin()).collect();
        // Deterministic pseudo-noise.
        let noisy: Vec<f64> = clean
            .iter()
            .enumerate()
            .map(|(i, &v)| v + 0.2 * (((i * 2654435761) % 1000) as f64 / 500.0 - 1.0))
            .collect();
        let den = wavelet_denoise(&noisy, Wavelet::Db(4), 4, Threshold::VisuShrink);
        let err_noisy: f64 = clean.iter().zip(&noisy).map(|(a, b)| (a - b).powi(2)).sum();
        let err_den: f64 = clean.iter().zip(&den).map(|(a, b)| (a - b).powi(2)).sum();
        assert!(err_den < 0.5 * err_noisy, "denoise: {err_den} vs {err_noisy}");
    }

    #[test]
    fn test_compress_full_fraction_lossless() {
        let x = sample_signal(100);
        let y = wavelet_compress(&x, Wavelet::Db(3), 3, 1.0);
        for (a, b) in x.iter().zip(&y) {
            assert!(approx(*a, *b, 1e-9));
        }
        // Heavy compression still tracks a smooth signal roughly.
        let smooth: Vec<f64> = (0..128).map(|i| (PI * i as f64 / 64.0).sin()).collect();
        let z = wavelet_compress(&smooth, Wavelet::Db(4), 4, 0.15);
        let mse: f64 =
            smooth.iter().zip(&z).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / 128.0;
        assert!(mse < 1e-2, "mse {mse}");
    }

    #[test]
    fn test_cwt_morlet_scale_localization() {
        let fs = 100.0;
        let f0 = 5.0;
        let n = 512;
        let x: Vec<f64> = (0..n).map(|i| (2.0 * PI * f0 * i as f64 / fs).sin()).collect();
        // Scales spanning 1–20 Hz for Morlet ω0 = 6.
        let scales: Vec<f64> = (1..=40).map(|k| k as f64).collect();
        let sc = scalogram(&x, &scales, Mother::Morlet(6.0), fs);
        // Find the scale with maximum mean power (interior samples).
        let mut best = (0usize, 0.0);
        for (si, row) in sc.iter().enumerate() {
            let mean: f64 = row[n / 4..3 * n / 4].iter().sum::<f64>() / (n / 2) as f64;
            if mean > best.1 {
                best = (si, mean);
            }
        }
        let f_est = scale_to_frequency(scales[best.0], Mother::Morlet(6.0), fs);
        assert!((f_est - f0).abs() < 1.0, "peak scale maps to {f_est} Hz");
    }

    #[test]
    fn test_wavelet_packets_count_and_energy() {
        let x = sample_signal(64);
        let leaves = wavelet_packet_decompose(&x, Wavelet::Haar, 3);
        assert_eq!(leaves.len(), 8);
        // Haar packets on a power-of-two signal conserve energy
        // approximately (boundary handling is symmetric).
        let ex: f64 = x.iter().map(|v| v * v).sum();
        let ec: f64 = leaves.iter().flat_map(|l| l.iter().map(|v| v * v)).sum();
        assert!((ec - ex).abs() / ex < 0.05, "{ec} vs {ex}");
    }

    #[test]
    fn test_lifting_53_lossless() {
        let orig: Vec<i32> = (0..64).map(|i| ((i * 89) % 251) - 125).collect();
        let mut x = orig.clone();
        lifting_dwt_53(&mut x);
        assert_ne!(x, orig);
        lifting_idwt_53(&mut x);
        assert_eq!(x, orig);
    }

    #[test]
    fn test_mra_components_sum_to_signal() {
        let x = sample_signal(90);
        for &w in &[Wavelet::Db(3), Wavelet::Bior(2, 4)] {
            let comps = multiresolution_analysis(&x, w, 3);
            assert_eq!(comps.len(), 4);
            for i in 0..x.len() {
                let sum: f64 = comps.iter().map(|c| c[i]).sum();
                assert!(approx(sum, x[i], 1e-9), "{w:?} at {i}");
            }
        }
    }

    #[test]
    fn test_bior_filters_biorthogonality() {
        // Cross-orthogonality: Σ dec_lo[k]·rec_lo[k+2m] = δ_m (up to
        // alignment) is what perfect reconstruction enforces; PR is
        // covered above, here just sanity-check filter sums.
        for &(p, q) in &[(1u8, 3u8), (2, 2), (2, 6), (3, 5), (4, 4), (5, 5), (6, 8)] {
            let (dec_lo, dec_hi, rec_lo, rec_hi) = wavelet_filters(Wavelet::Bior(p, q));
            let s_dec: f64 = dec_lo.iter().sum();
            let s_rec: f64 = rec_lo.iter().sum();
            assert!(approx(s_dec, std::f64::consts::SQRT_2, 1e-9), "bior{p}.{q} dec sum {s_dec}");
            assert!(approx(s_rec, std::f64::consts::SQRT_2, 1e-9), "bior{p}.{q} rec sum {s_rec}");
            let s_hi: f64 = dec_hi.iter().sum();
            let s_rhi: f64 = rec_hi.iter().sum();
            assert!(s_hi.abs() < 1e-9 && s_rhi.abs() < 1e-9);
        }
    }
}
