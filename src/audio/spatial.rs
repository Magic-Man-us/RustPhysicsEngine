//! Spatial audio: panning laws, VBAP, ambisonics, simple binaural cues,
//! Doppler, distance/air attenuation, geometric room acoustics (image
//! source and ray tracing), microphone arrays (beamforming, TDOA
//! localization), sonar, and loudspeaker system responses.

use crate::dsp::iir::{butterworth, Biquad, IirKind, Sos};
use crate::fractals::Complex;
use crate::geometry::mesh::Mesh;
use crate::math::Vec3;
use crate::monte_carlo::Rng;
use crate::special::legendre_p_assoc;

const TWO_PI: f64 = 2.0 * crate::math::constants::PI;
const PI: f64 = crate::math::constants::PI;

// --- Panning -------------------------------------------------------------

/// Linear pan law; `pos` in [-1 (left), 1 (right)].
#[must_use]
pub fn pan_linear(x: f64, pos: f64) -> (f64, f64) {
    let p = pos.clamp(-1.0, 1.0);
    (x * 0.5 * (1.0 - p), x * 0.5 * (1.0 + p))
}

/// Constant-power (-3 dB center) pan law.
#[must_use]
pub fn pan_constant_power(x: f64, pos: f64) -> (f64, f64) {
    let theta = (pos.clamp(-1.0, 1.0) + 1.0) * PI / 4.0;
    (x * theta.cos(), x * theta.sin())
}

/// -4.5 dB-center compromise pan law (geometric mean of the linear and
/// constant-power laws).
#[must_use]
pub fn pan_minus_4_5_db(x: f64, pos: f64) -> (f64, f64) {
    let p = pos.clamp(-1.0, 1.0);
    let theta = (p + 1.0) * PI / 4.0;
    let l = (0.5 * (1.0 - p) * theta.cos()).max(0.0).sqrt();
    let r = (0.5 * (1.0 + p) * theta.sin()).max(0.0).sqrt();
    (x * l, x * r)
}

/// 2D VBAP: gains for `speaker_angles` (radians, unsorted) reproducing a
/// source at `angle`; only the flanking pair is nonzero.
#[must_use]
pub fn pan_vbap_2d(angle: f64, speaker_angles: &[f64]) -> Vec<f64> {
    let n = speaker_angles.len();
    let mut gains = vec![0.0; n];
    if n == 0 {
        return gains;
    }
    if n == 1 {
        gains[0] = 1.0;
        return gains;
    }
    // Sort speakers by angle and find the pair whose arc contains angle.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| speaker_angles[a].partial_cmp(&speaker_angles[b]).unwrap());
    let wrap = |a: f64| a.rem_euclid(TWO_PI);
    let src = wrap(angle);
    for k in 0..n {
        let i = order[k];
        let j = order[(k + 1) % n];
        let a1 = wrap(speaker_angles[i]);
        let mut a2 = wrap(speaker_angles[j]);
        let mut s = src;
        if a2 <= a1 {
            a2 += TWO_PI;
        }
        if s < a1 {
            s += TWO_PI;
        }
        if s >= a1 && s <= a2 {
            // Solve [cos1 cos2; sin1 sin2] g = [cos s; sin s].
            let (c1, s1) = (a1.cos(), a1.sin());
            let (c2, s2) = (a2.cos(), a2.sin());
            let det = c1 * s2 - c2 * s1;
            if det.abs() < 1e-12 {
                gains[i] = 1.0;
                return gains;
            }
            let g1 = (src.cos() * s2 - c2 * src.sin()) / det;
            let g2 = (c1 * src.sin() - src.cos() * s1) / det;
            let norm = (g1 * g1 + g2 * g2).sqrt().max(1e-12);
            gains[i] = g1.max(0.0) / norm;
            gains[j] = g2.max(0.0) / norm;
            return gains;
        }
    }
    gains
}

fn solve3(m: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv_det = 1.0 / det;
    let adj = |r: usize, c: usize| -> f64 {
        let (r1, r2) = match r {
            0 => (1, 2),
            1 => (0, 2),
            _ => (0, 1),
        };
        let (c1, c2) = match c {
            0 => (1, 2),
            1 => (0, 2),
            _ => (0, 1),
        };
        let minor = m[r1][c1] * m[r2][c2] - m[r1][c2] * m[r2][c1];
        if (r + c).is_multiple_of(2) { minor } else { -minor }
    };
    let mut out = [0.0; 3];
    for (i, o) in out.iter_mut().enumerate() {
        // x = A^{-1} b with A^{-1} = adj(A)^T / det.
        *o = (adj(0, i) * b[0] + adj(1, i) * b[1] + adj(2, i) * b[2]) * inv_det;
    }
    Some(out)
}

/// 3D VBAP over speaker triplets: picks the triplet giving all-positive
/// gains with the best conditioning, normalized to unit power.
#[must_use]
pub fn pan_vbap_3d(dir: Vec3, speakers: &[Vec3]) -> Vec<f64> {
    let n = speakers.len();
    let mut gains = vec![0.0; n];
    let d = dir.normalized();
    let mut best: Option<(f64, [usize; 3], [f64; 3])> = None;
    for i in 0..n {
        for j in i + 1..n {
            for k in j + 1..n {
                let (a, b, c) =
                    (speakers[i].normalized(), speakers[j].normalized(), speakers[k].normalized());
                let m = [[a.x, b.x, c.x], [a.y, b.y, c.y], [a.z, b.z, c.z]];
                if let Some(g) = solve3(m, [d.x, d.y, d.z]) {
                    let min_g = g[0].min(g[1]).min(g[2]);
                    if best.as_ref().is_none_or(|(bm, _, _)| min_g > *bm) {
                        best = Some((min_g, [i, j, k], g));
                    }
                }
            }
        }
    }
    if let Some((_, idx, g)) = best {
        let norm = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt().max(1e-12);
        for (t, &i) in idx.iter().enumerate() {
            gains[i] = g[t].max(0.0) / norm;
        }
    }
    gains
}

// --- Ambisonics ----------------------------------------------------------

/// First-order B-format (FuMa WXYZ) encoding of one sample.
#[must_use]
pub fn ambisonics_encode_1st(x: f64, azimuth: f64, elevation: f64) -> [f64; 4] {
    let ce = elevation.cos();
    [
        x * std::f64::consts::FRAC_1_SQRT_2,
        x * azimuth.cos() * ce,
        x * azimuth.sin() * ce,
        x * elevation.sin(),
    ]
}

/// Real spherical harmonic, ACN index (l, m), SN3D normalization,
/// evaluated at azimuth/elevation.
fn sh_sn3d(l: u32, m: i32, az: f64, el: f64) -> f64 {
    let am = m.unsigned_abs();
    let x = el.sin();
    // Cancel the Condon-Shortley phase if the implementation includes it.
    let p = legendre_p_assoc(l, am as i32, x) * if am % 2 == 1 { -1.0 } else { 1.0 };
    let mut norm = 1.0;
    for i in (l - am + 1)..=(l + am) {
        norm /= i as f64;
    }
    let n = ((if m == 0 { 1.0 } else { 2.0 }) * norm).sqrt();
    let angular = if m >= 0 { (m as f64 * az).cos() } else { (am as f64 * az).sin() };
    n * p * angular
}

/// Higher-order ambisonic encoding (ACN channel order, SN3D weights) of
/// one sample; (order+1)² channels.
#[must_use]
pub fn ambisonics_encode(x: f64, az: f64, el: f64, order: u32) -> Vec<f64> {
    let mut out = Vec::with_capacity(((order + 1) * (order + 1)) as usize);
    for l in 0..=order {
        for m in -(l as i32)..=(l as i32) {
            out.push(x * sh_sn3d(l, m, az, el));
        }
    }
    out
}

/// Basic projection decode of an ACN/SN3D signal set to speakers at
/// (azimuth, elevation) pairs.
#[must_use]
pub fn ambisonics_decode(b: &[f64], speakers: &[(f64, f64)], order: u32) -> Vec<f64> {
    let n_ch = ((order + 1) * (order + 1)) as usize;
    speakers
        .iter()
        .map(|&(az, el)| {
            let y = ambisonics_encode(1.0, az, el, order);
            // N3D weighting makes the projection decoder exact on a
            // uniform layout: multiply each component by (2l+1).
            let mut acc = 0.0;
            let mut ch = 0;
            for l in 0..=order {
                for _m in -(l as i32)..=(l as i32) {
                    if ch < b.len() && ch < n_ch {
                        acc += b[ch] * y[ch] * (2 * l + 1) as f64;
                    }
                    ch += 1;
                }
            }
            acc / speakers.len() as f64
        })
        .collect()
}

fn rotate_zyx(v: Vec3, yaw: f64, pitch: f64, roll: f64) -> Vec3 {
    // Intrinsic yaw (about z), then pitch (about y), then roll (about x).
    let (cy, sy) = (yaw.cos(), yaw.sin());
    let v1 = Vec3::new(cy * v.x - sy * v.y, sy * v.x + cy * v.y, v.z);
    let (cp, sp) = (pitch.cos(), pitch.sin());
    let v2 = Vec3::new(cp * v1.x + sp * v1.z, v1.y, -sp * v1.x + cp * v1.z);
    let (cr, sr) = (roll.cos(), roll.sin());
    Vec3::new(v2.x, cr * v2.y - sr * v2.z, sr * v2.y + cr * v2.z)
}

/// Rotate an ACN/SN3D ambisonic frame by yaw/pitch/roll, via projection
/// onto a Fibonacci sphere sampling (exact for band-limited fields as the
/// sampling is dense; 256 points).
#[must_use]
pub fn ambisonics_rotate(b: &[f64], yaw: f64, pitch: f64, roll: f64, order: u32) -> Vec<f64> {
    let n_ch = ((order + 1) * (order + 1)) as usize;
    let n_pts = 256;
    let golden = PI * (3.0 - 5.0_f64.sqrt());
    let mut out = vec![0.0; n_ch];
    for p in 0..n_pts {
        let z = 1.0 - 2.0 * (p as f64 + 0.5) / n_pts as f64;
        let r = (1.0 - z * z).sqrt();
        let phi = golden * p as f64;
        let dir = Vec3::new(r * phi.cos(), r * phi.sin(), z);
        // Field value in direction `dir` (N3D-weighted synthesis).
        let az = dir.y.atan2(dir.x);
        let el = dir.z.asin();
        let y_here = ambisonics_encode(1.0, az, el, order);
        let mut value = 0.0;
        let mut ch = 0;
        for l in 0..=order {
            for _m in -(l as i32)..=(l as i32) {
                if ch < b.len() {
                    value += b[ch] * y_here[ch] * (2 * l + 1) as f64;
                }
                ch += 1;
            }
        }
        // Re-encode from the rotated direction.
        let rd = rotate_zyx(dir, yaw, pitch, roll);
        let raz = rd.y.atan2(rd.x);
        let rel = rd.z.asin().clamp(-PI / 2.0, PI / 2.0);
        let y_rot = ambisonics_encode(1.0, raz, rel, order);
        for ch in 0..n_ch {
            out[ch] += value * y_rot[ch] / n_pts as f64;
        }
    }
    out
}

// --- Binaural cues -------------------------------------------------------

/// Woodworth interaural time difference (s) for a spherical head of
/// radius `head_radius`; `azimuth` in radians from the median plane.
#[must_use]
pub fn itd_woodworth(azimuth: f64, head_radius: f64, c: f64) -> f64 {
    let az = azimuth.sin().asin(); // fold into [-π/2, π/2]
    head_radius / c * (az + az.sin())
}

/// Duda-Martens (Brown-Duda) spherical-head shadowing filter response at
/// one ear; `azimuth` is measured from that ear's axis (0 = ipsilateral).
#[must_use]
pub fn spherical_head_hrtf(azimuth: f64, freq: f64, head_radius: f64, c: f64) -> Complex {
    let omega0 = c / head_radius;
    let theta = azimuth.rem_euclid(TWO_PI);
    let theta = if theta > PI { TWO_PI - theta } else { theta };
    // Brown-Duda: α(θ) = 1.05 + 0.95 cos(θ · 180°/150°).
    let alpha = 1.05 + 0.95 * (theta * (180.0 / 150.0)).cos();
    let w = TWO_PI * freq;
    // H(s) = (α s + 2 ω0) / (s + 2 ω0), s = jw.
    let num = Complex::new(2.0 * omega0, alpha * w);
    let den = Complex::new(2.0 * omega0, w);
    num / den
}

/// Interaural level difference (dB, positive = louder in the near ear)
/// from the spherical-head model.
#[must_use]
pub fn ild_spherical_head(azimuth: f64, freq: f64, head_radius: f64) -> f64 {
    let c = 343.0;
    // Ear axes at ±90°.
    let near = spherical_head_hrtf(PI / 2.0 - azimuth, freq, head_radius, c).norm();
    let far = spherical_head_hrtf(PI / 2.0 + azimuth, freq, head_radius, c).norm();
    20.0 * (near / far.max(1e-12)).log10()
}

/// Simple binaural rendering: ITD (fractional delay) plus first-order
/// head-shadow filtering per ear.
#[must_use]
pub fn binaural_simple(x: &[f64], azimuth: f64, elevation: f64, fs: f64) -> (Vec<f64>, Vec<f64>) {
    let c = 343.0;
    let r = 0.0875;
    let itd = itd_woodworth(azimuth, r, c) * elevation.cos();
    let (delay_l, delay_r) = if itd >= 0.0 { (itd, 0.0) } else { (0.0, -itd) };
    // Head shadow: one-pole lowpass on the far ear.
    let omega0 = c / r;
    let alpha_l = 1.05 + 0.95 * ((PI / 2.0 + azimuth) * (180.0 / 150.0)).cos();
    let alpha_r = 1.05 + 0.95 * ((PI / 2.0 - azimuth) * (180.0 / 150.0)).cos();
    let shelf = |alpha: f64| -> Biquad {
        // Bilinear transform of (α s + 2ω0)/(s + 2ω0).
        let k = 2.0 * fs;
        let b0 = (alpha * k + 2.0 * omega0) / (k + 2.0 * omega0);
        let b1 = (2.0 * omega0 - alpha * k) / (k + 2.0 * omega0);
        let a1 = (2.0 * omega0 - k) / (k + 2.0 * omega0);
        Biquad::from_coeffs(b0, b1, 0.0, a1, 0.0)
    };
    let mut f_l = shelf(alpha_l);
    let mut f_r = shelf(alpha_r);
    let frac_delay = |x: &[f64], d_samples: f64| -> Vec<f64> {
        let d0 = d_samples.floor() as usize;
        let f = d_samples - d0 as f64;
        (0..x.len())
            .map(|i| {
                let a = if i >= d0 { x[i - d0] } else { 0.0 };
                let b = if i > d0 { x[i - d0 - 1] } else { 0.0 };
                a * (1.0 - f) + b * f
            })
            .collect()
    };
    let left = frac_delay(x, delay_l * fs);
    let right = frac_delay(x, delay_r * fs);
    (
        left.iter().map(|&v| f_l.process(v)).collect(),
        right.iter().map(|&v| f_r.process(v)).collect(),
    )
}

/// Doppler by retarded-time resampling: the source moves along
/// `source_path(t)`; each output sample reads the emission-time signal
/// value with 1/r distance attenuation.
#[must_use]
pub fn doppler_resample(
    x: &[f64],
    source_path: &dyn Fn(f64) -> Vec3,
    listener: Vec3,
    c: f64,
    fs: f64,
) -> Vec<f64> {
    (0..x.len())
        .map(|i| {
            let t = i as f64 / fs;
            // Solve t_e = t - |p(t_e) - listener| / c by fixed point.
            let mut te = t;
            for _ in 0..8 {
                let d = (source_path(te) - listener).magnitude();
                te = t - d / c;
            }
            let d = (source_path(te) - listener).magnitude().max(0.1);
            let pos = te * fs;
            if pos < 0.0 || pos >= (x.len() - 1) as f64 {
                0.0
            } else {
                let i0 = pos.floor() as usize;
                let f = pos - i0 as f64;
                (x[i0] * (1.0 - f) + x[i0 + 1] * f) / d
            }
        })
        .collect()
}

/// Inverse-distance gain with reference distance and rolloff exponent.
#[must_use]
pub fn distance_gain(d: f64, ref_d: f64, rolloff: f64) -> f64 {
    (ref_d / d.max(ref_d)).powf(rolloff)
}

/// Atmospheric absorption over distance `d` approximated as a 2nd-order
/// Butterworth lowpass whose cutoff gives 3 dB of ISO 9613-style
/// high-frequency loss at that range.
#[must_use]
pub fn air_absorption_filter(d: f64, humidity: f64, temp: f64, fs: f64) -> Sos {
    // Simplified absorption coefficient (dB/m) dominated by the O2/N2
    // relaxation terms; adequate for audible-range rendering.
    let alpha_db_per_m = |f: f64| -> f64 {
        let f2 = f * f;
        let t_k = temp + 273.15;
        let t_rel = t_k / 293.15;
        let h = humidity.clamp(1.0, 100.0);
        let fr_o = 24.0 + 40400.0 * h / (h + 4.04e4 / 1e4) / 1000.0; // ~kHz-scale relaxation
        let fr_n = (9.0 + 280.0 * h / 100.0) * t_rel.powf(-0.5);
        1.84e-11 * f2 / t_rel.sqrt()
            + t_rel.powf(-2.5)
                * (0.01275 * (-2239.1 / t_k).exp() * f2 / (fr_o + f2 / fr_o)
                    + 0.1068 * (-3352.0 / t_k).exp() * f2 / (fr_n + f2 / fr_n))
            * 8.686
    };
    // Find the frequency attenuated by 3 dB over distance d.
    let (mut lo, mut hi) = (100.0_f64, 0.49 * fs);
    if alpha_db_per_m(hi) * d < 3.0 {
        return butterworth(2, IirKind::Lowpass(hi), fs);
    }
    for _ in 0..60 {
        let mid = (lo * hi).sqrt();
        if alpha_db_per_m(mid) * d < 3.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    butterworth(2, IirKind::Lowpass((lo * hi).sqrt().max(200.0)), fs)
}

// --- Geometric room acoustics --------------------------------------------

/// Shoebox image-source impulse response (Allen-Berkley). `absorption`
/// holds wall absorption coefficients in the order
/// [-x, +x, -y, +y, -z, +z].
#[allow(clippy::too_many_arguments)] // physical parameter list
#[must_use]
pub fn image_source_ir(
    room: Vec3,
    source: Vec3,
    listener: Vec3,
    absorption: [f64; 6],
    max_order: usize,
    fs: f64,
    c: f64,
) -> Vec<f64> {
    let beta: Vec<f64> = absorption.iter().map(|a| (1.0 - a).max(0.0).sqrt()).collect();
    let n = max_order as i32;
    // Enough length for the farthest image.
    let max_d = (room.magnitude()) * (max_order as f64 + 1.0) + (source - listener).magnitude();
    let len = ((max_d / c) * fs) as usize + 64;
    let mut ir = vec![0.0; len];
    let axis = |l: i32, q: i32, room_l: f64, s: f64| -> (f64, i32, i32) {
        // Image coordinate and wall-hit counts (Allen-Berkley).
        let x = 2.0 * l as f64 * room_l + if q == 0 { s } else { -s };
        ((x), (l - q).abs(), l.abs())
    };
    for lx in -n..=n {
        for qx in 0..2 {
            let (ix, hx1, hx2) = axis(lx, qx, room.x, source.x);
            for ly in -n..=n {
                for qy in 0..2 {
                    let (iy, hy1, hy2) = axis(ly, qy, room.y, source.y);
                    for lz in -n..=n {
                        for qz in 0..2 {
                            let (iz, hz1, hz2) = axis(lz, qz, room.z, source.z);
                            let order = hx1 + hx2 + hy1 + hy2 + hz1 + hz2;
                            if order as usize > max_order {
                                continue;
                            }
                            let img = Vec3::new(ix, iy, iz);
                            let d = (img - listener).magnitude().max(1e-3);
                            let gain = beta[0].powi(hx1)
                                * beta[1].powi(hx2)
                                * beta[2].powi(hy1)
                                * beta[3].powi(hy2)
                                * beta[4].powi(hz1)
                                * beta[5].powi(hz2)
                                / (4.0 * PI * d);
                            let t = d / c * fs;
                            let i0 = t.floor() as usize;
                            let f = t - i0 as f64;
                            if i0 + 1 < ir.len() {
                                ir[i0] += gain * (1.0 - f);
                                ir[i0 + 1] += gain * f;
                            }
                        }
                    }
                }
            }
        }
    }
    ir
}

/// Stochastic ray-traced energy impulse response in an arbitrary closed
/// mesh; `absorption[i]` indexes by triangle material. Amplitude is the
/// square root of collected energy per sample bin.
#[allow(clippy::too_many_arguments)] // physical parameter list
#[must_use]
pub fn ray_tracing_ir(
    room_mesh: &Mesh,
    source: Vec3,
    listener: Vec3,
    absorption: &[f64],
    n_rays: usize,
    max_bounces: usize,
    fs: f64,
    c: f64,
    rng: &mut Rng,
) -> Vec<f64> {
    let detector_r = 0.5;
    let len = (2.0 * fs) as usize;
    let mut energy = vec![0.0; len];
    for _ in 0..n_rays {
        // Uniform random direction.
        let z = 2.0 * rng.next_f64() - 1.0;
        let phi = TWO_PI * rng.next_f64();
        let r = (1.0 - z * z).sqrt();
        let mut dir = Vec3::new(r * phi.cos(), r * phi.sin(), z);
        let mut pos = source;
        let mut e = 1.0 / n_rays as f64;
        let mut path = 0.0;
        for _ in 0..max_bounces {
            let Some(hit) = room_mesh.intersect_ray(pos, dir, 1e-6) else {
                break;
            };
            // Detector sphere crossing on this segment?
            let to_l = listener - pos;
            let proj = to_l.dot(&dir);
            if proj > 0.0 && proj < hit.t {
                let closest = (to_l - dir * proj).magnitude();
                if closest < detector_r {
                    let d_total = path + proj;
                    let idx = (d_total / c * fs) as usize;
                    if idx < len {
                        // Weight by the chord through the detector.
                        energy[idx] += e / (d_total * d_total).max(1.0);
                    }
                }
            }
            path += hit.t;
            pos = hit.point;
            let mat = room_mesh.materials.get(hit.triangle).copied().unwrap_or(0);
            let a = absorption.get(mat % absorption.len().max(1)).copied().unwrap_or(0.1);
            e *= (1.0 - a).max(0.0);
            if e < 1e-9 {
                break;
            }
            // Specular reflection.
            dir = (dir - hit.normal * (2.0 * dir.dot(&hit.normal))).normalized();
        }
    }
    energy.iter().map(|e| e.sqrt()).collect()
}

/// Direct sound plus the six first-order reflections of a shoebox room:
/// (arrival time s, 1/(4πd) gain, unit direction of arrival).
#[must_use]
pub fn early_reflections(
    room: Vec3,
    source: Vec3,
    listener: Vec3,
    c: f64,
) -> Vec<(f64, f64, Vec3)> {
    let mut images = vec![source];
    for axis in 0..3 {
        let (lo_img, hi_img) = match axis {
            0 => (
                Vec3::new(-source.x, source.y, source.z),
                Vec3::new(2.0 * room.x - source.x, source.y, source.z),
            ),
            1 => (
                Vec3::new(source.x, -source.y, source.z),
                Vec3::new(source.x, 2.0 * room.y - source.y, source.z),
            ),
            _ => (
                Vec3::new(source.x, source.y, -source.z),
                Vec3::new(source.x, source.y, 2.0 * room.z - source.z),
            ),
        };
        images.push(lo_img);
        images.push(hi_img);
    }
    images
        .iter()
        .map(|&img| {
            let dvec = img - listener;
            let d = dvec.magnitude().max(1e-6);
            (d / c, 1.0 / (4.0 * PI * d), dvec * (1.0 / d))
        })
        .collect()
}

// --- Microphone arrays ---------------------------------------------------

/// Delay-and-sum beamformer steered toward the unit direction `steer`
/// (plane-wave model): aligns and averages the mic signals.
#[must_use]
pub fn beamforming_delay_sum(
    mics: &[Vec3],
    signals: &[Vec<f64>],
    steer: Vec3,
    fs: f64,
    c: f64,
) -> Vec<f64> {
    let s = steer.normalized();
    let n = signals.iter().map(Vec::len).min().unwrap_or(0);
    // A wavefront from direction s reaches mic m at time -(m·s)/c relative
    // to the origin; delay each channel to align.
    let delays: Vec<f64> = mics.iter().map(|m| m.dot(&s) / c * fs).collect();
    let min_d = delays.iter().cloned().fold(f64::INFINITY, f64::min);
    let mut out = vec![0.0; n];
    for (sig, &d) in signals.iter().zip(&delays) {
        let shift = d - min_d;
        let i0 = shift.floor() as usize;
        let f = shift - i0 as f64;
        for (t, o) in out.iter_mut().enumerate() {
            let a = if t >= i0 { sig[t - i0] } else { 0.0 };
            let b = if t > i0 { sig[t - i0 - 1] } else { 0.0 };
            *o += a * (1.0 - f) + b * f;
        }
    }
    out.iter_mut().for_each(|v| *v /= mics.len() as f64);
    out
}

fn csolve(a: &mut [Vec<Complex>], b: &mut [Complex]) -> Option<Vec<Complex>> {
    let n = b.len();
    for col in 0..n {
        let piv = (col..n).max_by(|&i, &j| {
            a[i][col].norm().partial_cmp(&a[j][col].norm()).unwrap()
        })?;
        if a[piv][col].norm() < 1e-15 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        let pivot_row = a[col].clone();
        for row in col + 1..n {
            let factor = a[row][col] / pivot_row[col];
            for (k, &pv) in pivot_row.iter().enumerate().skip(col) {
                a[row][k] = a[row][k] - factor * pv;
            }
            b[row] = b[row] - factor * b[col];
        }
    }
    let mut x = vec![Complex::new(0.0, 0.0); n];
    for row in (0..n).rev() {
        let mut acc = b[row];
        for k in row + 1..n {
            acc = acc - a[row][k] * x[k];
        }
        x[row] = acc / a[row][row];
    }
    Some(x)
}

/// Narrowband frequency-domain MVDR beamformer at `freq`: per-block
/// spatial covariance with diagonal loading, steering toward `steer`.
#[allow(clippy::too_many_arguments)] // physical parameter list
#[must_use]
pub fn beamforming_mvdr(
    mics: &[Vec3],
    signals: &[Vec<f64>],
    steer: Vec3,
    freq: f64,
    fs: f64,
    c: f64,
    diagonal_loading: f64,
) -> Vec<f64> {
    let m = mics.len();
    let n = signals.iter().map(Vec::len).min().unwrap_or(0);
    if m == 0 || n == 0 {
        return Vec::new();
    }
    let s = steer.normalized();
    // Steering vector at this frequency (plane wave).
    let d: Vec<Complex> = mics
        .iter()
        .map(|mic| {
            let tau = mic.dot(&s) / c;
            let ph = TWO_PI * freq * tau;
            Complex::new(ph.cos(), ph.sin())
        })
        .collect();
    // Broadband covariance estimate from the raw signals (freq-flat
    // approximation): R = X Xᴴ using the analytic signal at `freq` via
    // quadrature demodulation.
    let mut base: Vec<Vec<Complex>> =
        vec![vec![Complex::new(0.0, 0.0); m]; m];
    let demod: Vec<Vec<Complex>> = signals
        .iter()
        .map(|sig| {
            (0..n)
                .map(|t| {
                    let ph = TWO_PI * freq * t as f64 / fs;
                    Complex::new(sig[t] * ph.cos(), -sig[t] * ph.sin())
                })
                .collect()
        })
        .collect();
    for i in 0..m {
        for j in 0..m {
            let mut acc = Complex::new(0.0, 0.0);
            for (zi, zj) in demod[i][..n].iter().zip(&demod[j][..n]) {
                acc = acc + *zi * zj.conjugate();
            }
            base[i][j] = acc * Complex::new(1.0 / n as f64, 0.0);
        }
    }
    let trace: f64 = (0..m).map(|i| base[i][i].re).sum();
    for (i, row) in base.iter_mut().enumerate() {
        row[i] = row[i] + Complex::new(diagonal_loading * trace / m as f64, 0.0);
    }
    // w = R⁻¹ d / (dᴴ R⁻¹ d).
    let mut a = base.clone();
    let mut rhs = d.clone();
    let Some(rinv_d) = csolve(&mut a, &mut rhs) else {
        return beamforming_delay_sum(mics, signals, steer, fs, c);
    };
    let mut denom = Complex::new(0.0, 0.0);
    for i in 0..m {
        denom = denom + d[i].conjugate() * rinv_d[i];
    }
    if denom.norm() < 1e-15 {
        return beamforming_delay_sum(mics, signals, steer, fs, c);
    }
    let w: Vec<Complex> = rinv_d.iter().map(|&v| v / denom).collect();
    // The beamformer output is y = wᴴx, so channel i is delayed by
    // +arg(w_i)/2πf (mod one period) and scaled by |w_i| (narrowband
    // approximation).
    let period = fs / freq;
    let delays: Vec<f64> =
        w.iter().map(|wi| (wi.arg() / (TWO_PI * freq) * fs).rem_euclid(period)).collect();
    let mut out = vec![0.0; n];
    for (ch, wi) in w.iter().enumerate() {
        let gain = wi.norm();
        let shift = delays[ch];
        let i0 = shift.floor() as usize;
        let f = shift - i0 as f64;
        for (t, o) in out.iter_mut().enumerate() {
            let a = if t >= i0 { signals[ch][t - i0] } else { 0.0 };
            let b = if t > i0 { signals[ch][t - i0 - 1] } else { 0.0 };
            *o += gain * (a * (1.0 - f) + b * f);
        }
    }
    out
}

/// GCC-PHAT time difference of arrival: delay of `b` relative to `a` in
/// seconds (positive = b lags a).
#[must_use]
pub fn tdoa_gcc_phat(a: &[f64], b: &[f64], fs: f64) -> f64 {
    use crate::transforms::fft::{fft, ifft};
    let n = crate::transforms::fft::next_power_of_two(a.len() + b.len());
    let mut fa: Vec<Complex> = a.iter().map(|&v| Complex::new(v, 0.0)).collect();
    let mut fb: Vec<Complex> = b.iter().map(|&v| Complex::new(v, 0.0)).collect();
    fa.resize(n, Complex::new(0.0, 0.0));
    fb.resize(n, Complex::new(0.0, 0.0));
    let sa = fft(&fa);
    let sb = fft(&fb);
    let cross: Vec<Complex> = sa
        .iter()
        .zip(&sb)
        .map(|(x, y)| {
            let p = *y * x.conjugate();
            let m = p.norm().max(1e-12);
            p * Complex::new(1.0 / m, 0.0)
        })
        .collect();
    let corr = ifft(&cross);
    let mags: Vec<f64> = corr.iter().map(|c| c.re).collect();
    let peak = (0..n).max_by(|&i, &j| mags[i].partial_cmp(&mags[j]).unwrap()).unwrap();
    // Parabolic refinement with circular indexing.
    let prev = mags[(peak + n - 1) % n];
    let next = mags[(peak + 1) % n];
    let denom = prev - 2.0 * mags[peak] + next;
    let off = if denom.abs() > 1e-30 { (0.5 * (prev - next) / denom).clamp(-0.5, 0.5) } else { 0.0 };
    let lag = if peak > n / 2 { peak as f64 - n as f64 } else { peak as f64 };
    (lag + off) / fs
}

/// Least-squares source localization from TDOAs relative to `mics[0]`
/// (`tdoas[i]` is the extra delay at mic i+1), by Gauss-Newton.
#[must_use]
pub fn localize_tdoa(mics: &[Vec3], tdoas: &[f64], c: f64) -> Vec3 {
    let mut x = Vec3::new(0.0, 0.0, 0.0);
    for m in mics {
        x = x + *m;
    }
    x = x * (1.0 / mics.len() as f64);
    x = x + Vec3::new(0.1, 0.1, 0.1); // avoid symmetric stationary point
    for _ in 0..50 {
        // Residuals r_i = (|x-m_i| - |x-m_0|) - c τ_i.
        let d0 = (x - mics[0]).magnitude().max(1e-9);
        let g0 = (x - mics[0]) * (1.0 / d0);
        let mut jt_j = [[0.0; 3]; 3];
        let mut jt_r = [0.0; 3];
        for (i, &tau) in tdoas.iter().enumerate() {
            let mi = mics[i + 1];
            let di = (x - mi).magnitude().max(1e-9);
            let gi = (x - mi) * (1.0 / di);
            let r = (di - d0) - c * tau;
            let j = gi - g0;
            let jv = [j.x, j.y, j.z];
            for a in 0..3 {
                for b in 0..3 {
                    jt_j[a][b] += jv[a] * jv[b];
                }
                jt_r[a] += jv[a] * r;
            }
        }
        for (a, row) in jt_j.iter_mut().enumerate() {
            row[a] += 1e-9;
        }
        if let Some(step) = solve3(jt_j, jt_r) {
            x = x - Vec3::new(step[0], step[1], step[2]);
            if (step[0].powi(2) + step[1].powi(2) + step[2].powi(2)).sqrt() < 1e-10 {
                break;
            }
        } else {
            break;
        }
    }
    x
}

// --- Sonar and loudspeakers ----------------------------------------------

/// Round-trip echo time to range.
#[must_use]
pub fn sonar_range(t_echo: f64, c: f64) -> f64 {
    0.5 * c * t_echo
}

/// Active sonar equation: echo excess = SL - 2 TL + TS - (NL - DI), dB.
#[must_use]
pub fn sonar_equation(sl: f64, tl: f64, ts: f64, nl: f64, di: f64) -> f64 {
    sl - 2.0 * tl + ts - (nl - di)
}

/// Linkwitz-Riley 4th-order crossover: (lowpass, highpass), each two
/// cascaded 2nd-order Butterworth sections; the pair sums to allpass.
#[must_use]
pub fn speaker_crossover_lr4(fc: f64, fs: f64) -> (Sos, Sos) {
    let lp1 = butterworth(2, IirKind::Lowpass(fc), fs);
    let hp1 = butterworth(2, IirKind::Highpass(fc), fs);
    let cascade = |a: &Sos| -> Sos {
        let mut sections = a.sections.clone();
        sections.extend(a.sections.clone());
        Sos { sections, gain: a.gain * a.gain }
    };
    (cascade(&lp1), cascade(&hp1))
}

/// Baffle-step compensation target: the +6 dB diffraction step of a
/// baffle of width `width_m`, centered at f3 = 115/width, as a high
/// shelf.
#[must_use]
pub fn speaker_baffle_step(width_m: f64, fs: f64) -> Sos {
    let f3 = 115.0 / width_m.max(0.05);
    Sos {
        sections: vec![Biquad::highshelf(f3, fs, 0.7, 6.0)],
        gain: 1.0,
    }
}

/// Sealed-box (2nd-order highpass) response magnitude in dB of a driver
/// with free-air resonance `fs_driver`, total Q `qts`, and compliance
/// volume `vas` in a box of `box_volume` (same units), at frequency `f`.
#[must_use]
pub fn thiele_small_response(fs_driver: f64, qts: f64, vas: f64, box_volume: f64, f: f64) -> f64 {
    let alpha = vas / box_volume;
    let fc = fs_driver * (1.0 + alpha).sqrt();
    let qtc = qts * (1.0 + alpha).sqrt();
    let w = f / fc;
    // |H|² of s²/(s² + s/Qtc + 1) at s = jw.
    let num = w.powi(4);
    let den = (1.0 - w * w).powi(2) + (w / qtc).powi(2);
    10.0 * (num / den).log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pan_laws() {
        // Constant power sums to unit power everywhere.
        for i in 0..21 {
            let pos = -1.0 + 0.1 * i as f64;
            let (l, r) = pan_constant_power(1.0, pos);
            assert!((l * l + r * r - 1.0).abs() < 1e-12, "at {pos}");
        }
        let (l, r) = pan_linear(1.0, 0.0);
        assert!((l - 0.5).abs() < 1e-12 && (r - 0.5).abs() < 1e-12);
        let (l, r) = pan_minus_4_5_db(1.0, 0.0);
        let db = 20.0 * l.log10();
        assert!((db + 4.5).abs() < 0.1, "center {db} dB");
        assert!((l - r).abs() < 1e-12);
        // Hard right.
        let (l, r) = pan_constant_power(1.0, 1.0);
        assert!(l.abs() < 1e-12 && (r - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_vbap() {
        let speakers = [-PI / 4.0, PI / 4.0, 3.0 * PI / 4.0, -3.0 * PI / 4.0];
        // Source exactly at a speaker: that speaker only.
        let g = pan_vbap_2d(PI / 4.0, &speakers);
        assert!((g[1] - 1.0).abs() < 1e-9, "{g:?}");
        assert!(g[0].abs() < 1e-9 && g[2].abs() < 1e-9);
        // Source at front center: equal on the front pair.
        let g = pan_vbap_2d(0.0, &speakers);
        assert!((g[0] - g[1]).abs() < 1e-9);
        assert!((g[0] * g[0] + g[1] * g[1] - 1.0).abs() < 1e-9);
        // 3D: cube layout, source toward one speaker.
        let cube: Vec<Vec3> = (0..8)
            .map(|i| {
                Vec3::new(
                    if i & 1 == 0 { -1.0 } else { 1.0 },
                    if i & 2 == 0 { -1.0 } else { 1.0 },
                    if i & 4 == 0 { -1.0 } else { 1.0 },
                )
            })
            .collect();
        let g3 = pan_vbap_3d(Vec3::new(1.0, 1.0, 1.0), &cube);
        assert!((g3.iter().map(|v| v * v).sum::<f64>() - 1.0).abs() < 1e-9);
        // The gain-weighted speaker directions reconstruct the source
        // direction.
        let mut recon = Vec3::new(0.0, 0.0, 0.0);
        for (g, s) in g3.iter().zip(&cube) {
            recon = recon + s.normalized() * *g;
        }
        let recon = recon.normalized();
        let target = Vec3::new(1.0, 1.0, 1.0).normalized();
        assert!((recon - target).magnitude() < 1e-9, "vbap3d direction {recon:?}");
    }

    #[test]
    fn test_ambisonics() {
        // 1st-order FuMa at front: W = 1/√2, X = 1, Y = Z = 0.
        let b = ambisonics_encode_1st(1.0, 0.0, 0.0);
        assert!((b[0] - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12);
        assert!((b[1] - 1.0).abs() < 1e-12);
        assert!(b[2].abs() < 1e-12 && b[3].abs() < 1e-12);
        // ACN/SN3D 1st order: [W, Y, Z, X] with unit weights.
        let acn = ambisonics_encode(1.0, 0.3, 0.2, 1);
        assert!((acn[0] - 1.0).abs() < 1e-12);
        assert!((acn[1] - (0.3_f64.sin() * 0.2_f64.cos())).abs() < 1e-9);
        assert!((acn[2] - 0.2_f64.sin()).abs() < 1e-9);
        assert!((acn[3] - (0.3_f64.cos() * 0.2_f64.cos())).abs() < 1e-9);
        // Projection decode: speaker at the source direction is loudest.
        let b2 = ambisonics_encode(1.0, 0.5, 0.0, 2);
        let spk = [(0.5, 0.0), (0.5 + PI, 0.0), (0.5 + PI / 2.0, 0.0), (0.5 - PI / 2.0, 0.0)];
        let out = ambisonics_decode(&b2, &spk, 2);
        assert!(out[0] > out[1] && out[0] > out[2] && out[0] > out[3], "{out:?}");
        // Rotation: encoding at az then yawing by -az matches encoding
        // at 0 (order 2, sampled rotation).
        let rot = ambisonics_rotate(&b2, -0.5, 0.0, 0.0, 2);
        let ref0 = ambisonics_encode(1.0, 0.0, 0.0, 2);
        for (r, e) in rot.iter().zip(&ref0) {
            assert!((r - e).abs() < 0.02, "rotate {rot:?} vs {ref0:?}");
        }
    }

    #[test]
    fn test_binaural_cues() {
        let (r, c) = (0.0875, 343.0);
        assert!(itd_woodworth(0.0, r, c).abs() < 1e-12);
        let side = itd_woodworth(PI / 2.0, r, c);
        assert!((side - r / c * (PI / 2.0 + 1.0)).abs() < 1e-9);
        assert!(side > 0.0006 && side < 0.0007);
        // Head shadow: high frequencies attenuated more on the far side.
        let far_hi = spherical_head_hrtf(PI, 6000.0, r, c).norm();
        let far_lo = spherical_head_hrtf(PI, 200.0, r, c).norm();
        assert!(far_hi < far_lo, "shadow {far_hi} vs {far_lo}");
        let near_hi = spherical_head_hrtf(0.0, 6000.0, r, c).norm();
        assert!(near_hi > 1.0, "boost {near_hi}");
        assert!(ild_spherical_head(PI / 3.0, 4000.0, r) > 3.0);
        assert!(ild_spherical_head(0.0, 4000.0, r).abs() < 1e-9);
        let x: Vec<f64> = (0..4800).map(|i| (TWO_PI * 500.0 * i as f64 / 48000.0).sin()).collect();
        let (l, rr) = binaural_simple(&x, 0.8, 0.0, 48000.0);
        let el: f64 = l.iter().map(|v| v * v).sum();
        let er: f64 = rr.iter().map(|v| v * v).sum();
        assert!(er > el, "source at +az should favor the right ear");
    }

    #[test]
    fn test_doppler_and_distance() {
        let fs = 48000.0;
        let c = 343.0;
        let f0 = 1000.0;
        let x: Vec<f64> = (0..96000).map(|i| (TWO_PI * f0 * i as f64 / fs).sin()).collect();
        // Source approaching the listener at 20 m/s along x.
        let v = 20.0;
        let path = move |t: f64| Vec3::new(100.0 - v * t, 2.0, 0.0);
        let y = doppler_resample(&x, &path, Vec3::new(0.0, 0.0, 0.0), c, fs);
        let (f, _) = crate::audio::analysis::pitch_yin(&y[24000..24000 + 4096], fs, 200.0, 4000.0, 0.3)
            .unwrap();
        let expected = f0 * c / (c - v);
        assert!((f / expected - 1.0).abs() < 0.01, "doppler {f} vs {expected}");
        assert!((distance_gain(2.0, 1.0, 1.0) - 0.5).abs() < 1e-12);
        assert!((distance_gain(0.5, 1.0, 1.0) - 1.0).abs() < 1e-12);
        // Air absorption: longer distances → lower cutoff.
        let near = air_absorption_filter(10.0, 50.0, 20.0, fs);
        let far = air_absorption_filter(1000.0, 50.0, 20.0, fs);
        let g_near = near.freq_response(10000.0, fs).norm();
        let g_far = far.freq_response(10000.0, fs).norm();
        assert!(g_far < g_near, "air absorption {g_far} vs {g_near}");
    }

    #[test]
    fn test_image_source_and_rays() {
        let fs = 16000.0;
        let c = 343.0;
        // Asymmetric geometry so no two images share an arrival bin.
        let room = Vec3::new(5.3, 4.1, 3.2);
        let src = Vec3::new(1.2, 1.1, 0.9);
        let lst = Vec3::new(3.9, 2.7, 1.8);
        let ir = image_source_ir(room, src, lst, [0.3; 6], 2, fs, c);
        // Direct sound: linear interpolation splits the pulse over two
        // bins whose sum is exactly 1/(4πd).
        let d_direct = (src - lst).magnitude();
        let idx_direct = (d_direct / c * fs) as usize;
        let direct: f64 = ir[idx_direct..=idx_direct + 1].iter().sum();
        let expect = 1.0 / (4.0 * PI * d_direct);
        assert!((direct / expect - 1.0).abs() < 0.02, "direct {direct} vs {expect}");
        // First floor reflection: geometric path time, β/(4πd) energy.
        let img = Vec3::new(src.x, src.y, -src.z);
        let d1 = (img - lst).magnitude();
        let i1 = (d1 / c * fs) as usize;
        let window: f64 = ir[i1..=i1 + 1].iter().sum();
        let expect1 = (1.0_f64 - 0.3).sqrt() / (4.0 * PI * d1);
        assert!((window / expect1 - 1.0).abs() < 0.05, "floor reflection {window} vs {expect1}");
        // Ray tracing in the same shoebox: energy arrives, decays.
        let mesh = Mesh::box_room(room);
        let mut rng = Rng::new(42);
        let rir = ray_tracing_ir(&mesh, src, lst, &[0.3], 2000, 20, fs, c, &mut rng);
        let total: f64 = rir.iter().map(|v| v * v).sum();
        assert!(total > 0.0, "no ray energy detected");
        let early: f64 = rir[..8000].iter().map(|v| v * v).sum();
        let late: f64 = rir[16000..].iter().map(|v| v * v).sum();
        assert!(early > late, "ray IR should decay");
        // Early reflections list: direct plus 6 walls, sorted times sane.
        let er = early_reflections(room, src, lst, c);
        assert_eq!(er.len(), 7);
        assert!((er[0].0 - d_direct / c).abs() < 1e-12);
        assert!(er.iter().skip(1).all(|&(t, _, _)| t > er[0].0));
    }

    #[test]
    fn test_beamforming_and_tdoa() {
        let fs = 48000.0;
        let c = 343.0;
        // GCC-PHAT with a known integer delay.
        let mut rng = Rng::new(7);
        let a: Vec<f64> = (0..4096).map(|_| 2.0 * rng.next_f64() - 1.0).collect();
        let delay = 37;
        let mut b = vec![0.0; a.len()];
        b[delay..].copy_from_slice(&a[..a.len() - delay]);
        let tau = tdoa_gcc_phat(&a, &b, fs);
        assert!((tau * fs - delay as f64).abs() < 0.05, "gcc-phat {}", tau * fs);
        // Delay-and-sum steering toward a plane wave boosts it.
        let mics = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.1, 0.0, 0.0),
            Vec3::new(0.2, 0.0, 0.0),
            Vec3::new(0.3, 0.0, 0.0),
        ];
        let dir = Vec3::new(1.0, 0.0, 0.0); // wave travelling toward -x? steer dir
        let f0 = 1000.0;
        let signals: Vec<Vec<f64>> = mics
            .iter()
            .map(|m| {
                let tau = m.dot(&dir) / c;
                (0..4800)
                    .map(|i| (TWO_PI * f0 * (i as f64 / fs + tau)).sin())
                    .collect()
            })
            .collect();
        let aligned = beamforming_delay_sum(&mics, &signals, dir, fs, c);
        let e_aligned: f64 = aligned[500..4000].iter().map(|v| v * v).sum();
        // Steering the wrong way misaligns and cancels partially.
        let wrong = beamforming_delay_sum(&mics, &signals, Vec3::new(-1.0, 0.0, 0.0), fs, c);
        let e_wrong: f64 = wrong[500..4000].iter().map(|v| v * v).sum();
        assert!(e_aligned > 1.5 * e_wrong, "beamforming gain {e_aligned} vs {e_wrong}");
        let mvdr = beamforming_mvdr(&mics, &signals, dir, f0, fs, c, 1e-2);
        assert!(mvdr.iter().all(|v| v.is_finite()));
        let e_mvdr: f64 = mvdr[500..4000].iter().map(|v| v * v).sum();
        assert!(e_mvdr > 0.1 * e_aligned, "mvdr collapsed: {e_mvdr}");
        // TDOA localization of a known source.
        let mics4 = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
        ];
        let src = Vec3::new(2.0, 1.0, 0.5);
        let d0 = (src - mics4[0]).magnitude();
        let tdoas: Vec<f64> = mics4[1..]
            .iter()
            .map(|m| ((src - *m).magnitude() - d0) / c)
            .collect();
        let found = localize_tdoa(&mics4, &tdoas, c);
        assert!((found - src).magnitude() < 0.01, "localized {found:?}");
    }

    #[test]
    fn test_sonar_and_speakers() {
        assert!((sonar_range(2.0, 1500.0) - 1500.0).abs() < 1e-9);
        assert!((sonar_equation(220.0, 60.0, 15.0, 70.0, 20.0) - 65.0).abs() < 1e-12);
        let fs = 48000.0;
        let (lp, hp) = speaker_crossover_lr4(1000.0, fs);
        // Each is -6 dB at fc; the sum is allpass (flat magnitude).
        let g_lp = lp.freq_response(1000.0, fs).norm();
        assert!((20.0 * g_lp.log10() + 6.02).abs() < 0.2, "LR4 LP at fc {g_lp}");
        for f in [100.0, 300.0, 1000.0, 3000.0, 10000.0] {
            let s = lp.freq_response(f, fs) + hp.freq_response(f, fs);
            assert!((s.norm() - 1.0).abs() < 0.01, "LR4 sum at {f}: {}", s.norm());
        }
        // Baffle step: +6 dB shelf above the step frequency.
        let bs = speaker_baffle_step(0.25, fs);
        let hi = bs.freq_response(10000.0, fs).norm();
        let lo = bs.freq_response(50.0, fs).norm();
        assert!((20.0 * (hi / lo).log10() - 6.0).abs() < 0.5);
        // Thiele-Small: flat well above fc, -12 dB/oct below.
        let flat = thiele_small_response(30.0, 0.4, 100.0, 50.0, 2000.0);
        assert!(flat.abs() < 0.5, "passband {flat}");
        let f_low = thiele_small_response(30.0, 0.4, 100.0, 50.0, 10.0);
        let f_half = thiele_small_response(30.0, 0.4, 100.0, 50.0, 5.0);
        assert!((f_low - f_half - 12.0).abs() < 1.5, "slope {} vs {}", f_low, f_half);
    }
}
