//! Coherent noise: Perlin gradient noise (Perlin 2002), OpenSimplex2
//! (ported from K.jpg's reference implementation), value noise,
//! Worley cellular noise, fractal combinators (fBm, turbulence,
//! ridged and hybrid multifractals, domain warping, curl noise), and
//! terrain synthesis (diamond-square, spectral synthesis, thermal
//! and hydraulic erosion, void-and-cluster blue noise).

use crate::math::{Vec2, Vec3};
use crate::mesh::isosurface::{ScalarField2, ScalarField3};
use crate::monte_carlo::Rng;
use crate::spatial::primitives::{Aabb, Rect};

fn fade(t: f64) -> f64 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

fn floor_i(x: f64) -> i64 {
    x.floor() as i64
}

/// Classic improved Perlin gradient noise (Perlin, "Improving
/// Noise", 2002) with a seeded permutation table. Values are in
/// [-1, 1] and zero at every integer lattice point.
#[derive(Debug, Clone)]
pub struct Perlin {
    perm: [u8; 512],
}

impl Perlin {
    /// Permutation table shuffled by the seed (Fisher-Yates over the
    /// crate Rng).
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let mut p: [u8; 256] = [0; 256];
        for (i, v) in p.iter_mut().enumerate() {
            *v = i as u8;
        }
        for i in (1..256).rev() {
            let j = (rng.next_f64() * (i + 1) as f64) as usize % (i + 1);
            p.swap(i, j);
        }
        let mut perm = [0u8; 512];
        for i in 0..512 {
            perm[i] = p[i & 255];
        }
        Self { perm }
    }

    fn hash2(&self, x: i64, y: i64) -> u8 {
        let xi = (x & 255) as usize;
        let yi = (y & 255) as usize;
        self.perm[self.perm[xi] as usize + yi]
    }

    fn hash3(&self, x: i64, y: i64, z: i64) -> u8 {
        let xi = (x & 255) as usize;
        let yi = (y & 255) as usize;
        let zi = (z & 255) as usize;
        self.perm[self.perm[self.perm[xi] as usize + yi] as usize + zi]
    }

    fn hash4(&self, x: i64, y: i64, z: i64, w: i64) -> u8 {
        let wi = (w & 255) as usize;
        self.perm[self.hash3(x, y, z) as usize + wi]
    }

    /// 1-D gradient noise: gradients ±1, ±2 at integer knots.
    #[must_use]
    pub fn noise_1d(&self, x: f64) -> f64 {
        let x0 = floor_i(x);
        let t = x - x0 as f64;
        let g = |h: u8, d: f64| -> f64 {
            let s = [1.0, -1.0, 2.0, -2.0][(h & 3) as usize];
            s * d
        };
        let a = g(self.hash2(x0, 0), t);
        let b = g(self.hash2(x0 + 1, 0), t - 1.0);
        // Normalize: max |value| for +-2 gradients is 2 * fade-lerp
        // peak 0.5 -> 1.
        lerp(a, b, fade(t)) * 0.5
    }

    fn grad2(h: u8, dx: f64, dy: f64) -> f64 {
        // 8 gradient directions (+-1, +-2 combinations, normalized
        // overall below).
        match h & 7 {
            0 => dx + dy,
            1 => dx - dy,
            2 => -dx + dy,
            3 => -dx - dy,
            4 => dx,
            5 => -dx,
            6 => dy,
            _ => -dy,
        }
    }

    /// 2-D Perlin noise in [-1, 1].
    #[must_use]
    pub fn noise_2d(&self, x: f64, y: f64) -> f64 {
        let (x0, y0) = (floor_i(x), floor_i(y));
        let (tx, ty) = (x - x0 as f64, y - y0 as f64);
        let (u, v) = (fade(tx), fade(ty));
        let n00 = Self::grad2(self.hash2(x0, y0), tx, ty);
        let n10 = Self::grad2(self.hash2(x0 + 1, y0), tx - 1.0, ty);
        let n01 = Self::grad2(self.hash2(x0, y0 + 1), tx, ty - 1.0);
        let n11 = Self::grad2(self.hash2(x0 + 1, y0 + 1), tx - 1.0, ty - 1.0);
        // 1/sqrt(2) scaling bounds the output by 1.
        lerp(lerp(n00, n10, u), lerp(n01, n11, u), v) * std::f64::consts::FRAC_1_SQRT_2
    }

    fn grad3(h: u8, x: f64, y: f64, z: f64) -> f64 {
        // The 12 edge-vector gradients of Improving Noise.
        match h & 15 {
            0 => x + y,
            1 => -x + y,
            2 => x - y,
            3 => -x - y,
            4 => x + z,
            5 => -x + z,
            6 => x - z,
            7 => -x - z,
            8 => y + z,
            9 => -y + z,
            10 => y - z,
            11 => -y - z,
            12 => y + x,
            13 => -y + z,
            14 => y - x,
            _ => -y - z,
        }
    }

    /// 3-D Perlin noise in [-1, 1].
    #[must_use]
    pub fn noise_3d(&self, x: f64, y: f64, z: f64) -> f64 {
        let (x0, y0, z0) = (floor_i(x), floor_i(y), floor_i(z));
        let (tx, ty, tz) = (x - x0 as f64, y - y0 as f64, z - z0 as f64);
        let (u, v, w) = (fade(tx), fade(ty), fade(tz));
        let g = |dx: i64, dy: i64, dz: i64| -> f64 {
            Self::grad3(
                self.hash3(x0 + dx, y0 + dy, z0 + dz),
                tx - dx as f64,
                ty - dy as f64,
                tz - dz as f64,
            )
        };
        let x00 = lerp(g(0, 0, 0), g(1, 0, 0), u);
        let x10 = lerp(g(0, 1, 0), g(1, 1, 0), u);
        let x01 = lerp(g(0, 0, 1), g(1, 0, 1), u);
        let x11 = lerp(g(0, 1, 1), g(1, 1, 1), u);
        lerp(lerp(x00, x10, v), lerp(x01, x11, v), w)
    }

    fn grad4(h: u8, x: f64, y: f64, z: f64, w: f64) -> f64 {
        // 32 gradients: (+-1, +-1, +-1, 0) with the zero in the
        // position selected by the top bits.
        let sx = if h & 1 == 0 { 1.0 } else { -1.0 };
        let sy = if h & 2 == 0 { 1.0 } else { -1.0 };
        let sz = if h & 4 == 0 { 1.0 } else { -1.0 };
        match (h >> 3) & 3 {
            0 => sx * y + sy * z + sz * w,
            1 => sx * x + sy * z + sz * w,
            2 => sx * x + sy * y + sz * w,
            _ => sx * x + sy * y + sz * z,
        }
    }

    /// 4-D Perlin noise in [-1, 1].
    #[must_use]
    pub fn noise_4d(&self, x: f64, y: f64, z: f64, w: f64) -> f64 {
        let (x0, y0, z0, w0) = (floor_i(x), floor_i(y), floor_i(z), floor_i(w));
        let t = [x - x0 as f64, y - y0 as f64, z - z0 as f64, w - w0 as f64];
        let f = [fade(t[0]), fade(t[1]), fade(t[2]), fade(t[3])];
        let mut acc = 0.0;
        for corner in 0..16 {
            let d = [corner & 1, (corner >> 1) & 1, (corner >> 2) & 1, (corner >> 3) & 1];
            let h = self.hash4(
                x0 + d[0] as i64,
                y0 + d[1] as i64,
                z0 + d[2] as i64,
                w0 + d[3] as i64,
            );
            let g = Self::grad4(
                h,
                t[0] - f64::from(d[0]),
                t[1] - f64::from(d[1]),
                t[2] - f64::from(d[2]),
                t[3] - f64::from(d[3]),
            );
            let mut weight = 1.0;
            for k in 0..4 {
                weight *= if d[k] == 1 { f[k] } else { 1.0 - f[k] };
            }
            acc += weight * g;
        }
        acc * 0.577 // ~1/sqrt(3): bounds the sum by 1
    }

    /// Gradient of the 2-D noise by central differences.
    #[must_use]
    pub fn gradient_2d(&self, x: f64, y: f64) -> Vec2 {
        let h = 1e-5;
        Vec2::new(
            (self.noise_2d(x + h, y) - self.noise_2d(x - h, y)) / (2.0 * h),
            (self.noise_2d(x, y + h) - self.noise_2d(x, y - h)) / (2.0 * h),
        )
    }

    /// Gradient of the 3-D noise by central differences.
    #[must_use]
    pub fn gradient_3d(&self, x: f64, y: f64, z: f64) -> Vec3 {
        let h = 1e-5;
        Vec3::new(
            (self.noise_3d(x + h, y, z) - self.noise_3d(x - h, y, z)) / (2.0 * h),
            (self.noise_3d(x, y + h, z) - self.noise_3d(x, y - h, z)) / (2.0 * h),
            (self.noise_3d(x, y, z + h) - self.noise_3d(x, y, z - h)) / (2.0 * h),
        )
    }
}

// ---------------------------------------------------------------
// OpenSimplex2 (K.jpg's "faster variant", ported from the public
// reference implementation; f64 throughout).
// ---------------------------------------------------------------

const OS2_PRIME_X: i64 = 0x5205_402B_9270_C86F;
const OS2_PRIME_Y: i64 = 0x598C_D327_0038_17B5;
const OS2_PRIME_Z: i64 = 0x5BCC_226E_9FA0_BACB;
const OS2_HASH_MULT: i64 = 0x53A3_F72D_EEC5_46F5;
const OS2_SEED_FLIP_3D: i64 = -0x52D5_47B2_E96E_D629;

const OS2_SKEW_2D: f64 = 0.366_025_403_784_439;
const OS2_UNSKEW_2D: f64 = -0.211_324_865_405_187_13;
const OS2_ROOT3OVER3: f64 = 0.577_350_269_189_626;
const OS2_NORMALIZER_2D: f64 = 0.010_016_341_213_657_12;
const OS2_NORMALIZER_3D: f64 = 0.079_698_376_689_353_31;
const OS2_R2_2D: f64 = 0.5;
const OS2_R2_3D: f64 = 0.6;

fn os2_gradients_2d() -> &'static [f64] {
    use std::sync::OnceLock;
    static TABLE: OnceLock<Vec<f64>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let base = [
            0.382_683_432_365_09,
            0.923_879_532_511_287,
            0.923_879_532_511_287,
            0.382_683_432_365_09,
            0.923_879_532_511_287,
            -0.382_683_432_365_09,
            0.382_683_432_365_09,
            -0.923_879_532_511_287,
            -0.382_683_432_365_09,
            -0.923_879_532_511_287,
            -0.923_879_532_511_287,
            -0.382_683_432_365_09,
            -0.923_879_532_511_287,
            0.382_683_432_365_09,
            -0.382_683_432_365_09,
            0.923_879_532_511_287,
            0.130_526_192_220_052,
            0.991_444_861_373_81,
            0.608_761_429_008_721,
            0.793_353_340_291_235,
            0.793_353_340_291_235,
            0.608_761_429_008_721,
            0.991_444_861_373_81,
            0.130_526_192_220_051,
            0.991_444_861_373_81,
            -0.130_526_192_220_051,
            0.793_353_340_291_235,
            -0.608_761_429_008_72,
            0.608_761_429_008_721,
            -0.793_353_340_291_235,
            0.130_526_192_220_052,
            -0.991_444_861_373_81,
            -0.130_526_192_220_052,
            -0.991_444_861_373_81,
            -0.608_761_429_008_721,
            -0.793_353_340_291_235,
            -0.793_353_340_291_235,
            -0.608_761_429_008_721,
            -0.991_444_861_373_81,
            -0.130_526_192_220_052,
            -0.991_444_861_373_81,
            0.130_526_192_220_051,
            -0.793_353_340_291_235,
            0.608_761_429_008_721,
            -0.608_761_429_008_721,
            0.793_353_340_291_235,
            -0.130_526_192_220_052,
            0.991_444_861_373_81,
        ];
        (0..256).map(|i| base[i % base.len()] / OS2_NORMALIZER_2D).collect()
    })
}

fn os2_gradients_3d() -> &'static [f64] {
    use std::sync::OnceLock;
    static TABLE: OnceLock<Vec<f64>> = OnceLock::new();
    TABLE.get_or_init(|| {
        const A: f64 = 2.224_744_871_39;
        const B: f64 = 3.086_266_468_797_201_7;
        const C: f64 = 1.172_151_342_246_497_8;
        #[rustfmt::skip]
        let base: [f64; 192] = [
            A, A, -1.0, 0.0,   A, A, 1.0, 0.0,   B, C, 0.0, 0.0,   C, B, 0.0, 0.0,
            -A, A, -1.0, 0.0,  -A, A, 1.0, 0.0,  -C, B, 0.0, 0.0,  -B, C, 0.0, 0.0,
            -1.0, -A, -A, 0.0, 1.0, -A, -A, 0.0, 0.0, -B, -C, 0.0, 0.0, -C, -B, 0.0,
            -1.0, -A, A, 0.0,  1.0, -A, A, 0.0,  0.0, -C, B, 0.0,  0.0, -B, C, 0.0,
            -A, -A, -1.0, 0.0, -A, -A, 1.0, 0.0, -B, -C, 0.0, 0.0, -C, -B, 0.0, 0.0,
            -A, -1.0, -A, 0.0, -A, 1.0, -A, 0.0, -C, 0.0, -B, 0.0, -B, 0.0, -C, 0.0,
            -A, -1.0, A, 0.0,  -A, 1.0, A, 0.0,  -B, 0.0, C, 0.0,  -C, 0.0, B, 0.0,
            -1.0, A, -A, 0.0,  1.0, A, -A, 0.0,  0.0, C, -B, 0.0,  0.0, B, -C, 0.0,
            -1.0, A, A, 0.0,   1.0, A, A, 0.0,   0.0, B, C, 0.0,   0.0, C, B, 0.0,
            A, -A, -1.0, 0.0,  A, -A, 1.0, 0.0,  C, -B, 0.0, 0.0,  B, -C, 0.0, 0.0,
            A, -1.0, -A, 0.0,  A, 1.0, -A, 0.0,  B, 0.0, -C, 0.0,  C, 0.0, -B, 0.0,
            A, -1.0, A, 0.0,   A, 1.0, A, 0.0,   C, 0.0, B, 0.0,   B, 0.0, C, 0.0,
        ];
        (0..1024).map(|i| base[i % base.len()] / OS2_NORMALIZER_3D).collect()
    })
}

/// OpenSimplex2 noise (the "faster" variant): visually isotropic
/// gradient noise on simplex-style lattices, in [-1, 1]. The 3-D
/// evaluator uses the ImproveXY lattice orientation; 4-D noise is
/// not ported — use [`Perlin::noise_4d`] when a fourth dimension is
/// needed.
#[derive(Debug, Clone, Copy)]
pub struct OpenSimplex2 {
    seed: i64,
}

impl OpenSimplex2 {
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self { seed: seed as i64 }
    }

    fn grad2(seed: i64, xsvp: i64, ysvp: i64, dx: f64, dy: f64) -> f64 {
        let mut hash = seed ^ xsvp ^ ysvp;
        hash = hash.wrapping_mul(OS2_HASH_MULT);
        hash ^= hash >> (64 - 7 + 1);
        let gi = (hash as i32 & ((128 - 1) << 1)) as usize;
        let g = os2_gradients_2d();
        g[gi] * dx + g[gi | 1] * dy
    }

    fn grad3(seed: i64, xrvp: i64, yrvp: i64, zrvp: i64, dx: f64, dy: f64, dz: f64) -> f64 {
        let mut hash = (seed ^ xrvp) ^ (yrvp ^ zrvp);
        hash = hash.wrapping_mul(OS2_HASH_MULT);
        hash ^= hash >> (64 - 8 + 2);
        let gi = (hash as i32 & ((256 - 1) << 2)) as usize;
        let g = os2_gradients_3d();
        g[gi] * dx + g[gi | 1] * dy + g[gi | 2] * dz
    }

    /// 2-D noise, standard lattice orientation.
    #[must_use]
    pub fn noise_2d(&self, x: f64, y: f64) -> f64 {
        let s = OS2_SKEW_2D * (x + y);
        let (xs, ys) = (x + s, y + s);
        let seed = self.seed;
        let xsb = xs.floor() as i64;
        let ysb = ys.floor() as i64;
        let xi = xs - xsb as f64;
        let yi = ys - ysb as f64;
        let xsbp = xsb.wrapping_mul(OS2_PRIME_X);
        let ysbp = ysb.wrapping_mul(OS2_PRIME_Y);
        let t = (xi + yi) * OS2_UNSKEW_2D;
        let dx0 = xi + t;
        let dy0 = yi + t;
        let mut value = 0.0;
        let a0 = OS2_R2_2D - dx0 * dx0 - dy0 * dy0;
        if a0 > 0.0 {
            value = (a0 * a0) * (a0 * a0) * Self::grad2(seed, xsbp, ysbp, dx0, dy0);
        }
        let a1 = (2.0 * (1.0 + 2.0 * OS2_UNSKEW_2D) * (1.0 / OS2_UNSKEW_2D + 2.0)) * t
            + (-2.0 * (1.0 + 2.0 * OS2_UNSKEW_2D) * (1.0 + 2.0 * OS2_UNSKEW_2D)) + a0;
        if a1 > 0.0 {
            let dx1 = dx0 - (1.0 + 2.0 * OS2_UNSKEW_2D);
            let dy1 = dy0 - (1.0 + 2.0 * OS2_UNSKEW_2D);
            value += (a1 * a1)
                * (a1 * a1)
                * Self::grad2(
                    seed,
                    xsbp.wrapping_add(OS2_PRIME_X),
                    ysbp.wrapping_add(OS2_PRIME_Y),
                    dx1,
                    dy1,
                );
        }
        if dy0 > dx0 {
            let dx2 = dx0 - OS2_UNSKEW_2D;
            let dy2 = dy0 - (OS2_UNSKEW_2D + 1.0);
            let a2 = OS2_R2_2D - dx2 * dx2 - dy2 * dy2;
            if a2 > 0.0 {
                value += (a2 * a2)
                    * (a2 * a2)
                    * Self::grad2(seed, xsbp, ysbp.wrapping_add(OS2_PRIME_Y), dx2, dy2);
            }
        } else {
            let dx2 = dx0 - (OS2_UNSKEW_2D + 1.0);
            let dy2 = dy0 - OS2_UNSKEW_2D;
            let a2 = OS2_R2_2D - dx2 * dx2 - dy2 * dy2;
            if a2 > 0.0 {
                value += (a2 * a2)
                    * (a2 * a2)
                    * Self::grad2(seed, xsbp.wrapping_add(OS2_PRIME_X), ysbp, dx2, dy2);
            }
        }
        value
    }

    /// 3-D noise, ImproveXY orientation (Z up the lattice diagonal;
    /// best for terrain and time-varied 2-D fields with z = time).
    #[must_use]
    pub fn noise_3d(&self, x: f64, y: f64, z: f64) -> f64 {
        let xy = x + y;
        let s2 = xy * OS2_UNSKEW_2D;
        let zz = z * OS2_ROOT3OVER3;
        let xr = x + s2 + zz;
        let yr = y + s2 + zz;
        let zr = xy * -OS2_ROOT3OVER3 + zz;
        self.noise3_unrotated(xr, yr, zr)
    }

    fn noise3_unrotated(&self, xr: f64, yr: f64, zr: f64) -> f64 {
        let mut seed = self.seed;
        let xrb = xr.round() as i64;
        let yrb = yr.round() as i64;
        let zrb = zr.round() as i64;
        let mut xri = xr - xrb as f64;
        let mut yri = yr - yrb as f64;
        let mut zri = zr - zrb as f64;
        let mut xn: i64 = if xri >= 0.0 { -1 } else { 1 };
        let mut yn: i64 = if yri >= 0.0 { -1 } else { 1 };
        let mut zn: i64 = if zri >= 0.0 { -1 } else { 1 };
        let mut ax0 = -(xn as f64) * xri;
        let mut ay0 = -(yn as f64) * yri;
        let mut az0 = -(zn as f64) * zri;
        let mut xrbp = xrb.wrapping_mul(OS2_PRIME_X);
        let mut yrbp = yrb.wrapping_mul(OS2_PRIME_Y);
        let mut zrbp = zrb.wrapping_mul(OS2_PRIME_Z);
        let mut value = 0.0;
        let mut a = (OS2_R2_3D - xri * xri) - (yri * yri + zri * zri);
        let mut l = 0;
        loop {
            if a > 0.0 {
                value += (a * a) * (a * a) * Self::grad3(seed, xrbp, yrbp, zrbp, xri, yri, zri);
            }
            if ax0 >= ay0 && ax0 >= az0 {
                let mut b = a + ax0 + ax0;
                if b > 1.0 {
                    b -= 1.0;
                    value += (b * b)
                        * (b * b)
                        * Self::grad3(
                            seed,
                            xrbp.wrapping_sub(xn.wrapping_mul(OS2_PRIME_X)),
                            yrbp,
                            zrbp,
                            xri + xn as f64,
                            yri,
                            zri,
                        );
                }
            } else if ay0 > ax0 && ay0 >= az0 {
                let mut b = a + ay0 + ay0;
                if b > 1.0 {
                    b -= 1.0;
                    value += (b * b)
                        * (b * b)
                        * Self::grad3(
                            seed,
                            xrbp,
                            yrbp.wrapping_sub(yn.wrapping_mul(OS2_PRIME_Y)),
                            zrbp,
                            xri,
                            yri + yn as f64,
                            zri,
                        );
                }
            } else {
                let mut b = a + az0 + az0;
                if b > 1.0 {
                    b -= 1.0;
                    value += (b * b)
                        * (b * b)
                        * Self::grad3(
                            seed,
                            xrbp,
                            yrbp,
                            zrbp.wrapping_sub(zn.wrapping_mul(OS2_PRIME_Z)),
                            xri,
                            yri,
                            zri + zn as f64,
                        );
                }
            }
            if l == 1 {
                break;
            }
            l += 1;
            ax0 = 0.5 - ax0;
            ay0 = 0.5 - ay0;
            az0 = 0.5 - az0;
            xri = xn as f64 * ax0;
            yri = yn as f64 * ay0;
            zri = zn as f64 * az0;
            a += (0.75 - ax0) - (ay0 + az0);
            xrbp = xrbp.wrapping_add(if xn < 0 { OS2_PRIME_X } else { 0 });
            yrbp = yrbp.wrapping_add(if yn < 0 { OS2_PRIME_Y } else { 0 });
            zrbp = zrbp.wrapping_add(if zn < 0 { OS2_PRIME_Z } else { 0 });
            xn = -xn;
            yn = -yn;
            zn = -zn;
            seed ^= OS2_SEED_FLIP_3D;
        }
        value
    }
}

/// Lattice value noise: random values at integer lattice points,
/// interpolated (quintic-smoothed bilinear, optional bicubic).
#[derive(Debug, Clone)]
pub struct ValueNoise {
    perm: [u8; 512],
}

impl ValueNoise {
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let p = Perlin::new(seed);
        Self { perm: p.perm }
    }

    fn lattice2(&self, x: i64, y: i64) -> f64 {
        let xi = (x & 255) as usize;
        let yi = (y & 255) as usize;
        let h = self.perm[self.perm[xi] as usize + yi];
        f64::from(h) / 127.5 - 1.0
    }

    fn lattice3(&self, x: i64, y: i64, z: i64) -> f64 {
        let xi = (x & 255) as usize;
        let yi = (y & 255) as usize;
        let zi = (z & 255) as usize;
        let h = self.perm[self.perm[self.perm[xi] as usize + yi] as usize + zi];
        f64::from(h) / 127.5 - 1.0
    }

    /// Smoothed bilinear value noise in [-1, 1].
    #[must_use]
    pub fn noise_2d(&self, x: f64, y: f64) -> f64 {
        let (x0, y0) = (floor_i(x), floor_i(y));
        let (tx, ty) = (x - x0 as f64, y - y0 as f64);
        let (u, v) = (fade(tx), fade(ty));
        lerp(
            lerp(self.lattice2(x0, y0), self.lattice2(x0 + 1, y0), u),
            lerp(self.lattice2(x0, y0 + 1), self.lattice2(x0 + 1, y0 + 1), u),
            v,
        )
    }

    /// Smoothed trilinear value noise in [-1, 1].
    #[must_use]
    pub fn noise_3d(&self, x: f64, y: f64, z: f64) -> f64 {
        let (x0, y0, z0) = (floor_i(x), floor_i(y), floor_i(z));
        let (tx, ty, tz) = (x - x0 as f64, y - y0 as f64, z - z0 as f64);
        let (u, v, w) = (fade(tx), fade(ty), fade(tz));
        let mut c = [0.0; 8];
        for (i, ci) in c.iter_mut().enumerate() {
            *ci = self.lattice3(
                x0 + (i & 1) as i64,
                y0 + ((i >> 1) & 1) as i64,
                z0 + ((i >> 2) & 1) as i64,
            );
        }
        lerp(
            lerp(lerp(c[0], c[1], u), lerp(c[2], c[3], u), v),
            lerp(lerp(c[4], c[5], u), lerp(c[6], c[7], u), v),
            w,
        )
    }

    /// Catmull-Rom bicubic value noise (C¹, wider support).
    #[must_use]
    pub fn noise_2d_cubic(&self, x: f64, y: f64) -> f64 {
        let (x0, y0) = (floor_i(x), floor_i(y));
        let (tx, ty) = (x - x0 as f64, y - y0 as f64);
        let catmull = |p: [f64; 4], t: f64| -> f64 {
            0.5 * ((2.0 * p[1])
                + (-p[0] + p[2]) * t
                + (2.0 * p[0] - 5.0 * p[1] + 4.0 * p[2] - p[3]) * t * t
                + (-p[0] + 3.0 * p[1] - 3.0 * p[2] + p[3]) * t * t * t)
        };
        let mut rows = [0.0; 4];
        for (j, row) in rows.iter_mut().enumerate() {
            let vals = [
                self.lattice2(x0 - 1, y0 + j as i64 - 1),
                self.lattice2(x0, y0 + j as i64 - 1),
                self.lattice2(x0 + 1, y0 + j as i64 - 1),
                self.lattice2(x0 + 2, y0 + j as i64 - 1),
            ];
            *row = catmull(vals, tx);
        }
        catmull(rows, ty).clamp(-1.0, 1.0)
    }
}

/// Distance metrics for Worley noise.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Metric {
    Euclidean,
    Manhattan,
    Chebyshev,
    Minkowski(f64),
}

fn metric_2d(m: Metric, d: Vec2) -> f64 {
    match m {
        Metric::Euclidean => d.magnitude(),
        Metric::Manhattan => d.x.abs() + d.y.abs(),
        Metric::Chebyshev => d.x.abs().max(d.y.abs()),
        Metric::Minkowski(p) => (d.x.abs().powf(p) + d.y.abs().powf(p)).powf(1.0 / p),
    }
}

fn metric_3d(m: Metric, d: Vec3) -> f64 {
    match m {
        Metric::Euclidean => d.magnitude(),
        Metric::Manhattan => d.x.abs() + d.y.abs() + d.z.abs(),
        Metric::Chebyshev => d.x.abs().max(d.y.abs()).max(d.z.abs()),
        Metric::Minkowski(p) => {
            (d.x.abs().powf(p) + d.y.abs().powf(p) + d.z.abs().powf(p)).powf(1.0 / p)
        }
    }
}

fn cell_hash(seed: u64, x: i64, y: i64, z: i64) -> u64 {
    let mut h = seed
        ^ (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
        ^ (z as u64).wrapping_mul(0x1656_67B1_9E37_79F9);
    h ^= h >> 30;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^ (h >> 31)
}

fn hash_unit(h: u64, k: u64) -> f64 {
    let mut v = h ^ k.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    v ^= v >> 33;
    v = v.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    v ^= v >> 33;
    (v >> 11) as f64 / (1u64 << 53) as f64
}

/// Worley (cellular) noise: one feature point per grid cell of size
/// `cell`, hashed from the seed; F1/F2 are the distances to the
/// nearest and second-nearest feature points under `metric`.
#[derive(Debug, Clone, Copy)]
pub struct Worley {
    seed: u64,
    cell: f64,
    pub metric: Metric,
}

impl Worley {
    /// # Panics
    /// Panics unless `cell > 0`.
    #[must_use]
    pub fn new(seed: u64, cell: f64) -> Self {
        assert!(cell > 0.0, "cell size must be positive");
        Self { seed, cell, metric: Metric::Euclidean }
    }

    fn feature_2d(&self, ix: i64, iy: i64) -> Vec2 {
        let h = cell_hash(self.seed, ix, iy, 0);
        Vec2::new(
            (ix as f64 + hash_unit(h, 1)) * self.cell,
            (iy as f64 + hash_unit(h, 2)) * self.cell,
        )
    }

    fn feature_3d(&self, ix: i64, iy: i64, iz: i64) -> Vec3 {
        let h = cell_hash(self.seed, ix, iy, iz);
        Vec3::new(
            (ix as f64 + hash_unit(h, 1)) * self.cell,
            (iy as f64 + hash_unit(h, 2)) * self.cell,
            (iz as f64 + hash_unit(h, 3)) * self.cell,
        )
    }

    fn f12_2d(&self, x: f64, y: f64) -> (f64, f64) {
        let p = Vec2::new(x, y);
        let cx = floor_i(x / self.cell);
        let cy = floor_i(y / self.cell);
        let (mut f1, mut f2) = (f64::INFINITY, f64::INFINITY);
        for dy in -2..=2 {
            for dx in -2..=2 {
                let d = metric_2d(self.metric, self.feature_2d(cx + dx, cy + dy) - p);
                if d < f1 {
                    f2 = f1;
                    f1 = d;
                } else if d < f2 {
                    f2 = d;
                }
            }
        }
        (f1, f2)
    }

    fn f12_3d(&self, x: f64, y: f64, z: f64) -> (f64, f64) {
        let p = Vec3::new(x, y, z);
        let cx = floor_i(x / self.cell);
        let cy = floor_i(y / self.cell);
        let cz = floor_i(z / self.cell);
        let (mut f1, mut f2) = (f64::INFINITY, f64::INFINITY);
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let d =
                        metric_3d(self.metric, self.feature_3d(cx + dx, cy + dy, cz + dz) - p);
                    if d < f1 {
                        f2 = f1;
                        f1 = d;
                    } else if d < f2 {
                        f2 = d;
                    }
                }
            }
        }
        (f1, f2)
    }

    /// Distance to the nearest feature point.
    #[must_use]
    pub fn f1_2d(&self, x: f64, y: f64) -> f64 {
        self.f12_2d(x, y).0
    }

    /// Distance to the second-nearest feature point.
    #[must_use]
    pub fn f2_2d(&self, x: f64, y: f64) -> f64 {
        self.f12_2d(x, y).1
    }

    /// F2 − F1 (ridged cell boundaries).
    #[must_use]
    pub fn f2_minus_f1_2d(&self, x: f64, y: f64) -> f64 {
        let (f1, f2) = self.f12_2d(x, y);
        f2 - f1
    }

    #[must_use]
    pub fn f1_3d(&self, x: f64, y: f64, z: f64) -> f64 {
        self.f12_3d(x, y, z).0
    }

    #[must_use]
    pub fn f2_3d(&self, x: f64, y: f64, z: f64) -> f64 {
        self.f12_3d(x, y, z).1
    }

    /// Stable id of the cell owning the nearest feature point.
    #[must_use]
    pub fn cell_id_2d(&self, x: f64, y: f64) -> u64 {
        let p = Vec2::new(x, y);
        let cx = floor_i(x / self.cell);
        let cy = floor_i(y / self.cell);
        let mut best = f64::INFINITY;
        let mut id = 0u64;
        for dy in -2..=2 {
            for dx in -2..=2 {
                let d = metric_2d(self.metric, self.feature_2d(cx + dx, cy + dy) - p);
                if d < best {
                    best = d;
                    id = cell_hash(self.seed, cx + dx, cy + dy, 0);
                }
            }
        }
        id
    }
}

/// Fractional Brownian motion parameters.
#[derive(Debug, Clone, Copy)]
pub struct FbmParams {
    pub octaves: u32,
    pub lacunarity: f64,
    pub gain: f64,
    pub frequency: f64,
    pub amplitude: f64,
}

impl Default for FbmParams {
    fn default() -> Self {
        Self { octaves: 5, lacunarity: 2.0, gain: 0.5, frequency: 1.0, amplitude: 1.0 }
    }
}

/// fBm: Σ amplitude·gainⁱ · n(frequency·lacunarityⁱ · x).
#[must_use]
pub fn fbm_2d(n: &dyn Fn(f64, f64) -> f64, x: f64, y: f64, p: &FbmParams) -> f64 {
    let mut sum = 0.0;
    let mut amp = p.amplitude;
    let mut freq = p.frequency;
    for _ in 0..p.octaves {
        sum += amp * n(x * freq, y * freq);
        amp *= p.gain;
        freq *= p.lacunarity;
    }
    sum
}

/// 3-D fBm.
#[must_use]
pub fn fbm_3d(n: &dyn Fn(f64, f64, f64) -> f64, x: f64, y: f64, z: f64, p: &FbmParams) -> f64 {
    let mut sum = 0.0;
    let mut amp = p.amplitude;
    let mut freq = p.frequency;
    for _ in 0..p.octaves {
        sum += amp * n(x * freq, y * freq, z * freq);
        amp *= p.gain;
        freq *= p.lacunarity;
    }
    sum
}

/// Turbulence: fBm of |n| (Perlin 1985's marble basis).
#[must_use]
pub fn turbulence_2d(n: &dyn Fn(f64, f64) -> f64, x: f64, y: f64, p: &FbmParams) -> f64 {
    let mut sum = 0.0;
    let mut amp = p.amplitude;
    let mut freq = p.frequency;
    for _ in 0..p.octaves {
        sum += amp * n(x * freq, y * freq).abs();
        amp *= p.gain;
        freq *= p.lacunarity;
    }
    sum
}

/// Musgrave's ridged multifractal: octaves of (offset − |n|)²,
/// each weighted by the previous octave's signal.
#[must_use]
pub fn ridged_multifractal_2d(
    n: &dyn Fn(f64, f64) -> f64,
    x: f64,
    y: f64,
    p: &FbmParams,
    offset: f64,
    gain: f64,
) -> f64 {
    let mut freq = p.frequency;
    let mut amp = p.amplitude;
    let mut weight = 1.0;
    let mut sum = 0.0;
    for _ in 0..p.octaves {
        let mut signal = offset - n(x * freq, y * freq).abs();
        signal = signal * signal * weight;
        weight = (signal * gain).clamp(0.0, 1.0);
        sum += signal * amp;
        amp *= p.gain;
        freq *= p.lacunarity;
    }
    sum
}

/// Musgrave's hybrid multifractal (3-D): additive multifractal with
/// octave weights damped by the running product.
#[must_use]
pub fn hybrid_multifractal(
    n: &dyn Fn(f64, f64, f64) -> f64,
    x: f64,
    y: f64,
    z: f64,
    p: &FbmParams,
    offset: f64,
) -> f64 {
    let mut freq = p.frequency;
    let mut amp = p.amplitude;
    let mut result = (n(x * freq, y * freq, z * freq) + offset) * amp;
    let mut weight = result;
    freq *= p.lacunarity;
    amp *= p.gain;
    for _ in 1..p.octaves {
        weight = weight.min(1.0);
        let signal = (n(x * freq, y * freq, z * freq) + offset) * amp;
        result += weight * signal;
        weight *= signal;
        freq *= p.lacunarity;
        amp *= p.gain;
    }
    result
}

/// Billow: fBm of 2|n| − 1 (puffy cloud look).
#[must_use]
pub fn billow_2d(n: &dyn Fn(f64, f64) -> f64, x: f64, y: f64, p: &FbmParams) -> f64 {
    let mut sum = 0.0;
    let mut amp = p.amplitude;
    let mut freq = p.frequency;
    for _ in 0..p.octaves {
        sum += amp * (2.0 * n(x * freq, y * freq).abs() - 1.0);
        amp *= p.gain;
        freq *= p.lacunarity;
    }
    sum
}

/// Iterated domain warping (Quilez): the sample point is repeatedly
/// displaced by an fBm offset field before the final evaluation.
#[must_use]
pub fn domain_warp_2d(
    n: &dyn Fn(f64, f64) -> f64,
    x: f64,
    y: f64,
    p: &FbmParams,
    warp_strength: f64,
    iterations: usize,
) -> f64 {
    let (mut wx, mut wy) = (x, y);
    for _ in 0..iterations {
        let ox = fbm_2d(n, wx + 5.2, wy + 1.3, p);
        let oy = fbm_2d(n, wx + 1.7, wy + 9.2, p);
        wx = x + warp_strength * ox;
        wy = y + warp_strength * oy;
    }
    fbm_2d(n, wx, wy, p)
}

/// 3-D iterated domain warping.
#[must_use]
pub fn domain_warp_3d(
    n: &dyn Fn(f64, f64, f64) -> f64,
    x: f64,
    y: f64,
    z: f64,
    p: &FbmParams,
    strength: f64,
    iterations: usize,
) -> f64 {
    let (mut wx, mut wy, mut wz) = (x, y, z);
    for _ in 0..iterations {
        let ox = fbm_3d(n, wx + 5.2, wy + 1.3, wz + 2.8, p);
        let oy = fbm_3d(n, wx + 1.7, wy + 9.2, wz + 4.6, p);
        let oz = fbm_3d(n, wx + 8.3, wy + 2.8, wz + 7.1, p);
        wx = x + strength * ox;
        wy = y + strength * oy;
        wz = z + strength * oz;
    }
    fbm_3d(n, wx, wy, wz, p)
}

/// Divergence-free 2-D flow from a scalar noise potential:
/// v = (∂ψ/∂y, −∂ψ/∂x) by central differences.
///
/// # Panics
/// Panics unless `eps > 0`.
#[must_use]
pub fn curl_noise_2d(n: &dyn Fn(f64, f64) -> f64, x: f64, y: f64, eps: f64) -> Vec2 {
    assert!(eps > 0.0, "step must be positive");
    let dpdy = (n(x, y + eps) - n(x, y - eps)) / (2.0 * eps);
    let dpdx = (n(x + eps, y) - n(x - eps, y)) / (2.0 * eps);
    Vec2::new(dpdy, -dpdx)
}

/// Divergence-free 3-D flow: curl of a vector potential whose three
/// components are offset copies of `n` (Bridson et al. 2007).
///
/// # Panics
/// Panics unless `eps > 0`.
#[must_use]
pub fn curl_noise_3d(n: &dyn Fn(f64, f64, f64) -> f64, x: f64, y: f64, z: f64, eps: f64) -> Vec3 {
    assert!(eps > 0.0, "step must be positive");
    let p1 = |x: f64, y: f64, z: f64| n(x, y, z);
    let p2 = |x: f64, y: f64, z: f64| n(x + 31.416, y + 47.853, z + 12.793);
    let p3 = |x: f64, y: f64, z: f64| n(x + 233.145, y + 113.021, z + 331.173);
    let d = |f: &dyn Fn(f64, f64, f64) -> f64, axis: usize| -> f64 {
        let (mut a, mut b) = ((x, y, z), (x, y, z));
        match axis {
            0 => {
                a.0 += eps;
                b.0 -= eps;
            }
            1 => {
                a.1 += eps;
                b.1 -= eps;
            }
            _ => {
                a.2 += eps;
                b.2 -= eps;
            }
        }
        (f(a.0, a.1, a.2) - f(b.0, b.1, b.2)) / (2.0 * eps)
    };
    Vec3::new(
        d(&p3, 1) - d(&p2, 2),
        d(&p1, 2) - d(&p3, 0),
        d(&p2, 0) - d(&p1, 1),
    )
}

/// Samples a noise function into a scalar field.
#[must_use]
pub fn noise_field_2d(
    n: &dyn Fn(f64, f64) -> f64,
    bounds: &Rect,
    w: usize,
    h: usize,
) -> ScalarField2 {
    ScalarField2::from_fn(*bounds, w, h, &|p| n(p.x, p.y))
}

/// Samples a noise function into a 3-D scalar field.
#[must_use]
pub fn noise_field_3d(
    n: &dyn Fn(f64, f64, f64) -> f64,
    bounds: &Aabb,
    res: (usize, usize, usize),
) -> ScalarField3 {
    ScalarField3::from_fn(*bounds, res.0, res.1, res.2, &|p| n(p.x, p.y, p.z))
}

/// fBm heightmap (row-major, `w` × `h`) with optional thermal
/// erosion: material moves down slopes exceeding the talus angle,
/// smoothing scree until the terrain settles.
///
/// # Panics
/// Panics unless the grid has at least 2×2 samples.
#[must_use]
pub fn terrain_heightmap(
    seed: u64,
    w: usize,
    h: usize,
    p: &FbmParams,
    erosion_iters: usize,
) -> Vec<f64> {
    assert!(w >= 2 && h >= 2, "heightmap needs at least 2x2 samples");
    let perlin = Perlin::new(seed);
    let mut height: Vec<f64> = (0..w * h)
        .map(|i| {
            let x = (i % w) as f64 / w as f64 * 4.0;
            let y = (i / w) as f64 / h as f64 * 4.0;
            fbm_2d(&|a, b| perlin.noise_2d(a, b), x, y, p)
        })
        .collect();
    let talus = 4.0 / w as f64;
    for _ in 0..erosion_iters {
        let snapshot = height.clone();
        for j in 0..h {
            for i in 0..w {
                let idx = j * w + i;
                let mut lowest = idx;
                let mut steepest = 0.0;
                for (di, dj) in [(1i64, 0i64), (-1, 0), (0, 1), (0, -1)] {
                    let (ni, nj) = (i as i64 + di, j as i64 + dj);
                    if ni < 0 || nj < 0 || ni >= w as i64 || nj >= h as i64 {
                        continue;
                    }
                    let n = nj as usize * w + ni as usize;
                    let d = snapshot[idx] - snapshot[n];
                    if d > steepest {
                        steepest = d;
                        lowest = n;
                    }
                }
                if steepest > talus {
                    let moved = 0.25 * (steepest - talus);
                    height[idx] -= moved;
                    height[lowest] += moved;
                }
            }
        }
    }
    height
}

/// Hydraulic erosion droplet parameters (Beyer 2015-style droplet
/// simulation).
#[derive(Debug, Clone, Copy)]
pub struct ErosionParams {
    /// Blend between old direction and downhill gradient (0..1).
    pub inertia: f64,
    /// Carry capacity multiplier.
    pub capacity: f64,
    pub min_capacity: f64,
    pub erode_speed: f64,
    pub deposit_speed: f64,
    pub evaporate_speed: f64,
    pub gravity: f64,
    pub max_lifetime: u32,
}

impl Default for ErosionParams {
    fn default() -> Self {
        Self {
            inertia: 0.05,
            capacity: 4.0,
            min_capacity: 0.01,
            erode_speed: 0.3,
            deposit_speed: 0.3,
            evaporate_speed: 0.01,
            gravity: 4.0,
            max_lifetime: 30,
        }
    }
}

fn bilinear_height_gradient(height: &[f64], w: usize, h: usize, x: f64, y: f64) -> (f64, Vec2) {
    let xi = (x.floor() as usize).min(w - 2);
    let yi = (y.floor() as usize).min(h - 2);
    let (fx, fy) = (x - xi as f64, y - yi as f64);
    let h00 = height[yi * w + xi];
    let h10 = height[yi * w + xi + 1];
    let h01 = height[(yi + 1) * w + xi];
    let h11 = height[(yi + 1) * w + xi + 1];
    let gx = (h10 - h00) * (1.0 - fy) + (h11 - h01) * fy;
    let gy = (h01 - h00) * (1.0 - fx) + (h11 - h10) * fx;
    let hh = h00 * (1.0 - fx) * (1.0 - fy) + h10 * fx * (1.0 - fy)
        + h01 * (1.0 - fx) * fy
        + h11 * fx * fy;
    (hh, Vec2::new(gx, gy))
}

/// Simulates `droplets` water droplets over the heightmap, eroding
/// and depositing material along their paths.
///
/// # Panics
/// Panics unless the grid is at least 3×3 and `height.len() == w·h`.
pub fn hydraulic_erosion(
    height: &mut [f64],
    w: usize,
    h: usize,
    droplets: usize,
    rng: &mut Rng,
    params: &ErosionParams,
) {
    assert!(w >= 3 && h >= 3, "erosion needs at least a 3x3 grid");
    assert_eq!(height.len(), w * h, "height buffer size mismatch");
    for _ in 0..droplets {
        let mut x = rng.next_f64() * (w - 1) as f64;
        let mut y = rng.next_f64() * (h - 1) as f64;
        let mut dir = Vec2::ZERO;
        let mut speed = 1.0;
        let mut water = 1.0;
        let mut sediment = 0.0;
        for _ in 0..params.max_lifetime {
            let (h0, grad) = bilinear_height_gradient(height, w, h, x, y);
            dir = dir * params.inertia - grad * (1.0 - params.inertia);
            let len = dir.magnitude();
            if len < 1e-12 {
                break;
            }
            dir = dir * (1.0 / len);
            let (nx, ny) = (x + dir.x, y + dir.y);
            if nx < 0.0 || ny < 0.0 || nx >= (w - 1) as f64 || ny >= (h - 1) as f64 {
                break;
            }
            let (h1, _) = bilinear_height_gradient(height, w, h, nx, ny);
            let dh = h1 - h0;
            let capacity =
                (-dh).max(params.min_capacity) * speed * water * params.capacity;
            let cell = (y.floor() as usize).min(h - 2) * w + (x.floor() as usize).min(w - 2);
            if sediment > capacity || dh > 0.0 {
                let deposit = if dh > 0.0 {
                    sediment.min(dh)
                } else {
                    (sediment - capacity) * params.deposit_speed
                };
                sediment -= deposit;
                height[cell] += deposit;
            } else {
                let erode = ((capacity - sediment) * params.erode_speed).min(-dh);
                height[cell] -= erode;
                sediment += erode;
            }
            speed = (speed * speed + dh.abs() * params.gravity).sqrt();
            water *= 1.0 - params.evaporate_speed;
            x = nx;
            y = ny;
        }
    }
}

/// Diamond-square (plasma) fractal heightmap on a
/// (2^size_pow2 + 1)² grid, row-major, roughness halving the random
/// amplitude at each subdivision.
///
/// # Panics
/// Panics unless `1 <= size_pow2 <= 12`.
#[must_use]
pub fn diamond_square(size_pow2: u32, roughness: f64, seed: u64) -> Vec<f64> {
    assert!((1..=12).contains(&size_pow2), "size must be in 1..=12");
    let n = (1usize << size_pow2) + 1;
    let mut rng = Rng::new(seed);
    let mut grid = vec![0.0f64; n * n];
    let mut rand = |amp: f64| (rng.next_f64() * 2.0 - 1.0) * amp;
    grid[0] = rand(1.0);
    grid[n - 1] = rand(1.0);
    grid[(n - 1) * n] = rand(1.0);
    grid[n * n - 1] = rand(1.0);
    let mut step = n - 1;
    let mut amp = roughness;
    while step > 1 {
        let half = step / 2;
        // Diamond step.
        for j in (half..n).step_by(step) {
            for i in (half..n).step_by(step) {
                let avg = (grid[(j - half) * n + i - half]
                    + grid[(j - half) * n + i + half]
                    + grid[(j + half) * n + i - half]
                    + grid[(j + half) * n + i + half])
                    / 4.0;
                grid[j * n + i] = avg + rand(amp);
            }
        }
        // Square step.
        for j in (0..n).step_by(half) {
            let start = if (j / half).is_multiple_of(2) { half } else { 0 };
            for i in (start..n).step_by(step) {
                let mut sum = 0.0;
                let mut count = 0.0;
                if i >= half {
                    sum += grid[j * n + i - half];
                    count += 1.0;
                }
                if i + half < n {
                    sum += grid[j * n + i + half];
                    count += 1.0;
                }
                if j >= half {
                    sum += grid[(j - half) * n + i];
                    count += 1.0;
                }
                if j + half < n {
                    sum += grid[(j + half) * n + i];
                    count += 1.0;
                }
                grid[j * n + i] = sum / count + rand(amp);
            }
        }
        step = half;
        amp *= roughness;
    }
    grid
}

/// 1/f^β spectral synthesis by direct summation of 64 random plane
/// waves with amplitudes f^{−β/2} (row-major, values roughly in
/// [-1, 1] after normalization).
///
/// # Panics
/// Panics unless the grid has at least 2×2 samples.
#[must_use]
pub fn spectral_synthesis_2d(w: usize, h: usize, beta: f64, seed: u64) -> Vec<f64> {
    assert!(w >= 2 && h >= 2, "grid needs at least 2x2 samples");
    let mut rng = Rng::new(seed);
    let waves: Vec<(f64, f64, f64, f64)> = (0..64)
        .map(|k| {
            let f = 1.0 + (k / 4) as f64; // 16 frequency bands, 4 waves each
            let angle = rng.next_f64() * std::f64::consts::TAU;
            let phase = rng.next_f64() * std::f64::consts::TAU;
            let amp = f.powf(-beta / 2.0);
            (f * angle.cos(), f * angle.sin(), phase, amp)
        })
        .collect();
    let norm: f64 = waves.iter().map(|&(_, _, _, a)| a * a).sum::<f64>().sqrt()
        * std::f64::consts::FRAC_1_SQRT_2;
    let mut out = Vec::with_capacity(w * h);
    for j in 0..h {
        for i in 0..w {
            let x = i as f64 / w as f64;
            let y = j as f64 / h as f64;
            let v: f64 = waves
                .iter()
                .map(|&(kx, ky, phase, amp)| {
                    amp * (std::f64::consts::TAU * (kx * x + ky * y) + phase).sin()
                })
                .sum();
            out.push(v / (2.0 * norm));
        }
    }
    out
}

/// Stateless hash white noise in [-1, 1]: the same (seed, x, y)
/// always yields the same value, with no correlation between
/// nearby inputs.
#[must_use]
pub fn white_noise_2d(seed: u64, x: f64, y: f64) -> f64 {
    let h = cell_hash(seed, x.to_bits() as i64, y.to_bits() as i64, 0);
    hash_unit(h, 7) * 2.0 - 1.0
}

/// Void-and-cluster blue-noise threshold texture (Ulichney 1993):
/// returns ranks normalized to [0, 1), toroidally tileable; every
/// rank appears exactly once.
///
/// # Panics
/// Panics unless `w·h >= 4` (and `w, h >= 2`).
#[must_use]
pub fn blue_noise_texture(w: usize, h: usize, seed: u64) -> Vec<f64> {
    assert!(w >= 2 && h >= 2, "texture needs at least 2x2 pixels");
    let n = w * h;
    let sigma = 1.5f64;
    // Precomputed wrapped Gaussian splat.
    let radius = (3.0 * sigma).ceil() as i64;
    let mut rng = Rng::new(seed);
    let mut pattern = vec![false; n];
    let mut energy = vec![0.0f64; n];
    let splat = |energy: &mut [f64], cx: usize, cy: usize, sign: f64| {
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let x = (cx as i64 + dx).rem_euclid(w as i64) as usize;
                let y = (cy as i64 + dy).rem_euclid(h as i64) as usize;
                let r2 = (dx * dx + dy * dy) as f64;
                energy[y * w + x] += sign * (-r2 / (2.0 * sigma * sigma)).exp();
            }
        }
    };
    // Seed with ~1/10 random minority points.
    let ones = (n / 10).max(2);
    let mut placed = 0;
    while placed < ones {
        let idx = (rng.next_f64() * n as f64) as usize % n;
        if !pattern[idx] {
            pattern[idx] = true;
            splat(&mut energy, idx % w, idx / w, 1.0);
            placed += 1;
        }
    }
    // Relax: move the tightest cluster to the largest void until
    // stable (bounded passes).
    for _ in 0..10 * n {
        let cluster = (0..n)
            .filter(|&i| pattern[i])
            .max_by(|&a, &b| energy[a].total_cmp(&energy[b]))
            .expect("minority points exist");
        pattern[cluster] = false;
        splat(&mut energy, cluster % w, cluster / w, -1.0);
        let void = (0..n)
            .filter(|&i| !pattern[i])
            .min_by(|&a, &b| energy[a].total_cmp(&energy[b]))
            .expect("void exists");
        pattern[void] = true;
        splat(&mut energy, void % w, void / w, 1.0);
        if void == cluster {
            break;
        }
    }
    let prototype = pattern.clone();
    let proto_energy = energy.clone();
    let mut rank = vec![0usize; n];
    // Phase 1: peel the tightest cluster down to nothing.
    let mut current = prototype.clone();
    let mut e = proto_energy.clone();
    for r in (0..ones).rev() {
        let cluster = (0..n)
            .filter(|&i| current[i])
            .max_by(|&a, &b| e[a].total_cmp(&e[b]))
            .expect("minority points remain");
        current[cluster] = false;
        splat(&mut e, cluster % w, cluster / w, -1.0);
        rank[cluster] = r;
    }
    // Phase 2 + 3: fill the largest void until the texture is full.
    let mut current = prototype;
    let mut e = proto_energy;
    for r in ones..n {
        let void = (0..n)
            .filter(|&i| !current[i])
            .min_by(|&a, &b| e[a].total_cmp(&e[b]))
            .expect("empty pixels remain");
        current[void] = true;
        splat(&mut e, void % w, void / w, 1.0);
        rank[void] = r;
    }
    rank.into_iter().map(|r| r as f64 / n as f64).collect()
}

/// One Gabor kernel: a Gaussian-windowed cosine wave.
#[derive(Debug, Clone, Copy)]
pub struct GaborKernel {
    pub center: Vec2,
    /// Cycles per unit length along the orientation.
    pub frequency: f64,
    /// Wave direction in radians.
    pub orientation: f64,
    /// Gaussian bandwidth (larger decays faster).
    pub bandwidth: f64,
    pub amplitude: f64,
    pub phase: f64,
}

/// Sparse Gabor noise: the sum of the kernels at (x, y)
/// (Lagae et al. 2009 with an explicit kernel list).
#[must_use]
pub fn gabor_noise_2d(x: f64, y: f64, kernels: &[GaborKernel]) -> f64 {
    let p = Vec2::new(x, y);
    kernels
        .iter()
        .map(|k| {
            let d = p - k.center;
            let envelope = (-std::f64::consts::PI
                * k.bandwidth
                * k.bandwidth
                * d.magnitude_squared())
            .exp();
            let (s, c) = k.orientation.sin_cos();
            let carrier = (std::f64::consts::TAU * k.frequency * (d.x * c + d.y * s)
                + k.phase)
                .cos();
            k.amplitude * envelope * carrier
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perlin_lattice_zeros_and_range() {
        let p = Perlin::new(42);
        for i in -6i64..6 {
            for j in -6i64..6 {
                assert_eq!(p.noise_2d(i as f64, j as f64), 0.0, "2-D lattice zero");
                assert_eq!(p.noise_3d(i as f64, j as f64, 1.0), 0.0, "3-D lattice zero");
            }
            assert_eq!(p.noise_1d(i as f64), 0.0);
        }
        let mut rng = Rng::new(1);
        let (mut lo, mut hi) = (0.0f64, 0.0f64);
        for _ in 0..100_000 {
            let (x, y) = (rng.next_f64() * 40.0, rng.next_f64() * 40.0);
            let v = p.noise_2d(x, y);
            assert!((-1.0..=1.0).contains(&v), "2-D in [-1, 1] ({v})");
            lo = lo.min(v);
            hi = hi.max(v);
            let v3 = p.noise_3d(x, y, x * 0.3);
            assert!((-1.0..=1.0).contains(&v3), "3-D in [-1, 1] ({v3})");
            let v4 = p.noise_4d(x, y, x * 0.3, y * 0.7);
            assert!((-1.1..=1.1).contains(&v4), "4-D bounded ({v4})");
        }
        assert!(hi > 0.3 && lo < -0.3, "noise actually varies ({lo}..{hi})");
    }

    #[test]
    fn test_noise_continuity_and_determinism() {
        let p = Perlin::new(9);
        let s = OpenSimplex2::new(9);
        let v = ValueNoise::new(9);
        let mut rng = Rng::new(5);
        for _ in 0..2000 {
            let (x, y) = (rng.next_f64() * 20.0 - 10.0, rng.next_f64() * 20.0 - 10.0);
            for (name, a, b) in [
                ("perlin", p.noise_2d(x, y), p.noise_2d(x + 1e-4, y)),
                ("simplex", s.noise_2d(x, y), s.noise_2d(x + 1e-4, y)),
                ("value", v.noise_2d(x, y), v.noise_2d(x + 1e-4, y)),
                ("cubic", v.noise_2d_cubic(x, y), v.noise_2d_cubic(x + 1e-4, y)),
                ("simplex3", s.noise_3d(x, y, 0.7), s.noise_3d(x + 1e-4, y, 0.7)),
            ] {
                assert!((a - b).abs() < 1e-2, "{name} continuous ({})", (a - b).abs());
            }
        }
        // Same seed -> identical, different seed -> different.
        let p2 = Perlin::new(9);
        let p3 = Perlin::new(10);
        assert_eq!(p.noise_2d(3.7, 1.2), p2.noise_2d(3.7, 1.2));
        assert_ne!(p.noise_2d(3.7, 1.2), p3.noise_2d(3.7, 1.2));
        let s2 = OpenSimplex2::new(9);
        assert_eq!(s.noise_2d(3.7, 1.2), s2.noise_2d(3.7, 1.2));
    }

    #[test]
    fn test_simplex_range_and_gradients() {
        let s = OpenSimplex2::new(77);
        let mut rng = Rng::new(3);
        let (mut lo, mut hi) = (0.0f64, 0.0f64);
        for _ in 0..100_000 {
            let (x, y) = (rng.next_f64() * 60.0, rng.next_f64() * 60.0);
            let v = s.noise_2d(x, y);
            assert!((-1.001..=1.001).contains(&v), "2-D in [-1, 1] ({v})");
            let v3 = s.noise_3d(x, y, x * 0.1);
            assert!((-1.001..=1.001).contains(&v3), "3-D in [-1, 1] ({v3})");
            lo = lo.min(v);
            hi = hi.max(v);
        }
        assert!(hi > 0.5 && lo < -0.5, "simplex uses its range ({lo}..{hi})");
        // Perlin gradient matches finite differences of the value.
        let p = Perlin::new(4);
        let g = p.gradient_2d(1.37, 2.81);
        let h = 1e-6;
        let gx = (p.noise_2d(1.37 + h, 2.81) - p.noise_2d(1.37 - h, 2.81)) / (2.0 * h);
        assert!((g.x - gx).abs() < 1e-4);
        let g3 = p.gradient_3d(0.5, 0.25, 0.75);
        assert!(g3.x.is_finite() && g3.y.is_finite() && g3.z.is_finite());
    }

    #[test]
    fn test_worley_properties() {
        let w = Worley::new(11, 1.0);
        // F1 = 0 exactly at a feature point; F2 >= F1 >= 0 everywhere.
        let f = w.feature_2d(3, 4);
        assert!(w.f1_2d(f.x, f.y) < 1e-12, "F1 vanishes at features");
        let mut rng = Rng::new(8);
        for _ in 0..2000 {
            let (x, y) = (rng.next_f64() * 20.0 - 10.0, rng.next_f64() * 20.0 - 10.0);
            let f1 = w.f1_2d(x, y);
            let f2 = w.f2_2d(x, y);
            assert!(f1 >= 0.0 && f2 >= f1, "0 <= F1 <= F2");
            assert!((w.f2_minus_f1_2d(x, y) - (f2 - f1)).abs() < 1e-12);
            let f13 = w.f1_3d(x, y, 0.5);
            let f23 = w.f2_3d(x, y, 0.5);
            assert!(f13 >= 0.0 && f23 >= f13);
            // Manhattan >= Euclidean >= Chebyshev distances.
            let mut wm = w;
            wm.metric = Metric::Manhattan;
            let mut wc = w;
            wc.metric = Metric::Chebyshev;
            assert!(wm.f1_2d(x, y) >= f1 - 1e-12);
            assert!(wc.f1_2d(x, y) <= f1 + 1e-12);
            let mut wk = w;
            wk.metric = Metric::Minkowski(2.0);
            assert!((wk.f1_2d(x, y) - f1).abs() < 1e-9, "Minkowski(2) = Euclidean");
        }
        // Same cell id within a cell interior.
        let id = w.cell_id_2d(f.x, f.y);
        assert_eq!(id, w.cell_id_2d(f.x + 0.01, f.y + 0.01));
    }

    #[test]
    fn test_value_noise_3d_is_trilinear_interpolation_of_its_lattice() {
        let v = ValueNoise::new(2024);
        // At integer lattice points the fade weights are exactly 0, so
        // the value *is* the lattice sample: one of the 256 levels
        // h/127.5 − 1 with h an integer byte.
        for i in -4i64..4 {
            for j in -4i64..4 {
                for k in -4i64..4 {
                    let s = v.noise_3d(i as f64, j as f64, k as f64);
                    assert!((-1.0..=1.0).contains(&s), "lattice value {s} out of range");
                    let h = (s + 1.0) * 127.5;
                    assert!(
                        (h - h.round()).abs() < 1e-9 && (0.0..=255.0).contains(&h.round()),
                        "lattice value {s} is not a byte level ({h})"
                    );
                }
            }
        }
        // fade(1/2) = 1/2, so the cell centre is the exact mean of the
        // eight corner lattice values, and each edge midpoint is the
        // mean of its two endpoints.
        let mut rng = Rng::new(31);
        for _ in 0..200 {
            let (i, j, k) = (
                (rng.next_f64() * 40.0 - 20.0).floor(),
                (rng.next_f64() * 40.0 - 20.0).floor(),
                (rng.next_f64() * 40.0 - 20.0).floor(),
            );
            let corner = |di: f64, dj: f64, dk: f64| v.noise_3d(i + di, j + dj, k + dk);
            let mean = (0..8)
                .map(|c| {
                    corner(
                        f64::from(c & 1),
                        f64::from((c >> 1) & 1),
                        f64::from((c >> 2) & 1),
                    )
                })
                .sum::<f64>()
                / 8.0;
            let centre = v.noise_3d(i + 0.5, j + 0.5, k + 0.5);
            assert!((centre - mean).abs() < 1e-12, "cell centre {centre} vs {mean}");
            let edge = v.noise_3d(i + 0.5, j, k);
            let edge_mean = 0.5 * (corner(0.0, 0.0, 0.0) + corner(1.0, 0.0, 0.0));
            assert!((edge - edge_mean).abs() < 1e-12, "edge {edge} vs {edge_mean}");
        }
    }

    #[test]
    fn test_value_noise_3d_bounded_deterministic_and_continuous() {
        let v = ValueNoise::new(7);
        let same = ValueNoise::new(7);
        let other = ValueNoise::new(8);
        let mut rng = Rng::new(41);
        let (mut lo, mut hi) = (0.0f64, 0.0f64);
        let mut differs = 0usize;
        for _ in 0..50_000 {
            let (x, y, z) = (
                rng.next_f64() * 60.0 - 30.0,
                rng.next_f64() * 60.0 - 30.0,
                rng.next_f64() * 60.0 - 30.0,
            );
            let a = v.noise_3d(x, y, z);
            // Trilinear blends of values in [-1, 1] stay in [-1, 1].
            assert!((-1.0..=1.0).contains(&a), "3-D value noise {a} out of range");
            // Deterministic: same seed, same input, bit-identical.
            assert_eq!(a, same.noise_3d(x, y, z), "same seed must agree exactly");
            assert_eq!(a, v.noise_3d(x, y, z), "repeatable");
            if a != other.noise_3d(x, y, z) {
                differs += 1;
            }
            // Lipschitz-ish continuity: the quintic fade has derivative
            // at most 15/8 per axis and lattice values span 2, so a step
            // of h moves the value by at most 3·(15/8)·h ≈ 5.63h.
            let h = 1e-4;
            for d in [
                (v.noise_3d(x + h, y, z) - a).abs(),
                (v.noise_3d(x, y + h, z) - a).abs(),
                (v.noise_3d(x, y, z + h) - a).abs(),
            ] {
                assert!(d < 5.7 * h, "continuity violated: {d} over {h}");
            }
            lo = lo.min(a);
            hi = hi.max(a);
        }
        // A different seed gives a different field almost everywhere.
        assert!(differs > 49_000, "seeds must decorrelate ({differs}/50000)");
        // The noise actually uses its range.
        assert!(hi > 0.6 && lo < -0.6, "3-D value noise range {lo}..{hi}");
        // Neighbouring lattice cells carry different values (the field
        // is not constant along any axis).
        let mut distinct = std::collections::HashSet::new();
        for i in 0..16 {
            distinct.insert(v.noise_3d(i as f64 + 0.5, 0.5, 0.5).to_bits());
        }
        assert!(distinct.len() > 12, "cells vary along x ({})", distinct.len());
    }

    #[test]
    fn test_fbm_family() {
        let p = Perlin::new(21);
        let n2 = |x: f64, y: f64| p.noise_2d(x, y);
        let n3 = |x: f64, y: f64, z: f64| p.noise_3d(x, y, z);
        let params = FbmParams::default();
        let max_amp: f64 = (0..params.octaves).map(|i| params.gain.powi(i as i32)).sum();
        let mut rng = Rng::new(2);
        for _ in 0..500 {
            let (x, y) = (rng.next_f64() * 10.0, rng.next_f64() * 10.0);
            let f = fbm_2d(&n2, x, y, &params);
            assert!(f.abs() <= max_amp + 1e-9, "fBm bounded by geometric sum");
            assert!(fbm_3d(&n3, x, y, 0.3, &params).abs() <= max_amp + 1e-9);
            let t = turbulence_2d(&n2, x, y, &params);
            assert!((0.0..=max_amp + 1e-9).contains(&t), "turbulence non-negative");
            assert!(billow_2d(&n2, x, y, &params).is_finite());
            assert!(ridged_multifractal_2d(&n2, x, y, &params, 1.0, 2.0).is_finite());
            assert!(hybrid_multifractal(&n3, x, y, 0.3, &params, 0.7).is_finite());
            assert!(domain_warp_2d(&n2, x, y, &params, 0.5, 2).is_finite());
            assert!(domain_warp_3d(&n3, x, y, 0.3, &params, 0.5, 1).is_finite());
        }
    }

    #[test]
    fn test_curl_noise_divergence_free() {
        let p = Perlin::new(33);
        let n2 = |x: f64, y: f64| p.noise_2d(x, y);
        let n3 = |x: f64, y: f64, z: f64| p.noise_3d(x, y, z);
        let eps = 1e-4;
        let mut rng = Rng::new(6);
        for _ in 0..200 {
            let (x, y, z) =
                (rng.next_f64() * 8.0, rng.next_f64() * 8.0, rng.next_f64() * 8.0);
            // Numerical divergence of the curl field.
            let h = 1e-3;
            let div2 = (curl_noise_2d(&n2, x + h, y, eps).x
                - curl_noise_2d(&n2, x - h, y, eps).x)
                / (2.0 * h)
                + (curl_noise_2d(&n2, x, y + h, eps).y - curl_noise_2d(&n2, x, y - h, eps).y)
                    / (2.0 * h);
            assert!(div2.abs() < 1e-3, "2-D curl noise divergence {div2}");
            let div3 = (curl_noise_3d(&n3, x + h, y, z, eps).x
                - curl_noise_3d(&n3, x - h, y, z, eps).x)
                / (2.0 * h)
                + (curl_noise_3d(&n3, x, y + h, z, eps).y
                    - curl_noise_3d(&n3, x, y - h, z, eps).y)
                    / (2.0 * h)
                + (curl_noise_3d(&n3, x, y, z + h, eps).z
                    - curl_noise_3d(&n3, x, y, z - h, eps).z)
                    / (2.0 * h);
            assert!(div3.abs() < 1e-3, "3-D curl noise divergence {div3}");
        }
    }

    #[test]
    fn test_fields_and_terrain() {
        let p = Perlin::new(14);
        let bounds = Rect { min: Vec2::new(-2.0, -2.0), max: Vec2::new(2.0, 2.0) };
        let field = noise_field_2d(&|x, y| p.noise_2d(x, y), &bounds, 16, 12);
        assert_eq!(field.data.len(), 16 * 12);
        let vol = Aabb { min: Vec3::ZERO, max: Vec3::new(1.0, 1.0, 1.0) };
        let field3 = noise_field_3d(&|x, y, z| p.noise_3d(x, y, z), &vol, (6, 5, 4));
        assert_eq!(field3.data.len(), 120);
        let terrain = terrain_heightmap(3, 33, 33, &FbmParams::default(), 10);
        assert_eq!(terrain.len(), 33 * 33);
        assert!(terrain.iter().all(|v| v.is_finite()));
        // Thermal erosion conserves material.
        let raw = terrain_heightmap(3, 33, 33, &FbmParams::default(), 0);
        let sum_raw: f64 = raw.iter().sum();
        let sum_eroded: f64 = terrain.iter().sum();
        assert!((sum_raw - sum_eroded).abs() < 1e-9, "thermal erosion conserves mass");
        // Hydraulic erosion runs and keeps the map finite.
        let mut eroded = raw.clone();
        let mut rng = Rng::new(4);
        hydraulic_erosion(&mut eroded, 33, 33, 200, &mut rng, &ErosionParams::default());
        assert!(eroded.iter().all(|v| v.is_finite()));
        assert!(eroded.iter().zip(&raw).any(|(a, b)| (a - b).abs() > 1e-12), "terrain changed");
    }

    #[test]
    fn test_diamond_square_and_spectral() {
        let g = diamond_square(5, 0.5, 77);
        assert_eq!(g.len(), 33 * 33);
        assert!(g.iter().all(|v| v.is_finite()));
        assert_eq!(diamond_square(5, 0.5, 77), g, "deterministic per seed");
        assert_ne!(diamond_square(5, 0.5, 78), g);
        let s = spectral_synthesis_2d(32, 32, 2.0, 5);
        assert_eq!(s.len(), 1024);
        let lo = s.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(hi > lo, "spectral field varies");
        assert!(lo >= -2.0 && hi <= 2.0, "roughly normalized ({lo}..{hi})");
        // Smoother than white noise: adjacent-sample correlation high
        // for beta = 2.
        let mut num = 0.0;
        let mut den = 0.0;
        for j in 0..32 {
            for i in 0..31 {
                num += s[j * 32 + i] * s[j * 32 + i + 1];
                den += s[j * 32 + i] * s[j * 32 + i];
            }
        }
        assert!(num / den > 0.5, "1/f^2 field is smooth ({})", num / den);
    }

    #[test]
    fn test_white_noise_and_blue_noise() {
        assert_eq!(white_noise_2d(1, 0.5, 0.25), white_noise_2d(1, 0.5, 0.25));
        assert_ne!(white_noise_2d(1, 0.5, 0.25), white_noise_2d(2, 0.5, 0.25));
        let mut rng = Rng::new(1);
        for _ in 0..1000 {
            let v = white_noise_2d(3, rng.next_f64(), rng.next_f64());
            assert!((-1.0..=1.0).contains(&v));
        }
        let (w, h) = (16, 16);
        let tex = blue_noise_texture(w, h, 9);
        assert_eq!(tex.len(), w * h);
        // Every rank appears exactly once.
        let mut ranks: Vec<usize> = tex.iter().map(|&v| (v * (w * h) as f64).round() as usize).collect();
        ranks.sort_unstable();
        assert!(ranks.iter().enumerate().all(|(i, &r)| i == r), "ranks are a permutation");
        // Blue-noise spacing: thresholding at 10% leaves points with
        // no tight pairs (toroidal min distance above 1 pixel).
        let pts: Vec<(usize, usize)> = (0..w * h)
            .filter(|&i| tex[i] < 0.1)
            .map(|i| (i % w, i / w))
            .collect();
        let mut min_d2 = usize::MAX;
        for i in 0..pts.len() {
            for j in i + 1..pts.len() {
                let dx = pts[i].0.abs_diff(pts[j].0).min(w - pts[i].0.abs_diff(pts[j].0));
                let dy = pts[i].1.abs_diff(pts[j].1).min(h - pts[i].1.abs_diff(pts[j].1));
                min_d2 = min_d2.min(dx * dx + dy * dy);
            }
        }
        assert!(min_d2 > 1, "thresholded blue noise avoids adjacent pairs ({min_d2})");
    }

    #[test]
    fn test_gabor_noise() {
        let kernels: Vec<GaborKernel> = (0..20)
            .map(|i| GaborKernel {
                center: Vec2::new((i % 5) as f64, (i / 5) as f64),
                frequency: 2.0,
                orientation: i as f64,
                bandwidth: 1.0,
                amplitude: 0.5,
                phase: 0.0,
            })
            .collect();
        // At a kernel center with phase 0 the kernel contributes its
        // full amplitude.
        let solo = [kernels[0]];
        assert!((gabor_noise_2d(0.0, 0.0, &solo) - 0.5).abs() < 1e-12);
        let mut rng = Rng::new(12);
        for _ in 0..500 {
            let v = gabor_noise_2d(rng.next_f64() * 5.0, rng.next_f64() * 4.0, &kernels);
            assert!(v.is_finite());
        }
        // Far from every kernel the envelope kills the noise.
        assert!(gabor_noise_2d(100.0, 100.0, &kernels).abs() < 1e-12);
    }
}
