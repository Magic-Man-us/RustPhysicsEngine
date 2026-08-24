//! Uniform-grid scalar fields.
//!
//! Minimal backfill of the Part 2 `ScalarField2`/`ScalarField3` types
//! that later roadmap phases build on: row-major storage with grid
//! spacing and bilinear/trilinear sampling.

/// 2D scalar field on an nx×ny uniform grid with spacing dx
/// (row-major: index = y·nx + x; physical position of node (i, j) is
/// (i·dx, j·dx)).
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarField2 {
    pub nx: usize,
    pub ny: usize,
    pub dx: f64,
    pub data: Vec<f64>,
}

impl ScalarField2 {
    /// Zero-filled field.
    #[must_use]
    pub fn new(nx: usize, ny: usize, dx: f64) -> Self {
        Self { nx, ny, dx, data: vec![0.0; nx * ny] }
    }

    /// Build from a function of the node position (x, y).
    #[must_use]
    pub fn from_fn(nx: usize, ny: usize, dx: f64, f: impl Fn(f64, f64) -> f64) -> Self {
        let mut data = Vec::with_capacity(nx * ny);
        for j in 0..ny {
            for i in 0..nx {
                data.push(f(i as f64 * dx, j as f64 * dx));
            }
        }
        Self { nx, ny, dx, data }
    }

    /// Node value.
    #[must_use]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[j * self.nx + i]
    }

    /// Set a node value.
    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[j * self.nx + i] = v;
    }

    /// Bilinear sample at a physical position (clamped to the grid).
    #[must_use]
    pub fn sample(&self, x: f64, y: f64) -> f64 {
        let fx = (x / self.dx).clamp(0.0, (self.nx - 1) as f64);
        let fy = (y / self.dx).clamp(0.0, (self.ny - 1) as f64);
        let i0 = fx.floor() as usize;
        let j0 = fy.floor() as usize;
        let i1 = (i0 + 1).min(self.nx - 1);
        let j1 = (j0 + 1).min(self.ny - 1);
        let tx = fx - i0 as f64;
        let ty = fy - j0 as f64;
        self.get(i0, j0) * (1.0 - tx) * (1.0 - ty)
            + self.get(i1, j0) * tx * (1.0 - ty)
            + self.get(i0, j1) * (1.0 - tx) * ty
            + self.get(i1, j1) * tx * ty
    }

    /// Smallest and largest node values.
    #[must_use]
    pub fn min_max(&self) -> (f64, f64) {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for &v in &self.data {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        (lo, hi)
    }
}

/// 3D scalar field on an nx×ny×nz uniform grid with spacing dx
/// (index = (k·ny + j)·nx + i).
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarField3 {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f64,
    pub data: Vec<f64>,
}

impl ScalarField3 {
    /// Zero-filled field.
    #[must_use]
    pub fn new(nx: usize, ny: usize, nz: usize, dx: f64) -> Self {
        Self { nx, ny, nz, dx, data: vec![0.0; nx * ny * nz] }
    }

    /// Node value.
    #[must_use]
    pub fn get(&self, i: usize, j: usize, k: usize) -> f64 {
        self.data[(k * self.ny + j) * self.nx + i]
    }

    /// Set a node value.
    pub fn set(&mut self, i: usize, j: usize, k: usize, v: f64) {
        self.data[(k * self.ny + j) * self.nx + i] = v;
    }

    /// Trilinear sample at a physical position (clamped to the grid).
    #[must_use]
    pub fn sample(&self, x: f64, y: f64, z: f64) -> f64 {
        let fx = (x / self.dx).clamp(0.0, (self.nx - 1) as f64);
        let fy = (y / self.dx).clamp(0.0, (self.ny - 1) as f64);
        let fz = (z / self.dx).clamp(0.0, (self.nz - 1) as f64);
        let i0 = fx.floor() as usize;
        let j0 = fy.floor() as usize;
        let k0 = fz.floor() as usize;
        let i1 = (i0 + 1).min(self.nx - 1);
        let j1 = (j0 + 1).min(self.ny - 1);
        let k1 = (k0 + 1).min(self.nz - 1);
        let tx = fx - i0 as f64;
        let ty = fy - j0 as f64;
        let tz = fz - k0 as f64;
        let c00 = self.get(i0, j0, k0) * (1.0 - tx) + self.get(i1, j0, k0) * tx;
        let c10 = self.get(i0, j1, k0) * (1.0 - tx) + self.get(i1, j1, k0) * tx;
        let c01 = self.get(i0, j0, k1) * (1.0 - tx) + self.get(i1, j0, k1) * tx;
        let c11 = self.get(i0, j1, k1) * (1.0 - tx) + self.get(i1, j1, k1) * tx;
        let c0 = c00 * (1.0 - ty) + c10 * ty;
        let c1 = c01 * (1.0 - ty) + c11 * ty;
        c0 * (1.0 - tz) + c1 * tz
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_field2_sampling() {
        let f = ScalarField2::from_fn(11, 11, 0.1, |x, y| 2.0 * x + 3.0 * y);
        assert!((f.get(5, 5) - 2.5).abs() < 1e-12);
        // Bilinear reproduces linear functions between nodes.
        assert!((f.sample(0.33, 0.47) - (0.66 + 1.41)).abs() < 1e-12);
        let (lo, hi) = f.min_max();
        assert!((lo - 0.0).abs() < 1e-12 && (hi - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_scalar_field2_new_is_zero_and_set_writes_one_node() {
        let mut f = ScalarField2::new(5, 4, 0.25);
        assert_eq!(f.nx, 5);
        assert_eq!(f.ny, 4);
        assert_eq!(f.dx, 0.25);
        assert_eq!(f.data.len(), 20);
        assert!(f.data.iter().all(|&v| v == 0.0), "new() is zero-filled");
        assert_eq!(f.min_max(), (0.0, 0.0));
        // set writes exactly one node and leaves the rest untouched.
        f.set(3, 2, -4.5);
        assert_eq!(f.get(3, 2), -4.5);
        assert_eq!(f.data[2 * 5 + 3], -4.5, "row-major index j*nx + i");
        assert_eq!(f.data.iter().filter(|&&v| v != 0.0).count(), 1);
        assert_eq!(f.min_max(), (-4.5, 0.0));
        // Sampling at that node returns the stored value; halfway to a
        // zero neighbour is half of it (bilinear on a linear segment).
        assert!((f.sample(3.0 * 0.25, 2.0 * 0.25) - (-4.5)).abs() < 1e-12);
        assert!((f.sample(3.5 * 0.25, 2.0 * 0.25) - (-2.25)).abs() < 1e-12);
    }

    #[test]
    fn test_scalar_field2_set_linear_field_has_constant_gradient() {
        // Build g(x, y) = 3x - 2y node by node with set(), then check
        // the central-difference gradient is exactly (3, -2) at every
        // interior node (central differences are exact for linear data).
        let (nx, ny, dx) = (9usize, 7usize, 0.5_f64);
        let mut f = ScalarField2::new(nx, ny, dx);
        for j in 0..ny {
            for i in 0..nx {
                f.set(i, j, 3.0 * (i as f64 * dx) - 2.0 * (j as f64 * dx));
            }
        }
        // set() must agree with the equivalent from_fn construction.
        let reference = ScalarField2::from_fn(nx, ny, dx, |x, y| 3.0 * x - 2.0 * y);
        assert_eq!(f, reference);
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let gx = (f.get(i + 1, j) - f.get(i - 1, j)) / (2.0 * dx);
                let gy = (f.get(i, j + 1) - f.get(i, j - 1)) / (2.0 * dx);
                assert!((gx - 3.0).abs() < 1e-12, "d/dx at ({i}, {j}) = {gx}");
                assert!((gy + 2.0).abs() < 1e-12, "d/dy at ({i}, {j}) = {gy}");
                // The 5-point Laplacian of a linear field vanishes.
                let lap = (f.get(i + 1, j) + f.get(i - 1, j) + f.get(i, j + 1)
                    + f.get(i, j - 1)
                    - 4.0 * f.get(i, j))
                    / (dx * dx);
                assert!(lap.abs() < 1e-11, "laplacian at ({i}, {j}) = {lap}");
            }
        }
        let (lo, hi) = f.min_max();
        assert!((lo - (-2.0 * (ny - 1) as f64 * dx)).abs() < 1e-12);
        assert!((hi - 3.0 * (nx - 1) as f64 * dx).abs() < 1e-12);
    }

    #[test]
    fn test_scalar_field3_sampling() {
        let mut f = ScalarField3::new(4, 4, 4, 1.0);
        f.set(1, 2, 3, 7.0);
        assert_eq!(f.get(1, 2, 3), 7.0);
        // Trilinear at the node itself.
        assert!((f.sample(1.0, 2.0, 3.0) - 7.0).abs() < 1e-12);
        // Halfway to a zero neighbor.
        assert!((f.sample(1.5, 2.0, 3.0) - 3.5).abs() < 1e-12);
    }
}
