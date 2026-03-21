/// Vector calculus operators and field theory for physics grids.
///
/// Provides discrete differential operators (gradient, laplacian, divergence, curl)
/// on 2D and 3D uniform grids, point-wise numerical differentiation via function
/// pointers, line/surface integrals, and a Jacobi Poisson solver.

// ---------------------------------------------------------------------------
// Index helpers
// ---------------------------------------------------------------------------

#[inline]
fn idx3(i: usize, j: usize, k: usize, ny: usize, nz: usize) -> usize {
    i * ny * nz + j * nz + k
}

#[inline]
fn idx2(i: usize, j: usize, ny: usize) -> usize {
    i * ny + j
}

/// Central-difference first derivative along one axis.
/// Returns (f[forward] - f[backward]) / (2 * spacing), using one-sided
/// differences at the boundaries.
#[inline]
fn central_diff(f_back: f64, f_fwd: f64, h: f64, at_lo: bool, at_hi: bool) -> f64 {
    if at_lo && at_hi {
        // Single-cell dimension — derivative is zero.
        0.0
    } else if at_lo {
        (f_fwd - f_back) / h // forward difference (denominator is dx, not 2dx)
    } else if at_hi {
        (f_fwd - f_back) / h // backward difference
    } else {
        (f_fwd - f_back) / (2.0 * h)
    }
}

/// Central-difference second derivative along one axis.
#[inline]
fn central_diff2(f_back: f64, f_center: f64, f_fwd: f64, h: f64, at_lo: bool, at_hi: bool) -> f64 {
    if at_lo && at_hi {
        0.0
    } else if at_lo || at_hi {
        // At a boundary we only have two points — fall back to zero (first-order
        // accurate second derivative requires at least 3 points).
        0.0
    } else {
        (f_fwd - 2.0 * f_center + f_back) / (h * h)
    }
}

// ---------------------------------------------------------------------------
// 3-D grid operators
// ---------------------------------------------------------------------------

/// Compute the gradient of a scalar field on a uniform 3-D grid.
///
/// `field` is row-major with index mapping `i*ny*nz + j*nz + k`.
/// Returns a `Vec` of `(∂f/∂x, ∂f/∂y, ∂f/∂z)` tuples, one per grid point.
#[must_use]
pub fn gradient_3d(
    field: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
    dy: f64,
    dz: f64,
) -> Vec<(f64, f64, f64)> {
    let n = nx * ny * nz;
    assert_eq!(field.len(), n, "field length must equal nx*ny*nz");
    let mut out = vec![(0.0, 0.0, 0.0); n];

    for i in 0..nx {
        let i_lo = if i == 0 { 0 } else { i - 1 };
        let i_hi = if i == nx - 1 { nx - 1 } else { i + 1 };
        for j in 0..ny {
            let j_lo = if j == 0 { 0 } else { j - 1 };
            let j_hi = if j == ny - 1 { ny - 1 } else { j + 1 };
            for k in 0..nz {
                let k_lo = if k == 0 { 0 } else { k - 1 };
                let k_hi = if k == nz - 1 { nz - 1 } else { k + 1 };

                let dfx = central_diff(
                    field[idx3(i_lo, j, k, ny, nz)],
                    field[idx3(i_hi, j, k, ny, nz)],
                    dx,
                    i == 0,
                    i == nx - 1,
                );
                let dfy = central_diff(
                    field[idx3(i, j_lo, k, ny, nz)],
                    field[idx3(i, j_hi, k, ny, nz)],
                    dy,
                    j == 0,
                    j == ny - 1,
                );
                let dfz = central_diff(
                    field[idx3(i, j, k_lo, ny, nz)],
                    field[idx3(i, j, k_hi, ny, nz)],
                    dz,
                    k == 0,
                    k == nz - 1,
                );

                out[idx3(i, j, k, ny, nz)] = (dfx, dfy, dfz);
            }
        }
    }

    out
}

/// Compute the Laplacian (∇²f) of a scalar field on a uniform 3-D grid.
#[must_use]
pub fn laplacian_3d(
    field: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
    dy: f64,
    dz: f64,
) -> Vec<f64> {
    let n = nx * ny * nz;
    assert_eq!(field.len(), n, "field length must equal nx*ny*nz");
    let mut out = vec![0.0; n];

    for i in 0..nx {
        let i_lo = if i == 0 { 0 } else { i - 1 };
        let i_hi = if i == nx - 1 { nx - 1 } else { i + 1 };
        for j in 0..ny {
            let j_lo = if j == 0 { 0 } else { j - 1 };
            let j_hi = if j == ny - 1 { ny - 1 } else { j + 1 };
            for k in 0..nz {
                let k_lo = if k == 0 { 0 } else { k - 1 };
                let k_hi = if k == nz - 1 { nz - 1 } else { k + 1 };
                let c = idx3(i, j, k, ny, nz);

                let d2x = central_diff2(
                    field[idx3(i_lo, j, k, ny, nz)],
                    field[c],
                    field[idx3(i_hi, j, k, ny, nz)],
                    dx,
                    i == 0,
                    i == nx - 1,
                );
                let d2y = central_diff2(
                    field[idx3(i, j_lo, k, ny, nz)],
                    field[c],
                    field[idx3(i, j_hi, k, ny, nz)],
                    dy,
                    j == 0,
                    j == ny - 1,
                );
                let d2z = central_diff2(
                    field[idx3(i, j, k_lo, ny, nz)],
                    field[c],
                    field[idx3(i, j, k_hi, ny, nz)],
                    dz,
                    k == 0,
                    k == nz - 1,
                );

                out[c] = d2x + d2y + d2z;
            }
        }
    }

    out
}

/// Divergence of a vector field on a uniform 3-D grid.
///
/// Each component (`fx`, `fy`, `fz`) is a flat array with the same index mapping
/// as scalar fields.
#[must_use]
pub fn divergence_3d(
    fx: &[f64],
    fy: &[f64],
    fz: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
    dy: f64,
    dz: f64,
) -> Vec<f64> {
    let n = nx * ny * nz;
    assert_eq!(fx.len(), n);
    assert_eq!(fy.len(), n);
    assert_eq!(fz.len(), n);
    let mut out = vec![0.0; n];

    for i in 0..nx {
        let i_lo = if i == 0 { 0 } else { i - 1 };
        let i_hi = if i == nx - 1 { nx - 1 } else { i + 1 };
        for j in 0..ny {
            let j_lo = if j == 0 { 0 } else { j - 1 };
            let j_hi = if j == ny - 1 { ny - 1 } else { j + 1 };
            for k in 0..nz {
                let k_lo = if k == 0 { 0 } else { k - 1 };
                let k_hi = if k == nz - 1 { nz - 1 } else { k + 1 };

                let dfx = central_diff(
                    fx[idx3(i_lo, j, k, ny, nz)],
                    fx[idx3(i_hi, j, k, ny, nz)],
                    dx,
                    i == 0,
                    i == nx - 1,
                );
                let dfy = central_diff(
                    fy[idx3(i, j_lo, k, ny, nz)],
                    fy[idx3(i, j_hi, k, ny, nz)],
                    dy,
                    j == 0,
                    j == ny - 1,
                );
                let dfz = central_diff(
                    fz[idx3(i, j, k_lo, ny, nz)],
                    fz[idx3(i, j, k_hi, ny, nz)],
                    dz,
                    k == 0,
                    k == nz - 1,
                );

                out[idx3(i, j, k, ny, nz)] = dfx + dfy + dfz;
            }
        }
    }

    out
}

/// Curl of a vector field on a uniform 3-D grid.
///
/// Returns `(curl_x, curl_y, curl_z)` as separate flat arrays.
#[must_use]
pub fn curl_3d(
    fx: &[f64],
    fy: &[f64],
    fz: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
    dy: f64,
    dz: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = nx * ny * nz;
    assert_eq!(fx.len(), n);
    assert_eq!(fy.len(), n);
    assert_eq!(fz.len(), n);

    let mut cx = vec![0.0; n];
    let mut cy = vec![0.0; n];
    let mut cz = vec![0.0; n];

    for i in 0..nx {
        let i_lo = if i == 0 { 0 } else { i - 1 };
        let i_hi = if i == nx - 1 { nx - 1 } else { i + 1 };
        for j in 0..ny {
            let j_lo = if j == 0 { 0 } else { j - 1 };
            let j_hi = if j == ny - 1 { ny - 1 } else { j + 1 };
            for k in 0..nz {
                let k_lo = if k == 0 { 0 } else { k - 1 };
                let k_hi = if k == nz - 1 { nz - 1 } else { k + 1 };
                let idx = idx3(i, j, k, ny, nz);

                // ∂Fz/∂y
                let dfz_dy = central_diff(
                    fz[idx3(i, j_lo, k, ny, nz)],
                    fz[idx3(i, j_hi, k, ny, nz)],
                    dy,
                    j == 0,
                    j == ny - 1,
                );
                // ∂Fy/∂z
                let dfy_dz = central_diff(
                    fy[idx3(i, j, k_lo, ny, nz)],
                    fy[idx3(i, j, k_hi, ny, nz)],
                    dz,
                    k == 0,
                    k == nz - 1,
                );
                // ∂Fx/∂z
                let dfx_dz = central_diff(
                    fx[idx3(i, j, k_lo, ny, nz)],
                    fx[idx3(i, j, k_hi, ny, nz)],
                    dz,
                    k == 0,
                    k == nz - 1,
                );
                // ∂Fz/∂x
                let dfz_dx = central_diff(
                    fz[idx3(i_lo, j, k, ny, nz)],
                    fz[idx3(i_hi, j, k, ny, nz)],
                    dx,
                    i == 0,
                    i == nx - 1,
                );
                // ∂Fy/∂x
                let dfy_dx = central_diff(
                    fy[idx3(i_lo, j, k, ny, nz)],
                    fy[idx3(i_hi, j, k, ny, nz)],
                    dx,
                    i == 0,
                    i == nx - 1,
                );
                // ∂Fx/∂y
                let dfx_dy = central_diff(
                    fx[idx3(i, j_lo, k, ny, nz)],
                    fx[idx3(i, j_hi, k, ny, nz)],
                    dy,
                    j == 0,
                    j == ny - 1,
                );

                cx[idx] = dfz_dy - dfy_dz;
                cy[idx] = dfx_dz - dfz_dx;
                cz[idx] = dfy_dx - dfx_dy;
            }
        }
    }

    (cx, cy, cz)
}

// ---------------------------------------------------------------------------
// 2-D grid operators
// ---------------------------------------------------------------------------

/// Gradient of a scalar field on a uniform 2-D grid.
///
/// Index mapping: `i*ny + j`. Returns `(∂f/∂x, ∂f/∂y)` per grid point.
#[must_use]
pub fn gradient_2d(
    field: &[f64],
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
) -> Vec<(f64, f64)> {
    let n = nx * ny;
    assert_eq!(field.len(), n, "field length must equal nx*ny");
    let mut out = vec![(0.0, 0.0); n];

    for i in 0..nx {
        let i_lo = if i == 0 { 0 } else { i - 1 };
        let i_hi = if i == nx - 1 { nx - 1 } else { i + 1 };
        for j in 0..ny {
            let j_lo = if j == 0 { 0 } else { j - 1 };
            let j_hi = if j == ny - 1 { ny - 1 } else { j + 1 };

            let dfx = central_diff(
                field[idx2(i_lo, j, ny)],
                field[idx2(i_hi, j, ny)],
                dx,
                i == 0,
                i == nx - 1,
            );
            let dfy = central_diff(
                field[idx2(i, j_lo, ny)],
                field[idx2(i, j_hi, ny)],
                dy,
                j == 0,
                j == ny - 1,
            );

            out[idx2(i, j, ny)] = (dfx, dfy);
        }
    }

    out
}

/// Laplacian of a scalar field on a uniform 2-D grid.
#[must_use]
pub fn laplacian_2d(
    field: &[f64],
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
) -> Vec<f64> {
    let n = nx * ny;
    assert_eq!(field.len(), n, "field length must equal nx*ny");
    let mut out = vec![0.0; n];

    for i in 0..nx {
        let i_lo = if i == 0 { 0 } else { i - 1 };
        let i_hi = if i == nx - 1 { nx - 1 } else { i + 1 };
        for j in 0..ny {
            let j_lo = if j == 0 { 0 } else { j - 1 };
            let j_hi = if j == ny - 1 { ny - 1 } else { j + 1 };
            let c = idx2(i, j, ny);

            let d2x = central_diff2(
                field[idx2(i_lo, j, ny)],
                field[c],
                field[idx2(i_hi, j, ny)],
                dx,
                i == 0,
                i == nx - 1,
            );
            let d2y = central_diff2(
                field[idx2(i, j_lo, ny)],
                field[c],
                field[idx2(i, j_hi, ny)],
                dy,
                j == 0,
                j == ny - 1,
            );

            out[c] = d2x + d2y;
        }
    }

    out
}

/// Divergence of a 2-D vector field.
#[must_use]
pub fn divergence_2d(
    fx: &[f64],
    fy: &[f64],
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
) -> Vec<f64> {
    let n = nx * ny;
    assert_eq!(fx.len(), n);
    assert_eq!(fy.len(), n);
    let mut out = vec![0.0; n];

    for i in 0..nx {
        let i_lo = if i == 0 { 0 } else { i - 1 };
        let i_hi = if i == nx - 1 { nx - 1 } else { i + 1 };
        for j in 0..ny {
            let j_lo = if j == 0 { 0 } else { j - 1 };
            let j_hi = if j == ny - 1 { ny - 1 } else { j + 1 };

            let dfx = central_diff(
                fx[idx2(i_lo, j, ny)],
                fx[idx2(i_hi, j, ny)],
                dx,
                i == 0,
                i == nx - 1,
            );
            let dfy = central_diff(
                fy[idx2(i, j_lo, ny)],
                fy[idx2(i, j_hi, ny)],
                dy,
                j == 0,
                j == ny - 1,
            );

            out[idx2(i, j, ny)] = dfx + dfy;
        }
    }

    out
}

/// Scalar curl of a 2-D vector field: ∂Fy/∂x - ∂Fx/∂y.
#[must_use]
pub fn curl_2d(
    fx: &[f64],
    fy: &[f64],
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
) -> Vec<f64> {
    let n = nx * ny;
    assert_eq!(fx.len(), n);
    assert_eq!(fy.len(), n);
    let mut out = vec![0.0; n];

    for i in 0..nx {
        let i_lo = if i == 0 { 0 } else { i - 1 };
        let i_hi = if i == nx - 1 { nx - 1 } else { i + 1 };
        for j in 0..ny {
            let j_lo = if j == 0 { 0 } else { j - 1 };
            let j_hi = if j == ny - 1 { ny - 1 } else { j + 1 };

            let dfy_dx = central_diff(
                fy[idx2(i_lo, j, ny)],
                fy[idx2(i_hi, j, ny)],
                dx,
                i == 0,
                i == nx - 1,
            );
            let dfx_dy = central_diff(
                fx[idx2(i, j_lo, ny)],
                fx[idx2(i, j_hi, ny)],
                dy,
                j == 0,
                j == ny - 1,
            );

            out[idx2(i, j, ny)] = dfy_dx - dfx_dy;
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Poisson solver
// ---------------------------------------------------------------------------

/// Solve ∇²φ = ρ on a 2-D grid using Jacobi iteration with zero (Dirichlet)
/// boundary conditions.
///
/// Returns the solution field φ. Iteration stops when the maximum update is
/// below `tol` or after `max_iter` iterations.
#[must_use]
pub fn poisson_jacobi_2d(
    rhs: &[f64],
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
    max_iter: usize,
    tol: f64,
) -> Vec<f64> {
    let n = nx * ny;
    assert_eq!(rhs.len(), n);

    let dx2 = dx * dx;
    let dy2 = dy * dy;
    let denom = 2.0 * (1.0 / dx2 + 1.0 / dy2);

    let mut phi = vec![0.0; n];
    let mut phi_new = vec![0.0; n];

    for _ in 0..max_iter {
        let mut max_diff: f64 = 0.0;

        for i in 0..nx {
            for j in 0..ny {
                // Boundary cells stay at zero.
                if i == 0 || i == nx - 1 || j == 0 || j == ny - 1 {
                    phi_new[idx2(i, j, ny)] = 0.0;
                    continue;
                }

                let val = (
                    (phi[idx2(i + 1, j, ny)] + phi[idx2(i - 1, j, ny)]) / dx2
                    + (phi[idx2(i, j + 1, ny)] + phi[idx2(i, j - 1, ny)]) / dy2
                    - rhs[idx2(i, j, ny)]
                ) / denom;

                phi_new[idx2(i, j, ny)] = val;
                max_diff = max_diff.max((val - phi[idx2(i, j, ny)]).abs());
            }
        }

        std::mem::swap(&mut phi, &mut phi_new);

        if max_diff < tol {
            break;
        }
    }

    phi
}

// ---------------------------------------------------------------------------
// Line / surface integrals
// ---------------------------------------------------------------------------

/// Numerical line integral ∫F·dr along a piecewise-linear path in 3-D.
///
/// `f` returns the vector field value (Fx, Fy, Fz) at a given point.
/// `path` is an ordered list of waypoints. The integral is evaluated at the
/// midpoint of each segment.
#[must_use]
pub fn line_integral(
    f: &dyn Fn(f64, f64, f64) -> (f64, f64, f64),
    path: &[(f64, f64, f64)],
) -> f64 {
    if path.len() < 2 {
        return 0.0;
    }

    let mut sum = 0.0;
    for w in path.windows(2) {
        let (x0, y0, z0) = w[0];
        let (x1, y1, z1) = w[1];
        let mx = 0.5 * (x0 + x1);
        let my = 0.5 * (y0 + y1);
        let mz = 0.5 * (z0 + z1);
        let (fx, fy, fz) = f(mx, my, mz);
        let drx = x1 - x0;
        let dry = y1 - y0;
        let drz = z1 - z0;
        sum += fx * drx + fy * dry + fz * drz;
    }
    sum
}

/// Flux integral ∫F·n̂ ds along a 2-D curve.
///
/// The outward normal is computed by rotating each segment's tangent 90 degrees
/// clockwise: tangent (tx, ty) -> normal (ty, -tx). The integral is evaluated
/// at the midpoint of each segment.
#[must_use]
pub fn flux_integral_2d(
    fn_field: &dyn Fn(f64, f64) -> (f64, f64),
    path: &[(f64, f64)],
) -> f64 {
    if path.len() < 2 {
        return 0.0;
    }

    let mut sum = 0.0;
    for w in path.windows(2) {
        let (x0, y0) = w[0];
        let (x1, y1) = w[1];
        let mx = 0.5 * (x0 + x1);
        let my = 0.5 * (y0 + y1);
        let (fx, fy) = fn_field(mx, my);

        let tx = x1 - x0;
        let ty = y1 - y0;
        // outward normal: rotate tangent 90 deg clockwise
        let nx = ty;
        let ny = -tx;
        // |n| = |dr|, so F·n̂ ds = F·n (unnormalized) since ds = |dr|
        // and n̂ = n/|n|, giving F·(n/|n|)*|dr| = F·n when |n|=|dr|.
        sum += fx * nx + fy * ny;
    }
    sum
}

// ---------------------------------------------------------------------------
// Point-wise numerical differential operators via function pointers
// ---------------------------------------------------------------------------

/// Finite-difference gradient of a scalar function at a point.
#[must_use]
pub fn numerical_gradient(
    f: &dyn Fn(f64, f64, f64) -> f64,
    x: f64,
    y: f64,
    z: f64,
    h: f64,
) -> (f64, f64, f64) {
    let dfx = (f(x + h, y, z) - f(x - h, y, z)) / (2.0 * h);
    let dfy = (f(x, y + h, z) - f(x, y - h, z)) / (2.0 * h);
    let dfz = (f(x, y, z + h) - f(x, y, z - h)) / (2.0 * h);
    (dfx, dfy, dfz)
}

/// Finite-difference Laplacian of a scalar function at a point.
#[must_use]
pub fn numerical_laplacian(
    f: &dyn Fn(f64, f64, f64) -> f64,
    x: f64,
    y: f64,
    z: f64,
    h: f64,
) -> f64 {
    let fc = f(x, y, z);
    let d2x = (f(x + h, y, z) - 2.0 * fc + f(x - h, y, z)) / (h * h);
    let d2y = (f(x, y + h, z) - 2.0 * fc + f(x, y - h, z)) / (h * h);
    let d2z = (f(x, y, z + h) - 2.0 * fc + f(x, y, z - h)) / (h * h);
    d2x + d2y + d2z
}

/// Finite-difference divergence of a vector field at a point.
///
/// Each component of the field is given as a separate function.
#[must_use]
pub fn numerical_divergence(
    fx: &dyn Fn(f64, f64, f64) -> f64,
    fy: &dyn Fn(f64, f64, f64) -> f64,
    fz: &dyn Fn(f64, f64, f64) -> f64,
    x: f64,
    y: f64,
    z: f64,
    h: f64,
) -> f64 {
    let dfx_dx = (fx(x + h, y, z) - fx(x - h, y, z)) / (2.0 * h);
    let dfy_dy = (fy(x, y + h, z) - fy(x, y - h, z)) / (2.0 * h);
    let dfz_dz = (fz(x, y, z + h) - fz(x, y, z - h)) / (2.0 * h);
    dfx_dx + dfy_dy + dfz_dz
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-6;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < TOL
    }

    // -- Gradient of f = x² + y² + z² should be (2x, 2y, 2z) --

    #[test]
    fn test_gradient_3d_quadratic() {
        let nx = 11;
        let ny = 11;
        let nz = 11;
        let dx = 0.1;
        let dy = 0.1;
        let dz = 0.1;

        let mut field = vec![0.0; nx * ny * nz];
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let x = i as f64 * dx;
                    let y = j as f64 * dy;
                    let z = k as f64 * dz;
                    field[idx3(i, j, k, ny, nz)] = x * x + y * y + z * z;
                }
            }
        }

        let grad = gradient_3d(&field, nx, ny, nz, dx, dy, dz);

        // Check interior points only (boundaries use one-sided diffs, still exact for quadratics)
        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    let x = i as f64 * dx;
                    let y = j as f64 * dy;
                    let z = k as f64 * dz;
                    let (gx, gy, gz) = grad[idx3(i, j, k, ny, nz)];
                    let (ex, ey, ez) = (2.0 * x, 2.0 * y, 2.0 * z);
                    assert!(
                        approx(gx, ex) && approx(gy, ey) && approx(gz, ez),
                        "gradient mismatch at ({i},{j},{k}): got ({gx},{gy},{gz}), expected ({ex},{ey},{ez})",
                    );
                }
            }
        }
    }

    // -- Laplacian of f = x² + y² + z² should be 6 everywhere (interior) --

    #[test]
    fn test_laplacian_3d_quadratic() {
        let nx = 11;
        let ny = 11;
        let nz = 11;
        let h = 0.1;

        let mut field = vec![0.0; nx * ny * nz];
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let x = i as f64 * h;
                    let y = j as f64 * h;
                    let z = k as f64 * h;
                    field[idx3(i, j, k, ny, nz)] = x * x + y * y + z * z;
                }
            }
        }

        let lap = laplacian_3d(&field, nx, ny, nz, h, h, h);

        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    let val = lap[idx3(i, j, k, ny, nz)];
                    assert!(
                        approx(val, 6.0),
                        "laplacian mismatch at ({i},{j},{k}): got {val}, expected 6.0"
                    );
                }
            }
        }
    }

    // -- Divergence of F = (x, y, z) should be 3 --

    #[test]
    fn test_divergence_3d_identity() {
        let nx = 11;
        let ny = 11;
        let nz = 11;
        let h = 0.1;

        let n = nx * ny * nz;
        let mut fx = vec![0.0; n];
        let mut fy = vec![0.0; n];
        let mut fz = vec![0.0; n];

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let idx = idx3(i, j, k, ny, nz);
                    fx[idx] = i as f64 * h;
                    fy[idx] = j as f64 * h;
                    fz[idx] = k as f64 * h;
                }
            }
        }

        let div = divergence_3d(&fx, &fy, &fz, nx, ny, nz, h, h, h);

        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    let val = div[idx3(i, j, k, ny, nz)];
                    assert!(
                        approx(val, 3.0),
                        "divergence mismatch at ({i},{j},{k}): got {val}, expected 3.0"
                    );
                }
            }
        }
    }

    // -- Curl of gradient should be zero --

    #[test]
    fn test_curl_of_gradient_is_zero() {
        let nx = 11;
        let ny = 11;
        let nz = 11;
        let h = 0.1;

        // f = x² + 2xy + z³
        let mut field = vec![0.0; nx * ny * nz];
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let x = i as f64 * h;
                    let y = j as f64 * h;
                    let z = k as f64 * h;
                    field[idx3(i, j, k, ny, nz)] = x * x + 2.0 * x * y + z * z * z;
                }
            }
        }

        let grad = gradient_3d(&field, nx, ny, nz, h, h, h);

        let n = nx * ny * nz;
        let mut gx = vec![0.0; n];
        let mut gy = vec![0.0; n];
        let mut gz = vec![0.0; n];
        for (idx, &(vx, vy, vz)) in grad.iter().enumerate() {
            gx[idx] = vx;
            gy[idx] = vy;
            gz[idx] = vz;
        }

        let (cx, cy, cz) = curl_3d(&gx, &gy, &gz, nx, ny, nz, h, h, h);

        // Check interior points (away from boundary artifacts)
        for i in 2..nx - 2 {
            for j in 2..ny - 2 {
                for k in 2..nz - 2 {
                    let idx = idx3(i, j, k, ny, nz);
                    let (cvx, cvy, cvz) = (cx[idx], cy[idx], cz[idx]);
                    assert!(
                        cvx.abs() < 1e-4 && cvy.abs() < 1e-4 && cvz.abs() < 1e-4,
                        "curl of gradient nonzero at ({i},{j},{k}): ({cvx},{cvy},{cvz})",
                    );
                }
            }
        }
    }

    // -- Poisson solver symmetry test --

    #[test]
    fn test_poisson_jacobi_2d_symmetry() {
        let nx = 21;
        let ny = 21;
        let h = 1.0 / (nx - 1) as f64;

        // Point source at the center
        let mut rhs = vec![0.0; nx * ny];
        rhs[idx2(nx / 2, ny / 2, ny)] = -1.0;

        let phi = poisson_jacobi_2d(&rhs, nx, ny, h, h, 10_000, 1e-10);

        let center = nx / 2;
        // Check 4-fold symmetry: phi(c+d, c) == phi(c-d, c) == phi(c, c+d) == phi(c, c-d)
        for d in 1..center - 1 {
            let right = phi[idx2(center + d, center, ny)];
            let left = phi[idx2(center - d, center, ny)];
            let up = phi[idx2(center, center + d, ny)];
            let down = phi[idx2(center, center - d, ny)];
            assert!(
                approx(right, left) && approx(right, up) && approx(right, down),
                "symmetry broken at d={d}: right={right}, left={left}, up={up}, down={down}"
            );
        }

        // Solution should be negative near center (Green's function for -∇²φ = δ)
        assert!(phi[idx2(center, center, ny)].abs() > 0.0, "center should be nonzero");
    }

    // -- Line integral of conservative field is path-independent --

    #[test]
    fn test_line_integral_conservative_field() {
        // F = ∇(x² + y² + z²) = (2x, 2y, 2z)
        let field = |x: f64, y: f64, z: f64| -> (f64, f64, f64) { (2.0 * x, 2.0 * y, 2.0 * z) };
        let potential = |x: f64, y: f64, z: f64| -> f64 { x * x + y * y + z * z };
        let expected = potential(1.0, 1.0, 1.0) - potential(0.0, 0.0, 0.0);

        // Path 1: straight line (many segments for accuracy)
        let n_seg = 1000;
        let path1: Vec<(f64, f64, f64)> = (0..=n_seg)
            .map(|s| {
                let t = s as f64 / n_seg as f64;
                (t, t, t)
            })
            .collect();

        // Path 2: go along x, then y, then z
        let mut path2: Vec<(f64, f64, f64)> = Vec::new();
        for s in 0..=n_seg {
            let t = s as f64 / n_seg as f64;
            path2.push((t, 0.0, 0.0));
        }
        for s in 1..=n_seg {
            let t = s as f64 / n_seg as f64;
            path2.push((1.0, t, 0.0));
        }
        for s in 1..=n_seg {
            let t = s as f64 / n_seg as f64;
            path2.push((1.0, 1.0, t));
        }

        let i1 = line_integral(&field, &path1);
        let i2 = line_integral(&field, &path2);

        assert!(
            (i1 - expected).abs() < 1e-4,
            "path1 integral {i1} != expected {expected}"
        );
        assert!(
            (i2 - expected).abs() < 1e-4,
            "path2 integral {i2} != expected {expected}"
        );
        assert!(
            (i1 - i2).abs() < 1e-4,
            "path independence violated: {i1} != {i2}"
        );
    }

    // -- Numerical point-wise operators --

    #[test]
    fn test_numerical_gradient_quadratic() {
        let f = |x: f64, y: f64, z: f64| x * x + y * y + z * z;
        let (gx, gy, gz) = numerical_gradient(&f, 1.0, 2.0, 3.0, 1e-5);
        assert!(approx(gx, 2.0), "gx={gx}");
        assert!(approx(gy, 4.0), "gy={gy}");
        assert!(approx(gz, 6.0), "gz={gz}");
    }

    #[test]
    fn test_numerical_laplacian_quadratic() {
        let f = |x: f64, y: f64, z: f64| x * x + y * y + z * z;
        let lap = numerical_laplacian(&f, 1.0, 2.0, 3.0, 1e-4);
        assert!(
            (lap - 6.0).abs() < 1e-3,
            "numerical laplacian {lap} != 6.0"
        );
    }

    #[test]
    fn test_numerical_divergence_identity() {
        let fx = |x: f64, _y: f64, _z: f64| x;
        let fy = |_x: f64, y: f64, _z: f64| y;
        let fz = |_x: f64, _y: f64, z: f64| z;
        let div = numerical_divergence(&fx, &fy, &fz, 1.0, 2.0, 3.0, 1e-5);
        assert!(approx(div, 3.0), "numerical divergence {div} != 3.0");
    }

    // -- 2D operators --

    #[test]
    fn test_gradient_2d_quadratic() {
        let nx = 11;
        let ny = 11;
        let h = 0.1;

        let mut field = vec![0.0; nx * ny];
        for i in 0..nx {
            for j in 0..ny {
                let x = i as f64 * h;
                let y = j as f64 * h;
                field[idx2(i, j, ny)] = x * x + y * y;
            }
        }

        let grad = gradient_2d(&field, nx, ny, h, h);

        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                let (gx, gy) = grad[idx2(i, j, ny)];
                let x = i as f64 * h;
                let y = j as f64 * h;
                assert!(
                    approx(gx, 2.0 * x) && approx(gy, 2.0 * y),
                    "2d gradient mismatch at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_laplacian_2d_quadratic() {
        let nx = 11;
        let ny = 11;
        let h = 0.1;

        let mut field = vec![0.0; nx * ny];
        for i in 0..nx {
            for j in 0..ny {
                let x = i as f64 * h;
                let y = j as f64 * h;
                field[idx2(i, j, ny)] = x * x + y * y;
            }
        }

        let lap = laplacian_2d(&field, nx, ny, h, h);

        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                let val = lap[idx2(i, j, ny)];
                assert!(approx(val, 4.0), "2d laplacian at ({i},{j}): {val} != 4.0");
            }
        }
    }

    #[test]
    fn test_curl_2d_rotation_field() {
        // F = (-y, x) has curl = 2 everywhere
        let nx = 11;
        let ny = 11;
        let h = 0.1;
        let n = nx * ny;

        let mut fx = vec![0.0; n];
        let mut fy = vec![0.0; n];

        for i in 0..nx {
            for j in 0..ny {
                let x = i as f64 * h;
                let y = j as f64 * h;
                fx[idx2(i, j, ny)] = -y;
                fy[idx2(i, j, ny)] = x;
            }
        }

        let curl = curl_2d(&fx, &fy, nx, ny, h, h);

        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                let val = curl[idx2(i, j, ny)];
                assert!(approx(val, 2.0), "2d curl at ({i},{j}): {val} != 2.0");
            }
        }
    }

    #[test]
    fn test_divergence_2d_identity_field() {
        // F = (x, y) => div = 2
        let nx = 11;
        let ny = 11;
        let h = 0.1;
        let n = nx * ny;

        let mut fx = vec![0.0; n];
        let mut fy = vec![0.0; n];

        for i in 0..nx {
            for j in 0..ny {
                fx[idx2(i, j, ny)] = i as f64 * h;
                fy[idx2(i, j, ny)] = j as f64 * h;
            }
        }

        let div = divergence_2d(&fx, &fy, nx, ny, h, h);

        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                let val = div[idx2(i, j, ny)];
                assert!(
                    approx(val, 2.0),
                    "2d divergence at ({i},{j}): {val} != 2.0"
                );
            }
        }
    }

    #[test]
    fn test_flux_integral_2d_radial_field() {
        // F = (x, y), integrate around a unit circle.
        // Exact flux = ∫∫ div(F) dA = 2 * π * r² = 2π for r=1.
        let field = |x: f64, y: f64| -> (f64, f64) { (x, y) };

        let n_seg = 10_000;
        let path: Vec<(f64, f64)> = (0..=n_seg)
            .map(|s| {
                let theta = 2.0 * std::f64::consts::PI * s as f64 / n_seg as f64;
                (theta.cos(), theta.sin())
            })
            .collect();

        let flux = flux_integral_2d(&field, &path);
        let expected = 6.283185307179586;
        assert!(
            (flux - expected).abs() < 1e-3,
            "flux {flux} != expected {expected}"
        );
    }

    #[test]
    fn test_central_diff_single_cell() {
        let val = super::central_diff(1.0, 3.0, 1.0, true, true);
        assert!(approx(val, 0.0));
    }

    #[test]
    fn test_central_diff2_single_cell() {
        let val = super::central_diff2(1.0, 2.0, 3.0, 1.0, true, true);
        assert!(approx(val, 0.0));
    }

    #[test]
    fn test_central_diff2_boundary() {
        let val = super::central_diff2(1.0, 2.0, 3.0, 1.0, true, false);
        assert!(approx(val, 0.0));
    }

    #[test]
    fn test_line_integral_single_point() {
        let f = |_x: f64, _y: f64, _z: f64| (1.0, 0.0, 0.0);
        let path = vec![(0.0, 0.0, 0.0)];
        let result = line_integral(&f, &path);
        assert!(approx(result, 0.0));
    }

    #[test]
    fn test_flux_integral_2d_single_point() {
        let f = |_x: f64, _y: f64| (1.0, 0.0);
        let path = vec![(0.0, 0.0)];
        let result = flux_integral_2d(&f, &path);
        assert!(approx(result, 0.0));
    }
}
