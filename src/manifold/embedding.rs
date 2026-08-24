//! Manifold learning and dimensionality reduction: spectral embeddings
//! (MDS, Isomap, LLE, Laplacian eigenmaps, diffusion maps), PCA and kernel
//! PCA, stochastic neighbor embeddings, intrinsic-dimension estimators,
//! embedding quality metrics, benchmark datasets, and optimization on
//! matrix manifolds.

use crate::linalg::{eigen_symmetric, lu_decompose, svd, Matrix};
use crate::manifold::metric::Metric;
use crate::manifold::vecn::VecN;
use crate::monte_carlo::Rng;

const PI: f64 = std::f64::consts::PI;

/// Pairwise Euclidean distance matrix.
#[must_use]
pub fn dist_matrix(points: &[VecN]) -> Matrix {
    let n = points.len();
    Matrix::from_fn(n, n, |i, j| points[i].sub(&points[j]).norm())
}

fn top_eigen_coords(b: &Matrix, dim: usize, descending: bool) -> Vec<VecN> {
    let n = b.rows;
    let eig = eigen_symmetric(b, 1e-12, 300).expect("eigen failed");
    let mut idx: Vec<usize> = (0..n).collect();
    if descending {
        idx.sort_by(|&a, &c| eig.values[c].partial_cmp(&eig.values[a]).unwrap());
    } else {
        idx.sort_by(|&a, &c| eig.values[a].partial_cmp(&eig.values[c]).unwrap());
    }
    (0..n)
        .map(|i| {
            VecN::from(
                &(0..dim)
                    .map(|d| {
                        let k = idx[d];
                        let lam = eig.values[k].max(0.0).sqrt();
                        eig.vectors.get(i, k) * if descending { lam } else { 1.0 }
                    })
                    .collect::<Vec<f64>>(),
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Spectral embeddings
// ---------------------------------------------------------------------------

/// Classical (Torgerson) multidimensional scaling from a distance matrix.
#[must_use]
pub fn classical_mds(dist: &Matrix, dim: usize) -> Vec<VecN> {
    let n = dist.rows;
    // B = -1/2 J D^2 J
    let mut row_mean = vec![0.0; n];
    let mut total = 0.0;
    for (i, rm) in row_mean.iter_mut().enumerate() {
        for j in 0..n {
            let d2 = dist.get(i, j).powi(2);
            *rm += d2 / n as f64;
            total += d2 / (n * n) as f64;
        }
    }
    let b = Matrix::from_fn(n, n, |i, j| {
        -0.5 * (dist.get(i, j).powi(2) - row_mean[i] - row_mean[j] + total)
    });
    top_eigen_coords(&b, dim, true)
}

/// Metric MDS by SMACOF stress majorization. Returns (embedding, stress).
#[must_use]
pub fn metric_mds_smacof(
    dist: &Matrix,
    dim: usize,
    iters: usize,
    rng: &mut Rng,
) -> (Vec<VecN>, f64) {
    let n = dist.rows;
    let mut x: Vec<VecN> = (0..n)
        .map(|_| VecN::random_gaussian(dim, rng).scale(0.1))
        .collect();
    // seed with classical MDS for stability
    let seed = classical_mds(dist, dim);
    if seed.iter().all(|p| p.data.iter().all(|v| v.is_finite())) {
        x = seed;
    }
    let mut stress = 0.0;
    for _ in 0..iters {
        // Guttman transform with uniform weights
        let mut x_new = vec![VecN::zeros(dim); n];
        for i in 0..n {
            let mut acc = VecN::zeros(dim);
            for j in 0..n {
                if i == j {
                    continue;
                }
                let dij = x[i].sub(&x[j]).norm().max(1e-12);
                let ratio = dist.get(i, j) / dij;
                acc = acc.add(&x[i].sub(&x[j]).scale(ratio).add(&x[j]));
            }
            x_new[i] = acc.scale(1.0 / (n as f64 - 1.0));
        }
        x = x_new;
        stress = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                stress += (x[i].sub(&x[j]).norm() - dist.get(i, j)).powi(2);
            }
        }
    }
    (x, stress)
}

/// Nonmetric MDS: SMACOF against monotone-regressed disparities.
#[must_use]
pub fn nonmetric_mds(dist: &Matrix, dim: usize, iters: usize, rng: &mut Rng) -> Vec<VecN> {
    let n = dist.rows;
    let (mut x, _) = metric_mds_smacof(dist, dim, 10, rng);
    // pairs sorted by dissimilarity
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            pairs.push((i, j));
        }
    }
    pairs.sort_by(|&(a, b), &(c, d)| dist.get(a, b).partial_cmp(&dist.get(c, d)).unwrap());
    for _ in 0..iters {
        // current embedded distances in dissimilarity order
        let d_emb: Vec<f64> = pairs.iter().map(|&(i, j)| x[i].sub(&x[j]).norm()).collect();
        // isotonic regression (pool adjacent violators)
        let mut disp = d_emb.clone();
        let mut blocks: Vec<(f64, usize)> = Vec::new();
        for &v in &disp {
            blocks.push((v, 1));
            while blocks.len() >= 2 {
                let (v2, c2) = blocks[blocks.len() - 1];
                let (v1, c1) = blocks[blocks.len() - 2];
                if v1 / c1 as f64 > v2 / c2 as f64 {
                    blocks.pop();
                    blocks.pop();
                    blocks.push((v1 + v2, c1 + c2));
                } else {
                    break;
                }
            }
        }
        let mut k = 0;
        for &(v, c) in &blocks {
            for _ in 0..c {
                disp[k] = v / c as f64;
                k += 1;
            }
        }
        // one Guttman step toward the disparities
        let mut disp_m = Matrix::zeros(n, n);
        for (p, &(i, j)) in pairs.iter().enumerate() {
            disp_m.set(i, j, disp[p]);
            disp_m.set(j, i, disp[p]);
        }
        let mut x_new = vec![VecN::zeros(dim); n];
        for i in 0..n {
            let mut acc = VecN::zeros(dim);
            for j in 0..n {
                if i == j {
                    continue;
                }
                let dij = x[i].sub(&x[j]).norm().max(1e-12);
                let ratio = disp_m.get(i, j) / dij;
                acc = acc.add(&x[i].sub(&x[j]).scale(ratio).add(&x[j]));
            }
            x_new[i] = acc.scale(1.0 / (n as f64 - 1.0));
        }
        x = x_new;
    }
    x
}

/// k-nearest-neighbor graph: for each point, its k neighbors and distances.
#[must_use]
pub fn knn_graph(points: &[VecN], k: usize) -> Vec<Vec<(usize, f64)>> {
    let n = points.len();
    (0..n)
        .map(|i| {
            let mut d: Vec<(usize, f64)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j, points[i].sub(&points[j]).norm()))
                .collect();
            d.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            d.truncate(k);
            d
        })
        .collect()
}

/// All-pairs shortest paths over a kNN graph (Floyd-Warshall; symmetrized).
#[must_use]
pub fn geodesic_distance_matrix(knn: &[Vec<(usize, f64)>]) -> Matrix {
    let n = knn.len();
    let mut d = Matrix::from_fn(n, n, |i, j| if i == j { 0.0 } else { f64::INFINITY });
    for (i, nb) in knn.iter().enumerate() {
        for &(j, w) in nb {
            if w < d.get(i, j) {
                d.set(i, j, w);
                d.set(j, i, w);
            }
        }
    }
    for k in 0..n {
        for i in 0..n {
            let dik = d.get(i, k);
            if dik.is_infinite() {
                continue;
            }
            for j in 0..n {
                let alt = dik + d.get(k, j);
                if alt < d.get(i, j) {
                    d.set(i, j, alt);
                }
            }
        }
    }
    d
}

/// Isomap: geodesic distances over the kNN graph fed to classical MDS.
#[must_use]
pub fn isomap(points: &[VecN], k_neighbors: usize, dim: usize) -> Vec<VecN> {
    let knn = knn_graph(points, k_neighbors);
    let mut d = geodesic_distance_matrix(&knn);
    // guard against disconnected components
    let mut dmax = 0.0_f64;
    for i in 0..d.rows {
        for j in 0..d.cols {
            if d.get(i, j).is_finite() {
                dmax = dmax.max(d.get(i, j));
            }
        }
    }
    for i in 0..d.rows {
        for j in 0..d.cols {
            if d.get(i, j).is_infinite() {
                d.set(i, j, 2.0 * dmax);
            }
        }
    }
    classical_mds(&d, dim)
}

/// Locally linear embedding.
#[must_use]
pub fn lle(points: &[VecN], k: usize, dim: usize, reg: f64) -> Vec<VecN> {
    let n = points.len();
    let knn = knn_graph(points, k);
    // reconstruction weights
    let mut w = Matrix::zeros(n, n);
    for i in 0..n {
        let nbrs = &knn[i];
        let kk = nbrs.len();
        // local Gram matrix of neighbor offsets
        let mut g = Matrix::zeros(kk, kk);
        for a in 0..kk {
            for b in 0..kk {
                let va = points[nbrs[a].0].sub(&points[i]);
                let vb = points[nbrs[b].0].sub(&points[i]);
                g.set(a, b, va.dot(&vb));
            }
        }
        let trace: f64 = (0..kk).map(|a| g.get(a, a)).sum();
        for a in 0..kk {
            g.set(a, a, g.get(a, a) + reg * trace.max(1e-12));
        }
        let ones = vec![1.0; kk];
        let sol = lu_decompose(&g)
            .and_then(|lu| lu.solve(&ones))
            .unwrap_or(ones.clone());
        let s: f64 = sol.iter().sum();
        for (a, &(j, _)) in nbrs.iter().enumerate() {
            w.set(i, j, sol[a] / s);
        }
    }
    // M = (I - W)^T (I - W); embedding = bottom eigenvectors 1..dim
    let iw = Matrix::from_fn(n, n, |i, j| {
        (if i == j { 1.0 } else { 0.0 }) - w.get(i, j)
    });
    let m = iw.transpose().mul(&iw).unwrap();
    let msym = Matrix::from_fn(n, n, |i, j| 0.5 * (m.get(i, j) + m.get(j, i)));
    let eig = eigen_symmetric(&msym, 1e-12, 400).expect("lle eigen");
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| eig.values[a].partial_cmp(&eig.values[b]).unwrap());
    (0..n)
        .map(|i| {
            VecN::from(
                &(1..=dim)
                    .map(|d| eig.vectors.get(i, idx[d]))
                    .collect::<Vec<f64>>(),
            )
        })
        .collect()
}

/// Laplacian eigenmaps with heat-kernel weights.
#[must_use]
pub fn laplacian_eigenmaps(points: &[VecN], k: usize, dim: usize, sigma: f64) -> Vec<VecN> {
    let n = points.len();
    let knn = knn_graph(points, k);
    let mut w = Matrix::zeros(n, n);
    for (i, nb) in knn.iter().enumerate() {
        for &(j, d) in nb {
            let wt = (-d * d / (2.0 * sigma * sigma)).exp();
            w.set(i, j, wt.max(w.get(i, j)));
            w.set(j, i, wt.max(w.get(j, i)));
        }
    }
    let deg: Vec<f64> = (0..n).map(|i| (0..n).map(|j| w.get(i, j)).sum()).collect();
    // normalized symmetric Laplacian
    let lsym = Matrix::from_fn(n, n, |i, j| {
        let d = (deg[i] * deg[j]).sqrt().max(1e-12);
        (if i == j { 1.0 } else { 0.0 }) - w.get(i, j) / d
    });
    let eig = eigen_symmetric(&lsym, 1e-12, 400).expect("eigenmaps eigen");
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| eig.values[a].partial_cmp(&eig.values[b]).unwrap());
    (0..n)
        .map(|i| {
            VecN::from(
                &(1..=dim)
                    .map(|d| eig.vectors.get(i, idx[d]) / deg[i].sqrt().max(1e-12))
                    .collect::<Vec<f64>>(),
            )
        })
        .collect()
}

/// Diffusion maps: eigenfunctions of the diffusion operator, scaled by
/// lambda^t.
#[must_use]
pub fn diffusion_maps(points: &[VecN], eps: f64, dim: usize, t: f64) -> Vec<VecN> {
    let n = points.len();
    let mut k = Matrix::from_fn(n, n, |i, j| {
        (-points[i].sub(&points[j]).dot(&points[i].sub(&points[j])) / eps).exp()
    });
    // alpha = 1 density normalization
    let q: Vec<f64> = (0..n).map(|i| (0..n).map(|j| k.get(i, j)).sum()).collect();
    for i in 0..n {
        for j in 0..n {
            k.set(i, j, k.get(i, j) / (q[i] * q[j]));
        }
    }
    let d: Vec<f64> = (0..n).map(|i| (0..n).map(|j| k.get(i, j)).sum()).collect();
    // symmetric conjugate of the Markov matrix
    let a = Matrix::from_fn(n, n, |i, j| k.get(i, j) / (d[i] * d[j]).sqrt());
    let eig = eigen_symmetric(&a, 1e-12, 400).expect("diffusion eigen");
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&x, &y| eig.values[y].partial_cmp(&eig.values[x]).unwrap());
    (0..n)
        .map(|i| {
            VecN::from(
                &(1..=dim)
                    .map(|dd| {
                        let kk = idx[dd];
                        let lam = eig.values[kk].max(0.0).powf(t);
                        lam * eig.vectors.get(i, kk) / d[i].sqrt().max(1e-12)
                    })
                    .collect::<Vec<f64>>(),
            )
        })
        .collect()
}

/// Spectral embedding of a graph adjacency matrix.
#[must_use]
pub fn spectral_embedding(adjacency: &Matrix, dim: usize) -> Vec<VecN> {
    let n = adjacency.rows;
    let deg: Vec<f64> = (0..n)
        .map(|i| (0..n).map(|j| adjacency.get(i, j)).sum())
        .collect();
    let lsym = Matrix::from_fn(n, n, |i, j| {
        let d = (deg[i] * deg[j]).sqrt().max(1e-12);
        (if i == j { 1.0 } else { 0.0 }) - adjacency.get(i, j) / d
    });
    let eig = eigen_symmetric(&lsym, 1e-12, 400).expect("spectral eigen");
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| eig.values[a].partial_cmp(&eig.values[b]).unwrap());
    (0..n)
        .map(|i| {
            VecN::from(
                &(1..=dim)
                    .map(|d| eig.vectors.get(i, idx[d]))
                    .collect::<Vec<f64>>(),
            )
        })
        .collect()
}

/// Principal component analysis: returns (projected points, explained
/// variance per component, components as rows).
#[must_use]
pub fn pca(points: &[VecN], dim: usize) -> (Vec<VecN>, Vec<f64>, Matrix) {
    let n = points.len();
    let d = points[0].dim();
    let mean = points
        .iter()
        .fold(VecN::zeros(d), |a, p| a.add(p))
        .scale(1.0 / n as f64);
    let cov = Matrix::from_fn(d, d, |a, b| {
        points
            .iter()
            .map(|p| (p[a] - mean[a]) * (p[b] - mean[b]))
            .sum::<f64>()
            / n as f64
    });
    let eig = eigen_symmetric(&cov, 1e-12, 300).expect("pca eigen");
    let mut idx: Vec<usize> = (0..d).collect();
    idx.sort_by(|&a, &b| eig.values[b].partial_cmp(&eig.values[a]).unwrap());
    let comps = Matrix::from_fn(dim, d, |r, c| eig.vectors.get(c, idx[r]));
    let vars: Vec<f64> = (0..dim).map(|r| eig.values[idx[r]].max(0.0)).collect();
    let proj: Vec<VecN> = points
        .iter()
        .map(|p| {
            let c = p.sub(&mean);
            VecN::from(
                &(0..dim)
                    .map(|r| (0..d).map(|cc| comps.get(r, cc) * c[cc]).sum())
                    .collect::<Vec<f64>>(),
            )
        })
        .collect();
    (proj, vars, comps)
}

/// Kernel PCA with a user-supplied kernel.
#[must_use]
pub fn kernel_pca(
    points: &[VecN],
    kernel: &dyn Fn(&VecN, &VecN) -> f64,
    dim: usize,
) -> Vec<VecN> {
    let n = points.len();
    let k = Matrix::from_fn(n, n, |i, j| kernel(&points[i], &points[j]));
    // center the kernel matrix
    let row: Vec<f64> = (0..n)
        .map(|i| (0..n).map(|j| k.get(i, j)).sum::<f64>() / n as f64)
        .collect();
    let total: f64 = row.iter().sum::<f64>() / n as f64;
    let kc = Matrix::from_fn(n, n, |i, j| k.get(i, j) - row[i] - row[j] + total);
    top_eigen_coords(&kc, dim, true)
}

// ---------------------------------------------------------------------------
// Stochastic embeddings
// ---------------------------------------------------------------------------

/// t-SNE (exact gradients; suitable for small point sets).
#[must_use]
pub fn tsne(
    points: &[VecN],
    dim: usize,
    perplexity: f64,
    iters: usize,
    lr: f64,
    rng: &mut Rng,
) -> Vec<VecN> {
    let n = points.len();
    // pairwise squared distances
    let d2 = Matrix::from_fn(n, n, |i, j| {
        points[i].sub(&points[j]).dot(&points[i].sub(&points[j]))
    });
    // per-point precision via binary search on perplexity
    let mut p = Matrix::zeros(n, n);
    let target_h = perplexity.ln();
    for i in 0..n {
        let (mut lo, mut hi) = (1e-9, 1e9);
        let mut beta = 1.0;
        for _ in 0..60 {
            let mut sum = 0.0;
            let mut hsum = 0.0;
            for j in 0..n {
                if j == i {
                    continue;
                }
                let e = (-beta * d2.get(i, j)).exp();
                sum += e;
                hsum += beta * d2.get(i, j) * e;
            }
            let h = if sum > 1e-300 { hsum / sum + sum.ln() } else { 0.0 };
            if (h - target_h).abs() < 1e-5 {
                break;
            }
            if h > target_h {
                lo = beta;
                beta = if hi < 1e8 { 0.5 * (beta + hi) } else { beta * 2.0 };
            } else {
                hi = beta;
                beta = 0.5 * (beta + lo);
            }
        }
        let mut sum = 0.0;
        for j in 0..n {
            if j != i {
                sum += (-beta * d2.get(i, j)).exp();
            }
        }
        for j in 0..n {
            if j != i {
                p.set(i, j, (-beta * d2.get(i, j)).exp() / sum.max(1e-300));
            }
        }
    }
    // symmetrize
    let psym = Matrix::from_fn(n, n, |i, j| {
        ((p.get(i, j) + p.get(j, i)) / (2.0 * n as f64)).max(1e-12)
    });
    let mut y: Vec<VecN> = (0..n)
        .map(|_| VecN::random_gaussian(dim, rng).scale(1e-2))
        .collect();
    let mut vel = vec![VecN::zeros(dim); n];
    for it in 0..iters {
        // low-dim affinities (Student t)
        let mut qnum = Matrix::zeros(n, n);
        let mut z = 0.0;
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let q = 1.0 / (1.0 + y[i].sub(&y[j]).dot(&y[i].sub(&y[j])));
                qnum.set(i, j, q);
                z += q;
            }
        }
        let exagger = if it < iters / 4 { 4.0 } else { 1.0 };
        let momentum = if it < iters / 4 { 0.5 } else { 0.8 };
        for i in 0..n {
            let mut grad = VecN::zeros(dim);
            for j in 0..n {
                if i == j {
                    continue;
                }
                let q = qnum.get(i, j) / z.max(1e-300);
                let coef = 4.0 * (exagger * psym.get(i, j) - q) * qnum.get(i, j);
                grad = grad.add(&y[i].sub(&y[j]).scale(coef));
            }
            vel[i] = vel[i].scale(momentum).sub(&grad.scale(lr));
            y[i] = y[i].add(&vel[i]);
        }
    }
    y
}

/// Lightweight UMAP: fuzzy kNN weights optimized by SGD attraction and
/// random-negative repulsion.
#[must_use]
pub fn umap_lite(
    points: &[VecN],
    k: usize,
    dim: usize,
    min_dist: f64,
    epochs: usize,
    rng: &mut Rng,
) -> Vec<VecN> {
    let n = points.len();
    let knn = knn_graph(points, k);
    // fuzzy weights: exp(-(d - rho)/sigma) with rho = nearest distance
    let mut edges: Vec<(usize, usize, f64)> = Vec::new();
    for (i, nb) in knn.iter().enumerate() {
        let rho = nb.first().map_or(0.0, |&(_, d)| d);
        let sigma = (nb.last().map_or(1.0, |&(_, d)| d) - rho).max(1e-6);
        for &(j, d) in nb {
            let w = (-(d - rho).max(0.0) / sigma).exp();
            edges.push((i, j, w));
        }
    }
    let mut y: Vec<VecN> = (0..n)
        .map(|_| VecN::random_gaussian(dim, rng).scale(0.1))
        .collect();
    for epoch in 0..epochs {
        let alpha = 1.0 - epoch as f64 / epochs as f64;
        for &(i, j, w) in &edges {
            if rng.next_f64() > w {
                continue;
            }
            // attract
            let d = y[i].sub(&y[j]);
            let dn = d.norm().max(1e-9);
            let pull = (dn - min_dist).max(0.0);
            let step = d.scale(-alpha * 0.1 * pull / dn);
            y[i] = y[i].add(&step);
            y[j] = y[j].sub(&step);
            // repel a random negative
            let neg = (rng.next_u64() as usize) % n;
            if neg != i {
                let dneg = y[i].sub(&y[neg]);
                let dn2 = dneg.dot(&dneg).max(1e-6);
                let push = dneg.scale(alpha * 0.02 / dn2);
                y[i] = y[i].add(&push);
            }
        }
    }
    y
}

// ---------------------------------------------------------------------------
// Intrinsic dimension
// ---------------------------------------------------------------------------

/// Levina-Bickel maximum-likelihood intrinsic dimension using k neighbors.
#[must_use]
pub fn intrinsic_dimension_mle(points: &[VecN], k: usize) -> f64 {
    let knn = knn_graph(points, k);
    let mut inv_sum = 0.0;
    let mut count = 0.0;
    for nb in &knn {
        if nb.len() < k {
            continue;
        }
        let rk = nb[k - 1].1.max(1e-300);
        let mut s = 0.0;
        for &(_, d) in nb.iter().take(k - 1) {
            s += (rk / d.max(1e-300)).ln();
        }
        if s > 1e-12 {
            inv_sum += (k as f64 - 2.0).max(1.0) / s;
            count += 1.0;
        }
    }
    if count > 0.0 {
        inv_sum / count
    } else {
        0.0
    }
}

/// Correlation-dimension estimate: log-log slope of the correlation
/// integral over the radius range.
#[must_use]
pub fn intrinsic_dimension_correlation(points: &[VecN], r_range: (f64, f64)) -> f64 {
    let n = points.len();
    let mut logs = Vec::new();
    let steps = 8;
    for s in 0..steps {
        let r = r_range.0 * (r_range.1 / r_range.0).powf(s as f64 / (steps - 1) as f64);
        let mut c = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                if points[i].sub(&points[j]).norm() < r {
                    c += 1.0;
                }
            }
        }
        if c > 0.0 {
            logs.push((r.ln(), (2.0 * c / (n as f64 * (n as f64 - 1.0))).ln()));
        }
    }
    // least squares slope
    let m = logs.len() as f64;
    if m < 2.0 {
        return 0.0;
    }
    let mx = logs.iter().map(|l| l.0).sum::<f64>() / m;
    let my = logs.iter().map(|l| l.1).sum::<f64>() / m;
    let num: f64 = logs.iter().map(|l| (l.0 - mx) * (l.1 - my)).sum();
    let den: f64 = logs.iter().map(|l| (l.0 - mx).powi(2)).sum();
    num / den
}

/// TwoNN intrinsic dimension (Facco et al.): d = n / sum ln(r2/r1).
#[must_use]
pub fn intrinsic_dimension_two_nn(points: &[VecN]) -> f64 {
    let knn = knn_graph(points, 2);
    let mut s = 0.0;
    let mut n = 0.0;
    for nb in &knn {
        if nb.len() >= 2 && nb[0].1 > 1e-12 {
            s += (nb[1].1 / nb[0].1).ln();
            n += 1.0;
        }
    }
    if s > 1e-12 {
        n / s
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Quality metrics
// ---------------------------------------------------------------------------

fn rank_matrix(points: &[VecN]) -> Vec<Vec<usize>> {
    let n = points.len();
    (0..n)
        .map(|i| {
            let mut idx: Vec<usize> = (0..n).filter(|&j| j != i).collect();
            idx.sort_by(|&a, &b| {
                points[i]
                    .sub(&points[a])
                    .norm()
                    .partial_cmp(&points[i].sub(&points[b]).norm())
                    .unwrap()
            });
            let mut rank = vec![0usize; n];
            for (r, &j) in idx.iter().enumerate() {
                rank[j] = r + 1;
            }
            rank
        })
        .collect()
}

/// Trustworthiness of a low-dimensional embedding (1 = perfect).
#[must_use]
pub fn trustworthiness(high: &[VecN], low: &[VecN], k: usize) -> f64 {
    let n = high.len();
    let rank_high = rank_matrix(high);
    let knn_low = knn_graph(low, k);
    let knn_high = knn_graph(high, k);
    let mut penalty = 0.0;
    for i in 0..n {
        let high_set: std::collections::HashSet<usize> =
            knn_high[i].iter().map(|&(j, _)| j).collect();
        for &(j, _) in &knn_low[i] {
            if !high_set.contains(&j) {
                penalty += (rank_high[i][j] as f64 - k as f64).max(0.0);
            }
        }
    }
    let norm = n as f64 * k as f64 * (2.0 * n as f64 - 3.0 * k as f64 - 1.0) / 2.0;
    1.0 - 2.0 * penalty / norm.max(1e-12)
}

/// Continuity of an embedding (trustworthiness with roles swapped).
#[must_use]
pub fn continuity(high: &[VecN], low: &[VecN], k: usize) -> f64 {
    trustworthiness(low, high, k)
}

/// Kruskal stress between two distance matrices.
#[must_use]
pub fn stress(dist_high: &Matrix, dist_low: &Matrix) -> f64 {
    let n = dist_high.rows;
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            num += (dist_high.get(i, j) - dist_low.get(i, j)).powi(2);
            den += dist_high.get(i, j).powi(2);
        }
    }
    (num / den.max(1e-300)).sqrt()
}

/// Fraction of k-nearest neighbors preserved by the embedding.
#[must_use]
pub fn neighborhood_preservation(high: &[VecN], low: &[VecN], k: usize) -> f64 {
    let n = high.len();
    let kh = knn_graph(high, k);
    let kl = knn_graph(low, k);
    let mut kept = 0.0;
    for i in 0..n {
        let hs: std::collections::HashSet<usize> = kh[i].iter().map(|&(j, _)| j).collect();
        for &(j, _) in &kl[i] {
            if hs.contains(&j) {
                kept += 1.0;
            }
        }
    }
    kept / (n * k) as f64
}

/// Procrustes alignment of b onto a (rotation + scale + translation);
/// returns (aligned b, residual).
#[must_use]
pub fn procrustes_align(a: &[VecN], b: &[VecN]) -> (Vec<VecN>, f64) {
    let n = a.len();
    let d = a[0].dim();
    let mean = |pts: &[VecN]| {
        pts.iter()
            .fold(VecN::zeros(d), |acc, p| acc.add(p))
            .scale(1.0 / n as f64)
    };
    let ma = mean(a);
    let mb = mean(b);
    let ac: Vec<VecN> = a.iter().map(|p| p.sub(&ma)).collect();
    let bc: Vec<VecN> = b.iter().map(|p| p.sub(&mb)).collect();
    // cross-covariance
    let cov = Matrix::from_fn(d, d, |r, c| {
        (0..n).map(|i| ac[i][r] * bc[i][c]).sum()
    });
    let s = svd(&cov).expect("procrustes svd");
    let rot = s.u.mul(&s.vt).unwrap();
    let scale_num: f64 = s.sigma.iter().sum();
    let scale_den: f64 = bc.iter().map(|p| p.dot(p)).sum();
    let scale = scale_num / scale_den.max(1e-300);
    let aligned: Vec<VecN> = bc
        .iter()
        .map(|p| {
            let rp = VecN::from(
                &(0..d)
                    .map(|r| (0..d).map(|c| rot.get(r, c) * p[c]).sum())
                    .collect::<Vec<f64>>(),
            );
            rp.scale(scale).add(&ma)
        })
        .collect();
    let resid: f64 = aligned
        .iter()
        .zip(a)
        .map(|(x, y)| x.sub(y).dot(&x.sub(y)))
        .sum::<f64>()
        .sqrt();
    (aligned, resid)
}

// ---------------------------------------------------------------------------
// Benchmark datasets
// ---------------------------------------------------------------------------

/// Swiss roll in R3 with the unrolled arc-length parameter as ground truth.
#[must_use]
pub fn swiss_roll(n: usize, noise: f64, rng: &mut Rng) -> (Vec<VecN>, Vec<f64>) {
    let mut pts = Vec::with_capacity(n);
    let mut ts = Vec::with_capacity(n);
    for _ in 0..n {
        let t = 1.5 * PI * (1.0 + 2.0 * rng.next_f64());
        let y = 21.0 * rng.next_f64();
        let p = VecN::from(&[
            t * t.cos() + noise * rng.next_gaussian(),
            y + noise * rng.next_gaussian(),
            t * t.sin() + noise * rng.next_gaussian(),
        ]);
        pts.push(p);
        // arc length of the spiral r = t: integral sqrt(1 + t^2) dt
        let arc = 0.5 * (t * (1.0 + t * t).sqrt() + (t + (1.0 + t * t).sqrt()).ln());
        ts.push(arc);
    }
    (pts, ts)
}

/// S-curve dataset with the curve parameter as ground truth.
#[must_use]
pub fn s_curve(n: usize, noise: f64, rng: &mut Rng) -> (Vec<VecN>, Vec<f64>) {
    let mut pts = Vec::with_capacity(n);
    let mut ts = Vec::with_capacity(n);
    for _ in 0..n {
        let t = 3.0 * PI * (rng.next_f64() - 0.5);
        let y = 2.0 * rng.next_f64();
        pts.push(VecN::from(&[
            t.sin() + noise * rng.next_gaussian(),
            y + noise * rng.next_gaussian(),
            t.signum() * (t.cos() - 1.0) + noise * rng.next_gaussian(),
        ]));
        ts.push(t);
    }
    (pts, ts)
}

/// Points on a torus with (u, v) angles as ground truth.
#[must_use]
pub fn torus_sample(n: usize, big_r: f64, small_r: f64, rng: &mut Rng) -> (Vec<VecN>, Vec<(f64, f64)>) {
    let mut pts = Vec::with_capacity(n);
    let mut uv = Vec::with_capacity(n);
    for _ in 0..n {
        let u = 2.0 * PI * rng.next_f64();
        let v = 2.0 * PI * rng.next_f64();
        pts.push(VecN::from(&[
            (big_r + small_r * v.cos()) * u.cos(),
            (big_r + small_r * v.cos()) * u.sin(),
            small_r * v.sin(),
        ]));
        uv.push((u, v));
    }
    (pts, uv)
}

/// Uniform points on the unit 2-sphere in R3.
#[must_use]
pub fn sphere_sample(n: usize, rng: &mut Rng) -> Vec<VecN> {
    (0..n).map(|_| VecN::random_unit(3, rng)).collect()
}

/// Helix in R3 with the parameter as ground truth.
#[must_use]
pub fn helix_sample(n: usize, rng: &mut Rng) -> (Vec<VecN>, Vec<f64>) {
    let mut pts = Vec::with_capacity(n);
    let mut ts = Vec::with_capacity(n);
    for _ in 0..n {
        let t = 4.0 * PI * rng.next_f64();
        pts.push(VecN::from(&[t.cos(), t.sin(), 0.3 * t]));
        ts.push(t);
    }
    (pts, ts)
}

/// Points on a Mobius band.
#[must_use]
pub fn mobius_sample(n: usize, rng: &mut Rng) -> Vec<VecN> {
    (0..n)
        .map(|_| {
            let u = 2.0 * PI * rng.next_f64();
            let v = rng.next_f64() - 0.5;
            VecN::from(&[
                (1.0 + 0.5 * v * (0.5 * u).cos()) * u.cos(),
                (1.0 + 0.5 * v * (0.5 * u).cos()) * u.sin(),
                0.5 * v * (0.5 * u).sin(),
            ])
        })
        .collect()
}

/// Points on the figure-8 immersion of the Klein bottle in R3.
#[must_use]
pub fn klein_sample(n: usize, rng: &mut Rng) -> Vec<VecN> {
    (0..n)
        .map(|_| {
            let u = 2.0 * PI * rng.next_f64();
            let v = 2.0 * PI * rng.next_f64();
            let r = 1.0 + 0.5 * (0.5 * u).cos() * v.sin() - 0.5 * (0.5 * u).sin() * (2.0 * v).sin();
            VecN::from(&[
                r * u.cos(),
                r * u.sin(),
                0.5 * (0.5 * u).sin() * v.sin() + 0.5 * (0.5 * u).cos() * (2.0 * v).sin(),
            ])
        })
        .collect()
}

/// The two-moons dataset with labels.
#[must_use]
pub fn two_moons(n: usize, noise: f64, rng: &mut Rng) -> (Vec<VecN>, Vec<usize>) {
    let mut pts = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);
    for i in 0..n {
        let t = PI * rng.next_f64();
        let (p, l) = if i % 2 == 0 {
            (VecN::from(&[t.cos(), t.sin()]), 0)
        } else {
            (VecN::from(&[1.0 - t.cos(), 0.5 - t.sin()]), 1)
        };
        pts.push(VecN::from(&[
            p[0] + noise * rng.next_gaussian(),
            p[1] + noise * rng.next_gaussian(),
        ]));
        labels.push(l);
    }
    (pts, labels)
}

/// Isotropic Gaussian blobs with labels.
#[must_use]
pub fn blobs(
    n: usize,
    centers: &[VecN],
    spread: f64,
    rng: &mut Rng,
) -> (Vec<VecN>, Vec<usize>) {
    let d = centers[0].dim();
    let mut pts = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);
    for i in 0..n {
        let c = i % centers.len();
        let mut p = centers[c].clone();
        for k in 0..d {
            p.data[k] += spread * rng.next_gaussian();
        }
        pts.push(p);
        labels.push(c);
    }
    (pts, labels)
}

// ---------------------------------------------------------------------------
// Local geometry and matrix manifolds
// ---------------------------------------------------------------------------

/// Local tangent space at a point by PCA of its k neighbors: rows are an
/// orthonormal basis.
#[must_use]
pub fn tangent_space_estimate(points: &[VecN], idx: usize, k: usize, dim: usize) -> Matrix {
    let knn = knn_graph(points, k);
    let nbrs: Vec<VecN> = knn[idx]
        .iter()
        .map(|&(j, _)| points[j].clone())
        .chain(std::iter::once(points[idx].clone()))
        .collect();
    let (_, _, comps) = pca(&nbrs, dim);
    comps
}

/// Curvature proxy at a point: residual variance fraction outside the local
/// tangent plane.
#[must_use]
pub fn manifold_curvature_estimate(points: &[VecN], idx: usize, k: usize) -> f64 {
    let knn = knn_graph(points, k);
    let nbrs: Vec<VecN> = knn[idx]
        .iter()
        .map(|&(j, _)| points[j].clone())
        .chain(std::iter::once(points[idx].clone()))
        .collect();
    let d = points[0].dim();
    let (_, vars, _) = pca(&nbrs, d);
    let total: f64 = vars.iter().sum();
    if total < 1e-300 {
        return 0.0;
    }
    // fraction beyond the top-2 components
    vars.iter().skip(2).sum::<f64>() / total
}

/// Grassmann distance between subspaces spanned by the rows of a and b
/// (square root of the sum of squared principal angles).
#[must_use]
pub fn grassmann_distance(a: &Matrix, b: &Matrix) -> f64 {
    // orthonormalize rows and compute singular values of Qa Qb^T
    let qa = orthonormal_rows(a);
    let qb = orthonormal_rows(b);
    let m = qa.mul(&qb.transpose()).unwrap();
    let s = svd(&m).expect("grassmann svd");
    s.sigma
        .iter()
        .map(|&c| c.clamp(-1.0, 1.0).acos().powi(2))
        .sum::<f64>()
        .sqrt()
}

fn orthonormal_rows(a: &Matrix) -> Matrix {
    let mut rows: Vec<VecN> = (0..a.rows)
        .map(|r| VecN::from(a.row(r)))
        .collect();
    let mut out = Vec::new();
    for mut v in rows.drain(..) {
        for u in &out {
            let uu: &VecN = u;
            v = v.sub(&uu.scale(v.dot(uu)));
        }
        let n = v.norm();
        if n > 1e-12 {
            out.push(v.scale(1.0 / n));
        }
    }
    Matrix::from_fn(out.len(), a.cols, |r, c| out[r][c])
}

/// Project a matrix onto the Stiefel manifold (nearest orthonormal-column
/// matrix, via the polar factor).
#[must_use]
pub fn stiefel_project(m: &Matrix) -> Matrix {
    let s = svd(m).expect("stiefel svd");
    s.u.mul(&s.vt).unwrap()
}

/// Riemannian gradient descent on the unit sphere.
#[must_use]
pub fn riemannian_gradient_descent_sphere(
    f: &dyn Fn(&VecN) -> f64,
    grad: &dyn Fn(&VecN) -> VecN,
    x0: &VecN,
    iters: usize,
    lr: f64,
) -> VecN {
    let mut x = x0.normalized();
    let mut best = x.clone();
    let mut best_f = f(&x);
    for _ in 0..iters {
        let g = grad(&x);
        // project to the tangent space and retract
        let gt = g.sub(&x.scale(g.dot(&x)));
        x = x.sub(&gt.scale(lr)).normalized();
        let fx = f(&x);
        if fx < best_f {
            best_f = fx;
            best = x.clone();
        }
    }
    best
}

/// Riemannian gradient descent on the Stiefel manifold (projection
/// retraction).
#[must_use]
pub fn riemannian_gradient_descent_stiefel(
    f: &dyn Fn(&Matrix) -> f64,
    grad: &dyn Fn(&Matrix) -> Matrix,
    x0: &Matrix,
    iters: usize,
    lr: f64,
) -> Matrix {
    let mut x = stiefel_project(x0);
    let mut best = x.clone();
    let mut best_f = f(&x);
    for _ in 0..iters {
        let g = grad(&x);
        let step = Matrix::from_fn(x.rows, x.cols, |i, j| x.get(i, j) - lr * g.get(i, j));
        x = stiefel_project(&step);
        let fx = f(&x);
        if fx < best_f {
            best_f = fx;
            best = x.clone();
        }
    }
    best
}

/// k-means with distances and means taken in a Riemannian metric (uses the
/// metric's exp/log maps). Returns (centroids, labels).
#[must_use]
pub fn geodesic_kmeans(
    metric: &Metric,
    points: &[VecN],
    k: usize,
    iters: usize,
    rng: &mut Rng,
) -> (Vec<VecN>, Vec<usize>) {
    let n = points.len();
    let mut centroids: Vec<VecN> = (0..k)
        .map(|_| points[(rng.next_u64() as usize) % n].clone())
        .collect();
    let mut labels = vec![0usize; n];
    for _ in 0..iters {
        for (i, p) in points.iter().enumerate() {
            let mut best = 0;
            let mut best_d = f64::MAX;
            for (c, cent) in centroids.iter().enumerate() {
                let d = match metric.log_map(cent, p) {
                    Ok(v) => metric.norm(cent, &v),
                    Err(_) => p.sub(cent).norm(),
                };
                if d < best_d {
                    best_d = d;
                    best = c;
                }
            }
            labels[i] = best;
        }
        for (c, cent) in centroids.iter_mut().enumerate() {
            let members: Vec<&VecN> = points
                .iter()
                .zip(&labels)
                .filter(|&(_, &l)| l == c)
                .map(|(p, _)| p)
                .collect();
            if members.is_empty() {
                continue;
            }
            // one Karcher step
            let mut avg = VecN::zeros(cent.dim());
            let mut cnt = 0.0;
            for p in &members {
                if let Ok(v) = metric.log_map(cent, p) {
                    avg = avg.add(&v);
                    cnt += 1.0;
                }
            }
            if cnt > 0.0 {
                *cent = metric.exp_map(cent, &avg.scale(1.0 / cnt));
            }
        }
    }
    (centroids, labels)
}

/// Radial-basis interpolation of scattered manifold data.
#[must_use]
pub fn manifold_interpolation_rbf(
    points: &[VecN],
    values: &[f64],
    query: &VecN,
    kernel: &dyn Fn(f64) -> f64,
) -> f64 {
    let n = points.len();
    let a = Matrix::from_fn(n, n, |i, j| kernel(points[i].sub(&points[j]).norm()));
    let w = lu_decompose(&a)
        .and_then(|lu| lu.solve(values))
        .unwrap_or_else(|_| values.to_vec());
    (0..n)
        .map(|i| w[i] * kernel(points[i].sub(query).norm()))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn correlation(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len() as f64;
        let ma = a.iter().sum::<f64>() / n;
        let mb = b.iter().sum::<f64>() / n;
        let mut num = 0.0;
        let mut da = 0.0;
        let mut db = 0.0;
        for (x, y) in a.iter().zip(b) {
            num += (x - ma) * (y - mb);
            da += (x - ma).powi(2);
            db += (y - mb).powi(2);
        }
        num / (da * db).sqrt().max(1e-300)
    }

    #[test]
    fn test_pca_matches_covariance_eigen() {
        let mut rng = Rng::new(3);
        // anisotropic Gaussian: variance 9 along a known direction
        let dir = VecN::from(&[0.6, 0.8, 0.0]);
        let mut pts = Vec::new();
        for _ in 0..300 {
            let main = 3.0 * rng.next_gaussian();
            let off1 = 0.5 * rng.next_gaussian();
            let off2 = 0.2 * rng.next_gaussian();
            pts.push(VecN::from(&[
                main * dir[0] - off1 * 0.8,
                main * dir[1] + off1 * 0.6,
                off2,
            ]));
        }
        let (proj, vars, comps) = pca(&pts, 2);
        assert_eq!(proj.len(), 300);
        // first component aligns with dir
        let c0 = VecN::from(comps.row(0));
        assert!(c0.dot(&dir).abs() > 0.99, "pca direction {c0:?}");
        assert!(vars[0] > 5.0 && vars[0] < 12.0, "explained var {vars:?}");
        assert!(vars[0] > vars[1]);
        // explained variance equals the projected variance
        let pvar: f64 = proj.iter().map(|p| p[0] * p[0]).sum::<f64>() / 300.0;
        assert!((pvar - vars[0]).abs() < 1e-9);
        // classical MDS of the Euclidean distances reproduces PCA shape
        let d = dist_matrix(&pts);
        let mds = classical_mds(&d, 2);
        let (_, resid) = procrustes_align(&mds, &proj);
        let scale: f64 = mds.iter().map(|p| p.dot(p)).sum::<f64>().sqrt();
        assert!(resid < 1e-6 * scale.max(1.0) + 1e-6, "mds vs pca {resid}");
    }

    #[test]
    fn test_isomap_swiss_roll() {
        let mut rng = Rng::new(11);
        // sample densely and keep the inner windings so the kNN graph
        // cannot short-circuit across the 2 pi winding gap
        let (raw, raw_t) = swiss_roll(900, 0.0, &mut rng);
        let mut pts = Vec::new();
        let mut t = Vec::new();
        for (p, a) in raw.into_iter().zip(raw_t) {
            if a < 55.0 && pts.len() < 280 {
                pts.push(p);
                t.push(a);
            }
        }
        let emb = isomap(&pts, 6, 2);
        // one embedding axis correlates with the roll parameter
        let e0: Vec<f64> = emb.iter().map(|p| p[0]).collect();
        let e1: Vec<f64> = emb.iter().map(|p| p[1]).collect();
        let c = correlation(&e0, &t).abs().max(correlation(&e1, &t).abs());
        assert!(c > 0.95, "isomap correlation {c}");
        // and the other with the width (y coordinate)
        let width: Vec<f64> = pts.iter().map(|p| p[1]).collect();
        let cw = correlation(&e0, &width)
            .abs()
            .max(correlation(&e1, &width).abs());
        assert!(cw > 0.9, "isomap width correlation {cw}");
    }

    #[test]
    fn test_intrinsic_dimensions() {
        let mut rng = Rng::new(17);
        // sphere sample: intrinsic dimension ~ 2
        let s2 = sphere_sample(400, &mut rng);
        let d_mle = intrinsic_dimension_mle(&s2, 10);
        assert!((d_mle - 2.0).abs() < 0.4, "MLE dim {d_mle}");
        let d_2nn = intrinsic_dimension_two_nn(&s2);
        assert!((d_2nn - 2.0).abs() < 0.5, "TwoNN dim {d_2nn}");
        // helix: dimension ~ 1
        let (h, _) = helix_sample(300, &mut rng);
        let d_h = intrinsic_dimension_mle(&h, 8);
        assert!((d_h - 1.0).abs() < 0.35, "helix dim {d_h}");
        // correlation dimension of a filled square ~ 2
        let sq: Vec<VecN> = (0..400)
            .map(|_| VecN::from(&[rng.next_f64(), rng.next_f64()]))
            .collect();
        let d_c = intrinsic_dimension_correlation(&sq, (0.05, 0.3));
        assert!((d_c - 2.0).abs() < 0.4, "correlation dim {d_c}");
    }

    #[test]
    fn test_embeddings_qualitative() {
        let mut rng = Rng::new(29);
        // LLE and Laplacian eigenmaps unroll the s-curve enough to keep
        // neighborhoods
        let (pts, _) = s_curve(200, 0.0, &mut rng);
        for (name, emb) in [
            ("lle", lle(&pts, 10, 2, 1e-3)),
            ("eigenmaps", laplacian_eigenmaps(&pts, 10, 2, 0.5)),
            ("diffusion", diffusion_maps(&pts, 1.0, 2, 1.0)),
        ] {
            assert!(emb.iter().all(|p| p.data.iter().all(|v| v.is_finite())), "{name} finite");
            let np = neighborhood_preservation(&pts, &emb, 8);
            assert!(np > 0.35, "{name} neighborhood preservation {np}");
        }
        // spectral embedding separates two graph clusters
        let n = 20;
        let adj = Matrix::from_fn(2 * n, 2 * n, |i, j| {
            if i == j {
                0.0
            } else if (i < n) == (j < n) || (i + j) % 37 == 0 {
                1.0 // in-cluster, plus sparse cross links
            } else {
                0.0
            }
        });
        let se = spectral_embedding(&adj, 1);
        let mean_a: f64 = (0..n).map(|i| se[i][0]).sum::<f64>() / n as f64;
        let mean_b: f64 = (n..2 * n).map(|i| se[i][0]).sum::<f64>() / n as f64;
        assert!(
            (mean_a - mean_b).abs() > 0.05,
            "spectral separation {mean_a} vs {mean_b}"
        );
        // kernel PCA with the linear kernel matches classical MDS shape
        let lin = |a: &VecN, b: &VecN| a.dot(b);
        let kp = kernel_pca(&pts, &lin, 2);
        let (p2, _, _) = pca(&pts, 2);
        let (_, resid) = procrustes_align(&kp, &p2);
        let scale: f64 = kp.iter().map(|p| p.dot(p)).sum::<f64>().sqrt();
        assert!(resid < 1e-5 * scale.max(1.0) + 1e-5, "kernel pca vs pca {resid}");
    }

    #[test]
    fn test_smacof_and_nonmetric() {
        let mut rng = Rng::new(41);
        // planar configuration: SMACOF recovers it with low stress
        let truth: Vec<VecN> = (0..25)
            .map(|k| VecN::from(&[(k % 5) as f64, (k / 5) as f64]))
            .collect();
        let d = dist_matrix(&truth);
        let (emb, stress_val) = metric_mds_smacof(&d, 2, 100, &mut rng);
        assert!(stress_val < 1e-6, "smacof stress {stress_val}");
        let (_, resid) = procrustes_align(&truth, &emb);
        assert!(resid < 1e-3, "smacof recovery {resid}");
        // nonmetric MDS keeps the rank order (finite and low stress-1)
        let nm = nonmetric_mds(&d, 2, 30, &mut rng);
        assert!(nm.iter().all(|p| p.data.iter().all(|v| v.is_finite())));
        let dl = dist_matrix(&nm);
        assert!(stress(&d, &dl) < 0.3, "nonmetric stress");
        // trustworthiness of a perfect embedding is 1
        let tw = trustworthiness(&truth, &truth, 5);
        assert!((tw - 1.0).abs() < 1e-12);
        let cy = continuity(&truth, &truth, 5);
        assert!((cy - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_tsne_umap_separate_blobs() {
        let mut rng = Rng::new(53);
        let centers = vec![
            VecN::from(&[0.0, 0.0, 0.0, 0.0]),
            VecN::from(&[8.0, 0.0, 0.0, 0.0]),
        ];
        let (pts, labels) = blobs(60, &centers, 0.5, &mut rng);
        let sep_score = |emb: &[VecN]| {
            // mean intra-cluster vs inter-cluster distance
            let mut intra = 0.0;
            let mut inter = 0.0;
            let mut ni = 0.0;
            let mut nx = 0.0;
            for i in 0..emb.len() {
                for j in (i + 1)..emb.len() {
                    let d = emb[i].sub(&emb[j]).norm();
                    if labels[i] == labels[j] {
                        intra += d;
                        ni += 1.0;
                    } else {
                        inter += d;
                        nx += 1.0;
                    }
                }
            }
            (inter / nx) / (intra / ni)
        };
        let ts = tsne(&pts, 2, 10.0, 250, 20.0, &mut rng);
        assert!(ts.iter().all(|p| p.data.iter().all(|v| v.is_finite())));
        let s_tsne = sep_score(&ts);
        assert!(s_tsne > 1.5, "tsne separation {s_tsne}");
        let um = umap_lite(&pts, 8, 2, 0.1, 150, &mut rng);
        assert!(um.iter().all(|p| p.data.iter().all(|v| v.is_finite())));
        let s_umap = sep_score(&um);
        assert!(s_umap > 1.5, "umap separation {s_umap}");
    }

    #[test]
    fn test_local_geometry_and_matrix_manifolds() {
        let mut rng = Rng::new(61);
        // tangent space of a sphere point is orthogonal to the radius
        let s2 = sphere_sample(300, &mut rng);
        let t = tangent_space_estimate(&s2, 0, 12, 2);
        let radial = &s2[0];
        for r in 0..2 {
            let row = VecN::from(t.row(r));
            assert!(row.dot(radial).abs() < 0.3, "tangent orthogonality");
        }
        // curvature estimate: plane ~ 0, sphere > plane
        let plane_pts: Vec<VecN> = (0..200)
            .map(|_| VecN::from(&[rng.next_gaussian(), rng.next_gaussian(), 0.0]))
            .collect();
        let c_plane = manifold_curvature_estimate(&plane_pts, 0, 15);
        let c_sphere = manifold_curvature_estimate(&s2, 0, 15);
        assert!(c_plane < 1e-9, "plane curvature {c_plane}");
        assert!(c_sphere > c_plane, "sphere curvature");
        // Grassmann distance: identical subspaces 0, orthogonal planes pi/2
        // per angle
        let a = Matrix::from_rows(&[&[1.0, 0.0, 0.0, 0.0], &[0.0, 1.0, 0.0, 0.0]]).unwrap();
        let b = Matrix::from_rows(&[&[0.0, 0.0, 1.0, 0.0], &[0.0, 0.0, 0.0, 1.0]]).unwrap();
        assert!(grassmann_distance(&a, &a) < 1e-9);
        let g = grassmann_distance(&a, &b);
        assert!((g - (2.0_f64).sqrt() * PI / 2.0).abs() < 1e-9, "grassmann {g}");
        // Stiefel projection gives orthonormal columns
        let m = Matrix::from_fn(4, 2, |i, j| (i + 2 * j) as f64 * 0.3 + 0.1);
        let q = stiefel_project(&m);
        let qtq = q.transpose().mul(&q).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((qtq.get(i, j) - want).abs() < 1e-9);
            }
        }
        // Riemannian GD on the sphere: minimize <x, v> -> x = -v
        let v = VecN::from(&[0.0, 0.0, 1.0]);
        let vf = v.clone();
        let vg = v.clone();
        let f = move |x: &VecN| x.dot(&vf);
        let grad = move |_: &VecN| vg.clone();
        let x = riemannian_gradient_descent_sphere(
            &f,
            &grad,
            &VecN::from(&[1.0, 0.0, 0.1]).normalized(),
            200,
            0.1,
        );
        assert!(x[2] < -0.99, "sphere GD {x:?}");
        // RBF interpolation reproduces values at the nodes
        let nodes: Vec<VecN> = (0..10)
            .map(|k| VecN::from(&[k as f64 * 0.3, (k * k) as f64 * 0.01]))
            .collect();
        let vals: Vec<f64> = nodes.iter().map(|p| p[0].sin() + p[1]).collect();
        let ker = |r: f64| (-r * r).exp();
        for (p, &v) in nodes.iter().zip(&vals) {
            let est = manifold_interpolation_rbf(&nodes, &vals, p, &ker);
            assert!((est - v).abs() < 1e-8, "rbf node {est} vs {v}");
        }
        // geodesic k-means on the Euclidean plane behaves like k-means
        let metric = Metric::euclidean(2);
        let cents = vec![VecN::from(&[0.0, 0.0]), VecN::from(&[6.0, 0.0])];
        let (data, labels_true) = blobs(40, &cents, 0.3, &mut rng);
        let (_, labels) = geodesic_kmeans(&metric, &data, 2, 8, &mut rng);
        // cluster agreement up to permutation
        let same: usize = labels
            .iter()
            .zip(&labels_true)
            .filter(|&(a, b)| a == b)
            .count();
        let agree = same.max(labels.len() - same);
        assert!(agree > 36, "kmeans agreement {agree}/40");
    }

    #[test]
    fn test_surface_samples_satisfy_their_equations() {
        let mut rng = Rng::new(97);
        // torus: the implicit equation (sqrt(x^2+y^2) - R)^2 + z^2 = r^2,
        // and the points reproduce the returned (u, v) parameters exactly
        let (big_r, small_r) = (3.0, 0.8);
        let (pts, uv) = torus_sample(200, big_r, small_r, &mut rng);
        assert_eq!(pts.len(), 200);
        assert_eq!(uv.len(), 200);
        for (p, &(u, v)) in pts.iter().zip(&uv) {
            assert_eq!(p.dim(), 3);
            let rho = (p[0] * p[0] + p[1] * p[1]).sqrt();
            let implicit = (rho - big_r).powi(2) + p[2] * p[2];
            assert!(
                (implicit - small_r * small_r).abs() < 1e-12,
                "torus implicit {implicit} vs {}",
                small_r * small_r
            );
            let want = [
                (big_r + small_r * v.cos()) * u.cos(),
                (big_r + small_r * v.cos()) * u.sin(),
                small_r * v.sin(),
            ];
            for k in 0..3 {
                assert!((p[k] - want[k]).abs() < 1e-12, "torus parametrization");
            }
            assert!((0.0..=2.0 * PI).contains(&u) && (0.0..=2.0 * PI).contains(&v));
            // the tube stays inside the annulus R -+ r
            assert!(rho >= big_r - small_r - 1e-12 && rho <= big_r + small_r + 1e-12);
        }
        // Mobius band: every point is at distance |v|/2 <= 1/4 from the unit
        // center circle, and lies on the ray at half-angle u/2 in the
        // (radial, z) plane -- z cos(u/2) = (rho - 1) sin(u/2)
        let mob = mobius_sample(150, &mut rng);
        assert_eq!(mob.len(), 150);
        for p in &mob {
            assert_eq!(p.dim(), 3);
            let rho = (p[0] * p[0] + p[1] * p[1]).sqrt();
            let u = p[1].atan2(p[0]);
            let (s, c) = (0.5 * u).sin_cos();
            let on_ray = p[2] * c - (rho - 1.0) * s;
            assert!(on_ray.abs() < 1e-12, "mobius half-twist residual {on_ray}");
            let dist = ((rho - 1.0).powi(2) + p[2] * p[2]).sqrt();
            assert!(dist <= 0.25 + 1e-12, "mobius half-width {dist}");
        }
        // and the band is genuinely twisted: some samples sit off the plane
        assert!(mob.iter().any(|p| p[2].abs() > 0.05), "mobius is not flat");
        // Klein bottle (figure-8 immersion): recovering (sin v, sin 2v) from
        // the point must satisfy sin^2(2v) = 4 sin^2 v (1 - sin^2 v)
        let klein = klein_sample(150, &mut rng);
        assert_eq!(klein.len(), 150);
        for p in &klein {
            assert_eq!(p.dim(), 3);
            let rho = (p[0] * p[0] + p[1] * p[1]).sqrt();
            assert!(rho > 0.3, "figure-8 radius stays positive: {rho}");
            let u = p[1].atan2(p[0]);
            let (b, a) = (0.5 * u).sin_cos();
            // [rho - 1; z] = 0.5 * R(u/2) * [sin v; sin 2v]
            let sv = 2.0 * ((rho - 1.0) * a + p[2] * b);
            let s2v = 2.0 * (-(rho - 1.0) * b + p[2] * a);
            assert!(sv.abs() <= 1.0 + 1e-9 && s2v.abs() <= 1.0 + 1e-9, "sines bounded");
            let resid = s2v * s2v - 4.0 * sv * sv * (1.0 - sv * sv);
            assert!(resid.abs() < 1e-9, "klein double-angle residual {resid}");
        }
        // two moons: labels alternate and are balanced, and with no noise the
        // two arcs lie exactly on their unit circles
        let (mp, ml) = two_moons(200, 0.0, &mut rng);
        assert_eq!(mp.len(), 200);
        let ones = ml.iter().filter(|&&l| l == 1).count();
        assert_eq!(ones, 100, "balanced labels");
        assert!(ml.iter().all(|&l| l < 2));
        for (p, &l) in mp.iter().zip(&ml) {
            assert_eq!(p.dim(), 2);
            if l == 0 {
                assert!((p[0] * p[0] + p[1] * p[1] - 1.0).abs() < 1e-12, "upper moon");
                assert!(p[1] >= -1e-12, "upper moon is the top half");
            } else {
                let d = (p[0] - 1.0).powi(2) + (p[1] - 0.5).powi(2);
                assert!((d - 1.0).abs() < 1e-12, "lower moon");
                assert!(p[1] <= 0.5 + 1e-12, "lower moon is the bottom half");
            }
        }
        // with noise the two moons stay well separated: nearest neighbors
        // share their label, and the closest pair across labels is far
        // beyond the within-moon spacing
        let (np, nl) = two_moons(400, 0.05, &mut rng);
        let nn = knn_graph(&np, 1);
        let agree = nn
            .iter()
            .enumerate()
            .filter(|&(i, nb)| nl[i] == nl[nb[0].0])
            .count();
        assert!(agree > 380, "nearest neighbours share a moon: {agree}/400");
        let mut min_cross = f64::MAX;
        let mut max_nn = 0.0_f64;
        for i in 0..np.len() {
            max_nn = max_nn.max(nn[i][0].1);
            for j in (i + 1)..np.len() {
                if nl[i] != nl[j] {
                    min_cross = min_cross.min(np[i].sub(&np[j]).norm());
                }
            }
        }
        assert!(min_cross > 0.15, "moon gap {min_cross}");
        assert!(max_nn < min_cross, "within-moon spacing {max_nn} vs gap {min_cross}");
    }

    #[test]
    fn test_stiefel_gradient_descent_finds_dominant_subspace() {
        // minimize f(X) = -tr(X^T A X) over 4x2 orthonormal-column matrices;
        // the optimum is the span of the top two eigenvectors of A, with
        // value -(lambda_1 + lambda_2)
        let diag = [3.0, 2.0, 1.0, 0.5];
        let a = Matrix::from_fn(4, 4, |i, j| if i == j { diag[i] } else { 0.0 });
        let af = a.clone();
        let ag = a.clone();
        let f = move |x: &Matrix| -> f64 {
            let ax = af.mul(x).unwrap();
            -(0..x.cols)
                .map(|c| (0..x.rows).map(|r| x.get(r, c) * ax.get(r, c)).sum::<f64>())
                .sum::<f64>()
        };
        let grad = move |x: &Matrix| -> Matrix {
            let ax = ag.mul(x).unwrap();
            Matrix::from_fn(x.rows, x.cols, |i, j| -2.0 * ax.get(i, j))
        };
        let x0 = Matrix::from_fn(4, 2, |i, j| ((i + 1) as f64 * 0.31 + (j + 1) as f64 * 0.17).sin());
        let start = stiefel_project(&x0);
        let x = riemannian_gradient_descent_stiefel(&f, &grad, &start, 300, 0.05);
        // the iterate stays on the Stiefel manifold: X^T X = I
        let xtx = x.transpose().mul(&x).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (xtx.get(i, j) - want).abs() < 1e-9,
                    "X^T X at {i},{j}: {}",
                    xtx.get(i, j)
                );
            }
        }
        // and it decreases the objective to the closed-form optimum
        assert!(f(&x) < f(&start) - 1e-6, "objective decreased");
        assert!(
            (f(&x) + 5.0).abs() < 1e-6,
            "optimal value {} vs -(3 + 2)",
            f(&x)
        );
        // the columns span e1, e2: the trailing coordinates vanish
        for c in 0..2 {
            assert!(
                x.get(2, c).abs() < 1e-4 && x.get(3, c).abs() < 1e-4,
                "column {c} leaks into the small eigen-directions"
            );
        }
        // a constant objective leaves the projected start point untouched
        let flat = |_: &Matrix| 1.0;
        let zero = |m: &Matrix| Matrix::zeros(m.rows, m.cols);
        let still = riemannian_gradient_descent_stiefel(&flat, &zero, &start, 20, 0.1);
        for i in 0..4 {
            for j in 0..2 {
                assert!((still.get(i, j) - start.get(i, j)).abs() < 1e-9);
            }
        }
    }
}
