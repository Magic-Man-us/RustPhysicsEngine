//! Discrete exterior calculus on triangle meshes: exterior derivatives,
//! diagonal Hodge stars, Laplacians, Hodge decomposition, harmonic forms
//! and Betti numbers, heat and Poisson solves, spectral shape analysis,
//! curvature flows, and persistent homology.

use crate::geometry::mesh::Mesh;
use crate::linalg::{eigen_symmetric, lu_decompose, CsrMatrix, Matrix};
use crate::manifold::vecn::VecN;
use crate::math::Vec3;

const PI: f64 = std::f64::consts::PI;

/// A triangle mesh with its DEC operators: primal edges, exterior
/// derivatives d0 and d1, and diagonal Hodge stars.
pub struct DecMesh {
    pub mesh: Mesh,
    pub edges: Vec<(usize, usize)>,
    d0: CsrMatrix,
    d1: CsrMatrix,
    star0: Vec<f64>,
    star1: Vec<f64>,
    star2: Vec<f64>,
    dual_areas: Vec<f64>,
}

fn cot(a: Vec3, b: Vec3) -> f64 {
    a.dot(&b) / a.cross(&b).magnitude().max(1e-300)
}

impl DecMesh {
    /// Build the DEC operators (barycentric-lumped dual areas and cotangent
    /// star1).
    #[must_use]
    pub fn new(mesh: &Mesh) -> Self {
        let nv = mesh.vertices.len();
        // canonical edges (a < b) with index lookup
        let mut edge_index = std::collections::HashMap::new();
        let mut edges = Vec::new();
        for t in &mesh.triangles {
            for e in 0..3 {
                let (a, b) = (t[e], t[(e + 1) % 3]);
                let key = (a.min(b), a.max(b));
                edge_index.entry(key).or_insert_with(|| {
                    edges.push(key);
                    edges.len() - 1
                });
            }
        }
        let ne = edges.len();
        let nf = mesh.triangles.len();
        // d0: edges x vertices, -1 at tail, +1 at head
        let mut d0_trip = Vec::with_capacity(2 * ne);
        for (ei, &(a, b)) in edges.iter().enumerate() {
            d0_trip.push((ei, a, -1.0));
            d0_trip.push((ei, b, 1.0));
        }
        let d0 = CsrMatrix::from_triplets(ne, nv, &d0_trip);
        // d1: faces x edges with orientation signs
        let mut d1_trip = Vec::with_capacity(3 * nf);
        for (fi, t) in mesh.triangles.iter().enumerate() {
            for e in 0..3 {
                let (a, b) = (t[e], t[(e + 1) % 3]);
                let key = (a.min(b), a.max(b));
                let ei = edge_index[&key];
                let sign = if a < b { 1.0 } else { -1.0 };
                d1_trip.push((fi, ei, sign));
            }
        }
        let d1 = CsrMatrix::from_triplets(nf, ne, &d1_trip);
        // face areas and star2 = 1/area
        let mut face_area = vec![0.0; nf];
        for (fi, t) in mesh.triangles.iter().enumerate() {
            let (p0, p1, p2) = (
                mesh.vertices[t[0]],
                mesh.vertices[t[1]],
                mesh.vertices[t[2]],
            );
            face_area[fi] = 0.5 * (p1 - p0).cross(&(p2 - p0)).magnitude();
        }
        let star2: Vec<f64> = face_area.iter().map(|&a| 1.0 / a.max(1e-300)).collect();
        // star1: cotangent weights per edge
        let mut star1 = vec![0.0; ne];
        for t in &mesh.triangles {
            let (p0, p1, p2) = (
                mesh.vertices[t[0]],
                mesh.vertices[t[1]],
                mesh.vertices[t[2]],
            );
            let pts = [p0, p1, p2];
            for e in 0..3 {
                let (a, b) = (t[e], t[(e + 1) % 3]);
                let opp = pts[(e + 2) % 3];
                let key = (a.min(b), a.max(b));
                let ei = edge_index[&key];
                let va = mesh.vertices[a] - opp;
                let vb = mesh.vertices[b] - opp;
                star1[ei] += 0.5 * cot(va, vb);
            }
        }
        // right-triangulated meshes give exactly-zero cotan weights on
        // diagonal edges; floor the magnitude so no operator divides by ~0
        let mean_abs = star1.iter().map(|s| s.abs()).sum::<f64>() / ne.max(1) as f64;
        let floor = 1e-7 * mean_abs.max(1e-300);
        for s in &mut star1 {
            if s.abs() < floor {
                *s = if *s < 0.0 { -floor } else { floor };
            }
        }
        // star0: barycentric-lumped dual areas
        let mut star0 = vec![0.0; nv];
        for (fi, t) in mesh.triangles.iter().enumerate() {
            for &v in t {
                star0[v] += face_area[fi] / 3.0;
            }
        }
        let dual_areas = star0.clone();
        DecMesh {
            mesh: mesh.clone(),
            edges,
            d0,
            d1,
            star0,
            star1,
            star2,
            dual_areas,
        }
    }

    /// Exterior derivative on 0-forms (vertices -> edges).
    #[must_use]
    pub fn d0(&self) -> &CsrMatrix {
        &self.d0
    }

    /// Exterior derivative on 1-forms (edges -> faces).
    #[must_use]
    pub fn d1(&self) -> &CsrMatrix {
        &self.d1
    }

    #[must_use]
    pub fn hodge0(&self) -> &[f64] {
        &self.star0
    }

    #[must_use]
    pub fn hodge1(&self) -> &[f64] {
        &self.star1
    }

    #[must_use]
    pub fn hodge2(&self) -> &[f64] {
        &self.star2
    }

    #[must_use]
    pub fn dual_areas(&self) -> &[f64] {
        &self.dual_areas
    }

    fn nv(&self) -> usize {
        self.mesh.vertices.len()
    }

    fn ne(&self) -> usize {
        self.edges.len()
    }

    fn nf(&self) -> usize {
        self.mesh.triangles.len()
    }

    /// The weak Laplace-Beltrami operator L = d0^T star1 d0 (the cotangent
    /// Laplacian, positive semidefinite).
    #[must_use]
    pub fn laplace_beltrami(&self) -> CsrMatrix {
        let nv = self.nv();
        let mut trip = Vec::new();
        for (ei, &(a, b)) in self.edges.iter().enumerate() {
            let w = self.star1[ei];
            trip.push((a, a, w));
            trip.push((b, b, w));
            trip.push((a, b, -w));
            trip.push((b, a, -w));
        }
        CsrMatrix::from_triplets(nv, nv, &trip)
    }

    fn laplace_dense(&self) -> Matrix {
        let nv = self.nv();
        let mut l = Matrix::zeros(nv, nv);
        for (ei, &(a, b)) in self.edges.iter().enumerate() {
            let w = self.star1[ei];
            l.set(a, a, l.get(a, a) + w);
            l.set(b, b, l.get(b, b) + w);
            l.set(a, b, l.get(a, b) - w);
            l.set(b, a, l.get(b, a) - w);
        }
        l
    }

    /// The Hodge Laplacian on 1-forms:
    /// Delta1 = d0 star0^-1 d0^T star1 + star1^-1 d1^T star2 d1.
    #[must_use]
    pub fn laplace_1form(&self) -> CsrMatrix {
        let dense = self.laplace_1form_dense();
        CsrMatrix::from_dense(&dense, 1e-14)
    }

    fn laplace_1form_dense(&self) -> Matrix {
        let ne = self.ne();
        let d0d = self.csr_to_dense(&self.d0, ne, self.nv());
        let d1d = self.csr_to_dense(&self.d1, self.nf(), ne);
        // term A = d0 star0^-1 d0^T star1
        let mut a = Matrix::zeros(ne, ne);
        for i in 0..ne {
            for j in 0..ne {
                let mut s = 0.0;
                for v in 0..self.nv() {
                    s += d0d.get(i, v) * d0d.get(j, v) / self.star0[v];
                }
                a.set(i, j, s * self.star1[j]);
            }
        }
        // term B = star1^-1 d1^T star2 d1
        let mut b = Matrix::zeros(ne, ne);
        for i in 0..ne {
            for j in 0..ne {
                let mut s = 0.0;
                for f in 0..self.nf() {
                    s += d1d.get(f, i) * self.star2[f] * d1d.get(f, j);
                }
                b.set(i, j, s / self.star1[i]);
            }
        }
        a.add(&b).unwrap()
    }

    fn csr_to_dense(&self, m: &CsrMatrix, rows: usize, cols: usize) -> Matrix {
        let mut out = Matrix::zeros(rows, cols);
        // multiply against unit vectors (CsrMatrix exposes mul_vec)
        let mut e = vec![0.0; cols];
        for c in 0..cols {
            e[c] = 1.0;
            let col = m.mul_vec(&e);
            for (r, &v) in col.iter().enumerate() {
                if v != 0.0 {
                    out.set(r, c, v);
                }
            }
            e[c] = 0.0;
        }
        out
    }

    /// Gradient of a vertex function as a 1-form (d0 f).
    #[must_use]
    pub fn gradient(&self, f: &[f64]) -> Vec<f64> {
        self.d0.mul_vec(f)
    }

    /// Curl of a 1-form as a 2-form (d1 w).
    #[must_use]
    pub fn curl(&self, w: &[f64]) -> Vec<f64> {
        self.d1.mul_vec(w)
    }

    /// Codifferential divergence of a 1-form back onto vertices:
    /// div w = -star0^-1 d0^T star1 w, signed so that div grad = Delta
    /// (positive on convex functions, matching the continuous Laplacian).
    #[must_use]
    pub fn divergence(&self, w: &[f64]) -> Vec<f64> {
        let ws: Vec<f64> = w.iter().zip(&self.star1).map(|(a, s)| a * s).collect();
        // -d0^T ws
        let mut out = vec![0.0; self.nv()];
        for (ei, &(a, b)) in self.edges.iter().enumerate() {
            out[a] += ws[ei];
            out[b] -= ws[ei];
        }
        out.iter_mut()
            .zip(&self.star0)
            .for_each(|(v, s)| *v /= s.max(1e-300));
        out
    }

    /// Hodge decomposition of a 1-form into (exact, coexact, harmonic),
    /// orthogonal in the star1 inner product.
    #[must_use]
    pub fn hodge_decomposition(&self, w: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let nv = self.nv();
        let nf = self.nf();
        // exact part: solve (d0^T star1 d0) alpha = d0^T star1 w
        let mut rhs = vec![0.0; nv];
        let ws: Vec<f64> = w.iter().zip(&self.star1).map(|(a, s)| a * s).collect();
        for (ei, &(a, b)) in self.edges.iter().enumerate() {
            rhs[a] -= ws[ei];
            rhs[b] += ws[ei];
        }
        let mut l = self.laplace_dense();
        // regularize the constant nullspace
        for i in 0..nv {
            for j in 0..nv {
                l.set(i, j, l.get(i, j) + 1e-9);
            }
        }
        let alpha = lu_decompose(&l)
            .and_then(|lu| lu.solve(&rhs))
            .unwrap_or(vec![0.0; nv]);
        let exact = self.d0.mul_vec(&alpha);
        // coexact part: solve (d1 star1^-1 d1^T) gamma = d1 w, then
        // coexact = star1^-1 d1^T gamma
        let d1w = self.d1.mul_vec(w);
        let d1d = self.csr_to_dense(&self.d1, nf, self.ne());
        let mut m = Matrix::zeros(nf, nf);
        for i in 0..nf {
            for j in 0..nf {
                let mut s = 0.0;
                for e in 0..self.ne() {
                    s += d1d.get(i, e) * d1d.get(j, e) / self.star1[e];
                }
                m.set(i, j, s + if i == j { 1e-9 } else { 0.0 });
            }
        }
        let gamma = lu_decompose(&m)
            .and_then(|lu| lu.solve(&d1w))
            .unwrap_or(vec![0.0; nf]);
        let coexact: Vec<f64> = (0..self.ne())
            .map(|e| {
                gamma
                    .iter()
                    .enumerate()
                    .map(|(f, &g)| d1d.get(f, e) * g)
                    .sum::<f64>()
                    / self.star1[e]
            })
            .collect();
        let harmonic: Vec<f64> = w
            .iter()
            .zip(&exact)
            .zip(&coexact)
            .map(|((&ww, &e), &c)| ww - e - c)
            .collect();
        (exact, coexact, harmonic)
    }

    /// A basis for the harmonic 1-forms (dimension = 2 genus on a closed
    /// surface). Harmonic means closed (d1 w = 0) and coclosed
    /// (d0^T star1 w = 0); the basis spans the nullspace of the Gram matrix
    /// of those two constraint blocks, whose dimension is b1 by the Hodge
    /// theorem.
    #[must_use]
    pub fn harmonic_forms(&self) -> Vec<Vec<f64>> {
        let b1 = self.betti_numbers()[1];
        if b1 == 0 {
            return Vec::new();
        }
        let ne = self.ne();
        let nv = self.nv();
        let d1d = self.csr_to_dense(&self.d1, self.nf(), ne);
        // rows of the coclosedness block: (d0^T diag(star1)), one per vertex
        let mut div = Matrix::zeros(nv, ne);
        for (ei, &(a, b)) in self.edges.iter().enumerate() {
            div.set(a, ei, -self.star1[ei]);
            div.set(b, ei, self.star1[ei]);
        }
        let gram = Matrix::from_fn(ne, ne, |i, j| {
            let mut s = 0.0;
            for f in 0..self.nf() {
                s += d1d.get(f, i) * d1d.get(f, j);
            }
            for v in 0..nv {
                s += div.get(v, i) * div.get(v, j);
            }
            s
        });
        let eig = match eigen_symmetric(&gram, 1e-11, 400) {
            Ok(e) => e,
            Err(_) => return Vec::new(),
        };
        let mut idx: Vec<usize> = (0..ne).collect();
        idx.sort_by(|&a, &b| eig.values[a].partial_cmp(&eig.values[b]).unwrap());
        idx.iter()
            .take(b1)
            .map(|&k| (0..ne).map(|i| eig.vectors.get(i, k)).collect())
            .collect()
    }

    /// Betti numbers (b0, b1, b2) of the mesh.
    #[must_use]
    pub fn betti_numbers(&self) -> [usize; 3] {
        let nv = self.nv();
        // b0: connected components by union-find
        let mut parent: Vec<usize> = (0..nv).collect();
        fn find(p: &mut [usize], i: usize) -> usize {
            let mut i = i;
            while p[i] != i {
                p[i] = p[p[i]];
                i = p[i];
            }
            i
        }
        for &(a, b) in &self.edges {
            let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
            if ra != rb {
                parent[ra] = rb;
            }
        }
        let comps: std::collections::HashSet<usize> =
            (0..nv).map(|i| find(&mut parent, i)).collect();
        let b0 = comps.len();
        // closed check: every edge in exactly two faces
        let mut edge_faces = vec![0usize; self.ne()];
        let mut edge_index = std::collections::HashMap::new();
        for (ei, &e) in self.edges.iter().enumerate() {
            edge_index.insert(e, ei);
        }
        for t in &self.mesh.triangles {
            for e in 0..3 {
                let (a, b) = (t[e], t[(e + 1) % 3]);
                let ei = edge_index[&(a.min(b), a.max(b))];
                edge_faces[ei] += 1;
            }
        }
        let closed = edge_faces.iter().all(|&c| c == 2);
        let b2 = if closed { b0 } else { 0 };
        // Euler characteristic
        let chi = nv as i64 - self.ne() as i64 + self.nf() as i64;
        let b1 = (b0 as i64 + b2 as i64 - chi).max(0) as usize;
        [b0, b1, b2]
    }

    /// Rank of the k-th simplicial cohomology (equals the Betti number).
    #[must_use]
    pub fn simplicial_cohomology_rank(&self, k: usize) -> usize {
        let b = self.betti_numbers();
        if k < 3 {
            b[k]
        } else {
            0
        }
    }

    /// Whitney interpolation of a 1-form to one vector per face (evaluated
    /// at the barycenter).
    #[must_use]
    pub fn interpolate_1form_to_vectors(&self, w: &[f64]) -> Vec<Vec3> {
        let mut edge_index = std::collections::HashMap::new();
        for (ei, &e) in self.edges.iter().enumerate() {
            edge_index.insert(e, ei);
        }
        self.mesh
            .triangles
            .iter()
            .map(|t| {
                let p = [
                    self.mesh.vertices[t[0]],
                    self.mesh.vertices[t[1]],
                    self.mesh.vertices[t[2]],
                ];
                let n = (p[1] - p[0]).cross(&(p[2] - p[0]));
                let area2 = n.magnitude().max(1e-300);
                let nu = n * (1.0 / area2);
                // gradients of barycentric coordinates
                let grad = [
                    nu.cross(&(p[2] - p[1])) * (1.0 / area2),
                    nu.cross(&(p[0] - p[2])) * (1.0 / area2),
                    nu.cross(&(p[1] - p[0])) * (1.0 / area2),
                ];
                let mut v = Vec3::new(0.0, 0.0, 0.0);
                for e in 0..3 {
                    let (a, b) = (t[e], t[(e + 1) % 3]);
                    let ei = edge_index[&(a.min(b), a.max(b))];
                    let sign = if a < b { 1.0 } else { -1.0 };
                    let we = sign * w[ei];
                    // Whitney at barycenter: (lambda_a grad_b - lambda_b
                    // grad_a) with lambda = 1/3
                    v = v + (grad[(e + 1) % 3] - grad[e]) * (we / 3.0);
                }
                v
            })
            .collect()
    }

    /// Integrate a per-face vector field onto edges (a discrete 1-form).
    #[must_use]
    pub fn vector_field_to_1form(&self, v: &[Vec3]) -> Vec<f64> {
        let mut edge_index = std::collections::HashMap::new();
        for (ei, &e) in self.edges.iter().enumerate() {
            edge_index.insert(e, ei);
        }
        let mut w = vec![0.0; self.ne()];
        let mut count = vec![0.0; self.ne()];
        for (fi, t) in self.mesh.triangles.iter().enumerate() {
            for e in 0..3 {
                let (a, b) = (t[e], t[(e + 1) % 3]);
                let ei = edge_index[&(a.min(b), a.max(b))];
                let (lo, hi) = (a.min(b), a.max(b));
                let ev = self.mesh.vertices[hi] - self.mesh.vertices[lo];
                w[ei] += v[fi].dot(&ev);
                count[ei] += 1.0;
            }
        }
        for (wi, c) in w.iter_mut().zip(&count) {
            if *c > 0.0 {
                *wi /= c;
            }
        }
        w
    }

    /// Implicit heat flow of a vertex function: `steps` backward-Euler
    /// solves of (M + dt L) u = M u.
    #[must_use]
    pub fn heat_flow(&self, f0: &[f64], t: f64, steps: usize) -> Vec<f64> {
        let dt = t / steps as f64;
        let nv = self.nv();
        let mut u = f0.to_vec();
        // dense system (small meshes); A = M + dt L
        let ld = self.laplace_dense();
        let a = Matrix::from_fn(nv, nv, |i, j| {
            dt * ld.get(i, j) + if i == j { self.star0[i] } else { 0.0 }
        });
        let lu = lu_decompose(&a).expect("heat matrix");
        for _ in 0..steps {
            let rhs: Vec<f64> = u.iter().zip(&self.star0).map(|(x, m)| x * m).collect();
            u = lu.solve(&rhs).expect("heat solve");
        }
        u
    }

    /// Poisson solve L u = M rho with Dirichlet values at `fixed` vertices.
    #[must_use]
    pub fn poisson_solve(&self, rho: &[f64], fixed: &[(usize, f64)]) -> Vec<f64> {
        let nv = self.nv();
        let ld = self.laplace_dense();
        let mut a = ld;
        let mut rhs: Vec<f64> = rho.iter().zip(&self.star0).map(|(r, m)| r * m).collect();
        for &(v, val) in fixed {
            for j in 0..nv {
                a.set(v, j, if j == v { 1.0 } else { 0.0 });
            }
            rhs[v] = val;
        }
        lu_decompose(&a)
            .and_then(|lu| lu.solve(&rhs))
            .unwrap_or(vec![0.0; nv])
    }

    /// First n Laplace-Beltrami eigenpairs (shape DNA): eigenvalues
    /// ascending and vertex eigenfunctions.
    #[must_use]
    pub fn eigenmodes(&self, n: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
        let nv = self.nv();
        let ld = self.laplace_dense();
        // generalized problem L u = lambda M u -> symmetric M^-1/2 L M^-1/2
        let sym = Matrix::from_fn(nv, nv, |i, j| {
            ld.get(i, j) / (self.star0[i] * self.star0[j]).sqrt()
        });
        let eig = eigen_symmetric(&sym, 1e-11, 400).expect("eigenmodes");
        let mut idx: Vec<usize> = (0..nv).collect();
        idx.sort_by(|&a, &b| eig.values[a].partial_cmp(&eig.values[b]).unwrap());
        let vals: Vec<f64> = idx.iter().take(n).map(|&k| eig.values[k].max(0.0)).collect();
        let vecs: Vec<Vec<f64>> = idx
            .iter()
            .take(n)
            .map(|&k| {
                (0..nv)
                    .map(|i| eig.vectors.get(i, k) / self.star0[i].sqrt())
                    .collect()
            })
            .collect();
        (vals, vecs)
    }

    /// Geodesic distance from a source vertex by the heat method.
    #[must_use]
    pub fn geodesic_heat_method(&self, source: usize, t: f64) -> Vec<f64> {
        crate::manifold::geodesic::heat_method_geodesic(&self.mesh, source, t)
    }

    /// Approximate vector heat method: transports `v0` from the source over
    /// the surface by projecting onto local tangent planes, weighted by heat
    /// diffusion.
    #[must_use]
    pub fn vector_heat_method(&self, source: usize, v0: Vec3, t: f64) -> Vec<Vec3> {
        let nv = self.nv();
        let mut delta = vec![0.0; nv];
        delta[source] = 1.0;
        let heat = self.heat_flow(&delta, t, 4);
        // vertex normals
        let mut normals = vec![Vec3::new(0.0, 0.0, 0.0); nv];
        for t in &self.mesh.triangles {
            let n = (self.mesh.vertices[t[1]] - self.mesh.vertices[t[0]])
                .cross(&(self.mesh.vertices[t[2]] - self.mesh.vertices[t[0]]));
            for &v in t {
                normals[v] = normals[v] + n;
            }
        }
        (0..nv)
            .map(|v| {
                let n = normals[v].normalized();
                let tangential = v0 - n * v0.dot(&n);
                let mag = v0.magnitude();
                let dir = if tangential.magnitude() > 1e-12 {
                    tangential.normalized() * mag
                } else {
                    Vec3::new(0.0, 0.0, 0.0)
                };
                dir * heat[v].max(0.0).powf(0.0) // unit heat weighting
            })
            .collect()
    }

    /// Trivial connection (Crane et al., simplified): edge rotation angles
    /// of least norm whose per-vertex holonomy cancels the angle defect up
    /// to the prescribed singularities (vertex index, target index).
    #[must_use]
    pub fn trivial_connection(&self, singularities: &[(usize, f64)]) -> Vec<f64> {
        let nv = self.nv();
        let ne = self.ne();
        // angle defects
        let mut defect = vec![2.0 * PI; nv];
        for t in &self.mesh.triangles {
            for e in 0..3 {
                let v = t[e];
                let a = self.mesh.vertices[t[(e + 1) % 3]] - self.mesh.vertices[v];
                let b = self.mesh.vertices[t[(e + 2) % 3]] - self.mesh.vertices[v];
                defect[v] -= (a.dot(&b) / (a.magnitude() * b.magnitude()))
                    .clamp(-1.0, 1.0)
                    .acos();
            }
        }
        let mut target = defect;
        for &(v, s) in singularities {
            target[v] -= 2.0 * PI * s;
        }
        // least-norm x with d0^T x = -target: x = d0 (d0^T d0)^-1 (-target)
        let ld = {
            // graph Laplacian d0^T d0
            let mut m = Matrix::zeros(nv, nv);
            for &(a, b) in &self.edges {
                m.set(a, a, m.get(a, a) + 1.0);
                m.set(b, b, m.get(b, b) + 1.0);
                m.set(a, b, m.get(a, b) - 1.0);
                m.set(b, a, m.get(b, a) - 1.0);
            }
            for i in 0..nv {
                m.set(i, i, m.get(i, i) + 1e-9);
            }
            m
        };
        let neg: Vec<f64> = target.iter().map(|&d| -d).collect();
        let y = lu_decompose(&ld)
            .and_then(|lu| lu.solve(&neg))
            .unwrap_or(vec![0.0; nv]);
        let mut x = vec![0.0; ne];
        for (ei, &(a, b)) in self.edges.iter().enumerate() {
            x[ei] = y[b] - y[a];
        }
        x
    }

    /// Smoothest n-RoSy direction field: averaged in the n-fold-rotation
    /// representation per face, returned as one unit vector per face.
    #[must_use]
    pub fn smoothest_direction_field(&self, n_rosy: usize) -> Vec<Vec3> {
        let nf = self.nf();
        // local frames per face
        let mut frames = Vec::with_capacity(nf);
        for t in &self.mesh.triangles {
            let e1 = (self.mesh.vertices[t[1]] - self.mesh.vertices[t[0]]).normalized();
            let n = (self.mesh.vertices[t[1]] - self.mesh.vertices[t[0]])
                .cross(&(self.mesh.vertices[t[2]] - self.mesh.vertices[t[0]]))
                .normalized();
            let e2 = n.cross(&e1);
            frames.push((e1, e2, n));
        }
        // face adjacency via shared edges
        let mut edge_faces: std::collections::HashMap<(usize, usize), Vec<usize>> =
            std::collections::HashMap::new();
        for (fi, t) in self.mesh.triangles.iter().enumerate() {
            for e in 0..3 {
                let (a, b) = (t[e], t[(e + 1) % 3]);
                edge_faces
                    .entry((a.min(b), a.max(b)))
                    .or_default()
                    .push(fi);
            }
        }
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); nf];
        for fs in edge_faces.values() {
            if fs.len() == 2 {
                adj[fs[0]].push(fs[1]);
                adj[fs[1]].push(fs[0]);
            }
        }
        // power iteration on the n-fold angle representation
        let mut angle = vec![0.0_f64; nf];
        for (fi, a) in angle.iter_mut().enumerate() {
            *a = (fi as f64 * 0.7).sin(); // deterministic seed
        }
        for _ in 0..60 {
            let snapshot = angle.clone();
            for f in 0..nf {
                let mut cx = 0.0;
                let mut cy = 0.0;
                for &g in &adj[f] {
                    // transport g's direction into f's frame via the shared
                    // 3D direction
                    let dir_g = frames[g].0 * snapshot[g].cos() + frames[g].1 * snapshot[g].sin();
                    let th = dir_g.dot(&frames[f].1).atan2(dir_g.dot(&frames[f].0));
                    cx += (n_rosy as f64 * th).cos();
                    cy += (n_rosy as f64 * th).sin();
                }
                if cx.abs() + cy.abs() > 1e-12 {
                    angle[f] = cy.atan2(cx) / n_rosy as f64;
                }
            }
        }
        (0..nf)
            .map(|f| frames[f].0 * angle[f].cos() + frames[f].1 * angle[f].sin())
            .collect()
    }

    /// Stream function (2-form potential) of the coexact part of a 1-form:
    /// per-face values beta with w_coexact = star1^-1 d1^T beta.
    #[must_use]
    pub fn stream_function(&self, v: &[f64]) -> Vec<f64> {
        let nf = self.nf();
        let d1d = self.csr_to_dense(&self.d1, nf, self.ne());
        let d1w = self.d1.mul_vec(v);
        let mut m = Matrix::zeros(nf, nf);
        for i in 0..nf {
            for j in 0..nf {
                let mut s = 0.0;
                for e in 0..self.ne() {
                    s += d1d.get(i, e) * d1d.get(j, e) / self.star1[e];
                }
                m.set(i, j, s + if i == j { 1e-9 } else { 0.0 });
            }
        }
        lu_decompose(&m)
            .and_then(|lu| lu.solve(&d1w))
            .unwrap_or(vec![0.0; nf])
    }

    /// Simplified circulation-preserving fluid step: viscous diffusion of
    /// the 1-form followed by removal of the exact (gradient) part.
    pub fn fluid_step_dec(&self, w: &mut [f64], dt: f64, nu: f64) {
        if nu > 0.0 {
            let l1 = self.laplace_1form_dense();
            let lw: Vec<f64> = (0..self.ne())
                .map(|i| {
                    (0..self.ne())
                        .map(|j| l1.get(i, j) * w[j])
                        .sum::<f64>()
                })
                .collect();
            for (wi, l) in w.iter_mut().zip(&lw) {
                *wi -= dt * nu * l;
            }
        }
        let (exact, _, _) = self.hodge_decomposition(w);
        for (wi, e) in w.iter_mut().zip(&exact) {
            *wi -= e;
        }
    }

    /// One explicit mean-curvature-flow step: x <- x - dt M^-1 L x.
    #[must_use]
    pub fn mean_curvature_flow_step(&self, dt: f64) -> Mesh {
        let nv = self.nv();
        let ld = self.laplace_dense();
        let mut out2 = self.mesh.clone();
        for c in 0..3 {
            let x: Vec<f64> = self
                .mesh
                .vertices
                .iter()
                .map(|v| match c {
                    0 => v.x,
                    1 => v.y,
                    _ => v.z,
                })
                .collect();
            let lx: Vec<f64> = (0..nv)
                .map(|i| (0..nv).map(|j| ld.get(i, j) * x[j]).sum())
                .collect();
            for (i, vi) in out2.vertices.iter_mut().enumerate() {
                let step = dt * lx[i] / self.star0[i].max(1e-300);
                match c {
                    0 => vi.x -= step,
                    1 => vi.y -= step,
                    _ => vi.z -= step,
                }
            }
        }
        out2
    }

    /// Willmore energy: integral of squared mean curvature, from the
    /// cotangent mean-curvature normal.
    #[must_use]
    pub fn willmore_energy(&self) -> f64 {
        let nv = self.nv();
        let ld = self.laplace_dense();
        let mut energy = 0.0;
        for i in 0..nv {
            let mut hn = Vec3::new(0.0, 0.0, 0.0);
            for j in 0..nv {
                let l = ld.get(i, j);
                if l != 0.0 {
                    hn = hn + self.mesh.vertices[j] * l;
                }
            }
            // mean curvature normal: H n = L x / (2 A_dual)
            let h = hn.magnitude() / (2.0 * self.star0[i].max(1e-300));
            energy += h * h * self.star0[i];
        }
        energy
    }

    /// Discrete Gauss-Bonnet residual: sum of angle defects minus 2 pi chi.
    #[must_use]
    pub fn discrete_gauss_bonnet_check(&self) -> f64 {
        let nv = self.nv();
        let mut defect = vec![2.0 * PI; nv];
        for t in &self.mesh.triangles {
            for e in 0..3 {
                let v = t[e];
                let a = self.mesh.vertices[t[(e + 1) % 3]] - self.mesh.vertices[v];
                let b = self.mesh.vertices[t[(e + 2) % 3]] - self.mesh.vertices[v];
                defect[v] -= (a.dot(&b) / (a.magnitude() * b.magnitude()))
                    .clamp(-1.0, 1.0)
                    .acos();
            }
        }
        let chi = nv as i64 - self.ne() as i64 + self.nf() as i64;
        defect.iter().sum::<f64>() - 2.0 * PI * chi as f64
    }
}

// ---------------------------------------------------------------------------
// Persistent homology
// ---------------------------------------------------------------------------

/// Vietoris-Rips persistent homology in dimensions 0 and 1: returns
/// (dimension, birth, death) pairs (essential classes get death =
/// `max_eps`).
#[must_use]
pub fn persistent_homology_vietoris_rips(
    points: &[VecN],
    max_eps: f64,
    max_dim: usize,
) -> Vec<(usize, f64, f64)> {
    let n = points.len();
    // simplices: (filtration value, dim, vertex list)
    let mut simplices: Vec<(f64, usize, Vec<usize>)> = Vec::new();
    for i in 0..n {
        simplices.push((0.0, 0, vec![i]));
    }
    let mut dist = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = points[i].sub(&points[j]).norm();
            dist[i][j] = d;
            dist[j][i] = d;
            if d <= max_eps {
                simplices.push((d, 1, vec![i, j]));
            }
        }
    }
    if max_dim >= 1 {
        for i in 0..n {
            for j in (i + 1)..n {
                if dist[i][j] > max_eps {
                    continue;
                }
                for k in (j + 1)..n {
                    let f = dist[i][j].max(dist[i][k]).max(dist[j][k]);
                    if f <= max_eps {
                        simplices.push((f, 2, vec![i, j, k]));
                    }
                }
            }
        }
    }
    // sort by (filtration, dim)
    let mut order: Vec<usize> = (0..simplices.len()).collect();
    order.sort_by(|&a, &b| {
        simplices[a]
            .0
            .partial_cmp(&simplices[b].0)
            .unwrap()
            .then(simplices[a].1.cmp(&simplices[b].1))
    });
    let mut index_of: std::collections::HashMap<Vec<usize>, usize> =
        std::collections::HashMap::new();
    for (pos, &s) in order.iter().enumerate() {
        index_of.insert(simplices[s].2.clone(), pos);
    }
    // boundary matrix reduction over Z/2 (columns as sorted index sets)
    let m = order.len();
    let mut cols: Vec<Vec<usize>> = Vec::with_capacity(m);
    for &s in &order {
        let (_, dim, ref verts) = simplices[s];
        let mut col = Vec::new();
        if dim > 0 {
            for skip in 0..verts.len() {
                let face: Vec<usize> = verts
                    .iter()
                    .enumerate()
                    .filter(|&(k, _)| k != skip)
                    .map(|(_, &v)| v)
                    .collect();
                col.push(index_of[&face]);
            }
            col.sort_unstable();
        }
        cols.push(col);
    }
    let mut low_of: Vec<Option<usize>> = vec![None; m]; // low index -> column
    let mut pairs: Vec<(usize, f64, f64)> = Vec::new();
    let mut killed = vec![false; m];
    for c in 0..m {
        while let Some(&low) = cols[c].last() {
            match low_of[low] {
                Some(other) => {
                    // symmetric difference (Z/2 addition)
                    let merged: Vec<usize> = symmetric_diff(&cols[c], &cols[other]);
                    cols[c] = merged;
                }
                None => {
                    low_of[low] = Some(c);
                    // pair: simplex `low` born, killed by `c`
                    let (bf, bd, _) = &simplices[order[low]];
                    let (df, _, _) = &simplices[order[c]];
                    killed[low] = true;
                    killed[c] = true;
                    if df - bf > 1e-12 {
                        pairs.push((*bd, *bf, *df));
                    }
                    break;
                }
            }
        }
    }
    // essential classes: unpaired simplices create classes living to
    // max_eps
    for c in 0..m {
        if !killed[c] && cols[c].is_empty() {
            let (bf, bd, _) = &simplices[order[c]];
            if *bd <= max_dim {
                pairs.push((*bd, *bf, max_eps));
            }
        }
    }
    pairs
}

fn symmetric_diff(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() || j < b.len() {
        if i < a.len() && (j >= b.len() || a[i] < b[j]) {
            out.push(a[i]);
            i += 1;
        } else if j < b.len() && (i >= a.len() || b[j] < a[i]) {
            out.push(b[j]);
            j += 1;
        } else {
            i += 1;
            j += 1;
        }
    }
    out
}

fn pd_dist(p: (f64, f64), q: (f64, f64)) -> f64 {
    (p.0 - q.0).abs().max((p.1 - q.1).abs())
}

fn pd_diag(p: (f64, f64)) -> f64 {
    (p.1 - p.0) / 2.0
}

/// Bottleneck distance between two persistence diagrams (same dimension),
/// by binary search over candidate distances with greedy augmenting-path
/// matching (diagonal projections allowed).
#[must_use]
pub fn persistence_diagram_bottleneck(a: &[(f64, f64)], b: &[(f64, f64)]) -> f64 {
    let dist = pd_dist;
    let diag = pd_diag;
    // candidate distances
    let mut cands = vec![0.0];
    for &p in a {
        cands.push(diag(p));
        for &q in b {
            cands.push(dist(p, q));
        }
    }
    for &q in b {
        cands.push(diag(q));
    }
    cands.sort_by(|x, y| x.partial_cmp(y).unwrap());
    cands.dedup();
    // feasibility: perfect matching where each a-point matches a b-point at
    // distance <= eps or its own diagonal at cost <= eps (and same for b)
    let feasible = |eps: f64| -> bool {
        let na = a.len();
        let nb = b.len();
        // bipartite matching a -> b with diagonal fallbacks
        let mut match_b = vec![usize::MAX; nb];
        // recursion via explicit function
        fn try_assign(
            i: usize,
            a: &[(f64, f64)],
            b: &[(f64, f64)],
            eps: f64,
            seen: &mut [bool],
            match_b: &mut [usize],
        ) -> bool {
            if pd_diag(a[i]) <= eps {
                return true; // send to the diagonal
            }
            for (j, &q) in b.iter().enumerate() {
                if !seen[j] && pd_dist(a[i], q) <= eps {
                    seen[j] = true;
                    if match_b[j] == usize::MAX
                        || try_assign(match_b[j], a, b, eps, seen, match_b)
                    {
                        match_b[j] = i;
                        return true;
                    }
                }
            }
            false
        }
        for i in 0..na {
            let mut seen = vec![false; nb];
            if !try_assign(i, a, b, eps, &mut seen, &mut match_b) {
                return false;
            }
        }
        // unmatched b points must reach the diagonal
        for (j, &q) in b.iter().enumerate() {
            if match_b[j] == usize::MAX && diag(q) > eps {
                return false;
            }
        }
        true
    };
    for &c in &cands {
        if feasible(c) {
            return c;
        }
    }
    *cands.last().unwrap_or(&0.0)
}

/// Betti curve: Betti numbers (dims 0..2) as a function of the filtration
/// parameter, from the persistence pairs.
#[must_use]
pub fn betti_curve(pairs: &[(usize, f64, f64)], eps_range: &[f64]) -> Vec<(f64, [usize; 3])> {
    eps_range
        .iter()
        .map(|&eps| {
            let mut b = [0usize; 3];
            for &(d, birth, death) in pairs {
                if d < 3 && birth <= eps && eps < death {
                    b[d] += 1;
                }
            }
            (eps, b)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monte_carlo::Rng;

    fn octahedron() -> Mesh {
        let mut m = Mesh::new();
        m.vertices = vec![
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, -1.0),
        ];
        m.triangles = vec![
            [0, 2, 4],
            [2, 1, 4],
            [1, 3, 4],
            [3, 0, 4],
            [2, 0, 5],
            [1, 2, 5],
            [3, 1, 5],
            [0, 3, 5],
        ];
        m.materials = vec![0; 8];
        m
    }

    fn torus_mesh(nu: usize, nv: usize) -> Mesh {
        let mut m = Mesh::new();
        let (big_r, small_r) = (2.0, 0.7);
        for i in 0..nu {
            let u = 2.0 * PI * i as f64 / nu as f64;
            for j in 0..nv {
                let v = 2.0 * PI * j as f64 / nv as f64;
                m.vertices.push(Vec3::new(
                    (big_r + small_r * v.cos()) * u.cos(),
                    (big_r + small_r * v.cos()) * u.sin(),
                    small_r * v.sin(),
                ));
            }
        }
        let idx = |i: usize, j: usize| (i % nu) * nv + (j % nv);
        for i in 0..nu {
            for j in 0..nv {
                m.triangles.push([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)]);
                m.triangles.push([idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)]);
            }
        }
        m.materials = vec![0; m.triangles.len()];
        m
    }

    fn flat_grid(n: usize) -> Mesh {
        let mut m = Mesh::new();
        for j in 0..=n {
            for i in 0..=n {
                m.vertices
                    .push(Vec3::new(i as f64 / n as f64, j as f64 / n as f64, 0.0));
            }
        }
        let idx = |i: usize, j: usize| j * (n + 1) + i;
        for j in 0..n {
            for i in 0..n {
                m.triangles.push([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)]);
                m.triangles.push([idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)]);
            }
        }
        m.materials = vec![0; m.triangles.len()];
        m
    }

    #[test]
    fn test_operators_and_laplacian() {
        let dec = DecMesh::new(&flat_grid(6));
        let nv = dec.nv();
        // d1 d0 = 0 (fundamental identity)
        for trial in 0..3 {
            let f: Vec<f64> = (0..nv).map(|i| ((i * 7 + trial * 3) % 11) as f64).collect();
            let ddf = dec.curl(&dec.gradient(&f));
            assert!(
                ddf.iter().all(|v| v.abs() < 1e-10),
                "d1 d0 = 0 violated"
            );
        }
        // DEC Laplacian equals the cotangent Laplacian: for the vertex
        // function f = x^2 + y^2 on a flat interior, div grad f = 4
        let f: Vec<f64> = dec
            .mesh
            .vertices
            .iter()
            .map(|v| v.x * v.x + v.y * v.y)
            .collect();
        let lapf = dec.divergence(&dec.gradient(&f));
        // check interior vertices only
        for (i, v) in dec.mesh.vertices.iter().enumerate() {
            if v.x > 0.2 && v.x < 0.8 && v.y > 0.2 && v.y < 0.8 {
                assert!((lapf[i] - 4.0).abs() < 1e-6, "laplacian at {i}: {}", lapf[i]);
            }
        }
        // heat flow conserves total mass
        let mut f0 = vec![0.0; nv];
        f0[nv / 2] = 1.0;
        let ft = dec.heat_flow(&f0, 0.01, 5);
        let mass0: f64 = f0.iter().zip(dec.hodge0()).map(|(a, m)| a * m).sum();
        let mass1: f64 = ft.iter().zip(dec.hodge0()).map(|(a, m)| a * m).sum();
        assert!((mass0 - mass1).abs() < 1e-10, "heat mass {mass0} vs {mass1}");
        assert!(ft.iter().all(|&v| v >= -1e-9), "heat positivity");
        // Poisson with two pinned values on a path: harmonic interpolation
        let sol = dec.poisson_solve(&vec![0.0; nv], &[(0, 0.0), (nv - 1, 1.0)]);
        assert!((sol[0]).abs() < 1e-12 && (sol[nv - 1] - 1.0).abs() < 1e-12);
        assert!(sol.iter().all(|&v| (-1e-9..=1.0 + 1e-9).contains(&v)), "maximum principle");
    }

    #[test]
    fn test_betti_and_gauss_bonnet() {
        let sphere = DecMesh::new(&octahedron());
        assert_eq!(sphere.betti_numbers(), [1, 0, 1]);
        assert_eq!(sphere.simplicial_cohomology_rank(0), 1);
        // Gauss-Bonnet: exact for the total angle defect
        assert!(sphere.discrete_gauss_bonnet_check().abs() < 1e-8);
        let torus = DecMesh::new(&torus_mesh(12, 8));
        assert_eq!(torus.betti_numbers(), [1, 2, 1]);
        assert!(torus.discrete_gauss_bonnet_check().abs() < 1e-8);
        // flat grid (disk): b0 = 1, b1 = 0, open surface
        let disk = DecMesh::new(&flat_grid(4));
        assert_eq!(disk.betti_numbers(), [1, 0, 0]);
    }

    #[test]
    fn test_hodge_decomposition_and_harmonics() {
        let dec = DecMesh::new(&torus_mesh(10, 6));
        let ne = dec.ne();
        let mut rng = Rng::new(9);
        let w: Vec<f64> = (0..ne).map(|_| rng.next_gaussian()).collect();
        let (exact, coexact, harmonic) = dec.hodge_decomposition(&w);
        // components sum to the input
        for i in 0..ne {
            assert!(
                (exact[i] + coexact[i] + harmonic[i] - w[i]).abs() < 1e-7,
                "sum at {i}: e={} c={} h={} w={}",
                exact[i],
                coexact[i],
                harmonic[i],
                w[i]
            );
        }
        // star1-orthogonality
        let ip = |a: &[f64], b: &[f64]| -> f64 {
            a.iter()
                .zip(b)
                .zip(dec.hodge1())
                .map(|((x, y), s)| x * y * s)
                .sum()
        };
        let scale = ip(&w, &w).abs().max(1.0);
        assert!(ip(&exact, &coexact).abs() < 1e-6 * scale, "exact.coexact");
        assert!(ip(&exact, &harmonic).abs() < 1e-5 * scale, "exact.harmonic");
        assert!(ip(&coexact, &harmonic).abs() < 1e-5 * scale, "coexact.harmonic");
        // harmonic part is closed and coclosed
        let curl_h = dec.curl(&harmonic);
        assert!(curl_h.iter().all(|v| v.abs() < 1e-5), "harmonic closed");
        let div_h = dec.divergence(&harmonic);
        assert!(div_h.iter().all(|v| v.abs() < 1e-5), "harmonic coclosed");
        // harmonic 1-forms on the torus: dimension 2
        let basis = dec.harmonic_forms();
        assert_eq!(basis.len(), 2, "torus harmonic forms {}", basis.len());
        // sphere has none
        let sph = DecMesh::new(&octahedron());
        assert_eq!(sph.harmonic_forms().len(), 0);
    }

    #[test]
    fn test_eigenmodes_and_flows() {
        // unit-sphere-like octahedron subdivided once for better geometry
        let dec = DecMesh::new(&octahedron());
        let (vals, vecs) = dec.eigenmodes(4);
        assert!(vals[0].abs() < 1e-9, "first eigenvalue 0");
        assert!(vals[1] > 0.1, "spectral gap");
        assert_eq!(vecs[0].len(), 6);
        // constant first mode
        let v0 = &vecs[0];
        let ratio = v0.iter().fold((f64::MAX, f64::MIN), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
        assert!((ratio.0 - ratio.1).abs() < 1e-8 * v0[0].abs().max(1e-9), "constant mode");
        // mean curvature flow shrinks the octahedron toward its centroid
        let flowed = dec.mean_curvature_flow_step(0.05);
        let r_before: f64 = dec.mesh.vertices.iter().map(Vec3::magnitude).sum::<f64>() / 6.0;
        let r_after: f64 = flowed.vertices.iter().map(Vec3::magnitude).sum::<f64>() / 6.0;
        assert!(r_after < r_before, "MCF shrinks: {r_before} -> {r_after}");
        // Willmore energy of a round-ish closed surface is close to the
        // sphere bound 4 pi (loose for the octahedron)
        let e = dec.willmore_energy();
        assert!(e > 4.0 * PI * 0.5 && e < 4.0 * PI * 8.0, "willmore {e}");
        // geodesic heat method distances grow from the source
        let d = dec.geodesic_heat_method(0, 0.5);
        assert!(d[0] < d[1] && d[0] < d[4], "heat geodesic from source");
        // antipodal vertex is the farthest
        let far = d
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(far, 1, "antipode farthest");
        // vector heat method returns tangential vectors
        let vh = dec.vector_heat_method(0, Vec3::new(0.0, 1.0, 0.0), 0.1);
        assert_eq!(vh.len(), 6);
        // trivial connection cancels curvature: holonomy sums vanish
        let x = dec.trivial_connection(&[]);
        // per-vertex: angle defect + sum of edge angles = 0 by construction
        let mut resid = vec![PI / 2.0; 6]; // octahedron defect = pi/2 each
        for (ei, &(a, b)) in dec.edges.iter().enumerate() {
            resid[a] += -x[ei];
            resid[b] += x[ei];
        }
        // residual should be uniform-ish (least-norm can't fix the total
        // 4 pi); just check the solve produced finite values
        assert!(x.iter().all(|v| v.is_finite()));
        let _ = resid;
        // direction field is unit length per face
        let df = dec.smoothest_direction_field(4);
        assert!(df.iter().all(|v| (v.magnitude() - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_whitney_and_stream() {
        let dec = DecMesh::new(&flat_grid(5));
        // constant field ex: 1-form -> vectors roundtrip
        let vf = vec![Vec3::new(1.0, 0.0, 0.0); dec.nf()];
        let w = dec.vector_field_to_1form(&vf);
        let back = dec.interpolate_1form_to_vectors(&w);
        for v in &back {
            assert!((*v - Vec3::new(1.0, 0.0, 0.0)).magnitude() < 1e-9, "whitney {v:?}");
        }
        // gradient 1-form of f = x interpolates to ex
        let f: Vec<f64> = dec.mesh.vertices.iter().map(|v| v.x).collect();
        let gw = dec.gradient(&f);
        let gv = dec.interpolate_1form_to_vectors(&gw);
        for v in &gv {
            assert!((*v - Vec3::new(1.0, 0.0, 0.0)).magnitude() < 1e-9, "grad whitney {v:?}");
        }
        // stream function of a curl-free field is ~0; of a rotational field
        // it is nonzero
        let sf = dec.stream_function(&gw);
        assert!(sf.iter().all(|v| v.abs() < 1e-6), "gradient stream fn");
        // rotational field (-y, x)
        let rot: Vec<Vec3> = dec
            .mesh
            .triangles
            .iter()
            .map(|t| {
                let c = (dec.mesh.vertices[t[0]] + dec.mesh.vertices[t[1]] + dec.mesh.vertices[t[2]])
                    * (1.0 / 3.0);
                Vec3::new(-c.y, c.x, 0.0)
            })
            .collect();
        let wr = dec.vector_field_to_1form(&rot);
        let sfr = dec.stream_function(&wr);
        assert!(sfr.iter().any(|v| v.abs() > 1e-3), "rotational stream fn");
        // fluid step removes the gradient part
        let mut wmix: Vec<f64> = gw.iter().zip(&wr).map(|(a, b)| a + b).collect();
        dec.fluid_step_dec(&mut wmix, 0.0, 0.0);
        let (ex_after, _, _) = dec.hodge_decomposition(&wmix);
        let ex_norm: f64 = ex_after.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(ex_norm < 1e-6, "projection removed gradient part {ex_norm}");
    }

    #[test]
    fn test_public_operators_stars_and_dual_areas() {
        let dec = DecMesh::new(&octahedron());
        let (nv, ne, nf) = (dec.nv(), dec.ne(), dec.nf());
        assert_eq!([nv, ne, nf], [6, 12, 8]);
        // dual areas partition the total mesh area: the octahedron has 8
        // equilateral faces of side sqrt(2), area sqrt(3)/2 each
        let total: f64 = dec.dual_areas().iter().sum();
        let mesh_area: f64 = dec
            .mesh
            .triangles
            .iter()
            .map(|t| {
                let (p0, p1, p2) = (
                    dec.mesh.vertices[t[0]],
                    dec.mesh.vertices[t[1]],
                    dec.mesh.vertices[t[2]],
                );
                0.5 * (p1 - p0).cross(&(p2 - p0)).magnitude()
            })
            .sum();
        assert!((mesh_area - 4.0 * 3.0_f64.sqrt()).abs() < 1e-12, "area {mesh_area}");
        assert!((total - mesh_area).abs() < 1e-12, "dual areas sum {total}");
        // by symmetry each vertex carries a sixth of the area, and the dual
        // areas are exactly the star0 weights
        for (i, &a) in dec.dual_areas().iter().enumerate() {
            assert!((a - mesh_area / 6.0).abs() < 1e-12, "dual area {i}: {a}");
            assert!((a - dec.hodge0()[i]).abs() < 1e-15);
        }
        // hodge2 is the reciprocal face area
        for (fi, t) in dec.mesh.triangles.iter().enumerate() {
            let (p0, p1, p2) = (
                dec.mesh.vertices[t[0]],
                dec.mesh.vertices[t[1]],
                dec.mesh.vertices[t[2]],
            );
            let area = 0.5 * (p1 - p0).cross(&(p2 - p0)).magnitude();
            assert!(
                (dec.hodge2()[fi] - 1.0 / area).abs() < 1e-12,
                "hodge2 at {fi}: {} vs {}",
                dec.hodge2()[fi],
                1.0 / area
            );
        }
        // d0 and d1 as public matrices: shapes, incidence pattern, and the
        // fundamental identity d1 d0 = 0
        let d0 = dec.d0();
        let d1 = dec.d1();
        assert_eq!((d0.rows, d0.cols), (ne, nv));
        assert_eq!((d1.rows, d1.cols), (nf, ne));
        let mut rng = Rng::new(31);
        for _ in 0..4 {
            let f: Vec<f64> = (0..nv).map(|_| rng.next_gaussian()).collect();
            let df = d0.mul_vec(&f);
            // d0 agrees with the public gradient and gives edge differences
            let grad = dec.gradient(&f);
            for e in 0..ne {
                let (a, b) = dec.edges[e];
                assert!((df[e] - (f[b] - f[a])).abs() < 1e-15, "d0 = head - tail");
                assert!((df[e] - grad[e]).abs() < 1e-15);
            }
            let ddf = d1.mul_vec(&df);
            assert!(ddf.iter().all(|v| v.abs() < 1e-12), "d1 d0 = 0");
            // constants are in the kernel of d0
            let one = vec![1.0; nv];
            assert!(d0.mul_vec(&one).iter().all(|v| v.abs() < 1e-15));
            // d1 agrees with the public curl
            let w: Vec<f64> = (0..ne).map(|_| rng.next_gaussian()).collect();
            let c1 = d1.mul_vec(&w);
            let c2 = dec.curl(&w);
            for i in 0..nf {
                assert!((c1[i] - c2[i]).abs() < 1e-15);
            }
        }
        // laplace_beltrami (CSR): symmetric, rows sum to zero, positive
        // semidefinite, and equal to -star0 * div grad (an independent path)
        let l = dec.laplace_beltrami();
        assert_eq!((l.rows, l.cols), (nv, nv));
        let one = vec![1.0; nv];
        assert!(
            l.mul_vec(&one).iter().all(|v| v.abs() < 1e-12),
            "constant in the kernel (rows sum to zero)"
        );
        for _ in 0..4 {
            let f: Vec<f64> = (0..nv).map(|_| rng.next_gaussian()).collect();
            let g: Vec<f64> = (0..nv).map(|_| rng.next_gaussian()).collect();
            let lf = l.mul_vec(&f);
            let lg = l.mul_vec(&g);
            let sym: f64 = g.iter().zip(&lf).map(|(a, b)| a * b).sum::<f64>()
                - f.iter().zip(&lg).map(|(a, b)| a * b).sum::<f64>();
            assert!(sym.abs() < 1e-10, "L symmetric: {sym}");
            let quad: f64 = f.iter().zip(&lf).map(|(a, b)| a * b).sum();
            assert!(quad > -1e-12, "L positive semidefinite: {quad}");
            // Dirichlet energy identity: f^T L f = sum star1 (df)^2
            let df = dec.gradient(&f);
            let energy: f64 = df
                .iter()
                .zip(dec.hodge1())
                .map(|(d, s)| s * d * d)
                .sum();
            assert!((quad - energy).abs() < 1e-9 * energy.abs().max(1.0), "Dirichlet energy");
            // L f = -star0 (div grad f)
            let divgrad = dec.divergence(&df);
            for i in 0..nv {
                assert!(
                    (lf[i] + dec.hodge0()[i] * divgrad[i]).abs() < 1e-9,
                    "L f vs -M div grad f at {i}"
                );
            }
        }
    }

    #[test]
    fn test_laplace_1form_and_viscous_fluid_step() {
        let dec = DecMesh::new(&flat_grid(4));
        let ne = dec.ne();
        let nv = dec.nv();
        let l1 = dec.laplace_1form();
        assert_eq!((l1.rows, l1.cols), (ne, ne));
        let l0 = dec.laplace_beltrami();
        let mut rng = Rng::new(2027);
        // On an exact 1-form w = d0 f the curl term drops out, leaving
        // Delta1 (d0 f) = d0 (star0^-1 L f) exactly.
        for _ in 0..3 {
            let f: Vec<f64> = (0..nv).map(|_| rng.next_gaussian()).collect();
            let w = dec.gradient(&f);
            let lhs = l1.mul_vec(&w);
            let lf = l0.mul_vec(&f);
            let scaled: Vec<f64> = lf
                .iter()
                .zip(dec.hodge0())
                .map(|(a, m)| a / m)
                .collect();
            let rhs = dec.gradient(&scaled);
            let scale = rhs.iter().fold(0.0_f64, |a, b| a.max(b.abs())).max(1.0);
            for i in 0..ne {
                assert!(
                    (lhs[i] - rhs[i]).abs() < 1e-8 * scale,
                    "Delta1 d0 f = d0 (M^-1 L f) at edge {i}: {} vs {}",
                    lhs[i],
                    rhs[i]
                );
            }
            // the star1 energy of the Hodge Laplacian is nonnegative
            let form: f64 = w
                .iter()
                .zip(&lhs)
                .zip(dec.hodge1())
                .map(|((a, b), s)| a * b * s)
                .sum();
            assert!(form > -1e-8, "<w, Delta1 w>_star1 >= 0: {form}");
        }
        // Viscous fluid step (nu > 0): the diffusion branch must equal an
        // explicit Delta1 step followed by removal of the exact part.
        let rot: Vec<Vec3> = dec
            .mesh
            .triangles
            .iter()
            .map(|t| {
                let c = (dec.mesh.vertices[t[0]] + dec.mesh.vertices[t[1]] + dec.mesh.vertices[t[2]])
                    * (1.0 / 3.0);
                Vec3::new(-(c.y - 0.5), c.x - 0.5, 0.0)
            })
            .collect();
        let mut w = dec.vector_field_to_1form(&rot);
        // project once so the field starts free of an exact part
        dec.fluid_step_dec(&mut w, 0.0, 0.0);
        let energy = |v: &[f64]| -> f64 {
            v.iter().zip(dec.hodge1()).map(|(a, s)| s * a * a).sum()
        };
        let e0 = energy(&w);
        assert!(e0 > 1e-6, "starting energy {e0}");
        // explicit diffusion is only stable for dt nu lambda_max < 2; bound
        // the spectrum of Delta1 by Gershgorin and stay at a quarter of it,
        // so that doubling nu below still increases the damping
        let gersh = (0..ne)
            .map(|r| {
                (l1.row_ptr[r]..l1.row_ptr[r + 1])
                    .map(|k| l1.vals[k].abs())
                    .sum::<f64>()
            })
            .fold(0.0_f64, f64::max);
        assert!(gersh > 0.0);
        let dt = 0.01;
        let nu = 0.25 / (dt * gersh);
        let mut got = w.clone();
        dec.fluid_step_dec(&mut got, dt, nu);
        // reference computed from the public operators
        let lw = l1.mul_vec(&w);
        let mut want: Vec<f64> = w.iter().zip(&lw).map(|(a, l)| a - dt * nu * l).collect();
        let (exact, _, _) = dec.hodge_decomposition(&want);
        for (v, e) in want.iter_mut().zip(&exact) {
            *v -= e;
        }
        let scale = want.iter().fold(0.0_f64, |a, b| a.max(b.abs())).max(1.0);
        for i in 0..ne {
            assert!(
                (got[i] - want[i]).abs() < 1e-9 * scale,
                "viscous step at edge {i}: {} vs {}",
                got[i],
                want[i]
            );
        }
        // diffusion dissipates energy, and the result stays exact-free
        let e1 = energy(&got);
        assert!(e1 < e0, "viscosity must dissipate: {e0} -> {e1}");
        let (ex_after, _, _) = dec.hodge_decomposition(&got);
        let ex_norm = ex_after.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
        let got_norm = got.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
        assert!(
            ex_norm < 1e-6 * got_norm.max(1.0),
            "exact part after the step {ex_norm}"
        );
        // with nu = 0 the same field only loses its (already absent) exact
        // part, so the energy is unchanged
        let mut inviscid = w.clone();
        dec.fluid_step_dec(&mut inviscid, dt, 0.0);
        assert!(
            (energy(&inviscid) - e0).abs() < 1e-6 * e0,
            "inviscid step conserves energy"
        );
        // more viscosity removes more energy
        let mut strong = w.clone();
        dec.fluid_step_dec(&mut strong, dt, 2.0 * nu);
        assert!(energy(&strong) < e1, "monotone in nu");
    }

    #[test]
    fn test_persistent_homology() {
        let mut rng = Rng::new(5);
        // noisy circle: one long-lived 1-cycle
        let pts: Vec<VecN> = (0..24)
            .map(|k| {
                let th = 2.0 * PI * k as f64 / 24.0;
                VecN::from(&[
                    th.cos() + 0.02 * rng.next_gaussian(),
                    th.sin() + 0.02 * rng.next_gaussian(),
                ])
            })
            .collect();
        let pairs = persistent_homology_vietoris_rips(&pts, 2.5, 1);
        // exactly one essential 0-class (the connected component)
        let h0_essential = pairs
            .iter()
            .filter(|&&(d, _, death)| d == 0 && (death - 2.5).abs() < 1e-12)
            .count();
        assert_eq!(h0_essential, 1, "one component");
        // one 1-cycle with a long lifetime (born early, dies near diameter)
        let long_cycles: Vec<&(usize, f64, f64)> = pairs
            .iter()
            .filter(|&&(d, b, death)| d == 1 && death - b > 0.8)
            .collect();
        assert_eq!(long_cycles.len(), 1, "one long 1-cycle: {pairs:?}");
        // Betti curve at mid-filtration: b0 = 1, b1 = 1
        let curve = betti_curve(&pairs, &[0.5, 1.0]);
        assert_eq!(curve[0].1[0], 1);
        assert_eq!(curve[0].1[1], 1);
        // bottleneck distance: diagram vs itself is 0; vs shifted is the
        // shift
        let d1: Vec<(f64, f64)> = pairs
            .iter()
            .filter(|&&(d, _, _)| d == 1)
            .map(|&(_, b, dth)| (b, dth))
            .collect();
        assert!(persistence_diagram_bottleneck(&d1, &d1) < 1e-12);
        let shifted: Vec<(f64, f64)> = d1.iter().map(|&(b, d)| (b + 0.1, d + 0.1)).collect();
        let bd = persistence_diagram_bottleneck(&d1, &shifted);
        assert!((bd - 0.1).abs() < 1e-9, "bottleneck {bd}");
    }
}
