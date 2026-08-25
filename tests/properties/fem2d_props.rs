//! Properties of the two-dimensional triangular finite element module.
//!
//! The tests split into two groups.
//!
//! *The mesh has to be a mesh.* Euler's formula `V - E + T = 1` holds for
//! any triangulated simply connected region, every edge belongs to one
//! triangle or two, and every triangle is oriented the same way. Uniform
//! refinement then has to preserve all of that while multiplying the
//! triangle count by four and leaving the worst angle *exactly* unchanged
//! -- the four children of a triangle are similar to their parent, so a
//! quality measure that drifts under refinement is measuring something
//! other than shape.
//!
//! *The solver has to be a Galerkin method.* The same theorems as in one
//! dimension apply verbatim, because none of their proofs mentions the
//! dimension: the error is orthogonal to the space, so the Pythagoras
//! identity holds exactly and Cea's lemma follows; refining lowers the
//! energy; a field already in the space is returned untouched. What is
//! new in two dimensions is geometry. The Laplacian does not care about
//! rotation or about which way the mesh happens to be cut, and the
//! stiffness matrix's off-diagonal entry is minus half the cotangent of
//! the opposite angle -- the identity that makes the Delaunay condition
//! and the discrete maximum principle the same statement.

use rust_physics_engine::error::SolveError;
use rust_physics_engine::fem::fem2d::{
    dirichlet_energy, element_gradient, element_stress, fem_2d_elasticity_plane_stress,
    fem_2d_heat_transient, fem_2d_helmholtz, fem_2d_poisson, fem_2d_reaction_diffusion,
    fem_eigenmodes_drum, fem_eigenvalues_drum, interpolate, mass_matrix, strain_energy,
    stiffness_matrix, von_mises_stress, FemMesh2,
};
use rust_physics_engine::math::Vec2;
use rust_physics_engine::monte_carlo::Rng;

/// Every distinct edge, with how many triangles use it.
fn edge_counts(mesh: &FemMesh2) -> std::collections::HashMap<(usize, usize), usize> {
    let mut counts = std::collections::HashMap::new();
    for t in &mesh.tris {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            *counts.entry(if a < b { (a, b) } else { (b, a) }).or_insert(0usize) += 1;
        }
    }
    counts
}

fn signed_area(mesh: &FemMesh2, t: &[usize; 3]) -> f64 {
    let (a, b, c) = (mesh.nodes[t[0]], mesh.nodes[t[1]], mesh.nodes[t[2]]);
    0.5 * ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y))
}

/// A dense lookup into a CSR matrix, for reading single entries in tests.
fn csr_get(m: &rust_physics_engine::linalg::sparse::CsrMatrix, i: usize, j: usize) -> f64 {
    (m.row_ptr[i]..m.row_ptr[i + 1])
        .filter(|&k| m.col_idx[k] == j)
        .map(|k| m.vals[k])
        .sum()
}

/// A spread of meshes covering the three generators.
fn meshes(rng: &mut Rng) -> Vec<FemMesh2> {
    let nx = 2 + (rng.next_u64() % 4) as usize;
    let ny = 2 + (rng.next_u64() % 4) as usize;
    let mut out = vec![
        FemMesh2::rect(0.5 + rng.next_f64(), 0.5 + rng.next_f64(), nx, ny).unwrap(),
        FemMesh2::disk(0.5 + rng.next_f64(), 1 + (rng.next_u64() % 4) as usize).unwrap(),
    ];
    // A jittered grid, triangulated by Delaunay. The jitter keeps the
    // points from being cocircular, which is the degenerate case.
    let mut points = Vec::new();
    for i in 0..5 {
        for j in 0..5 {
            points.push(Vec2::new(
                i as f64 + 0.2 * (rng.next_f64() - 0.5),
                j as f64 + 0.2 * (rng.next_f64() - 0.5),
            ));
        }
    }
    if let Ok(m) = FemMesh2::from_delaunay(&points) {
        out.push(m);
    }
    out
}

#[test]
fn prop_every_mesh_is_a_conforming_oriented_triangulation() {
    let mut rng = Rng::new(0x3c11_9d40);
    for _ in 0..25 {
        for m in meshes(&mut rng) {
            // Euler's formula for a simply connected region.
            let v = m.nodes.len() as i64;
            let e = m.edge_count() as i64;
            let t = m.tris.len() as i64;
            assert_eq!(v - e + t, 1, "V {v} E {e} T {t}");
            let counts = edge_counts(&m);
            assert_eq!(counts.len(), m.edge_count());
            assert!(counts.values().all(|&c| c == 1 || c == 2), "an edge had a third triangle");
            // Every triangle counterclockwise, and no degenerate ones.
            for tri in &m.tris {
                assert!(signed_area(&m, tri) > 0.0);
            }
            // The boundary nodes are exactly the endpoints of the
            // once-used edges, and each lies on exactly two of them: the
            // boundary is a union of simple closed curves.
            let mut degree = vec![0usize; m.nodes.len()];
            for (&(a, b), &c) in &counts {
                if c == 1 {
                    degree[a] += 1;
                    degree[b] += 1;
                }
            }
            let derived: Vec<usize> =
                (0..m.nodes.len()).filter(|&i| degree[i] > 0).collect();
            assert_eq!(derived, m.boundary);
            for &b in &m.boundary {
                assert_eq!(degree[b], 2, "boundary node {b} was a pinch point");
            }
            // The worst angle is a real angle.
            let q = m.quality_min_angle();
            assert!(q > 0.0 && q <= std::f64::consts::PI / 3.0 + 1e-12, "min angle {q}");
        }
    }
}

#[test]
fn prop_uniform_refinement_preserves_shape_and_area_exactly() {
    // The four children of a triangle are all similar to it, so the
    // worst angle in the mesh does not move at all. Preserving it only
    // approximately would mean the split is not the midpoint one.
    let mut rng = Rng::new(0x7a02_c5e1);
    for _ in 0..20 {
        for m in meshes(&mut rng) {
            let (v, e, t, a, q) =
                (m.nodes.len(), m.edge_count(), m.tris.len(), m.area(), m.quality_min_angle());
            let r = m.refine_uniform();
            assert_eq!(r.nodes.len(), v + e, "one new node per edge");
            assert_eq!(r.tris.len(), 4 * t);
            assert!((r.area() - a).abs() < 1e-12 * a, "area moved by {}", r.area() - a);
            assert!((r.quality_min_angle() - q).abs() < 1e-13, "the shape drifted");
            assert_eq!(
                r.nodes.len() as i64 - r.edge_count() as i64 + r.tris.len() as i64,
                1
            );
            // Each boundary edge gains its midpoint, and nothing else
            // joins the boundary.
            let boundary_edges = edge_counts(&m).values().filter(|&&c| c == 1).count();
            assert_eq!(r.boundary.len(), m.boundary.len() + boundary_edges);
        }
    }
}

#[test]
fn prop_the_stiffness_entry_is_the_cotangent_of_the_opposite_angle() {
    // K_ij for an edge is minus half the sum of the cotangents of the
    // angles facing it. That identity is the whole reason the Delaunay
    // condition and the M-matrix property coincide, and it is checked
    // here one triangle at a time so that no cancellation hides a sign.
    let mut rng = Rng::new(0x1b6e_2f93);
    for _ in 0..60 {
        let p: Vec<Vec2> = (0..3)
            .map(|_| Vec2::new(4.0 * rng.next_f64() - 2.0, 4.0 * rng.next_f64() - 2.0))
            .collect();
        let Ok(m) = FemMesh2::new(p.clone(), vec![[0, 1, 2]]) else { continue };
        if m.quality_min_angle() < 1e-3 {
            continue;
        }
        let k = stiffness_matrix(&m);
        for (i, j, opposite) in [(0usize, 1usize, 2usize), (1, 2, 0), (0, 2, 1)] {
            let (a, b) = (m.nodes[i] - m.nodes[opposite], m.nodes[j] - m.nodes[opposite]);
            let cross = a.x * b.y - a.y * b.x;
            let cot = a.dot(&b) / cross.abs();
            let got = csr_get(&k, i, j);
            assert!(
                (got + 0.5 * cot).abs() < 1e-9 * (1.0 + cot.abs()),
                "K[{i}][{j}] was {got}, cotangent rule says {}",
                -0.5 * cot
            );
            // Obtuse opposite angle means a positive off-diagonal, which
            // is exactly the failure of the M-matrix property.
            assert_eq!(got > 0.0, a.dot(&b) < 0.0);
        }
    }
}

#[test]
fn prop_the_stiffness_matrix_annihilates_constants_and_the_mass_matrix_totals_the_area() {
    // The three shape functions of a triangle sum to one, so their
    // gradients sum to zero and every stiffness row sums to zero. The
    // same partition of unity makes the mass matrix entries total the
    // area of the mesh.
    let mut rng = Rng::new(0x2d55_8b17);
    for _ in 0..20 {
        for m in meshes(&mut rng) {
            let k = stiffness_matrix(&m);
            let ones = vec![1.0; m.nodes.len()];
            let scale = k.vals.iter().fold(0.0f64, |a, v| a.max(v.abs())).max(1.0);
            for (i, r) in k.mul_vec(&ones).iter().enumerate() {
                assert!(r.abs() < 1e-11 * scale, "stiffness row {i} summed to {r}");
            }
            // Symmetry, entry by entry.
            for i in 0..m.nodes.len() {
                for idx in k.row_ptr[i]..k.row_ptr[i + 1] {
                    let j = k.col_idx[idx];
                    assert!((k.vals[idx] - csr_get(&k, j, i)).abs() < 1e-11 * scale);
                }
            }
            let mm = mass_matrix(&m);
            let total: f64 = mm.mul_vec(&ones).iter().sum();
            assert!((total - m.area()).abs() < 1e-11 * m.area(), "mass total {total}");
            // And the mass matrix is positive on the diagonal, since it
            // is the Gram matrix of linearly independent functions.
            for i in 0..m.nodes.len() {
                assert!(csr_get(&mm, i, i) > 0.0);
            }
        }
    }
}

#[test]
fn prop_a_linear_field_is_reproduced_and_interpolated_exactly() {
    // The patch test, plus the statement that the evaluator is the same
    // interpolation the space is built from.
    let mut rng = Rng::new(0x64ff_1c28);
    for _ in 0..20 {
        let (c0, cx, cy) = (
            2.0 * rng.next_f64() - 1.0,
            2.0 * rng.next_f64() - 1.0,
            2.0 * rng.next_f64() - 1.0,
        );
        let exact = move |p: Vec2| c0 + cx * p.x + cy * p.y;
        for m in meshes(&mut rng) {
            let u = fem_2d_poisson(&m, &|_| 0.0, &|p| Some(exact(p))).unwrap();
            for (i, &got) in u.iter().enumerate() {
                let want = exact(m.nodes[i]);
                assert!((got - want).abs() < 1e-9 * (1.0 + want.abs()), "node {i}");
            }
            let node_err = u
                .iter()
                .enumerate()
                .map(|(i, &g)| (g - exact(m.nodes[i])).abs())
                .fold(0.0, f64::max);
            // The gradient is the same constant on every triangle. It is
            // exact only to the accuracy of the nodal values, and a
            // gradient amplifies a nodal error by the sum of the shape
            // function gradient magnitudes -- which is what makes a
            // sliver element bad, and is a sharper thing to assert than
            // a fixed tolerance. Each shape function gradient is read
            // off by differentiating its own indicator vector.
            for t in 0..m.tris.len() {
                let amp: f64 = (0..3)
                    .map(|k| {
                        let mut e = vec![0.0; m.nodes.len()];
                        e[m.tris[t][k]] = 1.0;
                        element_gradient(&m, &e, t).unwrap().magnitude()
                    })
                    .sum();
                let g = element_gradient(&m, &u, t).unwrap();
                let err = (g - Vec2::new(cx, cy)).magnitude();
                assert!(err <= node_err * amp + 1e-12, "triangle {t}: {err} > {node_err} * {amp}");
            }
            // Sampling inside is a convex combination of nodal values,
            // so it cannot be further off than the worst node is.
            for t in 0..m.tris.len().min(6) {
                let tri = m.tris[t];
                let mid = Vec2::new(
                    (m.nodes[tri[0]].x + m.nodes[tri[1]].x + m.nodes[tri[2]].x) / 3.0,
                    (m.nodes[tri[0]].y + m.nodes[tri[1]].y + m.nodes[tri[2]].y) / 3.0,
                );
                let got = interpolate(&m, &u, mid).unwrap();
                assert!((got - exact(mid)).abs() <= node_err + 1e-12);
            }
        }
    }
}

#[test]
fn prop_the_solution_minimises_the_energy_and_the_excess_is_exact() {
    // The Ritz characterisation, which is equivalent to Galerkin
    // orthogonality: no other member of the space with the same boundary
    // values has a lower energy, and because the functional is quadratic
    // the amount by which a candidate loses is *exactly* half the energy
    // norm of its difference from the solution. The equality is the
    // stronger half -- an inequality can hold by accident, and the cross
    // term it hides is the orthogonality itself.
    //
    // A quadratic exact solution makes every integral in the assembly
    // exact, since its Laplacian is constant and the one-point rule
    // integrates a constant exactly, so this holds to rounding.
    let mut rng = Rng::new(0x4881_0ae5);
    let mut moved = 0;
    for _ in 0..30 {
        let (a, b, c) = (
            2.0 * rng.next_f64() - 1.0,
            2.0 * rng.next_f64() - 1.0,
            2.0 * rng.next_f64() - 1.0,
        );
        // u = a x^2 + b xy + c y^2, so -lap u = -2(a + c).
        let exact = move |p: Vec2| a * p.x * p.x + b * p.x * p.y + c * p.y * p.y;
        let load_density = -2.0 * (a + c);
        let m = FemMesh2::rect(1.0, 1.0, 5, 4).unwrap();
        let u_h = fem_2d_poisson(&m, &|_| load_density, &|p| Some(exact(p))).unwrap();
        // J(v) = (1/2) integral |grad v|^2 - integral f v, with the load
        // integrated the way the assembly does it.
        let j = |x: &[f64]| {
            let load: f64 = m
                .tris
                .iter()
                .map(|t| signed_area(&m, t) / 3.0 * (x[t[0]] + x[t[1]] + x[t[2]]))
                .sum();
            0.5 * dirichlet_energy(&m, x).unwrap() - load_density * load
        };
        let on_boundary: std::collections::HashSet<usize> = m.boundary.iter().copied().collect();
        for _ in 0..4 {
            let mut v = u_h.clone();
            for (i, slot) in v.iter_mut().enumerate() {
                if !on_boundary.contains(&i) {
                    *slot += 0.7 * (2.0 * rng.next_f64() - 1.0);
                }
            }
            let difference: Vec<f64> =
                v.iter().zip(u_h.iter()).map(|(p, q)| p - q).collect();
            let side = dirichlet_energy(&m, &difference).unwrap();
            let excess = j(&v) - j(&u_h);
            assert!(excess >= -1e-10, "a candidate had lower energy by {}", -excess);
            assert!(
                (excess - 0.5 * side).abs() < 1e-9 * (1.0 + excess),
                "excess {excess} was not half the energy {side}"
            );
            if side > 1e-6 {
                moved += 1;
            }
        }
    }
    assert!(moved > 100, "the candidates never left the solution");
}

#[test]
fn prop_refining_the_mesh_lowers_the_energy() {
    // The coarse space sits inside the refined one, so the minimum of
    // the energy functional over it cannot be smaller.
    let mut rng = Rng::new(0x0e93_77b2);
    for _ in 0..12 {
        let k = 1.0 + 2.0 * rng.next_f64();
        let f = move |p: Vec2| (k * p.x).cos() * (k * p.y).cos();
        let coarse = FemMesh2::rect(1.0, 1.0, 3, 3).unwrap();
        let fine = coarse.refine_uniform();
        let j = |m: &FemMesh2| {
            let u = fem_2d_poisson(m, &f, &|_| Some(0.0)).unwrap();
            let load: f64 = m
                .tris
                .iter()
                .map(|t| {
                    let mid = Vec2::new(
                        (m.nodes[t[0]].x + m.nodes[t[1]].x + m.nodes[t[2]].x) / 3.0,
                        (m.nodes[t[0]].y + m.nodes[t[1]].y + m.nodes[t[2]].y) / 3.0,
                    );
                    signed_area(m, t) / 3.0 * f(mid) * (u[t[0]] + u[t[1]] + u[t[2]])
                })
                .sum();
            0.5 * dirichlet_energy(m, &u).unwrap() - load
        };
        assert!(j(&fine) <= j(&coarse) + 1e-10, "refining raised the energy");
    }
}

#[test]
fn prop_the_solution_is_linear_in_its_data() {
    let mut rng = Rng::new(0x51c7_930f);
    for _ in 0..20 {
        let m = FemMesh2::rect(1.0, 1.5, 4, 3).unwrap();
        let (a1, a2) = (2.0 * rng.next_f64() - 1.0, 2.0 * rng.next_f64() - 1.0);
        let (b1, b2) = (2.0 * rng.next_f64() - 1.0, 2.0 * rng.next_f64() - 1.0);
        let f1 = move |p: Vec2| a1 * p.x + b1;
        let f2 = move |p: Vec2| a2 * p.y * p.y + b2;
        let g1 = move |p: Vec2| Some(a1 * p.x * p.y);
        let g2 = move |p: Vec2| Some(b2 - p.x);
        let u1 = fem_2d_poisson(&m, &f1, &g1).unwrap();
        let u2 = fem_2d_poisson(&m, &f2, &g2).unwrap();
        let both = fem_2d_poisson(&m, &|p| f1(p) + f2(p), &|p| {
            Some(g1(p).unwrap() + g2(p).unwrap())
        })
        .unwrap();
        for i in 0..m.nodes.len() {
            let want = u1[i] + u2[i];
            assert!((both[i] - want).abs() < 1e-8 * (1.0 + want.abs()), "node {i}");
        }
    }
}

#[test]
fn prop_a_nonnegative_load_stays_nonnegative_on_a_delaunay_mesh() {
    // The right-triangle rectangle mesh has no obtuse angle, so every
    // off-diagonal stiffness entry is nonpositive and the matrix is an
    // M-matrix. Its inverse is then entrywise nonnegative, which is the
    // discrete maximum principle.
    let mut rng = Rng::new(0x38b4_6d51);
    for _ in 0..25 {
        let m = FemMesh2::rect(1.0, 1.0, 6, 6).unwrap();
        let k = stiffness_matrix(&m);
        for i in 0..m.nodes.len() {
            for idx in k.row_ptr[i]..k.row_ptr[i + 1] {
                if k.col_idx[idx] != i {
                    assert!(k.vals[idx] <= 1e-12, "an off-diagonal entry was positive");
                }
            }
        }
        let (a, b) = (rng.next_f64(), rng.next_f64());
        let f = move |p: Vec2| (a * p.x + b * p.y).powi(2);
        let u = fem_2d_poisson(&m, &f, &|_| Some(0.0)).unwrap();
        assert!(u.iter().all(|&v| v >= -1e-10), "the solution went negative");
        // With no load the extremes are on the boundary.
        let g = move |p: Vec2| Some(a * p.x + b * p.y * p.y);
        let h = fem_2d_poisson(&m, &|_| 0.0, &g).unwrap();
        let on_boundary: std::collections::HashSet<usize> = m.boundary.iter().copied().collect();
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for &b in &m.boundary {
            lo = lo.min(h[b]);
            hi = hi.max(h[b]);
        }
        for (i, &v) in h.iter().enumerate() {
            if !on_boundary.contains(&i) {
                assert!(v >= lo - 1e-9 && v <= hi + 1e-9, "interior node {i} overshot at {v}");
            }
        }
    }
}

#[test]
fn prop_the_laplacian_does_not_care_how_the_plane_is_oriented() {
    // Rotating the mesh and the data rotates the solution and nothing
    // else. The shape function gradients are the only place a coordinate
    // direction enters, so this is a direct test of that formula.
    let mut rng = Rng::new(0x22e0_44b6);
    for _ in 0..20 {
        let theta = std::f64::consts::TAU * rng.next_f64();
        let (c, s) = (theta.cos(), theta.sin());
        let rot = move |p: Vec2| Vec2::new(c * p.x - s * p.y, s * p.x + c * p.y);
        let base = FemMesh2::rect(1.3, 0.8, 4, 3).unwrap();
        let turned =
            FemMesh2::new(base.nodes.iter().map(|&p| rot(p)).collect(), base.tris.clone())
                .unwrap();
        let k = 1.0 + rng.next_f64();
        // The source and the boundary data are carried along, so the
        // same physical problem is being solved in a turned frame.
        let plain = fem_2d_poisson(&base, &|p| (k * p.x).sin(), &|p| Some(p.y)).unwrap();
        let spun = fem_2d_poisson(
            &turned,
            &|p| {
                // Undo the rotation to sample the same physical point.
                let q = Vec2::new(c * p.x + s * p.y, -s * p.x + c * p.y);
                (k * q.x).sin()
            },
            &|p| Some(-s * p.x + c * p.y),
        )
        .unwrap();
        for i in 0..base.nodes.len() {
            assert!(
                (plain[i] - spun[i]).abs() < 1e-8 * (1.0 + plain[i].abs()),
                "node {i}: {} vs {}",
                plain[i],
                spun[i]
            );
        }
    }
}

#[test]
fn prop_scaling_the_domain_scales_the_laplacian_by_the_square() {
    // u(x) on the unit square solves -lap u = f; then u(x/s) on the
    // s-square solves -lap v = f(x/s)/s^2. Getting the power wrong is a
    // mistake the patch test cannot see, because a linear field's
    // Laplacian is zero either way.
    let mut rng = Rng::new(0x6f19_c0d3);
    for _ in 0..20 {
        let s = 0.4 + 2.0 * rng.next_f64();
        let k = 1.0 + 2.0 * rng.next_f64();
        let f = move |p: Vec2| (k * p.x).sin() * (k * p.y + 0.3).cos();
        let unit = FemMesh2::rect(1.0, 1.0, 5, 5).unwrap();
        let big = FemMesh2::rect(s, s, 5, 5).unwrap();
        let u = fem_2d_poisson(&unit, &f, &|_| Some(0.0)).unwrap();
        let v = fem_2d_poisson(&big, &|p| f(Vec2::new(p.x / s, p.y / s)) / (s * s), &|_| {
            Some(0.0)
        })
        .unwrap();
        for i in 0..unit.nodes.len() {
            assert!((u[i] - v[i]).abs() < 1e-8 * (1.0 + u[i].abs()), "node {i}");
        }
    }
}

#[test]
fn prop_a_pure_flux_problem_is_singular_unless_something_pins_it() {
    // With no Dirichlet node the constant is in the kernel, which is
    // exactly the statement that the stiffness rows sum to zero. A
    // reaction term removes it.
    let mut rng = Rng::new(0x5aa7_31e8);
    for _ in 0..20 {
        for m in meshes(&mut rng) {
            let f = |_: Vec2| 1.0;
            assert_eq!(fem_2d_poisson(&m, &f, &|_| None), Err(SolveError::Singular));
            let c = 0.5 + rng.next_f64();
            let v = fem_2d_reaction_diffusion(&m, &|_| c, &|_| c, &|_| None).unwrap();
            // -lap u + c u = c with no flux has the constant solution 1,
            // and the constant is in the element space, so it is found
            // exactly rather than approximately.
            for &g in &v {
                assert!((g - 1.0).abs() < 1e-8, "got {g}");
            }
        }
    }
}

#[test]
fn prop_convergence_is_second_order_in_the_mesh_size() {
    // The rate is what identifies the space. Measured on nested
    // refinements so that the comparison is between the same solutions
    // on strictly nested meshes.
    let mut rng = Rng::new(0x13da_9f27);
    let pi = std::f64::consts::PI;
    for _ in 0..6 {
        let (a, b) = (1 + (rng.next_u64() % 2) as i32, 1 + (rng.next_u64() % 2) as i32);
        let u = move |p: Vec2| (a as f64 * pi * p.x).sin() * (b as f64 * pi * p.y).sin();
        let lam = pi * pi * ((a * a) as f64 + (b * b) as f64);
        let mut errors = Vec::new();
        for n in [4usize, 8, 16] {
            let m = FemMesh2::rect(1.0, 1.0, n, n).unwrap();
            let v = fem_2d_poisson(&m, &|p| lam * u(p), &|_| Some(0.0)).unwrap();
            errors.push(
                v.iter()
                    .enumerate()
                    .map(|(i, &g)| (g - u(m.nodes[i])).abs())
                    .fold(0.0, f64::max),
            );
        }
        for w in errors.windows(2) {
            let ratio = w[0] / w[1];
            assert!((ratio - 4.0).abs() < 0.6, "halving h cut the error by {ratio}, not 4");
        }
    }
}

/// The assembled load vector for a source, using the same one-point
/// centroid rule the module's assembly does, so that identities the
/// discrete system satisfies exactly come out exact here.
fn load_vector(mesh: &FemMesh2, f: &dyn Fn(Vec2) -> f64) -> Vec<f64> {
    let mut load = vec![0.0; mesh.nodes.len()];
    for t in &mesh.tris {
        let mid = Vec2::new(
            (mesh.nodes[t[0]].x + mesh.nodes[t[1]].x + mesh.nodes[t[2]].x) / 3.0,
            (mesh.nodes[t[0]].y + mesh.nodes[t[1]].y + mesh.nodes[t[2]].y) / 3.0,
        );
        let share = signed_area(mesh, t) / 3.0 * f(mid);
        for k in 0..3 {
            load[t[k]] += share;
        }
    }
    load
}

#[test]
fn prop_the_operator_is_symmetric_so_sources_and_responses_reciprocate() {
    // Betti reciprocity: with the same homogeneous boundary condition,
    // the work one source does through the other's response equals the
    // work the other does through the first's. It is exactly the
    // symmetry of the stiffness matrix seen from outside the solver, and
    // it holds for the indefinite Helmholtz operator as well as for the
    // positive definite Laplacian -- symmetry has nothing to do with
    // definiteness.
    let mut rng = Rng::new(0x71b3_44c8);
    for _ in 0..25 {
        let m = FemMesh2::rect(1.0, 1.2, 5, 4).unwrap();
        let (a1, b1) = (2.0 * rng.next_f64() - 1.0, 2.0 * rng.next_f64() - 1.0);
        let (a2, b2) = (2.0 * rng.next_f64() - 1.0, 2.0 * rng.next_f64() - 1.0);
        let f1 = move |p: Vec2| a1 + b1 * p.x * p.y;
        let f2 = move |p: Vec2| a2 * p.y + b2 * p.x * p.x;
        let (l1, l2) = (load_vector(&m, &f1), load_vector(&m, &f2));
        for k in [0.0, 1.0, 3.0] {
            let u1 = fem_2d_helmholtz(&m, k, &f1, &|_| Some(0.0)).unwrap();
            let u2 = fem_2d_helmholtz(&m, k, &f2, &|_| Some(0.0)).unwrap();
            let one: f64 = l1.iter().zip(u2.iter()).map(|(a, b)| a * b).sum();
            let two: f64 = l2.iter().zip(u1.iter()).map(|(a, b)| a * b).sum();
            assert!(
                (one - two).abs() < 1e-9 * (1.0 + one.abs()),
                "k = {k}: {one} against {two}"
            );
        }
    }
}

#[test]
fn prop_helmholtz_is_linear_and_reduces_to_poisson_at_zero() {
    let mut rng = Rng::new(0x2c48_ff10);
    for _ in 0..20 {
        let m = FemMesh2::rect(1.0, 1.0, 5, 5).unwrap();
        let k = 3.0 * rng.next_f64();
        let (c1, c2) = (2.0 * rng.next_f64() - 1.0, 2.0 * rng.next_f64() - 1.0);
        let f1 = move |p: Vec2| c1 * (1.0 + p.x);
        let f2 = move |p: Vec2| c2 * p.y;
        let g1 = move |p: Vec2| Some(c1 * p.x);
        let g2 = move |p: Vec2| Some(c2 * p.y * p.y);
        let u1 = fem_2d_helmholtz(&m, k, &f1, &g1).unwrap();
        let u2 = fem_2d_helmholtz(&m, k, &f2, &g2).unwrap();
        let both = fem_2d_helmholtz(&m, k, &|p| f1(p) + f2(p), &|p| {
            Some(g1(p).unwrap() + g2(p).unwrap())
        })
        .unwrap();
        for i in 0..m.nodes.len() {
            let want = u1[i] + u2[i];
            assert!((both[i] - want).abs() < 1e-8 * (1.0 + want.abs()), "node {i}");
        }
        let zero = fem_2d_helmholtz(&m, 0.0, &f1, &g1).unwrap();
        let poisson = fem_2d_poisson(&m, &f1, &g1).unwrap();
        for i in 0..m.nodes.len() {
            assert!((zero[i] - poisson[i]).abs() < 1e-8 * (1.0 + poisson[i].abs()));
        }
    }
}

#[test]
fn prop_the_resonant_response_has_a_simple_pole_at_the_fundamental() {
    // Expanded in the drum modes the response is a sum of terms
    // c_j / (lambda_j - k^2). Approaching lambda_1 the first term takes
    // over, so the response times the gap converges to a constant --
    // a simple pole, not a pole of any other order and not an essential
    // singularity. Checking the residue converges is a much stronger
    // statement than checking the response grows.
    let mut rng = Rng::new(0x08f2_6a91);
    for _ in 0..10 {
        let m = FemMesh2::rect(1.0, 1.0, 6, 6).unwrap();
        let lambda = fem_eigenvalues_drum(&m, 1).unwrap()[0];
        let c = 0.5 + rng.next_f64();
        let f = move |_: Vec2| c;
        let probe = m
            .nodes
            .iter()
            .position(|p| (p.x - 0.5).abs() < 1e-12 && (p.y - 0.5).abs() < 1e-12)
            .unwrap();
        let residue = |gap: f64| {
            let k2 = lambda * (1.0 - gap);
            let u = fem_2d_helmholtz(&m, k2.sqrt(), &f, &|_| Some(0.0)).unwrap();
            u[probe] * (lambda - k2)
        };
        let (a, b, d) = (residue(0.05), residue(0.01), residue(0.002));
        // The residue settles down; the remaining drift is the other
        // modes' contribution, which falls off with the gap.
        assert!((b - d).abs() < 0.4 * (a - d).abs() + 1e-12, "the residue did not settle");
        assert!(d > 0.0, "the residue at the fundamental should be positive for a positive source");
    }
}

#[test]
fn prop_drum_eigenvalues_are_upper_bounds_that_fall_under_refinement() {
    // Every discrete eigenvalue is a Rayleigh quotient minimised over a
    // subspace of the true admissible space, so it is an upper bound on
    // the true one; and a nested refinement enlarges the subspace, so
    // the bound can only improve. Both halves are exact statements about
    // the method rather than asymptotic ones.
    let mut rng = Rng::new(0x4b70_2d3a);
    for _ in 0..10 {
        let nx = 3 + (rng.next_u64() % 3) as usize;
        let coarse = FemMesh2::rect(1.0, 1.0, nx, nx).unwrap();
        let fine = coarse.refine_uniform();
        let count = 3.min((nx - 1) * (nx - 1));
        let a = fem_eigenvalues_drum(&coarse, count).unwrap();
        let b = fem_eigenvalues_drum(&fine, count).unwrap();
        for i in 0..count {
            assert!(a[i] > 0.0, "eigenvalue {i} was not positive");
            assert!(b[i] <= a[i] + 1e-9, "refining raised eigenvalue {i}");
            // Both stay above the analytic fundamental, which is the
            // smallest thing any of them can be.
            assert!(b[i] >= 2.0 * std::f64::consts::PI.powi(2) - 1e-8);
            if i > 0 {
                assert!(a[i] >= a[i - 1] - 1e-9, "the eigenvalues came back unsorted");
            }
        }
    }
}

#[test]
fn prop_the_spectrum_scales_with_the_inverse_square_of_the_domain() {
    // Stretching a drum by s divides every eigenvalue by s^2. The mesh
    // topology is identical, so this is exact rather than a convergence
    // statement, and it is the check that catches a missing area factor
    // in either matrix.
    let mut rng = Rng::new(0x6d5c_11ae);
    for _ in 0..12 {
        let s = 0.4 + 2.0 * rng.next_f64();
        let unit = FemMesh2::rect(1.0, 1.0, 4, 4).unwrap();
        let big = FemMesh2::rect(s, s, 4, 4).unwrap();
        let a = fem_eigenvalues_drum(&unit, 4).unwrap();
        let b = fem_eigenvalues_drum(&big, 4).unwrap();
        for i in 0..4 {
            let want = a[i] / (s * s);
            assert!((b[i] - want).abs() < 1e-9 * want, "mode {i}: {} vs {want}", b[i]);
        }
        // And the spectrum does not care how the plane is turned.
        let theta = std::f64::consts::TAU * rng.next_f64();
        let (c, sn) = (theta.cos(), theta.sin());
        let turned = FemMesh2::new(
            unit.nodes.iter().map(|p| Vec2::new(c * p.x - sn * p.y, sn * p.x + c * p.y)).collect(),
            unit.tris.clone(),
        )
        .unwrap();
        let r = fem_eigenvalues_drum(&turned, 4).unwrap();
        for i in 0..4 {
            assert!((r[i] - a[i]).abs() < 1e-8 * a[i], "rotation moved mode {i}");
        }
    }
}

#[test]
fn prop_the_fundamental_mode_keeps_one_sign_and_the_next_does_not() {
    // Courant's nodal domain theorem: the first eigenfunction of the
    // Dirichlet Laplacian has no interior zero, and the k-th has at most
    // k nodal domains -- so the second must change sign. The discrete
    // modes inherit this on a mesh whose stiffness matrix is an
    // M-matrix, which the right-triangle rectangle mesh is.
    let mut rng = Rng::new(0x1fa9_3c60);
    for _ in 0..10 {
        let nx = 4 + (rng.next_u64() % 3) as usize;
        let m = FemMesh2::rect(1.0, 1.0, nx, nx).unwrap();
        let (values, modes) = fem_eigenmodes_drum(&m, 2).unwrap();
        let on_boundary: std::collections::HashSet<usize> = m.boundary.iter().copied().collect();
        let interior: Vec<usize> =
            (0..m.nodes.len()).filter(|i| !on_boundary.contains(i)).collect();
        let first: Vec<f64> = interior.iter().map(|&i| modes[0][i]).collect();
        assert!(
            first.iter().all(|&v| v > 1e-9) || first.iter().all(|&v| v < -1e-9),
            "the fundamental changed sign"
        );
        assert!(values[1] > values[0], "the second eigenvalue was not larger");
        let second: Vec<f64> = interior.iter().map(|&i| modes[1][i]).collect();
        assert!(
            second.iter().any(|&v| v > 1e-9) && second.iter().any(|&v| v < -1e-9),
            "the second mode kept one sign"
        );
    }
}

#[test]
fn prop_each_mode_is_mass_normalised_and_returns_its_own_rayleigh_quotient() {
    // phi^T M phi = 1 and phi^T K phi = lambda are the two halves of what
    // solving the generalised problem means, and distinct modes are
    // M-orthogonal. Getting the Cholesky transform backwards would leave
    // the eigenvalues plausible and these three identities broken.
    let mut rng = Rng::new(0x5e21_08d7);
    for _ in 0..10 {
        let m = if rng.next_f64() < 0.5 {
            FemMesh2::rect(1.0, 0.7, 5, 4).unwrap()
        } else {
            FemMesh2::disk(1.0, 4).unwrap()
        };
        let (values, modes) = fem_eigenmodes_drum(&m, 4).unwrap();
        let mass = mass_matrix(&m);
        let stiff = stiffness_matrix(&m);
        for (i, phi) in modes.iter().enumerate() {
            let mv = mass.mul_vec(phi);
            let norm: f64 = phi.iter().zip(mv.iter()).map(|(a, b)| a * b).sum();
            assert!((norm - 1.0).abs() < 1e-8, "mode {i} had mass norm {norm}");
            let kv = stiff.mul_vec(phi);
            let rayleigh: f64 = phi.iter().zip(kv.iter()).map(|(a, b)| a * b).sum();
            assert!((rayleigh - values[i]).abs() < 1e-6 * values[i], "mode {i}");
            for &b in &m.boundary {
                assert_eq!(phi[b], 0.0, "mode {i} was not clamped at node {b}");
            }
            for j in 0..i {
                if (values[i] - values[j]).abs() < 1e-6 * values[i] {
                    continue;
                }
                let dot: f64 = modes[j].iter().zip(mv.iter()).map(|(a, b)| a * b).sum();
                assert!(dot.abs() < 1e-7, "modes {j} and {i} overlapped by {dot}");
            }
        }
    }
}

#[test]
fn prop_no_rigid_motion_ever_strains_the_body() {
    // Two translations and an infinitesimal rotation span the kernel of
    // the stiffness matrix, so any combination of them is stress free
    // and energy free. A method that strained under a rotation would be
    // wrong at first order in the rotation angle and would look exactly
    // like a real load.
    let mut rng = Rng::new(0x62d9_1f04);
    for _ in 0..25 {
        for m in meshes(&mut rng) {
            let (tx, ty, w) = (
                4.0 * rng.next_f64() - 2.0,
                4.0 * rng.next_f64() - 2.0,
                2.0 * rng.next_f64() - 1.0,
            );
            let u: Vec<Vec2> = m
                .nodes
                .iter()
                .map(|p| Vec2::new(tx - w * p.y, ty + w * p.x))
                .collect();
            let e = 1.0 + 300.0 * rng.next_f64();
            let nu = 0.45 * rng.next_f64();
            let scale = e * (tx.abs() + ty.abs() + w.abs()).max(1.0);
            for (i, s) in von_mises_stress(&m, &u, e, nu).unwrap().iter().enumerate() {
                assert!(*s < 1e-10 * scale, "triangle {i} was stressed by {s}");
            }
            assert!(strain_energy(&m, &u, e, nu).unwrap().abs() < 1e-10 * scale);
        }
    }
}

#[test]
fn prop_a_uniform_strain_state_is_reproduced_and_uniform() {
    // The elasticity patch test. A linear displacement field lies in the
    // element space, so prescribing it on the boundary must reproduce it
    // inside, and every triangle must report the same stress no matter
    // its shape.
    let mut rng = Rng::new(0x1a4c_88b7);
    for _ in 0..25 {
        let (ex, ey, gamma) = (
            2e-3 * (rng.next_f64() - 0.5),
            2e-3 * (rng.next_f64() - 0.5),
            2e-3 * (rng.next_f64() - 0.5),
        );
        let field = move |p: Vec2| {
            Vec2::new(ex * p.x + 0.5 * gamma * p.y, 0.5 * gamma * p.x + ey * p.y)
        };
        let (e, nu) = (10.0 + 200.0 * rng.next_f64(), 0.45 * rng.next_f64());
        for m in meshes(&mut rng) {
            let pinned: Vec<(usize, Vec2)> =
                m.boundary.iter().map(|&b| (b, field(m.nodes[b]))).collect();
            let u = fem_2d_elasticity_plane_stress(&m, e, nu, &[], &pinned).unwrap();
            let scale = u.iter().fold(0.0f64, |a, d| a.max(d.x.abs()).max(d.y.abs()));
            for (i, got) in u.iter().enumerate() {
                let want = field(m.nodes[i]);
                let gap = (got.x - want.x).abs().max((got.y - want.y).abs());
                assert!(gap < 1e-10 * scale, "node {i} was off by {gap}");
            }
            // Every element reports the same stress: that is what makes
            // it a *uniform* strain state, and a shape-dependent answer
            // would mean the strain matrix is wrong.
            let first = element_stress(&m, &u, e, nu, 0).unwrap();
            let mag = first.iter().fold(0.0f64, |a, v| a.max(v.abs())).max(1e-12);
            for t in 1..m.tris.len() {
                let s = element_stress(&m, &u, e, nu, t).unwrap();
                for k in 0..3 {
                    assert!((s[k] - first[k]).abs() < 1e-8 * mag, "triangle {t} component {k}");
                }
            }
        }
    }
}

#[test]
fn prop_the_closed_form_stress_states_hold_for_every_material() {
    // Uniaxial strain gives sigma_x = E eps and a lateral stress of
    // exactly zero when the transverse strain is -nu eps; pure shear
    // gives tau = G gamma with G = E/(2(1+nu)) and a von Mises value of
    // sqrt(3) tau; and equal biaxial tension gives a von Mises value of
    // the tension itself -- not zero, because plane stress has a free
    // surface and so a state that is hydrostatic in plane is not
    // hydrostatic at all.
    let mut rng = Rng::new(0x0cd7_5e19);
    for _ in 0..40 {
        let (e, nu) = (1.0 + 500.0 * rng.next_f64(), 0.49 * rng.next_f64());
        let m = FemMesh2::rect(1.0, 1.0, 3, 3).unwrap();
        let eps = 1e-3 * (rng.next_f64() + 0.1);
        let uni: Vec<Vec2> =
            m.nodes.iter().map(|p| Vec2::new(eps * p.x, -nu * eps * p.y)).collect();
        let s = element_stress(&m, &uni, e, nu, 0).unwrap();
        assert!((s[0] - e * eps).abs() < 1e-9 * e * eps);
        assert!(s[1].abs() < 1e-9 * e * eps, "the lateral stress was {}", s[1]);
        assert!(s[2].abs() < 1e-9 * e * eps);
        assert!((von_mises_stress(&m, &uni, e, nu).unwrap()[0] - e * eps).abs() < 1e-9 * e * eps);
        let gamma = 1e-3 * (rng.next_f64() + 0.1);
        let sh: Vec<Vec2> = m
            .nodes
            .iter()
            .map(|p| Vec2::new(0.5 * gamma * p.y, 0.5 * gamma * p.x))
            .collect();
        let g_mod = e / (2.0 * (1.0 + nu));
        let ss = element_stress(&m, &sh, e, nu, 0).unwrap();
        assert!((ss[2] - g_mod * gamma).abs() < 1e-9 * g_mod * gamma);
        let vm = von_mises_stress(&m, &sh, e, nu).unwrap()[0];
        assert!((vm - 3.0f64.sqrt() * g_mod * gamma).abs() < 1e-9 * vm);
        let bi: Vec<Vec2> = m.nodes.iter().map(|p| Vec2::new(eps * p.x, eps * p.y)).collect();
        let bs = element_stress(&m, &bi, e, nu, 0).unwrap();
        assert!((bs[0] - bs[1]).abs() < 1e-9 * bs[0].abs());
        let bvm = von_mises_stress(&m, &bi, e, nu).unwrap()[0];
        assert!((bvm - bs[0].abs()).abs() < 1e-9 * bvm);
    }
}

#[test]
fn prop_von_mises_does_not_care_how_the_plane_is_turned() {
    // The equivalent stress is an invariant of the stress tensor, so
    // rotating the body and its displacement field together must leave
    // it alone element for element. Anything built from sigma_x and
    // sigma_y separately, rather than from their invariants, would fail
    // this.
    let mut rng = Rng::new(0x4f13_7ea2);
    for _ in 0..25 {
        let theta = std::f64::consts::TAU * rng.next_f64();
        let (c, sn) = (theta.cos(), theta.sin());
        let base = FemMesh2::rect(1.4, 0.9, 4, 3).unwrap();
        let turned = FemMesh2::new(
            base.nodes.iter().map(|p| Vec2::new(c * p.x - sn * p.y, sn * p.x + c * p.y)).collect(),
            base.tris.clone(),
        )
        .unwrap();
        let (e, nu) = (100.0, 0.3);
        let (a, b, d) = (
            2e-3 * (rng.next_f64() - 0.5),
            2e-3 * (rng.next_f64() - 0.5),
            2e-3 * (rng.next_f64() - 0.5),
        );
        let field = move |p: Vec2| Vec2::new(a * p.x + b * p.y, b * p.x + d * p.y);
        let u: Vec<Vec2> = base.nodes.iter().map(|&p| field(p)).collect();
        // Carry the displacement field around with the body.
        let spun: Vec<Vec2> = base
            .nodes
            .iter()
            .map(|&p| {
                let v = field(p);
                Vec2::new(c * v.x - sn * v.y, sn * v.x + c * v.y)
            })
            .collect();
        let plain = von_mises_stress(&base, &u, e, nu).unwrap();
        let rotated = von_mises_stress(&turned, &spun, e, nu).unwrap();
        let scale = plain.iter().fold(0.0f64, |x, &v| x.max(v)).max(1e-12);
        for t in 0..base.tris.len() {
            assert!((plain[t] - rotated[t]).abs() < 1e-9 * scale, "triangle {t}");
        }
    }
}

#[test]
fn prop_the_loads_do_twice_the_stored_energy() {
    // Clapeyron's theorem, which follows from the stiffness matrix being
    // symmetric and nothing else. Also the linear scalings: doubling
    // every load doubles the displacement and quadruples the energy, and
    // doubling the modulus halves the displacement.
    let mut rng = Rng::new(0x3ab6_02c4);
    for _ in 0..20 {
        let m = FemMesh2::rect(3.0, 1.0, 6, 2).unwrap();
        let (e, nu) = (50.0 + 200.0 * rng.next_f64(), 0.45 * rng.next_f64());
        let clamped: Vec<(usize, Vec2)> = m
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, p)| p.x < 1e-12)
            .map(|(i, _)| (i, Vec2::ZERO))
            .collect();
        let pull = Vec2::new(2.0 * rng.next_f64() - 1.0, 2.0 * rng.next_f64() - 1.0);
        let loads: Vec<(usize, Vec2)> = m
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, p)| (p.x - 3.0).abs() < 1e-12)
            .map(|(i, _)| (i, pull))
            .collect();
        let u = fem_2d_elasticity_plane_stress(&m, e, nu, &loads, &clamped).unwrap();
        let work: f64 = loads.iter().map(|&(i, f)| f.x * u[i].x + f.y * u[i].y).sum();
        let energy = strain_energy(&m, &u, e, nu).unwrap();
        assert!(work > 0.0, "the load did no work");
        assert!((work - 2.0 * energy).abs() < 1e-9 * work, "{work} against {}", 2.0 * energy);
        let doubled: Vec<(usize, Vec2)> =
            loads.iter().map(|&(i, f)| (i, Vec2::new(2.0 * f.x, 2.0 * f.y))).collect();
        let v = fem_2d_elasticity_plane_stress(&m, e, nu, &doubled, &clamped).unwrap();
        for i in 0..m.nodes.len() {
            assert!((v[i].x - 2.0 * u[i].x).abs() < 1e-9 * (1.0 + u[i].x.abs()));
            assert!((v[i].y - 2.0 * u[i].y).abs() < 1e-9 * (1.0 + u[i].y.abs()));
        }
        assert!(
            (strain_energy(&m, &v, e, nu).unwrap() - 4.0 * energy).abs() < 1e-9 * 4.0 * energy
        );
        let stiffer = fem_2d_elasticity_plane_stress(&m, 2.0 * e, nu, &loads, &clamped).unwrap();
        for i in 0..m.nodes.len() {
            assert!((stiffer[i].x - 0.5 * u[i].x).abs() < 1e-9 * (1.0 + u[i].x.abs()));
        }
    }
}

#[test]
fn prop_an_insulated_body_conserves_its_heat_to_the_last_digit() {
    // With no source and nothing prescribed, 1^T M u is unchanged by
    // every step and for every theta, because the stiffness rows sum to
    // zero. It is an algebraic consequence of the assembly rather than
    // an asymptotic property, so it holds at machine precision however
    // coarse the step.
    let mut rng = Rng::new(0x7e0b_3d55);
    for _ in 0..15 {
        let m = FemMesh2::rect(1.0, 1.0, 4, 4).unwrap();
        let k = 1.0 + 3.0 * rng.next_f64();
        let initial: Vec<f64> =
            m.nodes.iter().map(|p| (k * p.x).exp() + (k * p.y).sin()).collect();
        let mass = mass_matrix(&m);
        let total = |v: &[f64]| -> f64 { mass.mul_vec(v).iter().sum() };
        let start = total(&initial);
        for theta in [0.0, 0.5, 1.0] {
            let dt = 0.002 + 0.01 * rng.next_f64();
            let h = fem_2d_heat_transient(
                &m,
                &initial,
                0.1,
                dt,
                8,
                theta,
                &|_| 0.0,
                &|_| None,
            )
            .unwrap();
            for (n, step) in h.iter().enumerate() {
                assert!(
                    (total(step) - start).abs() < 1e-9 * start.abs(),
                    "theta {theta} step {n} changed the total heat"
                );
            }
            // Diffusion only flattens: the Dirichlet energy falls.
            let e0 = dirichlet_energy(&m, &h[0]).unwrap();
            let e1 = dirichlet_energy(&m, &h[8]).unwrap();
            assert!(e1 < e0 + 1e-12, "the field got rougher");
        }
    }
}

#[test]
fn prop_a_mode_follows_the_schemes_amplification_factor_exactly() {
    // Fed a discrete eigenmode the theta scheme is a scalar recurrence
    // with factor (1 - (1-theta)a)/(1 + theta a), a = alpha lambda dt.
    // That is an exact algebraic identity, so it separates the
    // time-stepping error from the spatial one completely -- there is no
    // spatial error left to confound it.
    let mut rng = Rng::new(0x2b8e_71c3);
    for _ in 0..12 {
        let m = FemMesh2::rect(1.0, 1.0, 4, 4).unwrap();
        let which = (rng.next_u64() % 3) as usize;
        let (values, modes) = fem_eigenmodes_drum(&m, which + 1).unwrap();
        let (lambda, phi) = (values[which], &modes[which]);
        let alpha = 0.1 + rng.next_f64();
        let dt = 0.005 + 0.02 * rng.next_f64();
        let a = alpha * lambda * dt;
        let peak = phi.iter().fold(0.0f64, |x, &v| x.max(v.abs()));
        for theta in [0.0, 0.5, 1.0] {
            // Forward Euler is only stable while a < 2; do not ask it
            // for something it does not promise.
            if theta == 0.0 && a >= 1.8 {
                continue;
            }
            let factor = (1.0 - (1.0 - theta) * a) / (1.0 + theta * a);
            let h = fem_2d_heat_transient(
                &m, phi, alpha, dt, 5, theta, &|_| 0.0, &|_| Some(0.0),
            )
            .unwrap();
            for (n, step) in h.iter().enumerate() {
                let want = factor.powi(n as i32);
                let worst = step
                    .iter()
                    .zip(phi.iter())
                    .filter(|(_, &p)| p.abs() > 0.5 * peak)
                    .map(|(&s, &p)| (s / p - want).abs())
                    .fold(0.0f64, f64::max);
                assert!(
                    worst < 1e-7 * (1.0 + want.abs()),
                    "theta {theta} step {n} drifted by {worst}"
                );
            }
        }
    }
}

#[test]
fn prop_crank_nicolson_is_a_stable_but_not_l_stable() {
    // For a mode too stiff to resolve, backward Euler's factor tends to
    // zero and Crank-Nicolson's tends to minus one. Both are stable;
    // only one damps. The stiff mode therefore dies under backward Euler
    // and survives under Crank-Nicolson, flipping sign every step --
    // which is why a discontinuous initial condition rings, and why the
    // usual remedy is to start with a couple of backward Euler steps.
    let mut rng = Rng::new(0x11c4_9a68);
    for _ in 0..12 {
        let m = FemMesh2::rect(1.0, 1.0, 4, 4).unwrap();
        let (values, modes) = fem_eigenmodes_drum(&m, 3).unwrap();
        let phi = &modes[2];
        let peak_at = phi
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .map(|(i, _)| i)
            .unwrap();
        let dt = (20.0 + 60.0 * rng.next_f64()) / values[2];
        let cn =
            fem_2d_heat_transient(&m, phi, 1.0, dt, 5, 0.5, &|_| 0.0, &|_| Some(0.0)).unwrap();
        let be =
            fem_2d_heat_transient(&m, phi, 1.0, dt, 5, 1.0, &|_| 0.0, &|_| Some(0.0)).unwrap();
        let start = phi[peak_at].abs();
        assert!(be[5][peak_at].abs() < 1e-3 * start, "backward Euler failed to damp");
        assert!(cn[5][peak_at].abs() > 0.4 * start, "Crank-Nicolson damped a stiff mode");
        for n in 0..5 {
            assert!(cn[n][peak_at] * cn[n + 1][peak_at] < 0.0, "no oscillation at step {n}");
        }
        // Both stay bounded, which is what A-stability means and is the
        // half forward Euler would fail here.
        assert!(cn[5][peak_at].abs() <= start * (1.0 + 1e-9));
    }
}

#[test]
fn prop_the_march_settles_onto_the_steady_solution() {
    // Long enough with a fixed source and fixed boundary, the transient
    // has to become the Poisson solution -- the two solvers are
    // consistent or one of them is wrong.
    let mut rng = Rng::new(0x59fa_c206);
    for _ in 0..10 {
        let m = FemMesh2::rect(1.0, 1.0, 4, 4).unwrap();
        let c = 2.0 * rng.next_f64() - 1.0;
        let source = move |p: Vec2| 1.0 + c * p.x;
        let hot = move |p: Vec2| Some(c * p.y);
        let steady = fem_2d_poisson(&m, &source, &hot).unwrap();
        let h = fem_2d_heat_transient(
            &m,
            &vec![0.0; m.nodes.len()],
            1.0,
            0.05,
            150,
            1.0,
            &source,
            &hot,
        )
        .unwrap();
        for i in 0..m.nodes.len() {
            assert!(
                (h[150][i] - steady[i]).abs() < 1e-7 * (1.0 + steady[i].abs()),
                "node {i}: {} against the steady {}",
                h[150][i],
                steady[i]
            );
        }
    }
}
