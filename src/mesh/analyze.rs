//! Mesh analysis: topology (manifoldness, orientation, boundary,
//! components, genus), quality statistics, QEM decimation, discrete
//! curvatures, and geodesic distances.

use crate::math::Vec3;
use crate::mesh::Mesh;
use crate::spatial::intersect::triangle_triangle;
use crate::spatial::Bvh;
use std::collections::{BinaryHeap, HashMap, HashSet};

type EdgeKey = (usize, usize);

fn edge_key(a: usize, b: usize) -> EdgeKey {
    (a.min(b), a.max(b))
}

fn edge_face_map(m: &Mesh) -> HashMap<EdgeKey, Vec<usize>> {
    let mut map: HashMap<EdgeKey, Vec<usize>> = HashMap::new();
    for (fi, &[a, b, c]) in m.indices.iter().enumerate() {
        for (u, v) in [(a, b), (b, c), (c, a)] {
            map.entry(edge_key(u, v)).or_default().push(fi);
        }
    }
    map
}

/// Summary statistics of a mesh.
#[derive(Debug, Clone, PartialEq)]
pub struct MeshStats {
    pub vertices: usize,
    pub edges: usize,
    pub faces: usize,
    pub euler: i64,
    /// Genus (2 − χ)/2 for a closed, connected, oriented manifold;
    /// `None` otherwise.
    pub genus: Option<u32>,
    pub boundary_loops: usize,
    pub is_manifold: bool,
    pub is_closed: bool,
    pub is_oriented: bool,
    pub min_angle_deg: f64,
    pub max_angle_deg: f64,
    pub min_edge: f64,
    pub max_edge: f64,
}

/// Computes all [`MeshStats`] in one pass over the mesh.
#[must_use]
pub fn stats(m: &Mesh) -> MeshStats {
    let edges = m.edges();
    let euler = euler_characteristic(m);
    let manifold = is_manifold(m);
    let closed = is_closed(m);
    let oriented = is_consistently_oriented(m);
    let single = connected_components(m).len() == 1;
    let genus = if manifold && closed && oriented && single && (2 - euler) % 2 == 0 && euler <= 2
    {
        Some(((2 - euler) / 2) as u32)
    } else {
        None
    };
    let mut min_angle = f64::INFINITY;
    let mut max_angle: f64 = 0.0;
    for t in m.triangles() {
        for (o, p, q) in [(t.a, t.b, t.c), (t.b, t.c, t.a), (t.c, t.a, t.b)] {
            let angle = (p - o).angle_between(&(q - o));
            min_angle = min_angle.min(angle);
            max_angle = max_angle.max(angle);
        }
    }
    let mut min_edge = f64::INFINITY;
    let mut max_edge: f64 = 0.0;
    for &(a, b) in &edges {
        let l = m.vertices[a].distance_to(&m.vertices[b]);
        min_edge = min_edge.min(l);
        max_edge = max_edge.max(l);
    }
    MeshStats {
        vertices: m.vertices.len(),
        edges: edges.len(),
        faces: m.indices.len(),
        euler,
        genus,
        boundary_loops: boundary_loops(m).len(),
        is_manifold: manifold,
        is_closed: closed,
        is_oriented: oriented,
        min_angle_deg: min_angle.to_degrees(),
        max_angle_deg: max_angle.to_degrees(),
        min_edge,
        max_edge,
    }
}

/// Euler characteristic χ = V − E + F.
#[must_use]
pub fn euler_characteristic(m: &Mesh) -> i64 {
    m.vertices.len() as i64 - m.edges().len() as i64 + m.indices.len() as i64
}

/// True when every edge belongs to one or two faces and the faces
/// around every vertex form a single fan (connected through shared
/// edges at that vertex).
#[must_use]
pub fn is_manifold(m: &Mesh) -> bool {
    let map = edge_face_map(m);
    if map.values().any(|f| f.len() > 2) {
        return false;
    }
    // Vertex fan connectivity.
    let mut vertex_faces: Vec<Vec<usize>> = vec![Vec::new(); m.vertices.len()];
    for (fi, &[a, b, c]) in m.indices.iter().enumerate() {
        for v in [a, b, c] {
            vertex_faces[v].push(fi);
        }
    }
    for (v, faces) in vertex_faces.iter().enumerate() {
        if faces.len() <= 1 {
            continue;
        }
        // BFS over faces sharing an edge incident to v.
        let set: HashSet<usize> = faces.iter().copied().collect();
        let mut seen = HashSet::new();
        let mut stack = vec![faces[0]];
        seen.insert(faces[0]);
        while let Some(f) = stack.pop() {
            let [a, b, c] = m.indices[f];
            for (p, q) in [(a, b), (b, c), (c, a)] {
                if p != v && q != v {
                    continue;
                }
                for &g in &map[&edge_key(p, q)] {
                    if set.contains(&g) && seen.insert(g) {
                        stack.push(g);
                    }
                }
            }
        }
        if seen.len() != faces.len() {
            return false;
        }
    }
    true
}

/// True when every edge belongs to exactly two faces.
#[must_use]
pub fn is_closed(m: &Mesh) -> bool {
    !m.indices.is_empty() && edge_face_map(m).values().all(|f| f.len() == 2)
}

/// True when every shared edge is traversed once in each direction
/// (faces agree on winding).
#[must_use]
pub fn is_consistently_oriented(m: &Mesh) -> bool {
    let mut directed: HashSet<(usize, usize)> = HashSet::new();
    for &[a, b, c] in &m.indices {
        for e in [(a, b), (b, c), (c, a)] {
            if !directed.insert(e) {
                return false; // same directed edge twice
            }
        }
    }
    true
}

/// Makes the orientation consistent per connected component by BFS
/// flipping. Returns false (leaving a best-effort result) when the
/// mesh is non-orientable (e.g. a Möbius band) or an edge has more
/// than two faces. A closed, consistently oriented component with
/// negative volume is flipped outward.
pub fn fix_orientation(m: &mut Mesh) -> bool {
    let map = edge_face_map(m);
    if map.values().any(|f| f.len() > 2) {
        return false;
    }
    let nf = m.indices.len();
    let mut visited = vec![false; nf];
    let mut ok = true;
    for start in 0..nf {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        let mut queue = vec![start];
        while let Some(f) = queue.pop() {
            let [a, b, c] = m.indices[f];
            for (u, v) in [(a, b), (b, c), (c, a)] {
                for &g in &map[&edge_key(u, v)] {
                    if g == f {
                        continue;
                    }
                    // Does g traverse (u, v) in the same direction?
                    let [x, y, z] = m.indices[g];
                    let same = [(x, y), (y, z), (z, x)].contains(&(u, v));
                    if !visited[g] {
                        visited[g] = true;
                        if same {
                            m.indices[g].swap(1, 2);
                        }
                        queue.push(g);
                    } else if same {
                        ok = false; // non-orientable
                    }
                }
            }
        }
    }
    if ok && is_closed(m) && m.volume() < 0.0 {
        m.flip_normals();
    }
    ok
}

/// Boundary loops as ordered vertex index cycles (each loop closed
/// implicitly; first vertex not repeated).
#[must_use]
pub fn boundary_loops(m: &Mesh) -> Vec<Vec<usize>> {
    let map = edge_face_map(m);
    // Directed boundary edges follow face orientation.
    let mut next: HashMap<usize, usize> = HashMap::new();
    for &[a, b, c] in &m.indices {
        for (u, v) in [(a, b), (b, c), (c, a)] {
            if map[&edge_key(u, v)].len() == 1 {
                next.insert(u, v);
            }
        }
    }
    let mut loops = Vec::new();
    let mut visited: HashSet<usize> = HashSet::new();
    let mut starts: Vec<usize> = next.keys().copied().collect();
    starts.sort_unstable();
    for start in starts {
        if visited.contains(&start) {
            continue;
        }
        let mut lp = vec![start];
        visited.insert(start);
        let mut cur = next[&start];
        while cur != start {
            visited.insert(cur);
            lp.push(cur);
            match next.get(&cur) {
                Some(&n) => cur = n,
                None => break, // dangling (non-manifold boundary)
            }
        }
        loops.push(lp);
    }
    loops
}

/// Splits into connected components (each with its own compacted
/// vertex list), ordered by smallest original vertex index.
#[must_use]
pub fn connected_components(m: &Mesh) -> Vec<Mesh> {
    let n = m.vertices.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    for &[a, b, c] in &m.indices {
        for (u, v) in [(a, b), (a, c)] {
            let (ru, rv) = (find(&mut parent, u), find(&mut parent, v));
            if ru != rv {
                parent[ru.max(rv)] = ru.min(rv);
            }
        }
    }
    let mut groups: HashMap<usize, usize> = HashMap::new(); // root -> component id
    let mut comps: Vec<(Vec<usize>, Vec<[usize; 3]>)> = Vec::new();
    for i in 0..n {
        let r = find(&mut parent, i);
        let id = *groups.entry(r).or_insert_with(|| {
            comps.push((Vec::new(), Vec::new()));
            comps.len() - 1
        });
        comps[id].0.push(i);
    }
    for &f in &m.indices {
        let id = groups[&find(&mut parent, f[0])];
        comps[id].1.push(f);
    }
    comps
        .into_iter()
        .filter(|(verts, faces)| !faces.is_empty() || verts.len() == 1)
        .map(|(verts, faces)| {
            let remap: HashMap<usize, usize> =
                verts.iter().enumerate().map(|(new, &old)| (old, new)).collect();
            Mesh {
                vertices: verts.iter().map(|&i| m.vertices[i]).collect(),
                indices: faces
                    .iter()
                    .map(|&[a, b, c]| [remap[&a], remap[&b], remap[&c]])
                    .collect(),
                normals: None,
                uvs: None,
            }
        })
        .collect()
}

/// Edges belonging to more than two faces.
#[must_use]
pub fn non_manifold_edges(m: &Mesh) -> Vec<EdgeKey> {
    let mut out: Vec<EdgeKey> = edge_face_map(m)
        .into_iter()
        .filter(|(_, f)| f.len() > 2)
        .map(|(e, _)| e)
        .collect();
    out.sort_unstable();
    out
}

/// Pairs of faces with identical vertex sets (regardless of winding),
/// each pair reported once as `(earlier, later)`.
#[must_use]
pub fn duplicate_faces(m: &Mesh) -> Vec<(usize, usize)> {
    let mut seen: HashMap<[usize; 3], Vec<usize>> = HashMap::new();
    for (fi, &f) in m.indices.iter().enumerate() {
        let mut s = f;
        s.sort_unstable();
        seen.entry(s).or_default().push(fi);
    }
    let mut out = Vec::new();
    for group in seen.values() {
        for w in 1..group.len() {
            out.push((group[0], group[w]));
        }
    }
    out.sort_unstable();
    out
}

/// Face pairs that intersect without sharing a vertex index. Pass the
/// mesh's BVH ([`Mesh::build_bvh`]) to prune candidate pairs.
#[must_use]
pub fn self_intersections(m: &Mesh, bvh: Option<&Bvh>) -> Vec<(usize, usize)> {
    let tris = m.to_triangles();
    let candidates: Vec<(usize, usize)> = match bvh {
        Some(b) => b.self_overlaps(),
        None => {
            let boxes: Vec<_> = tris
                .iter()
                .map(|t| crate::spatial::Aabb::from_points(&[t.a, t.b, t.c]))
                .collect();
            let mut pairs = Vec::new();
            for i in 0..boxes.len() {
                for j in i + 1..boxes.len() {
                    if boxes[i].intersection(&boxes[j]).is_some() {
                        pairs.push((i, j));
                    }
                }
            }
            pairs
        }
    };
    let mut out = Vec::new();
    for (i, j) in candidates {
        let (fa, fb) = (m.indices[i], m.indices[j]);
        if fa.iter().any(|v| fb.contains(v)) {
            continue; // shared vertex: adjacency, not intersection
        }
        if triangle_triangle(&tris[i], &tris[j]) {
            out.push((i.min(j), i.max(j)));
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Symmetric 4x4 quadric accumulated as a flat matrix.
type Quadric = [[f64; 4]; 4];

fn plane_quadric(t: &crate::spatial::Triangle) -> Quadric {
    let n = t.normal();
    let d = -n.dot(&t.a);
    let p = [n.x, n.y, n.z, d];
    let mut q = [[0.0; 4]; 4];
    for (r, row) in q.iter_mut().enumerate() {
        for (s, entry) in row.iter_mut().enumerate() {
            *entry = p[r] * p[s];
        }
    }
    q
}

fn quadric_add(a: &mut Quadric, b: &Quadric) {
    for r in 0..4 {
        for s in 0..4 {
            a[r][s] += b[r][s];
        }
    }
}

fn quadric_cost(q: &Quadric, v: Vec3) -> f64 {
    let x = [v.x, v.y, v.z, 1.0];
    let mut acc = 0.0;
    for (r, row) in q.iter().enumerate() {
        for (s, &entry) in row.iter().enumerate() {
            acc += x[r] * entry * x[s];
        }
    }
    acc
}

/// Optimal collapse position for a quadric, or `None` when the 3x3
/// system is (near) singular.
fn quadric_optimum(q: &Quadric) -> Option<Vec3> {
    let a = [
        [q[0][0], q[0][1], q[0][2]],
        [q[1][0], q[1][1], q[1][2]],
        [q[2][0], q[2][1], q[2][2]],
    ];
    let b = [-q[0][3], -q[1][3], -q[2][3]];
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    let scale: f64 = a.iter().flatten().map(|x| x.abs()).sum();
    if det.abs() < 1e-10 * scale.powi(3).max(1e-30) {
        return None;
    }
    let inv = |r: usize, s: usize| {
        // Cofactor expansion for the inverse times determinant.
        let (r1, r2) = match r {
            0 => (1, 2),
            1 => (0, 2),
            _ => (0, 1),
        };
        let (s1, s2) = match s {
            0 => (1, 2),
            1 => (0, 2),
            _ => (0, 1),
        };
        let sign = if (r + s).is_multiple_of(2) { 1.0 } else { -1.0 };
        sign * (a[s1][r1] * a[s2][r2] - a[s1][r2] * a[s2][r1])
    };
    let mut x = [0.0; 3];
    for (r, xr) in x.iter_mut().enumerate() {
        for (s, &bs) in b.iter().enumerate() {
            *xr += inv(r, s) * bs;
        }
        *xr /= det;
    }
    Some(Vec3::new(x[0], x[1], x[2]))
}

#[derive(PartialEq)]
struct Candidate {
    cost: f64,
    a: usize,
    b: usize,
    va: u64,
    vb: u64,
    pos: Vec3,
}

impl Eq for Candidate {}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Min-heap on cost.
        other.cost.total_cmp(&self.cost)
    }
}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Garland-Heckbert quadric error metric decimation ("Surface
/// Simplification Using Quadric Error Metrics", SIGGRAPH 1997):
/// greedily collapses the cheapest edge whose collapse keeps the mesh
/// manifold (link condition) and does not flip surviving faces, until
/// at most `target_faces` faces remain or no valid collapse exists.
#[must_use]
pub fn decimate_edge_collapse(m: &Mesh, target_faces: usize) -> Mesh {
    let mut verts = m.vertices.clone();
    let mut faces: Vec<Option<[usize; 3]>> = m.indices.iter().copied().map(Some).collect();
    let mut vertex_faces: Vec<HashSet<usize>> = vec![HashSet::new(); verts.len()];
    let mut quadrics: Vec<Quadric> = vec![[[0.0; 4]; 4]; verts.len()];
    for (fi, &f) in m.indices.iter().enumerate() {
        let t = m.triangle(fi);
        if t.area() == 0.0 {
            continue;
        }
        let q = plane_quadric(&t);
        for v in f {
            vertex_faces[v].insert(fi);
            quadric_add(&mut quadrics[v], &q);
        }
    }
    let mut versions: Vec<u64> = vec![0; verts.len()];
    let mut alive_faces = faces.iter().filter(|f| f.is_some()).count();
    let mut heap: BinaryHeap<Candidate> = BinaryHeap::new();
    let neighbors = |vf: &[HashSet<usize>], faces: &[Option<[usize; 3]>], v: usize| {
        let mut out: HashSet<usize> = HashSet::new();
        for &fi in &vf[v] {
            if let Some(f) = faces[fi] {
                for w in f {
                    if w != v {
                        out.insert(w);
                    }
                }
            }
        }
        out
    };
    let push_edges = |heap: &mut BinaryHeap<Candidate>,
                          quadrics: &[Quadric],
                          versions: &[u64],
                          verts: &[Vec3],
                          vf: &[HashSet<usize>],
                          faces: &[Option<[usize; 3]>],
                          v: usize| {
        for w in neighbors(vf, faces, v) {
            let (a, b) = (v.min(w), v.max(w));
            let mut q = quadrics[a];
            quadric_add(&mut q, &quadrics[b]);
            let pos = quadric_optimum(&q).unwrap_or_else(|| {
                let mid = (verts[a] + verts[b]) * 0.5;
                [verts[a], verts[b], mid]
                    .into_iter()
                    .min_by(|x, y| quadric_cost(&q, *x).total_cmp(&quadric_cost(&q, *y)))
                    .expect("nonempty")
            });
            heap.push(Candidate {
                cost: quadric_cost(&q, pos),
                a,
                b,
                va: versions[a],
                vb: versions[b],
                pos,
            });
        }
    };
    for v in 0..verts.len() {
        // Seed with edges from lower index only to halve duplicates.
        for w in neighbors(&vertex_faces, &faces, v) {
            if w > v {
                let mut q = quadrics[v];
                quadric_add(&mut q, &quadrics[w]);
                let pos = quadric_optimum(&q)
                    .unwrap_or_else(|| (verts[v] + verts[w]) * 0.5);
                heap.push(Candidate {
                    cost: quadric_cost(&q, pos),
                    a: v,
                    b: w,
                    va: 0,
                    vb: 0,
                    pos,
                });
            }
        }
    }
    while alive_faces > target_faces {
        let Some(c) = heap.pop() else { break };
        if versions[c.a] != c.va || versions[c.b] != c.vb {
            continue; // stale entry
        }
        let (a, b) = (c.a, c.b);
        // The edge must still exist.
        let shared: Vec<usize> = vertex_faces[a]
            .intersection(&vertex_faces[b])
            .copied()
            .filter(|&fi| faces[fi].is_some())
            .collect();
        if shared.is_empty() {
            continue;
        }
        // Link condition: common neighbors must be exactly the
        // opposite vertices of the shared faces.
        let na = neighbors(&vertex_faces, &faces, a);
        let nb = neighbors(&vertex_faces, &faces, b);
        let common: HashSet<usize> = na.intersection(&nb).copied().collect();
        let opposite: HashSet<usize> = shared
            .iter()
            .filter_map(|&fi| faces[fi])
            .flat_map(|f| f.into_iter())
            .filter(|&v| v != a && v != b)
            .collect();
        if common != opposite || shared.len() > 2 {
            continue;
        }
        // Reject collapses that flip a surviving face.
        let mut flips = false;
        for &fi in vertex_faces[a].union(&vertex_faces[b]) {
            let Some(f) = faces[fi] else { continue };
            if shared.contains(&fi) {
                continue;
            }
            let old = [verts[f[0]], verts[f[1]], verts[f[2]]];
            let mapped = f.map(|v| if v == a || v == b { c.pos } else { verts[v] });
            let n_old = (old[1] - old[0]).cross(&(old[2] - old[0]));
            let n_new = (mapped[1] - mapped[0]).cross(&(mapped[2] - mapped[0]));
            if n_old.dot(&n_new) <= 0.0 {
                flips = true;
                break;
            }
        }
        if flips {
            continue;
        }
        // Perform the collapse: b merges into a at the optimal point.
        verts[a] = c.pos;
        let qb = quadrics[b];
        quadric_add(&mut quadrics[a], &qb);
        for &fi in &shared {
            if faces[fi].take().is_some() {
                alive_faces -= 1;
            }
        }
        let b_faces: Vec<usize> = vertex_faces[b].iter().copied().collect();
        for fi in b_faces {
            if let Some(f) = faces[fi].as_mut() {
                for v in f.iter_mut() {
                    if *v == b {
                        *v = a;
                    }
                }
                vertex_faces[a].insert(fi);
            }
        }
        vertex_faces[b].clear();
        versions[a] += 1;
        versions[b] += 1;
        push_edges(&mut heap, &quadrics, &versions, &verts, &vertex_faces, &faces, a);
    }
    // Compact the result.
    let kept: Vec<[usize; 3]> = faces.into_iter().flatten().collect();
    let mut remap: HashMap<usize, usize> = HashMap::new();
    let mut vertices = Vec::new();
    let indices = kept
        .iter()
        .map(|&f| {
            f.map(|v| {
                *remap.entry(v).or_insert_with(|| {
                    vertices.push(verts[v]);
                    vertices.len() - 1
                })
            })
        })
        .collect();
    Mesh { vertices, indices, normals: None, uvs: None }
}

/// Number of distinct neighbors of each vertex.
#[must_use]
pub fn vertex_valence(m: &Mesh) -> Vec<usize> {
    m.adjacency().iter().map(|n| n.len()).collect()
}

/// Angle between the normals of the two faces at each interior edge
/// (0 for coplanar faces), aligned with [`Mesh::edges`] order;
/// boundary and non-manifold edges get 0.
#[must_use]
pub fn dihedral_angles(m: &Mesh) -> Vec<f64> {
    let map = edge_face_map(m);
    let normals = m.face_normals();
    m.edges()
        .iter()
        .map(|e| {
            let fs = &map[e];
            if fs.len() == 2 {
                normals[fs[0]].angle_between(&normals[fs[1]])
            } else {
                0.0
            }
        })
        .collect()
}

/// Interior edges whose faces' normals differ by more than the
/// threshold angle (radians).
#[must_use]
pub fn sharp_edges(m: &Mesh, angle_threshold_rad: f64) -> Vec<EdgeKey> {
    let edges = m.edges();
    dihedral_angles(m)
        .iter()
        .zip(&edges)
        .filter(|(&angle, _)| angle > angle_threshold_rad)
        .map(|(_, &e)| e)
        .collect()
}

/// Integrated Gaussian curvature per vertex as the angle deficit:
/// 2π − Σ incident angles (π − Σ on the boundary). Summing over a
/// closed mesh gives exactly 2πχ (discrete Gauss-Bonnet).
#[must_use]
pub fn discrete_gaussian_curvature(m: &Mesh) -> Vec<f64> {
    let map = edge_face_map(m);
    let mut on_boundary = vec![false; m.vertices.len()];
    for (&(a, b), fs) in &map {
        if fs.len() == 1 {
            on_boundary[a] = true;
            on_boundary[b] = true;
        }
    }
    let mut angle_sum = vec![0.0f64; m.vertices.len()];
    for &[a, b, c] in &m.indices {
        for (o, p, q) in [(a, b, c), (b, c, a), (c, a, b)] {
            angle_sum[o] +=
                (m.vertices[p] - m.vertices[o]).angle_between(&(m.vertices[q] - m.vertices[o]));
        }
    }
    angle_sum
        .iter()
        .zip(&on_boundary)
        .map(|(&s, &bd)| if bd { std::f64::consts::PI - s } else { 2.0 * std::f64::consts::PI - s })
        .collect()
}

/// Discrete (unsigned) mean curvature per vertex via the cotangent
/// Laplacian: H = |Σ (cot α + cot β)(vᵢ − vⱼ)| / (4 A) with A the
/// mixed Voronoi vertex area (Meyer et al. 2003). Zero on boundary
/// vertices.
#[must_use]
pub fn discrete_mean_curvature(m: &Mesh) -> Vec<f64> {
    let n = m.vertices.len();
    let mut lap = vec![Vec3::ZERO; n];
    let mut area = vec![0.0f64; n];
    let map = edge_face_map(m);
    let mut on_boundary = vec![false; n];
    for (&(a, b), fs) in &map {
        if fs.len() == 1 {
            on_boundary[a] = true;
            on_boundary[b] = true;
        }
    }
    for &[a, b, c] in &m.indices {
        let t_area = crate::spatial::Triangle {
            a: m.vertices[a],
            b: m.vertices[b],
            c: m.vertices[c],
        }
        .area();
        if t_area == 0.0 {
            continue;
        }
        // Corner data: (vertex, cot of its angle, obtuse?).
        let corner: Vec<(usize, f64, bool)> = [(a, b, c), (b, c, a), (c, a, b)]
            .into_iter()
            .map(|(o, p, q)| {
                let u = m.vertices[p] - m.vertices[o];
                let w = m.vertices[q] - m.vertices[o];
                let dot = u.dot(&w);
                (o, dot / u.cross(&w).magnitude(), dot < 0.0)
            })
            .collect();
        let any_obtuse = corner.iter().any(|&(_, _, ob)| ob);
        for k in 0..3 {
            let (o, _, obtuse_here) = corner[k];
            let (p, cot_p, _) = corner[(k + 1) % 3];
            let (q, cot_q, _) = corner[(k + 2) % 3];
            // Mixed Voronoi area (Meyer et al.): true Voronoi piece in
            // non-obtuse triangles, area/2 at the obtuse corner and
            // area/4 elsewhere otherwise.
            area[o] += if any_obtuse {
                if obtuse_here {
                    t_area / 2.0
                } else {
                    t_area / 4.0
                }
            } else {
                let d_op = (m.vertices[p] - m.vertices[o]).magnitude_squared();
                let d_oq = (m.vertices[q] - m.vertices[o]).magnitude_squared();
                (d_op * cot_q + d_oq * cot_p) / 8.0
            };
            // The corner's cot weights the opposite edge (p, q).
            let (_, cot_o, _) = corner[k];
            lap[p] = lap[p] + (m.vertices[p] - m.vertices[q]) * cot_o;
            lap[q] = lap[q] + (m.vertices[q] - m.vertices[p]) * cot_o;
        }
    }
    (0..n)
        .map(|i| {
            if on_boundary[i] || area[i] == 0.0 {
                0.0
            } else {
                lap[i].magnitude() / (4.0 * area[i])
            }
        })
        .collect()
}

#[derive(PartialEq)]
struct HeapEntry {
    dist: f64,
    vertex: usize,
}
impl Eq for HeapEntry {}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.dist.total_cmp(&self.dist)
    }
}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

fn dijkstra_with_pred(m: &Mesh, source: usize) -> (Vec<f64>, Vec<usize>) {
    assert!(source < m.vertices.len(), "source out of range");
    let adj = m.adjacency();
    let mut dist = vec![f64::INFINITY; m.vertices.len()];
    let mut pred = vec![usize::MAX; m.vertices.len()];
    dist[source] = 0.0;
    let mut heap = BinaryHeap::new();
    heap.push(HeapEntry { dist: 0.0, vertex: source });
    while let Some(HeapEntry { dist: d, vertex: v }) = heap.pop() {
        if d > dist[v] {
            continue;
        }
        for &w in &adj[v] {
            let nd = d + m.vertices[v].distance_to(&m.vertices[w]);
            if nd < dist[w] {
                dist[w] = nd;
                pred[w] = v;
                heap.push(HeapEntry { dist: nd, vertex: w });
            }
        }
    }
    (dist, pred)
}

/// Graph-shortest-path distance along mesh edges from `source` to
/// every vertex (an upper bound on true geodesic distance).
///
/// # Panics
/// Panics when `source` is out of range.
#[must_use]
pub fn geodesic_distance_dijkstra(m: &Mesh, source: usize) -> Vec<f64> {
    dijkstra_with_pred(m, source).0
}

/// First-order fast marching on the triangle mesh (Kimmel & Sethian
/// 1998): distances propagate as planar wavefronts across triangles,
/// converging to the true geodesic distance under refinement (unlike
/// edge-graph Dijkstra).
///
/// # Panics
/// Panics when `source` is out of range.
#[must_use]
pub fn geodesic_distance_fast_marching(m: &Mesh, source: usize) -> Vec<f64> {
    assert!(source < m.vertices.len(), "source out of range");
    let n = m.vertices.len();
    let mut vertex_faces: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (fi, &[a, b, c]) in m.indices.iter().enumerate() {
        for v in [a, b, c] {
            vertex_faces[v].push(fi);
        }
    }
    let mut dist = vec![f64::INFINITY; n];
    let mut known = vec![false; n];
    dist[source] = 0.0;
    let mut heap = BinaryHeap::new();
    heap.push(HeapEntry { dist: 0.0, vertex: source });
    // Planar wavefront update of vertex c from a triangle with both
    // other vertices known.
    let update = |c: Vec3, a: Vec3, b: Vec3, da: f64, db: f64| -> f64 {
        let fallback = (da + c.distance_to(&a)).min(db + c.distance_to(&b));
        // Local 2-D frame at c spanned by (a - c, b - c).
        let ca = a - c;
        let cb = b - c;
        let e1 = ca.normalized();
        let perp = cb - e1 * cb.dot(&e1);
        let pm = perp.magnitude();
        if pm < 1e-12 {
            return fallback;
        }
        let pa = crate::math::Vec2::new(ca.magnitude(), 0.0);
        let pb = crate::math::Vec2::new(cb.dot(&e1), pm);
        // Unit front normal g with g·(B - A) = db - da; d(x) = da +
        // g·(x - A); dc = da - g·A.
        let e = pb - pa;
        let u = db - da;
        let el2 = e.magnitude_squared();
        let w2 = el2 - u * u;
        if w2 <= 0.0 {
            return fallback;
        }
        let w = w2.sqrt();
        let mut best = fallback;
        let toward_a = pa.normalized();
        let toward_b = pb.normalized();
        let wedge = toward_a.cross(&toward_b);
        for sign in [-1.0, 1.0] {
            let g = (e * u + e.perp() * (sign * w)) * (1.0 / el2);
            // The wave reaches c travelling along +g, so the
            // characteristic through c comes from -g, which must lie
            // inside the wedge spanned by the two edges.
            let h = -g;
            if toward_a.cross(&h) * wedge < 0.0 || h.cross(&toward_b) * wedge < 0.0 {
                continue;
            }
            let dc = da - g.dot(&pa);
            if dc >= da.max(db) && dc < best {
                best = dc;
            }
        }
        best
    };
    while let Some(HeapEntry { dist: d, vertex: v }) = heap.pop() {
        if known[v] || d > dist[v] {
            continue;
        }
        known[v] = true;
        for &fi in &vertex_faces[v] {
            let f = m.indices[fi];
            for (i, &c) in f.iter().enumerate() {
                if known[c] {
                    continue;
                }
                let a = f[(i + 1) % 3];
                let b = f[(i + 2) % 3];
                let cand = if known[a] && known[b] {
                    update(m.vertices[c], m.vertices[a], m.vertices[b], dist[a], dist[b])
                } else if known[a] {
                    dist[a] + m.vertices[c].distance_to(&m.vertices[a])
                } else if known[b] {
                    dist[b] + m.vertices[c].distance_to(&m.vertices[b])
                } else {
                    continue;
                };
                if cand < dist[c] {
                    dist[c] = cand;
                    heap.push(HeapEntry { dist: cand, vertex: c });
                }
            }
        }
    }
    dist
}

/// Shortest edge path between two vertices (inclusive); empty when
/// unreachable.
///
/// # Panics
/// Panics when either endpoint is out of range.
#[must_use]
pub fn geodesic_path(m: &Mesh, from: usize, to: usize) -> Vec<usize> {
    assert!(to < m.vertices.len(), "target out of range");
    let (dist, pred) = dijkstra_with_pred(m, from);
    if dist[to].is_infinite() {
        return Vec::new();
    }
    let mut path = vec![to];
    let mut cur = to;
    while cur != from {
        cur = pred[cur];
        path.push(cur);
    }
    path.reverse();
    path
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::generate::{box_mesh, icosphere, plane_grid, torus, uv_sphere};

    #[test]
    fn test_stats_and_genus() {
        let s = stats(&icosphere(1.0, 2));
        assert!(s.is_manifold && s.is_closed && s.is_oriented);
        assert_eq!(s.euler, 2);
        assert_eq!(s.genus, Some(0));
        assert_eq!(s.boundary_loops, 0);
        assert!(s.min_angle_deg > 30.0 && s.max_angle_deg < 90.0);
        assert!(s.min_edge > 0.0 && s.max_edge < 1.5);

        let t = stats(&torus(2.0, 0.5, 24, 12));
        assert_eq!(t.euler, 0);
        assert_eq!(t.genus, Some(1));

        let p = stats(&plane_grid(1.0, 1.0, 3, 3));
        assert!(!p.is_closed && p.is_manifold);
        assert_eq!(p.boundary_loops, 1);
        assert_eq!(p.genus, None);
        assert_eq!(p.euler, 1);
    }

    #[test]
    fn test_manifold_detection() {
        // Three faces sharing one edge: non-manifold.
        let m = Mesh::new(
            vec![
                Vec3::ZERO,
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::new(0.0, -1.0, 0.0),
            ],
            vec![[0, 1, 2], [0, 1, 3], [0, 1, 4]],
        )
        .unwrap();
        assert!(!is_manifold(&m));
        assert_eq!(non_manifold_edges(&m), vec![(0, 1)]);
        // Two triangles touching only at a vertex: edge-manifold but
        // the vertex fan is disconnected.
        let bowtie = Mesh::new(
            vec![
                Vec3::ZERO,
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(1.0, -1.0, 0.0),
                Vec3::new(-1.0, 1.0, 0.0),
                Vec3::new(-1.0, -1.0, 0.0),
            ],
            vec![[0, 1, 2], [0, 3, 4]],
        )
        .unwrap();
        assert!(!is_manifold(&bowtie));
    }

    #[test]
    fn test_orientation_fixing() {
        let mut m = icosphere(1.0, 1);
        // Break a few windings.
        m.indices[3].swap(1, 2);
        m.indices[17].swap(1, 2);
        assert!(!is_consistently_oriented(&m));
        assert!(fix_orientation(&mut m));
        assert!(is_consistently_oriented(&m));
        assert!(m.volume() > 0.0, "closed mesh reoriented outward");
        // A Möbius band cannot be fixed.
        let n = 12;
        let mut verts = Vec::new();
        for i in 0..n {
            let a = std::f64::consts::PI * i as f64 / n as f64; // half twist
            let u = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
            let center = Vec3::new(u.cos() * 2.0, 0.0, u.sin() * 2.0);
            let dir = Vec3::new(u.cos() * a.cos(), a.sin(), u.sin() * a.cos());
            verts.push(center + dir * 0.4);
            verts.push(center - dir * 0.4);
        }
        let mut faces = Vec::new();
        for i in 0..n {
            let (a0, a1) = (2 * i, 2 * i + 1);
            // The strip closes with a flip: index parity swaps.
            let (b0, b1) = if i + 1 == n { (1, 0) } else { (2 * i + 2, 2 * i + 3) };
            faces.push([a0, b0, a1]);
            faces.push([a1, b0, b1]);
        }
        let mut mobius = Mesh::new(verts, faces).unwrap();
        assert!(!fix_orientation(&mut mobius), "Möbius band is non-orientable");
    }

    #[test]
    fn test_boundary_and_components() {
        let grid = plane_grid(1.0, 1.0, 2, 2);
        let loops = boundary_loops(&grid);
        assert_eq!(loops.len(), 1);
        assert_eq!(loops[0].len(), 8, "outer ring of a 2x2 grid has 8 vertices");
        let mut two = icosphere(1.0, 1);
        let mut far = icosphere(0.5, 1);
        far.translate(Vec3::new(5.0, 0.0, 0.0));
        two.merge(&far);
        let comps = connected_components(&two);
        assert_eq!(comps.len(), 2);
        assert_eq!(comps[0].indices.len(), 80);
        assert!((comps[1].volume() - far.volume()).abs() < 1e-12);
    }

    #[test]
    fn test_duplicates_and_self_intersections() {
        let mut m = box_mesh(Vec3::new(1.0, 1.0, 1.0));
        m.indices.push([m.indices[0][0], m.indices[0][2], m.indices[0][1]]);
        assert_eq!(duplicate_faces(&m), vec![(0, 12)]);

        // A clean box has no self-intersections.
        let b = box_mesh(Vec3::new(1.0, 1.0, 1.0));
        assert!(self_intersections(&b, None).is_empty());
        let bvh = b.build_bvh();
        assert!(self_intersections(&b, Some(&bvh)).is_empty());
        // Two interpenetrating boxes (merged) do intersect.
        let mut two = box_mesh(Vec3::new(1.0, 1.0, 1.0));
        let mut other = box_mesh(Vec3::new(1.0, 1.0, 1.0));
        other.translate(Vec3::new(0.9, 0.8, 0.7));
        two.merge(&other);
        let hits = self_intersections(&two, None);
        assert!(!hits.is_empty());
        let bvh = two.build_bvh();
        assert_eq!(self_intersections(&two, Some(&bvh)), hits);
    }

    #[test]
    fn test_gauss_bonnet_exact() {
        for (m, chi) in [
            (icosphere(1.0, 2), 2.0),
            (uv_sphere(2.0, 16, 9), 2.0),
            (torus(2.0, 0.5, 20, 10), 0.0),
            (box_mesh(Vec3::new(1.0, 2.0, 0.5)), 2.0),
        ] {
            let total: f64 = discrete_gaussian_curvature(&m).iter().sum();
            assert!(
                (total - 2.0 * std::f64::consts::PI * chi).abs() < 1e-9,
                "Gauss-Bonnet violated: {total}"
            );
        }
    }

    #[test]
    fn test_mean_curvature_sphere() {
        let r = 2.0;
        let m = icosphere(r, 3);
        let h = discrete_mean_curvature(&m);
        for &hi in &h {
            assert!((hi - 1.0 / r).abs() < 0.02 / r, "sphere mean curvature {hi}");
        }
    }

    #[test]
    fn test_valence_dihedral_sharp() {
        let ico = icosphere(1.0, 0);
        assert!(vertex_valence(&ico).iter().all(|&v| v == 5));
        let b = box_mesh(Vec3::new(1.0, 1.0, 1.0));
        let angles = dihedral_angles(&b);
        // Box edges are either flat (across a face diagonal) or 90°.
        for &a in &angles {
            assert!(a.abs() < 1e-9 || (a - std::f64::consts::FRAC_PI_2).abs() < 1e-9);
        }
        let sharp = sharp_edges(&b, 0.5);
        assert_eq!(sharp.len(), 12, "a box has 12 sharp edges");
    }

    #[test]
    fn test_decimation_preserves_genus() {
        let m = icosphere(1.0, 3);
        let d = decimate_edge_collapse(&m, 100);
        assert!(d.indices.len() <= 100);
        assert!(d.indices.len() > 20);
        let s = stats(&d);
        assert!(s.is_manifold && s.is_closed && s.is_oriented);
        assert_eq!(s.genus, Some(0), "decimation must preserve genus");
        // Still approximately the unit sphere.
        for v in &d.vertices {
            assert!((v.magnitude() - 1.0).abs() < 0.1);
        }
        let t = torus(2.0, 0.6, 24, 12);
        let dt = decimate_edge_collapse(&t, 150);
        assert!(dt.indices.len() <= 150);
        let st = stats(&dt);
        assert_eq!(st.genus, Some(1), "torus stays genus 1");
    }

    #[test]
    fn test_geodesics_on_sphere() {
        let r = 1.0;
        let m = icosphere(r, 4);
        let src = 0;
        let from = m.vertices[src];
        let dd = geodesic_distance_dijkstra(&m, src);
        let df = geodesic_distance_fast_marching(&m, src);
        let mut sum_d = 0.0;
        let mut sum_f = 0.0;
        let mut count = 0usize;
        for (i, v) in m.vertices.iter().enumerate() {
            let exact = r * from.angle_between(v);
            if exact < 0.5 {
                continue; // relative error unstable near the source
            }
            let ed = (dd[i] - exact).abs() / exact;
            let ef = (df[i] - exact).abs() / exact;
            // Dijkstra takes lattice-direction detours (up to ~20% on
            // a triangulated sphere); fast marching cuts across faces.
            assert!(ed < 0.25, "Dijkstra error {ed} at {i}");
            assert!(ef < 0.06, "fast marching error {ef} at {i}");
            sum_d += ed;
            sum_f += ef;
            count += 1;
        }
        let (avg_d, avg_f) = (sum_d / count as f64, sum_f / count as f64);
        assert!(avg_f < 0.03, "fast marching mean error {avg_f} should be under 3%");
        assert!(avg_f < avg_d / 2.0, "fast marching should clearly beat edge Dijkstra");
    }

    #[test]
    fn test_geodesic_path_endpoints() {
        let m = icosphere(1.0, 2);
        let to = m.vertices.len() - 1;
        let path = geodesic_path(&m, 0, to);
        assert_eq!(path[0], 0);
        assert_eq!(*path.last().unwrap(), to);
        // Consecutive path vertices are neighbors.
        let adj = m.adjacency();
        for w in path.windows(2) {
            assert!(adj[w[0]].contains(&w[1]));
        }
        // Path length equals the Dijkstra distance.
        let d = geodesic_distance_dijkstra(&m, 0);
        let len: f64 =
            path.windows(2).map(|w| m.vertices[w[0]].distance_to(&m.vertices[w[1]])).sum();
        assert!((len - d[to]).abs() < 1e-12);
    }
}
