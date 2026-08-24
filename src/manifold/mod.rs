//! Manifolds and higher-dimensional geometry: generic n-dimensional
//! vectors and tensors, metric-driven curvature, and (in later modules)
//! geodesics, Lie groups, constant-curvature spaces, polytopes, Clifford
//! algebras, embeddings, discrete exterior calculus, and spacetimes.

pub mod clifford;
pub mod dec;
pub mod embedding;
pub mod geodesic;
pub mod hyperbolic;
pub mod lie;
pub mod metric;
pub mod polytope4;
pub mod spacetime;
pub mod spherical;
pub mod vecn;

pub use clifford::{
    algebra_dimension, blade_name, cayley_table, cga3, cl3, is_isomorphic_to_known, pga3, sta,
    Multivector,
};
pub use dec::{
    betti_curve, persistence_diagram_bottleneck, persistent_homology_vietoris_rips, DecMesh,
};
pub use embedding::{
    blobs, classical_mds, continuity, diffusion_maps, dist_matrix, geodesic_distance_matrix,
    geodesic_kmeans, grassmann_distance, helix_sample, intrinsic_dimension_correlation,
    intrinsic_dimension_mle, intrinsic_dimension_two_nn, isomap, kernel_pca, klein_sample,
    knn_graph, laplacian_eigenmaps, lle, manifold_curvature_estimate,
    manifold_interpolation_rbf, metric_mds_smacof, mobius_sample, neighborhood_preservation,
    nonmetric_mds, pca, procrustes_align, riemannian_gradient_descent_sphere,
    riemannian_gradient_descent_stiefel, s_curve, sphere_sample, spectral_embedding,
    stiefel_project, stress, swiss_roll, tangent_space_estimate, torus_sample,
    trustworthiness, tsne, two_moons, umap_lite,
};
pub use geodesic::{
    geodesics_on_mesh_exact, great_circle_check, heat_method_geodesic, light_deflection,
    perihelion_precession, photon_orbit_stability, schwarzschild_orbit, shapiro_delay,
    GeodesicState, Integrator,
};
pub use hyperbolic::{
    apollonian_from_mobius, disk_to_hyperboloid, disk_to_klein, disk_to_uhp,
    equidistant_curve_disk, fundamental_polygon_genus, horocycle_disk, hyp_angle_of_parallelism,
    hyp_area_circle, hyp_area_triangle, hyp_centroid_disk, hyp_circle_disk, hyp_circumference,
    hyp_convex_hull_disk, hyp_delaunay_disk, hyp_distance_disk, hyp_distance_hyperboloid,
    hyp_distance_uhp, hyp_embed_graph_mds, hyp_embed_tree, hyp_geodesic_circle_disk,
    hyp_geodesic_disk, hyp_law_of_cosines, hyp_law_of_sines, hyp_mean_curvature_flow,
    hyp_tiling, hyp_tiling_exists, hyp_triangle_from_angles, hyp_volume_ball, hyp_voronoi_disk,
    hyperbolic_rotation, hyperbolic_translation, hyperboloid_to_disk,
    isometry_disk_from_two_points, klein_to_disk, limit_set_schottky,
    lorentz_boost_hyperboloid, mobius_disk, mobius_uhp, parabolic, poincare_embedding_train,
    uhp_to_disk, HypModel, HypPoint,
};
pub use lie::{
    casimir_so3, clebsch_gordan, hand_eye_calibration, killing_form, lie_bracket_matrix,
    matrix_exp, matrix_log, matrix_sqrt, pose_graph_optimize, rotate_spherical_harmonics,
    rotation_averaging, se3, so3, so3_haar_measure_density, so3_uniform_grid,
    structure_constants, umeyama_alignment, wigner_d, wigner_d_small, Heisenberg3, LieGroup,
    Se2, Se3, Sim3, Sl2C, Sl2Class, Sl2R, So2, So3, So4, Su2, Unitary,
};
pub use metric::{
    frw_metric, kerr_boyer_lindquist, schwarzschild_metric_fn,
    surface_metric_from_parametrization, warped_product, Metric, Sig,
};
pub use polytope4::{
    clifford_torus, clifford_torus_mesh, coxeter_plane_projection, cross_polytope_n,
    d4_lattice_points, e8_lattice_nearest, e8_roots, f4_roots, gaussian_concentration_radius,
    h4_roots, hypercube_graph_n, hypercube_n, hypercube_slicing_volume,
    hypersphere_cap_fraction, hypersphere_s3_points, hypersphere_volume, kissing_number_known,
    leech_lattice_min_vectors_count, petrie_polygon_projection, project_n_to_2, project_n_to_3,
    random_walk_n_return_prob, rotate_4d, rotate_4d_double, rotation_4d_planes, simplex_n,
    volume_ball_vs_cube_ratio, Polytope4, Vec4,
};
pub use spacetime::{
    bekenstein_entropy, black_hole_shadow_radius, cosmological_distances, eddington_finkelstein,
    evaporation_time, extra_dimension_gravity_law, frw_geodesic,
    gravitational_lens_einstein_radius, gw_chirp_mass, gw_waveform_inspiral, hawking_temperature,
    kaluza_klein_metric, kerr_geodesic_constants, kk_compactification_mass_spectrum,
    kk_reduce_geodesic_to_charged, kruskal_from_schwarzschild, lens_equation_solve,
    light_cone_check, orbit_schwarzschild_full, penrose_diagram_coords,
    photon_ray_trace_schwarzschild, point_lens_magnification, relativistic_rocket,
    rindler_coords, rindler_horizon, schwarzschild_geodesic_metric, simultaneity_plane,
    sta_vs_matrix_lorentz_check, twin_paradox_ages, unruh_temperature, Causal, FourVector,
    KerrConstants, LorentzTransform, Plane,
};
pub use spherical::{
    azimuthal_equidistant, equirectangular, gauss_legendre_sphere, gnomonic, haversine,
    healpix_ang2pix, healpix_npix, healpix_pix2ang, hopf_fiber, hopf_fiber_stereographic,
    hopf_fibration, inverse_stereographic, kent_distribution_pdf,
    lambert_azimuthal_equal_area, lebedev_quadrature, mercator, mollweide, orthographic,
    robinson, rotate_sphere_points, s3_geodesic, s3_uniform_points, sphere_cap_area,
    sphere_cap_volume, sphere_distance_n, sphere_exp_n, sphere_geodesic_n, sphere_log_n,
    sphere_parallel_transport_n, sphere_surface_n, sphere_uniform_points_n, sphere_volume_n,
    spherical_cap_packing, spherical_centroid, spherical_code_min_angle, spherical_convex_hull,
    spherical_convolution, spherical_delaunay, spherical_harmonic_inverse,
    spherical_harmonic_transform, spherical_harmonics_complex, spherical_heat_flow,
    spherical_kmeans, spherical_laplacian_spectral, spherical_law_of_cosines,
    spherical_law_of_sines, spherical_mean_weighted, spherical_polygon_area,
    spherical_t_design, spherical_triangle_angles, spherical_triangle_area,
    spherical_voronoi, spherical_wavelets, stereographic, stereographic_n, thomson_problem,
    vmf_fit, vmf_sample, von_mises_fisher_pdf,
};
pub use vecn::{
    determinant_n, exterior_derivative_numeric, wedge, TensorN, VecN,
};
