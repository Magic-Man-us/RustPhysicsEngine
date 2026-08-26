# ROADMAP PART 4: discrete math, exact arithmetic, stochastic processes, optimization, quantum, statistical mechanics, domains

> **Status: COMPLETE.** All thirty planned sessions are delivered, plus the
> two cross-references in "Cross-references and shared additions" that were
> not in the session list: `units::dimensional::dimensional_check_formula`
> (the `exact::symbolic` tie-in) and the `constants_codata` consolidation.
>
> Delivered against this plan: `exact/`, `discrete/`, `graph/`, `codes/`,
> `stochastic/`, `optimization/`, `quantum/`, `statistical_mechanics/`,
> `biophysics/`, `finance/`, `astrophysics/`, `fem/`, `learn/` and `units/`.
> The crate went from 2,981 to 4,193 unit tests and from 107 to 577 property
> tests over the course of it.
>
> The document below is kept as written — it is the specification the work was
> built against, and the function signatures in it are the ones that shipped.
> Where a signature changed in implementation the reason is recorded in the
> commit that changed it. Three specification readings that the plan left
> genuinely ambiguous are documented in the code rather than here:
> `quadratic_diophantine_solve` as a definite form, `frobenius_number` with a
> unit coin, and `stern_brocot_nth` indexed breadth-first.
>
> See the README for what the library now contains.

Same rules as Parts 1-3. This part covers everything computational that
the repo still lacks after 1-3: exact and big-number arithmetic, number
theory, combinatorics, graphs, error-correcting codes, stochastic
processes and time series, operations research and game theory, quantum
mechanics and quantum circuits, statistical mechanics and molecular
dynamics, epidemiology and population biology, quantitative finance,
astrodynamics, FEM/FDTD general PDE, machine learning from scratch, and
units.

Needs from earlier parts: Matrix, lu, eigen, svd, CsrMatrix, CG (Part 1),
distributions and quantiles (Part 1), fft (Part 3), Rng (existing),
SpatialHash (Part 2), dormand_prince (Part 1).

## Layout

```
src/
  exact/        bigint.rs rational.rs bigfloat.rs polynomial.rs contfrac.rs symbolic.rs
  discrete/     number_theory.rs primes.rs combinatorics.rs partitions.rs sequences.rs
  graph/        core.rs paths.rs flow.rs matching.rs spectral.rs coloring.rs layout.rs
  codes/        checksum.rs block.rs reed_solomon.rs convolutional.rs compression.rs crypto_math.rs
  stochastic/   markov.rs hmm.rs sde.rs point_process.rs queueing.rs timeseries.rs rmt.rs extreme.rs
  opt/          lp.rs integer.rs network.rs metaheuristics.rs convex.rs game_theory.rs schedule.rs
  quantum/      wavefunction.rs schrodinger.rs circuit.rs algorithms.rs spin.rs solid_state.rs
  statmech/     ising.rs md.rs kinetics.rs lattice_models.rs
  bio/          epidemiology.rs population.rs seq_align.rs phylo.rs neuro.rs
  finance/      options.rs rates.rs portfolio.rs risk.rs
  astro/        kepler.rs elements.rs lambert.rs maneuvers.rs time_systems.rs coords.rs
  fem/          fem1d.rs fem2d.rs fdtd.rs spectral_pde.rs
  learn/        nn.rs gp.rs cluster.rs tree.rs
  units/        quantity.rs dimensional.rs
```

## Phase A: exact arithmetic and symbolic

### 1. exact/bigint.rs
```rust
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct BigInt { pub sign: i8, pub limbs: Vec<u64> }   // little-endian base 2^64
impl BigInt {
    pub fn from_i64(n) / from_u64 / from_str_radix(s, radix) -> Result<Self, GeomError> / zero / one;
    pub fn to_string_radix(&self, radix) -> String;  to_f64;  to_i64() -> Option<i64>;  bits() -> usize;
    pub fn add / sub / neg / abs / mul (schoolbook, Karatsuba above 32 limbs) / div_rem -> (Self, Self);
    pub fn pow(&self, e: u64) -> Self;  mod_pow(&self, e: &Self, m: &Self) -> Self;   // sliding window
    pub fn gcd / lcm / extended_gcd -> (Self, Self, Self);
    pub fn mod_inverse(&self, m) -> Option<Self>;
    pub fn shl(bits) / shr / bit(i) / set_bit / and / or / xor;
    pub fn is_even / is_zero / is_negative;
    pub fn sqrt(&self) -> Self;  nth_root(n);  is_perfect_square;
    pub fn random_bits(bits, rng) -> Self;  random_below(bound, rng);
    pub fn factorial(n: u64) -> Self;  binomial(n, k: u64) -> Self;
    pub fn fibonacci(n: u64) -> Self;   // fast doubling
    pub fn cmp_abs(&self, o) -> Ordering;
}
impl Add/Sub/Mul/Div/Rem/Neg/Shl/Shr for BigInt
```
Property: (a*b)/b == a; extended_gcd Bezout identity holds; mod_pow matches
naive for small; from/to string roundtrip radix 2..=36; factorial(30)
matches known digits; Karatsuba matches schoolbook on random 100-limb.

### 2. exact/rational.rs and bigfloat.rs
```rust
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Rational { pub num: BigInt, pub den: BigInt }   // always reduced, den > 0
impl Rational { new(n, d) -> Option; from_i64(n, d); from_f64_exact(x) -> Option (IEEE bits); from_f64_approx(x, max_den) (Stern-Brocot); add/sub/mul/div/neg/recip/pow(i64); to_f64; floor/ceil/round/fract; cmp; abs; is_integer; to_continued_fraction() -> Vec<BigInt>; from_continued_fraction(cf); mediant(a, b); }
pub fn farey_sequence(n: u64) -> Vec<Rational>;
pub fn stern_brocot_path(r: &Rational) -> Vec<bool>;
pub fn best_rational_approximations(x: f64, max_den) -> Vec<Rational>;
pub fn solve_exact(a: &Matrix, b: &[f64]) -> Option<Vec<Rational>>;   // fraction-free Gaussian (Bareiss), exact answer for rational input
pub fn determinant_exact(a: &[Vec<Rational>]) -> Rational;
pub fn hilbert_matrix_inverse_exact(n) -> Vec<Vec<Rational>>;
pub struct BigFloat { pub mantissa: BigInt, pub exponent: i64, pub precision: usize }   // mantissa * 2^exponent, precision in bits
impl BigFloat { from_f64(x, prec); from_str(s, prec); add/sub/mul/div (correctly rounded to prec); sqrt; to_f64; to_string_decimal(digits); pi(prec) (Chudnovsky or AGM); e(prec); ln2(prec); exp/ln/sin/cos/atan (argument reduction + series); pow; agm(a, b); cmp; }
pub fn pi_digits(n_decimal) -> String;
pub fn e_digits(n) -> String;
pub fn sqrt2_digits(n) -> String;
pub fn machin_pi(prec) -> BigFloat;
pub fn compensated_to_bigfloat_check(xs: &[f64]) -> f64;   // validate Part1 sum_neumaier against exact
```
Property: Rational arithmetic exact (no drift over 1e4 ops); from_f64_exact
of 0.1 has denominator 2^k; solve_exact on Hilbert 8x8 gives exact known
inverse entries; pi_digits(100) matches published; BigFloat ops correctly
rounded vs higher precision reference.

### 3. exact/polynomial.rs, contfrac.rs, symbolic.rs
```rust
// polynomial.rs (coefficient vec, low to high; f64 and Rational variants via generic over Ring trait kept minimal: two concrete types)
pub struct Poly { pub c: Vec<f64> }
pub struct PolyQ { pub c: Vec<Rational> }
impl Poly { new(c); degree; eval (Horner); eval_complex; add/sub/mul/div_rem; derivative; integral(c0); compose; scale_arg; shift_arg; gcd (with tolerance); roots() (Part1 polynomial_roots); from_roots; resultant(o) -> f64; discriminant; sturm_sequence() -> Vec<Poly>; count_real_roots(a, b) (Sturm); isolate_real_roots() -> Vec<(f64, f64)>; refine_root(interval, tol); interpolate_lagrange(xs, ys); interpolate_newton(xs, ys); chebyshev_fit(f, a, b, n); chebyshev_eval(coeffs, x) (Clenshaw); to_chebyshev_basis / from_chebyshev_basis; pade(m, n) -> (Poly, Poly); wilkinson(n); cyclotomic(n) -> PolyQ; is_squarefree; squarefree_part; }
impl PolyQ { exact everything above minus roots; content; primitive_part; pseudo_div; gcd_exact (subresultant PRS); factor_rational_roots() -> Vec<(Rational, usize)>; eisenstein_check(p) -> bool; }
pub fn polynomial_multiply_fft(a: &Poly, b: &Poly) -> Poly;
pub fn bernstein_basis(n, i, t) -> f64;  pub fn to_bernstein(p: &Poly, a, b) -> Vec<f64>;
pub fn newton_identities(power_sums: &[f64]) -> Vec<f64>;   // elementary symmetric
pub fn vieta(roots: &[Complex]) -> Vec<Complex>;
// contfrac.rs
pub fn continued_fraction_f64(x, max_terms) -> Vec<i64>;
pub fn convergents(cf: &[i64]) -> Vec<Rational>;
pub fn periodic_cf_sqrt(n: u64) -> (Vec<u64>, Vec<u64>);   // sqrt(n) = [a0; period]
pub fn pell_fundamental_solution(d: u64) -> Option<(BigInt, BigInt)>;
pub fn generalized_cf_eval(a: &dyn Fn(usize) -> f64, b: &dyn Fn(usize) -> f64, n) -> f64;   // Lentz
pub fn cf_e(n) -> Vec<i64>;  pub fn cf_pi_terms(n) -> Vec<i64>;
pub fn gauss_map_orbit(x, n) -> Vec<f64>;  pub fn khinchin_estimate(x, n) -> f64;  pub fn levy_constant_estimate(x, n) -> f64;
// symbolic.rs (expression trees, numeric-first CAS-lite)
#[derive(Debug, Clone, PartialEq)]
pub enum Expr { Const(f64), Rat(Rational), Var(String), Add(Vec<Expr>), Mul(Vec<Expr>), Pow(Box<Expr>, Box<Expr>), Sin(Box<Expr>), Cos(Box<Expr>), Tan(..), Exp(..), Ln(..), Sqrt(..), Abs(..), Atan(..), Sinh(..), Cosh(..), Neg(..) }
impl Expr {
    pub fn parse(s: &str) -> Result<Expr, GeomError>;   // "3*x^2 + sin(y)/2", precedence climbing
    pub fn to_string(&self) -> String;  to_latex() -> String;
    pub fn eval(&self, vars: &[(&str, f64)]) -> Result<f64, GeomError>;
    pub fn diff(&self, var: &str) -> Expr;              // exact symbolic derivative
    pub fn gradient(&self, vars: &[&str]) -> Vec<Expr>;
    pub fn simplify(&self) -> Expr;                      // constant fold, flatten, collect like terms, x*0, x^1, sin^2+cos^2 -> 1, ln(exp)
    pub fn expand(&self) -> Expr;                        // distribute products over sums, multinomial pow with small integer exponent
    pub fn substitute(&self, var, replacement: &Expr) -> Expr;
    pub fn as_polynomial(&self, var) -> Option<Poly>;
    pub fn taylor(&self, var, at: f64, order) -> Poly;
    pub fn variables(&self) -> Vec<String>;
    pub fn depth(&self) / node_count;
    pub fn compile(&self) -> CompiledExpr;               // flatten to stack ops for fast repeated eval
    pub fn integrate_simple(&self, var) -> Option<Expr>; // table + linearity + power rule + u=ax+b; None if not elementary by these rules
    pub fn limit_numeric(&self, var, at, side) -> Option<f64>;
    pub fn equivalent_numeric(&self, o: &Expr, trials, rng) -> bool;
}
pub struct CompiledExpr { ops: Vec<Op>, ... }  impl { eval(&self, vals: &[f64]) -> f64 }
pub fn solve_univariate_numeric(e: &Expr, var, bracket) -> Vec<f64>;
pub fn critical_points(e: &Expr, var, range, n) -> Vec<(f64, f64)>;
pub fn hessian(e: &Expr, vars) -> Vec<Vec<Expr>>;
```
Property: diff of parsed "x^3 * sin(x)" evals equal to Part1 Dual
derivative at 100 random points; simplify(diff(sin^2 + cos^2)) == 0;
Sturm count on Wilkinson(10) over (0.5, 10.5) is 10; Chebyshev fit of
exp on [-1,1] degree 10 max error < 1e-9; Pell for d=61 gives the famous
(1766319049, 226153980); parse(to_string(e)) evaluates identically.

## Phase B: discrete mathematics

### 4. discrete/primes.rs and number_theory.rs
```rust
// primes.rs
pub fn sieve_eratosthenes(n) -> Vec<usize>;
pub fn sieve_segmented(lo, hi) -> Vec<u64>;
pub fn sieve_linear(n) -> (Vec<usize>, Vec<usize>);   // primes + smallest prime factor
pub fn is_prime_u64(n) -> bool;                        // deterministic Miller-Rabin bases
pub fn is_prime_bigint(n: &BigInt, rounds, rng) -> bool;   // Miller-Rabin + Lucas (BPSW)
pub fn next_prime(n) / prev_prime -> u64;
pub fn random_prime(bits, rng) -> BigInt;
pub fn pollard_rho(n: u64) -> Option<u64>;  pollard_rho_bigint(n, rng);  pollard_p_minus_1(n, bound);
pub fn factorize(n: u64) -> Vec<(u64, u32)>;  factorize_bigint(n, rng) -> Vec<(BigInt, u32)>;
pub fn trial_division(n, limit) -> (Vec<(u64, u32)>, u64);
pub fn fermat_factor(n) -> Option<(u64, u64)>;
pub fn prime_count_meissel(n) -> u64;   // pi(n) without full sieve
pub fn prime_count_li_approx(n: f64) -> f64;  riemann_r(n: f64) -> f64;
pub fn nth_prime(n) -> u64;
pub fn prime_gaps(n) -> Vec<u64>;  twin_primes(n) -> Vec<(u64, u64)>;
pub fn goldbach_partitions(n) -> Vec<(u64, u64)>;
pub fn primes_in_arithmetic_progression(a, d, count) -> Vec<u64>;
pub fn mersenne_lucas_lehmer(p: u32) -> bool;
pub fn wilson_check(p) -> bool;
// number_theory.rs
pub fn gcd_u64 / lcm_u64 / extended_gcd_i64 -> (i64, i64, i64);
pub fn mod_pow_u64(b, e, m) -> u64;  mod_inverse_u64(a, m) -> Option<u64>;
pub fn crt(residues: &[(u64, u64)]) -> Option<(u64, u64)>;   // (remainder, modulus) pairs, general moduli
pub fn euler_phi(n) -> u64;  phi_sieve(n) -> Vec<u64>;
pub fn mobius(n) -> i8;  mobius_sieve(n) -> Vec<i8>;
pub fn divisors(n) -> Vec<u64>;  divisor_count(n);  divisor_sum(n);  sigma_k(n, k);
pub fn is_perfect / is_abundant / is_deficient / amicable_pairs(limit);
pub fn multiplicative_order(a, n) -> Option<u64>;
pub fn primitive_root(p) -> Option<u64>;  all_primitive_roots(p);
pub fn discrete_log_bsgs(base, target, modulus) -> Option<u64>;   // baby-step giant-step
pub fn discrete_log_pohlig_hellman(base, target, p, factorization) -> Option<u64>;
pub fn legendre_symbol(a, p) -> i8;  jacobi_symbol(a, n) -> i8;
pub fn tonelli_shanks(a, p) -> Option<u64>;   // sqrt mod p
pub fn quadratic_residues(p) -> Vec<u64>;
pub fn carmichael_lambda(n) -> u64;  is_carmichael(n) -> bool;
pub fn digit_sum(n, base);  digital_root;  is_palindrome(n, base);  reverse_digits;
pub fn happy_number(n) -> bool;  collatz_trajectory(n) -> Vec<u64>;  collatz_stopping_time(n);
pub fn sum_of_two_squares(n) -> Option<(u64, u64)>;  sum_of_four_squares(n) -> (u64, u64, u64, u64);
pub fn pythagorean_triples_primitive(limit) -> Vec<(u64, u64, u64)>;   // tree generation
pub fn gaussian_integer_factor(re: i64, im: i64) -> Vec<(i64, i64)>;
pub fn frobenius_number(coins: &[u64]) -> Option<u64>;   // Chicken McNugget, n=2 exact, small n by DP
pub fn egyptian_fractions_greedy(r: &Rational) -> Vec<BigInt>;
pub fn zeckendorf(n) -> Vec<u64>;   // Fibonacci representation
pub fn lucas_sequence_u(p, q, n, m) -> u64;   // mod m
pub fn quadratic_diophantine_solve(a, b, c) -> Vec<(i64, i64)>;   // small cases
pub fn linear_diophantine(a, b, c) -> Option<(i64, i64, i64, i64)>;   // particular + homogeneous
pub fn stern_brocot_nth(n) -> Rational;
pub fn farey_next(a: &Rational, n) -> Rational;
pub fn dirichlet_convolution(f: &[i64], g: &[i64]) -> Vec<i64>;
```
Property: BPSW agrees with deterministic on all n < 1e7; factorize product
reconstructs n for 1e4 random u64; CRT result satisfies all congruences;
Tonelli-Shanks square is a mod p; euler_phi sieve matches direct; Meissel
pi(1e9) == 50847534; sum of divisors of perfect number is 2n; primitive
root has full multiplicative order.

### 5. discrete/combinatorics.rs, partitions.rs, sequences.rs
```rust
// combinatorics.rs
pub fn binomial_u64(n, k) -> Option<u64>;   // overflow-checked
pub fn binomial_mod_p(n, k, p) -> u64;      // Lucas theorem
pub fn multinomial(ks: &[u64]) -> BigInt;
pub fn permutations_count(n, k) -> Option<u64>;
pub fn permutations_iter(items: &[T]) -> impl Iterator;   // Heap's algorithm; concrete over Vec<usize>
pub fn permutations_lex_next(p: &mut [usize]) -> bool;
pub fn nth_permutation(n_items, index: &BigInt) -> Vec<usize>;   // factoradic
pub fn permutation_index(p: &[usize]) -> BigInt;
pub fn combinations_iter(n, k) -> impl Iterator<Item = Vec<usize>>;   // revolving door / lex
pub fn combinations_with_replacement_iter(n, k);
pub fn gray_code_iter(n_bits) -> impl Iterator<Item = u64>;
pub fn subsets_iter(n) -> impl Iterator<Item = u64>;
pub fn derangements_count(n) -> BigInt;  is_derangement(p);  random_derangement(n, rng);
pub fn random_permutation(n, rng) -> Vec<usize>;   // Fisher-Yates
pub fn permutation_compose / inverse / order / cycle_type(p) -> Vec<usize> / sign(p) -> i8 / from_cycles / to_cycles;
pub fn permutation_matrix(p) -> Matrix;
pub fn stirling_first(n, k) -> BigInt;  stirling_second(n, k) -> BigInt;  bell_number(n) -> BigInt;  bell_triangle(n);
pub fn catalan(n) -> BigInt;  catalan_mod(n, m);
pub fn eulerian_number(n, k) -> BigInt;
pub fn narayana(n, k) -> BigInt;  motzkin(n);  schroeder(n);  delannoy(m, n);
pub fn lah_number(n, k) -> BigInt;
pub fn ballot_number(p, q) -> BigInt;
pub fn dyck_paths_iter(n) -> impl Iterator<Item = Vec<bool>>;
pub fn set_partitions_iter(n) -> impl Iterator<Item = Vec<usize>>;   // restricted growth strings
pub fn compositions_iter(n) -> impl Iterator<Item = Vec<u64>>;
pub fn necklaces_count(n, k) -> BigInt;  bracelets_count(n, k);   // Burnside
pub fn burnside_orbit_count(group_element_fixed_counts: &[BigInt]) -> BigInt;
pub fn polya_enumeration(cycle_index: &PolyQ, colors) -> BigInt;
pub fn cycle_index_cyclic(n) -> ...;  cycle_index_dihedral(n);  cycle_index_symmetric(n);
pub fn inclusion_exclusion(sizes: &dyn Fn(&[usize]) -> BigInt, n) -> BigInt;
pub fn pigeonhole_min_overlap(items, boxes) -> u64;
pub fn ramsey_known(s, t) -> Option<u64>;
pub fn latin_square_random(n, rng) -> Vec<Vec<usize>>;  is_latin_square;
pub fn magic_square(n) -> Option<Vec<Vec<u64>>>;   // odd (Siamese), doubly even, singly even
pub fn de_bruijn_sequence(k, n) -> Vec<usize>;
pub fn perfect_shuffles_order(n_cards, out: bool) -> u64;   // ord of 2 mod n-1; ties to perfect-shuffle-cipher project
pub fn josephus(n, k) -> usize;
pub fn tower_of_hanoi_moves(n, from, to) -> Vec<(u8, u8)>;
pub fn twelvefold_way(n, k, injective: Option<bool>, surjective: Option<bool>, distinguishable_balls: bool, distinguishable_boxes: bool) -> BigInt;
// partitions.rs
pub fn partition_count(n) -> BigInt;   // pentagonal number theorem recurrence
pub fn partition_count_table(n) -> Vec<BigInt>;
pub fn partitions_iter(n) -> impl Iterator<Item = Vec<u64>>;
pub fn partitions_into_k(n, k) -> BigInt;  partitions_distinct(n);  partitions_odd(n);
pub fn partition_conjugate(p: &[u64]) -> Vec<u64>;
pub fn young_diagram(p) -> Vec<Vec<bool>>;  hook_lengths(p) -> Vec<Vec<u64>>;  standard_tableaux_count(p) -> BigInt (hook length formula);
pub fn rsk_correspondence(perm: &[usize]) -> (Vec<Vec<usize>>, Vec<Vec<usize>>);
pub fn durfee_square(p) -> u64;
pub fn hardy_ramanujan_estimate(n) -> f64;
pub fn goldbach_conjecture_verify(up_to) -> bool;
// sequences.rs (generating functions numerically + integer sequences)
pub fn ogf_coefficients(f: &dyn Fn(Complex) -> Complex, n, radius) -> Vec<f64>;   // Cauchy integral via FFT
pub fn egf_to_ogf(coeffs) -> Vec<f64>;
pub fn linear_recurrence(init: &[i64], coeffs: &[i64], n) -> BigInt;
pub fn linear_recurrence_mod(init, coeffs, n: u64, m) -> u64;   // Kitamasa / matrix power
pub fn find_linear_recurrence(seq: &[Rational]) -> Option<Vec<Rational>>;   // Berlekamp-Massey over Q
pub fn berlekamp_massey_gf2(seq: &[bool]) -> Vec<bool>;
pub fn fibonacci_mod(n: u64, m) -> u64;  pisano_period(m) -> u64;
pub fn lucas(n) -> BigInt;  tribonacci(n);  pell_number(n);  jacobsthal(n);
pub fn look_and_say(seed: &str, iterations) -> String;  conway_constant_estimate(iters) -> f64;
pub fn thue_morse(n) -> bool;  thue_morse_sequence(n) -> Vec<bool>;
pub fn kolakoski(n) -> Vec<u8>;
pub fn recaman(n) -> Vec<i64>;
pub fn ulam_sequence(a, b, n) -> Vec<u64>;
pub fn aliquot_sequence(n, max_steps) -> Vec<u64>;
pub fn ackermann_small(m, n) -> Option<BigInt>;
pub fn sequence_identify(terms: &[i64]) -> Vec<String>;   // test against ~50 known families + linear recurrence detection
```
Property: partition_count(100) == 190569292; hook length formula for (2,2)
gives 2; RSK shapes of P and Q equal; Berlekamp-Massey recovers Fibonacci
recurrence from 8 terms; Catalan via formula matches Dyck path iterator
count for n <= 12; Burnside necklace count matches brute force n <= 8;
perfect_shuffles_order(52, true) == 8; Pisano period of 10 is 60;
twelvefold entries match Stirling/binomial/partition identities.

### 6. graph/ (core, paths, flow, matching, spectral, coloring, layout)
```rust
// core.rs
pub struct Graph { pub n: usize, pub adj: Vec<Vec<(usize, f64)>>, pub directed: bool }
impl Graph { new(n, directed); add_edge(u, v, w); from_edges(n, edges, directed); from_adjacency_matrix(m); to_adjacency_matrix() -> Matrix; degree(v); in_degree; out_degree; edges() -> Vec<(usize, usize, f64)>; reverse; subgraph(vs); complement; is_connected; connected_components; strongly_connected_components (Tarjan); condensation; is_bipartite() -> Option<Vec<bool>>; is_tree; is_dag; topological_sort() -> Option<Vec<usize>>; bfs(s) -> Vec<Option<usize>>; dfs(s); bridges(); articulation_points(); eulerian_circuit() -> Option<Vec<usize>> (Hierholzer); eulerian_path; hamiltonian_path_small() -> Option<Vec<usize>> (DP bitmask n <= 20); girth(); diameter(); radius(); center(); density(); clustering_coefficient(v); average_clustering; transitivity; degree_distribution; assortativity; k_core(k) -> Vec<usize>; core_numbers(); }
pub fn complete_graph(n) / cycle_graph / path_graph / star / wheel / grid_2d(w, h) / hypercube_graph(d) / petersen() / complete_bipartite(m, n) -> Graph;
pub fn erdos_renyi(n, p, rng) / barabasi_albert(n, m, rng) / watts_strogatz(n, k, beta, rng) / random_regular(n, d, rng) / random_geometric(n, radius, rng) / stochastic_block_model(sizes, p_matrix, rng) -> Graph;
pub fn graph_from_mesh(mesh: &Mesh) -> Graph;
pub fn line_graph(g) -> Graph;  cartesian_product(g, h);  tensor_product(g, h);
pub fn is_isomorphic_small(g, h) -> bool;   // n <= 10 brute force with invariant pruning
pub fn canonical_form_small(g) -> Vec<u64>;
pub fn graph6_encode(g) -> String;  graph6_decode(s) -> Graph;
// paths.rs
pub fn dijkstra(g, s) -> (Vec<f64>, Vec<Option<usize>>);
pub fn dijkstra_target(g, s, t) -> Option<(f64, Vec<usize>)>;
pub fn bellman_ford(g, s) -> Result<(Vec<f64>, Vec<Option<usize>>), NegativeCycle>;
pub fn floyd_warshall(g) -> Matrix;
pub fn johnson(g) -> Result<Matrix, NegativeCycle>;
pub fn a_star(g, s, t, h: &dyn Fn(usize) -> f64) -> Option<(f64, Vec<usize>)>;
pub fn bidirectional_dijkstra(g, s, t) -> Option<(f64, Vec<usize>)>;
pub fn k_shortest_paths_yen(g, s, t, k) -> Vec<(f64, Vec<usize>)>;
pub fn widest_path(g, s, t) -> Option<(f64, Vec<usize>)>;
pub fn minimax_path(g, s, t) -> Option<(f64, Vec<usize>)>;
pub fn dag_shortest / dag_longest(g, s) -> Vec<f64>;
pub fn count_paths_dag(g, s, t) -> BigInt;
pub fn transitive_closure(g) -> Vec<Vec<bool>>;
pub fn minimum_spanning_tree_kruskal(g) -> (f64, Vec<(usize, usize)>);  prim(g);  boruvka(g);
pub fn second_best_mst(g) -> Option<(f64, Vec<(usize, usize)>)>;
pub fn steiner_tree_small(g, terminals) -> (f64, Vec<(usize, usize)>);   // Dreyfus-Wagner, |T| <= 12
pub fn traveling_salesman_exact(dist: &Matrix) -> (f64, Vec<usize>);      // Held-Karp n <= 20
pub fn tsp_nearest_neighbor(dist) -> (f64, Vec<usize>);  tsp_2opt(dist, tour) -> (f64, Vec<usize>);  tsp_or_opt;  tsp_christofides(dist) -> (f64, Vec<usize>) (needs matching);
pub fn chinese_postman(g) -> Option<(f64, Vec<usize>)>;
// flow.rs
pub fn max_flow_dinic(g, s, t) -> (f64, Vec<Vec<f64>>);
pub fn max_flow_push_relabel(g, s, t) -> f64;
pub fn min_cut(g, s, t) -> (f64, Vec<bool>);
pub fn global_min_cut_stoer_wagner(g) -> (f64, Vec<usize>);
pub fn min_cost_max_flow(g_with_costs, s, t) -> (f64, f64);   // SPFA/Johnson potentials
pub fn circulation_with_demands(...) -> Option<Vec<Vec<f64>>>;
pub fn max_bipartite_matching_via_flow(g) -> Vec<Option<usize>>;
pub fn vertex_disjoint_paths(g, s, t) -> usize;  edge_disjoint_paths;
pub fn gomory_hu_tree(g) -> Graph;
pub fn closure_problem(g, weights) -> (f64, Vec<bool>);   // max-weight closure via min cut
pub fn project_selection(...) -> f64;
// matching.rs
pub fn hopcroft_karp(left_n, right_n, edges) -> Vec<Option<usize>>;
pub fn hungarian(cost: &Matrix) -> (f64, Vec<usize>);   // assignment, O(n^3)
pub fn auction_assignment(cost, eps) -> (f64, Vec<usize>);
pub fn blossom_max_matching(g) -> Vec<Option<usize>>;   // general graphs, Edmonds
pub fn stable_marriage(prefs_a, prefs_b) -> Vec<usize>;   // Gale-Shapley
pub fn stable_roommates(prefs) -> Option<Vec<usize>>;
pub fn konig_vertex_cover(g_bipartite, matching) -> Vec<usize>;
pub fn hall_condition_check(g_bipartite) -> Result<(), Vec<usize>>;   // violating set
pub fn maximum_weight_bipartite(weights) -> (f64, Vec<Option<usize>>);
// spectral.rs (needs Part1 eigen)
pub fn laplacian_matrix(g) -> Matrix;  normalized_laplacian;  adjacency_spectrum(g) -> Vec<f64>;  laplacian_spectrum;
pub fn algebraic_connectivity(g) -> f64;  fiedler_vector(g) -> Vec<f64>;
pub fn spectral_bisection(g) -> Vec<bool>;  spectral_clustering(g, k) -> Vec<usize>;
pub fn number_spanning_trees(g) -> f64;   // Kirchhoff, det of reduced Laplacian
pub fn number_spanning_trees_exact(g) -> BigInt;   // Bareiss on integer Laplacian
pub fn pagerank(g, damping, tol) -> Vec<f64>;
pub fn hits(g, tol) -> (Vec<f64>, Vec<f64>);
pub fn eigenvector_centrality(g, tol) -> Vec<f64>;  katz_centrality(g, alpha);
pub fn betweenness_centrality(g) -> Vec<f64>;   // Brandes
pub fn closeness_centrality(g) -> Vec<f64>;  harmonic_centrality;
pub fn effective_resistance(g, u, v) -> f64;  resistance_matrix(g) -> Matrix;
pub fn commute_time(g, u, v) -> f64;
pub fn random_walk_stationary(g) -> Vec<f64>;  mixing_time_estimate(g, eps);
pub fn cheeger_bound(g) -> (f64, f64);
pub fn expander_check(g, target_gap) -> bool;
pub fn graph_energy(g) -> f64;  estrada_index(g);
pub fn isospectral_check(g, h, tol) -> bool;
pub fn community_louvain(g, rng) -> Vec<usize>;  modularity(g, communities) -> f64;
pub fn label_propagation(g, rng) -> Vec<usize>;
// coloring.rs
pub fn greedy_coloring(g, order: Order) -> Vec<usize>;   // natural, largest-first, smallest-last (degeneracy), DSATUR
pub fn chromatic_number_exact_small(g) -> usize;   // n <= 20, branch and bound
pub fn chromatic_polynomial_small(g) -> PolyQ;      // deletion-contraction with memo, n <= 12
pub fn edge_coloring_vizing(g) -> Vec<usize>;
pub fn is_k_colorable_sat_style(g, k, time_limit) -> Option<Vec<usize>>;
pub fn welsh_powell(g) -> Vec<usize>;
pub fn interval_graph_coloring(intervals) -> Vec<usize>;
pub fn map_coloring_backtrack(adjacency, k) -> Option<Vec<usize>>;
pub fn independent_set_greedy(g) -> Vec<usize>;  max_independent_set_small(g) (complement clique);
pub fn max_clique_bron_kerbosch(g) -> Vec<usize>;  all_maximal_cliques(g) -> Vec<Vec<usize>>;
pub fn vertex_cover_2approx(g) -> Vec<usize>;  vertex_cover_exact_small(g);
pub fn dominating_set_greedy(g) -> Vec<usize>;
pub fn feedback_arc_set_greedy(g) -> Vec<(usize, usize)>;
// layout.rs
pub fn fruchterman_reingold(g, iters, rng) -> Vec<Vec2>;
pub fn kamada_kawai(g, iters) -> Vec<Vec2>;
pub fn spectral_layout(g) -> Vec<Vec2>;
pub fn circular_layout(n) -> Vec<Vec2>;  shell_layout(g, shells);
pub fn tree_layout_reingold_tilford(g, root) -> Vec<Vec2>;
pub fn sugiyama_layered(dag) -> Vec<Vec2>;
pub fn planarity_test(g) -> bool;   // Euler bound + Kuratowski search small, or LR algorithm
pub fn planar_embedding_small(g) -> Option<Vec<Vec<usize>>>;
pub fn crossing_number_estimate(g, layout) -> usize;
pub fn stress_majorization(g, dim, iters) -> Vec<VecN>;
```
Property: Dijkstra equals Bellman-Ford equals Floyd-Warshall on random
graphs; max-flow equals min-cut; Kirchhoff on complete graph gives
Cayley n^(n-2); Hungarian matches brute force n <= 8; blossom matching
on Petersen graph has size 5; chromatic polynomial of C5 at 3 gives 30;
Held-Karp matches brute force n <= 10; Christofides tour <= 1.5 x
Held-Karp on metric instances; PageRank sums to 1 and matches power
iteration; betweenness of star center = (n-1)(n-2)/2; Louvain modularity
positive on planted partition and recovers blocks with high ARI.

### 7. codes/ (checksum, block, reed_solomon, convolutional, compression, crypto_math)
```rust
// checksum.rs
pub fn parity(bits) -> bool;  checksum_fletcher16(data) -> u16;  fletcher32;  adler32;
pub fn crc(data: &[u8], poly: u64, width, init, xor_out, reflect) -> u64;   // parametric
pub fn crc32_ieee(data) -> u32;  crc16_ccitt;  crc8;  crc_table(poly) -> [u32; 256];
pub fn luhn_check(digits) -> bool;  luhn_generate;
pub fn isbn10_check / isbn13_check / verhoeff_check / damm_check(digits) -> bool;
pub fn hamming_distance_bits(a: u64, b: u64) -> u32;  hamming_distance_bytes(a, b) -> Option<u32>;
// block.rs (GF(2) linear codes)
pub struct Gf2Matrix { pub rows, cols: usize, pub data: Vec<u64> }   // bit-packed
impl Gf2Matrix { rank; rref; solve; mul; transpose; identity; kernel_basis }
pub struct LinearCode { pub g: Gf2Matrix, pub h: Gf2Matrix, pub n, k, d: usize }
impl LinearCode { hamming(r) (2^r-1, 2^r-1-r, 3); extended_hamming(r); repetition(n); parity_check(n); golay23(); golay24(); reed_muller(r, m); from_generator(g); encode(msg_bits) -> Vec<bool>; decode_syndrome(recv) -> (Vec<bool>, usize /*corrected*/); minimum_distance_small(); weight_enumerator() -> Vec<BigInt>; dual(); is_self_dual(); syndrome(recv); standard_array_decode_small(); }
pub fn hamming_74_encode(nibble: u8) -> u8;  hamming_74_decode(byte) -> (u8, bool);
pub fn singleton_bound(n, k) -> usize;  hamming_bound(n, d) -> f64;  gilbert_varshamov(n, d) -> f64;  plotkin_bound;
pub fn ldpc_regular(n, wc, wr, rng) -> Gf2Matrix;
pub fn ldpc_decode_bp(h: &Gf2Matrix, llr: &[f64], iters) -> (Vec<bool>, bool);   // belief propagation
pub fn ldpc_decode_bitflip(h, recv, iters) -> Vec<bool>;
// reed_solomon.rs
pub struct Gf256 { pub log: [u8; 256], pub exp: [u8; 512] }   // GF(2^8), poly 0x11d
impl Gf256 { new(prim_poly); add; mul; div; pow; inv }
pub struct GfP { pub p: u64 }   // prime field
pub struct Gf2m { pub m: u32, pub prim: u64 }   // GF(2^m) general
impl Gf2m { add; mul; inv; pow; trace; all_elements; minimal_polynomial(alpha_power) }
pub struct ReedSolomon { pub n: usize, pub k: usize, gf: Gf256, gen_poly: Vec<u8> }
impl ReedSolomon { new(n, k); encode(msg: &[u8]) -> Vec<u8>; decode(recv: &[u8]) -> Result<(Vec<u8>, usize), TooManyErrors>;   // syndromes, Berlekamp-Massey, Chien, Forney
  decode_erasures(recv, erasure_pos) -> Result<Vec<u8>, _>; }
pub fn rs_ccsds() / rs_qr_code(version) / rs_dvd() -> ReedSolomon;
pub struct BchCode { ... }  impl { new(m, t); encode; decode }
pub fn cyclic_code_generator(n, factors_of_x_n_minus_1) -> Vec<Poly over GF2>;
// convolutional.rs
pub struct ConvolutionalCode { pub k: u32, pub polys: Vec<u64> }   // constraint length, generator polys octal
impl ConvolutionalCode { nasa_standard() (k=7, [171, 133]); encode(bits) -> Vec<bool>; viterbi_decode(recv_hard: &[bool]) -> Vec<bool>; viterbi_soft(llr: &[f64]) -> Vec<bool>; free_distance_estimate(); trellis_states(); puncture(pattern); }
pub struct TurboCode { ... two RSC + interleaver }  impl { encode; decode_bcjr(iters) }
pub fn interleaver_block(n, rows) -> Vec<usize>;  interleaver_random(n, rng);  qpp_interleaver(n, f1, f2);
pub fn ber_simulation(code: &dyn Code, snr_db_range, n_bits, rng) -> Vec<(f64, f64)>;
pub fn awgn_channel(bits, snr_db, rng) -> Vec<f64>;  bsc_channel(bits, p, rng) -> Vec<bool>;
pub fn shannon_limit_bpsk(rate) -> f64;   // Eb/N0 in dB
pub fn channel_capacity_awgn(snr) -> f64;  capacity_bsc(p);  capacity_bec(e);   // extends information_theory
// compression.rs
pub fn huffman_build(freqs: &[u64]) -> Vec<(u64, u8)>;   // (codeword, length)
pub fn huffman_encode(data: &[u8]) -> (Vec<u8>, Vec<(u64, u8)>, usize);
pub fn huffman_decode(bits, table, n) -> Vec<u8>;
pub fn canonical_huffman(lengths: &[u8]) -> Vec<u64>;
pub fn shannon_fano(freqs) -> Vec<(u64, u8)>;
pub fn arithmetic_encode(data, model: &[u64]) -> Vec<u8>;  arithmetic_decode;
pub fn lz77_compress(data, window, lookahead) -> Vec<Lz77Token>;  lz77_decompress;
pub fn lzw_compress(data) -> Vec<u16>;  lzw_decompress;
pub fn rle_compress(data) -> Vec<u8>;  rle_decompress;
pub fn bwt(data) -> (Vec<u8>, usize);  ibwt(data, idx) -> Vec<u8>;   // suffix-array based
pub fn mtf_encode(data) -> Vec<u8>;  mtf_decode;
pub fn delta_encode / delta_decode;
pub fn suffix_array(data) -> Vec<usize>;   // SA-IS or DC3
pub fn lcp_array(data, sa) -> Vec<usize>;
pub fn longest_repeated_substring(data) -> (usize, usize);
pub fn entropy_bytes(data) -> f64;  compression_bound(data) -> f64;
pub fn kolmogorov_estimate_by_compressors(data) -> f64;
pub fn normalized_compression_distance(a, b) -> f64;
// crypto_math.rs (educational, not constant-time; document loudly)
pub fn rsa_keygen(bits, rng) -> (BigInt, BigInt, BigInt);   // n, e, d
pub fn rsa_encrypt(m, e, n) -> BigInt;  rsa_decrypt(c, d, n);  rsa_crt_decrypt(c, d, p, q);
pub fn diffie_hellman_demo(p, g, rng) -> ((BigInt, BigInt), (BigInt, BigInt), BigInt);
pub struct EcCurve { pub a: BigInt, pub b: BigInt, pub p: BigInt }   // short Weierstrass over F_p
pub enum EcPoint { Infinity, Affine(BigInt, BigInt) }
impl EcCurve { is_on_curve(pt); add(p, q); double(p); scalar_mul(k, p); order_naive_small(); secp256k1(); p256(); random_point(rng); }
pub fn ecdh_demo(curve, g, rng) -> ...;
pub fn ec_count_points_small(curve) -> u64;  hasse_bound_check(count, p) -> bool;
pub fn shamir_split(secret: &BigInt, k, n, prime, rng) -> Vec<(u64, BigInt)>;
pub fn shamir_reconstruct(shares, prime) -> BigInt;   // Lagrange over F_p
pub fn one_time_pad(data, key) -> Vec<u8>;
pub fn lfsr(taps: u64, state: u64, n) -> Vec<bool>;  lfsr_period(taps, width);  berlekamp_massey_attack(stream) -> (u64, u64);
pub fn hash_avalanche_test(h: &dyn Fn(&[u8]) -> u64, trials, rng) -> f64;
pub fn birthday_bound(n_bits) -> f64;
pub fn frequency_analysis(text) -> [f64; 26];  index_of_coincidence(text);  kasiski_examination(text) -> Vec<usize>;  vigenere_break(text, max_key) -> String;  caesar_break(text) -> u8;
pub fn perfect_shuffle_permutation(n, out: bool) -> Vec<usize>;   // ties to shuffle cipher project
pub fn permutation_cipher_period(perm) -> u64;
```
Property: CRC32 of "123456789" is 0xCBF43926; Hamming(7,4) corrects any
single bit error exhaustively; Golay23 corrects all 3-bit errors on
random trials; RS(255,223) corrects 16 random byte errors; Viterbi at
high SNR recovers exactly; LDPC BP beats bit-flip BER at 2 dB; Huffman
average length within 1 bit of entropy; bwt+mtf+rle roundtrip exact;
suffix array sorted; RSA encrypt/decrypt roundtrip 100 trials; EC scalar
mul matches repeated addition; Hasse bound holds on random small curves;
Shamir reconstructs with any k shares, fails with k-1; Vigenere break
recovers key on 500-char English sample.

## Phase C: stochastic processes and time series

### 8. stochastic/markov.rs and hmm.rs
```rust
pub struct MarkovChain { pub p: Matrix }   // row-stochastic
impl MarkovChain {
    pub fn new(p) -> Result<Self, GeomError>;  from_counts(transitions);  from_sequence(states: &[usize], n_states);
    pub fn stationary(&self) -> Vec<f64>;      // eigenvector or linear solve
    pub fn step_dist(&self, dist: &[f64]) -> Vec<f64>;  n_step(&self, n) -> Matrix;
    pub fn simulate(&self, start, steps, rng) -> Vec<usize>;
    pub fn is_irreducible / is_aperiodic / period(state) / classify_states() -> Vec<StateClass>;
    pub fn absorbing_probabilities(&self) -> Matrix;   // fundamental matrix N = (I-Q)^-1
    pub fn expected_steps_to_absorption(&self) -> Vec<f64>;
    pub fn hitting_time(&self, from, target_set) -> f64;  hitting_probability;
    pub fn return_time(&self, state) -> f64;
    pub fn mixing_time(&self, eps) -> usize;  total_variation_distance(a, b);
    pub fn spectral_gap(&self) -> f64;
    pub fn reversible_check(&self, pi, tol) -> bool;   // detailed balance
    pub fn mfpt_matrix(&self) -> Matrix;
    pub fn entropy_rate(&self) -> f64;
    pub fn coupling_from_the_past_small(&self, rng) -> usize;   // exact sampling
    pub fn pagerank_chain(g: &Graph, damping) -> Self;
}
pub struct Mcmc;
impl Mcmc {
    pub fn metropolis_hastings(log_target: &dyn Fn(&[f64]) -> f64, x0, proposal_std, n, burn, rng) -> Vec<Vec<f64>>;   // generalizes existing metropolis_sample
    pub fn adaptive_metropolis(...) -> Vec<Vec<f64>>;
    pub fn gibbs(conditionals: &[&dyn Fn(&[f64], &mut Rng) -> f64], x0, n, burn, rng) -> Vec<Vec<f64>>;
    pub fn hamiltonian_mc(log_target, grad, x0, step, n_leapfrog, n, rng) -> Vec<Vec<f64>>;
    pub fn nuts_lite(...) -> Vec<Vec<f64>>;
    pub fn slice_sampler(log_target_1d, x0, w, n, rng) -> Vec<f64>;
    pub fn parallel_tempering(log_target, temps, ...) -> Vec<Vec<f64>>;
    pub fn effective_sample_size(chain: &[f64]) -> f64;
    pub fn gelman_rubin(chains: &[Vec<f64>]) -> f64;
    pub fn autocorrelation_time(chain) -> f64;
    pub fn simulated_annealing(energy: &dyn Fn(&[f64]) -> f64, x0, schedule: &dyn Fn(usize) -> f64, n, rng) -> (Vec<f64>, f64);
}
// hmm.rs
pub struct Hmm { pub a: Matrix, pub b: Matrix, pub pi: Vec<f64> }   // discrete emissions
impl Hmm {
    pub fn forward(&self, obs: &[usize]) -> (f64, Matrix);      // log-likelihood, scaled alphas
    pub fn backward(&self, obs) -> Matrix;
    pub fn viterbi(&self, obs) -> (f64, Vec<usize>);
    pub fn posterior_decode(&self, obs) -> Vec<usize>;
    pub fn baum_welch(&mut self, sequences: &[Vec<usize>], iters, tol) -> f64;
    pub fn simulate(&self, n, rng) -> (Vec<usize>, Vec<usize>);
    pub fn log_likelihood(&self, obs) -> f64;
    pub fn random_init(n_states, n_symbols, rng) -> Self;
}
pub struct GaussianHmm { pub a: Matrix, pub means: Vec<f64>, pub vars: Vec<f64>, pub pi: Vec<f64> }
impl GaussianHmm { forward; viterbi; baum_welch; simulate }
pub struct KalmanSmoother;   // RTS smoother, extends Part1 KalmanFilter
pub fn rts_smooth(kf: &KalmanFilter, xs, ps) -> (Vec<Vec<f64>>, Vec<Matrix>);
pub fn em_kalman(observations, ...) -> KalmanFilter;   // learn Q, R
pub struct ParticleFilter { pub particles: Vec<Vec<f64>>, pub weights: Vec<f64> }
impl ParticleFilter { new(n, init, rng); predict(dynamics, noise, rng); update(likelihood); resample_systematic(rng); estimate() -> Vec<f64>; effective_n(); }
```
Property: stationary satisfies pi P = pi; absorbing gambler's ruin matches
closed form; MH on Gaussian target has correct mean/var within MC error;
HMC and MH agree on target within ESS-scaled error; ESS <= n; Baum-Welch
log-likelihood monotone nondecreasing; Viterbi on noiseless chain recovers
states; RTS smoother variance <= filter variance; particle filter on
linear-Gaussian matches Kalman within MC error.

### 9. stochastic/sde.rs and point_process.rs
```rust
// sde.rs  dX = mu(t, X) dt + sigma(t, X) dW
pub fn brownian_motion(n, dt, rng) -> Vec<f64>;  brownian_bridge(n, dt, x0, x1, rng);  brownian_2d/3d;
pub fn geometric_brownian(x0, mu, sigma, n, dt, rng) -> Vec<f64>;  gbm_exact(x0, mu, sigma, t, z) -> f64;
pub fn ornstein_uhlenbeck(x0, theta, mu, sigma, n, dt, rng) -> Vec<f64>;  ou_exact_step(x, theta, mu, sigma, dt, z);
pub fn euler_maruyama(mu: &dyn Fn(f64, f64) -> f64, sigma: &dyn Fn(f64, f64) -> f64, x0, t_end, n, rng) -> Vec<f64>;
pub fn euler_maruyama_nd(mu: &dyn Fn(f64, &[f64]) -> Vec<f64>, sigma: &dyn Fn(f64, &[f64]) -> Matrix, x0, t_end, n, rng) -> Vec<Vec<f64>>;
pub fn milstein(mu, sigma, dsigma_dx, x0, t_end, n, rng) -> Vec<f64>;
pub fn stochastic_heun(mu, sigma, ...) -> Vec<f64>;   // Stratonovich
pub fn srk_order_1_5(...) -> Vec<f64>;
pub fn strong_convergence_order(scheme, exact_paths, dts) -> f64;  weak_convergence_order;
pub fn cir_process(x0, kappa, theta, sigma, n, dt, rng) -> Vec<f64>;   // full truncation, stays >= 0
pub fn heston_paths(s0, v0, params, n, dt, rng) -> (Vec<f64>, Vec<f64>);
pub fn jump_diffusion_merton(x0, mu, sigma, lambda, jump_mu, jump_sigma, n, dt, rng) -> Vec<f64>;
pub fn levy_stable_sample(alpha, beta, rng) -> f64;   // Chambers-Mallows-Stuck
pub fn fractional_brownian(h: f64, n, rng) -> Vec<f64>;   // Davies-Harte via fft
pub fn hurst_exponent_rs(x) -> f64;  hurst_dfa(x) -> f64;
pub fn first_passage_time_sim(process, barrier, n_paths, ...) -> Vec<f64>;
pub fn first_passage_bm_exact(barrier, drift, t) -> f64;   // inverse Gaussian density
pub fn feynman_kac_check(...) -> f64;   // SDE MC vs PDE solve of same problem
pub fn ito_isometry_check(sigma, t, n_paths, rng) -> (f64, f64);
pub fn langevin_underdamped(x0, v0, gamma, temp, mass, force, n, dt, rng) -> Vec<(f64, f64)>;   // BAOAB
pub fn fokker_planck_1d(p0: &[f64], mu, sigma, dx, dt, steps) -> Vec<f64>;   // Chang-Cooper
pub fn stationary_density_1d(mu, sigma, x_range, n) -> Vec<f64>;   // exp(-2 int mu/sigma^2)
pub fn kramers_escape_rate(barrier_height, temp, omega_well, omega_barrier, gamma) -> f64;
pub fn stochastic_resonance_sim(...) -> Vec<f64>;
// point_process.rs
pub fn poisson_process(rate, t_end, rng) -> Vec<f64>;
pub fn poisson_inhomogeneous(rate_fn, rate_max, t_end, rng) -> Vec<f64>;   // thinning
pub fn poisson_2d(rate, region: &Rect, rng) -> Vec<Vec2>;  poisson_3d;
pub fn compound_poisson(rate, jump_dist: &dyn Fn(&mut Rng) -> f64, t_end, rng) -> Vec<(f64, f64)>;
pub fn hawkes_process(mu, alpha, beta, t_end, rng) -> Vec<f64>;   // exponential kernel, Ogata thinning
pub fn hawkes_intensity(events, mu, alpha, beta, t) -> f64;
pub fn hawkes_fit_mle(events, t_end) -> (f64, f64, f64);
pub fn hawkes_branching_ratio(alpha, beta) -> f64;
pub fn renewal_process(interarrival: &dyn Fn(&mut Rng) -> f64, t_end, rng) -> Vec<f64>;
pub fn renewal_function_estimate(...) -> f64;
pub fn cox_process(...) -> Vec<f64>;
pub fn matern_cluster_process(parent_rate, cluster_radius, daughter_mean, region, rng) -> Vec<Vec2>;
pub fn thomas_process(...) -> Vec<Vec2>;
pub fn ripley_k(points, region, r_values) -> Vec<f64>;  l_function;  pair_correlation(points, region, r, dr);
pub fn nearest_neighbor_index(points, region) -> f64;   // Clark-Evans
pub fn quadrat_test(points, region, nx, ny) -> TestResult;
pub fn ks_test_exponential_interarrivals(events) -> TestResult;
pub fn branching_process_gw(offspring_pmf, generations, rng) -> Vec<u64>;   // Galton-Watson
pub fn extinction_probability(offspring_pgf_coeffs) -> f64;   // smallest fixed point
pub fn yule_process(birth_rate, t_end, rng) -> Vec<f64>;
pub fn birth_death_simulate(birth, death, n0, t_end, rng) -> Vec<(f64, u64)>;
```
Property: Euler-Maruyama strong order 0.5 and Milstein 1.0 measured on GBM
vs exact; OU stationary variance sigma^2/(2 theta); CIR never negative;
fBm with H=0.5 has Hurst estimate 0.5 +- 0.05; Ito isometry holds within
MC error; Poisson counts are Poisson-distributed (chi-squared p > 0.001);
Hawkes MLE recovers parameters within 10% on 1e4 events; Ripley K of CSR
matches pi r^2; Galton-Watson extinction prob matches PGF fixed point;
Fokker-Planck stationary matches analytic density.

### 10. stochastic/queueing.rs, timeseries.rs, rmt.rs, extreme.rs
```rust
// queueing.rs
pub fn mm1(lambda, mu) -> QueueMetrics { rho, l, lq, w, wq, p0, pn(n) };
pub fn mmc(lambda, mu, c) -> QueueMetrics;   // Erlang C
pub fn mm1k(lambda, mu, k) -> QueueMetrics;  mmck;  mm_inf;
pub fn erlang_b(offered_load, c) -> f64;  erlang_c(load, c);  erlang_b_inverse_capacity(load, blocking_target);
pub fn mg1_pollaczek_khinchine(lambda, service_mean, service_var) -> QueueMetrics;
pub fn gg1_kingman_approx(lambda, mu, ca2, cs2) -> f64;
pub fn littles_law_check(l, lambda, w) -> f64;
pub fn jackson_network(routing: &Matrix, external: &[f64], service: &[f64], servers: &[usize]) -> Vec<QueueMetrics>;
pub fn queue_simulate(arrival: &dyn Fn(&mut Rng) -> f64, service: &dyn Fn(&mut Rng) -> f64, c, t_end, rng) -> QueueSimResult;   // event-driven
pub fn priority_queue_simulate(...) -> ...;
pub fn queue_transient_mm1(lambda, mu, n0, t) -> Vec<f64>;   // via uniformization
pub fn uniformization(q_matrix: &Matrix, p0, t, eps) -> Vec<f64>;   // CTMC transient
pub struct Ctmc { pub q: Matrix }
impl Ctmc { stationary(); simulate(start, t_end, rng) -> Vec<(f64, usize)>; embedded_chain() -> MarkovChain; mean_holding_times(); first_passage(from, to) }
// timeseries.rs
pub fn acf(x, max_lag) -> Vec<f64>;  pacf(x, max_lag) (Durbin-Levinson);
pub fn ljung_box(x, lags) -> TestResult;  adf_test(x, lags) -> TestResult;  kpss_test(x) -> TestResult;
pub fn difference(x, d) -> Vec<f64>;  seasonal_difference(x, s);  undifference(diffed, initial);
pub struct Arma { pub ar: Vec<f64>, pub ma: Vec<f64>, pub sigma2: f64, pub mean: f64 }
impl Arma { fit_css(x, p, q) (conditional sum of squares via Part1 LM); fit_hannan_rissanen(x, p, q); simulate(n, rng); forecast(x, h) -> (Vec<f64>, Vec<f64>) (point + std err); log_likelihood(x); aic; bic; roots_check() -> (bool, bool) (stationary, invertible); impulse_response(n); spectral_density(freqs) }
pub struct Arima { pub d: usize, pub arma: Arma }  impl { fit(x, p, d, q); forecast }
pub fn auto_arima(x, max_p, max_d, max_q) -> Arima;   // AIC grid
pub struct Sarima { ... seasonal (P, D, Q, s) }
pub fn holt_winters(x, alpha, beta, gamma, season_len, multiplicative) -> (Vec<f64>, HwState);
pub fn holt_winters_optimize(x, season_len) -> (f64, f64, f64);
pub fn exponential_smoothing(x, alpha) -> Vec<f64>;  double_exponential;
pub struct Garch11 { pub omega, alpha, beta: f64 }
impl Garch11 { fit(returns) (MLE via Nelder-Mead); simulate(n, rng); conditional_variance(returns) -> Vec<f64>; forecast_variance(h); unconditional_variance(); persistence() }
pub fn ewma_variance(returns, lambda) -> Vec<f64>;
pub fn arch_lm_test(returns, lags) -> TestResult;
pub fn granger_causality(x, y, lags) -> TestResult;
pub fn cross_correlation_lags(x, y, max_lag) -> Vec<f64>;
pub struct Var { pub coeffs: Vec<Matrix>, pub intercept: Vec<f64> }   // vector autoregression
impl Var { fit(data: &[Vec<f64>], p); forecast(h); impulse_response(h); granger_matrix() }
pub fn cointegration_engle_granger(x, y) -> TestResult;
pub fn seasonal_decompose_stl_lite(x, period) -> (Vec<f64>, Vec<f64>, Vec<f64>);   // trend, seasonal, resid
pub fn changepoint_pelt(x, penalty) -> Vec<usize>;
pub fn changepoint_binary_segmentation(x, max_k) -> Vec<usize>;
pub fn cusum(x, target, k) -> (Vec<f64>, Vec<f64>);
pub fn matrix_profile_lite(x, m) -> (Vec<f64>, Vec<usize>);   // STOMP, motif/discord discovery
pub fn sample_entropy(x, m, r) -> f64;  approximate_entropy;  permutation_entropy(x, order, delay);
pub fn surrogate_test_iaaft(x, statistic, n_surrogates, rng) -> f64;
pub fn state_space_local_level(x) -> (Vec<f64>, f64, f64);   // Kalman-based, MLE variances
// rmt.rs
pub fn goe_sample(n, rng) -> Matrix;  gue_sample -> (Matrix, Matrix) (re, im);  ginibre_sample;
pub fn wishart_sample(n, p, rng) -> Matrix;
pub fn wigner_semicircle(x, r) -> f64;
pub fn marchenko_pastur(x, ratio, sigma2) -> f64;  mp_edges(ratio, sigma2) -> (f64, f64);
pub fn eigenvalue_spacing_distribution(eigs) -> Vec<f64>;
pub fn wigner_surmise_goe(s) -> f64;  gue(s);  poisson_spacing(s);
pub fn spectral_rigidity(eigs, l) -> f64;
pub fn tracy_widom_beta1_approx(x) -> f64;
pub fn participation_ratio(vec) -> f64;
pub fn level_spacing_ratio(eigs) -> f64;   // 0.5307 GOE, 0.3863 Poisson
pub fn correlation_matrix_denoise_mp(corr: &Matrix, t_over_n) -> Matrix;   // clip eigenvalues
// extreme.rs
pub fn gev_fit(maxima) -> (f64, f64, f64);   // mu, sigma, xi via MLE
pub fn gev_pdf/cdf/quantile(x, mu, sigma, xi) -> f64;
pub fn gumbel_fit(maxima) -> (f64, f64);
pub fn gpd_fit(exceedances) -> (f64, f64);   // peaks over threshold
pub fn mean_residual_life(x, thresholds) -> Vec<f64>;
pub fn hill_estimator(x, k) -> f64;
pub fn return_level(fit, period) -> f64;  return_period(fit, level);
pub fn block_maxima(x, block) -> Vec<f64>;
pub fn extremal_index(x, threshold) -> f64;
pub fn pickands_dependence(...) -> f64;
pub fn copula_gaussian_sample(corr, n, rng) -> Vec<Vec<f64>>;  copula_t_sample;  copula_clayton;  copula_gumbel;  copula_frank;
pub fn copula_fit_tau(data) -> f64;   // Kendall tau inversion
pub fn kendall_tau(x, y) -> f64;  spearman_rho(x, y);
pub fn tail_dependence_coefficient(data, q) -> (f64, f64);
pub fn empirical_copula(data) -> Vec<Vec<f64>>;
```
Property: M/M/1 simulation matches closed-form L and W within 3%; Erlang B
recursive matches formula; uniformization matches matrix exponential;
ACF of AR(1) is phi^k; ARMA fit recovers coefficients within 0.05 on 1e4
simulated points; GARCH persistence < 1 on fitted real-like data; forecast
intervals contain truth at nominal rate; GOE spacings match Wigner surmise
(KS p > 0.01); Marchenko-Pastur histogram matches density; GEV fit on
Gumbel data gives xi within 0.05 of 0; copula samples have target Kendall
tau within 0.02.

## Phase D: optimization, OR, games

### 11. opt/lp.rs, integer.rs, network.rs
```rust
// lp.rs  minimize c.x s.t. A x <= b, x >= 0 (standard form conversions provided)
pub struct LpProblem { pub c: Vec<f64>, pub a: Matrix, pub b: Vec<f64>, pub constraint_types: Vec<Cmp>, pub bounds: Vec<(f64, f64)>, pub maximize: bool }
pub enum LpResult { Optimal { x: Vec<f64>, objective: f64, duals: Vec<f64>, reduced_costs: Vec<f64> }, Infeasible, Unbounded }
pub fn simplex(p: &LpProblem) -> LpResult;   // revised, two-phase, Bland anti-cycling
pub fn dual_simplex(p, basis) -> LpResult;
pub fn interior_point(p, tol) -> LpResult;   // primal-dual path following
pub fn lp_dual(p) -> LpProblem;
pub fn sensitivity_ranges(p, result) -> (Vec<(f64, f64)>, Vec<(f64, f64)>);   // c ranges, b ranges
pub fn lp_from_str(s) -> Result<LpProblem, GeomError>;   // tiny modeling language
pub fn diet_problem(foods, nutrients, requirements) -> LpProblem;
pub fn transportation_problem(supply, demand, costs) -> LpResult;   // + Vogel initial, MODI
pub fn production_planning(...) -> LpProblem;
pub fn two_player_zero_sum_lp(payoff: &Matrix) -> (Vec<f64>, Vec<f64>, f64);
pub fn chebyshev_center(a, b) -> (Vec<f64>, f64);
pub fn l1_regression_lp(x, y) -> Vec<f64>;  linf_regression_lp;
// integer.rs
pub fn branch_and_bound(p: &LpProblem, integer_vars: &[usize], time_limit) -> Option<(Vec<f64>, f64)>;
pub fn gomory_cuts(p, result, max_cuts) -> LpProblem;
pub fn knapsack_01(values, weights, cap) -> (u64, Vec<bool>);   // DP
pub fn knapsack_unbounded / bounded / multiple;
pub fn knapsack_branch_bound(values, weights, cap) -> (u64, Vec<bool>);
pub fn subset_sum(xs, target) -> Option<Vec<usize>>;  subset_sum_count -> BigInt;
pub fn partition_min_diff(xs) -> (u64, Vec<bool>);
pub fn bin_packing_ffd(sizes, cap) -> Vec<Vec<usize>>;  bin_packing_exact_small;  bin_packing_lower_bound;
pub fn set_cover_greedy(universe_n, sets) -> Vec<usize>;  set_cover_exact_small;
pub fn facility_location_greedy(costs_open, costs_serve) -> (f64, Vec<bool>);
pub fn cutting_stock_column_generation(demand, lengths, stock_len) -> f64;
pub fn coin_change_min(coins, amount) -> Option<Vec<u64>>;  coin_change_count -> BigInt;
pub fn longest_increasing_subsequence(x) -> Vec<usize>;   // n log n
pub fn edit_distance(a, b) -> usize;  edit_distance_ops -> Vec<EditOp>;
pub fn longest_common_subsequence(a, b) -> Vec<u8>;
pub fn matrix_chain_order(dims) -> (u64, String);
pub fn rod_cutting(prices, n) -> (u64, Vec<usize>);
pub fn egg_drop(eggs, floors) -> u64;
pub fn optimal_bst(keys_freq) -> f64;
pub fn viterbi_generic(states, trans_cost, emit_cost, obs) -> Vec<usize>;
pub fn sudoku_solve(grid: &[[u8; 9]; 9]) -> Option<[[u8; 9]; 9]>;   // DLX or bitmask backtracking
pub fn exact_cover_dlx(matrix: &[Vec<bool>]) -> Option<Vec<usize>>;   // dancing links
pub fn n_queens(n) -> Vec<Vec<usize>>;  n_queens_count(n) -> u64;
pub fn constraint_propagation_ac3(domains, constraints) -> Option<Vec<Vec<u64>>>;
// network.rs
pub fn transshipment(...) -> LpResult;
pub fn shortest_path_lp_check(g, s, t) -> f64;
pub fn critical_path_method(tasks: &[(f64, Vec<usize>)]) -> (f64, Vec<usize>, Vec<(f64, f64, f64, f64)>);   // duration, critical tasks, ES EF LS LF
pub fn pert(tasks: &[(f64, f64, f64, Vec<usize>)]) -> (f64, f64);   // mean, variance of makespan
pub fn max_flow_lp_check(g, s, t) -> f64;
pub fn network_simplex_lite(...) -> LpResult;
pub fn vehicle_routing_savings(depot_dist, dist, demand, cap) -> Vec<Vec<usize>>;   // Clarke-Wright
pub fn job_shop_shifting_bottleneck_lite(...) -> f64;
pub fn scheduling_spt(jobs) -> Vec<usize>;  edd(jobs);  moore_hodgson(jobs) -> Vec<usize> (min late jobs);  johnson_two_machine(jobs) -> Vec<usize>;  lpt_makespan(jobs, machines) -> (f64, Vec<usize>);
pub fn interval_scheduling_max(intervals) -> Vec<usize>;
pub fn weighted_interval_scheduling(intervals) -> (f64, Vec<usize>);
pub fn gantt_data(schedule) -> Vec<(usize, f64, f64)>;
```
Property: simplex matches interior point on 100 random feasible LPs;
duality gap < 1e-8; dual of dual is primal; sensitivity: perturbing b
within range changes objective by dual * delta; Held-Karp equals branch
and bound on TSP-as-ILP small; knapsack DP equals branch and bound;
transportation optimal cost matches simplex; CPM makespan equals DAG
longest path; Johnson's rule optimal vs brute force n <= 8; n_queens_count
matches known 92 for n=8; sudoku solves the hard AI Escargot instance.

### 12. opt/metaheuristics.rs, convex.rs, game_theory.rs
```rust
// metaheuristics.rs (all take f: &dyn Fn(&[f64]) -> f64, bounds, budget, rng)
pub fn nelder_mead(f, x0, step, tol, max_iter) -> (Vec<f64>, f64);
pub fn pattern_search(f, x0, ...) -> (Vec<f64>, f64);
pub fn golden_section(f, a, b, tol) -> f64;  brent_minimize(f, a, b, tol) -> (f64, f64);
pub fn differential_evolution(f, bounds, pop, cr, f_weight, iters, rng) -> (Vec<f64>, f64);
pub fn particle_swarm(f, bounds, n_particles, w, c1, c2, iters, rng) -> (Vec<f64>, f64);
pub fn cma_es(f, x0, sigma0, budget, rng) -> (Vec<f64>, f64);   // full covariance adaptation
pub fn genetic_algorithm(fitness, encode: GaConfig, rng) -> (Vec<f64>, f64);
pub fn genetic_algorithm_permutation(fitness: &dyn Fn(&[usize]) -> f64, n, config, rng) -> (Vec<usize>, f64);   // OX/PMX crossover, for TSP-likes
pub fn simulated_annealing_generic<S>(energy, neighbor, s0, schedule, iters, rng) -> (S, f64);
pub fn tabu_search<S: Hash>(...) -> (S, f64);
pub fn basin_hopping(f, x0, ...) -> (Vec<f64>, f64);
pub fn multistart_lbfgs(f, grad, bounds, n_starts, rng) -> (Vec<f64>, f64);
pub fn bayesian_optimization_lite(f, bounds, n_init, n_iter, rng) -> (Vec<f64>, f64);   // GP + expected improvement, needs learn/gp
pub fn nsga2(objectives: &[&dyn Fn(&[f64]) -> f64], bounds, pop, gens, rng) -> Vec<(Vec<f64>, Vec<f64>)>;   // Pareto front
pub fn pareto_front(points: &[Vec<f64>]) -> Vec<usize>;
pub fn hypervolume_2d(front, ref_point) -> f64;
pub fn benchmark_functions() -> Vec<(&'static str, fn(&[f64]) -> f64, Vec<(f64, f64)>, f64)>;   // sphere, Rosenbrock, Rastrigin, Ackley, Griewank, Schwefel, Levy, Michalewicz, with known optima
pub fn convergence_curve(history) -> Vec<f64>;
// convex.rs
pub fn gradient_descent(f, grad, x0, lr: LrSchedule, iters) -> Vec<f64>;
pub fn momentum / nesterov / adagrad / rmsprop / adam / adamw(...) -> Vec<f64>;
pub fn lbfgs(f, grad, x0, m, tol, max_iter) -> (Vec<f64>, f64);
pub fn bfgs(f, grad, x0, tol, max_iter) -> (Vec<f64>, f64);
pub fn conjugate_gradient_nonlinear(f, grad, x0, ...) -> (Vec<f64>, f64);   // Polak-Ribiere
pub fn newton_method_nd(f, grad, hess, x0, ...) -> (Vec<f64>, f64);
pub fn trust_region_dogleg(...) -> (Vec<f64>, f64);
pub fn line_search_wolfe(f, grad, x, dir) -> f64;  backtracking(f, x, dir, grad_x);
pub fn projected_gradient(f, grad, project: &dyn Fn(&[f64]) -> Vec<f64>, x0, ...) -> Vec<f64>;
pub fn proximal_gradient(smooth_grad, prox: &dyn Fn(&[f64], f64) -> Vec<f64>, x0, ...) -> Vec<f64>;   // ISTA
pub fn fista(...) -> Vec<f64>;
pub fn prox_l1(x, t) -> Vec<f64>;  prox_l2;  prox_box(x, lo, hi);  prox_simplex(x);
pub fn admm_lasso(a, b, lambda, rho, iters) -> Vec<f64>;
pub fn admm_generic(...) -> Vec<f64>;
pub fn lasso_coordinate_descent(a, b, lambda) -> Vec<f64>;
pub fn ridge_closed_form(a, b, lambda) -> Vec<f64>;
pub fn elastic_net(a, b, l1, l2) -> Vec<f64>;
pub fn logistic_regression_fit(x, y, lambda) -> Vec<f64>;   // Newton or LBFGS
pub fn quadratic_program_active_set(q, c, a, b) -> Option<Vec<f64>>;
pub fn augmented_lagrangian(f, grad, constraints, ...) -> Vec<f64>;
pub fn penalty_method(...) -> Vec<f64>;
pub fn kkt_residual(f_grad, constraints, x, lambda) -> f64;
pub fn subgradient_method(...) -> Vec<f64>;
pub fn frank_wolfe(...) -> Vec<f64>;
pub fn mirror_descent_simplex(...) -> Vec<f64>;
pub fn dual_ascent(...) -> Vec<f64>;
pub fn convexity_check_numeric(f, bounds, trials, rng) -> bool;
pub fn condition_number_effect_demo(kappa) -> (usize, usize);   // GD vs CG iterations on quadratic
// game_theory.rs
pub fn minimax_value(payoff: &Matrix) -> (f64, Vec<f64>, Vec<f64>);   // zero-sum via LP
pub fn dominated_strategies(payoff) -> Vec<usize>;  iterated_elimination(a, b) -> (Vec<usize>, Vec<usize>);
pub fn nash_2x2(a: &Matrix, b: &Matrix) -> Vec<(Vec<f64>, Vec<f64>)>;   // all equilibria incl mixed
pub fn nash_bimatrix_lemke_howson(a, b) -> (Vec<f64>, Vec<f64>);
pub fn nash_support_enumeration(a, b, max_support) -> Vec<(Vec<f64>, Vec<f64>)>;
pub fn correlated_equilibrium_lp(a, b) -> Matrix;
pub fn best_response(payoff, opponent_mixed) -> Vec<usize>;
pub fn fictitious_play(a, b, iters) -> (Vec<f64>, Vec<f64>);
pub fn replicator_dynamics(payoff, x0, t_end, dt) -> Vec<Vec<f64>>;
pub fn evolutionarily_stable_check(payoff, strategy, tol) -> bool;
pub fn hawk_dove(v, c) -> Matrix;  prisoners_dilemma(t, r, p, s);  stag_hunt();  chicken();  matching_pennies();  rock_paper_scissors();
pub fn iterated_pd_tournament(strategies: &[&dyn IpdStrategy], rounds, noise, rng) -> Vec<(String, f64)>;   // tit-for-tat, grim, pavlov, random, etc built in
pub fn shapley_value(v: &dyn Fn(u64) -> f64, n) -> Vec<f64>;   // characteristic function on bitmask coalitions
pub fn shapley_monte_carlo(v, n, samples, rng) -> Vec<f64>;
pub fn banzhaf_index(v, n) -> Vec<f64>;
pub fn core_check_small(v, n, allocation) -> bool;
pub fn nucleolus_small(v, n) -> Vec<f64>;
pub fn voting_power_weighted(weights, quota) -> Vec<f64>;
pub fn first_price_auction_equilibrium_uniform(n) -> f64;   // bid shading factor
pub fn second_price_dominant_check() -> bool;
pub fn revenue_equivalence_sim(n, trials, rng) -> (f64, f64);
pub fn vcg_auction(bids: &[Vec<f64>], items) -> (Vec<Option<usize>>, Vec<f64>);
pub fn cake_cutting_divide_choose(...) -> (f64, f64);
pub fn gale_shapley_optimality_check(...) -> bool;
pub fn stackelberg_2x2(a, b) -> (usize, usize, f64, f64);
pub fn cournot_equilibrium(n, demand_intercept, demand_slope, costs) -> Vec<f64>;
pub fn bertrand_equilibrium(...) -> f64;
pub fn public_goods_game_sim(...) -> Vec<f64>;
pub fn colonel_blotto_sim(fields, troops, trials, rng) -> Matrix;
pub fn backward_induction(tree: &GameTree) -> (Vec<usize>, Vec<f64>);
pub fn alpha_beta_search(state: &dyn GameState, depth) -> (i64, usize);   // generic two-player
pub fn mcts_lite(state, iterations, c, rng) -> usize;
```
Property: CMA-ES solves Rosenbrock d=10 to 1e-8 within budget; DE finds
Rastrigin global for d=5 in 9/10 seeds; Adam on convex quadratic converges;
LBFGS matches Newton on quadratics in <= n iterations; prox_l1 is soft
threshold; ADMM lasso matches coordinate descent within 1e-6; minimax
value of RPS is 0 with uniform strategies; Lemke-Howson output is a Nash
(no profitable deviation, checked); replicator on RPS cycles (conserved
quantity); Shapley values sum to v(grand coalition); ESS of hawk-dove is
v/c hawks; alpha-beta equals minimax on tic-tac-toe (draw).

## Phase E: quantum

### 13. quantum/wavefunction.rs and schrodinger.rs
```rust
// wavefunction.rs
pub struct Wavefunction1D { pub psi: Vec<Complex>, pub dx: f64, pub x0: f64 }
impl Wavefunction1D { gaussian_packet(x0_c, k0, sigma, grid); plane_wave(k, grid); normalize(&mut self); norm(); probability_density() -> Vec<f64>; expectation_x(); expectation_p() (spectral); variance_x(); variance_p(); uncertainty_product(); momentum_space() -> Vec<Complex> (fft); overlap(other) -> Complex; energy(v: &[f64], hbar, m) -> f64; }
pub fn hermite_polynomial(n, x) -> f64;
pub fn harmonic_oscillator_eigenstate(n, x, m, omega, hbar) -> f64;
pub fn infinite_well_eigenstate(n, x, l) -> f64;  infinite_well_energy(n, l, m, hbar);
pub fn hydrogen_radial(n, l, r, a0) -> f64;   // associated Laguerre
pub fn hydrogen_energy(n) -> f64;   // eV
pub fn hydrogen_orbital_density(n, l, m, r, theta, phi) -> f64;   // with Part3 spherical harmonics
pub fn laguerre_associated(n, k, x) -> f64;
pub fn coherent_state(alpha: Complex, n_max) -> Vec<Complex>;   // Fock coefficients
pub fn squeezed_state(r, phi, n_max) -> Vec<Complex>;
pub fn wigner_function(psi: &[Complex], dx, x, p) -> f64;
pub fn husimi_q(psi, dx, x, p, sigma) -> f64;
// schrodinger.rs
pub fn tise_solve_fd(v: &[f64], dx, m, hbar, n_states) -> (Vec<f64>, Vec<Vec<f64>>);   // finite difference + Part1 eigen (tridiagonal)
pub fn tise_solve_numerov(v: &dyn Fn(f64) -> f64, x_range, n, e_range, m, hbar) -> Vec<(f64, Vec<f64>)>;   // shooting with node counting
pub fn tise_solve_matrix_basis(v, basis: Basis, n_basis) -> (Vec<f64>, Matrix);   // HO or plane-wave basis
pub fn tdse_split_operator(psi: &mut Wavefunction1D, v: &[f64], dt, steps, m, hbar);   // Strang, fft
pub fn tdse_crank_nicolson(psi, v, dt, steps, m, hbar);   // tridiagonal solve, unitary
pub fn tdse_2d_split_operator(psi: &mut [Complex], nx, ny, dx, v, dt, steps);
pub fn absorbing_boundary_cap(v: &mut [f64], width, strength);
pub fn transmission_coefficient(v, e, m, hbar) -> f64;   // transfer matrix, arbitrary barrier
pub fn tunneling_rectangular_exact(v0, width, e, m, hbar) -> f64;
pub fn wkb_tunneling(v: &dyn Fn(f64) -> f64, e, turning_points, m, hbar) -> f64;
pub fn wkb_quantization(v, n, m, hbar) -> f64;   // Bohr-Sommerfeld energy
pub fn reflection_step_potential(v0, e) -> f64;
pub fn ramsauer_townsend_wells(...) -> Vec<f64>;
pub fn double_well_splitting(v, ...) -> f64;
pub fn stark_shift_perturbative(field, n) -> f64;
pub fn perturbation_theory_1st(h0_states, h0_energies, v_pert) -> Vec<f64>;
pub fn perturbation_theory_2nd(...) -> Vec<f64>;
pub fn variational_ground_state(v, trial: &dyn Fn(f64, &[f64]) -> f64, params0, m, hbar) -> (f64, Vec<f64>);   // Nelder-Mead on <H>
pub fn imaginary_time_propagation(v, dx, n, dt, steps, m, hbar) -> (f64, Vec<f64>);
pub fn ehrenfest_check(psi_t: &[Wavefunction1D], v, dt) -> f64;
pub fn wavepacket_scattering(v, k0, sigma, ...) -> (f64, f64);   // T, R from split flux
pub fn gross_pitaevskii_1d(psi, v, g, dt, steps) -> ...;   // nonlinear split-step, BEC
pub fn soliton_bright_exact(x, t, ...) -> Complex;
pub fn quantum_carpet(l, psi0, times) -> Vec<Vec<f64>>;   // revivals in a box
pub fn revival_time(l, m, hbar) -> f64;
pub fn zeno_effect_sim(...) -> f64;
```
Property: FD eigenvalues of infinite well within 0.1% of n^2 formula for
n <= 5 at 2000 points; HO eigenvalues (n + 1/2) hbar omega within 1e-6;
split-operator conserves norm to 1e-12 and energy to 1e-8; free packet
spreads at analytic rate; rectangular barrier transfer matrix matches
closed form; hydrogen energies -13.6/n^2; imaginary time converges to FD
ground state; uncertainty product >= hbar/2, = hbar/2 for Gaussian;
Ehrenfest d<p>/dt = -<dV/dx> within discretization error; GP soliton
propagates without dispersion.

### 14. quantum/circuit.rs and algorithms.rs
```rust
// circuit.rs (state vector simulator, n <= 24 qubits practical)
pub struct QState { pub n: usize, pub amps: Vec<Complex> }
impl QState { zero(n); basis(n, idx); from_amps(v); plus_all(n); norm(); normalize; probability(idx); probabilities(); measure_all(rng) -> u64; measure_qubit(q, rng) -> (bool, Self); sample_counts(shots, rng) -> Vec<(u64, u64)>; expectation_z(q); expectation_pauli_string(s: &str) -> f64; fidelity(other); inner(other) -> Complex; entanglement_entropy(partition: &[usize]) -> f64; reduced_density_matrix(keep: &[usize]) -> Vec<Vec<Complex>>; schmidt_coefficients(partition); bloch_vector(q) -> Vec3; }
pub struct Gate { pub matrix: [[Complex; 2]; 2] }
impl Gate { x(); y(); z(); h(); s(); sdg(); t(); tdg(); rx(theta); ry; rz; phase(phi); u3(theta, phi, lambda); sqrt_x(); from_matrix(m) -> Result (unitarity check) }
pub struct Circuit { pub n: usize, pub ops: Vec<Op> }
pub enum Op { Single(usize, Gate), Controlled(usize, usize, Gate), CCX(usize, usize, usize), Swap(usize, usize), Measure(usize), Barrier, Custom(Vec<usize>, Vec<Vec<Complex>>) }
impl Circuit {
    pub fn new(n); x(q) -> &mut Self (builder); h(q); cx(c, t); cz; ccx; swap; rx(q, theta); ...; append(other); inverse(); depth(); gate_count();
    pub fn run(&self, initial: &QState) -> QState;
    pub fn run_shots(&self, shots, rng) -> Vec<(u64, u64)>;
    pub fn unitary_small(&self) -> Vec<Vec<Complex>>;   // n <= 10
    pub fn to_qasm_lite(&self) -> String;  from_qasm_lite(s);
    pub fn draw_ascii(&self) -> String;
}
pub struct DensityMatrix { pub n: usize, pub rho: Vec<Vec<Complex>> }
impl DensityMatrix { from_state(s); from_mixture(states, probs); purity(); von_neumann_entropy(); apply_gate; apply_channel(k: &[Vec<Vec<Complex>>]); partial_trace(keep); is_valid(tol); }
pub fn depolarizing_channel(p) -> Vec<Vec<Vec<Complex>>>;  amplitude_damping(gamma);  phase_damping(gamma);  bit_flip(p);  phase_flip(p);
pub fn pauli_decompose(h: &[Vec<Complex>]) -> Vec<(String, f64)>;
pub fn random_state(n, rng) -> QState;  random_unitary_2q(rng);
pub fn bell_state(which: u8) -> QState;  ghz(n);  w_state(n);
pub fn chsh_value(state, angles) -> f64;
pub fn quantum_teleportation_demo(rng) -> (Vec3, Vec3);   // input and output Bloch vectors
pub fn superdense_coding_demo(bits) -> (bool, bool);
pub fn no_cloning_fidelity_bound() -> f64;   // 5/6 optimal cloner check via channel
// algorithms.rs
pub fn qft_circuit(n) -> Circuit;  iqft(n);
pub fn qft_check_vs_fft(n) -> f64;
pub fn grover(oracle_marked: &[u64], n, rng) -> (u64, usize);   // returns found + iterations
pub fn grover_optimal_iterations(n_items, n_marked) -> usize;
pub fn deutsch_jozsa(f: &dyn Fn(u64) -> bool, n) -> bool;   // constant vs balanced
pub fn bernstein_vazirani(secret: u64, n) -> u64;
pub fn simon_lite(...) -> u64;
pub fn phase_estimation(unitary_pow: &dyn Fn(u64) -> Vec<Vec<Complex>>, eigenstate, n_ancilla) -> f64;
pub fn shor_period_finding_sim(a, n_mod, n_qubits) -> Option<u64>;   // small N (15, 21), full circuit sim
pub fn shor_classical_post(a, r, n) -> Option<(u64, u64)>;
pub fn vqe_lite(hamiltonian_paulis: &[(String, f64)], ansatz: &dyn Fn(&[f64]) -> Circuit, params0, rng) -> (f64, Vec<f64>);   // Nelder-Mead
pub fn h2_hamiltonian_sto3g(bond_length) -> Vec<(String, f64)>;   // tabulated coefficients
pub fn qaoa_maxcut(g: &Graph, p_layers, rng) -> (f64, Vec<f64>, u64);
pub fn trotter_evolution(paulis, t, steps) -> Circuit;
pub fn quantum_walk_line(steps, coin: Gate) -> Vec<f64>;   // position distribution
pub fn quantum_walk_graph(g, steps) -> Vec<f64>;
pub fn amplitude_amplification(...) -> Circuit;
pub fn quantum_counting(oracle, n, ancilla) -> f64;
pub fn hhl_lite_2x2(a, b) -> Vec<f64>;
pub fn error_correction_3bit_flip_demo(p, trials, rng) -> (f64, f64);   // logical vs physical error rate
pub fn shor_9_qubit_code_demo(...) -> (f64, f64);
pub fn steane_code_encode() -> Circuit;
pub fn randomized_benchmarking_sim(depths, noise_p, trials, rng) -> Vec<(usize, f64)>;
pub fn quantum_volume_sim(n, noise, trials, rng) -> bool;
```
Property: circuit unitary of H then H is identity; Bell state CHSH =
2 sqrt 2 at optimal angles and <= 2 for product states; QFT on basis
state matches DFT column; Grover finds marked in optimal iterations with
prob > 0.9; phase estimation recovers phase to n_ancilla bits; Shor sim
factors 15; teleportation output Bloch = input to 1e-10; entanglement
entropy of Bell = 1 bit, GHZ 1-vs-rest = 1 bit; density matrix trace 1,
purity of mixture < 1; VQE H2 at 0.74 A within 1e-3 Ha of FCI value
-1.137; depolarizing channel is trace-preserving; 3-bit code beats
physical below p = 0.5.

### 15. quantum/spin.rs and solid_state.rs
```rust
// spin.rs
pub fn pauli_matrices() -> [Vec<Vec<Complex>>; 3];
pub fn spin_operators(s: f64) -> (Matrix3C, Matrix3C, Matrix3C);   // any spin, (2s+1) dim
pub fn spin_coherent_state(s, theta, phi) -> Vec<Complex>;
pub struct SpinChain { pub n: usize, pub j: f64, pub jz: f64, pub h_field: f64, pub periodic: bool }
impl SpinChain {
    pub fn hamiltonian_dense(&self) -> Vec<Vec<Complex>>;   // n <= 14
    pub fn hamiltonian_sparse(&self) -> CsrComplexMatrix;    // n <= 20 with Lanczos
    pub fn ground_state_lanczos(&self, iters) -> (f64, Vec<Complex>);
    pub fn spectrum_small(&self) -> Vec<f64>;
    pub fn magnetization(state) -> f64;  correlation(state, i, j) -> f64;  structure_factor(state, k);
    pub fn time_evolve_krylov(state, t, dt) -> Vec<Complex>;
    pub fn entanglement_entropy_cut(state, cut) -> f64;
    pub fn bethe_ansatz_ground_energy_xxx(n) -> f64;   // reference values
}
pub fn heisenberg_2site_exact() -> Vec<f64>;   // singlet-triplet
pub fn ising_transverse_field_exact(n, g) -> f64;   // Jordan-Wigner free fermion ground energy
pub fn itf_critical_point() -> f64;   // g = 1
pub fn magnon_dispersion(j, k, s) -> f64;
pub fn lanczos(matvec: &dyn Fn(&[Complex]) -> Vec<Complex>, dim, iters, rng) -> (Vec<f64>, Vec<Vec<Complex>>);
pub fn dmrg_lite(...) -> f64;   // optional stretch goal, 2-site, small bond dim
pub fn clebsch_gordan_table(j1, j2) -> Vec<((f64, f64, f64, f64), f64)>;   // wraps Part3 lie
pub fn singlet_projector(...); pub fn total_spin_operator(n) -> ...;
pub fn larmor_precession(b: Vec3, mu, t) -> So3;
pub fn rabi_oscillation(omega_rabi, detuning, t) -> f64;
pub fn ramsey_fringes(...) -> f64;
pub fn spin_echo_sim(t2_star, t2, ...) -> f64;
pub fn bloch_equations(m0: Vec3, b: &dyn Fn(f64) -> Vec3, t1, t2, t_end, dt) -> Vec<Vec3>;
pub fn nmr_fid(frequencies, t2s, t, fs) -> Vec<f64>;
pub fn zeeman_splitting(b, g_factor, m_j) -> f64;
pub fn hyperfine_hydrogen_21cm() -> f64;
// solid_state.rs
pub fn tight_binding_1d(n, t_hop, on_site: &[f64], periodic) -> (Vec<f64>, Matrix);
pub fn tight_binding_band_1d(k, t_hop, a) -> f64;   // -2t cos(ka)
pub fn ssh_model(n, t1, t2) -> (Vec<f64>, Matrix);  ssh_winding_number(t1, t2) -> i32;  ssh_edge_states(n, t1, t2) -> usize;
pub fn tight_binding_square(nx, ny, t) -> Vec<f64>;  graphene_dispersion(kx, ky, t) -> (f64, f64);  dirac_points_graphene() -> Vec<Vec2>;
pub fn kronig_penney(v0, a, b, e, m, hbar) -> f64;   // dispersion determinant; bands where |f| <= 1
pub fn kronig_penney_bands(v0, a, b, e_range, n) -> Vec<(f64, f64)>;
pub fn density_of_states_1d/2d/3d_free(e, m, hbar) -> f64;
pub fn dos_from_bands(bands: &[f64], sigma) -> Vec<(f64, f64)>;
pub fn fermi_dirac(e, mu, t) -> f64;  bose_einstein(e, mu, t);   // may exist in thermodynamics; wrap
pub fn fermi_energy_free(n_density, m, hbar) -> f64;
pub fn sommerfeld_heat_capacity(t, t_fermi) -> f64;
pub fn debye_heat_capacity(t, theta_d) -> f64;  einstein_heat_capacity(t, theta_e);
pub fn phonon_dispersion_1d_monatomic(k, spring, m, a) -> f64;  diatomic(k, spring, m1, m2, a) -> (f64, f64);
pub fn bloch_oscillation_period(e_field, a, hbar) -> f64;
pub fn landau_levels(b, n, m, hbar) -> f64;
pub fn hofstadter_butterfly(q_max, n_k) -> Vec<(f64, f64)>;   // flux vs energy points
pub fn quantum_hall_conductance(n_filled) -> f64;
pub fn drude_conductivity(n, tau, m) -> f64;  hall_coefficient(n, q);
pub fn effective_mass_from_band(band: &dyn Fn(f64) -> f64, k0, h) -> f64;
pub fn semiconductor_carrier_density(e_gap, t, m_e, m_h) -> f64;
pub fn pn_junction_builtin(na, nd, ni, t) -> f64;  depletion_width(...);
pub fn bcs_gap_equation(t, tc) -> f64;  bcs_tc_from_coupling(...);
pub fn josephson_current(ic, phase) -> f64;  josephson_frequency(v);
pub fn anderson_localization_1d(n, w_disorder, trials, rng) -> f64;   // Lyapunov / localization length
pub fn conductance_landauer(transmissions: &[f64]) -> f64;
```
Property: ITF ground energy matches Jordan-Wigner exact for n = 10 across
g; Lanczos matches dense eigen n = 12; SSH winding 1 iff t1 < t2 and edge
states appear; graphene gap zero at K points; Kronig-Penney free limit
recovers parabola; Debye low-T goes as T^3 and high-T to Dulong-Petit;
Landau level spacing hbar omega_c; Hofstadter at flux 1/2 has 2 bands;
entanglement entropy at ITF critical point scales as (1/6) ln n within
20%; Bloch equations decay envelopes at T1, T2.

## Phase F: statistical mechanics and molecular

### 16. statmech/ising.rs and lattice_models.rs
```rust
pub struct Ising2D { pub n: usize, pub spins: Vec<i8>, pub j: f64, pub h: f64, pub beta: f64, pub periodic: bool }
impl Ising2D {
    pub fn random(n, beta, rng) / cold(n, beta);
    pub fn energy(&self) -> f64;  magnetization();  energy_per_site();
    pub fn metropolis_sweep(&mut self, rng);
    pub fn wolff_cluster_step(&mut self, rng) -> usize;   // cluster size
    pub fn heat_bath_sweep(&mut self, rng);
    pub fn sample(&mut self, sweeps, thermalize, measure_every, rng) -> IsingStats { e_mean, e_var, m_mean, m_abs, susceptibility, heat_capacity, binder_cumulant };
    pub fn correlation_function(&self, r) -> f64;  correlation_length_estimate();
    pub fn autocorrelation_time_metropolis(&self, rng) -> f64;
}
pub fn ising_tc_exact() -> f64;   // 2 / ln(1 + sqrt 2)
pub fn onsager_magnetization(beta, j) -> f64;
pub fn onsager_energy(beta, j) -> f64;   // elliptic integral, Part1
pub fn ising_1d_exact(beta, j, h, n) -> (f64, f64);   // transfer matrix
pub fn finite_size_scaling(sizes, stats: &[IsingStats]) -> (f64, f64);   // Tc and nu estimates via Binder crossing
pub fn ising_3d(n, beta, rng) -> ...;
pub struct Potts2D { pub q: u8, ... }  impl { metropolis; swendsen_wang; tc_exact(q) }
pub struct XyModel2D { pub theta: Vec<f64>, ... }
impl XyModel2D { metropolis_sweep; vorticity(); vortex_count(); helicity_modulus(); kt_transition_estimate() }
pub struct HeisenbergLattice { spins: Vec<Vec3>, ... }
pub fn partition_function_exact_small(hamiltonian: &dyn Fn(u64) -> f64, n_sites, beta) -> f64;
pub fn free_energy / entropy_from_z(...);
pub fn wang_landau(energy: &dyn Fn(u64) -> i64, n, f_final, rng) -> Vec<f64>;   // density of states
pub fn multicanonical_from_wl(...) -> IsingStats;
pub fn parallel_tempering_ising(betas, ...) -> Vec<IsingStats>;
pub fn jarzynski_check(...) -> (f64, f64);
pub fn fluctuation_dissipation_check(stats) -> f64;
// lattice_models.rs
pub fn percolation_site(n, p, rng) -> (Vec<bool>, bool);   // spans? extends Part2 automata percolation, union-find
pub fn percolation_bond(n, p, rng) -> bool;
pub fn percolation_threshold_binary_search(n, trials, rng) -> f64;
pub fn cluster_size_distribution(grid) -> Vec<u64>;   // union-find (add DisjointSet to discrete/)
pub fn fortuin_kasteleyn_map_check(...) -> f64;
pub fn self_avoiding_walk_count(n) -> BigInt;  saw_sample_rosenbluth(n, rng) -> (Vec<(i64, i64)>, f64);  connective_constant_estimate;
pub fn random_walk_lattice(n, dim, rng) -> Vec<Vec<i64>>;  return_probability(dim, steps);
pub fn polymer_end_to_end(saw_samples) -> f64;  flory_exponent_estimate;
pub fn lattice_gas_step(...);
pub fn asep_simulate(n, p, q, alpha, beta, t, rng) -> Vec<f64>;   // current, density profile
pub fn dimer_count_kasteleyn(m, n) -> BigInt;   // Pfaffian on planar grid
pub fn six_vertex_ice_count_small(n) -> BigInt;
pub fn hard_squares_entropy_estimate(n) -> f64;
pub fn kpz_growth_ballistic(n, t, rng) -> Vec<f64>;  interface_width(h) -> f64;  growth_exponents_estimate(...) -> (f64, f64);
pub fn eden_growth_stats(...);
pub fn sandpile_avalanche_distribution(n, drops, rng) -> Vec<u64>;   // extends Part2 sandpile, power-law fit
pub fn power_law_fit_clauset(data, x_min) -> (f64, f64);   // MLE alpha + KS
```
Property: Ising 2D energy and |m| match Onsager within 1% at beta away
from Tc (n=64, Wolff); Binder crossing gives Tc within 0.5%; Wolff
autocorrelation much shorter than Metropolis near Tc; 1D transfer matrix
matches enumeration n=10; Wang-Landau DOS reproduces exact small-lattice
Z; site percolation threshold 0.5927 within 0.005; SAW counts match OEIS
through n=15; dimer count 2x2 = 2, 8x8 = 12988816; KPZ beta ~ 1/3 within
0.05; power-law fit recovers alpha on synthetic.

### 17. statmech/md.rs and kinetics.rs
```rust
// md.rs
pub struct MdSystem { pub pos: Vec<Vec3>, pub vel: Vec<Vec3>, pub mass: Vec<f64>, pub box_size: Vec3, pub periodic: bool, pub potential: Potential, grid: SpatialHash, pub cutoff: f64 }
pub enum Potential { LennardJones { eps: f64, sigma: f64 }, Morse { d, a, r0 }, Coulomb { ke: f64 }, LjCoulomb { ... }, Harmonic { k, r0 }, Custom(Box<dyn Fn(f64) -> (f64, f64)>) }
impl MdSystem {
    pub fn lattice_fcc(cells, density, temp, eps, sigma, rng) -> Self;
    pub fn forces(&self) -> Vec<Vec3>;   // cell list, minimum image
    pub fn step_velocity_verlet(&mut self, dt);
    pub fn thermostat_berendsen(&mut self, t_target, tau, dt);
    pub fn thermostat_nose_hoover(&mut self, t_target, q, dt);
    pub fn thermostat_langevin(&mut self, t_target, gamma, dt, rng);
    pub fn barostat_berendsen(&mut self, p_target, ...);
    pub fn temperature(&self) -> f64;  kinetic_energy();  potential_energy();  pressure_virial();  total_momentum();
    pub fn rdf(&self, bins, r_max) -> Vec<f64>;                 // radial distribution
    pub fn msd(traj: &[Vec<Vec3>]) -> Vec<f64>;  diffusion_coefficient(msd, dt);
    pub fn vacf(traj_vel) -> Vec<f64>;  vdos_from_vacf(vacf, dt) -> Vec<f64>;
    pub fn structure_factor(&self, k_values) -> Vec<f64>;
    pub fn equilibrate(&mut self, steps, dt, t_target, rng);
    pub fn run_nve(&mut self, steps, dt) -> Vec<MdSample>;
    pub fn energy_drift(samples) -> f64;
    pub fn maxwell_boltzmann_check(&self) -> TestResult;
    pub fn melting_indicator_lindemann(&self, traj) -> f64;
}
pub fn lj_reduced_units_note() -> &'static str;
pub fn lj_phase_point(t_star, rho_star) -> &'static str;   // rough phase from known diagram
pub fn ewald_sum_energy_lite(charges, pos, box_l, alpha, k_max) -> f64;
pub fn harmonic_crystal_heat_capacity_check(...) -> f64;
pub fn virial_coefficient_b2(potential, t, r_max, n) -> f64;   // integral
pub fn mean_free_path(density, sigma) -> f64;  collision_rate(...);
pub fn green_kubo_viscosity_lite(...) -> f64;
pub fn umbrella_sampling_pmf(...) -> Vec<f64>;   // WHAM-lite
pub fn steered_pull(...) -> Vec<f64>;
// kinetics.rs
pub fn arrhenius(a, ea, t) -> f64;  eyring(dh, ds, t);
pub fn rate_equations(stoich: &Matrix, rates: &dyn Fn(&[f64]) -> Vec<f64>, c0, t_end, rtol) -> Vec<(f64, Vec<f64>)>;   // stiff, Part1 BDF
pub fn mass_action_rates(reactions: &[Reaction], k: &[f64]) -> impl Fn(&[f64]) -> Vec<f64>;
pub fn gillespie_ssa(reactions, k, x0: &[u64], t_end, rng) -> Vec<(f64, Vec<u64>)>;
pub fn tau_leaping(reactions, k, x0, t_end, tau, rng) -> Vec<(f64, Vec<u64>)>;
pub fn michaelis_menten(s, vmax, km) -> f64;  mm_fit(s, v) -> (f64, f64);  lineweaver_burk(s, v);
pub fn hill_equation(s, vmax, k, n) -> f64;  hill_fit(...);
pub fn equilibrium_constant_from_g(dg, t) -> f64;
pub fn equilibrium_composition(stoich, k_eq, totals) -> Vec<f64>;   // solve
pub fn steady_state_approx_check(...) -> f64;
pub fn oscillating_brusselator(a, b, c0, t_end) -> Vec<(f64, Vec<f64>)>;
pub fn oregonator(...) -> Vec<(f64, Vec<f64>)>;
pub fn lotka_volterra_chemical(...) -> ...;
pub fn autocatalysis_ignition(...) -> f64;
pub fn chain_reaction_criticality(k_branch, k_term) -> f64;
pub fn enzyme_inhibition(s, i, vmax, km, ki, kind: Inhibition) -> f64;
pub fn temperature_jump_relaxation(...) -> f64;
pub fn kinetic_isotope_effect_estimate(...) -> f64;
pub fn transition_state_theory_rate(...) -> f64;
pub fn kramers_rate_check(...) -> f64;   // cross-validate Part3-era sde kramers
pub fn nucleation_rate_cnt(delta_g_star, ...) -> f64;
pub fn jmak_avrami(t, k, n) -> f64;  avrami_fit(...);
pub fn photochemistry_quantum_yield(...) -> f64;
pub fn ph_from_equilibria(acids: &[(f64, f64)], base_conc) -> f64;   // polyprotic solve
pub fn titration_curve(...) -> Vec<(f64, f64)>;
pub fn buffer_henderson_hasselbalch(pka, ratio) -> f64;
pub fn debye_huckel_activity(z, ionic_strength) -> f64;
pub fn nernst(e0, z, ratio, t) -> f64;
pub fn butler_volmer(i0, alpha, eta, t) -> f64;
pub fn cottrell_current(...) -> f64;
```
Property: NVE energy drift < 1e-4 per 1e5 steps at dt = 0.005 LJ units;
Maxwell-Boltzmann speeds pass KS after equilibration; LJ RDF first peak
near 2^(1/6) sigma; diffusion from MSD matches VACF Green-Kubo within
15%; B2(T_Boyle) = 0 near known LJ Boyle temperature; Gillespie mean of
1e4 runs matches ODE for linear network; tau-leaping converges to SSA;
MM fit recovers Vmax, Km; Brusselator oscillates for b > 1 + a^2;
pH of 0.1 M weak acid matches quadratic solve; Nernst at 25 C gives
59.2 mV per decade.

## Phase G: domain packs

### 18. bio/ (epidemiology, population, seq_align, phylo, neuro)
```rust
// epidemiology.rs
pub fn sir(beta, gamma, s0, i0, t_end) -> Vec<(f64, f64, f64, f64)>;
pub fn seir(...) / sirs / sis / seirs / msir -> ...;
pub fn r0_sir(beta, gamma) -> f64;  herd_immunity_threshold(r0);  final_size_equation(r0) -> f64 (implicit solve);
pub fn sir_stochastic_gillespie(...) -> Vec<(f64, u64, u64, u64)>;
pub fn extinction_probability_epidemic(r0, i0) -> f64;
pub fn network_sir(g: &Graph, beta, gamma, patient_zero, rng) -> Vec<(usize, usize, usize)>;
pub fn epidemic_threshold_network(g) -> f64;   // 1 / spectral radius
pub fn sir_with_vaccination(...) / with_demography / two_strain / age_structured(contact_matrix, ...);
pub fn effective_r_estimate(incidence, serial_interval) -> Vec<f64>;   // Cori method lite
pub fn serial_interval_fit(...) -> (f64, f64);
pub fn seir_fit_to_incidence(data, ...) -> (f64, f64, f64);
pub fn wallinga_teunis(...) -> Vec<f64>;
// population.rs
pub fn logistic_growth(r, k, n0, t) -> f64;  gompertz;  richards;  allee_effect_ode(...);
pub fn lotka_volterra(alpha, beta, delta, gamma, x0, y0, t_end) -> Vec<(f64, f64, f64)>;   // exists in nonlinear? wrap/extend with conserved quantity
pub fn rosenzweig_macarthur(...) -> ...;  competition_lv(...) -> ...;  coexistence_condition(...);
pub fn leslie_matrix(fecundity, survival) -> Matrix;  leslie_growth_rate(l) -> f64;  stable_age_distribution(l);
pub fn euler_lotka_solve(lx, mx) -> f64;
pub fn ricker_map(r, k, n0, steps) -> Vec<f64>;  beverton_holt;  bifurcation_ricker(r_range, ...);
pub fn metapopulation_levins(c, e, p0, t_end) -> Vec<f64>;
pub fn wright_fisher(n, p0, generations, rng) -> Vec<f64>;
pub fn moran_process(n, i0, fitness, rng) -> (bool, u64);  fixation_probability_moran(n, i, r) -> f64;
pub fn genetic_drift_heterozygosity(n, t) -> f64;
pub fn hardy_weinberg(p) -> (f64, f64, f64);  hw_chi_square_test(observed) -> TestResult;
pub fn selection_one_locus(p0, w: [f64; 3], generations) -> Vec<f64>;
pub fn mutation_selection_balance(mu, s) -> f64;
pub fn coalescent_time_expected(n, k) -> f64;  coalescent_simulate(n_samples, rng) -> PhyloTree;
pub fn tajima_d(...) -> f64;  watterson_theta(segregating, n);  nucleotide_diversity(seqs);
pub fn fst(subpop_freqs) -> f64;
pub fn kin_selection_hamilton(r, b, c) -> bool;
pub fn price_equation_decompose(...) -> (f64, f64);
// seq_align.rs
pub fn needleman_wunsch(a: &[u8], b: &[u8], score: &Scoring) -> (i64, String, String);
pub fn smith_waterman(a, b, score) -> (i64, usize, usize, String, String);
pub fn gotoh_affine(a, b, match_s, mismatch, gap_open, gap_extend) -> (i64, String, String);
pub fn banded_alignment(a, b, band, score) -> i64;
pub fn hirschberg(a, b, score) -> (String, String);   // linear space
pub fn blosum62() -> [[i8; 24]; 24];  pam250();
pub fn kmer_index(seq, k) -> HashMap-free Vec-based index;
pub fn minimizers(seq, k, w) -> Vec<(usize, u64)>;
pub fn gc_content(seq) -> f64;  reverse_complement(seq) -> Vec<u8>;  transcribe;  translate(seq) -> Vec<u8> (codon table);  orf_find(seq, min_len) -> Vec<(usize, usize, i8)>;
pub fn codon_usage(seq) -> Vec<(String, f64)>;
pub fn melting_temperature_wallace(seq) -> f64;  tm_nearest_neighbor(seq, conc);
pub fn hamming_seqs(a, b) -> Option<usize>;  p_distance;  jukes_cantor_distance(p) -> f64;  kimura_2p(transitions, transversions);
pub fn msa_center_star(seqs, score) -> Vec<String>;
pub fn profile_from_msa(msa) -> Matrix;  consensus(msa);
pub fn pssm_score(profile, seq) -> Vec<f64>;
pub fn de_bruijn_assembly_lite(reads, k) -> Vec<Vec<u8>>;
pub fn burrows_wheeler_search(text_bwt_index, pattern) -> Vec<usize>;   // FM-index lite on codes::bwt
// phylo.rs
pub struct PhyloTree { pub parent: Vec<Option<usize>>, pub branch_length: Vec<f64>, pub labels: Vec<String> }
impl PhyloTree { from_newick(s) -> Result; to_newick(); leaves(); is_binary(); height(); total_length(); mrca(a, b); distance(a, b); robinson_foulds(other) -> usize; }
pub fn upgma(dist: &Matrix, labels) -> PhyloTree;
pub fn neighbor_joining(dist, labels) -> PhyloTree;
pub fn parsimony_fitch(tree, characters) -> u64;
pub fn likelihood_jc69(tree, seqs) -> f64;   // Felsenstein pruning
pub fn bootstrap_trees(seqs, n, method, rng) -> Vec<f64>;   // support values
pub fn birth_death_tree(lambda, mu, n_leaves, rng) -> PhyloTree;
pub fn gamma_statistic(tree) -> f64;
pub fn lineage_through_time(tree) -> Vec<(f64, usize)>;
// neuro.rs
pub fn hodgkin_huxley(i_ext: &dyn Fn(f64) -> f64, t_end, dt) -> Vec<(f64, f64, f64, f64, f64)>;   // V, m, h, n
pub fn hh_spike_threshold_estimate() -> f64;  hh_fi_curve(i_range) -> Vec<(f64, f64)>;
pub fn fitzhugh_nagumo_neuron(...) -> Vec<(f64, f64)>;
pub fn morris_lecar(...) / izhikevich(a, b, c, d, i, t_end, dt) / adex(...) -> Vec<(f64, f64)>;
pub fn izhikevich_presets() -> Vec<(&'static str, [f64; 4])>;   // RS, IB, CH, FS, LTS
pub fn lif_neuron(i, tau, v_th, v_reset, t_end, dt, rng_noise) -> Vec<f64>;  lif_fi_exact(i, ...);
pub fn interspike_intervals(spikes) -> Vec<f64>;  cv_isi;  fano_factor(counts);
pub fn poisson_spike_train(rate, t_end, rng) -> Vec<f64>;
pub fn psth(trains, bin) -> Vec<f64>;  raster_data(trains);
pub fn spike_triggered_average(stimulus, spikes, window) -> Vec<f64>;
pub fn tuning_curve_fit_von_mises(angles, rates) -> (f64, f64, f64);
pub fn izhikevich_network(n_exc, n_inh, ...) -> Vec<(f64, usize)>;   // the classic 1000-neuron net
pub fn synapse_exp(g_max, tau, spikes, t) -> f64;  alpha_synapse;  stdp_window(dt, ...);  stdp_train(...);
pub fn hopfield_store(patterns) -> Matrix;  hopfield_recall(w, probe, steps) -> Vec<i8>;  hopfield_capacity_check(n, trials, rng) -> f64;
pub fn wilson_cowan(...) -> Vec<(f64, f64)>;
pub fn cable_equation_1d(...) -> Vec<f64>;  length_constant(rm, ri, d);
pub fn nernst_potential(z, c_out, c_in, t) -> f64;  ghk_voltage(...);
pub fn reaction_time_ddm(drift, threshold, noise, trials, rng) -> Vec<(f64, bool)>;   // drift diffusion
pub fn ddm_analytic_accuracy(drift, threshold, noise) -> f64;
```
Property: SIR final size satisfies the implicit equation; network SIR dies
out below epidemic threshold; LV orbits conserve the invariant to 1e-6;
Leslie growth rate equals dominant eigenvalue; Wright-Fisher fixation
prob of neutral allele = p0 over 1e4 runs; NW on identical strings gives
max score and identity alignment; SW finds planted motif; NJ recovers
tree from its own distance matrix (RF = 0); HH fires above rheobase with
refractory period ~ known; Izhikevich presets produce their named
patterns; Hopfield recalls at 0.13n patterns, fails at 0.2n; DDM sim
accuracy matches analytic.

### 19. finance/, astro/, fem/, learn/, units/
```rust
// finance/options.rs
pub fn black_scholes(s, k, t, r, sigma, q, call: bool) -> f64;
pub fn bs_greeks(...) -> Greeks { delta, gamma, vega, theta, rho };
pub fn implied_volatility(price, s, k, t, r, call) -> Option<f64>;   // Jaeckel-style bracketing + Newton
pub fn binomial_crr(s, k, t, r, sigma, steps, call, american: bool) -> f64;
pub fn trinomial(...) -> f64;
pub fn monte_carlo_european(..., n_paths, rng) -> (f64, f64);   // price, std err; antithetic + control variate
pub fn monte_carlo_asian / barrier / lookback(...);
pub fn longstaff_schwartz_american(..., rng) -> f64;
pub fn heston_price_mc(...) -> f64;  heston_characteristic_fn_price(...) -> f64 (Part3 fft, Carr-Madan);
pub fn merton_jump_price(...) -> f64;
pub fn bs_pde_crank_nicolson(...) -> f64;
pub fn put_call_parity_check(...) -> f64;
pub fn volatility_smile_svi(params, k) -> f64;  svi_fit(strikes, ivs);
pub fn delta_hedging_sim(..., rng) -> (f64, f64);   // P&L mean, std vs rebalance freq
// finance/rates.rs
pub fn discount_factor(r, t, comp: Compounding) -> f64;  npv(rate, cashflows);  irr(cashflows) -> Option<f64>;  xirr(dates, cashflows);
pub fn bond_price(face, coupon, ytm, periods) -> f64;  ytm_solve(...);  duration_macaulay;  duration_modified;  convexity;
pub fn bootstrap_zero_curve(bonds) -> Vec<(f64, f64)>;
pub fn forward_rate(z1, t1, z2, t2) -> f64;
pub fn nelson_siegel(t, b0, b1, b2, tau) -> f64;  ns_fit(maturities, yields);
pub fn vasicek_bond_price(...) -> f64;  cir_bond_price(...);
pub fn amortization_schedule(principal, rate, n) -> Vec<(f64, f64, f64, f64)>;
pub fn mortgage_payment(p, r, n) -> f64;
// finance/portfolio.rs, risk.rs
pub fn returns_from_prices(p) -> Vec<f64>;  log_returns;
pub fn markowitz_frontier(mu, cov, n_points) -> Vec<(f64, f64, Vec<f64>)>;   // QP via Part1
pub fn min_variance_weights(cov) -> Vec<f64>;  tangency_portfolio(mu, cov, rf);
pub fn sharpe(returns, rf) -> f64;  sortino;  max_drawdown(prices);  calmar;  information_ratio;
pub fn capm_beta(asset, market) -> (f64, f64);
pub fn risk_parity_weights(cov) -> Vec<f64>;
pub fn kelly_fraction(p, b) -> f64;  kelly_continuous(mu, sigma, rf);
pub fn var_historical(returns, alpha) -> f64;  var_parametric;  cvar_historical;
pub fn var_cornish_fisher(...);
pub fn garch_var_forecast(...) -> f64;   // ties stochastic/timeseries
pub fn backtest_sma_crossover(prices, fast, slow) -> BacktestStats;
pub fn kupiec_test(violations, n, alpha) -> TestResult;
// astro/ (kepler, elements, lambert, maneuvers, time_systems, coords)
pub fn kepler_solve_elliptic(m_anomaly, e, tol) -> f64;   // E from M, Newton with good seed
pub fn kepler_solve_universal(...) -> f64;
pub fn true_from_eccentric(e_anom, e) -> f64;  eccentric_from_true;  mean_from_eccentric;
pub struct OrbitalElements { pub a, e, i, raan, argp, nu: f64 }
pub fn elements_from_state(r: Vec3, v: Vec3, mu) -> OrbitalElements;
pub fn state_from_elements(el, mu) -> (Vec3, Vec3);
pub fn propagate_kepler(r0, v0, dt, mu) -> (Vec3, Vec3);   // f and g functions
pub fn orbit_period(a, mu) -> f64;  vis_viva(r, a, mu);  specific_energy;  angular_momentum;
pub fn hohmann(r1, r2, mu) -> (f64, f64, f64);   // dv1, dv2, transfer time
pub fn bielliptic(r1, r2, rb, mu) -> (f64, f64);
pub fn plane_change_dv(v, delta_i) -> f64;  combined_maneuver(...);
pub fn lambert_universal(r1: Vec3, r2: Vec3, tof, mu, prograde: bool) -> Result<(Vec3, Vec3), SolveError>;   // Izzo or universal variables
pub fn porkchop_data(dep_states, arr_states, tofs, mu) -> Vec<Vec<f64>>;   // c3 grid
pub fn patched_conic_escape(...) -> f64;  sphere_of_influence(a, m_planet, m_sun);
pub fn gravity_assist_deflection(v_inf, r_p, mu) -> f64;
pub fn oberth_effect_dv(...) -> f64;
pub fn j2_raan_drift(a, e, i, j2, r_body, mu) -> f64;  sun_synchronous_inclination(a, e);
pub fn hill_radius(a, e, m, m_central) -> f64;  roche_limit(...);
pub fn tsiolkovsky(isp, m0, mf) -> f64;  staging_analysis(stages) -> f64;
pub fn ground_track(el, t_range, rotation_rate) -> Vec<(f64, f64)>;
pub fn julian_date(y, m, d, h, min, s) -> f64;  jd_to_calendar(jd);  gmst(jd) -> f64;  local_sidereal(jd, lon);
pub fn equatorial_to_horizontal(ra, dec, lat, lst) -> (f64, f64);  ecliptic_to_equatorial(...);  precession_approx(...);
pub fn sun_position_approx(jd) -> (f64, f64);  moon_position_approx(jd);  planet_position_low_precision(planet, jd) -> (f64, f64, f64);
pub fn rise_set_times(ra, dec, lat, lon, jd) -> Option<(f64, f64)>;
pub fn tle_parse_lite(l1, l2) -> Result<TleElements, GeomError>;   // parse only; SGP4 out of scope, document why
pub fn two_body_energy_check(...) -> f64;
// fem/ (fem1d, fem2d, fdtd, spectral_pde)
pub fn fem_1d_poisson(f, a, b, bc, n) -> Vec<f64>;   // linear elements, exact assembly
pub fn fem_1d_general(p, q, f, bc, n) -> Vec<f64>;   // -(p u')' + q u = f
pub fn fem_1d_quadratic(...) -> Vec<f64>;
pub fn fem_1d_error_h1(u_h, u_exact, du_exact) -> f64;  convergence_rate(errors, hs);
pub struct FemMesh2 { pub nodes: Vec<Vec2>, pub tris: Vec<[usize; 3]>, pub boundary: Vec<usize> }
impl FemMesh2 { rect(w, h, nx, ny); disk(r, n); from_delaunay(points); refine_uniform(); quality_min_angle() }
pub fn fem_2d_poisson(mesh, f, dirichlet: &dyn Fn(Vec2) -> Option<f64>) -> Vec<f64>;   // P1, CSR + CG
pub fn fem_2d_helmholtz(mesh, k, ...) -> Vec<f64>;
pub fn fem_2d_elasticity_plane_stress(mesh, e, nu, loads, fixed) -> Vec<Vec2>;   // displacement
pub fn von_mises_stress(mesh, u, e, nu) -> Vec<f64>;
pub fn fem_2d_heat_transient(mesh, ...) -> Vec<Vec<f64>>;
pub fn fem_eigenvalues_drum(mesh, n) -> Vec<f64>;   // compare Part3 cavity circular membrane
pub fn fdtd_1d(eps_r: &[f64], source, n, steps) -> Vec<Vec<f64>>;   // Yee, Mur ABC
pub fn fdtd_2d_tm(eps_r, source_pos, freq, nx, ny, steps, pml: usize) -> Vec<f64>;   // with PML
pub fn fdtd_courant_check(dx, dt, c) -> bool;
pub fn waveguide_cutoff_check_fdtd(...) -> f64;   // vs Part3 cavity modes
pub fn photonic_crystal_bandgap_1d(eps_a, eps_b, ...) -> Vec<(f64, f64)>;   // transfer matrix
pub fn spectral_poisson_periodic(f, n) -> Vec<f64>;   // wraps Part3 fft_poisson
pub fn chebyshev_collocation_bvp(p, q, f, bc, n) -> Vec<f64>;  cheb_diff_matrix(n) -> Matrix;
pub fn spectral_convergence_demo(...) -> Vec<f64>;   // exponential vs algebraic
// learn/ (nn, gp, cluster, tree)
pub struct Mlp { pub layers: Vec<(Matrix, Vec<f64>)>, pub activation: Act }
impl Mlp { new(sizes, act, rng); forward(x) -> Vec<f64>; backward(x, y, loss) -> Gradients; train_sgd(data, epochs, lr, batch, rng) -> Vec<f64>; train_adam(...); predict; loss(data); numerical_grad_check(x, y) -> f64; }
pub enum Act { Relu, Sigmoid, Tanh, Identity, Softmax }
pub enum Loss { Mse, CrossEntropy }
pub fn conv2d_forward(input, kernels, stride, pad) -> ...;   // one layer, for education
pub fn train_xor_demo(rng) -> f64;  train_mnist_like(data, rng) -> f64;
pub fn linear_regression_gd_check(...) -> f64;   // matches Part1 closed form
pub struct Gp { pub kernel: KernelFn, pub noise: f64, x_train: Vec<Vec<f64>>, ... }
pub enum KernelFn { Rbf { l: f64, s: f64 }, Matern32, Matern52, Periodic { l, p }, Linear, Sum(Box, Box), Product(Box, Box) }
impl Gp { fit(x, y) (Cholesky); predict(x_star) -> (Vec<f64>, Vec<f64>); log_marginal_likelihood(); optimize_hyperparams(bounds) (Nelder-Mead); sample_prior(x, n, rng); sample_posterior(...); }
pub fn kmeans(data, k, iters, rng) -> (Vec<Vec<f64>>, Vec<usize>);  kmeans_pp_init;  elbow_data(data, k_range);
pub fn dbscan(data, eps, min_pts) -> Vec<i32>;
pub fn hierarchical_agglomerative(data, linkage) -> Vec<(usize, usize, f64)>;  dendrogram_cut(merges, k);
pub fn gaussian_mixture_em(data, k, iters, rng) -> (Vec<f64>, Vec<Vec<f64>>, Vec<Matrix>);
pub fn silhouette_score(data, labels) -> f64;  adjusted_rand_index(a, b);  davies_bouldin;
pub fn knn_classify(train, labels, x, k) -> usize;  knn_regress;
pub fn decision_tree_fit(x, y, max_depth, min_leaf) -> Tree;  tree_predict;  gini;  feature_importance;
pub fn random_forest_fit(x, y, n_trees, ..., rng) -> Forest;
pub fn gradient_boosting_lite(x, y, n_rounds, lr, depth) -> Gbm;
pub fn train_test_split(n, frac, rng) -> (Vec<usize>, Vec<usize>);  k_fold(n, k, rng);
pub fn confusion_matrix(y, pred, k) -> Matrix;  precision_recall_f1;  roc_curve(scores, y) -> Vec<(f64, f64)>;  auc(roc);
pub fn standardize(data) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>);
pub fn naive_bayes_gaussian_fit / predict;
pub fn perceptron_train(...);  svm_smo_lite(x, y, c, kernel, tol) -> Svm;
// units/ (quantity, dimensional)
#[derive(Clone, Copy, PartialEq)] pub struct Dim { pub m: i8, pub kg: i8, pub s: i8, pub a: i8, pub k: i8, pub mol: i8, pub cd: i8 }
#[derive(Clone, Copy)] pub struct Quantity { pub value: f64, pub dim: Dim }
impl Quantity { meters(v); kg(v); seconds(v); newtons(v); joules(v); watts(v); volts(v); ... (50 common constructors);
  add(o) -> Result<Self, DimError>; sub; mul; div; pow(i8); sqrt() -> Result; to(unit: Unit) -> Result<f64, DimError>; format_si() -> String; }
pub fn parse_quantity(s: &str) -> Result<Quantity, GeomError>;   // "9.81 m/s^2", "3 kWh"
pub fn unit_convert(v, from: &str, to: &str) -> Result<f64, GeomError>;   // length/mass/time/temp/energy/pressure/volume/speed/data tables
pub fn buckingham_pi(dims: &[Dim]) -> Vec<Vec<Rational>>;   // null space of dimension matrix, exact
pub fn dimensionless_groups_named() -> Vec<(&'static str, &'static str)>;   // Re, Fr, We, Ma, Pr, Ra, ... with formulas
pub fn dimensional_check_formula(expr: &Expr, var_dims) -> Result<Dim, DimError>;   // uses exact/symbolic
pub fn natural_units_convert(...) -> f64;   // hbar = c = 1 helpers
pub fn planck_units() -> Vec<(&'static str, f64)>;
pub fn si_prefixes_format(v) -> String;
pub fn constants_codata() -> Vec<(&'static str, f64, &'static str)>;   // 2022 CODATA, consolidate the scattered constants
```
Property: BS price satisfies put-call parity to 1e-12; binomial converges
to BS at 1/n rate; implied vol roundtrips; Longstaff-Schwartz American
put >= European; IRR of annuity matches closed form; duration approximates
price sensitivity; Kepler solve residual < 1e-12 including e = 0.99;
elements-state roundtrip; Hohmann dv matches textbook Earth-Mars; Lambert
solution propagated hits r2 within 1e-6; J2 sun-synchronous inclination at
700 km is 98.2 deg; JD of 2000-01-01 12:00 is 2451545.0; FEM 1D quadratic
converges at h^3 in H1... (rate h^2 in H1, h^3 in L2); FEM drum eigenvalues
match Bessel zeros within mesh error; FDTD pulse travels at c and reflects
correctly at dielectric with Fresnel coefficient; MLP gradient check
< 1e-6; GP posterior mean interpolates noiseless training points; XOR
trains to 100%; k-means monotone decreasing inertia; AUC of perfect
classifier = 1; buckingham_pi on pendulum (g, l, m, t) yields t sqrt(g/l);
unit_convert("3.6 km/h" -> "m/s") == 1.0.

## Session plan

Grouped, one PR per session, 30 sessions:

1. exact/bigint.rs (1)
2. exact/rational.rs + bigfloat.rs (2)
3. exact/polynomial.rs + contfrac.rs (3a)
4. exact/symbolic.rs (3b)
5. discrete/primes.rs + number_theory.rs (4)
6. discrete/combinatorics.rs + partitions.rs + sequences.rs (5) + DisjointSet
7. graph/core.rs + paths.rs (6a)
8. graph/flow.rs + matching.rs (6b)
9. graph/spectral.rs + coloring.rs + layout.rs (6c)
10. codes/checksum.rs + block.rs (7a)
11. codes/reed_solomon.rs + convolutional.rs (7b)
12. codes/compression.rs + crypto_math.rs (7c)
13. stochastic/markov.rs + hmm.rs (8)
14. stochastic/sde.rs + point_process.rs (9)
15. stochastic/queueing.rs + timeseries.rs (10a)
16. stochastic/rmt.rs + extreme.rs (10b)
17. opt/lp.rs (11a)
18. opt/integer.rs + network.rs (11b)
19. opt/metaheuristics.rs + convex.rs (12a)
20. opt/game_theory.rs (12b)
21. quantum/wavefunction.rs + schrodinger.rs (13)
22. quantum/circuit.rs + algorithms.rs (14)
23. quantum/spin.rs + solid_state.rs (15)
24. statmech/ising.rs + lattice_models.rs (16)
25. statmech/md.rs + kinetics.rs (17)
26. bio/ all five files (18), split if large
27. finance/ all four (19a)
28. astro/ all six (19b)
29. fem/ all four (19c)
30. learn/ + units/ (19d)

Dependency-light starting points: 1, 5, 6, 7, 10, 13, 17, 21, 24, 28.

Per-session prompt identical to Parts 1-3.

## Cross-references and shared additions

- DisjointSet (union-find) -> discrete/, used by graph MST, percolation, clustering
- CsrComplexMatrix + Lanczos -> quantum/spin, shared with Part1 sparse
- TestResult reused from Part1 statistics/inference
- Part3 fft consumed by: sequences (Berlekamp-Massey check), sde (fBm), timeseries (spectral), quantum (split-operator, QFT check), finance (Carr-Madan), fem (spectral)
- learn/gp consumed by opt/metaheuristics bayesian_optimization
- exact/symbolic consumed by units/dimensional_check
- constants_codata: migrate the scattered physical constants in classical/gravitation/thermodynamics to reference one table (keep old consts as re-exports)

## What is deliberately out of scope (all parts)

GPU/SIMD, arbitrary-precision interval arithmetic, SGP4, full CAS
(integration beyond table rules, Groebner bases), cryptographically
secure implementations, HDF5/netCDF I/O, plotting. Each is either a
dependency violation, a correctness liability, or a project of its own.
