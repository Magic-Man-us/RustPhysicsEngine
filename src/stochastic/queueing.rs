//! Queueing theory: birth-death queues, Erlang loss and delay formulas,
//! networks of queues, and continuous-time Markov chains.
//!
//! Almost every closed form here is a birth-death chain in disguise. A queue
//! with Poisson arrivals and exponential service moves up one state at rate
//! `lambda` and down one at a rate set by how many servers are busy, so the
//! stationary distribution telescopes into a product of ratios and the means
//! follow by summation. The Erlang formulas are the two boundary cases of
//! that product: B when a full system turns customers away, C when it makes
//! them wait.
//!
//! Two results tie the whole module together and are worth stating because
//! the tests lean on them. Little's law, `L = lambda W`, holds for every
//! model below -- it is a statement about areas under a sample path and
//! assumes nothing about the arrival or service distributions. And the
//! Pollaczek-Khinchine formula shows what the exponential assumption was
//! buying: for a single server the mean queue depends on the service
//! distribution only through its first two moments, so M/D/1 has exactly half
//! the queue of M/M/1 at the same load.
//!
//! Where a model has no closed form the module simulates it instead. The
//! event-driven simulator tracks the number in system by integrating over a
//! merged event list rather than by invoking Little's law, so comparing its
//! output against `lambda W` is a real check rather than a tautology.

use crate::error::GeomError;
use crate::linalg::lu;
use crate::linalg::matrix::Matrix;
use crate::monte_carlo::Rng;

/// Below this the utilisation is treated as saturated and means are infinite.
const STABILITY_TOL: f64 = 1e-12;

/// Which birth-death chain a set of metrics came from.
///
/// Carried alongside the means so that [`QueueMetrics::pn`] can report the
/// exact stationary probability of `n` in the system. Models with no
/// product-form state distribution report [`QueueModel::MeanValueOnly`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QueueModel {
    /// `c` servers, unbounded queue, offered load `a = lambda / mu`.
    MMc { c: usize, a: f64 },
    /// `c` servers, at most `k` in the system, offered load `a`.
    MMcK { c: usize, k: usize, a: f64 },
    /// Unlimited servers: the state distribution is Poisson with mean `a`.
    MMInf { a: f64 },
    /// Means only -- M/G/1 and the diffusion approximations.
    MeanValueOnly,
}

/// The standard summary of a queue in steady state.
///
/// `l` and `lq` count customers, `w` and `wq` measure time. The two pairs are
/// linked by Little's law at the *effective* arrival rate, which differs from
/// the offered rate whenever the system turns customers away.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QueueMetrics {
    /// Fraction of server capacity in use, `lambda_eff / (c mu)`.
    pub rho: f64,
    /// Mean number in the system, waiting or in service.
    pub l: f64,
    /// Mean number waiting, not counting those in service.
    pub lq: f64,
    /// Mean time in the system.
    pub w: f64,
    /// Mean time waiting before service starts.
    pub wq: f64,
    /// Probability the system is empty.
    pub p0: f64,
    /// Arrival rate actually admitted; equals the offered rate unless the
    /// system is finite.
    pub lambda_eff: f64,
    /// The chain these came from, for [`QueueMetrics::pn`].
    pub model: QueueModel,
}

impl QueueMetrics {
    /// Stationary probability of exactly `n` customers in the system.
    ///
    /// Returns NaN for [`QueueModel::MeanValueOnly`], where only the means
    /// are determined by the inputs.
    #[must_use]
    pub fn pn(&self, n: usize) -> f64 {
        match self.model {
            QueueModel::MMc { c, a } => birth_death_pn(self.p0, a, c, n),
            QueueModel::MMcK { c, k, a } => {
                if n > k {
                    0.0
                } else {
                    birth_death_pn(self.p0, a, c, n)
                }
            }
            // Unlimited servers is the same recursion with no ceiling: every
            // extra customer divides by its own index, giving the Poisson law.
            QueueModel::MMInf { a } => birth_death_pn((-a).exp(), a, usize::MAX, n),
            QueueModel::MeanValueOnly => f64::NAN,
        }
    }
}

/// Stationary probability of `n` in a birth-death queue with `c` servers and
/// offered load `a`, given the empty-system probability.
///
/// Built as a running product rather than from `a^n / n!`. The closed form
/// is the same, but its two halves overflow independently -- `a^n` reaches
/// infinity and `n!` reaches infinity, and their ratio comes out NaN long
/// before the probability itself underflows. Multiplying `a / k` one step at
/// a time keeps every partial result near the magnitude of the answer, so the
/// tail decays to zero the way it should.
fn birth_death_pn(p0: f64, a: f64, c: usize, n: usize) -> f64 {
    let mut p = p0;
    for k in 1..=n.min(c) {
        p *= a / k as f64;
    }
    if n > c {
        // Past the last server the chain is a geometric walk with ratio a/c,
        // hung off the state where every server is busy.
        p *= (a / c as f64).powi((n - c) as i32);
    }
    p
}

/// A single-server queue with Poisson arrivals and exponential service.
///
/// The stationary distribution is geometric, `p_n = (1 - rho) rho^n`, which
/// gives `L = rho / (1 - rho)` directly.
///
/// # Panics
/// Panics unless `lambda` and `mu` are positive.
#[must_use]
pub fn mm1(lambda: f64, mu: f64) -> QueueMetrics {
    mmc(lambda, mu, 1)
}

/// `c` parallel servers, Poisson arrivals, exponential service, no limit on
/// the queue. The probability an arrival has to wait is Erlang C.
///
/// Unstable loads (`lambda >= c mu`) return infinite means with `rho >= 1`;
/// the queue really does grow without bound there, so that is the answer
/// rather than an error.
///
/// # Panics
/// Panics unless `lambda` and `mu` are positive and `c >= 1`.
#[must_use]
pub fn mmc(lambda: f64, mu: f64, c: usize) -> QueueMetrics {
    assert!(lambda > 0.0, "mmc requires lambda > 0");
    assert!(mu > 0.0, "mmc requires mu > 0");
    assert!(c >= 1, "mmc requires at least one server");

    let a = lambda / mu;
    let rho = a / c as f64;
    if rho >= 1.0 - STABILITY_TOL {
        return QueueMetrics {
            rho,
            l: f64::INFINITY,
            lq: f64::INFINITY,
            w: f64::INFINITY,
            wq: f64::INFINITY,
            p0: 0.0,
            lambda_eff: lambda,
            model: QueueModel::MMc { c, a },
        };
    }

    // sum_{k<c} a^k/k! by a running product, then the geometric tail hung
    // off a^c/c!. Formed stepwise for the reason given on `birth_death_pn`.
    let mut term = 1.0f64;
    let mut sum = 0.0f64;
    for k in 0..c {
        sum += term;
        term *= a / (k + 1) as f64;
    }
    let tail = term / (1.0 - rho);
    let p0 = 1.0 / (sum + tail);

    // Erlang C is the tail mass times the geometric sum above it.
    let c_erlang = tail * p0;
    let lq = c_erlang * rho / (1.0 - rho);
    let wq = lq / lambda;
    let w = wq + 1.0 / mu;
    let l = lambda * w;

    QueueMetrics { rho, l, lq, w, wq, p0, lambda_eff: lambda, model: QueueModel::MMc { c, a } }
}

/// A single server with room for `k` customers in total. Arrivals that find
/// the system full are lost, so the effective arrival rate is `lambda (1 - p_k)`
/// and the queue is stable at any load.
///
/// # Panics
/// Panics unless `lambda` and `mu` are positive and `k >= 1`.
#[must_use]
pub fn mm1k(lambda: f64, mu: f64, k: usize) -> QueueMetrics {
    mmck(lambda, mu, 1, k)
}

/// `c` servers with room for `k` in total, `k >= c`. Arrivals finding the
/// system full are lost.
///
/// # Panics
/// Panics unless `lambda` and `mu` are positive and `c <= k`.
#[must_use]
pub fn mmck(lambda: f64, mu: f64, c: usize, k: usize) -> QueueMetrics {
    assert!(lambda > 0.0, "mmck requires lambda > 0");
    assert!(mu > 0.0, "mmck requires mu > 0");
    assert!(c >= 1, "mmck requires at least one server");
    assert!(k >= c, "mmck requires capacity k >= server count c");

    let a = lambda / mu;
    // Unnormalised birth-death weights: a^n/n! while servers are still free,
    // then a geometric continuation at ratio a/c once all c are busy. Built
    // by a running product for the reason given on `birth_death_pn`.
    let mut weights = Vec::with_capacity(k + 1);
    let mut w = 1.0f64;
    weights.push(w);
    for n in 1..=k {
        w *= if n <= c { a / n as f64 } else { a / c as f64 };
        weights.push(w);
    }
    let total: f64 = weights.iter().sum();
    let p0 = 1.0 / total;

    let mut l = 0.0;
    let mut lq = 0.0;
    for n in 0..=k {
        let p = weights[n] * p0;
        l += n as f64 * p;
        lq += (n.saturating_sub(c)) as f64 * p;
    }
    let p_block = weights[k] * p0;
    let lambda_eff = lambda * (1.0 - p_block);
    let w = l / lambda_eff;
    let wq = lq / lambda_eff;
    let rho = lambda_eff / (c as f64 * mu);

    QueueMetrics { rho, l, lq, w, wq, p0, lambda_eff, model: QueueModel::MMcK { c, k, a } }
}

/// Unlimited servers: every arrival enters service at once. The number in
/// system is Poisson with mean `lambda / mu`, so nobody ever waits.
///
/// # Panics
/// Panics unless `lambda` and `mu` are positive.
#[must_use]
pub fn mm_inf(lambda: f64, mu: f64) -> QueueMetrics {
    assert!(lambda > 0.0, "mm_inf requires lambda > 0");
    assert!(mu > 0.0, "mm_inf requires mu > 0");
    let a = lambda / mu;
    QueueMetrics {
        rho: 0.0,
        l: a,
        lq: 0.0,
        w: 1.0 / mu,
        wq: 0.0,
        p0: (-a).exp(),
        lambda_eff: lambda,
        model: QueueModel::MMInf { a },
    }
}

/// Erlang's loss formula: the fraction of calls blocked by `c` trunks under
/// an offered load of `a` erlangs.
///
/// Computed by the recursion `B_c = a B_{c-1} / (c + a B_{c-1})` rather than
/// the ratio of factorial sums. The two agree exactly in real arithmetic, but
/// the direct form overflows near `c = 170` while the recursion stays in
/// `[0, 1]` at every step and is accurate for any `c`.
///
/// # Panics
/// Panics if `a` is negative.
#[must_use]
pub fn erlang_b(offered_load: f64, c: usize) -> f64 {
    assert!(offered_load >= 0.0, "erlang_b requires a non-negative load");
    let mut b = 1.0;
    for k in 1..=c {
        b = offered_load * b / (k as f64 + offered_load * b);
    }
    b
}

/// Erlang's delay formula: the probability an arrival to an `M/M/c` queue
/// finds every server busy and has to wait.
///
/// Returns 1 for a saturated system. Related to the loss formula by
/// `C = B / (1 - rho (1 - B))`, which is how it is evaluated here.
///
/// # Panics
/// Panics if `load` is negative or `c` is zero.
#[must_use]
pub fn erlang_c(load: f64, c: usize) -> f64 {
    assert!(load >= 0.0, "erlang_c requires a non-negative load");
    assert!(c >= 1, "erlang_c requires at least one server");
    let rho = load / c as f64;
    if rho >= 1.0 - STABILITY_TOL {
        return 1.0;
    }
    let b = erlang_b(load, c);
    b / (1.0 - rho * (1.0 - b))
}

/// The smallest number of trunks that holds blocking at or below
/// `blocking_target` for the given offered load.
///
/// Steps the Erlang B recursion upward, which is monotone decreasing in `c`,
/// so the first `c` that clears the target is the smallest one.
///
/// # Panics
/// Panics unless the target is in `(0, 1]` and the load is non-negative.
#[must_use]
pub fn erlang_b_inverse_capacity(load: f64, blocking_target: f64) -> usize {
    assert!(load >= 0.0, "erlang_b_inverse_capacity requires a non-negative load");
    assert!(
        blocking_target > 0.0 && blocking_target <= 1.0,
        "erlang_b_inverse_capacity requires a target in (0, 1]"
    );
    let mut b = 1.0;
    let mut c = 0usize;
    while b > blocking_target {
        c += 1;
        b = load * b / (c as f64 + load * b);
    }
    c
}

/// The Pollaczek-Khinchine mean-value formula for a single server with
/// Poisson arrivals and a general service distribution.
///
/// `Lq = lambda^2 (var + mean^2) / (2 (1 - rho))`. The service distribution
/// enters only through its first two moments: exponential service has
/// `var = mean^2` and recovers M/M/1, while deterministic service has
/// `var = 0` and halves the queue.
///
/// # Panics
/// Panics unless `lambda` and `service_mean` are positive and the variance is
/// non-negative.
#[must_use]
pub fn mg1_pollaczek_khinchine(lambda: f64, service_mean: f64, service_var: f64) -> QueueMetrics {
    assert!(lambda > 0.0, "mg1_pollaczek_khinchine requires lambda > 0");
    assert!(service_mean > 0.0, "mg1_pollaczek_khinchine requires a positive service mean");
    assert!(service_var >= 0.0, "mg1_pollaczek_khinchine requires a non-negative variance");

    let rho = lambda * service_mean;
    if rho >= 1.0 - STABILITY_TOL {
        return QueueMetrics {
            rho,
            l: f64::INFINITY,
            lq: f64::INFINITY,
            w: f64::INFINITY,
            wq: f64::INFINITY,
            p0: 0.0,
            lambda_eff: lambda,
            model: QueueModel::MeanValueOnly,
        };
    }
    let second_moment = service_var + service_mean * service_mean;
    let lq = lambda * lambda * second_moment / (2.0 * (1.0 - rho));
    let wq = lq / lambda;
    let w = wq + service_mean;
    let l = lq + rho;

    QueueMetrics {
        rho,
        l,
        lq,
        w,
        wq,
        // For M/G/1 the idle probability is 1 - rho whatever the service shape.
        p0: 1.0 - rho,
        lambda_eff: lambda,
        model: QueueModel::MeanValueOnly,
    }
}

/// Kingman's diffusion approximation for the mean wait in a G/G/1 queue,
/// given the squared coefficients of variation of the interarrival and
/// service times.
///
/// `Wq ~ (rho / (1 - rho)) ((ca2 + cs2) / 2) (1 / mu)`. It is exact for
/// M/M/1, where both coefficients are one and the middle factor drops out,
/// and is asymptotically exact as `rho -> 1` for any distribution.
///
/// # Panics
/// Panics unless the rates are positive and the coefficients non-negative.
#[must_use]
pub fn gg1_kingman_approx(lambda: f64, mu: f64, ca2: f64, cs2: f64) -> f64 {
    assert!(lambda > 0.0 && mu > 0.0, "gg1_kingman_approx requires positive rates");
    assert!(ca2 >= 0.0 && cs2 >= 0.0, "gg1_kingman_approx requires non-negative variability");
    let rho = lambda / mu;
    if rho >= 1.0 - STABILITY_TOL {
        return f64::INFINITY;
    }
    (rho / (1.0 - rho)) * ((ca2 + cs2) / 2.0) / mu
}

/// The residual `L - lambda W`, which any consistent set of steady-state
/// numbers must drive to zero.
///
/// Little's law is a pathwise identity, not a distributional one, so this is
/// a genuine check on measured or simulated quantities rather than an
/// assumption about the model.
#[must_use]
pub fn littles_law_check(l: f64, lambda: f64, w: f64) -> f64 {
    l - lambda * w
}

/// An open Jackson network of `M/M/c` nodes.
///
/// `routing[i][j]` is the probability a customer leaving node `i` goes to
/// node `j`; whatever is left over departs the network. Total arrival rates
/// solve the traffic equations `lambda_j = external_j + sum_i lambda_i r_ij`,
/// after which Jackson's theorem says each node behaves in steady state
/// exactly like an isolated `M/M/c_j` queue at its own total rate -- even
/// though the internal arrival streams are not Poisson.
///
/// # Errors
/// Returns [`GeomError::InvalidArgument`] if the shapes disagree, if a
/// routing row sums past one, or if the traffic equations are singular.
pub fn jackson_network(
    routing: &Matrix,
    external: &[f64],
    service: &[f64],
    servers: &[usize],
) -> Result<Vec<QueueMetrics>, GeomError> {
    let n = external.len();
    if !routing.is_square() || routing.rows != n {
        return Err(GeomError::InvalidArgument("jackson_network: routing must be n x n"));
    }
    if service.len() != n || servers.len() != n {
        return Err(GeomError::InvalidArgument("jackson_network: rate/server length mismatch"));
    }
    for i in 0..n {
        let row: f64 = (0..n).map(|j| routing.get(i, j)).sum();
        if !(row <= 1.0 + 1e-9) {
            return Err(GeomError::InvalidArgument(
                "jackson_network: a routing row sums past one",
            ));
        }
        for j in 0..n {
            if !(routing.get(i, j) >= 0.0) {
                return Err(GeomError::InvalidArgument(
                    "jackson_network: routing probabilities must be non-negative",
                ));
            }
        }
    }

    // (I - R^T) lambda = external.
    let mut a = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let delta = if i == j { 1.0 } else { 0.0 };
            a.set(i, j, delta - routing.get(j, i));
        }
    }
    let rates = lu::solve(&a, external)
        .map_err(|_| GeomError::Degenerate("jackson_network: traffic equations are singular"))?;

    rates
        .iter()
        .zip(service.iter().zip(servers.iter()))
        .map(|(&lam, (&mu, &c))| {
            if !(lam > 0.0) {
                return Err(GeomError::InvalidArgument(
                    "jackson_network: a node has non-positive total arrival rate",
                ));
            }
            Ok(mmc(lam, mu, c))
        })
        .collect()
}

/// What an event-driven run measured.
///
/// The time averages come from integrating the sample path over a merged
/// list of arrival and departure events, independently of the customer
/// averages, so `l` and `lambda_eff * w` are two separate measurements of the
/// same quantity rather than one derived from the other.
#[derive(Debug, Clone, PartialEq)]
pub struct QueueSimResult {
    /// Time-average number in the system, from the area under `n(t)`.
    pub l: f64,
    /// Time-average number waiting.
    pub lq: f64,
    /// Customer-average time in the system.
    pub w: f64,
    /// Customer-average time waiting.
    pub wq: f64,
    /// Fraction of the horizon each server was busy, averaged over servers.
    pub rho: f64,
    /// Arrivals per unit time over the horizon.
    pub lambda_eff: f64,
    /// Customers whose service completed within the horizon.
    pub served: usize,
}

/// A first-come-first-served queue with `c` identical servers, simulated
/// event by event.
///
/// `arrival` draws an interarrival gap and `service` a service duration, both
/// from the supplied generator, so any G/G/c queue can be run. Arrivals stop
/// at `t_end`; customers already admitted are followed to completion so the
/// customer averages are not truncated mid-service.
///
/// # Panics
/// Panics unless `c >= 1` and `t_end > 0`.
#[must_use]
pub fn queue_simulate(
    arrival: &dyn Fn(&mut Rng) -> f64,
    service: &dyn Fn(&mut Rng) -> f64,
    c: usize,
    t_end: f64,
    rng: &mut Rng,
) -> QueueSimResult {
    assert!(c >= 1, "queue_simulate requires at least one server");
    assert!(t_end > 0.0, "queue_simulate requires a positive horizon");

    // Each server is described only by the time it next becomes free, which
    // is all a FIFO assignment rule needs.
    let mut free_at = vec![0.0f64; c];
    let mut busy_time = vec![0.0f64; c];
    let mut arrivals: Vec<f64> = Vec::new();
    let mut departures: Vec<f64> = Vec::new();
    let mut starts: Vec<f64> = Vec::new();

    let mut t = 0.0f64;
    loop {
        t += arrival(rng).max(0.0);
        if t > t_end {
            break;
        }
        // FIFO with identical servers: the customer takes whichever server
        // frees up first, which is the earliest of the next-free times.
        let mut which = 0usize;
        for s in 1..c {
            if free_at[s] < free_at[which] {
                which = s;
            }
        }
        let start = t.max(free_at[which]);
        let duration = service(rng).max(0.0);
        free_at[which] = start + duration;
        busy_time[which] += duration;
        arrivals.push(t);
        starts.push(start);
        departures.push(start + duration);
    }

    let served = arrivals.len();
    if served == 0 {
        return QueueSimResult {
            l: 0.0,
            lq: 0.0,
            w: 0.0,
            wq: 0.0,
            rho: 0.0,
            lambda_eff: 0.0,
            served: 0,
        };
    }

    let w: f64 = departures.iter().zip(&arrivals).map(|(d, a)| d - a).sum::<f64>() / served as f64;
    let wq: f64 = starts.iter().zip(&arrivals).map(|(s, a)| s - a).sum::<f64>() / served as f64;

    // Time averages by integration. Merge the two sorted event streams and
    // accumulate n * dt; the queue count is the same walk with the number in
    // service subtracted, which is min(n, c).
    let mut sorted_dep = departures.clone();
    sorted_dep.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let horizon = sorted_dep[served - 1].max(t_end);

    let (mut i, mut j) = (0usize, 0usize);
    let (mut n, mut last, mut area, mut area_q) = (0usize, 0.0f64, 0.0f64, 0.0f64);
    while i < served || j < served {
        let next_arr = if i < served { arrivals[i] } else { f64::INFINITY };
        let next_dep = if j < served { sorted_dep[j] } else { f64::INFINITY };
        let next = next_arr.min(next_dep);
        area += n as f64 * (next - last);
        area_q += n.saturating_sub(c) as f64 * (next - last);
        last = next;
        // Ties go to the departure: a customer who leaves exactly when
        // another arrives frees the server first.
        if next_dep <= next_arr {
            n -= 1;
            j += 1;
        } else {
            n += 1;
            i += 1;
        }
    }

    let busy: f64 = busy_time.iter().sum();
    QueueSimResult {
        l: area / horizon,
        lq: area_q / horizon,
        w,
        wq,
        rho: busy / (c as f64 * horizon),
        lambda_eff: served as f64 / horizon,
        served,
    }
}

/// A non-preemptive priority queue with `c` servers and one exponential
/// class per entry of `lambdas`.
///
/// Class 0 has the highest priority. A waiting customer of a higher class is
/// always taken next, but a job already in service runs to completion.
/// Returns one result per class.
///
/// The discipline is work-conserving, so Kleinrock's conservation law applies:
/// `sum_k rho_k Wq_k` is the same here as under plain FIFO, however the
/// priorities are arranged. Only the split between classes changes.
///
/// # Panics
/// Panics unless the rate vectors match in length, are positive, and
/// `c >= 1`.
#[must_use]
pub fn priority_queue_simulate(
    lambdas: &[f64],
    mus: &[f64],
    c: usize,
    t_end: f64,
    rng: &mut Rng,
) -> Vec<QueueSimResult> {
    assert!(c >= 1, "priority_queue_simulate requires at least one server");
    assert!(t_end > 0.0, "priority_queue_simulate requires a positive horizon");
    assert!(
        !lambdas.is_empty() && lambdas.len() == mus.len(),
        "priority_queue_simulate requires one service rate per class"
    );
    assert!(
        lambdas.iter().all(|&l| l > 0.0) && mus.iter().all(|&m| m > 0.0),
        "priority_queue_simulate requires positive rates"
    );

    let classes = lambdas.len();
    let total_lambda: f64 = lambdas.iter().sum();

    // Superposition: the merged arrival stream is Poisson at the summed rate,
    // and each arrival belongs to class k with probability lambda_k / total.
    let mut pending: Vec<(f64, usize)> = Vec::new();
    let mut t = 0.0f64;
    loop {
        t += -rng.next_f64().max(1e-300).ln() / total_lambda;
        if t > t_end {
            break;
        }
        let u = rng.next_f64() * total_lambda;
        let mut acc = 0.0;
        let mut k = classes - 1;
        for (idx, &l) in lambdas.iter().enumerate() {
            acc += l;
            if u < acc {
                k = idx;
                break;
            }
        }
        pending.push((t, k));
    }

    let mut waiting: Vec<Vec<f64>> = vec![Vec::new(); classes];
    let mut free_at = vec![0.0f64; c];
    let mut busy_time = vec![0.0f64; c];
    let mut arrivals: Vec<Vec<f64>> = vec![Vec::new(); classes];
    let mut starts: Vec<Vec<f64>> = vec![Vec::new(); classes];
    let mut departures: Vec<Vec<f64>> = vec![Vec::new(); classes];

    let mut next_arrival = 0usize;
    loop {
        // The clock advances to whichever comes first: the next arrival, or
        // the moment a server frees up with someone already waiting.
        let earliest_free = free_at.iter().copied().fold(f64::INFINITY, f64::min);
        let queued = waiting.iter().any(|q| !q.is_empty());
        let next_arr =
            if next_arrival < pending.len() { pending[next_arrival].0 } else { f64::INFINITY };

        if !queued && next_arr.is_infinite() {
            break;
        }
        // Serve now if someone is waiting and a server is free by the time
        // the next arrival would land.
        if queued && earliest_free <= next_arr {
            let which = (0..c).min_by(|&x, &y| {
                free_at[x].partial_cmp(&free_at[y]).unwrap_or(std::cmp::Ordering::Equal)
            });
            let Some(which) = which else { break };
            let k = waiting.iter().position(|q| !q.is_empty()).unwrap_or(0);
            let arrived = waiting[k].remove(0);
            let start = earliest_free.max(arrived);
            let duration = -rng.next_f64().max(1e-300).ln() / mus[k];
            free_at[which] = start + duration;
            busy_time[which] += duration;
            arrivals[k].push(arrived);
            starts[k].push(start);
            departures[k].push(start + duration);
        } else if next_arrival < pending.len() {
            let (at, k) = pending[next_arrival];
            waiting[k].push(at);
            next_arrival += 1;
        } else {
            break;
        }
    }

    let horizon = free_at.iter().copied().fold(t_end, f64::max);
    (0..classes)
        .map(|k| {
            let served = arrivals[k].len();
            if served == 0 {
                return QueueSimResult {
                    l: 0.0,
                    lq: 0.0,
                    w: 0.0,
                    wq: 0.0,
                    rho: 0.0,
                    lambda_eff: 0.0,
                    served: 0,
                };
            }
            let w = departures[k]
                .iter()
                .zip(&arrivals[k])
                .map(|(d, a)| d - a)
                .sum::<f64>()
                / served as f64;
            let wq = starts[k].iter().zip(&arrivals[k]).map(|(s, a)| s - a).sum::<f64>()
                / served as f64;
            let lambda_eff = served as f64 / horizon;
            QueueSimResult {
                l: lambda_eff * w,
                lq: lambda_eff * wq,
                w,
                wq,
                rho: lambda_eff / mus[k],
                lambda_eff,
                served,
            }
        })
        .collect()
}

/// A continuous-time Markov chain, held as its generator matrix.
///
/// Rows of `q` sum to zero: the off-diagonal entries are transition rates and
/// the diagonal is minus their total, so `-q_ii` is the rate of leaving state
/// `i`. Where a discrete chain asks "what is the next state", a generator
/// asks "how long until something happens, and what".
#[derive(Debug, Clone, PartialEq)]
pub struct Ctmc {
    /// The generator. Square, non-negative off the diagonal, rows summing to zero.
    pub q: Matrix,
}

impl Ctmc {
    /// Wraps a generator after checking its shape.
    ///
    /// # Errors
    /// Returns [`GeomError::InvalidArgument`] if the matrix is not square,
    /// has a negative off-diagonal rate, or has a row that does not sum to
    /// zero within `1e-9`.
    pub fn new(q: Matrix) -> Result<Self, GeomError> {
        if !q.is_square() || q.rows == 0 {
            return Err(GeomError::InvalidArgument("Ctmc: generator must be square and non-empty"));
        }
        for i in 0..q.rows {
            let mut sum = 0.0;
            for j in 0..q.rows {
                let v = q.get(i, j);
                if i != j && !(v >= 0.0) {
                    return Err(GeomError::InvalidArgument(
                        "Ctmc: off-diagonal rates must be non-negative",
                    ));
                }
                sum += v;
            }
            if sum.abs() > 1e-9 {
                return Err(GeomError::InvalidArgument("Ctmc: generator rows must sum to zero"));
            }
        }
        Ok(Self { q })
    }

    /// Number of states.
    #[must_use]
    pub fn n(&self) -> usize {
        self.q.rows
    }

    /// Mean time spent in each state per visit, `1 / (-q_ii)`.
    ///
    /// Infinite for an absorbing state, which is never left.
    #[must_use]
    pub fn mean_holding_times(&self) -> Vec<f64> {
        (0..self.n())
            .map(|i| {
                let rate = -self.q.get(i, i);
                if rate <= 0.0 {
                    f64::INFINITY
                } else {
                    1.0 / rate
                }
            })
            .collect()
    }

    /// The jump chain: where the process goes, ignoring how long it waits.
    ///
    /// `P_ij = q_ij / (-q_ii)`. An absorbing state becomes a self-loop so the
    /// result is a valid stochastic matrix.
    ///
    /// # Errors
    /// Returns an error if the resulting matrix is rejected as a chain.
    pub fn embedded_chain(&self) -> Result<crate::stochastic::markov::MarkovChain, GeomError> {
        let n = self.n();
        let mut p = Matrix::zeros(n, n);
        for i in 0..n {
            let out = -self.q.get(i, i);
            if out <= 0.0 {
                p.set(i, i, 1.0);
                continue;
            }
            for j in 0..n {
                if i != j {
                    p.set(i, j, self.q.get(i, j) / out);
                }
            }
        }
        crate::stochastic::markov::MarkovChain::new(p)
    }

    /// The stationary distribution, solving `pi Q = 0` with `sum pi = 1`.
    ///
    /// Solved as a linear system rather than by iterating, so periodicity in
    /// the jump chain is irrelevant -- a continuous-time chain has no period
    /// to speak of, and the linear solve reflects that.
    ///
    /// # Errors
    /// Returns [`GeomError::Degenerate`] if the balance equations are
    /// singular, which happens when the chain is reducible.
    pub fn stationary(&self) -> Result<Vec<f64>, GeomError> {
        let n = self.n();
        // Q^T pi = 0 is rank-deficient by exactly one, so replace the last
        // row with the normalisation.
        let mut a = Matrix::zeros(n, n);
        for i in 0..n - 1 {
            for j in 0..n {
                a.set(i, j, self.q.get(j, i));
            }
        }
        for j in 0..n {
            a.set(n - 1, j, 1.0);
        }
        let mut b = vec![0.0; n];
        b[n - 1] = 1.0;
        lu::solve(&a, &b)
            .map_err(|_| GeomError::Degenerate("Ctmc::stationary: balance equations are singular"))
    }

    /// Simulates one trajectory by the Gillespie construction: hold in the
    /// current state for an exponential time set by its exit rate, then jump
    /// according to the embedded chain.
    ///
    /// Returns `(time of entry, state)` pairs, beginning at `(0, start)`.
    /// Stops early at an absorbing state.
    ///
    /// # Panics
    /// Panics if `start` is out of range or `t_end` is not positive.
    #[must_use]
    pub fn simulate(&self, start: usize, t_end: f64, rng: &mut Rng) -> Vec<(f64, usize)> {
        assert!(start < self.n(), "Ctmc::simulate: start state out of range");
        assert!(t_end > 0.0, "Ctmc::simulate requires a positive horizon");
        let mut out = vec![(0.0, start)];
        let mut t = 0.0f64;
        let mut s = start;
        loop {
            let rate = -self.q.get(s, s);
            if rate <= 0.0 {
                return out;
            }
            t += -rng.next_f64().max(1e-300).ln() / rate;
            if t > t_end {
                return out;
            }
            let mut u = rng.next_f64() * rate;
            let mut next = s;
            for j in 0..self.n() {
                if j == s {
                    continue;
                }
                u -= self.q.get(s, j);
                if u <= 0.0 {
                    next = j;
                    break;
                }
            }
            s = next;
            out.push((t, s));
        }
    }

    /// Mean time to reach `to` starting from `from`.
    ///
    /// Solves `m_i = 1/(-q_ii) + sum_{j != to} P_ij m_j` over the jump chain,
    /// which is the continuous-time analogue of a first-step decomposition:
    /// wait out the holding time, then start again from wherever you land.
    ///
    /// # Errors
    /// Returns an error if the system is singular, which means `to` is not
    /// reachable from every transient state.
    pub fn first_passage(&self, from: usize, to: usize) -> Result<f64, GeomError> {
        let n = self.n();
        if from >= n || to >= n {
            return Err(GeomError::InvalidArgument("Ctmc::first_passage: state out of range"));
        }
        if from == to {
            return Ok(0.0);
        }
        let mut a = Matrix::zeros(n, n);
        let mut b = vec![0.0; n];
        for i in 0..n {
            if i == to {
                a.set(i, i, 1.0);
                continue;
            }
            let out = -self.q.get(i, i);
            if out <= 0.0 {
                // Absorbing and not the target: the target is unreachable.
                return Ok(f64::INFINITY);
            }
            a.set(i, i, 1.0);
            for j in 0..n {
                if j != i && j != to {
                    a.set(i, j, -self.q.get(i, j) / out);
                }
            }
            b[i] = 1.0 / out;
        }
        let m = lu::solve(&a, &b)
            .map_err(|_| GeomError::Degenerate("Ctmc::first_passage: system is singular"))?;
        Ok(m[from])
    }
}

/// Transient distribution of a continuous-time chain by uniformization.
///
/// Writes `P(t) = exp(Qt)` as a Poisson mixture of powers of a discrete
/// chain: pick a rate `L` at least as large as every exit rate, set
/// `P = I + Q/L`, and then `p(t) = sum_k e^{-Lt} (Lt)^k / k! * p0 P^k`. Every
/// term is a probability vector and every weight is positive, so unlike a
/// truncated matrix exponential the partial sums never go negative, however
/// stiff the generator.
///
/// The sum is truncated when the remaining Poisson mass falls below `eps`.
///
/// # Errors
/// Returns [`GeomError::InvalidArgument`] if `p0` is the wrong length, is not
/// a distribution, or if `t` or `eps` are not positive.
pub fn uniformization(
    q_matrix: &Matrix,
    p0: &[f64],
    t: f64,
    eps: f64,
) -> Result<Vec<f64>, GeomError> {
    let n = q_matrix.rows;
    if !q_matrix.is_square() || p0.len() != n || n == 0 {
        return Err(GeomError::InvalidArgument("uniformization: shape mismatch"));
    }
    if !(t > 0.0) || !(eps > 0.0) {
        return Err(GeomError::InvalidArgument("uniformization: t and eps must be positive"));
    }
    let mass: f64 = p0.iter().sum();
    if (mass - 1.0).abs() > 1e-9 || p0.iter().any(|&x| x < 0.0) {
        return Err(GeomError::InvalidArgument("uniformization: p0 must be a distribution"));
    }

    let mut lambda = 0.0f64;
    for i in 0..n {
        lambda = lambda.max(-q_matrix.get(i, i));
    }
    if lambda <= 0.0 {
        // Nothing ever moves.
        return Ok(p0.to_vec());
    }

    let mut p = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let delta = if i == j { 1.0 } else { 0.0 };
            p.set(i, j, delta + q_matrix.get(i, j) / lambda);
        }
    }

    let lt = lambda * t;
    let mut out = vec![0.0; n];
    let mut vec_k = p0.to_vec();

    // The Poisson weights are carried as logarithms. Starting from e^{-Lt}
    // directly fails in both directions once Lt is large: past about 745 the
    // first term underflows to zero, and substituting the smallest subnormal
    // instead makes the recurrence w *= Lt/(k+1) climb by roughly e^{Lt} on
    // its way to the mode, overflowing to infinity long before it gets there.
    // In logarithms the same climb is a sum of bounded increments, and terms
    // are materialised only over the window where they are representable --
    // which is exactly the window where they matter, since everything outside
    // it is below e^{-745} of the total mass.
    let ln_lt = lt.ln();
    let mut log_weight = -lt;
    let mut accumulated = 0.0f64;
    for k in 0..MAX_UNIFORMIZATION_TERMS {
        let past_mode = (k as f64) > lt;
        if log_weight > LOG_UNDERFLOW {
            let weight = log_weight.exp();
            for i in 0..n {
                out[i] += weight * vec_k[i];
            }
            accumulated += weight;
        } else if past_mode {
            // Beyond the mode the weights only shrink, so once they have
            // dropped out of range again the remaining mass is negligible.
            break;
        }
        if past_mode && 1.0 - accumulated < eps {
            break;
        }
        // vec <- vec P, a row vector times the matrix.
        let mut next = vec![0.0; n];
        for i in 0..n {
            let v = vec_k[i];
            if v == 0.0 {
                continue;
            }
            for j in 0..n {
                next[j] += v * p.get(i, j);
            }
        }
        vec_k = next;
        log_weight += ln_lt - ((k + 1) as f64).ln();
    }

    // Renormalise against the truncated tail.
    let total: f64 = out.iter().sum();
    if total > 0.0 {
        for v in &mut out {
            *v /= total;
        }
    }
    Ok(out)
}

/// Guard against a non-terminating series when `eps` is set below the
/// resolution of double precision.
const MAX_UNIFORMIZATION_TERMS: usize = 1_000_000;

/// Below this a logarithm exponentiates to zero in double precision.
const LOG_UNDERFLOW: f64 = -745.0;

/// The distribution of the number in an M/M/1 queue at time `t`, starting
/// from exactly `n0` customers.
///
/// The state space is truncated well above the point where the stationary
/// geometric tail is negligible, then run through [`uniformization`]. Returns
/// the probability of each state from 0 up to the truncation point.
///
/// # Errors
/// Returns an error if the rates are not positive or the transient solve fails.
pub fn queue_transient_mm1(
    lambda: f64,
    mu: f64,
    n0: usize,
    t: f64,
) -> Result<Vec<f64>, GeomError> {
    if !(lambda > 0.0) || !(mu > 0.0) {
        return Err(GeomError::InvalidArgument("queue_transient_mm1 requires positive rates"));
    }
    if !(t > 0.0) {
        return Err(GeomError::InvalidArgument("queue_transient_mm1 requires t > 0"));
    }
    // The truncation has to hold essentially all the mass at time `t`, and
    // where that mass sits depends on stability. A stable queue is
    // positive-recurrent: it stays near its stationary geometric whatever the
    // horizon, so the reach is set by how many states it takes for rho^n to
    // fall below rounding, plus a fixed diffusive margin around the start.
    // An unstable queue genuinely drifts, at rate lambda - mu, so there the
    // bound has to follow the drift and its diffusive spread.
    let rho = lambda / mu;
    let net = lambda - mu;
    let reach = if rho < 1.0 {
        // rho^n < 1e-18 once n exceeds 18 ln(10) / ln(1/rho).
        let tail = 41.5 / (1.0 / rho).ln().max(1e-3);
        n0 as f64 + tail + 10.0 * (n0 as f64).sqrt() + 20.0
    } else {
        n0 as f64 + net * t + 10.0 * ((lambda + mu) * t).sqrt()
    };
    let cap = (reach.ceil() as usize).clamp(n0 + 20, 2000);

    let n = cap + 1;
    let mut q = Matrix::zeros(n, n);
    for i in 0..n {
        let up = if i + 1 < n { lambda } else { 0.0 };
        let down = if i > 0 { mu } else { 0.0 };
        if up > 0.0 {
            q.set(i, i + 1, up);
        }
        if down > 0.0 {
            q.set(i, i - 1, down);
        }
        q.set(i, i, -(up + down));
    }
    let mut p0 = vec![0.0; n];
    p0[n0.min(cap)] = 1.0;
    uniformization(&q, &p0, t, 1e-12)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exponential draw with the given rate, for the simulators.
    fn exponential(rate: f64) -> impl Fn(&mut Rng) -> f64 {
        move |rng: &mut Rng| -rng.next_f64().max(1e-300).ln() / rate
    }

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
    }

    // -----------------------------------------------------------------
    // The state distribution and the means have to be the same object
    // -----------------------------------------------------------------

    #[test]
    fn mm1_state_distribution_is_the_geometric_law() {
        let (lambda, mu) = (0.6, 1.0);
        let q = mm1(lambda, mu);
        let rho = lambda / mu;
        for n in 0..40 {
            let expected = (1.0 - rho) * rho.powi(n);
            assert!(
                (q.pn(n as usize) - expected).abs() < 1e-12,
                "p_{n} = {} but the geometric law gives {expected}",
                q.pn(n as usize)
            );
        }
    }

    #[test]
    fn closed_form_means_are_the_moments_of_the_reported_distribution() {
        // Summing n p_n has to reproduce L, and sum (n-c)^+ p_n has to
        // reproduce Lq. The means are computed from Erlang C and Little's law
        // rather than from the distribution, so this ties two independent
        // derivations together.
        let cases: Vec<(QueueMetrics, usize, usize)> = vec![
            (mm1(0.6, 1.0), 1, 400),
            (mmc(2.4, 1.0, 3), 3, 400),
            (mmc(7.0, 1.0, 9), 9, 400),
            (mm1k(0.8, 1.0, 12), 1, 13),
            (mmck(3.0, 1.0, 2, 9), 2, 10),
            (mm_inf(4.0, 1.0), usize::MAX, 200),
        ];
        for (q, c, terms) in cases {
            let mut mass = 0.0;
            let mut l = 0.0;
            let mut lq = 0.0;
            for n in 0..terms {
                let p = q.pn(n);
                mass += p;
                l += n as f64 * p;
                if c != usize::MAX {
                    lq += n.saturating_sub(c) as f64 * p;
                }
            }
            assert!((mass - 1.0).abs() < 1e-9, "probabilities summed to {mass}, not one");
            assert!(close(l, q.l, 1e-8), "sum n p_n = {l} but L = {}", q.l);
            if c != usize::MAX {
                assert!(close(lq, q.lq, 1e-8), "sum (n-c)+ p_n = {lq} but Lq = {}", q.lq);
            }
        }
    }

    #[test]
    fn littles_law_holds_for_every_closed_form_model() {
        let models = [
            mm1(0.7, 1.0),
            mmc(3.5, 1.0, 4),
            mm1k(1.3, 1.0, 8),
            mmck(5.0, 1.0, 3, 11),
            mm_inf(2.5, 1.0),
            mg1_pollaczek_khinchine(0.5, 1.0, 0.0),
            mg1_pollaczek_khinchine(0.5, 1.0, 4.0),
        ];
        for q in models {
            assert!(
                littles_law_check(q.l, q.lambda_eff, q.w).abs() < 1e-9,
                "L = {} but lambda W = {}",
                q.l,
                q.lambda_eff * q.w
            );
            assert!(
                littles_law_check(q.lq, q.lambda_eff, q.wq).abs() < 1e-9,
                "Lq = {} but lambda Wq = {}",
                q.lq,
                q.lambda_eff * q.wq
            );
        }
    }

    #[test]
    fn service_time_is_exactly_the_gap_between_sojourn_and_wait() {
        // W - Wq is the mean service time, and L - Lq is the mean number in
        // service, which by Little's law applied to the servers alone is
        // lambda_eff / mu.
        for (lambda, mu, c, k) in
            [(0.7, 1.0, 1, 0usize), (3.5, 1.1, 4, 0), (1.3, 1.0, 1, 8), (5.0, 1.0, 3, 11)]
        {
            let q = if k == 0 { mmc(lambda, mu, c) } else { mmck(lambda, mu, c, k) };
            assert!(close(q.w - q.wq, 1.0 / mu, 1e-9), "W - Wq = {}", q.w - q.wq);
            assert!(
                close(q.l - q.lq, q.lambda_eff / mu, 1e-9),
                "L - Lq = {} but lambda_eff / mu = {}",
                q.l - q.lq,
                q.lambda_eff / mu
            );
        }
    }

    // -----------------------------------------------------------------
    // Models nest inside one another
    // -----------------------------------------------------------------

    #[test]
    fn one_server_specialisations_agree() {
        let (lambda, mu) = (0.55, 1.3);
        let a = mm1(lambda, mu);
        let b = mmc(lambda, mu, 1);
        assert_eq!(a, b);

        let k = mm1k(lambda, mu, 7);
        let k2 = mmck(lambda, mu, 1, 7);
        assert_eq!(k, k2);

        // M/M/1 is M/G/1 with exponential service, whose variance is 1/mu^2.
        let g = mg1_pollaczek_khinchine(lambda, 1.0 / mu, 1.0 / (mu * mu));
        assert!(close(g.lq, a.lq, 1e-9), "P-K Lq = {} but M/M/1 Lq = {}", g.lq, a.lq);
        assert!(close(g.l, a.l, 1e-9), "P-K L = {} but M/M/1 L = {}", g.l, a.l);
        assert!(close(g.wq, a.wq, 1e-9));
        assert!(close(g.p0, a.p0, 1e-9));
    }

    #[test]
    fn finite_capacity_converges_to_the_infinite_queue() {
        let (lambda, mu) = (0.6, 1.0);
        let unbounded = mm1(lambda, mu);
        let mut previous = f64::INFINITY;
        for k in [5usize, 10, 20, 40, 80] {
            let bounded = mm1k(lambda, mu, k);
            let gap = (bounded.l - unbounded.l).abs();
            assert!(gap < previous, "capacity {k} did not improve on the previous truncation");
            // A finite buffer can only hold fewer customers than an unbounded one.
            assert!(bounded.l <= unbounded.l + 1e-12);
            previous = gap;
        }
        assert!(previous < 1e-12, "K = 80 still differs from M/M/1 by {previous}");
    }

    #[test]
    fn deterministic_service_halves_the_exponential_queue() {
        // The Pollaczek-Khinchine formula depends on the service law only
        // through its second moment, so at equal load M/D/1 has exactly half
        // the queue of M/M/1 -- var 0 against var 1/mu^2.
        for lambda in [0.2, 0.5, 0.8, 0.95] {
            let mu = 1.0;
            let md1 = mg1_pollaczek_khinchine(lambda, 1.0 / mu, 0.0);
            let mm1_ = mm1(lambda, mu);
            assert!(
                close(md1.lq, 0.5 * mm1_.lq, 1e-9),
                "at lambda = {lambda}, M/D/1 Lq = {} against half of {}",
                md1.lq,
                mm1_.lq
            );
        }
    }

    #[test]
    fn kingman_is_exact_for_markovian_arrivals_and_service() {
        for (lambda, mu) in [(0.3, 1.0), (0.5, 0.9), (0.85, 1.0)] {
            let approx = gg1_kingman_approx(lambda, mu, 1.0, 1.0);
            let exact = mm1(lambda, mu).wq;
            assert!(close(approx, exact, 1e-12), "Kingman {approx} against exact {exact}");
        }
        // Less variable service than exponential must predict a shorter wait,
        // more variable a longer one.
        let base = gg1_kingman_approx(0.7, 1.0, 1.0, 1.0);
        assert!(gg1_kingman_approx(0.7, 1.0, 1.0, 0.0) < base);
        assert!(gg1_kingman_approx(0.7, 1.0, 1.0, 4.0) > base);
    }

    // -----------------------------------------------------------------
    // Erlang's two formulas
    // -----------------------------------------------------------------

    #[test]
    fn erlang_b_recursion_matches_the_factorial_ratio() {
        // B(c, a) = (a^c / c!) / sum_{k=0}^{c} a^k / k!. The recursion avoids
        // the overflow in that ratio; for small c both are computable and
        // must agree.
        for &a in &[0.5, 1.0, 3.0, 7.5] {
            for c in 1..=15usize {
                let mut terms = Vec::with_capacity(c + 1);
                let mut term = 1.0f64;
                terms.push(term);
                for k in 1..=c {
                    term *= a / k as f64;
                    terms.push(term);
                }
                let direct = terms[c] / terms.iter().sum::<f64>();
                let recursive = erlang_b(a, c);
                assert!(
                    (direct - recursive).abs() < 1e-12,
                    "a = {a}, c = {c}: direct {direct} against recursion {recursive}"
                );
            }
        }
    }

    #[test]
    fn erlang_b_survives_a_trunk_count_that_overflows_the_direct_form() {
        // 200! is past f64::MAX, so the factorial ratio cannot be evaluated
        // at all here. The recursion stays inside [0, 1] at every step.
        let b = erlang_b(180.0, 200);
        assert!(b.is_finite() && b > 0.0 && b < 1.0, "B(200, 180) = {b}");
        // Blocking must still fall as trunks are added.
        assert!(erlang_b(180.0, 220) < b);
    }

    #[test]
    fn erlang_b_is_monotone_in_both_arguments() {
        let a = 4.0;
        let mut previous = 1.0;
        for c in 1..=30usize {
            let b = erlang_b(a, c);
            assert!(b < previous, "adding a trunk did not reduce blocking at c = {c}");
            assert!((0.0..=1.0).contains(&b));
            previous = b;
        }
        let c = 6;
        let mut previous = 0.0;
        for step in 1..=20 {
            let b = erlang_b(step as f64 * 0.5, c);
            assert!(b > previous, "more load did not raise blocking at step {step}");
            previous = b;
        }
    }

    #[test]
    fn erlang_c_is_the_tail_mass_of_the_mmc_distribution() {
        // C(c, a) is by definition the probability an arrival finds all c
        // servers busy, which is sum_{n >= c} p_n. Poisson arrivals see time
        // averages, so the two coincide.
        for (lambda, mu, c) in [(0.4, 1.0, 1usize), (2.4, 1.0, 3), (7.0, 1.0, 9), (11.0, 2.0, 7)] {
            let q = mmc(lambda, mu, c);
            let tail: f64 = (c..3000).map(|n| q.pn(n)).sum();
            let formula = erlang_c(lambda / mu, c);
            assert!(
                close(tail, formula, 1e-9),
                "c = {c}: tail mass {tail} against Erlang C {formula}"
            );
        }
    }

    #[test]
    fn erlang_c_exceeds_erlang_b_and_reduces_to_rho_for_one_server() {
        for &a in &[0.3, 1.0, 4.0] {
            // With a single server every arrival that finds it busy waits,
            // and the server is busy a fraction rho of the time.
            if a < 1.0 {
                assert!(close(erlang_c(a, 1), a, 1e-12), "C(1, {a}) = {}", erlang_c(a, 1));
            }
            for c in 1..=12usize {
                if a / c as f64 >= 1.0 {
                    continue;
                }
                let (b, cc) = (erlang_b(a, c), erlang_c(a, c));
                // Making a blocked customer wait rather than turning them away
                // can only increase the chance of finding the system full.
                assert!(cc >= b - 1e-12, "c = {c}, a = {a}: C = {cc} below B = {b}");
                assert!((0.0..=1.0).contains(&cc));
            }
        }
    }

    #[test]
    fn inverse_capacity_returns_the_smallest_sufficient_trunk_count() {
        for &(load, target) in &[(1.0, 0.01), (5.0, 0.02), (20.0, 0.001), (0.5, 0.5)] {
            let c = erlang_b_inverse_capacity(load, target);
            assert!(erlang_b(load, c) <= target, "c = {c} does not meet the target");
            assert!(
                c == 0 || erlang_b(load, c - 1) > target,
                "c = {c} is not minimal: c - 1 already meets the target"
            );
        }
    }

    #[test]
    fn mm_inf_is_poisson_and_nobody_waits() {
        let (lambda, mu) = (3.5, 0.7);
        let a = lambda / mu;
        let q = mm_inf(lambda, mu);
        assert_eq!(q.lq, 0.0);
        assert_eq!(q.wq, 0.0);
        assert!(close(q.l, a, 1e-12));
        assert!(close(q.w, 1.0 / mu, 1e-12));
        let mut term = (-a).exp();
        for n in 0..60 {
            assert!((q.pn(n) - term).abs() < 1e-12, "p_{n} is not the Poisson mass");
            term *= a / (n + 1) as f64;
        }
    }

    #[test]
    fn saturated_queues_report_infinite_means() {
        let q = mmc(2.0, 1.0, 2);
        assert!(q.l.is_infinite() && q.w.is_infinite());
        assert!(q.rho >= 1.0);
        assert!(gg1_kingman_approx(1.0, 1.0, 1.0, 1.0).is_infinite());
        assert!(mg1_pollaczek_khinchine(1.0, 1.0, 0.5).lq.is_infinite());
        // A finite buffer stays finite at any load: it simply blocks more.
        let bounded = mm1k(5.0, 1.0, 4);
        assert!(bounded.l.is_finite() && bounded.l <= 4.0);
        assert!(bounded.rho < 1.0);
    }

    // -----------------------------------------------------------------
    // Networks
    // -----------------------------------------------------------------

    #[test]
    fn jackson_rates_solve_the_traffic_equations_and_conserve_flow() {
        // Two nodes with feedback: half of node 0's output goes to node 1,
        // a quarter of node 1's comes back.
        let routing = Matrix::from_rows(&[&[0.0, 0.5], &[0.25, 0.0]]).unwrap();
        let external = [1.0, 0.5];
        let service = [4.0, 3.0];
        let servers = [1usize, 1];
        let out = jackson_network(&routing, &external, &service, &servers).unwrap();

        let rates: Vec<f64> = out.iter().map(|q| q.lambda_eff).collect();
        for j in 0..2 {
            let inflow: f64 =
                external[j] + (0..2).map(|i| rates[i] * routing.get(i, j)).sum::<f64>();
            assert!(
                close(rates[j], inflow, 1e-9),
                "node {j}: rate {} against inflow {inflow}",
                rates[j]
            );
        }
        // Flow conservation for the network as a whole: everything that
        // enters must leave.
        let entered: f64 = external.iter().sum();
        let departed: f64 = (0..2)
            .map(|i| rates[i] * (1.0 - (0..2).map(|j| routing.get(i, j)).sum::<f64>()))
            .sum();
        assert!(close(entered, departed, 1e-9), "{entered} in against {departed} out");

        // Jackson's theorem: each node is its own M/M/c at its total rate.
        for (j, q) in out.iter().enumerate() {
            assert_eq!(*q, mmc(rates[j], service[j], servers[j]));
        }
    }

    #[test]
    fn a_jackson_tandem_passes_its_arrival_rate_straight_through() {
        // Everything entering node 0 goes to node 1 and then leaves, so both
        // nodes see the same rate and the network is two independent M/M/1s.
        let routing = Matrix::from_rows(&[&[0.0, 1.0], &[0.0, 0.0]]).unwrap();
        let out = jackson_network(&routing, &[2.0, 0.0], &[5.0, 3.0], &[1, 1]).unwrap();
        assert!(close(out[0].lambda_eff, 2.0, 1e-12));
        assert!(close(out[1].lambda_eff, 2.0, 1e-12));
        // Total sojourn through the network is the sum of the two stages.
        let total = out[0].w + out[1].w;
        assert!(close(total, 1.0 / (5.0 - 2.0) + 1.0 / (3.0 - 2.0), 1e-12), "total W = {total}");
    }

    #[test]
    fn jackson_rejects_malformed_input() {
        let square = Matrix::from_rows(&[&[0.0, 0.5], &[0.25, 0.0]]).unwrap();
        assert!(jackson_network(&square, &[1.0], &[1.0], &[1]).is_err());
        assert!(jackson_network(&square, &[1.0, 1.0], &[1.0], &[1, 1]).is_err());
        let overfull = Matrix::from_rows(&[&[0.7, 0.7], &[0.0, 0.0]]).unwrap();
        assert!(jackson_network(&overfull, &[1.0, 1.0], &[9.0, 9.0], &[1, 1]).is_err());
        let negative = Matrix::from_rows(&[&[0.0, -0.5], &[0.0, 0.0]]).unwrap();
        assert!(jackson_network(&negative, &[1.0, 1.0], &[9.0, 9.0], &[1, 1]).is_err());
    }

    // -----------------------------------------------------------------
    // Simulation against the closed forms
    // -----------------------------------------------------------------

    #[test]
    fn simulated_mm1_reproduces_its_closed_form() {
        let (lambda, mu) = (0.6, 1.0);
        let mut rng = Rng::new(0x51DE_0001);
        let sim =
            queue_simulate(&exponential(lambda), &exponential(mu), 1, 400_000.0, &mut rng);
        let exact = mm1(lambda, mu);
        assert!(sim.served > 200_000, "only {} customers served", sim.served);
        assert!(close(sim.w, exact.w, 0.03), "W {} against {}", sim.w, exact.w);
        assert!(close(sim.wq, exact.wq, 0.04), "Wq {} against {}", sim.wq, exact.wq);
        assert!(close(sim.l, exact.l, 0.03), "L {} against {}", sim.l, exact.l);
        assert!(close(sim.lq, exact.lq, 0.05), "Lq {} against {}", sim.lq, exact.lq);
        assert!(close(sim.rho, exact.rho, 0.02), "rho {} against {}", sim.rho, exact.rho);
    }

    #[test]
    fn the_simulated_time_average_and_customer_average_satisfy_littles_law() {
        // `l` comes from integrating the sample path and `w` from averaging
        // over customers; nothing in the simulator derives one from the other,
        // so their agreement is evidence rather than arithmetic.
        let mut rng = Rng::new(0x51DE_0002);
        let sim = queue_simulate(&exponential(1.4), &exponential(2.0), 2, 200_000.0, &mut rng);
        let residual = littles_law_check(sim.l, sim.lambda_eff, sim.w);
        assert!(
            residual.abs() < 0.02 * sim.l,
            "L = {} but lambda W = {}",
            sim.l,
            sim.lambda_eff * sim.w
        );
        let residual_q = littles_law_check(sim.lq, sim.lambda_eff, sim.wq);
        assert!(residual_q.abs() < 0.02 * sim.l, "Lq = {} against lambda Wq", sim.lq);
    }

    #[test]
    fn simulated_md1_matches_pollaczek_khinchine() {
        // Deterministic service is the case the exponential formula gets
        // wrong by a factor of two, so this separates P-K from M/M/1.
        let (lambda, service) = (0.6, 1.0);
        let mut rng = Rng::new(0x51DE_0003);
        let sim = queue_simulate(
            &exponential(lambda),
            &move |_: &mut Rng| service,
            1,
            400_000.0,
            &mut rng,
        );
        let pk = mg1_pollaczek_khinchine(lambda, service, 0.0);
        let exponential_service = mm1(lambda, 1.0 / service);
        assert!(close(sim.wq, pk.wq, 0.04), "simulated Wq {} against P-K {}", sim.wq, pk.wq);
        // And it is genuinely distinguishable from the exponential answer.
        assert!(
            (sim.wq - exponential_service.wq).abs() > 0.3 * exponential_service.wq,
            "M/D/1 came out indistinguishable from M/M/1"
        );
    }

    #[test]
    fn simulated_mmc_reproduces_the_multi_server_closed_form() {
        let (lambda, mu, c) = (2.4, 1.0, 3usize);
        let mut rng = Rng::new(0x51DE_0004);
        let sim = queue_simulate(&exponential(lambda), &exponential(mu), c, 150_000.0, &mut rng);
        let exact = mmc(lambda, mu, c);
        assert!(close(sim.w, exact.w, 0.04), "W {} against {}", sim.w, exact.w);
        assert!(close(sim.lq, exact.lq, 0.07), "Lq {} against {}", sim.lq, exact.lq);
        assert!(close(sim.rho, exact.rho, 0.02), "rho {} against {}", sim.rho, exact.rho);
    }

    #[test]
    fn an_idle_horizon_produces_an_empty_but_well_formed_result() {
        let mut rng = Rng::new(7);
        // Arrivals every 100 time units, horizon 1: nobody shows up.
        let sim = queue_simulate(&|_: &mut Rng| 100.0, &exponential(1.0), 1, 1.0, &mut rng);
        assert_eq!(sim.served, 0);
        assert_eq!(sim.l, 0.0);
        assert_eq!(sim.rho, 0.0);
    }

    // -----------------------------------------------------------------
    // Priorities
    // -----------------------------------------------------------------

    #[test]
    fn non_preemptive_priority_matches_the_cobham_formula() {
        // With one server and equal service rates, class k's mean wait is
        // W0 / ((1 - sigma_{k-1})(1 - sigma_k)) where W0 = sum_j lambda_j
        // E[S_j^2] / 2 and sigma_k is the load of classes 0..=k.
        let lambdas = [0.2, 0.3, 0.15];
        let mu = 1.0;
        let mus = [mu; 3];
        let mut rng = Rng::new(0x51DE_0005);
        let sim = priority_queue_simulate(&lambdas, &mus, 1, 600_000.0, &mut rng);

        let w0: f64 = lambdas.iter().map(|&l| l * 2.0 / (mu * mu) / 2.0).sum();
        let mut sigma_prev = 0.0;
        for k in 0..3 {
            let sigma = sigma_prev + lambdas[k] / mu;
            let expected = w0 / ((1.0 - sigma_prev) * (1.0 - sigma));
            assert!(
                close(sim[k].wq, expected, 0.06),
                "class {k}: simulated Wq {} against Cobham {expected}",
                sim[k].wq
            );
            sigma_prev = sigma;
        }
    }

    #[test]
    fn priority_ordering_shortens_the_top_class_and_lengthens_the_bottom() {
        let lambdas = [0.25, 0.25, 0.2];
        let mus = [1.0; 3];
        let mut rng = Rng::new(0x51DE_0006);
        let sim = priority_queue_simulate(&lambdas, &mus, 1, 400_000.0, &mut rng);
        assert!(sim[0].wq < sim[1].wq, "class 0 did not beat class 1");
        assert!(sim[1].wq < sim[2].wq, "class 1 did not beat class 2");

        // Kleinrock's conservation law: sum_k rho_k Wq_k does not depend on
        // the order jobs are taken in, only on the work arriving. Compare
        // against plain first-come-first-served at the pooled rate.
        let total: f64 = lambdas.iter().sum();
        let fifo = mm1(total, 1.0);
        let weighted: f64 = (0..3).map(|k| lambdas[k] / mus[k] * sim[k].wq).sum();
        let reference = total / 1.0 * fifo.wq;
        assert!(
            close(weighted, reference, 0.06),
            "priority weighted wait {weighted} against FIFO {reference}"
        );
    }

    // -----------------------------------------------------------------
    // Continuous-time chains
    // -----------------------------------------------------------------

    fn birth_death_generator(lambda: f64, mu: f64, k: usize) -> Matrix {
        let n = k + 1;
        let mut q = Matrix::zeros(n, n);
        for i in 0..n {
            let up = if i + 1 < n { lambda } else { 0.0 };
            let down = if i > 0 { mu } else { 0.0 };
            if up > 0.0 {
                q.set(i, i + 1, up);
            }
            if down > 0.0 {
                q.set(i, i - 1, down);
            }
            q.set(i, i, -(up + down));
        }
        q
    }

    #[test]
    fn ctmc_rejects_matrices_that_are_not_generators() {
        assert!(Ctmc::new(Matrix::zeros(2, 3)).is_err());
        let bad_row = Matrix::from_rows(&[&[-1.0, 1.0], &[1.0, 0.0]]).unwrap();
        assert!(Ctmc::new(bad_row).is_err());
        let negative_rate = Matrix::from_rows(&[&[1.0, -1.0], &[1.0, -1.0]]).unwrap();
        assert!(Ctmc::new(negative_rate).is_err());
        assert!(Ctmc::new(birth_death_generator(1.0, 2.0, 3)).is_ok());
    }

    #[test]
    fn ctmc_stationary_satisfies_global_balance() {
        let chain = Ctmc::new(birth_death_generator(1.5, 2.0, 6)).unwrap();
        let pi = chain.stationary().unwrap();
        assert!(close(pi.iter().sum::<f64>(), 1.0, 1e-12));
        assert!(pi.iter().all(|&p| p >= -1e-12));
        // pi Q = 0, column by column.
        for j in 0..chain.n() {
            let flow: f64 = (0..chain.n()).map(|i| pi[i] * chain.q.get(i, j)).sum();
            assert!(flow.abs() < 1e-10, "state {j} has net probability flow {flow}");
        }
    }

    #[test]
    fn the_ctmc_of_a_finite_queue_has_the_analytic_stationary_distribution() {
        // The generator is written from the queue's transition rates alone;
        // mm1k derives its distribution from the birth-death product form.
        // Two separate routes to the same numbers.
        let (lambda, mu, k) = (1.5, 2.0, 9usize);
        let chain = Ctmc::new(birth_death_generator(lambda, mu, k)).unwrap();
        let pi = chain.stationary().unwrap();
        let analytic = mm1k(lambda, mu, k);
        for n in 0..=k {
            assert!(
                (pi[n] - analytic.pn(n)).abs() < 1e-10,
                "state {n}: solver {} against product form {}",
                pi[n],
                analytic.pn(n)
            );
        }
    }

    #[test]
    fn stationary_is_the_jump_chain_reweighted_by_holding_time() {
        // A continuous-time chain spends time in proportion to how often it
        // visits a state times how long it stays: pi_i is proportional to
        // nu_i h_i, where nu is the embedded chain's stationary law.
        let chain = Ctmc::new(birth_death_generator(1.0, 1.7, 5)).unwrap();
        let pi = chain.stationary().unwrap();
        let nu = chain.embedded_chain().unwrap().stationary();
        let h = chain.mean_holding_times();
        let unnormalised: Vec<f64> = nu.iter().zip(&h).map(|(&v, &t)| v * t).collect();
        let total: f64 = unnormalised.iter().sum();
        for i in 0..chain.n() {
            let predicted = unnormalised[i] / total;
            assert!(
                (pi[i] - predicted).abs() < 1e-9,
                "state {i}: {} against reweighted jump chain {predicted}",
                pi[i]
            );
        }
    }

    #[test]
    fn embedded_chain_is_stochastic_and_holds_no_self_transitions() {
        let chain = Ctmc::new(birth_death_generator(1.0, 1.7, 4)).unwrap();
        let jump = chain.embedded_chain().unwrap();
        for i in 0..chain.n() {
            let row: f64 = (0..chain.n()).map(|j| jump.p.get(i, j)).sum();
            assert!((row - 1.0).abs() < 1e-12, "row {i} sums to {row}");
            assert_eq!(jump.p.get(i, i), 0.0, "state {i} has a spurious self-loop");
        }
        // An absorbing state has no exit rate, so the jump chain makes it a
        // self-loop rather than an invalid all-zero row.
        let absorbing = Matrix::from_rows(&[&[-1.0, 1.0], &[0.0, 0.0]]).unwrap();
        let jump = Ctmc::new(absorbing).unwrap().embedded_chain().unwrap();
        assert_eq!(jump.p.get(1, 1), 1.0);
    }

    #[test]
    fn first_passage_up_a_pure_birth_chain_is_the_sum_of_holding_times() {
        // With no downward rates the walk can only climb, so the mean time
        // from 0 to k is exactly the sum of the mean holding times below k.
        let n = 6usize;
        let rates = [1.0, 2.0, 0.5, 3.0, 1.5];
        let mut q = Matrix::zeros(n, n);
        for i in 0..n - 1 {
            q.set(i, i + 1, rates[i]);
            q.set(i, i, -rates[i]);
        }
        let chain = Ctmc::new(q).unwrap();
        let expected: f64 = rates.iter().map(|r| 1.0 / r).sum();
        let got = chain.first_passage(0, n - 1).unwrap();
        assert!(close(got, expected, 1e-9), "{got} against {expected}");
        assert_eq!(chain.first_passage(3, 3).unwrap(), 0.0);
        assert!(chain.first_passage(9, 0).is_err());
        // Climbing only: the target below the start is never reached.
        assert!(chain.first_passage(4, 1).unwrap().is_infinite());
    }

    #[test]
    fn first_passage_matches_a_long_simulation() {
        let chain = Ctmc::new(birth_death_generator(1.0, 1.5, 4)).unwrap();
        let target = 4usize;
        let predicted = chain.first_passage(0, target).unwrap();

        let mut rng = Rng::new(0x51DE_0007);
        let trials = 4000;
        let mut total = 0.0;
        for _ in 0..trials {
            // Run far past the predicted mean so truncation is negligible.
            let path = chain.simulate(0, predicted * 60.0, &mut rng);
            let hit = path.iter().find(|&&(_, s)| s == target).map(|&(t, _)| t);
            total += hit.unwrap_or(predicted * 60.0);
        }
        let measured = total / trials as f64;
        assert!(close(measured, predicted, 0.06), "simulated {measured} against {predicted}");
    }

    #[test]
    fn simulated_occupancy_matches_the_stationary_distribution() {
        let chain = Ctmc::new(birth_death_generator(1.2, 1.8, 5)).unwrap();
        let pi = chain.stationary().unwrap();
        let horizon = 300_000.0;
        let mut rng = Rng::new(0x51DE_0008);
        let path = chain.simulate(0, horizon, &mut rng);

        let mut time_in = vec![0.0; chain.n()];
        for w in path.windows(2) {
            time_in[w[0].1] += w[1].0 - w[0].0;
        }
        if let Some(&(t, s)) = path.last() {
            time_in[s] += horizon - t;
        }
        for i in 0..chain.n() {
            let fraction = time_in[i] / horizon;
            assert!(
                (fraction - pi[i]).abs() < 0.01,
                "state {i}: occupied {fraction} of the time against pi = {}",
                pi[i]
            );
        }
    }

    // -----------------------------------------------------------------
    // Uniformization
    // -----------------------------------------------------------------

    #[test]
    fn uniformization_matches_the_two_state_closed_form() {
        // For Q = [[-a, a], [b, -b]] started in state 0,
        // p_0(t) = b/(a+b) + a/(a+b) e^{-(a+b)t}.
        let (a, b) = (0.7, 1.3);
        let q = Matrix::from_rows(&[&[-a, a], &[b, -b]]).unwrap();
        for &t in &[0.05, 0.5, 2.0, 10.0] {
            let p = uniformization(&q, &[1.0, 0.0], t, 1e-14).unwrap();
            let exact = b / (a + b) + a / (a + b) * (-(a + b) * t).exp();
            assert!(
                (p[0] - exact).abs() < 1e-9,
                "t = {t}: uniformization {} against exp(Qt) {exact}",
                p[0]
            );
            assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn uniformization_relaxes_to_the_stationary_distribution() {
        let chain = Ctmc::new(birth_death_generator(1.1, 1.9, 6)).unwrap();
        let pi = chain.stationary().unwrap();
        let n = chain.n();
        let mut start = vec![0.0; n];
        start[n - 1] = 1.0;

        let mut previous = f64::INFINITY;
        for &t in &[0.5, 2.0, 8.0, 40.0, 120.0] {
            let p = uniformization(&chain.q, &start, t, 1e-14).unwrap();
            assert!(p.iter().all(|&x| x >= -1e-12), "a probability went negative at t = {t}");
            assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            let distance: f64 =
                p.iter().zip(&pi).map(|(a, b)| (a - b).abs()).sum::<f64>() / 2.0;
            assert!(distance < previous, "distance to stationary grew at t = {t}");
            previous = distance;
        }
        assert!(previous < 1e-12, "still {previous} away from stationary at t = 120");
    }

    #[test]
    fn uniformization_is_stable_on_a_stiff_generator() {
        // Rates three orders of magnitude apart. A truncated Taylor series
        // for exp(Qt) would produce negative probabilities here; every term
        // of the Poisson mixture is non-negative by construction.
        let q = Matrix::from_rows(&[
            &[-1000.0, 1000.0, 0.0],
            &[0.0, -1000.5, 1000.5],
            &[0.5, 0.0, -0.5],
        ])
        .unwrap();
        let p = uniformization(&q, &[1.0, 0.0, 0.0], 1.0, 1e-12).unwrap();
        assert!(p.iter().all(|&x| x >= 0.0), "negative probability: {p:?}");
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        let pi = Ctmc::new(q).unwrap().stationary().unwrap();
        // At t = 1 the fast pair has long since equilibrated against the slow
        // return, so the answer should already be near stationary.
        let distance: f64 = p.iter().zip(&pi).map(|(a, b)| (a - b).abs()).sum::<f64>() / 2.0;
        assert!(distance < 0.35, "distance {distance} is implausibly large");
    }

    #[test]
    fn uniformization_survives_a_horizon_whose_poisson_mass_underflows() {
        // At Lt beyond about 745 the leading Poisson weight e^{-Lt} is zero in
        // double precision, and the weights only become representable again
        // some thousands of terms later, near the mode. Carrying them as
        // logarithms is what makes that window reachable; forming them
        // directly gives either all zeros or an overflow to infinity.
        let chain = Ctmc::new(birth_death_generator(3.0, 4.0, 8)).unwrap();
        let pi = chain.stationary().unwrap();
        let n = chain.n();
        let mut start = vec![0.0; n];
        start[n - 1] = 1.0;

        // The uniformization rate is the largest exit rate in the generator.
        let rate = (0..n).map(|i| -chain.q.get(i, i)).fold(0.0f64, f64::max);
        for &t in &[200.0f64, 800.0, 4000.0] {
            assert!(rate * t > 745.0, "Lt = {} is not past the underflow point", rate * t);
            let p = uniformization(&chain.q, &start, t, 1e-14).unwrap();
            assert!(p.iter().all(|v| v.is_finite()), "t = {t} produced {p:?}");
            assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12, "t = {t} is not a distribution");
            assert!(p.iter().all(|&v| v >= -1e-15));
            // Long past the relaxation time the answer is the stationary law.
            for i in 0..n {
                assert!(
                    (p[i] - pi[i]).abs() < 1e-9,
                    "t = {t}, state {i}: {} against stationary {}",
                    p[i],
                    pi[i]
                );
            }
        }
    }

    #[test]
    fn uniformization_rejects_malformed_input() {
        let q = birth_death_generator(1.0, 1.0, 2);
        assert!(uniformization(&q, &[1.0, 0.0], 1.0, 1e-9).is_err());
        assert!(uniformization(&q, &[0.5, 0.5, 0.5], 1.0, 1e-9).is_err());
        assert!(uniformization(&q, &[1.0, 0.0, 0.0], -1.0, 1e-9).is_err());
        assert!(uniformization(&q, &[1.0, 0.0, 0.0], 1.0, 0.0).is_err());
        // A generator with no transitions leaves the initial law untouched.
        let frozen = Matrix::zeros(2, 2);
        assert_eq!(uniformization(&frozen, &[0.3, 0.7], 5.0, 1e-9).unwrap(), vec![0.3, 0.7]);
    }

    #[test]
    fn transient_mm1_starts_where_it_was_put_and_relaxes_to_the_geometric() {
        let (lambda, mu, n0) = (1.0, 2.0, 3usize);
        let short = queue_transient_mm1(lambda, mu, n0, 1e-6).unwrap();
        assert!(short[n0] > 0.999, "at t = 1e-6 the mass had already left state {n0}");

        let long = queue_transient_mm1(lambda, mu, n0, 400.0).unwrap();
        let stationary = mm1(lambda, mu);
        for n in 0..25 {
            assert!(
                (long[n] - stationary.pn(n)).abs() < 1e-8,
                "state {n}: transient {} against stationary {}",
                long[n],
                stationary.pn(n)
            );
        }
        // The mean has to move monotonically from n0 down to rho/(1-rho).
        let mean = |p: &[f64]| p.iter().enumerate().map(|(n, &q)| n as f64 * q).sum::<f64>();
        let mut previous = n0 as f64;
        for &t in &[0.1, 0.5, 2.0, 10.0, 400.0] {
            let m = mean(&queue_transient_mm1(lambda, mu, n0, t).unwrap());
            assert!(m < previous + 1e-9, "the mean rose at t = {t}");
            previous = m;
        }
        assert!(close(previous, stationary.l, 1e-6), "settled at {previous}, not {}", stationary.l);
    }

    #[test]
    fn transient_mm1_rejects_bad_arguments() {
        assert!(queue_transient_mm1(0.0, 1.0, 0, 1.0).is_err());
        assert!(queue_transient_mm1(1.0, 0.0, 0, 1.0).is_err());
        assert!(queue_transient_mm1(1.0, 1.0, 0, 0.0).is_err());
    }
}
