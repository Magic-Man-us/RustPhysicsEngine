//! Network models and scheduling: project planning, flows on networks, and
//! the sequencing rules that provably optimise a stated objective.
//!
//! Two threads run through this module. The first is that several graph
//! problems are linear programs in disguise, and their constraint matrices
//! are totally unimodular, so the linear relaxation is automatically
//! integral. Shortest path and maximum flow both have this property, which is
//! why they can be solved by combinatorial algorithms *and* by a general
//! linear programming solver with the same answer. Having both is worth the
//! duplication: the graph module's algorithms are far faster, and the linear
//! programs are an independent check on them.
//!
//! The second is that scheduling is a subject of exact greedy rules rather
//! than heuristics. Sorting by processing time minimises mean flow time;
//! sorting by due date minimises maximum lateness; Moore and Hodgson's rule
//! minimises the *number* of late jobs; Johnson's rule minimises makespan on
//! two machines. Each is provably optimal for its own objective and provably
//! not for the others -- shortest-processing-time can make a job
//! catastrophically late while minimising the average -- so the objective
//! must be chosen before the rule. The tests check each rule against
//! exhaustive enumeration of every permutation, on the objective it claims
//! and on nothing else.

use crate::error::GeomError;
use crate::graph::core::Graph;
use crate::linalg::matrix::Matrix;
use crate::optimization::lp::{simplex, Cmp, LpProblem, LpResult};

/// Values within this of each other are treated as equal.
const TOL: f64 = 1e-9;

// ---------------------------------------------------------------------------
// Flows on networks
// ---------------------------------------------------------------------------

/// The transshipment problem: ship from sources to sinks through intermediate
/// nodes at least cost.
///
/// `supply[i]` is positive at a source, negative at a sink, and zero at a pure
/// transshipment node; the entries must sum to zero. `arcs` lists
/// `(from, to, unit cost, capacity)`.
///
/// Generalises the transportation problem by allowing goods to pass through a
/// node rather than only from a source directly to a sink, which is what makes
/// it a network rather than a bipartite matching.
///
/// # Errors
/// Returns an error if an arc names a node out of range or the supplies do not
/// balance.
pub fn transshipment(
    supply: &[f64],
    arcs: &[(usize, usize, f64, f64)],
) -> Result<LpResult, GeomError> {
    let n = supply.len();
    if n == 0 || arcs.is_empty() {
        return Err(GeomError::InvalidArgument("transshipment needs nodes and arcs"));
    }
    if arcs.iter().any(|&(a, b, _, cap)| a >= n || b >= n || cap < 0.0) {
        return Err(GeomError::InvalidArgument("transshipment: bad arc"));
    }
    if supply.iter().sum::<f64>().abs() > 1e-7 {
        return Err(GeomError::InvalidArgument("transshipment: supplies must sum to zero"));
    }

    // One variable per arc; one conservation row per node.
    let mut a = Matrix::zeros(n, arcs.len());
    for (k, &(from, to, _, _)) in arcs.iter().enumerate() {
        a.set(from, k, 1.0);
        a.set(to, k, -1.0);
    }
    let p = LpProblem {
        c: arcs.iter().map(|&(_, _, cost, _)| cost).collect(),
        a,
        b: supply.to_vec(),
        constraint_types: vec![Cmp::Eq; n],
        bounds: arcs.iter().map(|&(_, _, _, cap)| (0.0, cap)).collect(),
        maximize: false,
    };
    simplex(&p)
}

/// The length of a shortest path, computed as a linear program.
///
/// The dual of the shortest path problem asks for node potentials that
/// maximise the gap between source and target while no arc rises by more than
/// its length -- so the answer comes out of a linear program whose constraint
/// matrix is a node-arc incidence matrix, which is totally unimodular.
///
/// Its purpose is to check the graph module's Dijkstra against a completely
/// different method. Slower by a wide margin, and worth it only as
/// verification.
///
/// # Errors
/// Returns an error if the endpoints are out of range, or the graph has a
/// negative-length arc, where the linear program is unbounded rather than
/// merely wrong.
pub fn shortest_path_lp_check(g: &Graph, s: usize, t: usize) -> Result<Option<f64>, GeomError> {
    let n = g.n;
    if s >= n || t >= n {
        return Err(GeomError::InvalidArgument("shortest_path_lp_check: endpoint out of range"));
    }
    if s == t {
        return Ok(Some(0.0));
    }
    let arcs: Vec<(usize, usize, f64)> = directed_arcs(g);
    if arcs.iter().any(|&(_, _, w)| w < 0.0) {
        return Err(GeomError::InvalidArgument("shortest_path_lp_check: negative arc length"));
    }

    // Send one unit from s to t at least cost: the flow formulation.
    let mut a = Matrix::zeros(n, arcs.len());
    for (k, &(from, to, _)) in arcs.iter().enumerate() {
        a.set(from, k, 1.0);
        a.set(to, k, -1.0);
    }
    let mut b = vec![0.0; n];
    b[s] = 1.0;
    b[t] = -1.0;

    let p = LpProblem {
        c: arcs.iter().map(|&(_, _, w)| w).collect(),
        a,
        b,
        constraint_types: vec![Cmp::Eq; n],
        bounds: vec![(0.0, f64::INFINITY); arcs.len()],
        maximize: false,
    };
    Ok(simplex(&p)?.objective())
}

/// The value of a maximum flow, computed as a linear program.
///
/// Maximises the net outflow from the source subject to conservation at every
/// other node and each arc's capacity. Like the shortest path formulation this
/// exists to check the graph module's combinatorial algorithms rather than to
/// replace them.
///
/// # Errors
/// Returns an error if the endpoints are out of range or coincide.
pub fn max_flow_lp_check(g: &Graph, s: usize, t: usize) -> Result<Option<f64>, GeomError> {
    let n = g.n;
    if s >= n || t >= n || s == t {
        return Err(GeomError::InvalidArgument("max_flow_lp_check: bad endpoints"));
    }
    let arcs = directed_arcs(g);

    // Conservation at every node but the source and the sink; the objective is
    // the net flow leaving the source.
    let rows: Vec<usize> = (0..n).filter(|&v| v != s && v != t).collect();
    let mut a = Matrix::zeros(rows.len(), arcs.len());
    for (r, &v) in rows.iter().enumerate() {
        for (k, &(from, to, _)) in arcs.iter().enumerate() {
            if from == v {
                a.set(r, k, 1.0);
            }
            if to == v {
                a.set(r, k, a.get(r, k) - 1.0);
            }
        }
    }
    let c: Vec<f64> = arcs
        .iter()
        .map(|&(from, to, _)| {
            // Flow out of the source counts positively, flow back into it
            // negatively; everything else is invisible to the objective.
            f64::from(i8::from(from == s)) - f64::from(i8::from(to == s))
        })
        .collect();

    let p = LpProblem {
        c,
        a,
        b: vec![0.0; rows.len()],
        constraint_types: vec![Cmp::Eq; rows.len()],
        bounds: arcs.iter().map(|&(_, _, cap)| (0.0, cap)).collect(),
        maximize: true,
    };
    Ok(simplex(&p)?.objective())
}

/// The directed arcs of a graph, with an undirected edge appearing in both
/// directions.
fn directed_arcs(g: &Graph) -> Vec<(usize, usize, f64)> {
    let mut arcs = Vec::new();
    for u in 0..g.n {
        for &(v, w) in &g.adj[u] {
            arcs.push((u, v, w));
        }
    }
    arcs
}

/// A minimum-cost flow by the network simplex, expressed through the general
/// simplex method.
///
/// `arcs` are `(from, to, unit cost, capacity)` and `balance[i]` the net
/// supply at node `i`, summing to zero. The genuine network simplex maintains
/// a spanning tree basis and pivots in `O(m)` per step rather than solving a
/// linear system; this routes the same problem through the general solver,
/// which is correct and slower, and is named "lite" for that reason.
///
/// # Errors
/// Returns an error under the same conditions as [`transshipment`].
pub fn network_simplex_lite(
    balance: &[f64],
    arcs: &[(usize, usize, f64, f64)],
) -> Result<LpResult, GeomError> {
    transshipment(balance, arcs)
}

// ---------------------------------------------------------------------------
// Project scheduling
// ---------------------------------------------------------------------------

/// The four schedule times of one task: earliest start, earliest finish,
/// latest start, latest finish.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TaskTimes {
    /// Earliest the task can begin, given its predecessors.
    pub early_start: f64,
    /// Earliest it can end.
    pub early_finish: f64,
    /// Latest it can begin without delaying the project.
    pub late_start: f64,
    /// Latest it can end.
    pub late_finish: f64,
}

impl TaskTimes {
    /// How far the task can slip without delaying the project.
    ///
    /// Zero exactly on the critical path, which is what defines it.
    #[must_use]
    pub fn slack(&self) -> f64 {
        self.late_start - self.early_start
    }
}

/// The critical path method: the shortest possible project duration, which
/// tasks cannot slip, and every task's four schedule times.
///
/// `tasks[i]` is `(duration, predecessors)`. Returns
/// `(duration, critical task indices, times)`.
///
/// The critical path is the longest path through the precedence graph, and the
/// project cannot finish sooner than that however many resources are thrown at
/// it -- which is the point of computing it. A task is critical exactly when
/// its slack is zero, so shortening a non-critical task buys nothing at all.
///
/// # Errors
/// Returns an error if a predecessor is out of range or the precedences
/// contain a cycle, which makes the project unschedulable.
pub fn critical_path_method(
    tasks: &[(f64, Vec<usize>)],
) -> Result<(f64, Vec<usize>, Vec<TaskTimes>), GeomError> {
    let n = tasks.len();
    if n == 0 {
        return Err(GeomError::Empty);
    }
    if tasks.iter().any(|(d, preds)| *d < 0.0 || preds.iter().any(|&p| p >= n)) {
        return Err(GeomError::InvalidArgument("critical_path_method: bad task"));
    }

    // Topological order by Kahn's algorithm; a leftover node means a cycle.
    let mut indegree = vec![0usize; n];
    let mut successors = vec![Vec::new(); n];
    for (i, (_, preds)) in tasks.iter().enumerate() {
        indegree[i] = preds.len();
        for &p in preds {
            successors[p].push(i);
        }
    }
    let mut order = Vec::with_capacity(n);
    let mut ready: Vec<usize> = (0..n).filter(|&i| indegree[i] == 0).collect();
    while let Some(i) = ready.pop() {
        order.push(i);
        for &j in &successors[i] {
            indegree[j] -= 1;
            if indegree[j] == 0 {
                ready.push(j);
            }
        }
    }
    if order.len() != n {
        return Err(GeomError::Degenerate("critical_path_method: the precedences contain a cycle"));
    }

    // Forward pass: the earliest each task can start is when its last
    // predecessor finishes.
    let mut early_start = vec![0.0f64; n];
    let mut early_finish = vec![0.0f64; n];
    for &i in &order {
        early_start[i] = tasks[i]
            .1
            .iter()
            .map(|&p| early_finish[p])
            .fold(0.0f64, f64::max);
        early_finish[i] = early_start[i] + tasks[i].0;
    }
    let duration = early_finish.iter().copied().fold(0.0f64, f64::max);

    // Backward pass: the latest each task can finish is when its earliest
    // successor must start, or the project end if it has none.
    let mut late_finish = vec![duration; n];
    let mut late_start = vec![0.0f64; n];
    for &i in order.iter().rev() {
        if !successors[i].is_empty() {
            late_finish[i] = successors[i]
                .iter()
                .map(|&j| late_start[j])
                .fold(f64::INFINITY, f64::min);
        }
        late_start[i] = late_finish[i] - tasks[i].0;
    }

    let times: Vec<TaskTimes> = (0..n)
        .map(|i| TaskTimes {
            early_start: early_start[i],
            early_finish: early_finish[i],
            late_start: late_start[i],
            late_finish: late_finish[i],
        })
        .collect();
    let critical: Vec<usize> = (0..n).filter(|&i| times[i].slack().abs() < TOL).collect();
    Ok((duration, critical, times))
}

/// PERT: the mean and variance of the project duration under three-point
/// estimates.
///
/// `tasks[i]` is `(optimistic, most likely, pessimistic, predecessors)`. Each
/// task's duration is taken as a beta distribution with mean
/// `(a + 4m + b) / 6` and standard deviation `(b - a) / 6`, and the project
/// duration as the sum along the critical path.
///
/// The variance is the sum of the *critical path's* variances only, which is
/// the method's known weakness: a near-critical path with high variance can
/// overtake the critical one and PERT will not see it, so the figure
/// understates the true spread. It is reported because it is what PERT means,
/// not because it is the whole answer.
///
/// # Errors
/// Returns an error if an estimate is out of order or the precedences are
/// unschedulable.
pub fn pert(tasks: &[(f64, f64, f64, Vec<usize>)]) -> Result<(f64, f64), GeomError> {
    if tasks.iter().any(|&(a, m, b, _)| !(a <= m && m <= b)) {
        return Err(GeomError::InvalidArgument("pert: estimates must be ordered a <= m <= b"));
    }
    let expected: Vec<(f64, Vec<usize>)> = tasks
        .iter()
        .map(|(a, m, b, preds)| ((a + 4.0 * m + b) / 6.0, preds.clone()))
        .collect();
    let (duration, critical, _) = critical_path_method(&expected)?;
    let variance: f64 = critical
        .iter()
        .map(|&i| {
            let (a, _, b, _) = &tasks[i];
            let sd = (b - a) / 6.0;
            sd * sd
        })
        .sum();
    Ok((duration, variance))
}

// ---------------------------------------------------------------------------
// Vehicle routing
// ---------------------------------------------------------------------------

/// Clarke-Wright savings for the capacitated vehicle routing problem.
///
/// Every customer starts on its own out-and-back route. Merging the routes
/// ending at `i` and beginning at `j` saves `d(0,i) + d(0,j) - d(i,j)` -- the
/// two depot legs replaced by one direct leg -- so merges are tried in
/// decreasing order of that saving, subject to capacity.
///
/// Returns the routes as customer sequences, excluding the depot at each end.
///
/// # Errors
/// Returns an error if the distance matrix is the wrong shape, or a customer's
/// demand exceeds a vehicle's capacity, which makes routing impossible.
pub fn vehicle_routing_savings(
    distance: &Matrix,
    demand: &[f64],
    capacity: f64,
) -> Result<Vec<Vec<usize>>, GeomError> {
    // Node 0 is the depot; customers are 1..n.
    let n = demand.len();
    if !distance.is_square() || distance.rows != n || n < 2 {
        return Err(GeomError::InvalidArgument("vehicle_routing_savings: shape mismatch"));
    }
    if demand[1..].iter().any(|&d| d > capacity) {
        return Err(GeomError::InvalidArgument("a customer's demand exceeds the capacity"));
    }

    let mut routes: Vec<Vec<usize>> = (1..n).map(|i| vec![i]).collect();
    let mut load: Vec<f64> = (1..n).map(|i| demand[i]).collect();

    let mut savings: Vec<(f64, usize, usize)> = Vec::new();
    for i in 1..n {
        for j in i + 1..n {
            savings.push((
                distance.get(0, i) + distance.get(0, j) - distance.get(i, j),
                i,
                j,
            ));
        }
    }
    savings.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    for (saving, i, j) in savings {
        if saving <= 0.0 {
            break;
        }
        let Some(ri) = routes.iter().position(|r| r.last() == Some(&i)) else { continue };
        let Some(rj) = routes.iter().position(|r| r.first() == Some(&j)) else { continue };
        // Merging a route with itself would close it into a cycle that never
        // returns to the depot.
        if ri == rj || load[ri] + load[rj] > capacity + TOL {
            continue;
        }
        let tail = routes[rj].clone();
        routes[ri].extend(tail);
        load[ri] += load[rj];
        routes.remove(rj);
        load.remove(rj);
    }
    Ok(routes)
}

/// A lower bound on a job shop makespan by the shifting bottleneck idea,
/// simplified.
///
/// `jobs[j]` lists `(machine, duration)` in the order job `j` must visit them.
/// Returns the larger of the busiest machine's total load and the longest
/// job's total work -- both of which any schedule must exceed, since a machine
/// cannot process two operations at once and a job cannot be in two places.
///
/// The full shifting bottleneck procedure solves a one-machine sequencing
/// problem per machine and iterates; this reports the elementary bound those
/// iterations start from.
///
/// # Errors
/// Returns an error if a machine index exceeds the machine count.
pub fn job_shop_shifting_bottleneck_lite(
    jobs: &[Vec<(usize, f64)>],
    machines: usize,
) -> Result<f64, GeomError> {
    if machines == 0 || jobs.is_empty() {
        return Err(GeomError::InvalidArgument("job_shop needs machines and jobs"));
    }
    if jobs.iter().any(|ops| ops.iter().any(|&(m, d)| m >= machines || d < 0.0)) {
        return Err(GeomError::InvalidArgument("job_shop: bad operation"));
    }
    let mut machine_load = vec![0.0f64; machines];
    let mut longest_job = 0.0f64;
    for ops in jobs {
        let mut total = 0.0;
        for &(m, d) in ops {
            machine_load[m] += d;
            total += d;
        }
        longest_job = longest_job.max(total);
    }
    Ok(machine_load.iter().copied().fold(0.0f64, f64::max).max(longest_job))
}

// ---------------------------------------------------------------------------
// Sequencing rules
// ---------------------------------------------------------------------------

/// Shortest processing time first: the order minimising mean flow time on one
/// machine.
///
/// Optimal by an exchange argument -- swapping an adjacent out-of-order pair
/// always improves the total -- and optimal for nothing else. It can make one
/// long job arbitrarily late while the average looks excellent, which is why
/// the objective has to be chosen before the rule.
///
/// `jobs[i]` is a processing time. Returns the job order.
#[must_use]
pub fn scheduling_spt(jobs: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..jobs.len()).collect();
    order.sort_by(|&a, &b| {
        jobs[a].partial_cmp(&jobs[b]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b))
    });
    order
}

/// Earliest due date first: the order minimising maximum lateness on one
/// machine.
///
/// Jackson's rule. Also by an exchange argument, and again optimal only for
/// its own objective: it makes no attempt to reduce the *number* of late jobs,
/// which is what [`moore_hodgson`] is for.
///
/// `jobs[i]` is `(processing time, due date)`. Returns the job order.
#[must_use]
pub fn scheduling_edd(jobs: &[(f64, f64)]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..jobs.len()).collect();
    order.sort_by(|&a, &b| {
        jobs[a].1.partial_cmp(&jobs[b].1).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b))
    });
    order
}

/// The Moore-Hodgson rule: the order minimising the *number* of late jobs on
/// one machine.
///
/// Work through the jobs by due date; whenever the schedule falls behind,
/// throw out the longest job accepted so far. That one removal buys the most
/// time back, and the jobs thrown out are exactly the late ones, which is what
/// makes the rule optimal rather than merely sensible.
///
/// Returns the order: the on-time jobs first in due-date order, then the late
/// ones.
///
/// `jobs[i]` is `(processing time, due date)`.
#[must_use]
pub fn moore_hodgson(jobs: &[(f64, f64)]) -> Vec<usize> {
    let by_due = scheduling_edd(jobs);
    let mut accepted: Vec<usize> = Vec::new();
    let mut rejected: Vec<usize> = Vec::new();
    let mut clock = 0.0f64;
    for i in by_due {
        accepted.push(i);
        clock += jobs[i].0;
        if clock > jobs[i].1 + TOL {
            // Drop the longest accepted job: the single removal that recovers
            // the most time.
            let worst = accepted
                .iter()
                .enumerate()
                .max_by(|a, b| {
                    jobs[*a.1].0.partial_cmp(&jobs[*b.1].0).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(k, _)| k);
            if let Some(k) = worst {
                let dropped = accepted.remove(k);
                clock -= jobs[dropped].0;
                rejected.push(dropped);
            }
        }
    }
    accepted.extend(rejected);
    accepted
}

/// Johnson's rule: the order minimising makespan through two machines in
/// series.
///
/// Every job visits machine one then machine two. Jobs whose first operation
/// is the shorter go first, in increasing order of that operation; the rest go
/// last, in decreasing order of their second. The first group fills machine
/// two's queue quickly and the second keeps it busy at the end, which is what
/// the exchange argument formalises.
///
/// `jobs[i]` is `(time on machine one, time on machine two)`.
#[must_use]
pub fn johnson_two_machine(jobs: &[(f64, f64)]) -> Vec<usize> {
    let mut head: Vec<usize> = Vec::new();
    let mut tail: Vec<usize> = Vec::new();
    for (i, &(a, b)) in jobs.iter().enumerate() {
        if a <= b {
            head.push(i);
        } else {
            tail.push(i);
        }
    }
    head.sort_by(|&a, &b| {
        jobs[a].0.partial_cmp(&jobs[b].0).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b))
    });
    tail.sort_by(|&a, &b| {
        jobs[b].1.partial_cmp(&jobs[a].1).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b))
    });
    head.extend(tail);
    head
}

/// The makespan of a two-machine flow shop under a given order.
///
/// Machine two cannot start a job before machine one finishes it, nor before
/// it finishes the previous job, which is the whole recursion.
#[must_use]
pub fn two_machine_makespan(jobs: &[(f64, f64)], order: &[usize]) -> f64 {
    let mut first_free = 0.0f64;
    let mut second_free = 0.0f64;
    for &i in order {
        first_free += jobs[i].0;
        second_free = second_free.max(first_free) + jobs[i].1;
    }
    second_free
}

/// Longest processing time first onto identical parallel machines.
///
/// Returns the makespan and which machine each job went to. The rule finishes
/// within `4/3 - 1/(3m)` of the optimum, and that bound is tight -- so it is a
/// guarantee rather than an observation, and the tests check it against an
/// exact answer.
///
/// # Panics
/// Panics if `machines` is zero.
#[must_use]
pub fn lpt_makespan(jobs: &[f64], machines: usize) -> (f64, Vec<usize>) {
    assert!(machines > 0, "lpt_makespan requires at least one machine");
    let mut order: Vec<usize> = (0..jobs.len()).collect();
    order.sort_by(|&a, &b| {
        jobs[b].partial_cmp(&jobs[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b))
    });
    let mut load = vec![0.0f64; machines];
    let mut assignment = vec![0usize; jobs.len()];
    for &i in &order {
        // Onto whichever machine is least busy.
        let target = (0..machines)
            .min_by(|&a, &b| load[a].partial_cmp(&load[b]).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0);
        load[target] += jobs[i];
        assignment[i] = target;
    }
    (load.iter().copied().fold(0.0f64, f64::max), assignment)
}

/// The largest set of pairwise disjoint intervals, by earliest finish time.
///
/// The greedy choice is optimal, and the proof is the reason: whatever the
/// optimal set, replacing its first interval by the one that finishes earliest
/// leaves it still valid and no smaller, so an optimal solution containing the
/// greedy choice always exists.
///
/// `intervals[i]` is `(start, end)`. Returns the chosen indices.
#[must_use]
pub fn interval_scheduling_max(intervals: &[(f64, f64)]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..intervals.len()).collect();
    order.sort_by(|&a, &b| {
        intervals[a].1.partial_cmp(&intervals[b].1).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut chosen = Vec::new();
    let mut clock = f64::NEG_INFINITY;
    for i in order {
        if intervals[i].0 >= clock - TOL {
            clock = intervals[i].1;
            chosen.push(i);
        }
    }
    chosen.sort_unstable();
    chosen
}

/// The most valuable set of pairwise disjoint intervals.
///
/// Weights break the greedy argument completely -- one long valuable interval
/// can be worth more than any number of short ones -- so this is a table:
/// sort by finish time and, for each interval, either take it and jump to the
/// last compatible one or skip it.
///
/// `intervals[i]` is `(start, end, weight)`. Returns the total and the chosen
/// indices.
#[must_use]
pub fn weighted_interval_scheduling(intervals: &[(f64, f64, f64)]) -> (f64, Vec<usize>) {
    let n = intervals.len();
    if n == 0 {
        return (0.0, Vec::new());
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        intervals[a].1.partial_cmp(&intervals[b].1).unwrap_or(std::cmp::Ordering::Equal)
    });

    // `latest[k]` is the last interval finishing at or before interval k starts.
    let latest: Vec<Option<usize>> = (0..n)
        .map(|k| {
            let start = intervals[order[k]].0;
            (0..k).rev().find(|&j| intervals[order[j]].1 <= start + TOL)
        })
        .collect();

    let mut best = vec![0.0f64; n + 1];
    for k in 0..n {
        let take = intervals[order[k]].2 + latest[k].map_or(0.0, |j| best[j + 1]);
        best[k + 1] = best[k].max(take);
    }

    let mut chosen = Vec::new();
    let mut k = n;
    while k > 0 {
        let take = intervals[order[k - 1]].2 + latest[k - 1].map_or(0.0, |j| best[j + 1]);
        if take >= best[k] - TOL && (take - best[k]).abs() < TOL {
            chosen.push(order[k - 1]);
            k = latest[k - 1].map_or(0, |j| j + 1);
        } else {
            k -= 1;
        }
    }
    chosen.sort_unstable();
    (best[n], chosen)
}

/// Turns a single-machine job order into `(job, start, finish)` bars.
///
/// Jobs run back to back in the given order from time zero, which is what a
/// single-machine sequencing rule assumes.
#[must_use]
pub fn gantt_data(processing: &[f64], order: &[usize]) -> Vec<(usize, f64, f64)> {
    let mut clock = 0.0f64;
    order
        .iter()
        .map(|&i| {
            let start = clock;
            clock += processing[i];
            (i, start, clock)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monte_carlo::Rng;

    fn pick(rng: &mut Rng, n: usize) -> usize {
        ((u128::from(rng.next_u64()) * n as u128) >> 64) as usize
    }

    /// Every permutation of `0..n`, for checking a rule against brute force.
    fn permutations(n: usize) -> Vec<Vec<usize>> {
        let mut out = Vec::new();
        let mut current: Vec<usize> = (0..n).collect();
        permute(&mut current, 0, &mut out);
        out
    }

    fn permute(current: &mut Vec<usize>, k: usize, out: &mut Vec<Vec<usize>>) {
        if k == current.len() {
            out.push(current.clone());
            return;
        }
        for i in k..current.len() {
            current.swap(k, i);
            permute(current, k + 1, out);
            current.swap(k, i);
        }
    }

    // -----------------------------------------------------------------
    // Flow formulations against the graph module
    // -----------------------------------------------------------------

    #[test]
    fn the_shortest_path_program_agrees_with_dijkstra() {
        // Two completely different methods: a combinatorial priority-queue
        // sweep and a linear program over a node-arc incidence matrix. The
        // matrix is totally unimodular, so the relaxation is integral and the
        // two must give the same number.
        let mut rng = Rng::new(0x5407_0001);
        let mut compared = 0usize;
        for _ in 0..40 {
            let n = 4 + pick(&mut rng, 5);
            let mut g = Graph::new(n, true);
            for u in 0..n {
                for v in 0..n {
                    if u != v && rng.next_f64() < 0.45 {
                        g.add_edge(u, v, (rng.next_f64() * 9.0).round() + 1.0);
                    }
                }
            }
            let (s, t) = (0usize, n - 1);
            let (distances, _) = crate::graph::paths::dijkstra(&g, s);
            let lp = shortest_path_lp_check(&g, s, t).unwrap();
            match (distances[t].is_finite(), lp) {
                (true, Some(value)) => {
                    compared += 1;
                    assert!(
                        (value - distances[t]).abs() < 1e-7,
                        "the program gave {value}, Dijkstra {}",
                        distances[t]
                    );
                }
                (false, None) => {}
                (reachable, other) => {
                    panic!("disagreed on reachability: Dijkstra {reachable}, program {other:?}")
                }
            }
        }
        assert!(compared > 20, "only {compared} of 40 instances were comparable");

        let g = Graph::new(3, true);
        assert_eq!(shortest_path_lp_check(&g, 1, 1).unwrap(), Some(0.0));
        assert!(shortest_path_lp_check(&g, 5, 0).is_err());
        let mut negative = Graph::new(2, true);
        negative.add_edge(0, 1, -1.0);
        assert!(shortest_path_lp_check(&negative, 0, 1).is_err());
    }

    #[test]
    fn the_max_flow_program_agrees_with_the_combinatorial_algorithm() {
        let mut rng = Rng::new(0xF108_0001);
        for _ in 0..30 {
            let n = 4 + pick(&mut rng, 4);
            let mut g = Graph::new(n, true);
            for u in 0..n {
                for v in 0..n {
                    if u != v && rng.next_f64() < 0.5 {
                        g.add_edge(u, v, (rng.next_f64() * 8.0).round() + 1.0);
                    }
                }
            }
            let (s, t) = (0usize, n - 1);
            let combinatorial = crate::graph::flow::max_flow(&g, s, t);
            let lp = max_flow_lp_check(&g, s, t).unwrap().unwrap_or(0.0);
            assert!(
                (lp - combinatorial).abs() < 1e-6,
                "the program gave {lp}, the augmenting-path method {combinatorial}"
            );
        }
        let g = Graph::new(3, true);
        assert!(max_flow_lp_check(&g, 0, 0).is_err());
        assert!(max_flow_lp_check(&g, 0, 9).is_err());
    }

    #[test]
    fn transshipment_conserves_flow_and_respects_capacity() {
        // Two sources, one hub, two sinks. Everything supplied must arrive.
        let supply = [10.0, 5.0, 0.0, -8.0, -7.0];
        let arcs = [
            (0usize, 2usize, 1.0, 20.0),
            (1, 2, 2.0, 20.0),
            (2, 3, 1.0, 20.0),
            (2, 4, 3.0, 20.0),
            (0, 3, 6.0, 20.0),
        ];
        let LpResult::Optimal { x, objective, .. } = transshipment(&supply, &arcs).unwrap() else {
            panic!("expected an optimum");
        };
        for (k, &(_, _, _, cap)) in arcs.iter().enumerate() {
            assert!(x[k] >= -1e-7 && x[k] <= cap + 1e-7, "arc {k} carries {}", x[k]);
        }
        // Conservation at every node.
        for v in 0..supply.len() {
            let out: f64 =
                arcs.iter().enumerate().filter(|(_, a)| a.0 == v).map(|(k, _)| x[k]).sum();
            let into: f64 =
                arcs.iter().enumerate().filter(|(_, a)| a.1 == v).map(|(k, _)| x[k]).sum();
            assert!(
                (out - into - supply[v]).abs() < 1e-7,
                "node {v}: out {out}, in {into}, supply {}",
                supply[v]
            );
        }
        // The cheapest routing sends everything through the hub.
        assert!(objective > 0.0 && objective.is_finite());

        assert!(transshipment(&[1.0, 1.0], &[(0, 1, 1.0, 5.0)]).is_err());
        assert!(transshipment(&[1.0, -1.0], &[(0, 9, 1.0, 5.0)]).is_err());
        assert!(transshipment(&[], &[]).is_err());
        // The lite network simplex routes to the same place.
        assert_eq!(
            network_simplex_lite(&supply, &arcs).unwrap().objective(),
            Some(objective)
        );
    }

    // -----------------------------------------------------------------
    // Project scheduling
    // -----------------------------------------------------------------

    #[test]
    fn the_critical_path_is_the_longest_path_and_has_no_slack() {
        // A small project: A -> C, B -> C, C -> D, with B also feeding D.
        let tasks = vec![
            (3.0, vec![]),
            (2.0, vec![]),
            (4.0, vec![0usize, 1]),
            (1.0, vec![2usize, 1]),
        ];
        let (duration, critical, times) = critical_path_method(&tasks).unwrap();
        // A then C then D is 3 + 4 + 1 = 8, the longest chain.
        assert!((duration - 8.0).abs() < 1e-9, "duration {duration}");
        assert_eq!(critical, vec![0, 2, 3], "critical tasks {critical:?}");

        for (i, t) in times.iter().enumerate() {
            assert!(
                (t.early_finish - t.early_start - tasks[i].0).abs() < 1e-9,
                "task {i}: finish minus start is not its duration"
            );
            assert!(
                (t.late_finish - t.late_start - tasks[i].0).abs() < 1e-9,
                "task {i}: the late pair is not its duration apart"
            );
            assert!(t.late_start >= t.early_start - 1e-9, "task {i} starts late before early");
            assert!(
                (t.slack() - (t.late_finish - t.early_finish)).abs() < 1e-9,
                "task {i}: slack differs by which end it is measured from"
            );
            // A task is critical exactly when its slack is zero.
            assert_eq!(critical.contains(&i), t.slack().abs() < 1e-9);
            // Every predecessor finishes before this one starts.
            for &p in &tasks[i].1 {
                assert!(
                    times[p].early_finish <= t.early_start + 1e-9,
                    "task {i} starts before its predecessor {p} finishes"
                );
            }
        }
        // Shortening a non-critical task buys nothing.
        let mut relaxed = tasks.clone();
        relaxed[1].0 = 0.5;
        assert!((critical_path_method(&relaxed).unwrap().0 - duration).abs() < 1e-9);
        // Shortening a critical one does.
        let mut shortened = tasks.clone();
        shortened[2].0 = 1.0;
        assert!(critical_path_method(&shortened).unwrap().0 < duration - 1e-9);

        assert!(critical_path_method(&[]).is_err());
        assert!(critical_path_method(&[(1.0, vec![5])]).is_err());
        // A cycle is unschedulable, not merely slow.
        assert!(critical_path_method(&[(1.0, vec![1]), (1.0, vec![0])]).is_err());
    }

    #[test]
    fn the_critical_path_matches_a_longest_path_search() {
        let mut rng = Rng::new(0x0C97_0001);
        for _ in 0..60 {
            let n = 3 + pick(&mut rng, 6);
            // Predecessors only from earlier indices, so the graph is acyclic
            // by construction.
            let tasks: Vec<(f64, Vec<usize>)> = (0..n)
                .map(|i| {
                    let preds: Vec<usize> =
                        (0..i).filter(|_| rng.next_f64() < 0.4).collect();
                    ((rng.next_f64() * 9.0).round() + 1.0, preds)
                })
                .collect();
            let (duration, critical, times) = critical_path_method(&tasks).unwrap();

            // The longest chain, computed independently by recursion.
            let mut longest = vec![0.0f64; n];
            for i in 0..n {
                longest[i] = tasks[i].0
                    + tasks[i].1.iter().map(|&p| longest[p]).fold(0.0f64, f64::max);
            }
            let expected = longest.iter().copied().fold(0.0f64, f64::max);
            assert!(
                (duration - expected).abs() < 1e-9,
                "the method gave {duration}, the longest chain {expected}"
            );
            assert!(!critical.is_empty(), "every project has a critical path");
            // The critical tasks' durations chain up to the whole project.
            for &i in &critical {
                assert!(times[i].slack().abs() < 1e-9);
            }
        }
    }

    #[test]
    fn pert_averages_its_estimates_and_sums_the_critical_variances() {
        // One chain of two tasks, so the critical path is unambiguous.
        let tasks = vec![
            (2.0, 4.0, 12.0, vec![]),
            (1.0, 2.0, 3.0, vec![0usize]),
        ];
        let (mean, variance) = pert(&tasks).unwrap();
        // Beta means: (2 + 16 + 12)/6 = 5 and (1 + 8 + 3)/6 = 2.
        assert!((mean - 7.0).abs() < 1e-9, "mean {mean}");
        // Variances: ((12 - 2)/6)^2 + ((3 - 1)/6)^2.
        let expected = (10.0f64 / 6.0).powi(2) + (2.0f64 / 6.0).powi(2);
        assert!((variance - expected).abs() < 1e-9, "variance {variance} against {expected}");
        assert!(variance > 0.0);

        // A symmetric estimate has the same mean as its most likely value.
        let symmetric = vec![(1.0, 5.0, 9.0, vec![])];
        assert!((pert(&symmetric).unwrap().0 - 5.0).abs() < 1e-9);
        // A certain estimate has no variance.
        let certain = vec![(4.0, 4.0, 4.0, vec![])];
        assert!(pert(&certain).unwrap().1.abs() < 1e-12);

        assert!(pert(&[(5.0, 1.0, 9.0, vec![])]).is_err(), "unordered estimates");
        assert!(pert(&[(1.0, 5.0, 3.0, vec![])]).is_err());
    }

    // -----------------------------------------------------------------
    // Sequencing rules, each against brute force on its own objective
    // -----------------------------------------------------------------

    #[test]
    fn shortest_processing_time_minimises_mean_flow_and_nothing_else() {
        let mut rng = Rng::new(0x05B7_0001);
        for _ in 0..60 {
            let n = 2 + pick(&mut rng, 5);
            let jobs: Vec<f64> = (0..n).map(|_| (rng.next_f64() * 9.0).round() + 1.0).collect();

            let flow = |order: &[usize]| -> f64 {
                let mut clock = 0.0;
                let mut total = 0.0;
                for &i in order {
                    clock += jobs[i];
                    total += clock;
                }
                total
            };
            let rule = scheduling_spt(&jobs);
            assert_eq!(rule.len(), n);
            let best = permutations(n).iter().map(|p| flow(p)).fold(f64::INFINITY, f64::min);
            assert!(
                (flow(&rule) - best).abs() < 1e-9,
                "the rule gives total flow {}, the best is {best}",
                flow(&rule)
            );
        }

        // And it is genuinely not optimal for maximum lateness: one long job
        // due early is pushed to the back.
        let jobs = [(10.0, 10.0), (1.0, 100.0)];
        let by_time = scheduling_spt(&[10.0, 1.0]);
        let by_due = scheduling_edd(&jobs);
        let lateness = |order: &[usize]| -> f64 {
            let mut clock = 0.0f64;
            let mut worst = f64::NEG_INFINITY;
            for &i in order {
                clock += jobs[i].0;
                worst = worst.max(clock - jobs[i].1);
            }
            worst
        };
        assert!(
            lateness(&by_due) < lateness(&by_time),
            "earliest-due-date should beat shortest-processing-time on lateness"
        );
    }

    #[test]
    fn earliest_due_date_minimises_maximum_lateness() {
        let mut rng = Rng::new(0x0EDD_0001);
        for _ in 0..60 {
            let n = 2 + pick(&mut rng, 5);
            let jobs: Vec<(f64, f64)> = (0..n)
                .map(|_| {
                    (
                        (rng.next_f64() * 6.0).round() + 1.0,
                        (rng.next_f64() * 20.0).round() + 1.0,
                    )
                })
                .collect();
            let lateness = |order: &[usize]| -> f64 {
                let mut clock = 0.0f64;
                let mut worst = f64::NEG_INFINITY;
                for &i in order {
                    clock += jobs[i].0;
                    worst = worst.max(clock - jobs[i].1);
                }
                worst
            };
            let rule = scheduling_edd(&jobs);
            let best =
                permutations(n).iter().map(|p| lateness(p)).fold(f64::INFINITY, f64::min);
            assert!(
                (lateness(&rule) - best).abs() < 1e-9,
                "the rule gives maximum lateness {}, the best is {best}",
                lateness(&rule)
            );
        }
    }

    #[test]
    fn moore_hodgson_minimises_the_number_of_late_jobs() {
        let mut rng = Rng::new(0x3007_0001);
        for _ in 0..60 {
            let n = 2 + pick(&mut rng, 5);
            let jobs: Vec<(f64, f64)> = (0..n)
                .map(|_| {
                    (
                        (rng.next_f64() * 6.0).round() + 1.0,
                        (rng.next_f64() * 18.0).round() + 1.0,
                    )
                })
                .collect();
            let late_count = |order: &[usize]| -> usize {
                let mut clock = 0.0f64;
                let mut late = 0usize;
                for &i in order {
                    clock += jobs[i].0;
                    if clock > jobs[i].1 + 1e-9 {
                        late += 1;
                    }
                }
                late
            };
            let rule = moore_hodgson(&jobs);
            assert_eq!(rule.len(), n, "the rule dropped a job from the order");
            let mut sorted = rule.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, (0..n).collect::<Vec<usize>>(), "the order is not a permutation");

            let best = permutations(n).iter().map(|p| late_count(p)).min().unwrap_or(0);
            assert_eq!(
                late_count(&rule),
                best,
                "the rule leaves {} jobs late, the best is {best}",
                late_count(&rule)
            );
        }
    }

    #[test]
    fn johnsons_rule_minimises_the_two_machine_makespan() {
        let mut rng = Rng::new(0x1085_0001);
        for _ in 0..60 {
            let n = 2 + pick(&mut rng, 5);
            let jobs: Vec<(f64, f64)> = (0..n)
                .map(|_| {
                    (
                        (rng.next_f64() * 8.0).round() + 1.0,
                        (rng.next_f64() * 8.0).round() + 1.0,
                    )
                })
                .collect();
            let rule = johnson_two_machine(&jobs);
            assert_eq!(rule.len(), n);
            let best = permutations(n)
                .iter()
                .map(|p| two_machine_makespan(&jobs, p))
                .fold(f64::INFINITY, f64::min);
            assert!(
                (two_machine_makespan(&jobs, &rule) - best).abs() < 1e-9,
                "the rule gives makespan {}, the best is {best}",
                two_machine_makespan(&jobs, &rule)
            );
            // The makespan is at least the busier machine's total load.
            let load_one: f64 = jobs.iter().map(|j| j.0).sum();
            let load_two: f64 = jobs.iter().map(|j| j.1).sum();
            assert!(best >= load_one.max(load_two) - 1e-9);
        }
    }

    #[test]
    fn longest_processing_time_stays_within_its_proven_ratio() {
        let mut rng = Rng::new(0x1B70_0001);
        for _ in 0..60 {
            let machines = 2 + pick(&mut rng, 3);
            let n = machines + pick(&mut rng, 6);
            let jobs: Vec<f64> =
                (0..n).map(|_| (rng.next_f64() * 9.0).round() + 1.0).collect();

            let (makespan, assignment) = lpt_makespan(&jobs, machines);
            assert_eq!(assignment.len(), n);
            // The reported makespan is the busiest machine's load.
            let loads: Vec<f64> = (0..machines)
                .map(|m| {
                    jobs.iter()
                        .enumerate()
                        .filter(|(i, _)| assignment[*i] == m)
                        .map(|(_, &d)| d)
                        .sum()
                })
                .collect();
            assert!(
                (loads.iter().copied().fold(0.0f64, f64::max) - makespan).abs() < 1e-9,
                "the reported makespan does not match the loads {loads:?}"
            );

            // Exact optimum by assigning every job to every machine.
            let mut best = f64::INFINITY;
            let mut counter = vec![0usize; n];
            loop {
                let mut load = vec![0.0f64; machines];
                for (i, &m) in counter.iter().enumerate() {
                    load[m] += jobs[i];
                }
                best = best.min(load.iter().copied().fold(0.0f64, f64::max));
                let mut k = 0usize;
                while k < n {
                    counter[k] += 1;
                    if counter[k] < machines {
                        break;
                    }
                    counter[k] = 0;
                    k += 1;
                }
                if k == n || n > 8 {
                    break;
                }
            }
            if n > 8 {
                continue;
            }
            let ratio = 4.0 / 3.0 - 1.0 / (3.0 * machines as f64);
            assert!(
                makespan <= ratio * best + 1e-9,
                "makespan {makespan} exceeds {ratio} times the optimum {best}"
            );
            assert!(makespan >= best - 1e-9, "the greedy answer beat the optimum");
        }
    }

    #[test]
    fn interval_scheduling_takes_as_many_as_possible() {
        let mut rng = Rng::new(0x1275_0001);
        for _ in 0..80 {
            let n = 1 + pick(&mut rng, 10);
            let intervals: Vec<(f64, f64)> = (0..n)
                .map(|_| {
                    let s = (rng.next_f64() * 15.0).round();
                    (s, s + (rng.next_f64() * 6.0).round() + 1.0)
                })
                .collect();
            let chosen = interval_scheduling_max(&intervals);

            // The chosen intervals really are pairwise disjoint.
            for a in 0..chosen.len() {
                for b in a + 1..chosen.len() {
                    let (i, j) = (chosen[a], chosen[b]);
                    let overlap = intervals[i].0.max(intervals[j].0)
                        < intervals[i].1.min(intervals[j].1) - 1e-9;
                    assert!(!overlap, "intervals {i} and {j} overlap");
                }
            }
            // And no larger disjoint set exists.
            let mut best = 0usize;
            for mask in 0u32..(1u32 << n) {
                let members: Vec<usize> = (0..n).filter(|k| mask & (1 << k) != 0).collect();
                let disjoint = members.iter().enumerate().all(|(a, &i)| {
                    members[a + 1..].iter().all(|&j| {
                        intervals[i].0.max(intervals[j].0)
                            >= intervals[i].1.min(intervals[j].1) - 1e-9
                    })
                });
                if disjoint {
                    best = best.max(members.len());
                }
            }
            assert_eq!(chosen.len(), best, "took {} of a possible {best}", chosen.len());
        }
    }

    #[test]
    fn weighted_interval_scheduling_takes_the_most_valuable_set() {
        let mut rng = Rng::new(0x7215_0001);
        for _ in 0..80 {
            let n = 1 + pick(&mut rng, 9);
            let intervals: Vec<(f64, f64, f64)> = (0..n)
                .map(|_| {
                    let s = (rng.next_f64() * 12.0).round();
                    (
                        s,
                        s + (rng.next_f64() * 5.0).round() + 1.0,
                        (rng.next_f64() * 9.0).round() + 1.0,
                    )
                })
                .collect();
            let (total, chosen) = weighted_interval_scheduling(&intervals);

            // The reported set is disjoint and worth what was claimed.
            for a in 0..chosen.len() {
                for b in a + 1..chosen.len() {
                    let (i, j) = (chosen[a], chosen[b]);
                    assert!(
                        intervals[i].0.max(intervals[j].0)
                            >= intervals[i].1.min(intervals[j].1) - 1e-9,
                        "intervals {i} and {j} overlap"
                    );
                }
            }
            let claimed: f64 = chosen.iter().map(|&i| intervals[i].2).sum();
            assert!((claimed - total).abs() < 1e-7, "the set is worth {claimed}, not {total}");

            // No disjoint set is worth more.
            let mut best = 0.0f64;
            for mask in 0u32..(1u32 << n) {
                let members: Vec<usize> = (0..n).filter(|k| mask & (1 << k) != 0).collect();
                let disjoint = members.iter().enumerate().all(|(a, &i)| {
                    members[a + 1..].iter().all(|&j| {
                        intervals[i].0.max(intervals[j].0)
                            >= intervals[i].1.min(intervals[j].1) - 1e-9
                    })
                });
                if disjoint {
                    best = best.max(members.iter().map(|&i| intervals[i].2).sum::<f64>());
                }
            }
            assert!((total - best).abs() < 1e-7, "took {total} of a possible {best}");
        }
        assert_eq!(weighted_interval_scheduling(&[]), (0.0, Vec::new()));
        // Weights break the greedy argument: one long valuable interval beats
        // two short cheap ones.
        let (value, picks) =
            weighted_interval_scheduling(&[(0.0, 10.0, 100.0), (0.0, 1.0, 1.0), (2.0, 3.0, 1.0)]);
        assert_eq!(picks, vec![0]);
        assert!((value - 100.0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------
    // Routing and shop floor
    // -----------------------------------------------------------------

    #[test]
    fn savings_routing_visits_every_customer_within_capacity() {
        let mut rng = Rng::new(0xC147_0001);
        for _ in 0..40 {
            let customers = 3 + pick(&mut rng, 6);
            let n = customers + 1;
            // A symmetric distance matrix from random points.
            let points: Vec<(f64, f64)> =
                (0..n).map(|_| (rng.next_f64() * 50.0, rng.next_f64() * 50.0)).collect();
            let mut distance = Matrix::zeros(n, n);
            for i in 0..n {
                for j in 0..n {
                    let d = ((points[i].0 - points[j].0).powi(2)
                        + (points[i].1 - points[j].1).powi(2))
                    .sqrt();
                    distance.set(i, j, d);
                }
            }
            let mut demand = vec![0.0f64; n];
            for entry in demand.iter_mut().skip(1) {
                *entry = (rng.next_f64() * 8.0).round() + 1.0;
            }
            let capacity = 20.0f64;

            let routes = vehicle_routing_savings(&distance, &demand, capacity).unwrap();
            // Every customer appears exactly once across all routes.
            let mut seen = vec![0usize; n];
            for route in &routes {
                let load: f64 = route.iter().map(|&i| demand[i]).sum();
                assert!(load <= capacity + 1e-9, "a route carries {load}");
                for &i in route {
                    assert!(i >= 1 && i < n, "the depot appeared inside a route");
                    seen[i] += 1;
                }
            }
            assert!(
                seen[1..].iter().all(|&k| k == 1),
                "a customer was visited {seen:?} times"
            );
            // Merging can only reduce the number of routes.
            assert!(routes.len() <= customers, "more routes than customers");
        }

        let d = Matrix::from_rows(&[&[0.0, 1.0], &[1.0, 0.0]]).unwrap();
        assert!(vehicle_routing_savings(&d, &[0.0, 99.0], 10.0).is_err());
        assert!(vehicle_routing_savings(&d, &[0.0], 10.0).is_err());
    }

    #[test]
    fn the_job_shop_bound_is_a_bound_no_schedule_can_beat() {
        let jobs = vec![
            vec![(0usize, 3.0), (1usize, 2.0)],
            vec![(1usize, 4.0), (0usize, 1.0)],
            vec![(0usize, 2.0), (1usize, 5.0)],
        ];
        let bound = job_shop_shifting_bottleneck_lite(&jobs, 2).unwrap();
        // Machine 0 carries 3 + 1 + 2 = 6; machine 1 carries 2 + 4 + 5 = 11;
        // the longest job is 7. The bound is the largest of those.
        assert!((bound - 11.0).abs() < 1e-9, "bound {bound}");
        // Neither a machine nor a job can be beaten.
        let busiest = 11.0f64;
        let longest = 7.0f64;
        assert!(bound >= busiest - 1e-9 && bound >= longest - 1e-9);

        assert!(job_shop_shifting_bottleneck_lite(&jobs, 0).is_err());
        assert!(job_shop_shifting_bottleneck_lite(&[], 2).is_err());
        assert!(job_shop_shifting_bottleneck_lite(&[vec![(5usize, 1.0)]], 2).is_err());
    }

    #[test]
    fn gantt_bars_run_back_to_back_in_the_given_order() {
        let processing = [3.0, 1.0, 4.0];
        let order = scheduling_spt(&processing);
        let bars = gantt_data(&processing, &order);
        assert_eq!(bars.len(), 3);
        assert!((bars[0].1 - 0.0).abs() < 1e-12, "the first bar does not start at zero");
        for w in bars.windows(2) {
            assert!((w[0].2 - w[1].1).abs() < 1e-12, "a gap or overlap between bars");
        }
        for &(job, start, finish) in &bars {
            assert!(
                (finish - start - processing[job]).abs() < 1e-12,
                "bar {job} is not its own length"
            );
        }
        // The last bar ends at the total work, whatever the order.
        assert!((bars[2].2 - processing.iter().sum::<f64>()).abs() < 1e-12);
        assert!(gantt_data(&processing, &[]).is_empty());
    }
}
