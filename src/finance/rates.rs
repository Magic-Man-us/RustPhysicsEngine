//! Interest rates: discounting, bonds, curves and short-rate models.
//!
//! # Two things a "rate" can mean
//!
//! A quoted rate is meaningless without its compounding convention. 10%
//! compounded annually, semi-annually and continuously produce growth
//! factors of 1.1, 1.1025 and 1.10517 over a year -- differences that are
//! small over one period and decisive over thirty. [`Compounding`] makes
//! the convention explicit at every call site rather than leaving it to a
//! comment, and [`equivalent_rate`] converts between them.
//!
//! The second distinction is between a *zero rate*, which discounts a
//! single payment at one maturity, and a *yield*, which is the single
//! rate that reproduces a whole bond's price. They coincide only for a
//! zero-coupon bond. A coupon bond's yield is a weighted average of the
//! zero rates along its life, weighted by the discounted cashflows -- so
//! two bonds of the same maturity and different coupons have different
//! yields off the same curve, which is what makes a yield a property of
//! the instrument rather than of the market.
//!
//! # What is solved and what is assumed
//!
//! [`irr`], [`ytm_solve`] and [`bootstrap_zero_curve`] invert a price to
//! find a rate, and each has a uniqueness condition that the
//! documentation states and the code checks where it can. A yield always
//! exists and is unique for a bond with positive cashflows; an internal
//! rate of return need not be either, and the sign-change test is the
//! only cheap guarantee available.

use crate::error::GeomError;

/// How often a quoted rate compounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compounding {
    /// Once per year.
    Annual,
    /// Twice per year, the convention for most government bonds.
    SemiAnnual,
    /// Four times per year.
    Quarterly,
    /// Twelve times per year, the convention for consumer lending.
    Monthly,
    /// Continuously, the convention for derivative pricing.
    Continuous,
}

impl Compounding {
    /// Periods per year, or `None` for continuous compounding.
    #[must_use]
    pub fn periods_per_year(self) -> Option<f64> {
        match self {
            Compounding::Annual => Some(1.0),
            Compounding::SemiAnnual => Some(2.0),
            Compounding::Quarterly => Some(4.0),
            Compounding::Monthly => Some(12.0),
            Compounding::Continuous => None,
        }
    }
}

/// The present value of one unit paid at time `t`.
///
/// `(1 + r/m)^(-m t)` for `m` compounding periods a year, and `e^(-r t)`
/// continuously. The two agree in the limit `m -> infinity`, which is the
/// whole reason continuous compounding is used in pricing: it turns a
/// product over periods into an exponential and makes rates additive
/// across maturities.
///
/// # Errors
/// Returns an error for a negative time, a non-finite rate, or a periodic
/// rate at or below `-100%` per period, where the growth factor is
/// non-positive and the discount factor does not exist.
pub fn discount_factor(rate: f64, t: f64, compounding: Compounding) -> Result<f64, GeomError> {
    if t < 0.0 || !t.is_finite() || !rate.is_finite() {
        return Err(GeomError::InvalidArgument("discount_factor: bad rate or time"));
    }
    match compounding.periods_per_year() {
        None => Ok((-rate * t).exp()),
        Some(m) => {
            let growth = 1.0 + rate / m;
            if growth <= 0.0 {
                return Err(GeomError::Degenerate("the periodic growth factor is not positive"));
            }
            Ok(growth.powf(-m * t))
        }
    }
}

/// Converts a rate between compounding conventions, preserving the growth
/// factor over a year.
///
/// The number changes but the money does not: 10% semi-annual and 9.7580%
/// continuous are the same investment written two ways. Quoting the
/// smaller number is a real practice and this is what makes the two
/// comparable.
///
/// # Errors
/// Returns an error for a non-finite rate or a periodic rate at or below
/// `-100%` per period.
pub fn equivalent_rate(
    rate: f64,
    from: Compounding,
    to: Compounding,
) -> Result<f64, GeomError> {
    if !rate.is_finite() {
        return Err(GeomError::InvalidArgument("equivalent_rate: the rate is not finite"));
    }
    let annual_growth = match from.periods_per_year() {
        None => rate.exp(),
        Some(m) => {
            let growth = 1.0 + rate / m;
            if growth <= 0.0 {
                return Err(GeomError::Degenerate("the periodic growth factor is not positive"));
            }
            growth.powf(m)
        }
    };
    Ok(match to.periods_per_year() {
        None => annual_growth.ln(),
        Some(m) => m * (annual_growth.powf(1.0 / m) - 1.0),
    })
}

/// The net present value of cashflows at times `0, 1, ..., n-1` periods.
///
/// Discounted at the periodic rate `rate`, so `cashflows[0]` is undiscounted.
///
/// # Errors
/// Returns an error for no cashflows, a non-finite value, or a rate at or
/// below `-100%`.
pub fn npv(rate: f64, cashflows: &[f64]) -> Result<f64, GeomError> {
    if cashflows.is_empty() || cashflows.iter().any(|c| !c.is_finite()) {
        return Err(GeomError::InvalidArgument("npv: bad cashflows"));
    }
    if !(rate > -1.0) || !rate.is_finite() {
        return Err(GeomError::InvalidArgument("npv: the rate is at or below -100%"));
    }
    let growth = 1.0 + rate;
    Ok(cashflows.iter().enumerate().map(|(k, c)| c / growth.powi(k as i32)).sum())
}

/// The internal rate of return: the periodic rate at which the cashflows'
/// net present value is zero.
///
/// Returns `None` when no rate in `(-99.99%, 1e6)` does, or when the
/// cashflows change sign more than once and the answer would not be
/// unique. That second case is the one worth knowing about: Descartes'
/// rule bounds the number of positive roots by the number of sign changes,
/// so a single change guarantees at most one rate, and a project that
/// alternates between spending and earning can genuinely have several
/// internal rates of return or none at all. Reporting one of them as
/// *the* return would be a mistake this refuses to make.
///
/// # Errors
/// Returns an error for fewer than two cashflows, or a non-finite value.
pub fn irr(cashflows: &[f64]) -> Result<Option<f64>, GeomError> {
    if cashflows.len() < 2 || cashflows.iter().any(|c| !c.is_finite()) {
        return Err(GeomError::InvalidArgument("irr: bad cashflows"));
    }
    let signs: Vec<f64> = cashflows.iter().copied().filter(|c| *c != 0.0).collect();
    let changes = signs.windows(2).filter(|p| p[0] * p[1] < 0.0).count();
    if changes != 1 {
        return Ok(None);
    }
    let value = |rate: f64| npv(rate, cashflows).unwrap_or(f64::NAN);
    bracket_and_solve(&value)
}

/// Brackets a monotone sign change on `(-1, inf)` and bisects it.
fn bracket_and_solve(f: &dyn Fn(f64) -> f64) -> Result<Option<f64>, GeomError> {
    // The lower end cannot simply be pinned near -1: discounting a long
    // schedule at -99.99% raises 1e-4 to the power of the term and
    // overflows to infinity, so a fixed bound loses every deeply negative
    // yield. Walk down toward -1 instead, keeping the last point at which
    // the function is still finite.
    let mut low = -0.5;
    let mut at_low = f(low);
    for _ in 0..60 {
        let next = 0.5 * (low - 1.0);
        let value = f(next);
        if !value.is_finite() {
            break;
        }
        low = next;
        at_low = value;
        if at_low * f(0.0) <= 0.0 {
            break;
        }
    }
    if !at_low.is_finite() {
        return Ok(None);
    }
    let mut high = 0.0;
    let mut at_high = f(high);
    let mut attempts = 0;
    while at_low * at_high > 0.0 {
        high = if high == 0.0 { 0.1 } else { high * 2.0 };
        if high > 1e6 {
            return Ok(None);
        }
        at_high = f(high);
        if !at_high.is_finite() {
            return Ok(None);
        }
        attempts += 1;
        if attempts > 200 {
            return Ok(None);
        }
    }
    for _ in 0..200 {
        let mid = 0.5 * (low + high);
        if f(mid) * at_low > 0.0 {
            low = mid;
        } else {
            high = mid;
        }
        if high - low < 1e-14 * (1.0 + low.abs()) {
            break;
        }
    }
    Ok(Some(0.5 * (low + high)))
}

/// The annualised internal rate of return for cashflows at irregular
/// dates, given in years from the first.
///
/// The rate is annual with annual compounding, so a payment at 0.5 years
/// is discounted by `(1 + r)^-0.5`. That fractional exponent is why this
/// needs its own function rather than being IRR on a padded schedule: real
/// cashflows do not fall on period boundaries, and forcing them there
/// misprices by days of interest.
///
/// # Errors
/// Returns an error for fewer than two flows, mismatched lengths, times
/// that are not increasing from zero, or a non-finite value.
pub fn xirr(times: &[f64], cashflows: &[f64]) -> Result<Option<f64>, GeomError> {
    if times.len() < 2 || times.len() != cashflows.len() {
        return Err(GeomError::InvalidArgument("xirr: mismatched or too few flows"));
    }
    if times[0] != 0.0 || times.windows(2).any(|p| p[1] <= p[0]) {
        return Err(GeomError::InvalidArgument("xirr: the times must start at zero and increase"));
    }
    if times.iter().chain(cashflows.iter()).any(|x| !x.is_finite()) {
        return Err(GeomError::InvalidArgument("xirr: a value is not finite"));
    }
    let signs: Vec<f64> = cashflows.iter().copied().filter(|c| *c != 0.0).collect();
    if signs.windows(2).filter(|p| p[0] * p[1] < 0.0).count() != 1 {
        return Ok(None);
    }
    let value = |rate: f64| -> f64 {
        let growth = 1.0 + rate;
        if growth <= 0.0 {
            return f64::NAN;
        }
        times.iter().zip(cashflows.iter()).map(|(t, c)| c * growth.powf(-t)).sum()
    };
    bracket_and_solve(&value)
}

// ---------------------------------------------------------------------------
// Bonds
// ---------------------------------------------------------------------------

/// The price of a bond paying `coupon` per period for `periods` periods
/// and `face` at the end, discounted at the periodic yield `ytm`.
///
/// All three arguments are *per period*, not per year: a 6% annual coupon
/// on 100 face paid semi-annually for five years is `coupon = 3`,
/// `periods = 10`, and a yield quoted semi-annually.
///
/// A bond trades above face when its coupon exceeds its yield and below
/// when it does not, and that is not a market opinion but arithmetic: the
/// price is the yield's own discounting applied to a coupon stream that
/// pays more or less than the yield demands.
///
/// # Errors
/// Returns an error for zero periods, more than ten thousand, a
/// non-finite input, or a yield at or below `-100%` per period.
pub fn bond_price(face: f64, coupon: f64, ytm: f64, periods: usize) -> Result<f64, GeomError> {
    if periods == 0 || periods > 10_000 {
        return Err(GeomError::InvalidArgument("bond_price: bad period count"));
    }
    if !face.is_finite() || !coupon.is_finite() || !(ytm > -1.0) || !ytm.is_finite() {
        return Err(GeomError::InvalidArgument("bond_price: bad face, coupon or yield"));
    }
    let growth = 1.0 + ytm;
    let mut total = 0.0;
    for period in 1..=periods {
        let discount = growth.powi(-(period as i32));
        total += coupon * discount;
        if period == periods {
            total += face * discount;
        }
    }
    Ok(total)
}

/// The periodic yield that reproduces an observed bond price.
///
/// Unique whenever the coupons and face are non-negative and at least one
/// is positive: the price is then strictly decreasing in the yield, so
/// there is exactly one root. That is why a bond has *a* yield where a
/// project may have several internal rates of return -- the cashflows
/// after the purchase all point the same way.
///
/// # Errors
/// Returns an error for a non-positive price, bad bond parameters, or a
/// price no yield in `(-99.99%, 1e6)` reaches.
pub fn ytm_solve(
    price: f64,
    face: f64,
    coupon: f64,
    periods: usize,
) -> Result<f64, GeomError> {
    if !(price > 0.0) || !price.is_finite() {
        return Err(GeomError::InvalidArgument("ytm_solve: the price must be positive"));
    }
    if face < 0.0 || coupon < 0.0 || !(face + coupon > 0.0) {
        return Err(GeomError::InvalidArgument("ytm_solve: the cashflows must be non-negative"));
    }
    let residual = |ytm: f64| bond_price(face, coupon, ytm, periods).map_or(f64::NAN, |p| p - price);
    bracket_and_solve(&residual)?
        .ok_or(GeomError::Degenerate("no yield reproduces that price"))
}

/// The Macaulay duration in periods: the discounted-cashflow-weighted
/// average time to payment.
///
/// It is a *centre of mass*, which is why it has units of time and why a
/// zero-coupon bond's duration is exactly its maturity: all the weight
/// sits at one date. Coupons pull the centre earlier, so a higher coupon
/// always shortens duration at the same maturity.
///
/// # Errors
/// As [`bond_price`], plus a bond whose price comes out non-positive.
pub fn duration_macaulay(
    face: f64,
    coupon: f64,
    ytm: f64,
    periods: usize,
) -> Result<f64, GeomError> {
    let price = bond_price(face, coupon, ytm, periods)?;
    if !(price > 0.0) {
        return Err(GeomError::Degenerate("the bond has no positive price to weight against"));
    }
    let growth = 1.0 + ytm;
    let mut weighted = 0.0;
    for period in 1..=periods {
        let discount = growth.powi(-(period as i32));
        let flow = coupon + if period == periods { face } else { 0.0 };
        weighted += period as f64 * flow * discount;
    }
    Ok(weighted / price)
}

/// The modified duration: Macaulay duration divided by `1 + ytm`.
///
/// This is the one that answers "how much does the price move": it is
/// exactly `-(1/P) dP/dy`, so a modified duration of 7 means a price fall
/// of about 7% for a one-point rise in yield. The word "about" is doing
/// real work -- duration is the first derivative and the relationship is
/// convex, so it overstates the loss on a rise and understates the gain
/// on a fall. [`convexity`] is the correction.
///
/// # Errors
/// As [`duration_macaulay`].
pub fn duration_modified(
    face: f64,
    coupon: f64,
    ytm: f64,
    periods: usize,
) -> Result<f64, GeomError> {
    Ok(duration_macaulay(face, coupon, ytm, periods)? / (1.0 + ytm))
}

/// The convexity in periods squared: `(1/P) d2P/dy2`.
///
/// Always positive for an ordinary bond, which is the reason duration
/// alone is pessimistic in both directions. Between two bonds of equal
/// duration the more convex one gains more when yields move either way,
/// and its price reflects that -- convexity is not a free lunch, it is
/// paid for in yield.
///
/// # Errors
/// As [`duration_macaulay`].
pub fn convexity(face: f64, coupon: f64, ytm: f64, periods: usize) -> Result<f64, GeomError> {
    let price = bond_price(face, coupon, ytm, periods)?;
    if !(price > 0.0) {
        return Err(GeomError::Degenerate("the bond has no positive price to weight against"));
    }
    let growth = 1.0 + ytm;
    let mut total = 0.0;
    for period in 1..=periods {
        let flow = coupon + if period == periods { face } else { 0.0 };
        let n = period as f64;
        total += n * (n + 1.0) * flow * growth.powi(-(period as i32 + 2));
    }
    Ok(total / price)
}

/// One instrument on the curve to bootstrap: a bond quoted by price.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurveBond {
    /// Years to maturity; must be a whole number of coupon periods.
    pub maturity: f64,
    /// Coupon paid each period, per unit of face.
    pub coupon: f64,
    /// The observed price, per unit of face.
    pub price: f64,
    /// Coupon payments per year.
    pub frequency: f64,
}

/// Bootstraps a zero-coupon curve from bonds of increasing maturity,
/// returning `(maturity, continuously compounded zero rate)`.
///
/// Each bond is stripped in turn: its earlier coupons are discounted at
/// the zero rates already recovered, and whatever discount factor the
/// final payment needs to make the price work is the new point. The
/// method is exact and sequential, and that is also its weakness -- an
/// error in an early quote propagates into every later rate, and there is
/// no least-squares smoothing to absorb it.
///
/// Coupon dates that fall between known maturities are interpolated
/// linearly *in the zero rate*, which is a choice: interpolating in the
/// discount factor or the forward rate gives different curves from the
/// same bonds, and no market convention makes one correct.
///
/// # Errors
/// Returns an error for no bonds, maturities that do not increase, a
/// non-positive price, frequency or maturity, a maturity that is not a
/// whole number of periods, or a final cashflow whose implied discount
/// factor is non-positive -- which means the quotes admit an arbitrage.
pub fn bootstrap_zero_curve(bonds: &[CurveBond]) -> Result<Vec<(f64, f64)>, GeomError> {
    if bonds.is_empty() {
        return Err(GeomError::InvalidArgument("bootstrap_zero_curve: no bonds"));
    }
    if bonds.windows(2).any(|p| p[1].maturity <= p[0].maturity) {
        return Err(GeomError::InvalidArgument("the maturities must strictly increase"));
    }
    let mut curve: Vec<(f64, f64)> = Vec::with_capacity(bonds.len());
    for bond in bonds {
        if !(bond.maturity > 0.0) || !(bond.price > 0.0) || !(bond.frequency > 0.0) {
            return Err(GeomError::InvalidArgument("bootstrap_zero_curve: bad bond"));
        }
        let periods = bond.maturity * bond.frequency;
        if (periods - periods.round()).abs() > 1e-9 || periods.round() < 1.0 {
            return Err(GeomError::InvalidArgument(
                "a maturity is not a whole number of coupon periods",
            ));
        }
        let periods = periods.round() as usize;
        let mut discounted_coupons = 0.0;
        for period in 1..periods {
            let t = period as f64 / bond.frequency;
            let zero = interpolate_zero(&curve, t)?;
            discounted_coupons += bond.coupon * (-zero * t).exp();
        }
        let final_flow = 1.0 + bond.coupon;
        let remaining = bond.price - discounted_coupons;
        if !(remaining > 0.0) {
            return Err(GeomError::Degenerate(
                "the quotes leave no positive value for the final cashflow",
            ));
        }
        let discount = remaining / final_flow;
        if !(discount > 0.0) {
            return Err(GeomError::Degenerate("the implied discount factor is not positive"));
        }
        curve.push((bond.maturity, -discount.ln() / bond.maturity));
    }
    Ok(curve)
}

/// The zero rate at `t`, interpolated linearly and held flat beyond the
/// last point.
fn interpolate_zero(curve: &[(f64, f64)], t: f64) -> Result<f64, GeomError> {
    if curve.is_empty() {
        return Err(GeomError::Degenerate("a coupon falls before any zero rate is known"));
    }
    if t <= curve[0].0 {
        return Ok(curve[0].1);
    }
    if t >= curve[curve.len() - 1].0 {
        return Ok(curve[curve.len() - 1].1);
    }
    for pair in curve.windows(2) {
        if t <= pair[1].0 {
            let span = pair[1].0 - pair[0].0;
            let weight = (t - pair[0].0) / span;
            return Ok(pair[0].1 * (1.0 - weight) + pair[1].1 * weight);
        }
    }
    Ok(curve[curve.len() - 1].1)
}

/// The continuously compounded forward rate between two maturities:
/// `(z2 t2 - z1 t1) / (t2 - t1)`.
///
/// This is the rate the curve implies for borrowing from `t1` to `t2`, and
/// it follows from no-arbitrage alone: investing to `t2` must pay the same
/// as investing to `t1` and rolling. It is far more volatile than the zero
/// rates it comes from, because it is a *difference* of two nearly equal
/// products -- a small error in a long zero rate becomes a large error in
/// the forward, which is why bootstrapped curves are usually smoothed
/// before forwards are read off them.
///
/// # Errors
/// Returns an error for non-increasing maturities, a non-positive first
/// maturity, or a non-finite rate.
pub fn forward_rate(z1: f64, t1: f64, z2: f64, t2: f64) -> Result<f64, GeomError> {
    if !(t1 >= 0.0) || !(t2 > t1) || !z1.is_finite() || !z2.is_finite() {
        return Err(GeomError::InvalidArgument("forward_rate: bad maturities or rates"));
    }
    Ok((z2 * t2 - z1 * t1) / (t2 - t1))
}

/// The Nelson-Siegel zero rate at maturity `t`.
///
/// `b0 + (b1 + b2) (1 - e^-x)/x - b2 e^-x` with `x = t/tau`. The three
/// coefficients are usually read as level, slope and curvature: `b0` is
/// the long rate the curve tends to, `b0 + b1` is the short rate it starts
/// from, and `b2` is a hump whose position `tau` sets.
///
/// Four parameters is not many for a yield curve, and that is the point:
/// the shape cannot fit noise, so it smooths, and it extrapolates to a
/// finite long rate rather than diverging as a polynomial would. What it
/// cannot do is fit more than one hump, which is where the Svensson
/// extension with two decay terms is used instead.
///
/// # Errors
/// Returns an error for a non-positive `tau`, a negative `t`, or a
/// non-finite coefficient.
pub fn nelson_siegel(t: f64, b0: f64, b1: f64, b2: f64, tau: f64) -> Result<f64, GeomError> {
    if !(tau > 0.0) || t < 0.0 || ![b0, b1, b2, t].iter().all(|x| x.is_finite()) {
        return Err(GeomError::InvalidArgument("nelson_siegel: bad parameters"));
    }
    if t == 0.0 {
        // The limit as t -> 0 is the short rate b0 + b1.
        return Ok(b0 + b1);
    }
    let x = t / tau;
    let decay = (-x).exp();
    // `(1 - e^-x) / x` cancels catastrophically for small x: at
    // x = 1e-10 the subtraction keeps about six digits, and the slope
    // term inherits that error. `exp_m1` computes `e^-x - 1` accurately
    // all the way down, so negating it gives the numerator directly.
    let slope = -(-x).exp_m1() / x;
    Ok(b0 + (b1 + b2) * slope - b2 * decay)
}

/// Fits Nelson-Siegel to observed yields, returning `(b0, b1, b2, tau)`.
///
/// For a fixed `tau` the model is *linear* in the three coefficients, so
/// the fit is a three-parameter least squares that solves exactly. Only
/// `tau` needs searching, and it is searched over a grid rather than by
/// gradient because the objective in `tau` is not convex and a local
/// method lands wherever it started. That split -- exact where the model
/// is linear, brute force where it is not -- is what makes this reliable
/// where a five-parameter nonlinear search is not.
///
/// # Errors
/// Returns an error for fewer than four points, mismatched lengths, a
/// non-positive maturity, a non-finite value, or maturities that do not
/// determine the fit.
pub fn ns_fit(maturities: &[f64], yields: &[f64]) -> Result<(f64, f64, f64, f64), GeomError> {
    if maturities.len() < 4 || maturities.len() != yields.len() {
        return Err(GeomError::InvalidArgument("ns_fit needs at least four matched points"));
    }
    if maturities.iter().any(|t| !(*t > 0.0)) || yields.iter().any(|y| !y.is_finite()) {
        return Err(GeomError::InvalidArgument("ns_fit: bad observations"));
    }
    let longest = maturities.iter().fold(0.0f64, |a, b| a.max(*b));
    let grid = 400usize;
    let spacing = longest / grid as f64;
    let mut best: Option<(f64, [f64; 3], f64)> = None;
    let mut candidates: Vec<f64> = (1..=grid).map(|k| longest * k as f64 / grid as f64).collect();
    // Two golden-section refinements around the grid's winner. The
    // objective in tau is not convex, so the grid is what finds the right
    // basin and the refinement only sharpens it.
    for _ in 0..2 {
        for tau in candidates.clone() {
        let basis = |t: f64| -> [f64; 3] {
            let x = t / tau;
            let decay = (-x).exp();
            let slope = -(-x).exp_m1() / x;
            [1.0, slope, slope - decay]
        };
        // Normal equations for the three linear coefficients.
        let mut matrix = [[0.0f64; 3]; 3];
        let mut rhs = [0.0f64; 3];
        for (t, y) in maturities.iter().zip(yields.iter()) {
            let row = basis(*t);
            for i in 0..3 {
                rhs[i] += row[i] * y;
                for j in 0..3 {
                    matrix[i][j] += row[i] * row[j];
                }
            }
        }
        let Some(beta) = solve3(&matrix, &rhs) else { continue };
        let error: f64 = maturities
            .iter()
            .zip(yields.iter())
            .map(|(t, y)| {
                let row = basis(*t);
                (row[0] * beta[0] + row[1] * beta[1] + row[2] * beta[2] - y).powi(2)
            })
            .sum();
        if best.is_none_or(|(_, _, e)| error < e) {
            best = Some((tau, beta, error));
        }
        }
        let Some((centre, _, _)) = best else { break };
        let width = spacing;
        candidates = (0..=40)
            .map(|k| centre - width + 2.0 * width * k as f64 / 40.0)
            .filter(|t| *t > 1e-6)
            .collect();
    }
    let (tau, beta, _) =
        best.ok_or(GeomError::Degenerate("no decay parameter gave a solvable fit"))?;
    // Matching coefficients between `b0 + (b1 + b2) slope - b2 decay` and
    // the fitted `beta0 + beta1 slope + beta2 (slope - decay)` gives
    // beta1 = b1 and beta2 = b2 directly: the `slope - decay` basis vector
    // already carries the `+ b2 slope` term.
    Ok((beta[0], beta[1], beta[2], tau))
}

/// Gaussian elimination on a 3x3 system, or `None` if it is singular.
fn solve3(matrix: &[[f64; 3]; 3], rhs: &[f64; 3]) -> Option<[f64; 3]> {
    let mut a = [
        [matrix[0][0], matrix[0][1], matrix[0][2], rhs[0]],
        [matrix[1][0], matrix[1][1], matrix[1][2], rhs[1]],
        [matrix[2][0], matrix[2][1], matrix[2][2], rhs[2]],
    ];
    let scale = a.iter().flatten().fold(0.0f64, |m, v| m.max(v.abs())).max(1.0);
    for column in 0..3 {
        let pivot = (column..3).max_by(|i, j| {
            a[*i][column]
                .abs()
                .partial_cmp(&a[*j][column].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
        a.swap(column, pivot);
        if a[column][column].abs() < 1e-12 * scale {
            return None;
        }
        for row in 0..3 {
            if row == column {
                continue;
            }
            let factor = a[row][column] / a[column][column];
            for entry in column..4 {
                a[row][entry] -= factor * a[column][entry];
            }
        }
    }
    Some([a[0][3] / a[0][0], a[1][3] / a[1][1], a[2][3] / a[2][2]])
}

// ---------------------------------------------------------------------------
// Short-rate models
// ---------------------------------------------------------------------------

/// The Vasicek zero-coupon bond price under `dr = kappa (theta - r) dt +
/// sigma dW`.
///
/// `P(t) = A(t) e^(-B(t) r0)` with `B = (1 - e^(-kappa t))/kappa`. The
/// model is affine and Gaussian, which is what makes the price a closed
/// form and also what makes the rate able to go negative -- for decades
/// that was the standard objection to Vasicek, and since 2014 it has been
/// the reason to use it.
///
/// The long-run mean of the *rate* is `theta`, but the long-run mean of
/// the yield is `theta - sigma^2/(2 kappa^2)`, lower by a convexity term
/// that grows with volatility. Discounting is convex in the rate, so
/// uncertainty about future rates makes bonds worth more than the average
/// rate alone would say.
///
/// # Errors
/// Returns an error for a non-positive `kappa`, a negative `sigma`, a
/// negative maturity, or a non-finite parameter.
pub fn vasicek_bond_price(
    r0: f64,
    kappa: f64,
    theta: f64,
    sigma: f64,
    t: f64,
) -> Result<f64, GeomError> {
    if !(kappa > 0.0) || sigma < 0.0 || t < 0.0 || ![r0, theta, sigma, t].iter().all(|x| x.is_finite())
    {
        return Err(GeomError::InvalidArgument("vasicek_bond_price: bad parameters"));
    }
    if t == 0.0 {
        return Ok(1.0);
    }
    let b = (1.0 - (-kappa * t).exp()) / kappa;
    let long_run = theta - sigma * sigma / (2.0 * kappa * kappa);
    let log_a = long_run * (b - t) - sigma * sigma * b * b / (4.0 * kappa);
    Ok((log_a - b * r0).exp())
}

/// The Cox-Ingersoll-Ross zero-coupon bond price under
/// `dr = kappa (theta - r) dt + sigma sqrt(r) dW`.
///
/// The `sqrt(r)` diffusion is what keeps the rate non-negative: volatility
/// vanishes as the rate approaches zero, so the process cannot cross it.
/// Whether zero is even reached depends on the Feller condition
/// `2 kappa theta >= sigma^2` -- satisfied, the rate stays strictly
/// positive; violated, it touches zero and reflects. The price is still a
/// closed form either way, and [`cir_feller_condition`] reports which
/// regime the parameters are in.
///
/// The formula raises a base tending to one to the power
/// `2 kappa theta / sigma^2`, so it loses precision as `sigma` shrinks: at
/// `sigma = 1e-6` the answer is off by about `1e-6` relative, which is a
/// thousand times larger than the convexity effect it is trying to
/// capture. `sigma = 0` is handled exactly by the deterministic limit;
/// between them, below roughly `1e-5`, the price is dominated by rounding
/// and [`vasicek_bond_price`] with a zero volatility is the better answer.
///
/// # Errors
/// Returns an error for a negative initial rate, a non-positive `kappa` or
/// `theta`, a negative `sigma`, a negative maturity, or a non-finite
/// parameter.
pub fn cir_bond_price(
    r0: f64,
    kappa: f64,
    theta: f64,
    sigma: f64,
    t: f64,
) -> Result<f64, GeomError> {
    if r0 < 0.0 || !(kappa > 0.0) || !(theta > 0.0) || sigma < 0.0 || t < 0.0 {
        return Err(GeomError::InvalidArgument("cir_bond_price: bad parameters"));
    }
    if ![r0, kappa, theta, sigma, t].iter().all(|x| x.is_finite()) {
        return Err(GeomError::InvalidArgument("cir_bond_price: a parameter is not finite"));
    }
    if t == 0.0 {
        return Ok(1.0);
    }
    if sigma == 0.0 {
        // With no diffusion the rate is deterministic and both models
        // give the same price. The affine formula cannot be evaluated
        // here: its exponent is `2 kappa theta / sigma^2`, and the base
        // tends to one, so the limit arrives as `1^infinity` and floating
        // point resolves it to one -- silently dropping the whole `A(t)`
        // factor and returning `e^(-B r0)` alone.
        return vasicek_bond_price(r0, kappa, theta, 0.0, t);
    }
    let gamma = (kappa * kappa + 2.0 * sigma * sigma).sqrt();
    let expanded = (gamma * t).exp() - 1.0;
    let denominator = (gamma + kappa) * expanded + 2.0 * gamma;
    if !(denominator > 0.0) {
        return Err(GeomError::Degenerate("the CIR denominator vanished"));
    }
    let b = 2.0 * expanded / denominator;
    let a = (2.0 * gamma * ((kappa + gamma) * t / 2.0).exp() / denominator)
        .powf(2.0 * kappa * theta / (sigma * sigma));
    Ok(a * (-b * r0).exp())
}

/// Whether the Feller condition `2 kappa theta >= sigma^2` holds, which
/// decides whether a CIR rate can reach zero.
#[must_use]
pub fn cir_feller_condition(kappa: f64, theta: f64, sigma: f64) -> bool {
    2.0 * kappa * theta >= sigma * sigma
}

// ---------------------------------------------------------------------------
// Amortisation
// ---------------------------------------------------------------------------

/// The level payment that repays `principal` over `n` periods at the
/// periodic rate `rate`.
///
/// `P r / (1 - (1+r)^-n)`, which is the principal divided by the annuity
/// factor. At zero rate it degenerates to `P/n`, handled directly.
///
/// # Errors
/// Returns an error for a non-positive principal, zero periods, more than
/// a hundred thousand periods, or a rate at or below `-100%`.
pub fn mortgage_payment(principal: f64, rate: f64, n: usize) -> Result<f64, GeomError> {
    if !(principal > 0.0) || !principal.is_finite() || n == 0 || n > 100_000 {
        return Err(GeomError::InvalidArgument("mortgage_payment: bad principal or term"));
    }
    if !(rate > -1.0) || !rate.is_finite() {
        return Err(GeomError::InvalidArgument("mortgage_payment: bad rate"));
    }
    if rate == 0.0 {
        return Ok(principal / n as f64);
    }
    let factor = (1.0 + rate).powi(-(n as i32));
    Ok(principal * rate / (1.0 - factor))
}

/// The amortisation schedule as `(payment, interest, principal, balance)`
/// per period.
///
/// The payment is level; what changes is its split. Early on almost all of
/// it is interest, because interest is charged on a balance that has
/// barely fallen, and the crossover to mostly-principal comes surprisingly
/// late -- past the halfway point of the term for any rate above a few
/// percent. That is the single most counterintuitive fact about a
/// mortgage and it falls straight out of the arithmetic.
///
/// The final balance is forced to exactly zero, absorbing the accumulated
/// rounding into the last principal payment, which is what a lender does.
///
/// # Errors
/// As [`mortgage_payment`].
pub fn amortization_schedule(
    principal: f64,
    rate: f64,
    n: usize,
) -> Result<Vec<(f64, f64, f64, f64)>, GeomError> {
    let payment = mortgage_payment(principal, rate, n)?;
    let mut balance = principal;
    let mut schedule = Vec::with_capacity(n);
    for period in 1..=n {
        let interest = balance * rate;
        let mut repaid = payment - interest;
        if period == n {
            // The last payment clears whatever is left, so rounding never
            // leaves a balance behind.
            repaid = balance;
        }
        balance -= repaid;
        schedule.push((interest + repaid, interest, repaid, balance.max(0.0)));
    }
    if let Some(last) = schedule.last_mut() {
        last.3 = 0.0;
    }
    Ok(schedule)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_compounding_convention_changes_the_number_and_not_the_money() {
        // 10% quoted three ways over a year: the growth factors differ,
        // and the differences are small over one period and decisive over
        // thirty.
        assert!((discount_factor(0.1, 1.0, Compounding::Annual).unwrap() - 1.0 / 1.1).abs() < 1e-15);
        assert!(
            (discount_factor(0.1, 1.0, Compounding::SemiAnnual).unwrap() - 1.0 / 1.1025).abs()
                < 1e-15
        );
        assert!(
            (discount_factor(0.1, 1.0, Compounding::Continuous).unwrap() - (-0.1f64).exp()).abs()
                < 1e-15
        );
        // More frequent compounding discounts harder at the same quoted
        // rate, and continuous is the limit of the sequence.
        let mut previous = f64::INFINITY;
        for convention in [
            Compounding::Annual,
            Compounding::SemiAnnual,
            Compounding::Quarterly,
            Compounding::Monthly,
            Compounding::Continuous,
        ] {
            let factor = discount_factor(0.1, 1.0, convention).unwrap();
            assert!(factor < previous, "{convention:?} did not discount harder");
            previous = factor;
        }
        // Over thirty years the annual and continuous conventions differ
        // by more than a fifth of the present value.
        // 0.0573 against 0.0498: fifteen percent more present value from
        // the convention alone.
        let annual = discount_factor(0.1, 30.0, Compounding::Annual).unwrap();
        let continuous = discount_factor(0.1, 30.0, Compounding::Continuous).unwrap();
        let ratio = annual / continuous;
        assert!((1.15..1.16).contains(&ratio), "the ratio was {ratio}");
    }

    #[test]
    fn converting_a_rate_between_conventions_leaves_the_growth_factor_alone() {
        // A round trip through every pair must return the rate exactly,
        // and the converted rate must discount to the same number.
        let conventions = [
            Compounding::Annual,
            Compounding::SemiAnnual,
            Compounding::Quarterly,
            Compounding::Monthly,
            Compounding::Continuous,
        ];
        for from in conventions {
            for to in conventions {
                for rate in [-0.02f64, 0.001, 0.05, 0.35, 1.5] {
                    let moved = equivalent_rate(rate, from, to).unwrap();
                    let back = equivalent_rate(moved, to, from).unwrap();
                    assert!((back - rate).abs() < 1e-12, "{from:?}->{to:?} at {rate} gave {back}");
                    for t in [0.5f64, 1.0, 7.0] {
                        let here = discount_factor(rate, t, from).unwrap();
                        let there = discount_factor(moved, t, to).unwrap();
                        assert!(
                            (here - there).abs() < 1e-12,
                            "{from:?}->{to:?}: {here} against {there}"
                        );
                    }
                }
            }
        }
        // The textbook conversion: 10% semi-annual is 9.7580% continuous.
        let continuous =
            equivalent_rate(0.1, Compounding::SemiAnnual, Compounding::Continuous).unwrap();
        assert!((continuous - 2.0 * 1.05f64.ln()).abs() < 1e-15, "got {continuous}");
        assert!((continuous - 0.097_580_328_338_864_0).abs() < 1e-15, "got {continuous}");
    }

    #[test]
    fn the_internal_rate_of_return_is_the_rate_that_zeroes_the_value() {
        let flows = [-1000.0, 300.0, 400.0, 500.0];
        let rate = irr(&flows).unwrap().expect("one sign change, so one rate");
        assert!((rate - 0.088_963_394_693).abs() < 1e-9, "the rate came out at {rate}");
        assert!(npv(rate, &flows).unwrap().abs() < 1e-9, "the value at its own rate is not zero");
        // Net present value falls as the discount rate rises, which is
        // what makes the root unique here.
        let mut previous = f64::INFINITY;
        for r in [-0.5f64, 0.0, 0.05, 0.2, 1.0, 5.0] {
            let value = npv(r, &flows).unwrap();
            assert!(value < previous, "the value rose at {r}");
            previous = value;
        }
    }

    #[test]
    fn cashflows_that_change_sign_twice_get_no_single_rate() {
        // Descartes bounds the positive roots by the sign changes, so one
        // change guarantees at most one rate. Two changes can give two
        // rates or none, and reporting either as *the* return would be a
        // mistake. Here both 0% and 100% zero the value.
        let alternating = [-100.0, 230.0, -132.0];
        assert_eq!(irr(&alternating).unwrap(), None);
        assert!(npv(0.1, &alternating).unwrap().abs() < 1e-12, "0.1 is a root");
        assert!(npv(0.2, &alternating).unwrap().abs() < 1e-12, "0.2 is a root");

        // No sign change at all means no rate either.
        assert_eq!(irr(&[100.0, 200.0, 300.0]).unwrap(), None);
        assert_eq!(irr(&[-100.0, -200.0]).unwrap(), None);
        assert!(irr(&[100.0]).is_err());
        assert!(irr(&[100.0, f64::NAN]).is_err());
        assert!(npv(-1.0, &[1.0, 2.0]).is_err());
        assert!(npv(0.1, &[]).is_err());
    }

    #[test]
    fn irregular_dates_need_the_fractional_discounting_xirr_does() {
        // The same flows on the same schedule: forcing them onto period
        // boundaries changes the answer by real money.
        let times = [0.0, 0.5, 1.2, 2.0];
        let flows = [-1000.0, 300.0, 400.0, 500.0];
        let rate = xirr(&times, &flows).unwrap().expect("one sign change");
        let value: f64 =
            times.iter().zip(flows.iter()).map(|(t, c)| c * (1.0 + rate).powf(-t)).sum();
        assert!(value.abs() < 1e-9, "the value at its own rate is {value}");
        // Paid earlier than the annual schedule assumes, so the return is
        // higher than the whole-period IRR.
        let annual = irr(&flows).unwrap().unwrap();
        assert!(rate > annual, "{rate} against the whole-period {annual}");

        // On whole years the two agree exactly.
        let whole = xirr(&[0.0, 1.0, 2.0, 3.0], &flows).unwrap().unwrap();
        assert!((whole - annual).abs() < 1e-9, "{whole} against {annual}");

        assert!(xirr(&[0.0, 1.0], &[1.0]).is_err());
        assert!(xirr(&[1.0, 2.0], &[-1.0, 2.0]).is_err(), "times must start at zero");
        assert!(xirr(&[0.0, 1.0, 1.0], &[-1.0, 1.0, 1.0]).is_err(), "times must increase");
    }

    #[test]
    fn a_bond_prices_at_par_when_its_coupon_equals_its_yield() {
        // Not a market observation but arithmetic: the discounting is the
        // yield's own, applied to a stream paying exactly what the yield
        // asks.
        for rate in [0.001f64, 0.025, 0.07, 0.3] {
            for periods in [1usize, 5, 10, 60] {
                let price = bond_price(100.0, 100.0 * rate, rate, periods).unwrap();
                assert!((price - 100.0).abs() < 1e-10, "at {rate} over {periods} it was {price}");
            }
        }
        // Above the yield it trades over par, below it under.
        assert!(bond_price(100.0, 4.0, 0.025, 10).unwrap() > 100.0);
        assert!(bond_price(100.0, 1.0, 0.025, 10).unwrap() < 100.0);
        // And a zero-coupon bond is just the discount factor.
        let zero = bond_price(100.0, 0.0, 0.03, 10).unwrap();
        assert!((zero - 100.0 * 1.03f64.powi(-10)).abs() < 1e-12);
    }

    #[test]
    fn solving_for_the_yield_inverts_the_price_it_was_given() {
        for coupon in [0.0f64, 1.0, 3.0, 12.0] {
            for periods in [1usize, 4, 20, 100] {
                for ytm in [-0.02f64, 0.001, 0.045, 0.25] {
                    let price = bond_price(100.0, coupon, ytm, periods).unwrap();
                    let recovered = ytm_solve(price, 100.0, coupon, periods).unwrap();
                    assert!(
                        (recovered - ytm).abs() < 1e-9,
                        "coupon {coupon} over {periods}: {recovered} not {ytm}"
                    );
                }
            }
        }
        // The price is strictly decreasing in the yield, which is what
        // makes the root unique -- unlike an internal rate of return.
        let mut previous = f64::INFINITY;
        for ytm in [-0.05f64, 0.0, 0.02, 0.1, 0.5, 2.0] {
            let price = bond_price(100.0, 3.0, ytm, 20).unwrap();
            assert!(price < previous, "the price rose at {ytm}");
            previous = price;
        }
        assert!(ytm_solve(0.0, 100.0, 3.0, 10).is_err());
        assert!(ytm_solve(100.0, 0.0, 0.0, 10).is_err());
        assert!(bond_price(100.0, 3.0, 0.02, 0).is_err());
        assert!(bond_price(100.0, 3.0, -1.0, 10).is_err());
    }

    #[test]
    fn a_zero_coupon_bonds_duration_is_exactly_its_maturity() {
        // Duration is a centre of mass, and a zero-coupon bond has all of
        // its weight at one date. Coupons pull it earlier, always.
        for periods in [1usize, 5, 30] {
            for ytm in [0.0f64, 0.03, 0.15] {
                let zero = duration_macaulay(100.0, 0.0, ytm, periods).unwrap();
                assert!((zero - periods as f64).abs() < 1e-10, "got {zero} for {periods}");
            }
        }
        let mut previous = f64::INFINITY;
        for coupon in [0.0f64, 1.0, 3.0, 8.0, 20.0] {
            let duration = duration_macaulay(100.0, coupon, 0.04, 30).unwrap();
            assert!(duration < previous, "a coupon of {coupon} did not shorten duration");
            assert!(duration > 0.0 && duration <= 30.0);
            previous = duration;
        }
    }

    #[test]
    fn duration_and_convexity_are_the_derivatives_they_claim_to_be() {
        // Modified duration is exactly -(1/P) dP/dy and convexity is
        // (1/P) d2P/dy2. Checking them against differences of the price is
        // what catches the factor of (1 + y) that separates the two
        // durations.
        for (coupon, ytm, periods) in
            [(3.0f64, 0.025f64, 10usize), (0.0, 0.05, 30), (8.0, 0.12, 5), (1.0, 0.001, 40)]
        {
            let price = |y: f64| bond_price(100.0, coupon, y, periods).unwrap();
            let base = price(ytm);
            let h = 1e-5;
            let modified = duration_modified(100.0, coupon, ytm, periods).unwrap();
            let expected = -(price(ytm + h) - price(ytm - h)) / (2.0 * h) / base;
            assert!(
                (modified - expected).abs() < 1e-6,
                "modified duration {modified} against {expected}"
            );
            // And Macaulay is modified times one plus the yield.
            let macaulay = duration_macaulay(100.0, coupon, ytm, periods).unwrap();
            assert!((macaulay - modified * (1.0 + ytm)).abs() < 1e-12);

            // The second difference needs a larger step: dividing by h^2
            // amplifies the cancellation.
            let hc = 1e-3;
            let second = |h: f64| (price(ytm + h) - 2.0 * base + price(ytm - h)) / (h * h) / base;
            let extrapolated = (4.0 * second(0.5 * hc) - second(hc)) / 3.0;
            let convex = convexity(100.0, coupon, ytm, periods).unwrap();
            assert!(convex > 0.0, "convexity was not positive");
            assert!(
                (convex - extrapolated).abs() < 1e-4 * convex,
                "convexity {convex} against {extrapolated}"
            );
        }
    }

    #[test]
    fn duration_alone_is_pessimistic_in_both_directions() {
        // The price is convex in the yield, so the linear estimate
        // overstates the loss on a rise and understates the gain on a
        // fall. Adding the convexity term fixes both.
        let (face, coupon, ytm, periods) = (100.0, 3.0, 0.04, 30);
        let base = bond_price(face, coupon, ytm, periods).unwrap();
        let modified = duration_modified(face, coupon, ytm, periods).unwrap();
        let convex = convexity(face, coupon, ytm, periods).unwrap();
        for shift in [-0.02f64, -0.01, 0.01, 0.02] {
            let actual = bond_price(face, coupon, ytm + shift, periods).unwrap();
            let linear = base * (1.0 - modified * shift);
            assert!(actual > linear, "the linear estimate beat the price at {shift}");
            let quadratic = base * (1.0 - modified * shift + 0.5 * convex * shift * shift);
            assert!(
                (quadratic - actual).abs() < 0.2 * (linear - actual).abs(),
                "the convexity term did not improve the estimate at {shift}"
            );
        }
    }

    /// Prices a bond off a known continuous zero curve.
    fn price_from_curve(zero: &dyn Fn(f64) -> f64, coupon: f64, years: usize) -> f64 {
        let mut price = 0.0;
        for period in 1..=years {
            let t = period as f64;
            price += coupon * (-zero(t) * t).exp();
        }
        price + (-zero(years as f64) * years as f64).exp()
    }

    #[test]
    fn bootstrapping_recovers_the_curve_the_bonds_were_priced_from() {
        // The method is exact and sequential, so given prices that came
        // from a curve it returns that curve to rounding -- not to a
        // tolerance. Anything else would be an error in the stripping.
        let truth = |t: f64| 0.02 + 0.015 * (1.0 - (-t / 2.0).exp());
        for coupon in [0.0f64, 0.01, 0.03, 0.09] {
            let bonds: Vec<CurveBond> = (1..=6)
                .map(|years| CurveBond {
                    maturity: years as f64,
                    coupon,
                    price: price_from_curve(&truth, coupon, years),
                    frequency: 1.0,
                })
                .collect();
            let curve = bootstrap_zero_curve(&bonds).unwrap();
            assert_eq!(curve.len(), 6);
            for (t, zero) in &curve {
                assert!(
                    (zero - truth(*t)).abs() < 1e-12,
                    "coupon {coupon} at t={t}: {zero} against {}",
                    truth(*t)
                );
            }
        }
    }

    #[test]
    fn a_flat_curve_bootstraps_flat_whatever_the_coupons() {
        // The simplest sanity check with an answer known in advance, and
        // the one that catches an interpolation that leaks.
        let flat = 0.04;
        for coupon in [0.0f64, 0.04, 0.15] {
            let bonds: Vec<CurveBond> = (1..=8)
                .map(|years| CurveBond {
                    maturity: years as f64,
                    coupon,
                    price: price_from_curve(&|_| flat, coupon, years),
                    frequency: 1.0,
                })
                .collect();
            for (_, zero) in bootstrap_zero_curve(&bonds).unwrap() {
                assert!((zero - flat).abs() < 1e-12, "got {zero} on a flat curve");
            }
        }
    }

    #[test]
    fn bootstrapping_refuses_quotes_that_would_imply_an_arbitrage() {
        let sound = CurveBond { maturity: 1.0, coupon: 0.03, price: 1.0, frequency: 1.0 };
        assert!(bootstrap_zero_curve(&[sound]).is_ok());
        assert!(bootstrap_zero_curve(&[]).is_err());
        // Out of order.
        let second = CurveBond { maturity: 0.5, ..sound };
        assert!(bootstrap_zero_curve(&[sound, second]).is_err());
        // A maturity that is not a whole number of coupon periods.
        assert!(bootstrap_zero_curve(&[CurveBond { maturity: 1.5, ..sound }]).is_err());
        assert!(bootstrap_zero_curve(&[CurveBond { price: 0.0, ..sound }]).is_err());
        assert!(bootstrap_zero_curve(&[CurveBond { frequency: 0.0, ..sound }]).is_err());
        // A five-year bond priced so low that its coupons alone exceed it
        // leaves nothing for the principal, which is not a curve but an
        // arbitrage.
        let cheap = [
            CurveBond { maturity: 1.0, coupon: 0.5, price: 1.4, frequency: 1.0 },
            CurveBond { maturity: 2.0, coupon: 0.5, price: 0.4, frequency: 1.0 },
        ];
        assert!(bootstrap_zero_curve(&cheap).is_err());
    }

    #[test]
    fn a_forward_rate_is_what_stops_the_curve_from_arbitraging_itself() {
        // Investing to t2 must pay the same as investing to t1 and rolling
        // at the forward. That is the definition, and it is checkable
        // directly against the discount factors.
        for (z1, t1, z2, t2) in
            [(0.02f64, 1.0f64, 0.03f64, 2.0f64), (0.05, 0.25, 0.045, 10.0), (0.0, 0.0, 0.04, 5.0)]
        {
            let forward = forward_rate(z1, t1, z2, t2).unwrap();
            let rolled = (-z1 * t1).exp() * (-forward * (t2 - t1)).exp();
            let direct = (-z2 * t2).exp();
            assert!((rolled - direct).abs() < 1e-14, "{rolled} against {direct}");
        }
        // A rising curve implies forwards above the spot rates it came
        // from, which is the sense in which a steep curve "predicts" rate
        // rises -- it is arithmetic, not a forecast.
        let forward = forward_rate(0.02, 1.0, 0.03, 2.0).unwrap();
        assert!(forward > 0.03, "the forward {forward} did not exceed the longer zero rate");
        assert!((forward - 0.04).abs() < 1e-14);
        // A flat curve implies a forward equal to it.
        assert!((forward_rate(0.035, 2.0, 0.035, 7.0).unwrap() - 0.035).abs() < 1e-15);
        assert!(forward_rate(0.02, 2.0, 0.03, 1.0).is_err());
        assert!(forward_rate(0.02, 1.0, 0.03, 1.0).is_err());
    }

    #[test]
    fn nelson_siegel_starts_at_the_short_rate_and_ends_at_the_long_one() {
        let (b0, b1, b2, tau) = (0.045, -0.02, 0.03, 2.5);
        // At zero maturity the slope term is one and the curvature term
        // cancels: the limit is b0 + b1.
        assert!((nelson_siegel(0.0, b0, b1, b2, tau).unwrap() - (b0 + b1)).abs() < 1e-15);
        // Approached from above it agrees, so the limit is continuous.
        assert!((nelson_siegel(1e-9, b0, b1, b2, tau).unwrap() - (b0 + b1)).abs() < 1e-8);
        // Far out both shape terms vanish and only the level survives.
        assert!((nelson_siegel(1e6, b0, b1, b2, tau).unwrap() - b0).abs() < 1e-4);
        assert!((nelson_siegel(1e9, b0, b1, b2, tau).unwrap() - b0).abs() < 1e-7);
        // With a positive curvature the curve humps above the straight
        // line between its two ends.
        let humped = nelson_siegel(tau, b0, b1, b2, tau).unwrap();
        assert!(humped > b0 + b1 && humped < b0, "the hump sat at {humped}");
        assert!(nelson_siegel(1.0, b0, b1, b2, 0.0).is_err());
        assert!(nelson_siegel(-1.0, b0, b1, b2, tau).is_err());
    }

    #[test]
    fn the_nelson_siegel_fit_recovers_the_curve_it_was_shown() {
        let (b0, b1, b2, tau) = (0.045, -0.02, 0.03, 2.5);
        let maturities = [0.25f64, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0];
        let yields: Vec<f64> =
            maturities.iter().map(|t| nelson_siegel(*t, b0, b1, b2, tau).unwrap()).collect();
        let (f0, f1, f2, ftau) = ns_fit(&maturities, &yields).unwrap();
        // The parameters come back, which they need not in general -- but
        // with noiseless data from the model itself they do.
        assert!((f0 - b0).abs() < 1e-4, "level {f0}");
        assert!((f1 - b1).abs() < 1e-3, "slope {f1}");
        assert!((f2 - b2).abs() < 1e-3, "curvature {f2}");
        assert!((ftau - tau).abs() < 0.05, "decay {ftau}");
        // And the fitted curve matches everywhere, which is what a smile
        // or a curve fit is actually for.
        for t in [0.1f64, 0.75, 4.0, 15.0, 25.0, 40.0] {
            let want = nelson_siegel(t, b0, b1, b2, tau).unwrap();
            let got = nelson_siegel(t, f0, f1, f2, ftau).unwrap();
            assert!((got - want).abs() < 1e-5, "at t={t}: {got} against {want}");
        }
        assert!(ns_fit(&[1.0, 2.0], &[0.02, 0.03]).is_err());
        assert!(ns_fit(&[1.0, 2.0, 3.0, 4.0], &[0.02, 0.03, 0.03]).is_err());
        assert!(ns_fit(&[0.0, 2.0, 3.0, 4.0], &[0.02; 4]).is_err());
    }

    #[test]
    fn a_short_rate_model_with_no_diffusion_is_just_deterministic_discounting() {
        // With sigma at zero the rate follows an ODE with a closed-form
        // integral, and both models must reproduce it exactly -- and each
        // other. This is the check that catches a mis-set A(t) factor,
        // which no plausibility check on the price would.
        let (r0, kappa, theta) = (0.03, 0.5, 0.04);
        for t in [0.5f64, 5.0, 30.0] {
            // integral of theta + (r0 - theta) e^(-kappa s) over [0, t]
            let integral =
                theta * t + (r0 - theta) * (1.0 - (-kappa * t).exp()) / kappa;
            let expected = (-integral).exp();
            let vasicek = vasicek_bond_price(r0, kappa, theta, 0.0, t).unwrap();
            assert!((vasicek - expected).abs() < 1e-13, "Vasicek gave {vasicek} not {expected}");
            let cir = cir_bond_price(r0, kappa, theta, 0.0, t).unwrap();
            assert!((cir - expected).abs() < 1e-13, "CIR gave {cir} not {expected}");
        }
    }

    #[test]
    fn volatility_makes_a_bond_worth_more_than_its_average_rate_would_say() {
        // Discounting is convex in the rate, so uncertainty about future
        // rates raises the price. In Vasicek the effect is explicit: the
        // long-run yield is theta minus sigma^2/(2 kappa^2).
        let (r0, kappa, theta) = (0.04, 0.5, 0.04);
        let mut previous = 0.0;
        for sigma in [0.0f64, 0.005, 0.01, 0.02, 0.04] {
            let price = vasicek_bond_price(r0, kappa, theta, sigma, 10.0).unwrap();
            assert!(price > previous, "volatility {sigma} did not raise the price");
            previous = price;
        }
        // The long yield falls by the convexity term, which is 1/(2 k^2)
        // times the variance.
        let sigma = 0.02;
        let long = 60.0;
        let yield_at_long =
            -vasicek_bond_price(theta, kappa, theta, sigma, long).unwrap().ln() / long;
        let expected = theta - sigma * sigma / (2.0 * kappa * kappa);
        assert!((yield_at_long - expected).abs() < 2e-3, "{yield_at_long} against {expected}");

        // The same is true in CIR, which has its own convexity term.
        let mut previous = 0.0;
        for sigma in [0.0f64, 0.02, 0.05, 0.1] {
            let price = cir_bond_price(r0, kappa, theta, sigma, 10.0).unwrap();
            assert!(price > previous, "CIR volatility {sigma} did not raise the price");
            previous = price;
        }
    }

    #[test]
    fn both_short_rate_models_price_a_bond_the_way_a_bond_behaves() {
        // Worth one at maturity, falling with maturity, and never above
        // one for a positive rate.
        for (r0, kappa, theta, sigma) in
            [(0.03f64, 0.5f64, 0.04f64, 0.01f64), (0.001, 2.0, 0.05, 0.03), (0.08, 0.2, 0.02, 0.02)]
        {
            assert!((vasicek_bond_price(r0, kappa, theta, sigma, 0.0).unwrap() - 1.0).abs() < 1e-15);
            assert!((cir_bond_price(r0, kappa, theta, sigma, 0.0).unwrap() - 1.0).abs() < 1e-15);
            let mut previous = 1.0;
            for t in [0.1f64, 1.0, 5.0, 20.0, 50.0] {
                for price in [
                    vasicek_bond_price(r0, kappa, theta, sigma, t).unwrap(),
                    cir_bond_price(r0, kappa, theta, sigma, t).unwrap(),
                ] {
                    assert!(price > 0.0 && price < 1.0, "at t={t} the price was {price}");
                }
                let price = cir_bond_price(r0, kappa, theta, sigma, t).unwrap();
                assert!(price < previous, "the CIR price rose at t={t}");
                previous = price;
            }
        }
        // The Feller condition decides whether a CIR rate can reach zero,
        // and it is a statement about the parameters alone.
        assert!(cir_feller_condition(0.5, 0.04, 0.05), "2*0.5*0.04 = 0.04 exceeds 0.0025");
        assert!(!cir_feller_condition(0.5, 0.04, 0.3), "0.04 does not reach 0.09");
        assert!(cir_bond_price(0.03, 0.0, 0.04, 0.01, 1.0).is_err());
        assert!(cir_bond_price(0.03, 0.5, 0.0, 0.01, 1.0).is_err());
        assert!(cir_bond_price(-0.01, 0.5, 0.04, 0.01, 1.0).is_err());
        assert!(vasicek_bond_price(0.03, -1.0, 0.04, 0.01, 1.0).is_err());
        assert!(vasicek_bond_price(0.03, 0.5, 0.04, -0.01, 1.0).is_err());
    }

    #[test]
    fn a_level_payment_repays_the_loan_exactly_and_no_more() {
        // The schedule's principal repayments must sum to the loan and its
        // balance must reach zero, at every rate including nothing at all.
        for rate in [0.0f64, 0.0001, 0.05 / 12.0, 0.02] {
            for n in [1usize, 12, 360] {
                let principal = 300_000.0;
                let payment = mortgage_payment(principal, rate, n).unwrap();
                let schedule = amortization_schedule(principal, rate, n).unwrap();
                assert_eq!(schedule.len(), n);
                assert!((schedule.last().unwrap().3).abs() < 1e-9, "a balance was left over");
                let repaid: f64 = schedule.iter().map(|row| row.2).sum();
                assert!(
                    (repaid - principal).abs() < 1e-6,
                    "at rate {rate} over {n} it repaid {repaid}"
                );
                // Every payment is the level one, and every row's parts add
                // up to it.
                for row in &schedule {
                    assert!((row.0 - payment).abs() < 1e-6, "a payment was {} not {payment}", row.0);
                    assert!((row.1 + row.2 - row.0).abs() < 1e-9);
                    assert!(row.1 >= -1e-12 && row.2 > -1e-12);
                }
                // The balance falls monotonically.
                let mut previous = principal;
                for row in &schedule {
                    assert!(row.3 <= previous + 1e-9, "the balance rose");
                    previous = row.3;
                }
            }
        }
        assert!(mortgage_payment(0.0, 0.01, 12).is_err());
        assert!(mortgage_payment(1000.0, 0.01, 0).is_err());
        assert!(mortgage_payment(1000.0, -1.0, 12).is_err());
        // At zero rate the payment is just the principal split evenly.
        assert!((mortgage_payment(1200.0, 0.0, 12).unwrap() - 100.0).abs() < 1e-12);
    }

    #[test]
    fn a_mortgage_is_mostly_interest_until_past_its_halfway_point() {
        // The single most counterintuitive fact about level repayment, and
        // it falls straight out of the arithmetic: interest is charged on
        // a balance that has barely moved, so the crossover comes late.
        let schedule = amortization_schedule(300_000.0, 0.05 / 12.0, 360).unwrap();
        let crossover =
            schedule.iter().position(|row| row.2 > row.1).expect("principal overtakes eventually");
        assert!(crossover > 180, "principal overtook interest at period {}", crossover + 1);
        assert!(crossover < 240, "it should not take that long: {}", crossover + 1);

        // Total interest on a thirty-year loan at 5% is most of the
        // principal again.
        let interest: f64 = schedule.iter().map(|row| row.1).sum();
        assert!(interest > 0.9 * 300_000.0, "the interest was only {interest}");
        assert!(interest < 300_000.0, "the interest was {interest}");

        // A higher rate pushes the crossover later still.
        let dearer = amortization_schedule(300_000.0, 0.09 / 12.0, 360).unwrap();
        let later = dearer.iter().position(|row| row.2 > row.1).unwrap();
        assert!(later > crossover, "a higher rate did not delay the crossover");
    }
}
