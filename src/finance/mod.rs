//! Quantitative finance: derivative pricing, interest rates, portfolio
//! construction and risk measurement.
//!
//! # What the models are and are not
//!
//! Every pricing model here is a statement about a *hypothetical* market:
//! continuous trading, no transaction costs, a known volatility, and a
//! price process of a stated form. None of those is true. What the models
//! buy is not a prediction of price but a consistent way to quote one
//! instrument in terms of another -- which is why the quantity traders
//! actually exchange is implied volatility, the number that makes the
//! formula reproduce the market price, rather than the price itself.
//!
//! The tests in this module lean hard on that internal consistency. Put-call
//! parity is a no-arbitrage identity independent of the model; a binomial
//! tree must converge to Black-Scholes as its steps grow; Monte Carlo must
//! agree with the closed form within its own standard error; and the
//! Greeks must match finite differences of the price they are derivatives
//! of. Those are checkable. Whether the model describes a real market is
//! not, and nothing here claims it.

pub mod options;
