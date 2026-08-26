//! A small computer algebra system over expression trees.
//!
//! Expressions are built from constants, exact rationals, named variables,
//! n-ary sums and products, powers, and the usual elementary functions.
//! The design is numeric-first: everything can be evaluated, differentiated
//! exactly, simplified enough to make cancellation visible, and compiled to
//! a stack machine for repeated evaluation.

use crate::error::GeomError;
use crate::exact::polynomial::Poly;
use crate::exact::rational::Rational;
use crate::monte_carlo::Rng;

/// A symbolic expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Const(f64),
    Rat(Rational),
    Var(String),
    Add(Vec<Expr>),
    Mul(Vec<Expr>),
    Pow(Box<Expr>, Box<Expr>),
    Neg(Box<Expr>),
    Sin(Box<Expr>),
    Cos(Box<Expr>),
    Tan(Box<Expr>),
    Exp(Box<Expr>),
    Ln(Box<Expr>),
    Sqrt(Box<Expr>),
    Abs(Box<Expr>),
    Atan(Box<Expr>),
    Sinh(Box<Expr>),
    Cosh(Box<Expr>),
}

/// Which side a one-sided limit approaches from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Left,
    Right,
    Both,
}

// ---------------------------------------------------------------------------
// constructors and small helpers
// ---------------------------------------------------------------------------

impl Expr {
    #[must_use]
    pub fn c(v: f64) -> Self {
        Expr::Const(v)
    }

    #[must_use]
    pub fn var(name: &str) -> Self {
        Expr::Var(name.to_string())
    }

    #[must_use]
    pub fn zero() -> Self {
        Expr::Const(0.0)
    }

    #[must_use]
    pub fn one() -> Self {
        Expr::Const(1.0)
    }

    #[must_use]
    pub fn add(terms: Vec<Expr>) -> Self {
        Expr::Add(terms)
    }

    #[must_use]
    pub fn mul(factors: Vec<Expr>) -> Self {
        Expr::Mul(factors)
    }

    #[must_use]
    pub fn pow(base: Expr, exp: Expr) -> Self {
        Expr::Pow(Box::new(base), Box::new(exp))
    }

    /// The numeric value of a constant leaf, if this is one.
    #[must_use]
    pub fn as_number(&self) -> Option<f64> {
        match self {
            Expr::Const(v) => Some(*v),
            Expr::Rat(q) => Some(q.to_f64()),
            _ => None,
        }
    }

    fn is_const(&self, v: f64) -> bool {
        self.as_number().is_some_and(|x| x == v)
    }

    /// The direct children of this node.
    fn children(&self) -> Vec<&Expr> {
        match self {
            Expr::Const(_) | Expr::Rat(_) | Expr::Var(_) => Vec::new(),
            Expr::Add(v) | Expr::Mul(v) => v.iter().collect(),
            Expr::Pow(a, b) => vec![a.as_ref(), b.as_ref()],
            Expr::Neg(a)
            | Expr::Sin(a)
            | Expr::Cos(a)
            | Expr::Tan(a)
            | Expr::Exp(a)
            | Expr::Ln(a)
            | Expr::Sqrt(a)
            | Expr::Abs(a)
            | Expr::Atan(a)
            | Expr::Sinh(a)
            | Expr::Cosh(a) => vec![a.as_ref()],
        }
    }

    /// Rebuild this node with new children, in the order `children` returns.
    fn rebuild(&self, kids: Vec<Expr>) -> Expr {
        match self {
            Expr::Const(_) | Expr::Rat(_) | Expr::Var(_) => self.clone(),
            Expr::Add(_) => Expr::Add(kids),
            Expr::Mul(_) => Expr::Mul(kids),
            Expr::Pow(_, _) => {
                Expr::Pow(Box::new(kids[0].clone()), Box::new(kids[1].clone()))
            }
            _ => {
                let a = Box::new(kids[0].clone());
                match self {
                    Expr::Neg(_) => Expr::Neg(a),
                    Expr::Sin(_) => Expr::Sin(a),
                    Expr::Cos(_) => Expr::Cos(a),
                    Expr::Tan(_) => Expr::Tan(a),
                    Expr::Exp(_) => Expr::Exp(a),
                    Expr::Ln(_) => Expr::Ln(a),
                    Expr::Sqrt(_) => Expr::Sqrt(a),
                    Expr::Abs(_) => Expr::Abs(a),
                    Expr::Atan(_) => Expr::Atan(a),
                    Expr::Sinh(_) => Expr::Sinh(a),
                    Expr::Cosh(_) => Expr::Cosh(a),
                    _ => unreachable!("handled above"),
                }
            }
        }
    }

    /// The number of nodes in the tree.
    #[must_use]
    pub fn node_count(&self) -> usize {
        1 + self.children().iter().map(|c| c.node_count()).sum::<usize>()
    }

    /// The height of the tree; a leaf has depth 1.
    #[must_use]
    pub fn depth(&self) -> usize {
        1 + self.children().iter().map(|c| c.depth()).max().unwrap_or(0)
    }

    /// Every variable name appearing in the expression, sorted and unique.
    ///
    /// Collected into a `BTreeSet` rather than deduplicated by scanning a
    /// growing vector, which would cost `O(v^2)` string comparisons in the
    /// number of distinct variables. The set also supplies the sort.
    #[must_use]
    pub fn variables(&self) -> Vec<String> {
        let mut out = std::collections::BTreeSet::new();
        fn walk(e: &Expr, out: &mut std::collections::BTreeSet<String>) {
            if let Expr::Var(n) = e {
                out.insert(n.clone());
            }
            for c in e.children() {
                walk(c, out);
            }
        }
        walk(self, &mut out);
        out.into_iter().collect()
    }

    /// Replace every occurrence of `var` with `replacement`.
    #[must_use]
    pub fn substitute(&self, var: &str, replacement: &Expr) -> Expr {
        if let Expr::Var(n) = self {
            if n == var {
                return replacement.clone();
            }
            return self.clone();
        }
        let kids: Vec<Expr> = self
            .children()
            .into_iter()
            .map(|c| c.substitute(var, replacement))
            .collect();
        self.rebuild(kids)
    }

    /// Evaluate at the given variable bindings.
    ///
    /// # Errors
    /// Returns [`GeomError::InvalidArgument`] if a variable in the
    /// expression has no binding.
    pub fn eval(&self, vars: &[(&str, f64)]) -> Result<f64, GeomError> {
        Ok(match self {
            Expr::Const(v) => *v,
            Expr::Rat(q) => q.to_f64(),
            Expr::Var(n) => vars
                .iter()
                .find(|(k, _)| k == n)
                .map(|(_, v)| *v)
                .ok_or(GeomError::InvalidArgument("unbound variable"))?,
            Expr::Add(t) => {
                let mut s = 0.0;
                for e in t {
                    s += e.eval(vars)?;
                }
                s
            }
            Expr::Mul(f) => {
                let mut p = 1.0;
                for e in f {
                    p *= e.eval(vars)?;
                }
                p
            }
            Expr::Pow(a, b) => a.eval(vars)?.powf(b.eval(vars)?),
            Expr::Neg(a) => -a.eval(vars)?,
            Expr::Sin(a) => a.eval(vars)?.sin(),
            Expr::Cos(a) => a.eval(vars)?.cos(),
            Expr::Tan(a) => a.eval(vars)?.tan(),
            Expr::Exp(a) => a.eval(vars)?.exp(),
            Expr::Ln(a) => a.eval(vars)?.ln(),
            Expr::Sqrt(a) => a.eval(vars)?.sqrt(),
            Expr::Abs(a) => a.eval(vars)?.abs(),
            Expr::Atan(a) => a.eval(vars)?.atan(),
            Expr::Sinh(a) => a.eval(vars)?.sinh(),
            Expr::Cosh(a) => a.eval(vars)?.cosh(),
        })
    }
}

// ---------------------------------------------------------------------------
// printing
// ---------------------------------------------------------------------------

/// Binding power used to decide where parentheses are needed.
fn prec(e: &Expr) -> u8 {
    match e {
        Expr::Add(_) => 1,
        Expr::Mul(_) => 2,
        Expr::Neg(_) => 3,
        Expr::Pow(_, _) => 4,
        _ => 5,
    }
}

fn wrap(child: &Expr, parent_prec: u8) -> String {
    let s = child.to_string();
    if prec(child) < parent_prec {
        format!("({s})")
    } else {
        s
    }
}

fn fmt_num(v: f64) -> String {
    if v == v.trunc() && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

impl std::fmt::Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expr::Const(v) => write!(f, "{}", fmt_num(*v)),
            Expr::Rat(q) => {
                if q.is_integer() {
                    write!(f, "{q}")
                } else {
                    write!(f, "({q})")
                }
            }
            Expr::Var(n) => write!(f, "{n}"),
            Expr::Add(t) => {
                if t.is_empty() {
                    return write!(f, "0");
                }
                let mut s = wrap(&t[0], 1);
                for e in &t[1..] {
                    // Render a negative leading coefficient as a subtraction.
                    match e {
                        Expr::Neg(inner) => s += &format!(" - {}", wrap(inner, 2)),
                        _ if e.as_number().is_some_and(|v| v < 0.0) => {
                            s += &format!(" - {}", fmt_num(-e.as_number().unwrap()));
                        }
                        _ => s += &format!(" + {}", wrap(e, 1)),
                    }
                }
                write!(f, "{s}")
            }
            Expr::Mul(v) => {
                if v.is_empty() {
                    return write!(f, "1");
                }
                let parts: Vec<String> = v.iter().map(|e| wrap(e, 2)).collect();
                write!(f, "{}", parts.join("*"))
            }
            Expr::Pow(a, b) => write!(f, "{}^{}", wrap(a, 5), wrap(b, 5)),
            Expr::Neg(a) => write!(f, "-{}", wrap(a, 3)),
            Expr::Sin(a) => write!(f, "sin({a})"),
            Expr::Cos(a) => write!(f, "cos({a})"),
            Expr::Tan(a) => write!(f, "tan({a})"),
            Expr::Exp(a) => write!(f, "exp({a})"),
            Expr::Ln(a) => write!(f, "ln({a})"),
            Expr::Sqrt(a) => write!(f, "sqrt({a})"),
            Expr::Abs(a) => write!(f, "abs({a})"),
            Expr::Atan(a) => write!(f, "atan({a})"),
            Expr::Sinh(a) => write!(f, "sinh({a})"),
            Expr::Cosh(a) => write!(f, "cosh({a})"),
        }
    }
}

impl Expr {
    /// Render as LaTeX.
    #[must_use]
    pub fn to_latex(&self) -> String {
        fn wrapl(child: &Expr, parent_prec: u8) -> String {
            let s = child.to_latex();
            if prec(child) < parent_prec {
                format!("\\left({s}\\right)")
            } else {
                s
            }
        }
        match self {
            Expr::Const(v) => fmt_num(*v),
            Expr::Rat(q) => {
                if q.is_integer() {
                    format!("{}", q.num)
                } else {
                    format!("\\frac{{{}}}{{{}}}", q.num, q.den)
                }
            }
            Expr::Var(n) => n.clone(),
            Expr::Add(t) => {
                if t.is_empty() {
                    return "0".to_string();
                }
                let mut s = wrapl(&t[0], 1);
                for e in &t[1..] {
                    match e {
                        Expr::Neg(inner) => s += &format!(" - {}", wrapl(inner, 2)),
                        _ => s += &format!(" + {}", wrapl(e, 1)),
                    }
                }
                s
            }
            Expr::Mul(v) => v.iter().map(|e| wrapl(e, 2)).collect::<Vec<_>>().join(" \\cdot "),
            Expr::Pow(a, b) => format!("{}^{{{}}}", wrapl(a, 5), b.to_latex()),
            Expr::Neg(a) => format!("-{}", wrapl(a, 3)),
            Expr::Sin(a) => format!("\\sin\\left({}\\right)", a.to_latex()),
            Expr::Cos(a) => format!("\\cos\\left({}\\right)", a.to_latex()),
            Expr::Tan(a) => format!("\\tan\\left({}\\right)", a.to_latex()),
            Expr::Exp(a) => format!("e^{{{}}}", a.to_latex()),
            Expr::Ln(a) => format!("\\ln\\left({}\\right)", a.to_latex()),
            Expr::Sqrt(a) => format!("\\sqrt{{{}}}", a.to_latex()),
            Expr::Abs(a) => format!("\\left|{}\\right|", a.to_latex()),
            Expr::Atan(a) => format!("\\arctan\\left({}\\right)", a.to_latex()),
            Expr::Sinh(a) => format!("\\sinh\\left({}\\right)", a.to_latex()),
            Expr::Cosh(a) => format!("\\cosh\\left({}\\right)", a.to_latex()),
        }
    }
}

// ---------------------------------------------------------------------------
// parsing
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(f64),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    LParen,
    RParen,
}

fn tokenize(s: &str) -> Result<Vec<Tok>, GeomError> {
    let b: Vec<char> = s.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let ch = b[i];
        match ch {
            c if c.is_whitespace() => i += 1,
            '+' => {
                out.push(Tok::Plus);
                i += 1;
            }
            '-' => {
                out.push(Tok::Minus);
                i += 1;
            }
            '*' => {
                out.push(Tok::Star);
                i += 1;
            }
            '/' => {
                out.push(Tok::Slash);
                i += 1;
            }
            '^' => {
                out.push(Tok::Caret);
                i += 1;
            }
            '(' => {
                out.push(Tok::LParen);
                i += 1;
            }
            ')' => {
                out.push(Tok::RParen);
                i += 1;
            }
            c if c.is_ascii_digit() || c == '.' => {
                let start = i;
                while i < b.len() && (b[i].is_ascii_digit() || b[i] == '.') {
                    i += 1;
                }
                // Accept an exponent suffix only when it really is one.
                if i < b.len() && (b[i] == 'e' || b[i] == 'E') {
                    let mut j = i + 1;
                    if j < b.len() && (b[j] == '+' || b[j] == '-') {
                        j += 1;
                    }
                    if j < b.len() && b[j].is_ascii_digit() {
                        i = j;
                        while i < b.len() && b[i].is_ascii_digit() {
                            i += 1;
                        }
                    }
                }
                let text: String = b[start..i].iter().collect();
                let v = text
                    .parse::<f64>()
                    .map_err(|_| GeomError::InvalidArgument("malformed number"))?;
                out.push(Tok::Num(v));
            }
            c if c.is_alphabetic() || c == '_' => {
                let start = i;
                while i < b.len() && (b[i].is_alphanumeric() || b[i] == '_') {
                    i += 1;
                }
                out.push(Tok::Ident(b[start..i].iter().collect()));
            }
            _ => return Err(GeomError::InvalidArgument("unexpected character")),
        }
    }
    Ok(out)
}

struct Parser<'a> {
    t: &'a [Tok],
    pos: usize,
}

impl Parser<'_> {
    fn peek(&self) -> Option<&Tok> {
        self.t.get(self.pos)
    }

    fn next(&mut self) -> Option<Tok> {
        let v = self.t.get(self.pos).cloned();
        self.pos += 1;
        v
    }

    fn expect(&mut self, tok: &Tok) -> Result<(), GeomError> {
        if self.peek() == Some(tok) {
            self.pos += 1;
            Ok(())
        } else {
            Err(GeomError::InvalidArgument("expected a closing parenthesis"))
        }
    }

    /// Precedence climbing: parse operators binding at least `min_prec`.
    fn expr(&mut self, min_prec: u8) -> Result<Expr, GeomError> {
        let mut lhs = self.unary()?;
        while let Some(op) = self.peek().cloned() {
            // `^` is right associative, the arithmetic operators are left.
            let (p, right_assoc) = match op {
                Tok::Plus | Tok::Minus => (1u8, false),
                Tok::Star | Tok::Slash => (2, false),
                Tok::Caret => (3, true),
                _ => break,
            };
            if p < min_prec {
                break;
            }
            self.pos += 1;
            let next_min = if right_assoc { p } else { p + 1 };
            let rhs = self.expr(next_min)?;
            lhs = match op {
                Tok::Plus => Expr::Add(vec![lhs, rhs]),
                Tok::Minus => Expr::Add(vec![lhs, Expr::Neg(Box::new(rhs))]),
                Tok::Star => Expr::Mul(vec![lhs, rhs]),
                Tok::Slash => Expr::Mul(vec![
                    lhs,
                    Expr::Pow(Box::new(rhs), Box::new(Expr::Const(-1.0))),
                ]),
                Tok::Caret => Expr::Pow(Box::new(lhs), Box::new(rhs)),
                _ => unreachable!("filtered above"),
            };
        }
        Ok(lhs)
    }

    fn unary(&mut self) -> Result<Expr, GeomError> {
        match self.peek() {
            Some(Tok::Minus) => {
                self.pos += 1;
                // Bind tighter than `*` so -x*y parses as (-x)*y, and
                // looser than `^` so -x^2 is -(x^2).
                Ok(Expr::Neg(Box::new(self.unary()?)))
            }
            Some(Tok::Plus) => {
                self.pos += 1;
                self.unary()
            }
            _ => self.postfix(),
        }
    }

    fn postfix(&mut self) -> Result<Expr, GeomError> {
        let base = self.primary()?;
        if self.peek() == Some(&Tok::Caret) {
            self.pos += 1;
            let rhs = self.unary()?;
            return Ok(Expr::Pow(Box::new(base), Box::new(rhs)));
        }
        Ok(base)
    }

    fn primary(&mut self) -> Result<Expr, GeomError> {
        match self.next() {
            Some(Tok::Num(v)) => Ok(Expr::Const(v)),
            Some(Tok::LParen) => {
                let e = self.expr(1)?;
                self.expect(&Tok::RParen)?;
                Ok(e)
            }
            Some(Tok::Ident(name)) => {
                if self.peek() == Some(&Tok::LParen) {
                    self.pos += 1;
                    let arg = self.expr(1)?;
                    self.expect(&Tok::RParen)?;
                    let b = Box::new(arg);
                    return Ok(match name.as_str() {
                        "sin" => Expr::Sin(b),
                        "cos" => Expr::Cos(b),
                        "tan" => Expr::Tan(b),
                        "exp" => Expr::Exp(b),
                        "ln" | "log" => Expr::Ln(b),
                        "sqrt" => Expr::Sqrt(b),
                        "abs" => Expr::Abs(b),
                        "atan" => Expr::Atan(b),
                        "sinh" => Expr::Sinh(b),
                        "cosh" => Expr::Cosh(b),
                        _ => return Err(GeomError::InvalidArgument("unknown function")),
                    });
                }
                Ok(match name.as_str() {
                    "pi" => Expr::Const(std::f64::consts::PI),
                    _ => Expr::Var(name),
                })
            }
            _ => Err(GeomError::InvalidArgument("unexpected end of input")),
        }
    }
}

impl Expr {
    /// Parse an infix expression such as `"3*x^2 + sin(y)/2"`.
    ///
    /// Supports `+ - * / ^`, parentheses, unary minus, the elementary
    /// functions named by the variants of this enum, and `pi`. `^` is
    /// right associative; `log` is accepted as a synonym for `ln`.
    ///
    /// # Errors
    /// Returns [`GeomError::InvalidArgument`] for an unexpected character,
    /// a malformed number, an unknown function, unbalanced parentheses, or
    /// trailing input.
    pub fn parse(s: &str) -> Result<Expr, GeomError> {
        let toks = tokenize(s)?;
        if toks.is_empty() {
            return Err(GeomError::Empty);
        }
        let mut p = Parser { t: &toks, pos: 0 };
        let e = p.expr(1)?;
        if p.pos != toks.len() {
            return Err(GeomError::InvalidArgument("trailing input"));
        }
        Ok(e)
    }
}

// ---------------------------------------------------------------------------
// differentiation
// ---------------------------------------------------------------------------

impl Expr {
    /// The exact symbolic derivative with respect to `var`.
    ///
    /// The result is not simplified; call [`Expr::simplify`] on it.
    #[must_use]
    pub fn diff(&self, var: &str) -> Expr {
        let d = |e: &Expr| e.diff(var);
        match self {
            Expr::Const(_) | Expr::Rat(_) => Expr::zero(),
            Expr::Var(n) => {
                if n == var {
                    Expr::one()
                } else {
                    Expr::zero()
                }
            }
            Expr::Add(t) => Expr::Add(t.iter().map(d).collect()),
            Expr::Mul(fs) => {
                // Product rule over an n-ary product.
                let mut terms = Vec::with_capacity(fs.len());
                for i in 0..fs.len() {
                    let mut factors: Vec<Expr> = fs.clone();
                    factors[i] = d(&fs[i]);
                    terms.push(Expr::Mul(factors));
                }
                Expr::Add(terms)
            }
            Expr::Pow(a, b) => {
                match b.as_number() {
                    // Power rule for a constant exponent.
                    Some(n) => Expr::Mul(vec![
                        Expr::Const(n),
                        Expr::Pow(a.clone(), Box::new(Expr::Const(n - 1.0))),
                        d(a),
                    ]),
                    // General case: d(a^b) = a^b * (b' ln a + b a'/a).
                    None => Expr::Mul(vec![
                        self.clone(),
                        Expr::Add(vec![
                            Expr::Mul(vec![d(b), Expr::Ln(a.clone())]),
                            Expr::Mul(vec![
                                b.as_ref().clone(),
                                d(a),
                                Expr::Pow(a.clone(), Box::new(Expr::Const(-1.0))),
                            ]),
                        ]),
                    ]),
                }
            }
            Expr::Neg(a) => Expr::Neg(Box::new(d(a))),
            Expr::Sin(a) => Expr::Mul(vec![Expr::Cos(a.clone()), d(a)]),
            Expr::Cos(a) => Expr::Neg(Box::new(Expr::Mul(vec![Expr::Sin(a.clone()), d(a)]))),
            // d tan = 1 + tan^2, which avoids introducing a division.
            Expr::Tan(a) => Expr::Mul(vec![
                Expr::Add(vec![
                    Expr::one(),
                    Expr::Pow(Box::new(Expr::Tan(a.clone())), Box::new(Expr::Const(2.0))),
                ]),
                d(a),
            ]),
            Expr::Exp(a) => Expr::Mul(vec![self.clone(), d(a)]),
            Expr::Ln(a) => Expr::Mul(vec![
                d(a),
                Expr::Pow(a.clone(), Box::new(Expr::Const(-1.0))),
            ]),
            Expr::Sqrt(a) => Expr::Mul(vec![
                Expr::Const(0.5),
                Expr::Pow(a.clone(), Box::new(Expr::Const(-0.5))),
                d(a),
            ]),
            // d|x| = sign(x) = x/|x|, valid away from zero.
            Expr::Abs(a) => Expr::Mul(vec![
                a.as_ref().clone(),
                Expr::Pow(Box::new(Expr::Abs(a.clone())), Box::new(Expr::Const(-1.0))),
                d(a),
            ]),
            Expr::Atan(a) => Expr::Mul(vec![
                Expr::Pow(
                    Box::new(Expr::Add(vec![
                        Expr::one(),
                        Expr::Pow(a.clone(), Box::new(Expr::Const(2.0))),
                    ])),
                    Box::new(Expr::Const(-1.0)),
                ),
                d(a),
            ]),
            Expr::Sinh(a) => Expr::Mul(vec![Expr::Cosh(a.clone()), d(a)]),
            Expr::Cosh(a) => Expr::Mul(vec![Expr::Sinh(a.clone()), d(a)]),
        }
    }

    /// The gradient with respect to several variables.
    #[must_use]
    pub fn gradient(&self, vars: &[&str]) -> Vec<Expr> {
        vars.iter().map(|v| self.diff(v).simplify()).collect()
    }
}

/// The Hessian matrix of second partial derivatives, simplified.
#[must_use]
pub fn hessian(e: &Expr, vars: &[&str]) -> Vec<Vec<Expr>> {
    vars.iter()
        .map(|a| {
            let da = e.diff(a);
            vars.iter().map(|b| da.diff(b).simplify()).collect()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// simplification
// ---------------------------------------------------------------------------

/// A sort key giving a deterministic operand order: numbers first, then
/// everything else by printed form.
fn sort_key(e: &Expr) -> (u8, String) {
    match e {
        Expr::Const(_) | Expr::Rat(_) => (0, String::new()),
        _ => (1, e.to_string()),
    }
}

/// Ordering for the terms of a sum: constants last, so a polynomial
/// prints as `x^2 - 1` rather than `-1 + x^2`. Products want the opposite
/// (`5*x`, not `x*5`), which is why [`sort_key`] exists separately.
fn add_sort_key(e: &Expr) -> (u8, String) {
    match e {
        Expr::Const(_) | Expr::Rat(_) => (1, String::new()),
        _ => (0, e.to_string()),
    }
}

/// Split a product into its numeric coefficient and remaining factors.
///
/// A negation contributes -1 to the coefficient rather than becoming an
/// opaque factor; without that, `x - x` never collects, because the two
/// terms hash under different keys.
fn split_coeff(e: &Expr) -> (f64, Vec<Expr>) {
    match e {
        Expr::Const(v) => (*v, Vec::new()),
        Expr::Rat(q) => (q.to_f64(), Vec::new()),
        Expr::Neg(a) => {
            let (c, rest) = split_coeff(a);
            (-c, rest)
        }
        Expr::Mul(fs) => {
            let mut coeff = 1.0;
            let mut rest = Vec::new();
            for f in fs {
                match f.as_number() {
                    Some(v) => coeff *= v,
                    None => rest.push(f.clone()),
                }
            }
            (coeff, rest)
        }
        _ => (1.0, vec![e.clone()]),
    }
}

/// Split a factor into a base and a numeric exponent.
fn split_pow(e: &Expr) -> (Expr, f64) {
    match e {
        Expr::Pow(b, x) => match x.as_number() {
            Some(v) => ((**b).clone(), v),
            None => (e.clone(), 1.0),
        },
        _ => (e.clone(), 1.0),
    }
}

/// Rebuild a product from a coefficient and factors, tidying the ends.
fn build_mul(coeff: f64, mut factors: Vec<Expr>) -> Expr {
    if coeff == 0.0 {
        return Expr::zero();
    }
    factors.sort_by_key(sort_key);
    if factors.is_empty() {
        return Expr::Const(coeff);
    }
    if coeff == 1.0 {
        return if factors.len() == 1 {
            factors.pop().expect("non-empty")
        } else {
            Expr::Mul(factors)
        };
    }
    // A leading -1 reads better as a negation than as a factor.
    if coeff == -1.0 {
        let inner = if factors.len() == 1 {
            factors.pop().expect("non-empty")
        } else {
            Expr::Mul(factors)
        };
        return Expr::Neg(Box::new(inner));
    }
    let mut all = vec![Expr::Const(coeff)];
    all.extend(factors);
    Expr::Mul(all)
}

impl Expr {
    /// Simplify: fold constants, flatten nested sums and products, collect
    /// like terms and repeated factors, and apply the standard identities
    /// for powers, exponentials and logarithms.
    ///
    /// This is deliberately a normaliser rather than a prover. It makes
    /// cancellation visible -- the derivative of `sin(x)^2 + cos(x)^2`
    /// collapses to zero because the two terms collect -- but it does not
    /// search for trigonometric rewrites.
    #[must_use]
    pub fn simplify(&self) -> Expr {
        // Simplify children first.
        let kids: Vec<Expr> = self.children().into_iter().map(Expr::simplify).collect();
        let e = self.rebuild(kids);
        match e {
            // A negation is normalised into a -1 coefficient so that terms
            // collect uniformly, then rebuilt by `build_mul`.
            Expr::Neg(a) => {
                let (c, f) = split_coeff(&a);
                build_mul(-c, f)
            }
            Expr::Add(terms) => {
                // Flatten nested sums.
                let mut flat = Vec::new();
                let mut stack = terms;
                stack.reverse();
                while let Some(t) = stack.pop() {
                    match t {
                        Expr::Add(inner) => {
                            for x in inner.into_iter().rev() {
                                stack.push(x);
                            }
                        }
                        other => flat.push(other),
                    }
                }
                // Collect like terms: group by the non-numeric part.
                let mut constant = 0.0;
                let mut groups: Vec<(String, Vec<Expr>, f64)> = Vec::new();
                for t in flat {
                    let (c, rest) = split_coeff(&t);
                    if rest.is_empty() {
                        constant += c;
                        continue;
                    }
                    let mut sorted = rest;
                    sorted.sort_by_key(sort_key);
                    let key = sorted
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join("*");
                    match groups.iter_mut().find(|(k, _, _)| *k == key) {
                        Some((_, _, acc)) => *acc += c,
                        None => groups.push((key, sorted, c)),
                    }
                }
                let mut out: Vec<Expr> = Vec::new();
                for (_, factors, c) in groups {
                    if c != 0.0 {
                        out.push(build_mul(c, factors));
                    }
                }
                if constant != 0.0 {
                    out.push(Expr::Const(constant));
                }
                match out.len() {
                    0 => Expr::zero(),
                    1 => out.pop().expect("non-empty"),
                    _ => {
                        out.sort_by_key(add_sort_key);
                        Expr::Add(out)
                    }
                }
            }
            Expr::Mul(factors) => {
                // Flatten nested products.
                let mut flat = Vec::new();
                let mut stack = factors;
                stack.reverse();
                while let Some(f) = stack.pop() {
                    match f {
                        Expr::Mul(inner) => {
                            for x in inner.into_iter().rev() {
                                stack.push(x);
                            }
                        }
                        Expr::Neg(a) => {
                            flat.push(Expr::Const(-1.0));
                            stack.push(*a);
                        }
                        other => flat.push(other),
                    }
                }
                let mut coeff = 1.0;
                // Group repeated bases, summing their exponents.
                let mut bases: Vec<(String, Expr, f64)> = Vec::new();
                for f in flat {
                    if let Some(v) = f.as_number() {
                        coeff *= v;
                        continue;
                    }
                    let (b, x) = split_pow(&f);
                    let key = b.to_string();
                    match bases.iter_mut().find(|(k, _, _)| *k == key) {
                        Some((_, _, acc)) => *acc += x,
                        None => bases.push((key, b, x)),
                    }
                }
                if coeff == 0.0 {
                    return Expr::zero();
                }
                let mut out = Vec::new();
                // exp(a)*exp(b) = exp(a+b): collect every exponential factor
                // into one argument sum, so reciprocal pairs cancel.
                let mut exp_args: Vec<Expr> = Vec::new();
                for (_, b, x) in bases {
                    if x == 0.0 {
                        continue;
                    }
                    if let Expr::Exp(inner) = &b {
                        exp_args.push(if x == 1.0 {
                            (**inner).clone()
                        } else {
                            Expr::Mul(vec![Expr::Const(x), (**inner).clone()])
                        });
                        continue;
                    }
                    if x == 1.0 {
                        out.push(b);
                    } else {
                        out.push(Expr::Pow(Box::new(b), Box::new(Expr::Const(x))));
                    }
                }
                if !exp_args.is_empty() {
                    // Simplify only the argument; wrapping it back in Exp
                    // here would re-enter this branch.
                    let arg = Expr::Add(exp_args).simplify();
                    if !arg.is_const(0.0) {
                        out.push(Expr::Exp(Box::new(arg)));
                    }
                }
                build_mul(coeff, out)
            }
            Expr::Pow(b, x) => {
                if x.is_const(0.0) {
                    return Expr::one();
                }
                if x.is_const(1.0) {
                    return *b;
                }
                if b.is_const(1.0) {
                    return Expr::one();
                }
                if b.is_const(0.0) {
                    return Expr::zero();
                }
                if let (Some(bv), Some(xv)) = (b.as_number(), x.as_number()) {
                    return Expr::Const(bv.powf(xv));
                }
                // (a^m)^n collapses when both exponents are numeric.
                if let Expr::Pow(inner_b, inner_x) = b.as_ref() {
                    if let (Some(m), Some(n)) = (inner_x.as_number(), x.as_number()) {
                        return Expr::Pow(inner_b.clone(), Box::new(Expr::Const(m * n)))
                            .simplify();
                    }
                }
                Expr::Pow(b, x)
            }
            Expr::Ln(a) => match a.as_ref() {
                Expr::Exp(inner) => (**inner).clone(),
                _ if a.is_const(1.0) => Expr::zero(),
                _ => Expr::Ln(a),
            },
            Expr::Exp(a) => match a.as_ref() {
                Expr::Ln(inner) => (**inner).clone(),
                _ if a.is_const(0.0) => Expr::one(),
                _ => Expr::Exp(a),
            },
            Expr::Sin(a) if a.is_const(0.0) => Expr::zero(),
            Expr::Cos(a) if a.is_const(0.0) => Expr::one(),
            Expr::Tan(a) if a.is_const(0.0) => Expr::zero(),
            Expr::Sinh(a) if a.is_const(0.0) => Expr::zero(),
            Expr::Cosh(a) if a.is_const(0.0) => Expr::one(),
            Expr::Atan(a) if a.is_const(0.0) => Expr::zero(),
            Expr::Sqrt(a) => match a.as_number() {
                Some(v) if v >= 0.0 => Expr::Const(v.sqrt()),
                _ => Expr::Sqrt(a),
            },
            Expr::Abs(a) => match a.as_number() {
                Some(v) => Expr::Const(v.abs()),
                None => Expr::Abs(a),
            },
            other => other,
        }
    }

    /// Distribute products over sums and expand small integer powers, then
    /// simplify.
    #[must_use]
    pub fn expand(&self) -> Expr {
        fn go(e: &Expr) -> Expr {
            let kids: Vec<Expr> = e.children().into_iter().map(go).collect();
            let e = e.rebuild(kids);
            match e {
                Expr::Mul(factors) => {
                    // Multiply out one factor at a time, keeping a list of
                    // summands as the running product.
                    let mut acc: Vec<Expr> = vec![Expr::one()];
                    for f in factors {
                        let terms: Vec<Expr> = match f {
                            Expr::Add(t) => t,
                            other => vec![other],
                        };
                        let mut next = Vec::with_capacity(acc.len() * terms.len());
                        for a in &acc {
                            for t in &terms {
                                next.push(Expr::Mul(vec![a.clone(), t.clone()]));
                            }
                        }
                        acc = next;
                    }
                    if acc.len() == 1 {
                        acc.pop().expect("non-empty")
                    } else {
                        Expr::Add(acc)
                    }
                }
                Expr::Pow(b, x) => {
                    // Expand (sum)^n for small non-negative integer n by
                    // repeated multiplication.
                    if let Some(n) = x.as_number() {
                        if n.fract() == 0.0 && (0.0..=16.0).contains(&n) {
                            let k = n as usize;
                            if k == 0 {
                                return Expr::one();
                            }
                            let mut acc = (*b).clone();
                            for _ in 1..k {
                                acc = go(&Expr::Mul(vec![acc, (*b).clone()]));
                            }
                            return acc;
                        }
                    }
                    Expr::Pow(b, x)
                }
                other => other,
            }
        }
        go(self).simplify()
    }
}

// ---------------------------------------------------------------------------
// polynomials, Taylor series, compilation
// ---------------------------------------------------------------------------

impl Expr {
    /// Extract the coefficients of a univariate polynomial in `var`, or
    /// `None` if the expanded expression is not one.
    #[must_use]
    pub fn as_polynomial(&self, var: &str) -> Option<Poly> {
        let e = self.expand();
        let terms: Vec<Expr> = match &e {
            Expr::Add(t) => t.clone(),
            other => vec![other.clone()],
        };
        let mut coeffs: Vec<f64> = Vec::new();
        for t in terms {
            let (c, rest) = split_coeff(&t);
            let mut power = 0usize;
            let mut coeff = c;
            for f in rest {
                let (b, x) = split_pow(&f);
                match &b {
                    Expr::Var(n) if n == var => {
                        // Only non-negative integer powers of `var` qualify.
                        if x.fract() != 0.0 || x < 0.0 {
                            return None;
                        }
                        power += x as usize;
                    }
                    // Any other factor must be free of `var` and constant.
                    other => {
                        if other.variables().iter().any(|v| v == var) {
                            return None;
                        }
                        let v = other.eval(&[]).ok()?;
                        coeff *= v.powf(x);
                    }
                }
            }
            if coeffs.len() <= power {
                coeffs.resize(power + 1, 0.0);
            }
            coeffs[power] += coeff;
        }
        if coeffs.is_empty() {
            coeffs.push(0.0);
        }
        Some(Poly::new(coeffs))
    }

    /// The Taylor polynomial of degree `order` about `at`, in `var`.
    ///
    /// Coefficients are the derivatives `f^(k)(at) / k!`, computed by
    /// differentiating symbolically and evaluating, so they are exact up to
    /// the evaluation itself.
    ///
    /// # Errors
    /// Returns `None` if any derivative fails to evaluate at `at`, which
    /// happens when the expression is undefined there or mentions another
    /// variable.
    #[must_use]
    pub fn taylor(&self, var: &str, at: f64, order: usize) -> Option<Poly> {
        let mut coeffs = Vec::with_capacity(order + 1);
        let mut d = self.clone();
        let mut factorial = 1.0f64;
        for k in 0..=order {
            if k > 0 {
                d = d.diff(var).simplify();
                factorial *= k as f64;
            }
            let v = d.eval(&[(var, at)]).ok()?;
            if !v.is_finite() {
                return None;
            }
            coeffs.push(v / factorial);
        }
        Some(Poly::new(coeffs))
    }

    /// Flatten to a stack program for fast repeated evaluation.
    ///
    /// The compiled program reads variables positionally, in the order
    /// given by [`Expr::variables`].
    #[must_use]
    pub fn compile(&self) -> CompiledExpr {
        let vars = self.variables();
        let mut ops = Vec::new();
        fn emit(e: &Expr, vars: &[String], ops: &mut Vec<Op>) {
            match e {
                Expr::Const(v) => ops.push(Op::Push(*v)),
                Expr::Rat(q) => ops.push(Op::Push(q.to_f64())),
                Expr::Var(n) => {
                    let idx = vars.iter().position(|v| v == n).expect("variable listed");
                    ops.push(Op::Load(idx));
                }
                Expr::Add(t) => {
                    for x in t {
                        emit(x, vars, ops);
                    }
                    ops.push(Op::Sum(t.len()));
                }
                Expr::Mul(t) => {
                    for x in t {
                        emit(x, vars, ops);
                    }
                    ops.push(Op::Prod(t.len()));
                }
                Expr::Pow(a, b) => {
                    emit(a, vars, ops);
                    emit(b, vars, ops);
                    ops.push(Op::Pow);
                }
                other => {
                    let kid = other.children()[0];
                    emit(kid, vars, ops);
                    ops.push(match other {
                        Expr::Neg(_) => Op::Neg,
                        Expr::Sin(_) => Op::Sin,
                        Expr::Cos(_) => Op::Cos,
                        Expr::Tan(_) => Op::Tan,
                        Expr::Exp(_) => Op::Exp,
                        Expr::Ln(_) => Op::Ln,
                        Expr::Sqrt(_) => Op::Sqrt,
                        Expr::Abs(_) => Op::Abs,
                        Expr::Atan(_) => Op::Atan,
                        Expr::Sinh(_) => Op::Sinh,
                        Expr::Cosh(_) => Op::Cosh,
                        _ => unreachable!("leaf and n-ary cases handled above"),
                    });
                }
            }
        }
        emit(self, &vars, &mut ops);
        CompiledExpr { ops, vars }
    }

    /// Antiderivative with respect to `var` by linearity, the power rule,
    /// a small table of elementary forms, and the linear substitution
    /// `u = a*var + b`.
    ///
    /// Returns `None` when none of those rules apply; it does not attempt
    /// integration by parts or partial fractions.
    #[must_use]
    pub fn integrate_simple(&self, var: &str) -> Option<Expr> {
        let e = self.simplify();
        // Free of the variable: integrates to c*var.
        if !e.variables().iter().any(|v| v == var) {
            return Some(Expr::Mul(vec![e, Expr::var(var)]).simplify());
        }
        match &e {
            // Linearity over sums.
            Expr::Add(terms) => {
                let mut out = Vec::with_capacity(terms.len());
                for t in terms {
                    out.push(t.integrate_simple(var)?);
                }
                Some(Expr::Add(out).simplify())
            }
            // Pull constant factors out of a product.
            Expr::Mul(_) => {
                let (c, rest) = split_coeff(&e);
                if rest.len() == 1 && c != 1.0 {
                    let inner = rest[0].integrate_simple(var)?;
                    return Some(Expr::Mul(vec![Expr::Const(c), inner]).simplify());
                }
                if rest.len() == 1 {
                    return rest[0].integrate_simple(var);
                }
                None
            }
            Expr::Neg(a) => Some(Expr::Neg(Box::new(a.integrate_simple(var)?)).simplify()),
            Expr::Var(n) if n == var => Some(
                Expr::Mul(vec![
                    Expr::Const(0.5),
                    Expr::pow(Expr::var(var), Expr::Const(2.0)),
                ])
                .simplify(),
            ),
            Expr::Pow(b, x) => {
                // The power rule, including the logarithmic exception.
                let n = x.as_number()?;
                match b.as_ref() {
                    Expr::Var(v) if v == var => {
                        if n == -1.0 {
                            Some(Expr::Ln(Box::new(Expr::Abs(Box::new(Expr::var(var))))))
                        } else {
                            Some(
                                Expr::Mul(vec![
                                    Expr::Const(1.0 / (n + 1.0)),
                                    Expr::pow(Expr::var(var), Expr::Const(n + 1.0)),
                                ])
                                .simplify(),
                            )
                        }
                    }
                    _ => None,
                }
            }
            // Elementary forms, each allowed an inner linear argument.
            Expr::Sin(a) | Expr::Cos(a) | Expr::Exp(a) => {
                let (slope, _) = linear_in(a, var)?;
                let scale = Expr::Const(1.0 / slope);
                let anti = match &e {
                    Expr::Sin(_) => Expr::Neg(Box::new(Expr::Cos(a.clone()))),
                    Expr::Cos(_) => Expr::Sin(a.clone()),
                    _ => Expr::Exp(a.clone()),
                };
                Some(Expr::Mul(vec![scale, anti]).simplify())
            }
            _ => None,
        }
    }

    /// A one-sided or two-sided numeric limit, by sampling a geometric
    /// sequence of offsets and requiring the values to settle.
    ///
    /// Returns `None` when the samples do not agree, which covers a genuine
    /// divergence and a two-sided limit whose sides disagree.
    #[must_use]
    pub fn limit_numeric(&self, var: &str, at: f64, side: Side) -> Option<f64> {
        let approach = |sign: f64| -> Option<f64> {
            let mut last: Option<f64> = None;
            let mut stable = 0;
            let mut h = 1e-3;
            for _ in 0..12 {
                let v = self.eval(&[(var, at + sign * h)]).ok()?;
                if !v.is_finite() {
                    return None;
                }
                if let Some(p) = last {
                    // Settled once successive samples agree to a relative
                    // tolerance well inside f64's reach.
                    if (v - p).abs() <= 1e-7 * v.abs().max(1.0) {
                        stable += 1;
                        if stable >= 2 {
                            return Some(v);
                        }
                    } else {
                        stable = 0;
                    }
                }
                last = Some(v);
                h *= 0.25;
            }
            last
        };
        match side {
            Side::Left => approach(-1.0),
            Side::Right => approach(1.0),
            Side::Both => {
                let l = approach(-1.0)?;
                let r = approach(1.0)?;
                if (l - r).abs() <= 1e-6 * l.abs().max(1.0) {
                    Some(0.5 * (l + r))
                } else {
                    None
                }
            }
        }
    }

    /// Test whether two expressions agree numerically at random points.
    ///
    /// This is a probabilistic check, not a proof: it samples the shared
    /// variables and compares. Points where either side is undefined are
    /// skipped rather than counted as disagreement.
    #[must_use]
    pub fn equivalent_numeric(&self, other: &Expr, trials: usize, rng: &mut Rng) -> bool {
        let mut vars = self.variables();
        for v in other.variables() {
            if !vars.contains(&v) {
                vars.push(v);
            }
        }
        let mut checked = 0usize;
        for _ in 0..trials {
            let vals: Vec<(String, f64)> = vars
                .iter()
                .map(|v| (v.clone(), rng.next_f64() * 4.0 - 2.0))
                .collect();
            let binding: Vec<(&str, f64)> =
                vals.iter().map(|(k, v)| (k.as_str(), *v)).collect();
            let (Ok(a), Ok(b)) = (self.eval(&binding), other.eval(&binding)) else {
                continue;
            };
            if !a.is_finite() || !b.is_finite() {
                continue;
            }
            if (a - b).abs() > 1e-9 * a.abs().max(b.abs()).max(1.0) {
                return false;
            }
            checked += 1;
        }
        checked > 0
    }
}

/// Decompose `e` as `slope * var + intercept`, or `None` if it is not
/// linear in `var` with constant coefficients.
fn linear_in(e: &Expr, var: &str) -> Option<(f64, f64)> {
    let p = e.as_polynomial(var)?;
    match p.c.len() {
        0 => Some((0.0, 0.0)),
        1 => Some((0.0, p.c[0])),
        2 => {
            if p.c[1] == 0.0 {
                Some((0.0, p.c[0]))
            } else {
                Some((p.c[1], p.c[0]))
            }
        }
        _ => None,
    }
}

/// A single instruction of a [`CompiledExpr`].
#[derive(Debug, Clone, Copy, PartialEq)]
enum Op {
    Push(f64),
    Load(usize),
    Sum(usize),
    Prod(usize),
    Pow,
    Neg,
    Sin,
    Cos,
    Tan,
    Exp,
    Ln,
    Sqrt,
    Abs,
    Atan,
    Sinh,
    Cosh,
}

/// An expression flattened to a stack program.
#[derive(Debug, Clone)]
pub struct CompiledExpr {
    ops: Vec<Op>,
    vars: Vec<String>,
}

impl CompiledExpr {
    /// The variable order the program expects.
    #[must_use]
    pub fn vars(&self) -> &[String] {
        &self.vars
    }

    /// The number of instructions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Evaluate with variable values in [`CompiledExpr::vars`] order.
    ///
    /// # Panics
    /// Panics if `vals` is shorter than the variable list.
    #[must_use]
    pub fn eval(&self, vals: &[f64]) -> f64 {
        assert!(vals.len() >= self.vars.len(), "too few variable values");
        let mut st: Vec<f64> = Vec::with_capacity(16);
        for op in &self.ops {
            match *op {
                Op::Push(v) => st.push(v),
                Op::Load(i) => st.push(vals[i]),
                Op::Sum(n) => {
                    let at = st.len() - n;
                    let s = st.drain(at..).sum();
                    st.push(s);
                }
                Op::Prod(n) => {
                    let at = st.len() - n;
                    let p = st.drain(at..).product();
                    st.push(p);
                }
                Op::Pow => {
                    let b = st.pop().expect("stack underflow");
                    let a = st.pop().expect("stack underflow");
                    st.push(a.powf(b));
                }
                _ => {
                    let a = st.pop().expect("stack underflow");
                    st.push(match op {
                        Op::Neg => -a,
                        Op::Sin => a.sin(),
                        Op::Cos => a.cos(),
                        Op::Tan => a.tan(),
                        Op::Exp => a.exp(),
                        Op::Ln => a.ln(),
                        Op::Sqrt => a.sqrt(),
                        Op::Abs => a.abs(),
                        Op::Atan => a.atan(),
                        Op::Sinh => a.sinh(),
                        Op::Cosh => a.cosh(),
                        _ => unreachable!("handled above"),
                    });
                }
            }
        }
        st.pop().unwrap_or(0.0)
    }
}

/// Real roots of `e` in a bracket, by scanning for sign changes and
/// bisecting each one.
///
/// Only sign-changing roots are found; a root of even multiplicity, where
/// the curve touches the axis without crossing, is invisible to this
/// method.
///
/// # Panics
/// Panics if the bracket is empty or reversed.
#[must_use]
pub fn solve_univariate_numeric(e: &Expr, var: &str, bracket: (f64, f64)) -> Vec<f64> {
    let (lo, hi) = bracket;
    assert!(hi > lo, "bracket must be non-empty");
    let n = 2000;
    let f = |x: f64| e.eval(&[(var, x)]).ok().filter(|v| v.is_finite());
    let mut out = Vec::new();
    let step = (hi - lo) / n as f64;
    let mut prev_x = lo;
    let mut prev_v = f(lo);
    for k in 1..=n {
        let x = lo + step * k as f64;
        let v = f(x);
        if let (Some(a), Some(b)) = (prev_v, v) {
            if a == 0.0 {
                out.push(prev_x);
            } else if a * b < 0.0 {
                // Bisect: 80 halvings takes the bracket well below f64
                // resolution for any realistic interval.
                let (mut l, mut r) = (prev_x, x);
                let mut fl = a;
                for _ in 0..80 {
                    let m = 0.5 * (l + r);
                    let Some(fm) = f(m) else { break };
                    if fl * fm <= 0.0 {
                        r = m;
                    } else {
                        l = m;
                        fl = fm;
                    }
                }
                out.push(0.5 * (l + r));
            }
        }
        prev_x = x;
        prev_v = v;
    }
    if let Some(v) = f(hi) {
        if v == 0.0 {
            out.push(hi);
        }
    }
    out
}

/// Critical points of `e` in a range: the points where the derivative
/// changes sign, paired with the value of `e` there.
///
/// `n` is unused beyond selecting the search resolution and is kept for
/// signature compatibility.
///
/// # Panics
/// Panics if the range is empty or reversed.
#[must_use]
pub fn critical_points(e: &Expr, var: &str, range: (f64, f64), n: usize) -> Vec<(f64, f64)> {
    let _ = n;
    let d = e.diff(var).simplify();
    solve_univariate_numeric(&d, var, range)
        .into_iter()
        .filter_map(|x| e.eval(&[(var, x)]).ok().map(|y| (x, y)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::dual::{derivative, Dual};

    fn p(s: &str) -> Expr {
        Expr::parse(s).expect("parses")
    }

    #[test]
    fn test_parse_precedence_and_round_trip() {
        // Precedence and associativity, checked by value rather than shape.
        assert_eq!(p("2+3*4").eval(&[]).unwrap(), 14.0);
        assert_eq!(p("(2+3)*4").eval(&[]).unwrap(), 20.0);
        assert_eq!(p("2^3^2").eval(&[]).unwrap(), 512.0, "^ is right associative");
        assert_eq!(p("(2^3)^2").eval(&[]).unwrap(), 64.0);
        assert_eq!(p("8/4/2").eval(&[]).unwrap(), 1.0, "/ is left associative");
        assert_eq!(p("-2^2").eval(&[]).unwrap(), -4.0, "unary minus binds looser than ^");
        assert_eq!(p("2-3-4").eval(&[]).unwrap(), -5.0);
        assert_eq!(p("1e3").eval(&[]).unwrap(), 1000.0);
        assert_eq!(p("2*pi").eval(&[]).unwrap(), std::f64::consts::TAU);
        assert!((p("sin(pi/2)").eval(&[]).unwrap() - 1.0).abs() < 1e-15);

        // Rejected inputs.
        assert!(Expr::parse("").is_err());
        assert!(Expr::parse("2+").is_err());
        assert!(Expr::parse("(2+3").is_err());
        assert!(Expr::parse("2+3)").is_err());
        assert!(Expr::parse("frobnicate(x)").is_err());
        assert!(Expr::parse("2 $ 3").is_err());

        // The roadmap's property: parse(to_string(e)) evaluates identically.
        let sources = [
            "3*x^2 + sin(y)/2",
            "x^3*sin(x)",
            "exp(-x^2)",
            "ln(abs(x)+2) - sqrt(x^2+1)",
            "atan(x)*cosh(y) + sinh(x*y)",
            "-x*y + 2^x",
            "(x+y)^3",
            "tan(x/4) + 1/(x^2+1)",
        ];
        let mut rng = Rng::new(7);
        for src in sources {
            let e = p(src);
            let round = Expr::parse(&e.to_string())
                .unwrap_or_else(|_| panic!("re-parse of {} failed", e));
            for _ in 0..40 {
                let (xv, yv) = (rng.next_f64() * 2.0 + 0.3, rng.next_f64() * 2.0 + 0.3);
                let b = [("x", xv), ("y", yv)];
                let (a, c) = (e.eval(&b).unwrap(), round.eval(&b).unwrap());
                assert!((a - c).abs() <= 1e-12 * a.abs().max(1.0),
                        "round-trip of {src} changed value: {a} vs {c}");
            }
            // And the simplified form must agree with the original too.
            assert!(e.simplify().equivalent_numeric(&e, 40, &mut rng),
                    "simplify changed the value of {src}");
        }
    }

    #[test]
    fn test_diff_matches_dual_numbers() {
        // The roadmap's property: the symbolic derivative of x^3*sin(x)
        // agrees with the Part 1 dual-number derivative at 100 points.
        let e = p("x^3*sin(x)");
        let d = e.diff("x").simplify();
        let mut rng = Rng::new(11);
        for _ in 0..100 {
            let x = rng.next_f64() * 6.0 - 3.0;
            let symbolic = d.eval(&[("x", x)]).unwrap();
            let dual = derivative(|t: Dual| t.powi(3) * t.sin(), x);
            assert!((symbolic - dual).abs() <= 1e-9 * dual.abs().max(1.0),
                    "at x={x}: symbolic {symbolic} vs dual {dual}");
        }
        // The same check across the whole function table.
        let cases: [(&str, fn(Dual) -> Dual); 9] = [
            ("sin(x)*cos(x)", |t| t.sin() * t.cos()),
            ("exp(x)/(1+x^2)", |t| t.exp() / (Dual::constant(1.0) + t * t)),
            ("ln(x^2+2)", |t| (t * t + Dual::constant(2.0)).ln()),
            ("sqrt(x^2+1)", |t| (t * t + Dual::constant(1.0)).sqrt()),
            ("tan(x/3)", |t| (t / Dual::constant(3.0)).tan()),
            ("atan(2*x)", |t| (t * Dual::constant(2.0)).atan()),
            ("sinh(x)*cosh(x)", |t| t.sinh() * t.cosh()),
            ("x^5 - 3*x^2 + 7", |t| t.powi(5) - t.powi(2) * Dual::constant(3.0) + Dual::constant(7.0)),
            ("exp(sin(x))", |t| t.sin().exp()),
        ];
        for (src, f) in cases {
            let d = p(src).diff("x").simplify();
            for _ in 0..40 {
                let x = rng.next_f64() * 2.0 - 1.0;
                let symbolic = d.eval(&[("x", x)]).unwrap();
                let dual = derivative(f, x);
                assert!((symbolic - dual).abs() <= 1e-8 * dual.abs().max(1.0),
                        "{src} at x={x}: {symbolic} vs {dual}");
            }
        }
    }

    #[test]
    fn test_simplify_cancels_and_normalises() {
        // The roadmap's property: the derivative of the Pythagorean
        // identity collapses to exactly zero. The two product-rule terms
        // are +2*cos*sin and -2*cos*sin, so this is real term collection,
        // not a hard-coded trig rewrite.
        let e = p("sin(x)^2 + cos(x)^2");
        assert_eq!(e.diff("x").simplify(), Expr::zero());

        // Identities.
        assert_eq!(p("x*0").simplify(), Expr::zero());
        assert_eq!(p("x*1").simplify(), Expr::var("x"));
        assert_eq!(p("x+0").simplify(), Expr::var("x"));
        assert_eq!(p("x^1").simplify(), Expr::var("x"));
        assert_eq!(p("x^0").simplify(), Expr::one());
        assert_eq!(p("ln(exp(x))").simplify(), Expr::var("x"));
        assert_eq!(p("exp(ln(x))").simplify(), Expr::var("x"));
        assert_eq!(p("x-x").simplify(), Expr::zero());
        assert_eq!(p("2*x+3*x").simplify().to_string(), "5*x");
        assert_eq!(p("x*x").simplify().to_string(), "x^2");
        assert_eq!(p("x/x").simplify(), Expr::one());
        assert_eq!(p("2+3").simplify(), Expr::Const(5.0));
        assert_eq!(p("(x^2)^3").simplify().to_string(), "x^6");
        assert_eq!(p("sin(0)").simplify(), Expr::zero());
        assert_eq!(p("cos(0)").simplify(), Expr::one());

        // Simplification must never change the value.
        let mut rng = Rng::new(19);
        for src in ["x^3*sin(x)", "(x+1)^2 - x^2 - 2*x - 1", "exp(x)*exp(-x)",
                    "ln(x^2+1)*atan(x)", "x*y - y*x + 3", "sqrt(x^2+4)/(x^2+4)"] {
            let e = p(src);
            let s = e.simplify();
            assert!(s.equivalent_numeric(&e, 60, &mut rng), "simplify broke {src}");
            // Idempotence: simplifying again is a fixed point.
            assert_eq!(s.simplify(), s, "simplify is not idempotent on {src}");
        }
        assert_eq!(p("(x+1)^2 - x^2 - 2*x - 1").expand(), Expr::zero());
        assert_eq!(p("exp(x)*exp(-x)").simplify(), Expr::one());

        // Expansion is value-preserving and reaches polynomial form.
        for src in ["(x+1)^3", "(x+y)^2", "(x-1)*(x+1)", "(x+2)^4"] {
            let e = p(src);
            assert!(e.expand().equivalent_numeric(&e, 60, &mut rng), "expand broke {src}");
        }
        assert_eq!(p("(x-1)*(x+1)").expand().to_string(), "x^2 - 1");
    }

    #[test]
    fn test_polynomial_extraction_and_taylor() {
        let q = p("3*x^2 + 2*x - 5").as_polynomial("x").unwrap();
        assert_eq!(q.c, vec![-5.0, 2.0, 3.0]);
        let q = p("(x+1)^3").as_polynomial("x").unwrap();
        assert_eq!(q.c, vec![1.0, 3.0, 3.0, 1.0], "binomial coefficients");
        assert_eq!(p("7").as_polynomial("x").unwrap().c, vec![7.0]);
        // Not polynomials in x.
        assert!(p("sin(x)").as_polynomial("x").is_none());
        assert!(p("x^(-1)").as_polynomial("x").is_none());
        assert!(p("x^0.5").as_polynomial("x").is_none());
        // A coefficient carrying a different variable is not constant.
        assert!(p("y*x^2").as_polynomial("x").is_none());

        // Taylor coefficients against the known series.
        let t = p("exp(x)").taylor("x", 0.0, 6).unwrap();
        for (k, c) in t.c.iter().enumerate() {
            let want = 1.0 / (1..=k).map(|i| i as f64).product::<f64>().max(1.0);
            assert!((c - want).abs() < 1e-12, "exp coefficient {k}: {c} vs {want}");
        }
        let s = p("sin(x)").taylor("x", 0.0, 7).unwrap();
        assert!(s.c[0].abs() < 1e-15 && (s.c[1] - 1.0).abs() < 1e-12);
        assert!((s.c[3] + 1.0 / 6.0).abs() < 1e-12, "-1/6 x^3");
        assert!((s.c[5] - 1.0 / 120.0).abs() < 1e-12, "+1/120 x^5");
        assert!(s.c[2].abs() < 1e-12 && s.c[4].abs() < 1e-12, "even terms vanish");
        // The series approximates the function near the expansion point.
        let t = p("cos(x)").taylor("x", 0.5, 8).unwrap();
        for dx in [-0.2_f64, -0.05, 0.0, 0.05, 0.2] {
            let approx = t.eval(dx);
            assert!((approx - (0.5 + dx).cos()).abs() < 1e-9, "Taylor at dx={dx}");
        }
    }

    #[test]
    fn test_compile_matches_eval() {
        let mut rng = Rng::new(23);
        for src in ["3*x^2 + sin(y)/2", "exp(-x^2)*cosh(y)", "ln(x^2+2) + atan(y)",
                    "sqrt(x^2+y^2)", "x^3*sin(x)*tan(y/4)", "abs(x-y) + sinh(x)"] {
            let e = p(src);
            let c = e.compile();
            assert!(!c.is_empty() && c.len() >= e.node_count() / 2);
            let names: Vec<&str> = c.vars().iter().map(String::as_str).collect();
            for _ in 0..60 {
                let vals: Vec<f64> = names.iter().map(|_| rng.next_f64() * 2.0 + 0.2).collect();
                let binding: Vec<(&str, f64)> =
                    names.iter().copied().zip(vals.iter().copied()).collect();
                let want = e.eval(&binding).unwrap();
                let got = c.eval(&vals);
                assert!((got - want).abs() <= 1e-12 * want.abs().max(1.0),
                        "{src}: compiled {got} vs interpreted {want}");
            }
        }
        // Structure: variables are reported sorted, and a constant compiles.
        assert_eq!(p("y+x").compile().vars(), ["x", "y"]);
        assert_eq!(p("2+3").compile().eval(&[]), 5.0);
    }

    #[test]
    fn test_integration_limits_and_solving() {
        // Antiderivatives verified by differentiating back.
        let mut rng = Rng::new(29);
        for src in ["x^2", "x^3 + 2*x", "sin(x)", "cos(2*x)", "exp(3*x)", "5",
                    "x^(-1)", "sin(2*x+1)", "4*x^7"] {
            let f = p(src);
            let anti = f.integrate_simple("x")
                .unwrap_or_else(|| panic!("no antiderivative for {src}"));
            let back = anti.diff("x").simplify();
            assert!(back.equivalent_numeric(&f, 60, &mut rng),
                    "d/dx of the antiderivative of {src} gave {back}");
        }
        // Rules that do not apply return None rather than a wrong answer.
        assert!(p("sin(x^2)").integrate_simple("x").is_none(), "no substitution rule");
        assert!(p("x*sin(x)").integrate_simple("x").is_none(), "no parts rule");

        // Limits, including the removable singularity of sin(x)/x.
        let l = p("sin(x)/x").limit_numeric("x", 0.0, Side::Both).unwrap();
        assert!((l - 1.0).abs() < 1e-6, "sin(x)/x -> 1, got {l}");
        let l = p("(exp(x)-1)/x").limit_numeric("x", 0.0, Side::Both).unwrap();
        assert!((l - 1.0).abs() < 1e-5, "(e^x-1)/x -> 1, got {l}");
        let l = p("(1-cos(x))/x^2").limit_numeric("x", 0.0, Side::Both).unwrap();
        assert!((l - 0.5).abs() < 1e-4, "(1-cos x)/x^2 -> 1/2, got {l}");
        // A jump has one-sided limits but no two-sided one.
        let jump = p("abs(x)/x");
        assert!((jump.limit_numeric("x", 0.0, Side::Right).unwrap() - 1.0).abs() < 1e-12);
        assert!((jump.limit_numeric("x", 0.0, Side::Left).unwrap() + 1.0).abs() < 1e-12);
        assert!(jump.limit_numeric("x", 0.0, Side::Both).is_none(), "sides disagree");

        // Root finding and critical points.
        let roots = solve_univariate_numeric(&p("x^2 - 4"), "x", (-10.0, 10.0));
        assert_eq!(roots.len(), 2);
        assert!((roots[0] + 2.0).abs() < 1e-9 && (roots[1] - 2.0).abs() < 1e-9);
        let roots = solve_univariate_numeric(&p("sin(x)"), "x", (-0.5, 7.0));
        assert_eq!(roots.len(), 3, "0, pi, 2pi");
        assert!((roots[1] - std::f64::consts::PI).abs() < 1e-9);
        // A parabola's only critical point is its vertex.
        let cp = critical_points(&p("x^2 - 4*x + 7"), "x", (-10.0, 10.0), 100);
        assert_eq!(cp.len(), 1);
        assert!((cp[0].0 - 2.0).abs() < 1e-9 && (cp[0].1 - 3.0).abs() < 1e-9);
        // sin has extrema at pi/2 and 3pi/2.
        let cp = critical_points(&p("sin(x)"), "x", (0.0, 6.0), 100);
        assert_eq!(cp.len(), 2);
        assert!((cp[0].1 - 1.0).abs() < 1e-9 && (cp[1].1 + 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_gradient_hessian_and_structure() {
        // The Hessian of a quadratic form is its constant coefficient
        // matrix, and is symmetric.
        let e = p("3*x^2 + 2*x*y + 5*y^2");
        let g = e.gradient(&["x", "y"]);
        let mut rng = Rng::new(31);
        for _ in 0..40 {
            let (xv, yv) = (rng.next_f64() * 2.0 - 1.0, rng.next_f64() * 2.0 - 1.0);
            let b = [("x", xv), ("y", yv)];
            assert!((g[0].eval(&b).unwrap() - (6.0 * xv + 2.0 * yv)).abs() < 1e-12);
            assert!((g[1].eval(&b).unwrap() - (2.0 * xv + 10.0 * yv)).abs() < 1e-12);
        }
        let h = hessian(&e, &["x", "y"]);
        assert_eq!(h[0][0].eval(&[]).unwrap(), 6.0);
        assert_eq!(h[0][1].eval(&[]).unwrap(), 2.0);
        assert_eq!(h[1][0].eval(&[]).unwrap(), 2.0);
        assert_eq!(h[1][1].eval(&[]).unwrap(), 10.0);
        // Symmetry of second derivatives on a transcendental case.
        let f = p("exp(x*y) + sin(x)*y^3");
        let h = hessian(&f, &["x", "y"]);
        for _ in 0..40 {
            let (xv, yv) = (rng.next_f64() - 0.5, rng.next_f64() - 0.5);
            let b = [("x", xv), ("y", yv)];
            let (a, c) = (h[0][1].eval(&b).unwrap(), h[1][0].eval(&b).unwrap());
            assert!((a - c).abs() < 1e-10, "mixed partials differ: {a} vs {c}");
        }

        // Structural accessors and substitution.
        let e = p("3*x^2 + sin(y)/2");
        assert_eq!(e.variables(), ["x", "y"]);
        assert!(e.node_count() > 5 && e.depth() >= 3);
        assert_eq!(p("x").depth(), 1);
        let sub = e.substitute("y", &p("2*x"));
        assert_eq!(sub.variables(), ["x"]);
        for _ in 0..30 {
            let xv = rng.next_f64() * 2.0;
            let want = 3.0 * xv * xv + (2.0 * xv).sin() / 2.0;
            assert!((sub.eval(&[("x", xv)]).unwrap() - want).abs() < 1e-12);
        }
        // Unbound variables are an error, not a silent zero.
        assert!(p("x+z").eval(&[("x", 1.0)]).is_err());

        // LaTeX rendering.
        assert_eq!(p("x^2").to_latex(), "x^{2}");
        assert_eq!(p("sqrt(x)").to_latex(), "\\sqrt{x}");
        assert_eq!(p("sin(x)").to_latex(), "\\sin\\left(x\\right)");
        assert!(p("x/y").to_latex().contains("^{-1}") || p("x/y").to_latex().contains("cdot"));

        // equivalent_numeric distinguishes genuinely different functions.
        assert!(!p("sin(x)").equivalent_numeric(&p("cos(x)"), 50, &mut rng));
        assert!(p("sin(x)^2").equivalent_numeric(&p("1-cos(x)^2"), 50, &mut rng));
    }

    /// `variables()` must stay sorted and unique after the switch from a
    /// linear-scan dedup to a set, and must scale rather than degrade
    /// quadratically in the number of distinct variables.
    #[test]
    fn variables_are_sorted_unique_and_complete() {
        // Order of first appearance must not leak into the result.
        let e = Expr::parse("z + a*y + z*a + b").unwrap();
        assert_eq!(e.variables(), vec!["a", "b", "y", "z"]);
        // A repeated variable appears once however deeply nested.
        let deep = Expr::parse("sin(x + cos(x * exp(x)))").unwrap();
        assert_eq!(deep.variables(), vec!["x"]);
        // No variables at all.
        assert!(Expr::parse("1 + 2*3").unwrap().variables().is_empty());
        // Many distinct variables: every one is found, exactly once, in order.
        let names: Vec<String> = (0..300).map(|i| format!("v{i:03}")).collect();
        let sum = names.join(" + ");
        let big = Expr::parse(&sum).unwrap();
        let found = big.variables();
        let mut want = names.clone();
        want.sort();
        assert_eq!(found, want);
        // And repeating the whole sum does not duplicate anything.
        let doubled = Expr::parse(&format!("({sum}) + ({sum})")).unwrap();
        assert_eq!(doubled.variables(), want);
    }

}
