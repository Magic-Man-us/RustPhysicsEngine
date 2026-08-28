# rust_physics_engine, from Python

Python bindings for [rust_physics_engine][crate] — 4,086 functions, 2,254
methods, 416 classes and 106 constants across 71 domains, from Newtonian
mechanics to Reed–Solomon codes, with no runtime dependencies on either
side.

```console
$ pip install ./bindings/python
```

or, to work on the bindings themselves, from an activated virtualenv:

```console
$ pip install maturin
$ maturin develop --release -m bindings/python/Cargo.toml
```

```python
>>> import math
>>> import rust_physics_engine as rpe
>>> rpe.classical.projectile_range(speed=20.0, angle_rad=math.pi / 4, g=9.80665)
40.78864851911713
```

Every Python module mirrors a Rust module of the same name, and every
function keeps the name, the argument order and the units it has in Rust.
If you can read `docs/MODULE_MAP.md`, you can find your way around here.

## What the bindings add

The Rust API is not changed, but five things are translated so that it
reads as Python rather than as Rust seen through glass.

**Errors are exceptions.** `Result<T, SolveError>` becomes a return value
and a raise; the variants that carry data carry it onto the exception.

```python
>>> try:
...     rpe.linalg.lu.solve([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0])
... except rpe.SingularMatrixError as e:
...     print(e)
matrix is singular or pivot below threshold
```

Everything raised derives from `PhysicsError`, so `except PhysicsError`
catches all of it and nothing else:

```
PhysicsError
├── InvalidArgumentError      a documented precondition was violated
├── SolverError
│   ├── SingularMatrixError
│   ├── NotPositiveDefiniteError
│   ├── ConvergenceError          .iterations, .residual
│   └── DimensionMismatchError    .expected, .got
├── GeometryError
│   ├── DegenerateGeometryError
│   ├── NotManifoldError
│   └── EmptyInputError
└── UnitsError
```

The library also validates arguments with `assert!`, which in Rust is the
right call — a negative mass is a programming error, not a runtime
condition. Those become `InvalidArgumentError` carrying the assertion's
own message, rather than aborting the interpreter:

```python
>>> rpe.classical.acceleration(force=10.0, mass=-1.0)
Traceback (most recent call last):
rust_physics_engine.InvalidArgumentError: mass must be positive
```

**Small value types accept literals.** Anywhere a `Vec2`, `Vec3`, `Vec4`,
`Mat3` or `Quaternion` is expected, a sequence of the right length will
do; anywhere a `Matrix` is expected, a list of rows will do. The wrapper
classes still exist, with their methods, their operators and their
`tolist()`.

```python
>>> rpe.classical.position_3d((0, 0, 100), (5, 0, 0), (0, 0, -9.81), 2.0).tolist()
[10.0, 0.0, 80.38]
>>> v = rpe.math.Vec3(1, 2, 2)
>>> v.magnitude(), (v + (1, 0, 0)).tolist(), v[0], list(v)
(3.0, [2.0, 2.0, 2.0], 1.0, [1.0, 2.0, 2.0])
```

**Three Rust types are Python types.** They have exact counterparts, so
they are translated rather than wrapped, and the round trip loses nothing:

| Rust | Python |
|---|---|
| `fractals::Complex` | `complex` |
| `exact::bigint::BigInt` | `int`, of any size |
| `exact::rational::Rational` | `fractions.Fraction` |

```python
>>> rpe.exact.bigint.factorial(30)
265252859812191058636308480000000
>>> rpe.transforms.fft.fft([1, 0, 0, 0])
[(1+0j), (1+0j), (1+0j), (1+0j)]
```

**Builders chain.** A Rust method that takes `&mut self` and returns
`&mut Self` hands the same Python object back, so a circuit reads the way
it does in Rust:

```python
>>> c = rpe.quantum.circuit.Circuit(2)
>>> c.h(0).cx(0, 1)                                  # a Bell pair
>>> state = c.run(rpe.quantum.circuit.QState.zero(2))
>>> [round(abs(z) ** 2, 3) for z in state.amps]
[0.5, 0.0, 0.0, 0.5]
```

**Functions can be Python functions.** Anywhere the library takes a
`&dyn Fn`, pass a callable. An exception raised inside it comes back out
of the call with its own traceback, rather than turning into a NaN:

```python
>>> import math
>>> rpe.numerical.integrate.simpson(math.sin, 0.0, math.pi, 1000)
2.0000000000010805
>>> rpe.numerical.roots.newton_raphson(lambda x: x*x - 2, lambda x: 2*x, 1.0, 1e-12, 50)
1.414213562373095
```

## Types, and your editor

The package ships `py.typed` and a `.pyi` stub for every module, so
`mypy`, `pyright` and editor completion all work without importing the
extension.

## What is not bound

4,086 of the library's 4,149 free functions, 2,254 of its 2,277 methods,
416 of its 426 types and all 106 of its constants, across 296 modules.

The rest is mostly three things: functions generic over a type parameter,
which cannot be monomorphised without knowing what to monomorphise to;
`&dyn Trait` arguments for traits with no Python equivalent; and routines
returning a closure. [COVERAGE.md](COVERAGE.md) lists every unbound item
by name with its reason, and is regenerated with the bindings, so it
cannot drift from them.

## How this is built

`generate.py` reads the library's source with `rustscan.py` and writes the
wrapper for every item it can bind. The alternative — writing 6,000
wrappers by hand — fails quietly: the first commit that adds a function to
the library leaves the binding stale, and nothing breaks to tell you.
Here, regenerating is one command, and CI runs

```console
$ python3 bindings/python/generate.py --check
```

which fails if what is committed differs from what the current source
produces.

To work on the bindings:

```console
$ python3 bindings/python/generate.py     # after changing the library
$ maturin develop --release -m bindings/python/Cargo.toml
$ python -m pytest bindings/python/tests
```

The hand-written half is small and lives in `src/runtime/`: the exception
hierarchy and the panic guard (`errors.rs`), the literal coercions and the
three type identifications (`coerce.rs`), and the callable adapter
(`callback.rs`). Anything a generator cannot reach — a Python protocol
like `__getitem__`, a method defined by a `macro_rules!` the scanner
cannot see — is declared in a table at the top of `generate.py` and
spliced in, so there is one place to look.

## Performance notes

The GIL is released around calls that do real work — those taking or
returning arrays — so several threads can compute at once. It is held for
scalar calls, where releasing it would cost more than the call, and for
anything involving a Python callable, which needs it.

[crate]: https://github.com/Magic-Man-us/RustPhysicsEngine
