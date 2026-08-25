//! Properties of the feed-forward network module.
//!
//! A learning algorithm is unusually easy to test badly. A falling
//! training curve is not evidence that anything is right: gradient
//! descent reduces the loss under a wrong gradient too, just more slowly
//! and towards a different place. So none of these tests looks at a loss
//! curve as its main assertion.
//!
//! *The gradient check is the test.* Reverse-mode differentiation is
//! exact arithmetic, not an approximation, so the analytic gradient must
//! agree with a central difference to about eight digits across every
//! architecture, activation and loss. That single property covers the
//! whole of backpropagation, and nothing else covers any of it.
//!
//! *Exact identities.* Softmax is invariant under adding a constant to
//! its input. Fused with cross-entropy its output gradient is exactly
//! the prediction minus the target. A rectifier network with no biases
//! is positively homogeneous: scaling the input scales the output by the
//! same factor, layer after layer. Permuting the units of a hidden layer
//! and the corresponding weights leaves the function computed unchanged.
//! Convolution is linear and, away from the padding, commutes with a
//! shift.
//!
//! *Somewhere to be checked against.* Linear least squares has a closed
//! form, so descent on the same problem has an exact answer to reach.

use rust_physics_engine::learn::nn::{
    conv2d_forward, linear_regression_gd_check, Act, Loss, Mlp,
};
use rust_physics_engine::linalg::matrix::Matrix;
use rust_physics_engine::monte_carlo::Rng;

/// A random architecture with two or three hidden layers.
fn architecture(rng: &mut Rng) -> Vec<usize> {
    let depth = 2 + (rng.next_u64() % 2) as usize;
    let mut sizes = vec![1 + (rng.next_u64() % 4) as usize];
    for _ in 0..depth {
        sizes.push(1 + (rng.next_u64() % 5) as usize);
    }
    sizes.push(1 + (rng.next_u64() % 4) as usize);
    sizes
}

#[test]
fn prop_the_gradient_agrees_with_a_central_difference_everywhere() {
    // Across random architectures, activations, losses and inputs. This
    // is the property that says backpropagation is implemented; every
    // other test in this file assumes it.
    let mut rng = Rng::new(0x4d70_1cb2);
    for _ in 0..40 {
        let sizes = architecture(&mut rng);
        let hidden = match rng.next_u64() % 3 {
            0 => Act::Tanh,
            1 => Act::Sigmoid,
            _ => Act::Identity,
        };
        let classify = rng.next_f64() < 0.5;
        let (output, loss) = if classify {
            (Act::Softmax, Loss::CrossEntropy)
        } else {
            match rng.next_u64() % 3 {
                0 => (Act::Identity, Loss::Mse),
                1 => (Act::Sigmoid, Loss::Mse),
                _ => (Act::Tanh, Loss::Mse),
            }
        };
        let net = Mlp::new(&sizes, hidden, output, &mut rng).unwrap();
        let x: Vec<f64> = (0..net.input_size()).map(|_| 2.0 * rng.next_f64() - 1.0).collect();
        let y: Vec<f64> = if classify {
            let mut t = vec![0.0; net.output_size()];
            t[(rng.next_u64() as usize) % net.output_size()] = 1.0;
            t
        } else {
            (0..net.output_size()).map(|_| rng.next_gaussian()).collect()
        };
        let relative = net.numerical_grad_check(&x, &y, loss).unwrap();
        assert!(
            relative < 1e-7,
            "{sizes:?} {hidden:?}/{output:?}/{loss:?} disagreed by {relative}"
        );
    }
}

#[test]
fn prop_a_rectifier_only_disagrees_at_its_kinks() {
    // A rectifier is not differentiable at zero, so a central difference
    // that straddles the kink measures a slope the derivative does not
    // have. That is a real disagreement for a real reason, and the
    // useful assertion is not that it happens rarely -- a rate depends
    // on the architecture and says nothing -- but that it happens *only*
    // there. Every draw whose gradient check fails is required to have a
    // pre-activation within a few difference steps of zero; a wrong
    // gradient would fail draws that are nowhere near one.
    let mut rng = Rng::new(0x21e9_5f70);
    let step = 1e-5;
    let mut disagreements = 0;
    let mut total = 0;
    for _ in 0..25 {
        let sizes = architecture(&mut rng);
        let net = Mlp::new(&sizes, Act::Relu, Act::Identity, &mut rng).unwrap();
        for _ in 0..8 {
            let x: Vec<f64> =
                (0..net.input_size()).map(|_| 2.0 * rng.next_f64() - 1.0).collect();
            let y: Vec<f64> = (0..net.output_size()).map(|_| rng.next_gaussian()).collect();
            total += 1;
            let relative = net.numerical_grad_check(&x, &y, Loss::Mse).unwrap();
            if relative < 1e-7 {
                continue;
            }
            disagreements += 1;
            // Only the hidden layers are rectified; the output is
            // linear and has no kink to blame.
            let z = net.preactivations(&x).unwrap();
            let closest = z[..z.len() - 1]
                .iter()
                .flat_map(|layer| layer.iter())
                .map(|v| v.abs())
                .fold(f64::INFINITY, f64::min);
            assert!(
                closest < 1000.0 * step,
                "a disagreement of {relative} with the nearest kink {closest} away"
            );
        }
    }
    // And the check is doing something: if nothing ever disagreed the
    // assertion above would be vacuous.
    assert!(disagreements > 0, "no rectifier draw ever straddled a kink");
    assert!(disagreements * 2 < total, "{disagreements} of {total} draws disagreed");
}

#[test]
fn prop_softmax_ignores_a_shift_of_its_input() {
    // The invariance that makes subtracting the maximum a free
    // improvement rather than a change of answer, checked including at
    // magnitudes where the naive computation would overflow.
    let mut rng = Rng::new(0x0b64_9a31);
    for _ in 0..40 {
        let n = 2 + (rng.next_u64() % 6) as usize;
        let z: Vec<f64> = (0..n).map(|_| 20.0 * rng.next_gaussian()).collect();
        let net = Mlp {
            layers: vec![(Matrix::identity(n), vec![0.0; n])],
            activation: Act::Tanh,
            output_activation: Act::Softmax,
        };
        let p = net.forward(&z).unwrap();
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(p.iter().all(|v| v.is_finite() && *v >= 0.0));
        for shift in [-500.0, -1.0, 7.5, 600.0] {
            let moved: Vec<f64> = z.iter().map(|v| v + shift).collect();
            let q = net.forward(&moved).unwrap();
            for k in 0..n {
                assert!((p[k] - q[k]).abs() < 1e-12, "shift {shift} moved component {k}");
            }
        }
        // Order is preserved: a larger logit always gets a larger share.
        for i in 0..n {
            for j in 0..n {
                assert_eq!(z[i] < z[j], p[i] < p[j], "softmax reordered its inputs");
            }
        }
    }
}

#[test]
fn prop_the_fused_output_gradient_is_exactly_the_residual() {
    // Softmax with cross-entropy, and identity with squared error, both
    // collapse to p - y at the output pre-activation. The last layer's
    // bias gradient *is* that quantity, so it can be read off directly.
    let mut rng = Rng::new(0x7c05_e3aa);
    for _ in 0..30 {
        let n = 2 + (rng.next_u64() % 5) as usize;
        for (output, loss) in [(Act::Softmax, Loss::CrossEntropy), (Act::Identity, Loss::Mse)] {
            let net = Mlp::new(&[3, 4, n], Act::Tanh, output, &mut rng).unwrap();
            let x: Vec<f64> = (0..3).map(|_| rng.next_gaussian()).collect();
            let y: Vec<f64> = if loss == Loss::CrossEntropy {
                let mut t = vec![0.0; n];
                t[(rng.next_u64() as usize) % n] = 1.0;
                t
            } else {
                (0..n).map(|_| rng.next_gaussian()).collect()
            };
            let p = net.forward(&x).unwrap();
            let g = net.backward(&x, &y, loss).unwrap();
            let bias = &g.layers[g.layers.len() - 1].1;
            for k in 0..n {
                assert!(
                    (bias[k] - (p[k] - y[k])).abs() < 1e-12,
                    "{output:?}/{loss:?} component {k}"
                );
            }
        }
    }
}

#[test]
fn prop_a_bias_free_rectifier_network_is_positively_homogeneous() {
    // max(0, cx) = c max(0, x) for positive c, and an affine map without
    // a bias is homogeneous too, so the whole network scales with its
    // input. This is exactly why removing the biases changes what a
    // rectifier network can express, and it holds for any depth.
    let mut rng = Rng::new(0x58c1_02de);
    for _ in 0..30 {
        let sizes = architecture(&mut rng);
        let mut net = Mlp::new(&sizes, Act::Relu, Act::Relu, &mut rng).unwrap();
        for (_, b) in net.layers.iter_mut() {
            for v in b.iter_mut() {
                *v = 0.0;
            }
        }
        let x: Vec<f64> = (0..net.input_size()).map(|_| rng.next_gaussian()).collect();
        let base = net.forward(&x).unwrap();
        for c in [0.25, 1.0, 7.0] {
            let scaled: Vec<f64> = x.iter().map(|v| c * v).collect();
            let got = net.forward(&scaled).unwrap();
            for k in 0..got.len() {
                let want = c * base[k];
                assert!(
                    (got[k] - want).abs() < 1e-10 * (1.0 + want.abs()),
                    "scale {c} component {k}"
                );
            }
        }
    }
}

#[test]
fn prop_permuting_a_hidden_layer_computes_the_same_function() {
    // Hidden units carry no identity: permuting one layer's rows, and
    // the matching columns of the next layer, leaves the function alone.
    // That symmetry is why two networks trained from different
    // initialisations cannot be compared parameter by parameter, and it
    // is exact.
    let mut rng = Rng::new(0x33fa_7168);
    for _ in 0..30 {
        let hidden = 2 + (rng.next_u64() % 5) as usize;
        let net = Mlp::new(&[3, hidden, 2], Act::Tanh, Act::Identity, &mut rng).unwrap();
        let mut order: Vec<usize> = (0..hidden).collect();
        for i in (1..hidden).rev() {
            let j = (rng.next_u64() % (i as u64 + 1)) as usize;
            order.swap(i, j);
        }
        let mut shuffled = net.clone();
        {
            let (w0, b0) = &net.layers[0];
            let (nw0, nb0) = &mut shuffled.layers[0];
            for (new_row, &old_row) in order.iter().enumerate() {
                nb0[new_row] = b0[old_row];
                for j in 0..w0.cols {
                    nw0.set(new_row, j, w0.get(old_row, j));
                }
            }
            let w1 = &net.layers[1].0;
            let nw1 = &mut shuffled.layers[1].0;
            for (new_col, &old_col) in order.iter().enumerate() {
                for i in 0..w1.rows {
                    nw1.set(i, new_col, w1.get(i, old_col));
                }
            }
        }
        for _ in 0..5 {
            let x: Vec<f64> = (0..3).map(|_| rng.next_gaussian()).collect();
            let a = net.forward(&x).unwrap();
            let b = shuffled.forward(&x).unwrap();
            for k in 0..a.len() {
                assert!((a[k] - b[k]).abs() < 1e-12, "permutation changed component {k}");
            }
        }
    }
}

#[test]
fn prop_descent_reaches_the_closed_form_least_squares_answer() {
    // A convex problem whose answer is known exactly. Anything else
    // about an optimiser is a matter of degree; this is not.
    let mut rng = Rng::new(0x6ee2_bc59);
    for _ in 0..20 {
        let cols = 2 + (rng.next_u64() % 4) as usize;
        let rows = cols + 10 + (rng.next_u64() % 30) as usize;
        let mut x = Matrix::zeros(rows, cols);
        for i in 0..rows {
            x.set(i, 0, 1.0);
            for j in 1..cols {
                x.set(i, j, rng.next_gaussian());
            }
        }
        let truth: Vec<f64> = (0..cols).map(|_| 2.0 * rng.next_gaussian()).collect();
        let y: Vec<f64> = (0..rows)
            .map(|i| {
                (0..cols).map(|j| x.get(i, j) * truth[j]).sum::<f64>()
                    + 0.05 * rng.next_gaussian()
            })
            .collect();
        let relative = linear_regression_gd_check(&x, &y, 5000).unwrap();
        assert!(relative < 1e-5, "descent stopped {relative} from the exact answer");
    }
}

#[test]
fn prop_convolution_is_linear_and_shift_equivariant() {
    let mut rng = Rng::new(0x158d_40f7);
    for _ in 0..25 {
        let w = 7 + (rng.next_u64() % 5) as usize;
        let h = 7 + (rng.next_u64() % 5) as usize;
        let k = 1 + 2 * (rng.next_u64() % 2) as usize; // 1 or 3, so it is odd
        let a: Vec<f64> = (0..w * h).map(|_| rng.next_gaussian()).collect();
        let b: Vec<f64> = (0..w * h).map(|_| rng.next_gaussian()).collect();
        let kernel: (Vec<f64>, usize, usize) =
            ((0..k * k).map(|_| rng.next_gaussian()).collect(), k, k);
        let run = |img: &[f64]| conv2d_forward(img, w, h, std::slice::from_ref(&kernel), 1, 0).unwrap();
        let (ra, out_w, out_h) = run(&a);
        let (rb, _, _) = run(&b);
        let (alpha, beta) = (2.0 * rng.next_f64() - 1.0, 2.0 * rng.next_f64() - 1.0);
        let mixed: Vec<f64> = a.iter().zip(&b).map(|(x, y)| alpha * x + beta * y).collect();
        let (rm, _, _) = run(&mixed);
        for i in 0..out_w * out_h {
            let want = alpha * ra[0][i] + beta * rb[0][i];
            assert!((rm[0][i] - want).abs() < 1e-11 * (1.0 + want.abs()), "linearity at {i}");
        }
        // A shift of the input shifts the output, away from the edges
        // where the zero padding is not shift invariant and cannot be.
        let mut shifted = vec![0.0; w * h];
        for y in 0..h {
            for x in 1..w {
                shifted[y * w + x] = a[y * w + x - 1];
            }
        }
        let (rs, _, _) = run(&shifted);
        for y in 0..out_h {
            for x in 1..out_w {
                let want = ra[0][y * out_w + x - 1];
                assert!(
                    (rs[0][y * out_w + x] - want).abs() < 1e-11 * (1.0 + want.abs()),
                    "shift at ({x}, {y})"
                );
            }
        }
        // The output size is what the formula says, for every stride.
        for stride in 1..=3 {
            for pad in 0..=2 {
                let padded_w = w + 2 * pad;
                let padded_h = h + 2 * pad;
                let (_, ow, oh) =
                    conv2d_forward(&a, w, h, std::slice::from_ref(&kernel), stride, pad).unwrap();
                assert_eq!(ow, (padded_w - k) / stride + 1);
                assert_eq!(oh, (padded_h - k) / stride + 1);
            }
        }
    }
}

#[test]
fn prop_a_uniform_kernel_averages_and_a_delta_copies() {
    // Two kernels whose effect is known in closed form, at every size
    // and on every image: the mean over the window, and the identity.
    let mut rng = Rng::new(0x2a71_c8e4);
    for _ in 0..25 {
        let w = 6 + (rng.next_u64() % 6) as usize;
        let h = 6 + (rng.next_u64() % 6) as usize;
        let k = 3;
        let img: Vec<f64> = (0..w * h).map(|_| rng.next_gaussian()).collect();
        let mean_kernel = (vec![1.0 / (k * k) as f64; k * k], k, k);
        let (blurred, ow, oh) = conv2d_forward(&img, w, h, &[mean_kernel], 1, 0).unwrap();
        for oy in 0..oh {
            for ox in 0..ow {
                let want: f64 = (0..k)
                    .flat_map(|dy| (0..k).map(move |dx| (dx, dy)))
                    .map(|(dx, dy)| img[(oy + dy) * w + ox + dx])
                    .sum::<f64>()
                    / (k * k) as f64;
                let got = blurred[0][oy * ow + ox];
                assert!((got - want).abs() < 1e-12 * (1.0 + want.abs()), "mean at ({ox},{oy})");
            }
        }
        let mut delta = vec![0.0; k * k];
        delta[k * k / 2] = 1.0;
        let (copied, _, _) = conv2d_forward(&img, w, h, &[(delta, k, k)], 1, 1).unwrap();
        for i in 0..w * h {
            assert!((copied[0][i] - img[i]).abs() < 1e-15, "the delta did not copy at {i}");
        }
    }
}
