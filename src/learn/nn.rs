//! Feed-forward networks, trained by backpropagation.
//!
//! # Backpropagation is the chain rule with the products reassociated
//!
//! The derivative of the loss with respect to an early weight is a
//! product of Jacobians, one per layer. Multiplying them left to right
//! costs a matrix-matrix product per layer; multiplying right to left,
//! starting from the scalar loss, costs a matrix-*vector* product per
//! layer. Backpropagation is the second association, and that is the
//! whole of it. It is not an approximation and it is not specific to
//! neural networks -- it is reverse-mode differentiation, and the cost
//! of one gradient is a small multiple of the cost of one forward pass
//! however many parameters there are.
//!
//! Which is why [`Mlp::numerical_grad_check`] is the test that matters.
//! Descent will reduce a loss using wrong gradients, just more slowly
//! and towards somewhere else, so a falling training curve is no
//! evidence at all. A central difference agreeing with the analytic
//! gradient to eight digits is.
//!
//! # Softmax and cross-entropy belong together
//!
//! Taken separately, softmax has a Jacobian and cross-entropy has a
//! gradient, and composing them involves a matrix. Taken together the
//! product collapses: the gradient of cross-entropy with respect to the
//! *logits* is exactly `p - y`, the predicted distribution minus the
//! target. That cancellation is worth having for accuracy as well as
//! speed -- computing the two separately loses precision exactly where
//! the network is confident and the softmax output is near zero or one.
//! The two are therefore fused here, and [`Loss::CrossEntropy`] requires
//! [`Act::Softmax`] on the output layer.
//!
//! # Initialisation is not cosmetic
//!
//! Weights start from a scaled normal draw -- the He scaling
//! `sqrt(2/fan_in)` for rectifiers, the Xavier scaling
//! `sqrt(1/fan_in)` otherwise. Initialising everything to zero makes
//! every hidden unit in a layer compute the same thing and receive the
//! same gradient forever, so the layer has one effective unit no matter
//! how wide it is; initialising too large saturates the sigmoid and
//! tanh, whose derivative is then near zero and whose gradient
//! therefore vanishes.

use crate::error::SolveError;
use crate::linalg::matrix::Matrix;
use crate::monte_carlo::Rng;

/// The activation applied after a layer's affine map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Act {
    /// `max(0, x)`. Cheap, and its derivative does not vanish for large
    /// input, which is what lets deep rectifier networks train at all.
    /// A unit whose input is negative for every example is dead: its
    /// gradient is exactly zero and it never recovers.
    Relu,
    /// `1 / (1 + e^-x)`, saturating at zero and one.
    Sigmoid,
    /// `tanh x`, saturating at minus one and one. Zero-centred, which
    /// makes it better behaved than the sigmoid in a hidden layer.
    Tanh,
    /// No activation at all, for a regression output.
    Identity,
    /// The normalised exponential over a whole layer, for a
    /// distribution over classes. Unlike the others it couples the
    /// units of its layer to each other.
    Softmax,
}

impl Act {
    /// Applies the activation to a whole layer.
    fn apply(self, z: &[f64]) -> Vec<f64> {
        match self {
            Act::Relu => z.iter().map(|v| v.max(0.0)).collect(),
            Act::Sigmoid => z.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect(),
            Act::Tanh => z.iter().map(|v| v.tanh()).collect(),
            Act::Identity => z.to_vec(),
            Act::Softmax => {
                // Subtracting the maximum changes nothing mathematically
                // -- softmax is invariant under a shift of its input --
                // and everything numerically: without it a logit of 800
                // overflows and the answer is NaN rather than the
                // one-hot vector it should be.
                let peak = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let raw: Vec<f64> = z.iter().map(|v| (v - peak).exp()).collect();
                let total: f64 = raw.iter().sum();
                raw.iter().map(|v| v / total).collect()
            }
        }
    }

    /// The derivative of the activation with respect to its input, given
    /// the *output*, for the element-wise activations only.
    fn derivative_from_output(self, a: f64) -> f64 {
        match self {
            Act::Relu => {
                if a > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Act::Sigmoid => a * (1.0 - a),
            Act::Tanh => 1.0 - a * a,
            Act::Identity => 1.0,
            // Softmax is not element-wise; it is only ever used fused
            // with cross-entropy, where the Jacobian cancels.
            Act::Softmax => f64::NAN,
        }
    }
}

/// What the network is asked to minimise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Loss {
    /// Mean squared error, halved so that its gradient is the plain
    /// residual.
    Mse,
    /// Categorical cross-entropy, which must be paired with
    /// [`Act::Softmax`] on the output.
    CrossEntropy,
}

/// A fully connected feed-forward network.
#[derive(Debug, Clone, PartialEq)]
pub struct Mlp {
    /// Weight matrix and bias vector for each layer. The weight matrix
    /// of layer `k` is `out x in`.
    pub layers: Vec<(Matrix, Vec<f64>)>,
    /// Activation on every hidden layer.
    pub activation: Act,
    /// Activation on the output layer, which is usually not the same
    /// one -- a regression wants [`Act::Identity`] and a classifier
    /// wants [`Act::Softmax`].
    pub output_activation: Act,
}

/// The gradient of the loss with respect to every parameter, shaped
/// like the network itself.
#[derive(Debug, Clone, PartialEq)]
pub struct Gradients {
    /// One weight-gradient matrix and bias-gradient vector per layer.
    pub layers: Vec<(Matrix, Vec<f64>)>,
}

impl Gradients {
    /// Zero gradients shaped like the given network.
    fn zeros_like(net: &Mlp) -> Self {
        Self {
            layers: net
                .layers
                .iter()
                .map(|(w, b)| (Matrix::zeros(w.rows, w.cols), vec![0.0; b.len()]))
                .collect(),
        }
    }

    /// Adds another set of gradients into this one.
    fn add(&mut self, other: &Gradients) {
        for ((w, b), (ow, ob)) in self.layers.iter_mut().zip(other.layers.iter()) {
            for i in 0..w.rows {
                for j in 0..w.cols {
                    w.set(i, j, w.get(i, j) + ow.get(i, j));
                }
            }
            for (v, o) in b.iter_mut().zip(ob.iter()) {
                *v += o;
            }
        }
    }

    /// Scales every entry.
    fn scale(&mut self, k: f64) {
        for (w, b) in self.layers.iter_mut() {
            for i in 0..w.rows {
                for j in 0..w.cols {
                    w.set(i, j, w.get(i, j) * k);
                }
            }
            for v in b.iter_mut() {
                *v *= k;
            }
        }
    }

    /// The Euclidean norm over every parameter, used to compare a
    /// gradient against a finite-difference estimate.
    pub fn norm(&self) -> f64 {
        let mut total = 0.0;
        for (w, b) in &self.layers {
            for i in 0..w.rows {
                for j in 0..w.cols {
                    total += w.get(i, j) * w.get(i, j);
                }
            }
            total += b.iter().map(|v| v * v).sum::<f64>();
        }
        total.sqrt()
    }
}

impl Mlp {
    /// Builds a network with the given layer sizes, the first being the
    /// input width and the last the output width.
    ///
    /// Weights are drawn from a normal distribution scaled by fan-in --
    /// He for rectifiers, Xavier otherwise -- and biases start at zero.
    /// See the module note on why neither choice is cosmetic.
    ///
    /// # Errors
    ///
    /// [`SolveError::InvalidArgument`] for fewer than two sizes or any
    /// zero-width layer.
    pub fn new(
        sizes: &[usize],
        activation: Act,
        output_activation: Act,
        rng: &mut Rng,
    ) -> Result<Self, SolveError> {
        if sizes.len() < 2 {
            return Err(SolveError::InvalidArgument("need an input and an output size"));
        }
        if sizes.contains(&0) {
            return Err(SolveError::InvalidArgument("every layer needs at least one unit"));
        }
        let mut layers = Vec::with_capacity(sizes.len() - 1);
        for k in 0..sizes.len() - 1 {
            let (fan_in, fan_out) = (sizes[k], sizes[k + 1]);
            let scale = if activation == Act::Relu {
                (2.0 / fan_in as f64).sqrt()
            } else {
                (1.0 / fan_in as f64).sqrt()
            };
            let mut w = Matrix::zeros(fan_out, fan_in);
            for i in 0..fan_out {
                for j in 0..fan_in {
                    w.set(i, j, scale * rng.next_gaussian());
                }
            }
            layers.push((w, vec![0.0; fan_out]));
        }
        Ok(Self { layers, activation, output_activation })
    }

    /// The input width the network expects.
    pub fn input_size(&self) -> usize {
        self.layers[0].0.cols
    }

    /// The output width.
    pub fn output_size(&self) -> usize {
        self.layers[self.layers.len() - 1].0.rows
    }

    /// The total parameter count.
    pub fn parameter_count(&self) -> usize {
        self.layers.iter().map(|(w, b)| w.rows * w.cols + b.len()).sum()
    }

    /// Runs the network forward, returning the activations of every
    /// layer including the input.
    fn forward_all(&self, x: &[f64]) -> Result<Vec<Vec<f64>>, SolveError> {
        if x.len() != self.input_size() {
            return Err(SolveError::DimensionMismatch {
                expected: self.input_size(),
                got: x.len(),
            });
        }
        let mut acts = Vec::with_capacity(self.layers.len() + 1);
        acts.push(x.to_vec());
        for (k, (w, b)) in self.layers.iter().enumerate() {
            let last = acts.last().expect("the input is always present");
            let mut z = w.mul_vec(last)?;
            for (v, bias) in z.iter_mut().zip(b.iter()) {
                *v += bias;
            }
            let act = if k + 1 == self.layers.len() {
                self.output_activation
            } else {
                self.activation
            };
            acts.push(act.apply(&z));
        }
        Ok(acts)
    }

    /// The pre-activation of every layer -- the affine map's output,
    /// before the activation is applied.
    ///
    /// Worth having in public because it is what says how close a
    /// rectifier unit is to its kink. A unit whose pre-activation is
    /// near zero is where a finite-difference gradient check is entitled
    /// to disagree with the analytic gradient, and where a unit is about
    /// to die or come back to life.
    ///
    /// # Errors
    ///
    /// [`SolveError::DimensionMismatch`] if the input width is wrong.
    pub fn preactivations(&self, x: &[f64]) -> Result<Vec<Vec<f64>>, SolveError> {
        if x.len() != self.input_size() {
            return Err(SolveError::DimensionMismatch {
                expected: self.input_size(),
                got: x.len(),
            });
        }
        let mut out = Vec::with_capacity(self.layers.len());
        let mut current = x.to_vec();
        for (k, (w, b)) in self.layers.iter().enumerate() {
            let mut z = w.mul_vec(&current)?;
            for (v, bias) in z.iter_mut().zip(b.iter()) {
                *v += bias;
            }
            let act = if k + 1 == self.layers.len() {
                self.output_activation
            } else {
                self.activation
            };
            current = act.apply(&z);
            out.push(z);
        }
        Ok(out)
    }

    /// Runs the network forward.
    ///
    /// # Errors
    ///
    /// [`SolveError::DimensionMismatch`] if the input width is wrong.
    pub fn forward(&self, x: &[f64]) -> Result<Vec<f64>, SolveError> {
        Ok(self.forward_all(x)?.pop().expect("there is always an output"))
    }

    /// The index of the largest output, for a classifier.
    ///
    /// # Errors
    ///
    /// As [`Mlp::forward`].
    pub fn predict(&self, x: &[f64]) -> Result<usize, SolveError> {
        let out = self.forward(x)?;
        Ok(out
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .expect("the output layer is never empty"))
    }

    /// The loss on one example.
    ///
    /// # Errors
    ///
    /// [`SolveError::DimensionMismatch`] on a width mismatch;
    /// [`SolveError::InvalidArgument`] if cross-entropy is asked for
    /// without a softmax output.
    pub fn example_loss(&self, x: &[f64], y: &[f64], loss: Loss) -> Result<f64, SolveError> {
        let out = self.forward(x)?;
        if y.len() != out.len() {
            return Err(SolveError::DimensionMismatch { expected: out.len(), got: y.len() });
        }
        match loss {
            Loss::Mse => {
                Ok(0.5 * out.iter().zip(y).map(|(p, t)| (p - t) * (p - t)).sum::<f64>())
            }
            Loss::CrossEntropy => {
                if self.output_activation != Act::Softmax {
                    return Err(SolveError::InvalidArgument(
                        "cross-entropy needs a softmax output layer",
                    ));
                }
                // Clamped away from zero: a confident wrong answer would
                // otherwise give an infinite loss and a NaN average,
                // losing the information that the rest of the batch
                // carries.
                Ok(-out
                    .iter()
                    .zip(y)
                    .map(|(p, t)| t * p.max(1e-300).ln())
                    .sum::<f64>())
            }
        }
    }

    /// The mean loss over a dataset.
    ///
    /// # Errors
    ///
    /// As [`Mlp::example_loss`], plus
    /// [`SolveError::InvalidArgument`] for an empty dataset.
    pub fn loss(&self, data: &[(Vec<f64>, Vec<f64>)], loss: Loss) -> Result<f64, SolveError> {
        if data.is_empty() {
            return Err(SolveError::InvalidArgument("the dataset is empty"));
        }
        let mut total = 0.0;
        for (x, y) in data {
            total += self.example_loss(x, y, loss)?;
        }
        Ok(total / data.len() as f64)
    }

    /// The gradient of the loss on one example, by backpropagation.
    ///
    /// # Errors
    ///
    /// As [`Mlp::example_loss`].
    pub fn backward(&self, x: &[f64], y: &[f64], loss: Loss) -> Result<Gradients, SolveError> {
        let acts = self.forward_all(x)?;
        let out = acts.last().expect("there is always an output");
        if y.len() != out.len() {
            return Err(SolveError::DimensionMismatch { expected: out.len(), got: y.len() });
        }
        if loss == Loss::CrossEntropy && self.output_activation != Act::Softmax {
            return Err(SolveError::InvalidArgument(
                "cross-entropy needs a softmax output layer",
            ));
        }
        // The error signal at the output layer's pre-activation.
        //
        // For softmax with cross-entropy, and for identity with squared
        // error, this is `p - y` and the activation's Jacobian has
        // already cancelled against the loss's gradient. For any other
        // pairing the element-wise derivative has to be applied.
        let mut delta: Vec<f64> = out.iter().zip(y).map(|(p, t)| p - t).collect();
        let fused = (loss == Loss::CrossEntropy && self.output_activation == Act::Softmax)
            || (loss == Loss::Mse && self.output_activation == Act::Identity);
        if !fused {
            if self.output_activation == Act::Softmax {
                return Err(SolveError::InvalidArgument(
                    "a softmax output is only supported with cross-entropy",
                ));
            }
            for (d, &a) in delta.iter_mut().zip(out.iter()) {
                *d *= self.output_activation.derivative_from_output(a);
            }
        }
        let mut grads = Gradients::zeros_like(self);
        for k in (0..self.layers.len()).rev() {
            let input = &acts[k];
            let (gw, gb) = &mut grads.layers[k];
            for i in 0..gw.rows {
                gb[i] = delta[i];
                for j in 0..gw.cols {
                    gw.set(i, j, delta[i] * input[j]);
                }
            }
            if k > 0 {
                // Propagate to the previous layer: W^T delta, then the
                // element-wise derivative there.
                let w = &self.layers[k].0;
                let mut next = vec![0.0; w.cols];
                for (j, slot) in next.iter_mut().enumerate() {
                    *slot = (0..w.rows).map(|i| w.get(i, j) * delta[i]).sum();
                }
                for (d, &a) in next.iter_mut().zip(acts[k].iter()) {
                    *d *= self.activation.derivative_from_output(a);
                }
                delta = next;
            }
        }
        Ok(grads)
    }

    /// Every parameter as one flat vector, in a fixed order.
    fn parameters(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.parameter_count());
        for (w, b) in &self.layers {
            for i in 0..w.rows {
                for j in 0..w.cols {
                    out.push(w.get(i, j));
                }
            }
            out.extend_from_slice(b);
        }
        out
    }

    /// Writes a flat parameter vector back into the network.
    fn set_parameters(&mut self, flat: &[f64]) {
        let mut k = 0;
        for (w, b) in self.layers.iter_mut() {
            for i in 0..w.rows {
                for j in 0..w.cols {
                    w.set(i, j, flat[k]);
                    k += 1;
                }
            }
            for v in b.iter_mut() {
                *v = flat[k];
                k += 1;
            }
        }
    }

    /// Compares the analytic gradient against a central difference,
    /// returning the relative difference of the two as vectors.
    ///
    /// This is the test that decides whether backpropagation was
    /// implemented correctly. Training curves do not: descent reduces
    /// the loss under a wrong gradient too, just more slowly and towards
    /// somewhere else.
    ///
    /// A central difference is used rather than a forward one because
    /// its truncation error is `O(h^2)` instead of `O(h)`, which with
    /// `h = 1e-5` puts the truncation and the rounding at about the same
    /// size and leaves eight digits of agreement to look for. A forward
    /// difference would leave four, which is not enough to distinguish a
    /// correct gradient from a nearly correct one.
    ///
    /// # Errors
    ///
    /// As [`Mlp::backward`].
    pub fn numerical_grad_check(
        &self,
        x: &[f64],
        y: &[f64],
        loss: Loss,
    ) -> Result<f64, SolveError> {
        let analytic = self.backward(x, y, loss)?;
        let flat = self.flatten(&analytic);
        let mut probe = self.clone();
        let base = self.parameters();
        let h = 1e-5;
        let mut numeric = Vec::with_capacity(base.len());
        for k in 0..base.len() {
            let mut up = base.clone();
            up[k] += h;
            probe.set_parameters(&up);
            let plus = probe.example_loss(x, y, loss)?;
            let mut down = base.clone();
            down[k] -= h;
            probe.set_parameters(&down);
            let minus = probe.example_loss(x, y, loss)?;
            numeric.push((plus - minus) / (2.0 * h));
        }
        let diff: f64 = flat
            .iter()
            .zip(numeric.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        let scale = flat.iter().map(|v| v * v).sum::<f64>().sqrt()
            + numeric.iter().map(|v| v * v).sum::<f64>().sqrt();
        Ok(if scale > 0.0 { diff / scale } else { diff })
    }

    /// Flattens gradients in the same order as [`Mlp::parameters`].
    fn flatten(&self, g: &Gradients) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.parameter_count());
        for (w, b) in &g.layers {
            for i in 0..w.rows {
                for j in 0..w.cols {
                    out.push(w.get(i, j));
                }
            }
            out.extend_from_slice(b);
        }
        out
    }

    /// Applies a gradient step, `p -= lr * g`.
    fn step(&mut self, g: &Gradients, lr: f64) {
        for ((w, b), (gw, gb)) in self.layers.iter_mut().zip(g.layers.iter()) {
            for i in 0..w.rows {
                for j in 0..w.cols {
                    w.set(i, j, w.get(i, j) - lr * gw.get(i, j));
                }
            }
            for (v, d) in b.iter_mut().zip(gb.iter()) {
                *v -= lr * d;
            }
        }
    }

    /// The mean gradient over a batch.
    fn batch_gradient(
        &self,
        batch: &[&(Vec<f64>, Vec<f64>)],
        loss: Loss,
    ) -> Result<Gradients, SolveError> {
        let mut total = Gradients::zeros_like(self);
        for (x, y) in batch {
            total.add(&self.backward(x, y, loss)?);
        }
        total.scale(1.0 / batch.len() as f64);
        Ok(total)
    }

    /// Trains by mini-batch stochastic gradient descent, returning the
    /// mean loss after each epoch.
    ///
    /// # Errors
    ///
    /// [`SolveError::InvalidArgument`] for an empty dataset, a
    /// non-positive batch size, or a non-finite learning rate.
    pub fn train_sgd(
        &mut self,
        data: &[(Vec<f64>, Vec<f64>)],
        loss: Loss,
        epochs: usize,
        lr: f64,
        batch: usize,
        rng: &mut Rng,
    ) -> Result<Vec<f64>, SolveError> {
        if data.is_empty() {
            return Err(SolveError::InvalidArgument("the dataset is empty"));
        }
        if batch == 0 {
            return Err(SolveError::InvalidArgument("the batch size must be positive"));
        }
        if !lr.is_finite() || lr <= 0.0 {
            return Err(SolveError::InvalidArgument("the learning rate must be positive"));
        }
        let mut history = Vec::with_capacity(epochs);
        let mut order: Vec<usize> = (0..data.len()).collect();
        for _ in 0..epochs {
            shuffle(&mut order, rng);
            for chunk in order.chunks(batch) {
                let picked: Vec<&(Vec<f64>, Vec<f64>)> =
                    chunk.iter().map(|&i| &data[i]).collect();
                let g = self.batch_gradient(&picked, loss)?;
                self.step(&g, lr);
            }
            history.push(self.loss(data, loss)?);
        }
        Ok(history)
    }

    /// Trains with Adam, returning the mean loss after each epoch.
    ///
    /// Adam keeps a running mean and a running mean square of each
    /// parameter's gradient and steps by their ratio, which makes the
    /// step size roughly scale-free: multiplying every gradient by a
    /// constant leaves the update almost unchanged. The bias correction
    /// matters most at the start, where both running averages begin at
    /// zero and would otherwise make the first steps far too small.
    ///
    /// # Errors
    ///
    /// As [`Mlp::train_sgd`].
    pub fn train_adam(
        &mut self,
        data: &[(Vec<f64>, Vec<f64>)],
        loss: Loss,
        epochs: usize,
        lr: f64,
        batch: usize,
        rng: &mut Rng,
    ) -> Result<Vec<f64>, SolveError> {
        if data.is_empty() {
            return Err(SolveError::InvalidArgument("the dataset is empty"));
        }
        if batch == 0 {
            return Err(SolveError::InvalidArgument("the batch size must be positive"));
        }
        if !lr.is_finite() || lr <= 0.0 {
            return Err(SolveError::InvalidArgument("the learning rate must be positive"));
        }
        const B1: f64 = 0.9;
        const B2: f64 = 0.999;
        const EPS: f64 = 1e-8;
        let n = self.parameter_count();
        let mut m = vec![0.0; n];
        let mut v = vec![0.0; n];
        let mut t = 0u32;
        let mut history = Vec::with_capacity(epochs);
        let mut order: Vec<usize> = (0..data.len()).collect();
        for _ in 0..epochs {
            shuffle(&mut order, rng);
            for chunk in order.chunks(batch) {
                let picked: Vec<&(Vec<f64>, Vec<f64>)> =
                    chunk.iter().map(|&i| &data[i]).collect();
                let g = self.flatten(&self.batch_gradient(&picked, loss)?);
                t += 1;
                let c1 = 1.0 - B1.powi(t as i32);
                let c2 = 1.0 - B2.powi(t as i32);
                let mut p = self.parameters();
                for k in 0..n {
                    m[k] = B1 * m[k] + (1.0 - B1) * g[k];
                    v[k] = B2 * v[k] + (1.0 - B2) * g[k] * g[k];
                    p[k] -= lr * (m[k] / c1) / ((v[k] / c2).sqrt() + EPS);
                }
                self.set_parameters(&p);
            }
            history.push(self.loss(data, loss)?);
        }
        Ok(history)
    }
}

/// A Fisher-Yates shuffle with the crate's own generator.
fn shuffle(order: &mut [usize], rng: &mut Rng) {
    for i in (1..order.len()).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
}

/// One convolution layer's forward pass: `kernels` applied to a single
/// channel image, with the given stride and zero padding.
///
/// Returns one output plane per kernel, each row-major, along with the
/// output width and height. The convolution here is the cross-correlation
/// that every machine learning library calls a convolution -- the kernel
/// is *not* flipped. Against a symmetric kernel the two agree and the
/// distinction never shows; against an asymmetric one they differ by a
/// reflection, so a signal-processing convolution needs the kernel
/// reversed on the way in.
///
/// # Errors
///
/// [`SolveError::InvalidArgument`] for a zero stride, an empty kernel
/// set, a kernel larger than the padded image, or mismatched sizes;
/// [`SolveError::DimensionMismatch`] if the image is not `w * h`.
pub fn conv2d_forward(
    input: &[f64],
    w: usize,
    h: usize,
    kernels: &[(Vec<f64>, usize, usize)],
    stride: usize,
    pad: usize,
) -> Result<(Vec<Vec<f64>>, usize, usize), SolveError> {
    if input.len() != w * h {
        return Err(SolveError::DimensionMismatch { expected: w * h, got: input.len() });
    }
    if stride == 0 {
        return Err(SolveError::InvalidArgument("the stride must be positive"));
    }
    if kernels.is_empty() {
        return Err(SolveError::InvalidArgument("need at least one kernel"));
    }
    let (kw, kh) = (kernels[0].1, kernels[0].2);
    for (k, a, b) in kernels {
        if *a != kw || *b != kh {
            return Err(SolveError::InvalidArgument("the kernels differ in size"));
        }
        if k.len() != a * b {
            return Err(SolveError::DimensionMismatch { expected: a * b, got: k.len() });
        }
        if *a == 0 || *b == 0 {
            return Err(SolveError::InvalidArgument("a kernel cannot be empty"));
        }
    }
    let padded_w = w + 2 * pad;
    let padded_h = h + 2 * pad;
    if kw > padded_w || kh > padded_h {
        return Err(SolveError::InvalidArgument("the kernel is larger than the padded image"));
    }
    let out_w = (padded_w - kw) / stride + 1;
    let out_h = (padded_h - kh) / stride + 1;
    let sample = |x: i64, y: i64| -> f64 {
        if x < 0 || y < 0 || x >= w as i64 || y >= h as i64 {
            0.0
        } else {
            input[y as usize * w + x as usize]
        }
    };
    let mut planes = Vec::with_capacity(kernels.len());
    for (kernel, _, _) in kernels {
        let mut plane = vec![0.0; out_w * out_h];
        for oy in 0..out_h {
            for ox in 0..out_w {
                let mut total = 0.0;
                for ky in 0..kh {
                    for kx in 0..kw {
                        let sx = (ox * stride + kx) as i64 - pad as i64;
                        let sy = (oy * stride + ky) as i64 - pad as i64;
                        total += kernel[ky * kw + kx] * sample(sx, sy);
                    }
                }
                plane[oy * out_w + ox] = total;
            }
        }
        planes.push(plane);
    }
    Ok((planes, out_w, out_h))
}

/// Fits `y = X b` by gradient descent and reports how far the answer is
/// from the closed-form least-squares solution, relative to its size.
///
/// The point is the comparison. Least squares has an exact answer
/// through the normal equations, so an iterative method solving the same
/// problem has somewhere to be checked against -- and that check is
/// worth more than any amount of watching a loss go down, because a
/// descent with the wrong gradient also produces a loss that goes down.
///
/// The step size is taken as `1 / L` with `L` the largest eigenvalue of
/// `X^T X`, estimated by a few power iterations. That is the largest
/// step for which gradient descent on a quadratic is guaranteed to
/// converge, and going past it diverges rather than converging slowly.
///
/// # Errors
///
/// [`SolveError::InvalidArgument`] for an empty or ill-shaped problem;
/// whatever the least-squares solver reports otherwise.
pub fn linear_regression_gd_check(
    x: &Matrix,
    y: &[f64],
    iterations: usize,
) -> Result<f64, SolveError> {
    if x.rows == 0 || x.cols == 0 {
        return Err(SolveError::InvalidArgument("the design matrix is empty"));
    }
    if y.len() != x.rows {
        return Err(SolveError::DimensionMismatch { expected: x.rows, got: y.len() });
    }
    let exact = crate::linalg::qr::least_squares(x, y)?;
    let n = x.cols;
    // The Lipschitz constant of the gradient is the largest eigenvalue
    // of X^T X; a few power iterations bound it well enough, and the
    // slight overestimate from stopping early is on the safe side.
    let mut v = vec![1.0; n];
    let mut lipschitz = 1.0;
    for _ in 0..50 {
        let xv = x.mul_vec(&v)?;
        let mut next = vec![0.0; n];
        for j in 0..n {
            next[j] = (0..x.rows).map(|i| x.get(i, j) * xv[i]).sum();
        }
        let norm = next.iter().map(|a| a * a).sum::<f64>().sqrt();
        if norm <= 0.0 {
            return Err(SolveError::Singular);
        }
        lipschitz = norm;
        for (slot, value) in v.iter_mut().zip(next.iter()) {
            *slot = value / norm;
        }
    }
    let lr = 1.0 / lipschitz;
    let mut beta = vec![0.0; n];
    for _ in 0..iterations {
        let residual: Vec<f64> =
            x.mul_vec(&beta)?.iter().zip(y).map(|(p, t)| p - t).collect();
        for j in 0..n {
            let g: f64 = (0..x.rows).map(|i| x.get(i, j) * residual[i]).sum();
            beta[j] -= lr * g;
        }
    }
    let diff: f64 = beta
        .iter()
        .zip(exact.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt();
    let scale = exact.iter().map(|v| v * v).sum::<f64>().sqrt();
    Ok(if scale > 0.0 { diff / scale } else { diff })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The four XOR examples, as a regression target.
    fn xor() -> Vec<(Vec<f64>, Vec<f64>)> {
        vec![
            (vec![0.0, 0.0], vec![0.0]),
            (vec![0.0, 1.0], vec![1.0]),
            (vec![1.0, 0.0], vec![1.0]),
            (vec![1.0, 1.0], vec![0.0]),
        ]
    }

    #[test]
    fn the_analytic_gradient_matches_a_central_difference() {
        // The test that decides whether backpropagation is right.
        // Smooth activations only here: a rectifier has a kink, and a
        // pre-activation that lands within h of zero makes the central
        // difference straddle it and disagree for a good reason.
        let mut rng = Rng::new(0x2c4a_71b9);
        for (hidden, output, loss) in [
            (Act::Tanh, Act::Identity, Loss::Mse),
            (Act::Sigmoid, Act::Identity, Loss::Mse),
            (Act::Tanh, Act::Sigmoid, Loss::Mse),
            (Act::Tanh, Act::Softmax, Loss::CrossEntropy),
            (Act::Sigmoid, Act::Softmax, Loss::CrossEntropy),
        ] {
            let net = Mlp::new(&[3, 5, 4, 3], hidden, output, &mut rng).unwrap();
            for _ in 0..5 {
                let x: Vec<f64> = (0..3).map(|_| 2.0 * rng.next_f64() - 1.0).collect();
                let y = if loss == Loss::CrossEntropy {
                    let mut t = vec![0.0; 3];
                    t[(rng.next_u64() % 3) as usize] = 1.0;
                    t
                } else {
                    (0..3).map(|_| rng.next_gaussian()).collect()
                };
                let relative = net.numerical_grad_check(&x, &y, loss).unwrap();
                assert!(
                    relative < 1e-8,
                    "{hidden:?}/{output:?}/{loss:?} disagreed by {relative}"
                );
            }
        }
    }

    #[test]
    fn a_rectifier_gradient_is_right_away_from_its_kink() {
        // A rectifier is not differentiable at zero, so a central
        // difference straddling the kink measures a slope the derivative
        // does not have. What is asserted is that this is the only cause
        // -- every disagreement has a pre-activation within a few
        // difference steps of zero -- rather than a pass rate, which
        // depends on the architecture and says nothing.
        let mut rng = Rng::new(0x77d0_1e42);
        let net = Mlp::new(&[4, 6, 2], Act::Relu, Act::Identity, &mut rng).unwrap();
        let step = 1e-5;
        for _ in 0..40 {
            let x: Vec<f64> = (0..4).map(|_| 2.0 * rng.next_f64() - 1.0).collect();
            let y: Vec<f64> = (0..2).map(|_| rng.next_gaussian()).collect();
            let relative = net.numerical_grad_check(&x, &y, Loss::Mse).unwrap();
            if relative < 1e-8 {
                continue;
            }
            let z = net.preactivations(&x).unwrap();
            let closest =
                z[0].iter().map(|v| v.abs()).fold(f64::INFINITY, f64::min);
            assert!(
                closest < 1000.0 * step,
                "a disagreement of {relative} with the nearest kink {closest} away"
            );
        }
    }

    #[test]
    fn softmax_is_a_distribution_and_ignores_a_shift() {
        let z = vec![1.0, -2.0, 0.5, 3.0];
        let p = Act::Softmax.apply(&z);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-15);
        assert!(p.iter().all(|&v| v > 0.0));
        // Adding a constant to every logit changes nothing, which is
        // what makes subtracting the maximum safe.
        let shifted: Vec<f64> = z.iter().map(|v| v + 137.0).collect();
        for (a, b) in p.iter().zip(Act::Softmax.apply(&shifted).iter()) {
            assert!((a - b).abs() < 1e-15);
        }
        // And it does not overflow where a naive version would.
        let huge = Act::Softmax.apply(&[800.0, 799.0, -800.0]);
        assert!(huge.iter().all(|v| v.is_finite()));
        assert!((huge.iter().sum::<f64>() - 1.0).abs() < 1e-15);
        // The largest logit takes the largest share, in order.
        let ordered = Act::Softmax.apply(&[0.0, 1.0, 2.0]);
        assert!(ordered[0] < ordered[1] && ordered[1] < ordered[2]);
    }

    #[test]
    fn the_fused_output_gradient_is_the_prediction_minus_the_target() {
        // With softmax and cross-entropy the Jacobian of the activation
        // cancels against the gradient of the loss exactly, leaving
        // p - y at the output pre-activation. Checked through the bias
        // gradient of the last layer, which *is* that quantity.
        let mut rng = Rng::new(0x4b21_9de0);
        let net = Mlp::new(&[3, 3], Act::Tanh, Act::Softmax, &mut rng).unwrap();
        let x = vec![0.4, -1.1, 0.2];
        let mut y = vec![0.0; 3];
        y[1] = 1.0;
        let p = net.forward(&x).unwrap();
        let g = net.backward(&x, &y, Loss::CrossEntropy).unwrap();
        let bias = &g.layers[0].1;
        for k in 0..3 {
            assert!((bias[k] - (p[k] - y[k])).abs() < 1e-14, "component {k}");
        }
    }

    #[test]
    fn xor_needs_a_hidden_layer() {
        // XOR is the standard demonstration that a linear model is not
        // merely bad at some problems but incapable of them: no line
        // separates the two classes, so the best a single layer can do
        // is predict the mean and take a loss of 0.125. One hidden
        // layer of two units is already enough to solve it.
        let data = xor();
        let mut rng = Rng::new(0x0e57_2b4c);
        let mut flat = Mlp::new(&[2, 1], Act::Tanh, Act::Identity, &mut rng).unwrap();
        let history = flat.train_adam(&data, Loss::Mse, 400, 0.05, 4, &mut rng).unwrap();
        let best = history.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(best > 0.11, "a linear model reached {best} on XOR");
        // A hidden layer, and it is solved.
        let mut deep = Mlp::new(&[2, 4, 1], Act::Tanh, Act::Identity, &mut rng).unwrap();
        let history = deep.train_adam(&data, Loss::Mse, 1200, 0.05, 4, &mut rng).unwrap();
        let last = *history.last().unwrap();
        assert!(last < 1e-3, "a hidden layer only reached {last}");
        for (x, y) in &data {
            let got = deep.forward(x).unwrap()[0];
            assert!((got - y[0]).abs() < 0.1, "XOR{x:?} gave {got}");
        }
    }

    #[test]
    fn gradient_descent_finds_the_least_squares_answer() {
        // A convex problem with a closed form: descent has somewhere to
        // be checked against, and agreeing with it is worth more than
        // any training curve.
        let mut rng = Rng::new(0x39ba_c105);
        let (rows, cols) = (40, 4);
        let mut x = Matrix::zeros(rows, cols);
        for i in 0..rows {
            x.set(i, 0, 1.0);
            for j in 1..cols {
                x.set(i, j, rng.next_gaussian());
            }
        }
        let truth = [0.7, -1.3, 2.0, 0.4];
        let y: Vec<f64> = (0..rows)
            .map(|i| {
                (0..cols).map(|j| x.get(i, j) * truth[j]).sum::<f64>() + 0.05 * rng.next_gaussian()
            })
            .collect();
        let relative = linear_regression_gd_check(&x, &y, 4000).unwrap();
        assert!(relative < 1e-6, "descent stopped {relative} away from the exact answer");
        assert!(linear_regression_gd_check(&x, &y[..3], 10).is_err());
        // Underdetermined: least squares has no unique answer to check
        // against, and says so rather than returning one of the many.
        let wide = Matrix::zeros(2, 5);
        assert!(linear_regression_gd_check(&wide, &[1.0, 2.0], 10).is_err());
    }

    #[test]
    fn convolution_does_what_its_kernel_says() {
        // A single one in the middle of a kernel is the identity; a
        // uniform kernel is a mean. Both are exact.
        let w = 5;
        let h = 4;
        let input: Vec<f64> = (0..w * h).map(|k| k as f64).collect();
        let identity = (vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], 3, 3);
        let (planes, ow, oh) = conv2d_forward(&input, w, h, &[identity], 1, 1).unwrap();
        assert_eq!((ow, oh), (w, h), "unit stride with one of padding preserves the size");
        for k in 0..w * h {
            assert!((planes[0][k] - input[k]).abs() < 1e-15, "cell {k}");
        }
        // A box filter over a constant image gives that constant back.
        let flat = vec![3.0; w * h];
        let box_filter = (vec![1.0 / 9.0; 9], 3, 3);
        let (blurred, _, _) = conv2d_forward(&flat, w, h, std::slice::from_ref(&box_filter), 1, 0).unwrap();
        for v in &blurred[0] {
            assert!((v - 3.0).abs() < 1e-14, "the box filter changed a constant");
        }
        // The output size follows the standard formula.
        let (_, ow, oh) = conv2d_forward(&input, w, h, &[box_filter], 2, 2).unwrap();
        assert_eq!(ow, (w + 4 - 3) / 2 + 1);
        assert_eq!(oh, (h + 4 - 3) / 2 + 1);
    }

    #[test]
    fn convolution_is_linear_and_commutes_with_a_shift() {
        // Both properties define what a convolution is. Shift
        // equivariance holds away from the edges, where the zero
        // padding is not shift invariant and cannot be.
        let mut rng = Rng::new(0x6b39_ff02);
        let (w, h) = (9, 8);
        let a: Vec<f64> = (0..w * h).map(|_| rng.next_gaussian()).collect();
        let b: Vec<f64> = (0..w * h).map(|_| rng.next_gaussian()).collect();
        let kernel = ((0..9).map(|_| rng.next_gaussian()).collect::<Vec<f64>>(), 3, 3);
        let run = |img: &[f64]| conv2d_forward(img, w, h, std::slice::from_ref(&kernel), 1, 0).unwrap().0;
        let (ra, rb) = (run(&a), run(&b));
        let sum: Vec<f64> = a.iter().zip(&b).map(|(x, y)| 2.0 * x - 3.0 * y).collect();
        let rs = run(&sum);
        for k in 0..rs[0].len() {
            let want = 2.0 * ra[0][k] - 3.0 * rb[0][k];
            assert!((rs[0][k] - want).abs() < 1e-12, "linearity at {k}");
        }
        // Shift the image one cell right; the interior of the output
        // shifts with it.
        let mut shifted = vec![0.0; w * h];
        for y in 0..h {
            for x in 1..w {
                shifted[y * w + x] = a[y * w + x - 1];
            }
        }
        let out_w = w - 2;
        let base = run(&a);
        let moved = run(&shifted);
        for y in 0..h - 2 {
            for x in 1..out_w {
                let want = base[0][y * out_w + x - 1];
                assert!(
                    (moved[0][y * out_w + x] - want).abs() < 1e-12,
                    "shift equivariance at ({x}, {y})"
                );
            }
        }
    }

    #[test]
    fn both_optimisers_reduce_the_loss_they_are_given() {
        let data = xor();
        let mut rng = Rng::new(0x1d80_44ae);
        for adam in [false, true] {
            let mut net = Mlp::new(&[2, 6, 1], Act::Tanh, Act::Identity, &mut rng).unwrap();
            let before = net.loss(&data, Loss::Mse).unwrap();
            let history = if adam {
                net.train_adam(&data, Loss::Mse, 300, 0.05, 2, &mut rng).unwrap()
            } else {
                net.train_sgd(&data, Loss::Mse, 300, 0.5, 2, &mut rng).unwrap()
            };
            assert_eq!(history.len(), 300);
            let after = *history.last().unwrap();
            assert!(after < before * 0.2, "adam={adam}: {before} only fell to {after}");
        }
    }

    #[test]
    fn classification_learns_a_separable_problem() {
        // Three well-separated clusters, softmax and cross-entropy.
        let mut rng = Rng::new(0x5fa2_10c7);
        let centres = [[2.0, 0.0], [-2.0, 1.5], [0.0, -2.5]];
        let mut data = Vec::new();
        for (label, c) in centres.iter().enumerate() {
            for _ in 0..40 {
                let x = vec![c[0] + 0.3 * rng.next_gaussian(), c[1] + 0.3 * rng.next_gaussian()];
                let mut y = vec![0.0; 3];
                y[label] = 1.0;
                data.push((x, y));
            }
        }
        let mut net = Mlp::new(&[2, 8, 3], Act::Tanh, Act::Softmax, &mut rng).unwrap();
        net.train_adam(&data, Loss::CrossEntropy, 120, 0.05, 16, &mut rng).unwrap();
        let correct = data
            .iter()
            .filter(|(x, y)| {
                let want = y.iter().position(|&v| v > 0.5).unwrap();
                net.predict(x).unwrap() == want
            })
            .count();
        assert!(correct >= data.len() - 2, "only {correct} of {} correct", data.len());
    }

    #[test]
    fn the_network_refuses_impossible_arguments() {
        let mut rng = Rng::new(1);
        assert!(Mlp::new(&[3], Act::Tanh, Act::Identity, &mut rng).is_err());
        assert!(Mlp::new(&[3, 0, 2], Act::Tanh, Act::Identity, &mut rng).is_err());
        let net = Mlp::new(&[2, 3, 2], Act::Tanh, Act::Identity, &mut rng).unwrap();
        assert_eq!(net.input_size(), 2);
        assert_eq!(net.output_size(), 2);
        assert_eq!(net.parameter_count(), 2 * 3 + 3 + 3 * 2 + 2);
        assert!(net.forward(&[1.0]).is_err());
        assert!(net.example_loss(&[1.0, 2.0], &[1.0], Loss::Mse).is_err());
        // Cross-entropy without a softmax output is refused rather than
        // silently computing something else.
        assert!(net.example_loss(&[1.0, 2.0], &[1.0, 0.0], Loss::CrossEntropy).is_err());
        assert!(net.backward(&[1.0, 2.0], &[1.0, 0.0], Loss::CrossEntropy).is_err());
        // And a softmax output with squared error, which would need the
        // full Jacobian, is refused too.
        let soft = Mlp::new(&[2, 2], Act::Tanh, Act::Softmax, &mut rng).unwrap();
        assert!(soft.backward(&[1.0, 2.0], &[1.0, 0.0], Loss::Mse).is_err());
        assert!(net.loss(&[], Loss::Mse).is_err());
        let mut m = net.clone();
        let data = vec![(vec![0.0, 0.0], vec![0.0, 0.0])];
        assert!(m.train_sgd(&[], Loss::Mse, 1, 0.1, 1, &mut rng).is_err());
        assert!(m.train_sgd(&data, Loss::Mse, 1, 0.1, 0, &mut rng).is_err());
        assert!(m.train_sgd(&data, Loss::Mse, 1, -1.0, 1, &mut rng).is_err());
        assert!(m.train_adam(&[], Loss::Mse, 1, 0.1, 1, &mut rng).is_err());
        assert!(m.train_adam(&data, Loss::Mse, 1, 0.1, 0, &mut rng).is_err());
        assert!(m.train_adam(&data, Loss::Mse, 1, f64::NAN, 1, &mut rng).is_err());
        // Convolution arguments.
        let img = vec![0.0; 12];
        let k = (vec![1.0; 4], 2, 2);
        assert!(conv2d_forward(&img, 3, 3, std::slice::from_ref(&k), 1, 0).is_err());
        assert!(conv2d_forward(&img, 4, 3, std::slice::from_ref(&k), 0, 0).is_err());
        assert!(conv2d_forward(&img, 4, 3, &[], 1, 0).is_err());
        assert!(conv2d_forward(&img, 4, 3, &[(vec![1.0; 3], 2, 2)], 1, 0).is_err());
        assert!(conv2d_forward(&img, 4, 3, &[k.clone(), (vec![1.0; 9], 3, 3)], 1, 0).is_err());
        assert!(conv2d_forward(&img, 4, 3, &[(vec![1.0; 100], 10, 10)], 1, 0).is_err());
    }
}
