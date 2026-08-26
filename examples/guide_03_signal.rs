//! Guide, chapter 3: finding a tone buried in noise, and filtering it out.
//!
//! Run with `cargo run --example guide_03_signal`. CI runs it too, so the
//! guide chapter built from this file cannot go stale.

use rust_physics_engine::dsp::fir::{fir_apply, fir_lowpass};
use rust_physics_engine::dsp::windows::WindowKind;
use rust_physics_engine::monte_carlo::Rng;
use rust_physics_engine::transforms::fft::rfft;
use rust_physics_engine::transforms::spectral::welch;

fn main() {
    let fs = 8_000.0; // sample rate, Hz
    let n = 4_096; // a power of two, so the FFT is radix-2

    // Two tones and a lot of noise. 440 Hz is the one we want; 2,600 Hz is
    // interference we intend to filter away.
    let mut rng = Rng::new(0x5EED_1234);
    let signal: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 / fs;
            let wanted = (2.0 * std::f64::consts::PI * 440.0 * t).sin();
            let interference = 0.8 * (2.0 * std::f64::consts::PI * 2_600.0 * t).sin();
            let noise = 1.5 * (rng.next_f64() - 0.5);
            wanted + interference + noise
        })
        .collect();

    // rfft returns only the non-negative frequencies, which is all a real
    // signal has: bin k is at k·fs/n Hz.
    let spectrum = rfft(&signal);
    let peak = |from: f64, to: f64| -> (f64, f64) {
        let lo = (from * n as f64 / fs) as usize;
        let hi = (to * n as f64 / fs) as usize;
        let mut best = (0.0, 0.0);
        for (k, c) in spectrum.iter().enumerate().take(hi + 1).skip(lo) {
            let mag = (c.re * c.re + c.im * c.im).sqrt();
            if mag > best.1 {
                best = (k as f64 * fs / n as f64, mag);
            }
        }
        best
    };

    let (f_wanted, m_wanted) = peak(300.0, 600.0);
    let (f_interf, m_interf) = peak(2_400.0, 2_800.0);
    println!("before filtering");
    println!("  tone found at      {f_wanted:.0} Hz  (magnitude {m_wanted:.0})");
    println!("  interference at    {f_interf:.0} Hz  (magnitude {m_interf:.0})");

    // The FFT recovers both tones despite noise at more than the amplitude
    // of the signal, because the noise is spread across every bin while a
    // sinusoid concentrates into one.
    assert!((f_wanted - 440.0).abs() < 5.0);
    assert!((f_interf - 2_600.0).abs() < 5.0);

    // A windowed-sinc low-pass. The cutoff is in cycles per sample, so
    // 1,000 Hz at fs = 8 kHz is 0.125. More taps means a sharper edge.
    let taps = fir_lowpass(101, 1_000.0 / fs, WindowKind::Hamming);
    let filtered = fir_apply(&taps, &signal);

    let spectrum = rfft(&filtered[..n]);
    let peak2 = |from: f64, to: f64| -> f64 {
        let lo = (from * n as f64 / fs) as usize;
        let hi = (to * n as f64 / fs) as usize;
        spectrum[lo..=hi]
            .iter()
            .map(|c| (c.re * c.re + c.im * c.im).sqrt())
            .fold(0.0, f64::max)
    };
    let after_wanted = peak2(300.0, 600.0);
    let after_interf = peak2(2_400.0, 2_800.0);
    println!();
    println!("after a 1 kHz low-pass");
    println!("  440 Hz kept        magnitude {after_wanted:.0}");
    println!("  2.6 kHz rejected   magnitude {after_interf:.0}");
    println!(
        "  rejection          {:.0} dB",
        20.0 * (m_interf / after_interf.max(1e-12)).log10()
    );

    // The tone in the passband survives; the one in the stopband does not.
    assert!(after_wanted > 0.5 * m_wanted, "the passband tone was attenuated");
    assert!(after_interf < 0.05 * m_interf, "the stopband tone survived");

    // Welch's method trades frequency resolution for a variance reduction,
    // by averaging periodograms over overlapping segments. It is the right
    // tool when you want the noise floor rather than the exact peak.
    let (freqs, psd) = welch(&signal, fs, 512, 256, WindowKind::Hann);
    let loudest = psd
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| freqs[i])
        .unwrap();
    println!();
    println!("Welch PSD over {} segments peaks at {loudest:.0} Hz", n / 256 - 1);
    assert!((loudest - 440.0).abs() < 20.0);
}
