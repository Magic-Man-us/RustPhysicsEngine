//! WAV (RIFF) reading and writing: PCM 8/16/24/32-bit, IEEE float
//! 32/64-bit, and WAVE_FORMAT_EXTENSIBLE containers. Samples are
//! normalized to −1..1 per channel.

use crate::error::SolveError;

/// Decoded audio: sample rate, channel count, and per-channel samples
/// in −1..1.
#[derive(Debug, Clone, PartialEq)]
pub struct WavData {
    pub fs: u32,
    pub channels: u16,
    pub samples: Vec<Vec<f64>>,
}

const FORMAT_PCM: u16 = 1;
const FORMAT_IEEE_FLOAT: u16 = 3;
const FORMAT_EXTENSIBLE: u16 = 0xFFFE;

fn rd_u16(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([b[off], b[off + 1]])
}

fn rd_u32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

/// Parse the fmt and data chunks: (format, channels, fs, bits,
/// data_offset, data_len).
fn parse_chunks(bytes: &[u8]) -> Result<(u16, u16, u32, u16, usize, usize), SolveError> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(SolveError::InvalidArgument("not a RIFF/WAVE file"));
    }
    let mut off = 12usize;
    let mut fmt: Option<(u16, u16, u32, u16)> = None;
    let mut data: Option<(usize, usize)> = None;
    while off + 8 <= bytes.len() {
        let id = &bytes[off..off + 4];
        let size = rd_u32(bytes, off + 4) as usize;
        let body = off + 8;
        if body + size > bytes.len() {
            return Err(SolveError::InvalidArgument("truncated WAV chunk"));
        }
        match id {
            b"fmt " => {
                if size < 16 {
                    return Err(SolveError::InvalidArgument("fmt chunk too small"));
                }
                let mut format = rd_u16(bytes, body);
                let channels = rd_u16(bytes, body + 2);
                let fs = rd_u32(bytes, body + 4);
                let bits = rd_u16(bytes, body + 14);
                if format == FORMAT_EXTENSIBLE {
                    if size < 40 {
                        return Err(SolveError::InvalidArgument("extensible fmt chunk too small"));
                    }
                    // SubFormat GUID: first two bytes carry the format tag.
                    format = rd_u16(bytes, body + 24);
                }
                fmt = Some((format, channels, fs, bits));
            }
            b"data" => {
                data = Some((body, size));
            }
            _ => {}
        }
        off = body + size + (size % 2); // chunks are word-aligned
    }
    match (fmt, data) {
        (Some((format, ch, fs, bits)), Some((doff, dlen))) => Ok((format, ch, fs, bits, doff, dlen)),
        _ => Err(SolveError::InvalidArgument("missing fmt or data chunk")),
    }
}

/// Decode a WAV byte stream.
///
/// # Errors
/// `InvalidArgument` for malformed containers or unsupported encodings.
pub fn wav_read(bytes: &[u8]) -> Result<WavData, SolveError> {
    let (format, channels, fs, bits, doff, dlen) = parse_chunks(bytes)?;
    if channels == 0 {
        return Err(SolveError::InvalidArgument("zero channels"));
    }
    let bytes_per_sample = (bits as usize).div_ceil(8);
    let frame = bytes_per_sample * channels as usize;
    if frame == 0 {
        return Err(SolveError::InvalidArgument("zero-size frame"));
    }
    let n_frames = dlen / frame;
    let data = &bytes[doff..doff + dlen];
    let mut samples = vec![Vec::with_capacity(n_frames); channels as usize];
    for f in 0..n_frames {
        for (c, chan) in samples.iter_mut().enumerate() {
            let s = f * frame + c * bytes_per_sample;
            let v = match (format, bits) {
                (FORMAT_PCM, 8) => (data[s] as f64 - 128.0) / 127.0,
                (FORMAT_PCM, 16) => {
                    i16::from_le_bytes([data[s], data[s + 1]]) as f64 / 32767.0
                }
                (FORMAT_PCM, 24) => {
                    let raw = ((data[s] as i32) << 8
                        | (data[s + 1] as i32) << 16
                        | (data[s + 2] as i32) << 24)
                        >> 8;
                    raw as f64 / 8_388_607.0
                }
                (FORMAT_PCM, 32) => {
                    i32::from_le_bytes([data[s], data[s + 1], data[s + 2], data[s + 3]]) as f64
                        / 2_147_483_647.0
                }
                (FORMAT_IEEE_FLOAT, 32) => {
                    f32::from_le_bytes([data[s], data[s + 1], data[s + 2], data[s + 3]]) as f64
                }
                (FORMAT_IEEE_FLOAT, 64) => f64::from_le_bytes([
                    data[s],
                    data[s + 1],
                    data[s + 2],
                    data[s + 3],
                    data[s + 4],
                    data[s + 5],
                    data[s + 6],
                    data[s + 7],
                ]),
                _ => {
                    return Err(SolveError::InvalidArgument("unsupported WAV encoding"));
                }
            };
            chan.push(v);
        }
    }
    Ok(WavData { fs, channels, samples })
}

/// Encode to WAV bytes: PCM at 8/16/24/32 bits, or IEEE float at 32/64.
///
/// # Panics
/// Panics for unsupported bit depths or mismatched channel lengths.
#[must_use]
pub fn wav_write(data: &WavData, bits: u16, float: bool) -> Vec<u8> {
    assert!(
        (float && (bits == 32 || bits == 64)) || (!float && [8, 16, 24, 32].contains(&bits)),
        "unsupported bit depth"
    );
    let channels = data.channels as usize;
    assert_eq!(data.samples.len(), channels, "channel count mismatch");
    let n_frames = data.samples.first().map_or(0, Vec::len);
    for ch in &data.samples {
        assert_eq!(ch.len(), n_frames, "channels must have equal length");
    }
    let bps = bits as usize / 8;
    let frame = bps * channels;
    let data_len = n_frames * frame;
    let format = if float { FORMAT_IEEE_FLOAT } else { FORMAT_PCM };
    let mut out = Vec::with_capacity(44 + data_len);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&((36 + data_len) as u32).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&format.to_le_bytes());
    out.extend_from_slice(&data.channels.to_le_bytes());
    out.extend_from_slice(&data.fs.to_le_bytes());
    out.extend_from_slice(&((data.fs as usize * frame) as u32).to_le_bytes());
    out.extend_from_slice(&(frame as u16).to_le_bytes());
    out.extend_from_slice(&bits.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data_len as u32).to_le_bytes());
    for f in 0..n_frames {
        for ch in &data.samples {
            let v = ch[f].clamp(-1.0, 1.0);
            match (float, bits) {
                (false, 8) => out.push(((v * 127.0).round() + 128.0) as u8),
                (false, 16) => {
                    let q = (v * 32767.0).round() as i16;
                    out.extend_from_slice(&q.to_le_bytes());
                }
                (false, 24) => {
                    let q = (v * 8_388_607.0).round() as i32;
                    out.extend_from_slice(&q.to_le_bytes()[..3]);
                }
                (false, 32) => {
                    let q = (v * 2_147_483_647.0).round() as i32;
                    out.extend_from_slice(&q.to_le_bytes());
                }
                (true, 32) => out.extend_from_slice(&(v as f32).to_le_bytes()),
                (true, 64) => out.extend_from_slice(&v.to_le_bytes()),
                _ => unreachable!(),
            }
        }
    }
    out
}

/// Read a WAV file from disk.
///
/// # Errors
/// I/O errors from the filesystem; decode failures become
/// `InvalidData`.
pub fn wav_read_file(path: &str) -> std::io::Result<WavData> {
    let bytes = std::fs::read(path)?;
    wav_read(&bytes).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

/// Write a WAV file to disk.
///
/// # Errors
/// I/O errors from the filesystem.
pub fn wav_write_file(path: &str, data: &WavData, bits: u16, float: bool) -> std::io::Result<()> {
    std::fs::write(path, wav_write(data, bits, float))
}

/// Header summary (fs, channels, bits, frames) without decoding samples.
///
/// # Errors
/// `InvalidArgument` for malformed containers.
pub fn wav_info(bytes: &[u8]) -> Result<(u32, u16, u16, usize), SolveError> {
    let (_, channels, fs, bits, _, dlen) = parse_chunks(bytes)?;
    let frame = (bits as usize).div_ceil(8) * channels as usize;
    Ok((fs, channels, bits, if frame > 0 { dlen / frame } else { 0 }))
}

/// Average all channels down to one.
#[must_use]
pub fn to_mono(d: &WavData) -> Vec<f64> {
    let n = d.samples.first().map_or(0, Vec::len);
    (0..n)
        .map(|i| d.samples.iter().map(|ch| ch[i]).sum::<f64>() / d.samples.len().max(1) as f64)
        .collect()
}

/// Interleave channels (frame-major).
#[must_use]
pub fn to_interleaved(d: &WavData) -> Vec<f64> {
    let n = d.samples.first().map_or(0, Vec::len);
    let mut out = Vec::with_capacity(n * d.samples.len());
    for i in 0..n {
        for ch in &d.samples {
            out.push(ch[i]);
        }
    }
    out
}

/// Split an interleaved stream into per-channel vectors.
///
/// # Panics
/// Panics if `channels == 0`.
#[must_use]
pub fn from_interleaved(x: &[f64], channels: u16) -> Vec<Vec<f64>> {
    assert!(channels > 0, "need at least one channel");
    let ch = channels as usize;
    let frames = x.len() / ch;
    (0..ch).map(|c| (0..frames).map(|f| x[f * ch + c]).collect()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_signal(n: usize) -> Vec<f64> {
        (0..n).map(|i| (i as f64 * 0.13).sin() * 0.8).collect()
    }

    #[test]
    fn test_wav_16bit_roundtrip_exact() {
        // Quantize first so the round trip is bit-exact.
        let raw = test_signal(500);
        let quantized: Vec<f64> = raw
            .iter()
            .map(|&v| (v * 32767.0).round() / 32767.0)
            .collect();
        let data = WavData { fs: 44100, channels: 1, samples: vec![quantized.clone()] };
        let bytes = wav_write(&data, 16, false);
        let back = wav_read(&bytes).unwrap();
        assert_eq!(back.fs, 44100);
        assert_eq!(back.channels, 1);
        for (a, b) in quantized.iter().zip(&back.samples[0]) {
            assert_eq!(a, b, "16-bit round trip must be exact");
        }
    }

    #[test]
    fn test_wav_float_roundtrip_bitexact() {
        let sig = test_signal(300);
        let data = WavData { fs: 48000, channels: 2, samples: vec![sig.clone(), sig.iter().map(|v| -v).collect()] };
        let bytes = wav_write(&data, 64, true);
        let back = wav_read(&bytes).unwrap();
        assert_eq!(back.samples[0], sig);
        // 32-bit float within f32 precision.
        let bytes32 = wav_write(&data, 32, true);
        let back32 = wav_read(&bytes32).unwrap();
        for (a, b) in sig.iter().zip(&back32.samples[0]) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_wav_24_and_8_bit() {
        let sig = test_signal(200);
        let data = WavData { fs: 32000, channels: 1, samples: vec![sig.clone()] };
        let back24 = wav_read(&wav_write(&data, 24, false)).unwrap();
        for (a, b) in sig.iter().zip(&back24.samples[0]) {
            assert!((a - b).abs() < 2.0 / 8_388_607.0);
        }
        let back8 = wav_read(&wav_write(&data, 8, false)).unwrap();
        for (a, b) in sig.iter().zip(&back8.samples[0]) {
            assert!((a - b).abs() < 2.0 / 127.0);
        }
        let back32 = wav_read(&wav_write(&data, 32, false)).unwrap();
        for (a, b) in sig.iter().zip(&back32.samples[0]) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_wav_info_and_extensible() {
        let sig = test_signal(123);
        let data = WavData { fs: 22050, channels: 2, samples: vec![sig.clone(), sig] };
        let mut bytes = wav_write(&data, 16, false);
        let (fs, ch, bits, frames) = wav_info(&bytes).unwrap();
        assert_eq!((fs, ch, bits, frames), (22050, 2, 16, 123));
        // Fake an extensible header: patch format tag to 0xFFFE with a
        // 40-byte fmt chunk carrying PCM in the SubFormat.
        let mut ext = Vec::new();
        ext.extend_from_slice(&bytes[..16]); // through "fmt " id
        ext.extend_from_slice(&40u32.to_le_bytes());
        ext.extend_from_slice(&FORMAT_EXTENSIBLE.to_le_bytes());
        ext.extend_from_slice(&bytes[22..36]); // rest of the 16-byte fmt body
        ext.extend_from_slice(&22u16.to_le_bytes()); // cbSize
        ext.extend_from_slice(&16u16.to_le_bytes()); // valid bits
        ext.extend_from_slice(&3u32.to_le_bytes()); // channel mask
        ext.extend_from_slice(&FORMAT_PCM.to_le_bytes()); // SubFormat tag
        ext.extend_from_slice(&[0u8; 14]); // rest of GUID
        ext.extend_from_slice(&bytes[36..]); // data chunk onward
        // Fix the RIFF size.
        let riff_size = (ext.len() - 8) as u32;
        ext[4..8].copy_from_slice(&riff_size.to_le_bytes());
        let parsed = wav_read(&ext).unwrap();
        assert_eq!(parsed.channels, 2);
        assert_eq!(parsed.samples[0].len(), 123);
        // Garbage rejects cleanly.
        bytes[0] = b'X';
        assert!(wav_read(&bytes).is_err());
    }

    #[test]
    fn test_file_roundtrip_and_channel_utils() {
        let sig = test_signal(64);
        let data = WavData {
            fs: 8000,
            channels: 2,
            samples: vec![sig.clone(), vec![0.25; 64]],
        };
        let dir = std::env::temp_dir().join("rpe_wav_test.wav");
        let path = dir.to_str().unwrap();
        wav_write_file(path, &data, 16, false).unwrap();
        let back = wav_read_file(path).unwrap();
        assert_eq!(back.channels, 2);
        std::fs::remove_file(path).ok();

        let mono = to_mono(&data);
        assert!((mono[0] - (sig[0] + 0.25) / 2.0).abs() < 1e-12);
        let inter = to_interleaved(&data);
        assert_eq!(inter.len(), 128);
        assert_eq!(inter[1], 0.25);
        let split = from_interleaved(&inter, 2);
        assert_eq!(split[0], data.samples[0]);
        assert_eq!(split[1], data.samples[1]);
    }
}
