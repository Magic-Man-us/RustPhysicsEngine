//! Properties of the sequence alignment module.
//!
//! Alignment is unusually well supplied with exact cross-checks. A dynamic
//! program reports a score and an alignment, and the two must agree -- which
//! is checkable directly by rescoring, and is the failure a
//! score-only comparison cannot see. Beyond that, four of the algorithms
//! here compute the same optimum by different means: Needleman-Wunsch's
//! quadratic table, Hirschberg's linear-space recursion, Gotoh's three
//! tables with a free gap opening, and a band wide enough to contain the
//! whole table. Any disagreement between them is a defect in one of them.

use rust_physics_engine::biophysics::seq_align::{
    alignment_score, alignment_score_affine, banded_alignment, blosum62, burrows_wheeler_search,
    consensus, de_bruijn_assembly_lite, gc_content, gotoh_affine, hirschberg, jukes_cantor_distance,
    kimura_2p, kmer_index, minimizers, msa_center_star, needleman_wunsch, p_distance, pam250,
    profile_from_msa, reverse_complement, smith_waterman, transcribe, translate, Scoring,
};
use rust_physics_engine::monte_carlo::Rng;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

fn pick(rng: &mut Rng, n: usize) -> usize {
    ((u128::from(rng.next_u64()) * n as u128) >> 64) as usize
}

/// A random DNA sequence of a given length.
fn dna(rng: &mut Rng, len: usize) -> Vec<u8> {
    (0..len).map(|_| b"ACGT"[pick(rng, 4)]).collect()
}

/// A random DNA sequence of length `0..max`, drawing its own length so the
/// generator is borrowed once rather than twice.
fn dna_upto(rng: &mut Rng, max: usize) -> Vec<u8> {
    let len = pick(rng, max);
    dna(rng, len)
}

/// A random DNA sequence of length `min..min + span`.
fn dna_min(rng: &mut Rng, min: usize, span: usize) -> Vec<u8> {
    let len = min + pick(rng, span);
    dna(rng, len)
}

/// A random scoring scheme with a negative gap penalty.
fn scoring(rng: &mut Rng) -> Scoring {
    Scoring::simple(
        1 + pick(rng, 5) as i64,
        -(1 + pick(rng, 5) as i64),
        -(1 + pick(rng, 6) as i64),
    )
}

// ---------------------------------------------------------------------------
// Alignment
// ---------------------------------------------------------------------------

#[test]
fn prop_every_alignment_achieves_the_score_it_reports() {
    // The check a score-only comparison cannot make: rescoring the returned
    // alignment catches a program that reports a maximum it did not reach.
    let mut rng = Rng::new(0x05EA_0001);
    for _ in 0..300 {
        let s = scoring(&mut rng);
        let a = dna_upto(&mut rng, 30);
        let b = dna_upto(&mut rng, 30);
        let (score, top, bottom) = needleman_wunsch(&a, &b, &s).unwrap();
        assert_eq!(top.len(), bottom.len(), "the aligned strings differ in length");
        assert_eq!(
            alignment_score(&top, &bottom, &s).unwrap(),
            score,
            "the alignment does not score what was reported:\n{top}\n{bottom}"
        );
        // No column of two gaps, and both sequences are spelled out.
        assert!(!top.bytes().zip(bottom.bytes()).any(|(x, y)| x == b'-' && y == b'-'));
        assert_eq!(top.bytes().filter(|c| *c != b'-').collect::<Vec<_>>(), a);
        assert_eq!(bottom.bytes().filter(|c| *c != b'-').collect::<Vec<_>>(), b);
        // Symmetric in its arguments.
        assert_eq!(needleman_wunsch(&b, &a, &s).unwrap().0, score);
        // And bounded above by the self-alignment of either sequence.
        let identity = needleman_wunsch(&a, &a, &s).unwrap().0;
        assert_eq!(identity, s.match_score * a.len() as i64);
        assert!(score <= identity);
    }
}

#[test]
fn prop_four_routes_to_the_global_optimum_agree() {
    // The quadratic table, the linear-space recursion, the affine model with
    // free opening, and a full-width band. Four implementations, one number.
    let mut rng = Rng::new(0x05EA_0002);
    for _ in 0..200 {
        let match_score = 1 + pick(&mut rng, 5) as i64;
        let mismatch = -(1 + pick(&mut rng, 5) as i64);
        let gap = -(1 + pick(&mut rng, 6) as i64);
        let s = Scoring::simple(match_score, mismatch, gap);
        let a = dna_upto(&mut rng, 25);
        let b = dna_upto(&mut rng, 25);
        let table = needleman_wunsch(&a, &b, &s).unwrap().0;

        // Hirschberg: linear space, same optimum. Several alignments can
        // share it, so the score is what must agree, not the strings.
        let (top, bottom) = hirschberg(&a, &b, &s).unwrap();
        assert_eq!(
            alignment_score(&top, &bottom, &s).unwrap(),
            table,
            "Hirschberg found a different optimum"
        );
        assert_eq!(top.bytes().filter(|c| *c != b'-').collect::<Vec<_>>(), a);
        assert_eq!(bottom.bytes().filter(|c| *c != b'-').collect::<Vec<_>>(), b);

        // Gotoh with no opening cost is the linear model.
        let (affine, atop, abottom) =
            gotoh_affine(&a, &b, match_score, mismatch, 0, gap).unwrap();
        assert_eq!(affine, table, "Gotoh with free opening differs from the linear model");
        assert_eq!(
            alignment_score_affine(&atop, &abottom, match_score, mismatch, 0, gap).unwrap(),
            affine
        );

        // A band wide enough to hold the whole table.
        let wide = banded_alignment(&a, &b, a.len().max(b.len()), &s).unwrap();
        assert_eq!(wide, table, "a full-width band differs from the table");
        // And no band can beat the unrestricted optimum.
        for band in a.len().abs_diff(b.len())..=a.len().max(b.len()) {
            assert!(banded_alignment(&a, &b, band, &s).unwrap() <= table);
        }
    }
}

#[test]
fn prop_affine_gaps_never_charge_more_than_linear_ones_for_a_single_run() {
    // A gap of length k costs open + k * extend under the affine model and
    // k * extend under the linear one, so the affine score is at most the
    // linear score at the same extend cost -- with equality exactly when the
    // alignment has no gaps at all.
    let mut rng = Rng::new(0x05EA_0003);
    for _ in 0..150 {
        let match_score = 1 + pick(&mut rng, 5) as i64;
        let mismatch = -(1 + pick(&mut rng, 4) as i64);
        let extend = -(1 + pick(&mut rng, 4) as i64);
        let open = -(pick(&mut rng, 10) as i64);
        let a = dna_upto(&mut rng, 25);
        let b = dna_upto(&mut rng, 25);
        let linear = Scoring::simple(match_score, mismatch, extend);
        let straight = needleman_wunsch(&a, &b, &linear).unwrap().0;
        let (affine, top, bottom) =
            gotoh_affine(&a, &b, match_score, mismatch, open, extend).unwrap();
        assert!(
            affine <= straight,
            "affine {affine} beat linear {straight} at open = {open}"
        );
        assert_eq!(
            alignment_score_affine(&top, &bottom, match_score, mismatch, open, extend).unwrap(),
            affine,
            "the affine alignment does not score what was reported"
        );
        // With no gaps in it, the *same* alignment scores identically under
        // either model -- the gap terms are what differ and there are none.
        // That is not the same as the two optima coinciding: with a costly
        // opening the affine optimum may take mismatches where the linear
        // one buys gaps, so `affine` and `straight` legitimately differ.
        if !top.contains('-') && !bottom.contains('-') {
            assert_eq!(
                alignment_score(&top, &bottom, &linear).unwrap(),
                affine,
                "a gapless alignment scored differently under the two models"
            );
        }
        // A harsher opening cost can only lower the score.
        let harsher = gotoh_affine(&a, &b, match_score, mismatch, open - 5, extend).unwrap().0;
        assert!(harsher <= affine, "a harsher opening cost raised the score");
    }
}

#[test]
fn prop_the_local_score_is_never_negative_and_never_below_the_global_one_on_a_match() {
    let mut rng = Rng::new(0x05EA_0004);
    for _ in 0..200 {
        let s = scoring(&mut rng);
        let a = dna_min(&mut rng, 1, 25);
        let b = dna_min(&mut rng, 1, 25);
        let (local, start_a, start_b, top, bottom) = smith_waterman(&a, &b, &s).unwrap();
        assert!(local >= 0, "a local score went negative: {local}");
        assert_eq!(
            alignment_score(&top, &bottom, &s).unwrap(),
            local,
            "the local alignment does not score what was reported"
        );
        // The reported start positions really are where the alignment sits.
        let consumed_a: Vec<u8> = top.bytes().filter(|c| *c != b'-').collect();
        let consumed_b: Vec<u8> = bottom.bytes().filter(|c| *c != b'-').collect();
        assert_eq!(&a[start_a..start_a + consumed_a.len()], consumed_a.as_slice());
        assert_eq!(&b[start_b..start_b + consumed_b.len()], consumed_b.as_slice());
        // Local can never do worse than global on the same inputs, since
        // the global alignment is one of the local candidates plus flanks.
        let global = needleman_wunsch(&a, &b, &s).unwrap().0;
        assert!(local >= global.max(0), "local {local} fell below global {global}");
        // A sequence against itself is a perfect local match.
        assert_eq!(
            smith_waterman(&a, &a, &s).unwrap().0,
            s.match_score * a.len() as i64
        );
    }
}

// ---------------------------------------------------------------------------
// Sequences
// ---------------------------------------------------------------------------

#[test]
fn prop_the_reverse_complement_is_an_involution() {
    let mut rng = Rng::new(0x05EA_0010);
    for _ in 0..400 {
        let seq = dna_min(&mut rng, 1, 40);
        let once = reverse_complement(&seq);
        assert_eq!(reverse_complement(&once), seq, "not an involution");
        assert_eq!(once.len(), seq.len());
        // The GC fraction is a property of the duplex, not of the strand.
        assert!(close(gc_content(&once).unwrap(), gc_content(&seq).unwrap(), 1e-12));
        // Complementing reverses the order of the A/T and G/C counts.
        let at = seq.iter().filter(|c| matches!(**c, b'A' | b'T')).count();
        let at_back = once.iter().filter(|c| matches!(**c, b'A' | b'T')).count();
        assert_eq!(at, at_back, "the A/T count changed");
        // Transcription is idempotent on an already-transcribed sequence.
        let rna = transcribe(&seq);
        assert_eq!(transcribe(&rna), rna);
        assert!(!rna.contains(&b'T'));
        // Translating from DNA and from its RNA gives the same protein.
        assert_eq!(translate(&seq), translate(&rna));
    }
}

#[test]
fn prop_the_corrected_distances_are_monotone_and_invert_their_own_formulas() {
    let mut rng = Rng::new(0x05EA_0011);
    for _ in 0..500 {
        let p = rng.next_f64() * 0.7499;
        let d = jukes_cantor_distance(p).unwrap();
        assert!(d >= p - 1e-12, "the correction is below the observed proportion");
        assert!(d.is_finite() && d >= 0.0);
        // Inverting the closed form recovers p exactly.
        let recovered = 0.75 * (1.0 - (-4.0 * d / 3.0).exp());
        assert!(close(recovered, p, 1e-9), "inverting gave {recovered} against {p}");
        // Monotone in the observed proportion.
        if p < 0.74 {
            assert!(jukes_cantor_distance(p + 0.005).unwrap() > d);
        }
        // Kimura reduces to Jukes-Cantor when transitions and transversions
        // are in the ratio the uncorrected model implicitly assumes: one
        // transition to two transversions.
        let third = p / 3.0;
        if let Ok(k) = kimura_2p(third, 2.0 * third) {
            assert!(
                close(k, d, 1e-9),
                "at the Jukes-Cantor ratio Kimura gives {k} against {d}"
            );
        }
        // And exceeds it when transitions dominate.
        if let Ok(heavy) = kimura_2p(p * 0.8, p * 0.2) {
            assert!(heavy >= d - 1e-9, "a transition bias lowered the distance");
        }
    }
    // The p-distance is the Hamming count normalised.
    let mut rng = Rng::new(0x05EA_0012);
    for _ in 0..200 {
        let len = 1 + pick(&mut rng, 40);
        let a = dna(&mut rng, len);
        let b = dna(&mut rng, len);
        let differences = a.iter().zip(&b).filter(|(x, y)| x != y).count();
        assert!(close(
            p_distance(&a, &b).unwrap(),
            differences as f64 / len as f64,
            1e-12
        ));
        assert!(close(p_distance(&a, &a).unwrap(), 0.0, 1e-15));
    }
}

// ---------------------------------------------------------------------------
// Indexing
// ---------------------------------------------------------------------------

#[test]
fn prop_the_index_and_the_search_agree_with_a_naive_scan() {
    // The naive scan is the only thing here that is obviously right, so both
    // structures are checked against it rather than against each other.
    let mut rng = Rng::new(0x05EA_0020);
    for _ in 0..60 {
        let text = dna_min(&mut rng, 20, 60);
        let k = 1 + pick(&mut rng, 6.min(text.len()));
        let index = kmer_index(&text, k).unwrap();
        let total: usize = index.iter().map(|(_, p)| p.len()).sum();
        assert_eq!(total, text.len() - k + 1, "the index lost a position");
        for (kmer, positions) in &index {
            let naive: Vec<usize> =
                (0..=text.len() - k).filter(|i| &text[*i..*i + k] == kmer.as_slice()).collect();
            assert_eq!(*positions, naive, "the index disagrees with a scan");
            assert_eq!(
                burrows_wheeler_search(&text, kmer).unwrap(),
                naive,
                "the BWT search disagrees with a scan"
            );
        }
        for pair in index.windows(2) {
            assert!(pair[0].0 < pair[1].0, "the index is not sorted");
        }
        // A random pattern, present or not.
        let pattern = dna_min(&mut rng, 1, 8);
        let naive: Vec<usize> = if pattern.len() <= text.len() {
            (0..=text.len() - pattern.len())
                .filter(|i| &text[*i..*i + pattern.len()] == pattern.as_slice())
                .collect()
        } else {
            Vec::new()
        };
        assert_eq!(burrows_wheeler_search(&text, &pattern).unwrap(), naive);
    }
}

#[test]
fn prop_minimizers_are_the_smallest_in_some_window_and_reduce_the_k_mer_count() {
    let mut rng = Rng::new(0x05EA_0021);
    for _ in 0..60 {
        let k = 3 + pick(&mut rng, 5);
        let w = 2 + pick(&mut rng, 8);
        let text = dna_min(&mut rng, k + w, 80);
        let selected = minimizers(&text, k, w).unwrap();
        assert!(!selected.is_empty(), "no minimizer was selected");
        let kmers = text.len() - k + 1;
        assert!(selected.len() <= kmers, "more minimizers than k-mers");
        // Each reported k-mer must be minimal over at least one window that
        // contains it. Recomputed here from the text directly, so the check
        // does not go through the function it is testing.
        let hashes: Vec<u64> = (0..kmers)
            .map(|i| {
                let mut hash = 0xcbf2_9ce4_8422_2325u64;
                for byte in &text[i..i + k] {
                    hash ^= u64::from(*byte);
                    hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
                }
                hash
            })
            .collect();
        for (position, hash) in &selected {
            assert!(position + k <= text.len(), "a minimizer runs off the end");
            assert_eq!(*hash, hashes[*position], "the reported hash is not the k-mer's");
            let lo = position.saturating_sub(w - 1);
            let hi = (*position).min(kmers - w);
            assert!(
                (lo..=hi).any(|start| (start..start + w).all(|j| hashes[*position] <= hashes[j])),
                "the k-mer at {position} is not minimal over any window containing it"
            );
        }
        // Positions are strictly increasing, so nothing is reported twice.
        for pair in selected.windows(2) {
            assert!(pair[1].0 >= pair[0].0, "the minimizers are out of order");
        }
        // Two sequences sharing a substring of at least w + k - 1 share a
        // minimizer -- the guarantee that makes the sampling consistent.
        let shared = dna_min(&mut rng, k + w - 1, 20);
        let mut a = dna(&mut rng, 10);
        a.extend_from_slice(&shared);
        a.extend_from_slice(&dna(&mut rng, 10));
        let mut b = dna(&mut rng, 15);
        b.extend_from_slice(&shared);
        b.extend_from_slice(&dna(&mut rng, 5));
        let ma = minimizers(&a, k, w).unwrap();
        let mb = minimizers(&b, k, w).unwrap();
        let shared_hashes = ma.iter().filter(|(_, h)| mb.iter().any(|(_, g)| g == h)).count();
        assert!(shared_hashes > 0, "a shared substring produced no shared minimizer");
    }
}

// ---------------------------------------------------------------------------
// Multiple alignment and assembly
// ---------------------------------------------------------------------------

#[test]
fn prop_the_multiple_alignment_is_rectangular_and_lossless() {
    let mut rng = Rng::new(0x05EA_0030);
    for _ in 0..40 {
        let s = scoring(&mut rng);
        let count = 2 + pick(&mut rng, 4);
        let sequences: Vec<Vec<u8>> =
            (0..count).map(|_| dna_min(&mut rng, 3, 15)).collect();
        let msa = msa_center_star(&sequences, &s).unwrap();
        assert_eq!(msa.len(), count);
        let width = msa[0].len();
        for (row, original) in msa.iter().zip(&sequences) {
            assert_eq!(row.len(), width, "the alignment is not rectangular");
            let stripped: Vec<u8> = row.bytes().filter(|c| *c != b'-').collect();
            assert_eq!(&stripped, original, "a row does not spell out its sequence");
        }
        // No column is all gaps: that would be a column carrying nothing.
        for column in 0..width {
            assert!(
                msa.iter().any(|row| row.as_bytes()[column] != b'-'),
                "column {column} is entirely gaps"
            );
        }
        // The profile is a distribution in every column, and the consensus
        // only uses residues that appear there.
        let profile = profile_from_msa(&msa).unwrap();
        for column in 0..width {
            let total: f64 = profile.iter().map(|(_, f)| f[column]).sum();
            assert!(close(total, 1.0, 1e-12), "column {column} sums to {total}");
            assert!(profile.iter().all(|(_, f)| f[column] >= 0.0));
        }
        let agreed = consensus(&msa).unwrap();
        assert_eq!(agreed.len(), width);
        for (column, c) in agreed.bytes().enumerate() {
            assert!(msa.iter().any(|row| row.as_bytes()[column] == c));
        }
    }
}

#[test]
fn prop_every_assembled_contig_is_spelled_from_observed_k_mers() {
    // An assembler may fail to join things, but it must never invent
    // sequence -- every k-mer in every contig has to come from a read.
    let mut rng = Rng::new(0x05EA_0031);
    for _ in 0..40 {
        let k = 4 + pick(&mut rng, 4);
        let genome = dna_min(&mut rng, 30, 40);
        let read_len = k + 4 + pick(&mut rng, 6);
        if read_len > genome.len() {
            continue;
        }
        let reads: Vec<Vec<u8>> = (0..=genome.len() - read_len)
            .step_by(1 + pick(&mut rng, 3))
            .map(|i| genome[i..i + read_len].to_vec())
            .collect();
        let contigs = de_bruijn_assembly_lite(&reads, k).unwrap();
        assert!(!contigs.is_empty(), "no contig was produced");
        for contig in &contigs {
            assert!(contig.len() >= k, "a contig is shorter than k");
            for window in contig.windows(k) {
                assert!(
                    reads.iter().any(|r| r.windows(k).any(|w| w == window)),
                    "a contig contains a k-mer no read holds"
                );
            }
        }
        // Every observed k-mer appears in some contig, so nothing is lost.
        let mut observed: Vec<&[u8]> =
            reads.iter().flat_map(|r| r.windows(k)).collect();
        observed.sort_unstable();
        observed.dedup();
        for kmer in observed {
            assert!(
                contigs.iter().any(|c| c.windows(k).any(|w| w == kmer)),
                "an observed k-mer is missing from every contig"
            );
        }
    }
}

#[test]
fn prop_the_substitution_matrices_are_symmetric_and_score_identity_highest() {
    // A substitution matrix derived from symmetric alignment counts must be
    // symmetric, and a residue must never score higher against a different
    // residue than against itself -- otherwise the matrix would prefer a
    // mutation to a conservation.
    for matrix in [blosum62(), pam250()] {
        assert!(matrix.is_symmetric());
        // Symmetry holds across the whole table, ambiguity codes included.
        for a in &matrix.alphabet {
            for b in &matrix.alphabet {
                assert_eq!(
                    matrix.lookup(*a, *b).unwrap(),
                    matrix.lookup(*b, *a).unwrap(),
                    "asymmetric at {}/{}",
                    *a as char,
                    *b as char
                );
            }
        }
        // "Identity scores highest" is a statement about *residues*, and the
        // ambiguity codes B, Z and X are not residues: X is a wildcard whose
        // scores are averages over the alphabet, so BLOSUM62 gives X/A zero
        // against X/X of minus one. Asserting the property over them would
        // be asserting something false about what they mean.
        const RESIDUES: &[u8; 20] = b"ARNDCQEGHILKMFPSTWYV";
        for a in RESIDUES {
            let self_score = matrix.lookup(*a, *a).unwrap();
            for b in RESIDUES {
                let cross = matrix.lookup(*a, *b).unwrap();
                if a != b {
                    assert!(
                        cross <= self_score,
                        "{}/{} scores {cross}, above {}'s self-score {self_score}",
                        *a as char,
                        *b as char,
                        *a as char
                    );
                }
            }
        }
        // Using it in an alignment reproduces its own diagonal.
        let s = Scoring { match_score: 0, mismatch: 0, gap: -20, matrix: Some(matrix.clone()) };
        for residue in RESIDUES {
            let sequence = vec![*residue; 4];
            let (score, top, bottom) = needleman_wunsch(&sequence, &sequence, &s).unwrap();
            assert_eq!(score, 4 * matrix.lookup(*residue, *residue).unwrap());
            assert_eq!(alignment_score(&top, &bottom, &s).unwrap(), score);
        }
    }
}
