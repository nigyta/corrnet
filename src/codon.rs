use anyhow::Result;
use ndarray::{Array2, ArrayBase};
use ndarray_stats::*;
use std::collections::BTreeMap;

use crate::rank;

fn make_codon_map() -> BTreeMap<String, usize> {
    let nuc = &['A', 'G', 'C', 'T'];
    let stop_codons = &["TGA", "TAA", "TAG"];
    let mut map: BTreeMap<String, usize> = BTreeMap::new();

    for n1 in nuc.iter() {
        for n2 in nuc.iter() {
            for n3 in nuc.iter() {
                let codon = format!("{}{}{}", n1, n2, n3);
                if stop_codons.contains(&codon.as_ref()) {
                    continue;
                }
                map.entry(codon).or_insert(0);
            }
        }
    }

    map
}

fn make_codon_vec(seq: &str) -> Vec<usize> {
    let mut codon_map = make_codon_map();

    let seq = seq.to_ascii_uppercase();

    for codon in seq.as_bytes().chunks_exact(3) {
        let codon = match std::str::from_utf8(codon) {
            Ok(codon) => codon,
            Err(_) => continue,
        };
        if let Some(cnt) = codon_map.get_mut(codon) {
            *cnt += 1;
        }
    }

    codon_map.into_iter().map(|(_, x)| x).collect()
}

fn make_codon_arr(seqs: &[String]) -> Result<Array2<usize>> {
    const CODON_SIZE: usize = 61;
    let seq_len = seqs.len();

    let mut vec: Vec<usize> = vec![];

    for seq in seqs.iter() {
        let codon_vec = make_codon_vec(seq);
        vec.extend(codon_vec);
    }

    Ok(ArrayBase::from_shape_vec((seq_len, CODON_SIZE), vec)?)
}

fn make_codon_corr(seqs: &[String]) -> Result<Array2<f64>> {
    let codon_arr = make_codon_arr(seqs)?;
    Ok(codon_arr.mapv(|x| x as f64).pearson_correlation()?)
}

pub fn make_codon_rank(seqs: &[String]) -> Result<Array2<usize>> {
    let codon_corr = make_codon_corr(seqs)?;
    rank::construct_rank_matrix(&codon_corr)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_make_codon_map() {
        let codon_map = make_codon_map();
        // check size (4 * 4 * 4 - 3)
        assert_eq!(codon_map.iter().count(), 61);
    }

    #[test]
    fn test_make_codon_vec_1() {
        let seq = "ATGCAGCCCCAGTGA";
        let codon_vec = make_codon_vec(seq);

        let mut codon_map = make_codon_map();
        codon_map.insert("ATG".to_string(), 1);
        codon_map.insert("CAG".to_string(), 2);
        codon_map.insert("CCC".to_string(), 1);

        for (i, (k, v)) in codon_map.iter().enumerate() {
            // check index order
            if *v == 1 {
                assert_eq!(codon_vec[i], 1);
            }

            // check stop codon
            if k == "TGA" {
                assert_eq!(codon_vec[i], 0)
            }
        }
    }

    #[test]
    fn test_make_codon_vec_last_codon_and_short_seq() {
        let codon_map = make_codon_map();
        let pos = |c: &str| codon_map.keys().position(|k| k == c).unwrap();

        // the last codon is counted
        let codon_vec = make_codon_vec("ATGCCC");
        assert_eq!(codon_vec[pos("ATG")], 1);
        assert_eq!(codon_vec[pos("CCC")], 1);

        // lowercase input and a trailing partial codon
        let codon_vec = make_codon_vec("atgcc");
        assert_eq!(codon_vec[pos("ATG")], 1);
        assert_eq!(codon_vec.iter().sum::<usize>(), 1);

        // sequences shorter than one codon do not panic
        assert_eq!(make_codon_vec("AT").iter().sum::<usize>(), 0);
        assert_eq!(make_codon_vec("").iter().sum::<usize>(), 0);
    }
}
