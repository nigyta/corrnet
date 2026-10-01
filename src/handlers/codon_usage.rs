use anyhow::Result;
use csv::Reader;
use itertools::Itertools;
use ordered_float::OrderedFloat;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{collections::HashMap, path::Path};

use crate::codon;
use crate::io;
use crate::math;
use crate::rank;
use crate::similarity;

pub fn parse_args(input_graph: &Path, input_fasta: &Path, percent: &f64) -> Result<()> {
    info!("start caluculate coden score...");

    info!("start read fasta...");
    let (index, seqs) = io::read_fasta(input_fasta)?;

    info!("start construct codon rank matrix");
    // codon arr: convert to rank
    let codon_rank = codon::make_codon_rank(&seqs)?;

    // graph: sort by rank
    info!("start to read graph...");
    let mut rdr = Reader::from_path(input_graph)?;
    let mut map: HashMap<String, Vec<(String, OrderedFloat<f64>)>> = HashMap::new();

    for _r in rdr.deserialize() {
        let r: io::CsvRecord = _r?;
        let (gene_1, gene_2) = r.genes();
        let rank = OrderedFloat::from(r.rank::<f64>());
        map.entry(gene_1.clone())
            .or_insert_with(Vec::new)
            .push((gene_2.clone(), rank));
        map.entry(gene_2)
            .or_insert_with(Vec::new)
            .push((gene_1, rank));
    }

    let corr_index_cnt = map.keys().count();
    let fasta_index_cnt = index.len();

    let k = (std::cmp::min(corr_index_cnt, fasta_index_cnt) as f64 * percent) as usize;

    info!(
        "In rank based graph, there is {} genes and in fasta, there is {} genes",
        corr_index_cnt, fasta_index_cnt
    );
    info!("This tool calculate top_k as {}", k);

    info!("start calculate coxsim score...");

    // sort neighbours of each gene by rank once, then share the map read-only
    let map: HashMap<String, Vec<String>> = map
        .into_par_iter()
        .map(|(gene, mut neighbours)| {
            neighbours.sort_by_key(|x| x.1);
            (gene, neighbours.into_iter().map(|x| x.0).collect_vec())
        })
        .collect();

    // genes with fewer than k neighbours in the graph (e.g. after a rank cutoff) cannot be scored
    let skipped = AtomicUsize::new(0);
    let coxsim_values: Vec<f64> = (0..index.len())
        .into_par_iter()
        .filter_map(|i| {
            let corr_ranked_vec = map.get(&index[i])?;
            if corr_ranked_vec.len() < k {
                skipped.fetch_add(1, Ordering::Relaxed);
                return None;
            }
            let codon_ranked_vec: Vec<String> =
                rank::get_index_sorted_by_rank(&codon_rank, i, &index);

            Some(similarity::coxsim(corr_ranked_vec, &codon_ranked_vec, k))
        })
        .collect();

    let skipped = skipped.into_inner();
    if skipped > 0 {
        warn!(
            "{} genes have fewer than k = {} neighbours in the graph and were skipped",
            skipped, k
        );
    }
    anyhow::ensure!(!coxsim_values.is_empty(), "no gene could be scored");

    info!("caluculation is done!");

    // print median of coxsim values
    println!("Codon Score: {}", math::median(&coxsim_values));

    Ok(())
}

#[allow(dead_code)]
fn sort_corr_by_rank(
    map: &mut HashMap<String, Vec<(String, OrderedFloat<f64>)>>,
    key: &str,
) -> Option<Vec<String>> {
    let corr_ranked = match map.get_mut(key) {
        Some(v) => v,
        None => return None,
    };

    corr_ranked.sort_by(|a, b| a.1.cmp(&b.1));

    Some(corr_ranked.iter().map(|x| x.0.to_owned()).collect())
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_sort_corr_by_rank() {
        let mut map: HashMap<String, Vec<(String, OrderedFloat<f64>)>> = HashMap::new();
        let gene_1 = "gene_1".to_string();
        let gene_1_vec: Vec<(String, OrderedFloat<f64>)> = vec![
            ("gene_2".to_string(), OrderedFloat::from(1.0)),
            ("gene_3".to_string(), OrderedFloat::from(4.0)),
            ("gene_4".to_string(), OrderedFloat::from(3.0)),
            ("gene_5".to_string(), OrderedFloat::from(2.0)),
        ];
        map.insert(gene_1.clone(), gene_1_vec);

        let corr_ranked_vec = sort_corr_by_rank(&mut map, &gene_1).unwrap();
        assert_eq!(
            corr_ranked_vec,
            ["gene_2", "gene_5", "gene_4", "gene_3"]
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<String>>()
        );

        assert_eq!(sort_corr_by_rank(&mut map, &"gene_2".to_string()), None,);
    }
}
